import tempfile
import os
import logging
from gestor_db import get_or_create_conversation, get_attached_file_by_id, save_mongo_id_for_file
from llm_wrapper import LLMWrapper
from langchain_community.document_loaders import PyPDFLoader, UnstructuredEPubLoader

# BLIP image analysis imports
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

from langchain_core.documents import Document
from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from fastapi.responses import JSONResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from uuid import uuid4
import tempfile, os, re


llm_wrapper = LLMWrapper(mocked=False)
logger = logging.getLogger("restllm")
# Inicializa BLIP solo una vez, con manejo de error por rate limit o conexión
try:
    # Usa use_fast=False explícitamente para evitar el warning de transformers
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", use_fast=False)
    blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
except Exception as e:
    logger.error("Error cargando BLIP (puede ser por límite de HuggingFace o conexión): %s", e)
    blip_processor = None
    blip_model = None

try:
    from langchain_community.document_loaders import UnstructuredEPubLoader
    UNSTRUCTURED_EPUB_AVAILABLE = True
except ImportError as e:
    logger.error("No se pudo importar UnstructuredEPubLoader: %s", e)
    UNSTRUCTURED_EPUB_AVAILABLE = False

try:
    from ebooklib import epub
    from bs4 import BeautifulSoup
    EBOOKLIB_AVAILABLE = True
    # Define ITEM_DOCUMENT si no existe (ebooklib <0.18)
    if not hasattr(epub, "ITEM_DOCUMENT"):
        # En versiones antiguas, los documentos suelen tener media_type 'application/xhtml+xml'
        def is_document(item):
            return getattr(item, "media_type", "") == "application/xhtml+xml"
    else:
        def is_document(item):
            return item.get_type() == epub.ITEM_DOCUMENT
except ImportError as e:
    logger.error("No se pudo importar ebooklib o bs4 para EPUB: %s", e)
    EBOOKLIB_AVAILABLE = False
    def is_document(item):
        return False



async def process_llm_with_attachments(text, conversation_id, files):
    attached_files = []
    file_contents = []
    temp_pdf_paths = []
    temp_img_paths = []
    temp_epub_paths = []
    if files:
        for file in files:
            file_bytes = await file.read()
            attached_file = {
                "filename": file.filename,
                "content_type": file.content_type,
                "content": file_bytes,
            }
            attached_files.append(attached_file)
            # PDF
            if file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_bytes)
                    tmp.flush()
                    temp_pdf_paths.append(tmp.name)
                    loader = PyPDFLoader(tmp.name)
                    pages = loader.load()
                    pdf_text = "\n".join([doc.page_content for doc in pages])
                    file_contents.append(f"[Contenido PDF {file.filename}]:\n{pdf_text}")
            # EPUB
            # elif (
            #     (file.content_type in ["application/epub+zip", "application/x-epub"] or file.filename.lower().endswith(".epub"))
            #     and UNSTRUCTURED_EPUB_AVAILABLE
            # ):
            #     with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
            #         tmp.write(file_bytes)
            #         tmp.flush()
            #         temp_epub_paths.append(tmp.name)
            #         try:
            #             loader = UnstructuredEPubLoader(tmp.name)
            #             pages = loader.load()
            #             epub_text = "\n".join([doc.page_content for doc in pages])
            #             file_contents.append(f"[Contenido EPUB {file.filename}]:\n{epub_text}")
            #         except Exception as e:
            #             logger.error("Error procesando EPUB con UnstructuredEPubLoader: %s", e)
            #             file_contents.append(f"[Error al procesar EPUB {file.filename}]")
            # EPUB alternativa con ebooklib+BeautifulSoup
            elif (
                (file.content_type in ["application/epub+zip", "application/x-epub"] or file.filename.lower().endswith(".epub"))
                and EBOOKLIB_AVAILABLE
            ):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
                    tmp.write(file_bytes)
                    tmp.flush()
                    temp_epub_paths.append(tmp.name)
                    try:
                        book = epub.read_epub(tmp.name)
                        text_content = []
                        for item in book.get_items():
                            if is_document(item):
                                soup = BeautifulSoup(item.get_content(), "html.parser")
                                text_content.append(soup.get_text(separator="\n"))
                        epub_text = "\n".join(text_content)
                        file_contents.append(f"[Contenido EPUB {file.filename}]:\n{epub_text}")
                    except Exception as e:
                        logger.error("Error procesando EPUB con ebooklib: %s", e)
                        file_contents.append(f"[Error al procesar EPUB {file.filename} con ebooklib]")
            elif (
                file.content_type in ["application/epub+zip", "application/x-epub"] or file.filename.lower().endswith(".epub")
            ):
                logger.error("No se puede procesar EPUB: falta el paquete 'unstructured' o 'ebooklib/bs4'")
                file_contents.append(f"[No se puede procesar EPUB {file.filename}: falta el paquete 'unstructured' o 'ebooklib/bs4']")
            # Imagen
            elif file.content_type.startswith("image/") or file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
                    tmp.write(file_bytes)
                    tmp.flush()
                    temp_img_paths.append(tmp.name)
                    if blip_processor and blip_model:
                        try:
                            image = Image.open(tmp.name).convert('RGB')
                            question = text if text.strip() else "Describe la imagen"
                            inputs = blip_processor(image, question, return_tensors="pt")
                            out = blip_model.generate(**inputs)
                            caption = blip_processor.decode(out[0], skip_special_tokens=True)
                            file_contents.append(f"[Análisis imagen {file.filename}]: {caption}")
                        except Exception as e:
                            logger.error("Error procesando imagen con BLIP: %s", e)
                            file_contents.append(f"[BLIP no disponible o error en análisis de imagen {file.filename}]")
                    else:
                        file_contents.append(f"[BLIP no disponible para analizar imagen {file.filename}]")
            else:
                # Otros tipos de archivo como texto
                try:
                    file_content = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    file_content = file_bytes.decode("latin-1")
                file_contents.append(file_content)
    conversation_id_result = get_or_create_conversation(
        user_id="anonymous", conversation_id=conversation_id
    )
    try:
        response_content = llm_wrapper.invoke(text, file_contents)
        # Limpia los archivos temporales
        for path in temp_pdf_paths + temp_img_paths + temp_epub_paths:
            try:
                os.remove(path)
            except Exception:
                pass
        return response_content, attached_files, conversation_id_result
    except Exception as e:
        logger.exception("Error in process_llm_with_attachments")
        return "", [], None

class MockVectorStore:
    def add_documents(self, documents, ids=None):
        print("[MockVectorStore] add_documents called")
        print("Documents:", documents)
        print("IDs:", ids)
        # Simula un ID de MongoDB
        return ["mocked-mongo-id"]

USE_MOCK_VECTOR_STORE = False

# async def vectorize_document_to_mongo(idDocumento: int):
#     """
#     Recupera un documento adjunto por id, lo vectoriza y lo añade a la base de datos MongoDB Atlas Vector Search.
#     """
#     doc = get_attached_file_by_id(idDocumento)
#     if not doc:
#         return JSONResponse(status_code=404, content={"error": "Documento no encontrado"})
#     if doc.get("id_mongo"):
#         return JSONResponse(status_code=400, content={"error": "El documento ya está vectorizado"})
#     try:
#         content = ""
#         # PDF
#         if doc["content_type"] == "application/pdf" or doc["filename"].lower().endswith(".pdf"):
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#                 tmp.write(doc["content"])
#                 tmp.flush()
#                 loader = PyPDFLoader(tmp.name)
#                 pages = loader.load()
#                 content = "\n".join([p.page_content for p in pages])
#             os.remove(tmp.name)
#         # EPUB
#         elif (
#             (doc["content_type"] in ["application/epub+zip", "application/x-epub"] or doc["filename"].lower().endswith(".epub"))
#             and 'EBOOKLIB_AVAILABLE' in globals() and EBOOKLIB_AVAILABLE
#         ):
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
#                 tmp.write(doc["content"])
#                 tmp.flush()
#                 tmp_path = tmp.name
#             try:
#                 book = epub.read_epub(tmp_path)
#                 text_content = []
#                 for item in book.get_items():
#                     if is_document(item):
#                         soup = BeautifulSoup(item.get_content(), "html.parser")
#                         text_content.append(soup.get_text(separator="\n"))
#                 content = "\n".join(text_content)
#             finally:
#                 try:
#                     os.remove(tmp_path)
#                 except Exception as e:
#                     logger.warning(f"No se pudo eliminar el archivo temporal EPUB: {e}")
#         else:
#             try:
#                 content = doc["content"].decode("utf-8")
#             except Exception:
#                 content = doc["content"].decode("latin-1")

#         document = Document(
#             page_content=content,
#             metadata={
#                 "source": doc["filename"],
#                 "conversation_id": doc["conversation_id"],
#                 "file_id": idDocumento,
#             }
#         )
#         from uuid import uuid4
#         doc_id = str(uuid4())

#         if USE_MOCK_VECTOR_STORE:
#             vector_store = MockVectorStore()
#             mongo_ids = vector_store.add_documents(documents=[document], ids=[doc_id])
#         else:
#             MONGODB_ATLAS_CLUSTER_URI = os.environ.get("MONGODB_ATLAS_CLUSTER_URI")
#             DB_NAME = os.environ.get("MONGODB_ATLAS_DB", "langchain_test_db")
#             COLLECTION_NAME = os.environ.get("MONGODB_ATLAS_COLLECTION", "langchain_test_vectorstores")
#             ATLAS_VECTOR_SEARCH_INDEX_NAME = os.environ.get("MONGODB_ATLAS_INDEX", "langchain-test-index-vectorstores")

#             client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
#             collection = client[DB_NAME][COLLECTION_NAME]

#             embeddings = OllamaEmbeddings(model="bge-m3:latest")
#             vector_store = MongoDBAtlasVectorSearch(
#                 collection=collection,
#                 embedding=embeddings,
#                 index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
#                 relevance_score_fn="cosine",
#             )
#             mongo_ids = vector_store.add_documents(documents=[document], ids=[doc_id])

#         mongo_id = mongo_ids[0] if mongo_ids else doc_id
#         try:
#             save_mongo_id_for_file(idDocumento, mongo_id)
#         except Exception as e:
#             print(f"Error guardando id_mongo en la base de datos: {e}")

#         return {"status": "vectorizado", "file_id": idDocumento, "mongo_id": mongo_id}
#     except Exception as e:
#         logger.exception("Error in vectorize_document_to_mongo")
#         return JSONResponse(status_code=500, content={"error": str(e)})

# async def query_vectorized_document(idDocumento: int, prompt: str):
#     """
#     Realiza una consulta (prompt) sobre el documento vectorizado en MongoDB Atlas Vector Search.
#     """
#     doc = get_attached_file_by_id(idDocumento)
#     if not doc:
#         return JSONResponse(status_code=404, content={"error": "Documento no encontrado"})
#     if not doc.get("id_mongo"):
#         return JSONResponse(status_code=400, content={"error": "El documento no está vectorizado"})
#     try:
#         # Configuración MongoDB Atlas
#         MONGODB_ATLAS_CLUSTER_URI = os.environ.get("MONGODB_ATLAS_CLUSTER_URI")
#         DB_NAME = os.environ.get("MONGODB_ATLAS_DB", "langchain_test_db")
#         COLLECTION_NAME = os.environ.get("MONGODB_ATLAS_COLLECTION", "langchain_test_vectorstores")
#         ATLAS_VECTOR_SEARCH_INDEX_NAME = os.environ.get("MONGODB_ATLAS_INDEX", "langchain-test-index-vectorstores")

#         client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
#         collection = client[DB_NAME][COLLECTION_NAME]

#         embeddings = OllamaEmbeddings(model="bge-m3:latest")
#         vector_store = MongoDBAtlasVectorSearch(
#             collection=collection,
#             embedding=embeddings,
#             index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
#             relevance_score_fn="cosine",
#         )

#         mongo_id = doc["id_mongo"]
#         # Corrige: solo pasa el filtro por 'pre_filter', no por 'filter'
#         pre_filter = {"_id": mongo_id}
#         results = vector_store.similarity_search(prompt, k=3, pre_filter=pre_filter)

#         docs_content = "\n\n".join([doc.page_content for doc in results])

#         llm_prompt = f"Contexto:\n{docs_content}\n\nPregunta: {prompt}\nRespuesta:"
#         response_content = llm_wrapper.invoke(llm_prompt)

#         return {
#             "context": docs_content,
#             "response": response_content
#         }
#     except Exception as e:
#         logger.exception("Error in query_vectorized_document")
#         return JSONResponse(status_code=500, content={"error": str(e)})        
#     except Exception as e:
#         logger.exception("Error in query_vectorized_document")
#         return JSONResponse(status_code=500, content={"error": str(e)})
#         logger.exception("Error in query_vectorized_document")
#         return JSONResponse(status_code=500, content={"error": str(e)})

def extract_sections_from_text(text: str):
    """
    Divide un texto completo en secciones basadas en encabezados tipo 'Capítulo 1', etc.
    Devuelve lista de tuplas: (título de capítulo, texto).
    """
    pattern = r"(CAP[IÍ]TULO\s+\d+[^\n]*|Cap[ií]tulo\s+\d+[^\n]*|\n\d+\.\s+[^\n]+)"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group().strip()
        section_text = text[start:end].strip()
        sections.append((title, section_text))

    if not sections:
        sections = [("Documento completo", text)]

    return sections

async def vectorize_document_to_mongo(idDocumento: int):
    doc = get_attached_file_by_id(idDocumento)
    if not doc:
        return JSONResponse(status_code=404, content={"error": "Documento no encontrado"})
    if doc.get("id_mongo"):
        return JSONResponse(status_code=400, content={"error": "El documento ya está vectorizado"})
    try:
        content = ""
        # Cargar PDF
        if doc["content_type"] == "application/pdf" or doc["filename"].lower().endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(doc["content"])
                tmp.flush()
                loader = PyPDFLoader(tmp.name)
                pages = loader.load()
                content = "\n".join([p.page_content for p in pages])
            os.remove(tmp.name)

        # Cargar EPUB
        elif (
            (doc["content_type"] in ["application/epub+zip", "application/x-epub"] or doc["filename"].lower().endswith(".epub"))
            and 'EBOOKLIB_AVAILABLE' in globals() and EBOOKLIB_AVAILABLE
        ):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
                tmp.write(doc["content"])
                tmp.flush()
                tmp_path = tmp.name
            try:
                book = epub.read_epub(tmp_path)
                text_content = []
                for item in book.get_items():
                    if is_document(item):
                        soup = BeautifulSoup(item.get_content(), "html.parser")
                        text_content.append(soup.get_text(separator="\n"))
                content = "\n".join(text_content)
            finally:
                os.remove(tmp_path)

        # Texto plano
        else:
            try:
                content = doc["content"].decode("utf-8")
            except Exception:
                content = doc["content"].decode("latin-1")

        # Dividir por secciones y chunks
        sections = extract_sections_from_text(content)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        doc_id = str(uuid4())
        documents = []

        for section_title, section_text in sections:
            chunks = text_splitter.split_text(section_text)
            for i, chunk in enumerate(chunks):
                documents.append(LangchainDocument(
                    page_content=chunk,
                    metadata={
                        "source": doc["filename"],
                        "conversation_id": doc["conversation_id"],
                        "file_id": idDocumento,
                        "document_id": doc_id,
                        "chunk_index": i,
                        "chapter": section_title
                    }
                ))

        # Vectorizar y guardar
        MONGODB_ATLAS_CLUSTER_URI = os.environ.get("MONGODB_ATLAS_CLUSTER_URI")
        DB_NAME = os.environ.get("MONGODB_ATLAS_DB", "langchain_test_db")
        COLLECTION_NAME = os.environ.get("MONGODB_ATLAS_COLLECTION", "langchain_test_vectorstores")
        ATLAS_VECTOR_SEARCH_INDEX_NAME = os.environ.get("MONGODB_ATLAS_INDEX", "langchain-test-index-vectorstores")

        client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
        collection = client[DB_NAME][COLLECTION_NAME]

        embeddings = OllamaEmbeddings(model="bge-m3:latest")
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            relevance_score_fn="cosine",
        )

        vector_store.add_documents(documents)
        save_mongo_id_for_file(idDocumento, doc_id)

        return {"status": "vectorizado", "file_id": idDocumento, "mongo_id": doc_id}

    except Exception as e:
        logger.exception("Error en vectorize_document_to_mongo")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
async def query_vectorized_document(idDocumento: int, prompt: str, chapter: str = None):
    doc = get_attached_file_by_id(idDocumento)
    if not doc:
        return JSONResponse(status_code=404, content={"error": "Documento no encontrado"})
    if not doc.get("id_mongo"):
        return JSONResponse(status_code=400, content={"error": "El documento no está vectorizado"})
    try:
        MONGODB_ATLAS_CLUSTER_URI = os.environ.get("MONGODB_ATLAS_CLUSTER_URI")
        DB_NAME = os.environ.get("MONGODB_ATLAS_DB", "langchain_test_db")
        COLLECTION_NAME = os.environ.get("MONGODB_ATLAS_COLLECTION", "langchain_test_vectorstores")
        ATLAS_VECTOR_SEARCH_INDEX_NAME = os.environ.get("MONGODB_ATLAS_INDEX", "langchain-test-index-vectorstores")

        client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
        collection = client[DB_NAME][COLLECTION_NAME]

        # Verifica que el índice de vector search esté correctamente configurado
        # y que el campo de embedding esté indexado como knnVector en Atlas.
        # Si no, devuelve un error claro.
        try:
            # Test simple para ver si el índice existe y es de tipo knnVector
            index_info = collection.index_information()
            # Esto es solo informativo, la comprobación real depende de la configuración Atlas
        except Exception as idx_err:
            logger.error("Error comprobando el índice de vector search: %s", idx_err)
            return JSONResponse(status_code=500, content={"error": "Error comprobando el índice de vector search en MongoDB Atlas. Asegúrate de que el campo de embedding esté indexado como knnVector."})

        embeddings = OllamaEmbeddings(model="bge-m3:latest")
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            relevance_score_fn="cosine",
        )

        pre_filter = {"document_id": doc["id_mongo"]}
        if chapter:
            pre_filter["chapter"] = chapter

        try:
            results = vector_store.similarity_search(prompt, k=5, pre_filter=pre_filter)
        except Exception as ve:
            logger.error("Error en similarity_search: %s", ve)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Error en la búsqueda vectorial. Verifica que el índice de vector search esté correctamente configurado en MongoDB Atlas y que el campo de embedding esté indexado como knnVector.",
                    "details": str(ve)
                }
            )

        if not results:
            return JSONResponse(status_code=404, content={"error": "No se encontraron resultados relevantes"})

        docs_content = "\n\n".join([r.page_content for r in results])
        llm_prompt = f"Contexto:\n{docs_content}\n\nPregunta: {prompt}\nRespuesta:"
        response_content = llm_wrapper.invoke(llm_prompt)

        return {
            "context": docs_content,
            "response": response_content
        }
    except Exception as e:
        logger.exception("Error en query_vectorized_document")
        return JSONResponse(status_code=500, content={"error": str(e)})