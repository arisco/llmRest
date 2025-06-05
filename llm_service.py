import tempfile
import os
from gestor_db import get_or_create_conversation
from llm_wrapper import LLMWrapper
from langchain_community.document_loaders import PyPDFLoader

llm_wrapper = LLMWrapper(mocked=False)

async def process_llm_with_attachments(text, conversation_id, files):
    attached_files = []
    file_contents = []
    temp_pdf_paths = []
    if files:
        for file in files:
            file_bytes = await file.read()
            attached_file = {
                "filename": file.filename,
                "content_type": file.content_type,
                "content": file_bytes,
            }
            attached_files.append(attached_file)
            # Si es un PDF, extrae el texto usando PyPDFLoader
            if file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_bytes)
                    tmp.flush()
                    temp_pdf_paths.append(tmp.name)
                    loader = PyPDFLoader(tmp.name)
                    pages = loader.load()
                    pdf_text = "\n".join([doc.page_content for doc in pages])
                    file_contents.append(f"[Contenido PDF {file.filename}]:\n{pdf_text}")
            else:
                # Si no es PDF, intenta decodificar como texto
                try:
                    file_content = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    file_content = file_bytes.decode("latin-1")
                file_contents.append(file_content)
    conversation_id_result = get_or_create_conversation(
        user_id="anonymous", conversation_id=conversation_id
    )
    # Llama al LLM o mock, pasando el texto y el contenido extraído de los PDFs
    response_content = llm_wrapper.invoke(text, file_contents)
    # Limpia los archivos temporales PDF
    for path in temp_pdf_paths:
        try:
            os.remove(path)
        except Exception:
            pass
    return response_content, attached_files, conversation_id_result
