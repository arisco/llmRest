from fastapi import FastAPI, Query, Request, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse
import os
# from langchain_ollama import ChatOllama
# from langchain.schema import HumanMessage
import logging
import io
from gestor_db import (
    get_or_create_conversation,
    save_message,
    save_file,
)
from typing import List, Optional
from llm_wrapper import LLMWrapper  # <--- importa el wrapper externo
from llm_service import process_llm_with_attachments, vectorize_document_to_mongo  # Nuevo servicio para LLM y ficheros

# Añade import para PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama
from threading import Lock
import uuid
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI

app = FastAPI()

# Habilitar CORS para Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock para ChatOllama
# class MockChatOllama:
#     def __init__(self, model=None):
#         self.model = model

#     def invoke(self, messages):
#         # Simula una respuesta del modelo
#         class Response:
#             def __init__(self, content):
#                 self.content = content
#         # Puedes personalizar la respuesta mock aquí
#         prompt = messages[0].content if messages else ""
#         return Response(content=f"[MOCKED OLLAMA RESPONSE] Recibido: {prompt}")

# Selecciona el LLM real o el mock según variable de entorno
# if os.environ.get("MOCK_OLLAMA") == "1":
#     llm = MockChatOllama(model="llama4:scout")
# else:
#     llm = ChatOllama(model="llama4:scout")


class ChatRequest(BaseModel):
    text: str
    conversation_id: int | None = None  # Permite conversation_id opcional

# Cambia a True para mock, False para real
llm_wrapper = LLMWrapper(mocked=False)

# --- Sustituye ConversationChain por LangGraph ---

# Define el grafo y el modelo globalmente
workflow = StateGraph(state_schema=MessagesState)
model = ChatOllama(model="llama4:scout")  # Usa el modelo Ollama actual

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

def get_langgraph_app_and_config(conversation_id: int):
    """
    Devuelve el grafo langgraph y la config para el conversation_id.
    """
    config = {"configurable": {"thread_id": conversation_id}}
    return app_graph, config

@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    logging.info(f"Received: {request}")
    try:
        conversation_id = get_or_create_conversation(
            user_id="anonymous", conversation_id=request.conversation_id
        )
        user_msg_id = save_message(conversation_id, "user", request.text)

        # --- Conversación con memoria usando LangGraph ---
        app_graph, config = get_langgraph_app_and_config(conversation_id)
        input_message = HumanMessage(content=request.text)
        response_content = ""
        # Procesa el mensaje usando el grafo y recupera la respuesta
        for event in app_graph.stream({"messages": [input_message]}, config, stream_mode="values"):
            # Tomamos el último mensaje generado por el modelo
            if event["messages"]:
                response_content = event["messages"][-1].content

        assistant_msg_id = save_message(conversation_id, "assistant", response_content)
        return JSONResponse(
            content={"response": response_content, "conversation_id": conversation_id}
        )
    except Exception as e:
        logging.exception("Error in /chat endpoint")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/chat-with-attachment")
async def chat_with_attachment(
    text: str = Form(...),
    conversation_id: int = Form(None),
    files: Optional[List[UploadFile]] = File(None),
):
    logging.info(
        f"Received text: {text}, files: {[f.filename for f in files] if files else 'No files'}, conversation_id: {conversation_id}"
    )
    try:
        response_content, attached_files, conversation_id_result = await process_llm_with_attachments(
            text, conversation_id, files
        )
        # Guarda archivos adjuntos en la base de datos
        for attached_file in attached_files:
            save_file(conversation_id_result, attached_file)
        # Guarda el mensaje del usuario y la respuesta del asistente
        user_msg_id = save_message(conversation_id_result, "user", text)
        assistant_msg_id = save_message(conversation_id_result, "assistant", response_content)
        return JSONResponse(
            content={"response": response_content, "conversation_id": conversation_id_result}
        )
    except Exception as e:
        logging.exception("Error in /chat-with-attachment endpoint")
        return JSONResponse(status_code=500, content={"error": str(e)})


# @app.get("/documents")
# async def get_documents(user_id: str = None):
#     """
#     Devuelve la lista de documentos adjuntos para un usuario.
#     """
#     from gestor_db import get_user_documents
#     if not user_id:
#         return JSONResponse(status_code=400, content={"error": "user_id es requerido"})
#     documents = get_user_documents(user_id)
#     return {"documents": documents}


@app.get("/chats")
async def get_chats(user_id: str):
    """
    Devuelve el resumen de los chats de un usuario (primeros 30 caracteres del primer prompt y conversation_id).
    """
    from gestor_db import get_user_chats_summary
    return {"chats": get_user_chats_summary(user_id)}


@app.get("/chat/{id}")
async def get_chat(id: int):
    """
    Devuelve el chat completo (pregunta, respuesta, etc) dado su id.
    """
    # Importa aquí para evitar error de import circular
    from gestor_db import get_chat_by_id
    chat = get_chat_by_id(id)
    if chat:
        return {"chat": chat}
    else:
        return JSONResponse(status_code=404, content={"error": "Chat not found"})


@app.get("/documents")
async def get_documents():
    """
    Devuelve un listado de ids y nombres de todos los ficheros adjuntos.
    """
    from gestor_db import get_all_attached_filenames
    files = get_all_attached_filenames()
    return {"files": files}


@app.get("/document/{file_id}")
async def get_document(file_id: int):
    """
    Devuelve el archivo adjunto dado su id.
    """
    from gestor_db import get_attached_file_by_id
    file = get_attached_file_by_id(file_id)
    if file:
        return StreamingResponse(
            io.BytesIO(file["content"]),
            media_type=file["content_type"],
            headers={"Content-Disposition": f'attachment; filename="{file["filename"]}"'}
        )
    else:
        return JSONResponse(status_code=404, content={"error": "Archivo no encontrado"})


@app.delete("/document/{file_id}")
async def delete_document(file_id: int):
    """
    Elimina un archivo adjunto dado su id.
    """
    from gestor_db import delete_attached_file_by_id
    try:
        deleted = delete_attached_file_by_id(file_id)
        if deleted:
            return {"status": "deleted", "file_id": file_id}
        else:
            return JSONResponse(status_code=404, content={"error": "Archivo no encontrado"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: int):
    """
    Elimina un chat dado su id.
    """
    from gestor_db import delete_chat_by_id
    try:
        deleted = delete_chat_by_id(chat_id)
        if deleted:
            return {"status": "deleted", "chat_id": chat_id}
        else:
            return JSONResponse(status_code=404, content={"error": "Chat no encontrado"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/rag/save/{idDocumento}")
async def rag_save_document(idDocumento: int):
    """
    Recupera un documento adjunto por id, lo vectoriza y lo añade a la base de datos MongoDB Atlas Vector Search.
    """
    try:
        result = await vectorize_document_to_mongo(idDocumento)
        return result
    except Exception as e:
        logging.exception("Error en vectorización RAG" +  str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/rag/query/{idDocumento}")
async def rag_query_document(idDocumento: int, prompt: str = Body(..., embed=True)):
    """
    Realiza una consulta (prompt) sobre el documento vectorizado en MongoDB Atlas Vector Search.
    """
    try:
        from llm_service import query_vectorized_document
        # Asegúrate de que la función existe y está correctamente importada
        result = await query_vectorized_document(idDocumento, prompt)
        return result
    except Exception as e:
        logging.exception("Error en consulta RAG: " + str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})


# Configuración de logging para consola y archivo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Consola
        logging.FileHandler("restllm.log", encoding="utf-8")  # Archivo
    ]
)

# Opcional: para desarrollo local
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("restLLM:app", host="0.0.0.0", port=8000, reload=True)

