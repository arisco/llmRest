from fastapi import FastAPI, Query, Request, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
import logging
import io
from gestor_db import (
    get_or_create_conversation,
    save_message,
    save_file,
)

app = FastAPI()

# Habilitar CORS para Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOllama(model="llama4:scout")


class ChatRequest(BaseModel):
    text: str
    conversation_id: int | None = None  # Permite conversation_id opcional


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    logging.info(f"Received: {request}")
    try:
        # 1. Obtén o crea la conversación
        conversation_id = get_or_create_conversation(
            user_id="anonymous", conversation_id=request.conversation_id
        )
        # 2. Guarda el mensaje del usuario
        user_msg_id = save_message(conversation_id, "user", request.text)
        # 3. Llama al LLM
        response = llm.invoke([HumanMessage(content=request.text)])
        # 4. Guarda la respuesta del asistente
        assistant_msg_id = save_message(conversation_id, "assistant", response.content)
        return JSONResponse(
            content={"response": response.content, "conversation_id": conversation_id}
        )
    except Exception as e:
        logging.exception("Error in /chat endpoint")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/chat-with-attachment")
async def chat_with_attachment(
    text: str = Form(...),
    conversation_id: int = Form(None),
    file: UploadFile = File(None),
):
    logging.info(
        f"Received text: {text}, file: {file.filename if file else 'No file'}, conversation_id: {conversation_id}"
    )
    try:
        file_bytes = None
        attached_file = None
        file_content = ""
        if file is not None:
            file_bytes = await file.read()
            try:
                file_content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                file_content = file_bytes.decode("latin-1")
            attached_file = {
                "filename": file.filename,
                "content_type": file.content_type,
                "content": file_bytes,
            }

        # 1. Obtén o crea la conversación
        conversation_id_result = get_or_create_conversation(
            user_id="anonymous", conversation_id=conversation_id
        )
        # 2. Guarda el mensaje del usuario
        user_msg_id = save_message(conversation_id_result, "user", text)
        # 3. Si hay archivo, guárdalo
        if attached_file:
            save_file(user_msg_id, attached_file)
        # 4. Llama al LLM pasando el texto y el contenido del fichero como contexto
        llm_input = text
        if file_content:
            llm_input += f"\n\n[Adjunto]:\n{file_content}"
        response = llm.invoke([HumanMessage(content=llm_input)])
        # 5. Guarda la respuesta del asistente
        assistant_msg_id = save_message(conversation_id_result, "assistant", response.content)
        print("DEBUG: assistant_msg_id =", assistant_msg_id)
        logging.info(f"assistant_msg_id: {assistant_msg_id}")
        return JSONResponse(
            content={"response": response.content, "conversation_id": conversation_id_result}
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


# Opcional: para desarrollo local
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("restLLM:app", host="0.0.0.0", port=8000, reload=True)

