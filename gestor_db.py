import os
from dotenv import load_dotenv
import psycopg2
from datetime import datetime
import re

# Carga las variables del .env
load_dotenv()

# Obtén los datos de conexión desde variables de entorno
DB_USER = os.environ.get("DATASOURCE_USR")
DB_PASSWORD = os.environ.get("DATASOURCE_PWD")
DB_URL = os.environ.get("DATASOURCE_URL")

# Imprime las variables de conexión por terminal
print(f"DB_USER: {DB_USER}")
print(f"DB_PASSWORD: {DB_PASSWORD}")
print(f"DB_URL: {DB_URL}")

# Si necesitas extraer host, port y dbname del URL:
match = re.match(r"jdbc:postgresql://([^:/]+):(\d+)/(.+)", DB_URL or "")
if match:
    DB_HOST = match.group(1)
    DB_PORT = match.group(2)
    DB_NAME = match.group(3)
else:
    DB_HOST = DB_PORT = DB_NAME = None

print(f"DB_HOST: {DB_HOST}")
print(f"DB_PORT: {DB_PORT}")
print(f"DB_NAME: {DB_NAME}")

def get_or_create_conversation(user_id: str, conversation_id: int = None):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    if conversation_id:
        # Verifica que exista
        cur.execute("SELECT id FROM conversations WHERE id = %s AND user_id = %s", (conversation_id, user_id))
        row = cur.fetchone()
        if row:
            cur.close()
            conn.close()
            return conversation_id
    # Si no existe, crea una nueva
    cur.execute(
        "INSERT INTO conversations (user_id, created_at) VALUES (%s, %s) RETURNING id",
        (user_id, datetime.utcnow())
    )
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return new_id

def save_message(conversation_id: int, role: str, content: str):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (%s, %s, %s, %s) RETURNING id",
        (conversation_id, role, content, datetime.utcnow())
    )
    message_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return message_id

def save_file(conversation_id: int, attached_file: dict):
    """
    Guarda un archivo adjunto asociado a una conversación.
    """
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO conversation_files (conversation_id, filename, content_type, content, uploaded_at)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        conversation_id,
        attached_file["filename"],
        attached_file.get("content_type", "application/octet-stream"),
        psycopg2.Binary(attached_file["content"]),
        datetime.utcnow()
    ))
    conn.commit()
    cur.close()
    conn.close()

def get_user_chats_summary(user_id: str):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT c.id, COALESCE((
            SELECT LEFT(m.content, 30)
            FROM messages m
            WHERE m.conversation_id = c.id AND m.role = 'user'
            ORDER BY m.timestamp ASC
            LIMIT 1
        ), '') AS summary
        FROM conversations c
        WHERE c.user_id = %s
        ORDER BY c.created_at DESC
    """, (user_id,))
    chats = [
        {
            "conversation_id": row[0],
            "summary": row[1]
        }
        for row in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return chats

def get_chat_by_id(conversation_id: int):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    # Recupera los mensajes de la conversación ordenados por timestamp
    cur.execute("""
        SELECT m.role, m.content, m.timestamp
        FROM messages m
        WHERE m.conversation_id = %s
        ORDER BY m.timestamp ASC
    """, (conversation_id,))
    messages = [
        {
            "role": row[0],
            "content": row[1],
            "timestamp": row[2]
        }
        for row in cur.fetchall()
    ]
    # Recupera los documentos asociados a la conversación
    cur.execute("""
        SELECT id, filename, content_type, uploaded_at
        FROM conversation_files
        WHERE conversation_id = %s
        ORDER BY uploaded_at ASC
    """, (conversation_id,))
    documents = [
        {
            "id": row[0],
            "filename": row[1],
            "content_type": row[2],
            "uploaded_at": row[3]
        }
        for row in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "documents": documents
    }

def get_all_attached_filenames():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("SELECT id, filename, conversation_id FROM conversation_files ORDER BY uploaded_at DESC")
    files = [
        {"id": row[0], "filename": row[1], "conversation_id": row[2]}
        for row in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return files

def get_attached_file_by_id(file_id: int):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT filename, content_type, content, conversation_id
        FROM conversation_files
        WHERE id = %s
    """, (file_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return {
            "filename": row[0],
            "content_type": row[1],
            "content": row[2],
            "conversation_id": row[3]
        }
    else:
        return None

def delete_attached_file_by_id(file_id: int):
    """
    Elimina un archivo adjunto por su id.
    Devuelve True si se eliminó, False si no existía.
    """
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("DELETE FROM conversation_files WHERE id = %s", (file_id,))
    deleted = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return deleted

def delete_chat_by_id(chat_id: int):
    """
    Elimina un chat (conversación), sus mensajes y archivos asociados.
    Devuelve True si se eliminó, False si no existía.
    """
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    # Elimina archivos asociados a la conversación
    cur.execute("""
        DELETE FROM conversation_files
        WHERE conversation_id = %s
    """, (chat_id,))
    # Elimina los mensajes de la conversación
    cur.execute("DELETE FROM messages WHERE conversation_id = %s", (chat_id,))
    # Elimina la conversación
    cur.execute("DELETE FROM conversations WHERE id = %s", (chat_id,))
    deleted = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return deleted