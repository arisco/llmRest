import psycopg2
from datetime import datetime

def get_or_create_conversation(user_id: str, conversation_id: int = None):
    conn = psycopg2.connect(
        dbname="base",
        user="base",
        password="",  # Añade la contraseña si es necesaria
        host="localhost",
        port=5432
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
        dbname="base",
        user="base",
        password="",  # Añade la contraseña si es necesaria
        host="localhost",
        port=5432
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

def save_file(message_id: int, attached_file: dict):
    conn = psycopg2.connect(
        dbname="base",
        user="base",
        password="",  # Añade la contraseña si es necesaria
        host="localhost",
        port=5432
    )
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO conversation_files (message_id, filename, content_type, content, uploaded_at)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        message_id,
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
        dbname="base",
        user="base",
        password="",  # Añade la contraseña si es necesaria
        host="localhost",
        port=5432
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
        dbname="base",
        user="base",
        password="",  # Añade la contraseña si es necesaria
        host="localhost",
        port=5432
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
    cur.close()
    conn.close()
    return {
        "conversation_id": conversation_id,
        "messages": messages
    }

def get_all_attached_filenames():
    conn = psycopg2.connect(
        dbname="base",
        user="base",
        password="",  # Añade la contraseña si es necesaria
        host="localhost",
        port=5432
    )
    cur = conn.cursor()
    cur.execute("SELECT filename FROM conversation_files ORDER BY uploaded_at DESC")
    filenames = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return filenames