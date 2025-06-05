-- Tabla de conversaciones (por si no existe)
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tabla de mensajes (necesaria para el funcionamiento del backend)
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(50),
    content TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);

DROP TABLE IF EXISTS conversation_files;

CREATE TABLE conversation_files (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    filename TEXT,
    content_type TEXT,
    content BYTEA,
    uploaded_at TIMESTAMP
);