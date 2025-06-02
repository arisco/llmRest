-- Tabla de conversaciones (por si no existe)
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    question TEXT,
    response TEXT,
    timestamp TIMESTAMP
);

-- Tabla de archivos adjuntos
CREATE TABLE conversation_files (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    filename VARCHAR(255),
    content_type VARCHAR(100),
    content BYTEA,
    uploaded_at TIMESTAMP
);