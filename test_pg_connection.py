import psycopg2
import os

DB_USER = "postgres"
DB_PASSWORD = "admin1"
DB_URL = "jdbc:postgresql://localhost:5432/postgres"

import re
match = re.match(r"jdbc:postgresql://([^:/]+):(\d+)/(.+)", DB_URL or "")
if match:
    DB_HOST = match.group(1)
    DB_PORT = match.group(2)
    DB_NAME = match.group(3)
else:
    print("No se pudo parsear el URL de la base de datos")
    exit(1)

try:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    print("Conexión exitosa a PostgreSQL")
    conn.close()
except Exception as e:
    print("Error al conectar:", e)
