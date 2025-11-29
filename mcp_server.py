from mcp.server.fastmcp import FastMCP
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
import json
import os

# 1. Definimos el nombre de nuestro Servidor MCP
mcp = FastMCP("Enterprise-Clinical-Trials-Service")

# --- CONFIGURACIÓN DE RUTAS DINÁMICAS ---
# Esto hace que funcione en cualquier ordenador sin cambiar código
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, "clinical_trials.db")
VECTOR_DB_PATH = os.path.join(CURRENT_DIR, "chroma_db_data")

# Cargamos modelos una sola vez al inicio
# IMPORTANTE: No usar print() aquí para evitar ensuciar la salida stdio del protocolo MCP
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_collection("clinical_protocols")

# 2. HERRAMIENTA 1: Búsqueda SQL (Estructurada)
@mcp.tool()
def search_trials_sql(query_condition: str) -> str:
    """
    Search for clinical trials in the SQL database by condition or title.
    Useful for listing trials, checking phases, or finding IDs.
    Args:
        query_condition: The disease or keyword to search (e.g. 'Breast Cancer')
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT nct_id, title, status, phase FROM trials WHERE conditions LIKE ? OR title LIKE ? LIMIT 5", 
            (f"%{query_condition}%", f"%{query_condition}%")
        )
        results = [dict(row) for row in cursor.fetchall()]
        return json.dumps(results, indent=2) if results else "No trials found."
    finally:
        conn.close()

# 3. HERRAMIENTA 2: Búsqueda Vectorial (RAG / No estructurada)
@mcp.tool()
def get_protocol_details_rag(user_question: str, specific_nct_id: str = None) -> str:
    """
    Retrieves detailed information from clinical protocols using vector search (RAG).
    Use this when asking about exclusion criteria, methodology, or specific trial details.
    Args:
        user_question: The specific question about the protocol.
        specific_nct_id: Optional. If known, filters by Trial ID (e.g., NCT05196035).
    """
    # Vectorizar la pregunta
    query_vector = embedding_model.encode([user_question]).tolist()
    
    # Filtros
    search_filters = {"nct_id": specific_nct_id} if specific_nct_id else None
    
    results = collection.query(
        query_embeddings=query_vector, 
        n_results=2,
        where=search_filters 
    )
    
    if results['documents'] and results['documents'][0]:
        context = "\n\n".join(results['documents'][0])
        return f"DATA RETRIEVED FROM PROTOCOLS:\n{context}"
    else:
        return "No relevant protocol details found."

# 4. Iniciar el servidor
if __name__ == "__main__":
    mcp.run()