import requests
import sqlite3
import chromadb
import time
import os
from sentence_transformers import SentenceTransformer

# --- CONFIGURACIÓN DE VOLUMEN ---
TARGET_TOTAL_TRIALS = 200 

# IDs DE EJEMPLO (Puedes añadir los que necesites específicos)
VIP_TRIALS = [] 

# Configuración de Rutas (Relativas para que funcione en cualquier PC)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, "clinical_trials.db")
VECTOR_DB_PATH = os.path.join(CURRENT_DIR, "chroma_db_data")
API_URL = "https://clinicaltrials.gov/api/v2/studies"

def setup_databases():
    """Reinicia las bases de datos."""
    print("🛠️  Inicializando bases de datos...")
    
    # 1. SQL
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS trials") 
    cursor.execute('''
        CREATE TABLE trials (
            nct_id TEXT PRIMARY KEY,
            title TEXT,
            status TEXT,
            phase TEXT,
            conditions TEXT
        )
    ''')
    conn.commit()
    conn.close()

    # 2. Vector DB
    try:
        chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        try:
            chroma_client.delete_collection(name="clinical_protocols")
        except:
            pass
        collection = chroma_client.create_collection(name="clinical_protocols")
        return collection
    except Exception as e:
        print(f"⚠️ Error inicializando ChromaDB: {e}")
        return None

def process_and_insert(studies, collection, model, cursor):
    """Procesa un lote de estudios y los guarda."""
    documents = []
    metadatas = []
    ids = []
    count = 0

    for study in studies:
        try:
            protocol = study.get('protocolSection', {})
            id_module = protocol.get('identificationModule', {})
            status_module = protocol.get('statusModule', {})
            design_module = protocol.get('designModule', {})
            cond_module = protocol.get('conditionsModule', {})
            desc_module = protocol.get('descriptionModule', {})
            elig_module = protocol.get('eligibilityModule', {})

            # --- SQL Data ---
            nct_id = id_module.get('nctId')
            if not nct_id: continue

            title = id_module.get('briefTitle', 'No Title')
            status = status_module.get('overallStatus', 'Unknown')
            phases = ", ".join(design_module.get('phases', [])) if 'phases' in design_module else "Not Applicable"
            conditions = ", ".join(cond_module.get('conditions', []))
            
            cursor.execute("INSERT OR REPLACE INTO trials VALUES (?,?,?,?,?)", 
                           (nct_id, title, status, phases, conditions))

            # --- RAG Data ---
            description = desc_module.get('detailedDescription', '')
            eligibility = elig_module.get('eligibilityCriteria', '')
            
            full_text = f"TRIAL ID: {nct_id}\nTITLE: {title}\nCONDITIONS: {conditions}\nDESCRIPTION: {description}\nELIGIBILITY CRITERIA: {eligibility}"
            
            # Solo guardamos si hay texto suficiente
            if len(full_text) > 100:
                documents.append(full_text)
                metadatas.append({"nct_id": nct_id, "source": "ClinicalTrials.gov"})
                ids.append(nct_id)
                count += 1

        except Exception as e:
            continue

    # Insertar Vectores
    if documents:
        embeddings = model.encode(documents).tolist()
        collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
    
    return count

def run_heavy_etl(collection):
    print(f"🚀 Iniciando Carga Masiva. Objetivo: {TARGET_TOTAL_TRIALS} ensayos.")
    
    # Preparamos modelo y conexión SQL
    print("🧠 Cargando modelo neuronal (esto tarda la primera vez)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json"
    }

    total_loaded = 0

    # --- FASE 1: DESCARGA VIP ---
    if VIP_TRIALS:
        print(f"\n⭐ FASE 1: Descargando Ensayos VIP ({len(VIP_TRIALS)})...")
        vip_params = {
            "filter.ids": ",".join(VIP_TRIALS),
            "fields": "NCTId,BriefTitle,OverallStatus,DesignModule,ConditionsModule,DescriptionModule,EligibilityModule"
        }
        try:
            resp = requests.get(API_URL, params=vip_params, headers=headers)
            if resp.status_code == 200:
                studies = resp.json().get('studies', [])
                c = process_and_insert(studies, collection, model, cursor)
                conn.commit()
                total_loaded += c
                print(f"   ✅ {c} Ensayos VIP cargados correctamente.")
        except Exception as e:
            print(f"   ❌ Error cargando VIPs: {e}")

    # --- FASE 2: CARGA MASIVA (Relleno) ---
    print("\n🌊 FASE 2: Iniciando Carga Masiva (Paginación)...")
    next_page_token = None
    page_num = 1

    while total_loaded < TARGET_TOTAL_TRIALS:
        remaining = TARGET_TOTAL_TRIALS - total_loaded
        batch_size = min(50, remaining) if remaining > 0 else 50

        bulk_params = {
            "query.term": "Cancer OR Cardiology OR Alzheimer OR Diabetes",
            "pageSize": batch_size,
            "fields": "NCTId,BriefTitle,OverallStatus,DesignModule,ConditionsModule,DescriptionModule,EligibilityModule"
        }
        
        if next_page_token:
            bulk_params["pageToken"] = next_page_token

        try:
            print(f"   ... Descargando Página {page_num} (Llevamos {total_loaded}/{TARGET_TOTAL_TRIALS})...")
            resp = requests.get(API_URL, params=bulk_params, headers=headers, timeout=30)
            
            if resp.status_code != 200:
                print(f"   ⚠️ Error API (Code {resp.status_code}). Parando.")
                break
            
            data = resp.json()
            studies = data.get('studies', [])
            
            if not studies:
                print("   ⚠️ No hay más ensayos disponibles.")
                break

            c = process_and_insert(studies, collection, model, cursor)
            conn.commit()
            total_loaded += c
            
            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                break 
            
            page_num += 1
            time.sleep(1) 

        except Exception as e:
            print(f"   ❌ Error en paginación: {e}")
            break

    conn.close()
    print(f"\n🎉 FIN DEL ETL. Total en Base de Datos: {total_loaded} ensayos.")

if __name__ == "__main__":
    coll = setup_databases()
    if coll:
        run_heavy_etl(coll)