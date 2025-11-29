# Enterprise Clinical Trials Agent (MCP Architecture)

This project implements a robust Retrieval-Augmented Generation (RAG) agent using the **Model Context Protocol (MCP)** standard. It acts as an integration middleware that fetches real-world clinical data, stores it in hybrid databases (SQL + Vector), and exposes it to LLM clients for autonomous reasoning.

## 🚀 Key Features

- **REST API Integration:** Implements a custom ETL pipeline (`etl_pipeline.py`) that consumes the ClinicalTrials.gov API with pagination and error handling.
- **Hybrid Data Architecture:**
  - **Structured (SQL):** Uses SQLite for high-precision filtering (Trial Status, Phase, Conditions).
  - **Unstructured (Vector DB):** Uses ChromaDB with `sentence-transformers` for semantic search within medical protocols.
- **MCP Server:** A centralized integration server (`mcp_server.py`) that standardizes tools for any AI client (Claude Desktop, Cursor, IDEs).

## 🛠️ Installation

1. **Clone the repository:**

    git clone https://github.com/tonih23/clinical-trials-mcp-agent.git  
    cd clinical-trials-mcp-agent

2. **Set up the environment:**

    python -m venv venv  
    # Windows:  
    venv\Scripts\activate  

    # Mac/Linux:  
    source venv/bin/activate  

    pip install -r requirements.txt

## 🔄 Usage Workflow

### 1. Data Ingestion (ETL)

Run the pipeline to fetch live data from the API and populate local databases. This demonstrates the REST integration pattern:

    python etl_pipeline.py

This will fetch the latest trials related to Oncology, Cardiology, and Diabetes.

### 2. Run the MCP Server (Dev Mode)

Start the integration server to test connections:

    mcp dev mcp_server.py

### 3. Connect to LLM Client (Production-like)

To use this with an MCP-compliant client (like Claude Desktop), add this configuration to your `claude_desktop_config.json`:

> Note: Replace the path below with the absolute path to your project folder.  
> On Windows, use double backslashes (`\\`).

    {
      "mcpServers": {
        "clinical-agent": {
          "command": "python",
          "args": ["C:\\Users\\YOUR_USER\\Desktop\\clinical-trials-mcp-agent\\mcp_server.py"]
        }
      }
    }

## 🧠 Example Queries

Once connected, the Agent can perform autonomous tool calling:

### Structured Query (SQL)

"Find active Phase 3 trials for Diabetes."  
(The LLM will automatically route this to `search_trials_sql`.)

### Semantic Query (RAG)

"For trial NCT00528879, what are the specific exclusion criteria regarding kidney function?"  
(The LLM will automatically route this to `get_protocol_details_rag`.)

---

Developed as a Proof of Concept for Enterprise AI Integration.
