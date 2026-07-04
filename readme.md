# Autonomous Multi-Agent Financial Portfolio Strategy Pipeline

An advanced, fault-tolerant multi-agent AI system built using LangGraph and Ollama. The pipeline ingests raw corporate financial data, executes unsupervised machine learning (K-Means Clustering) and supervised classification (Logistic Regression), grounds its strategic analysis using a local Retrieval-Augmented Generation (RAG) vector database, and automatically handles quality control and tone humanization via a Supervisor node.

---

## Architecture & System Flow

The system is designed as a Directed Acyclic Graph (DAG) managed by **LangGraph**. State information is shared securely via a centralized schema to ensure context persistence without data corruption.

                [ Shared State Schema ]
                           │

┌─────────────────────────────┼─────────────────────────────┐
▼                             ▼                             ▼
[Agent 1: Data] ──► [Agent 2: Quant] ──► [Agent 3.5: RAG] ──► [Agent 3: Interpreter] ──► [Agent 4: Supervisor] ──► END
│                       │                 │
└───────────────┬───────┴─────────────────┘
▼
🛑 [Guardrail Router] ──► [Fallback Agent] ──► END


### The 5-Node Agentic Pipeline

1. **Agent 1: Data Configurator (`agents/data.py`)**
   Loads and profiles the raw fundamental stock metrics (`historical_fundamentals.csv`). Calculates core financial health metrics including Debt-to-Equity, Return on Equity (ROE), and Current Ratios.
2. **Agent 2: Quant Engine (`agents/quant.py`)**
   Spins up an autonomous machine learning loop. It pings an LLM Critic to determine the optimal number of clusters ($k$) and isolate extreme market outliers. It then executes **K-Means Clustering** and trains a **Logistic Regression** classifier to recognize distinct market regimes (achieving ~98% classification accuracy).
3. **Agent 3.5: Offline RAG Agent (`agents/rag.py`)**
   Acts as the corporate knowledge base. It extracts the numerical profiles of each cluster, generates natural language queries, and performs a similarity search against a localized **ChromaDB** vector instance containing your offline investment reference manual.
4. **Agent 3: Strategy Interpreter (`agents/interpreter.py`)**
   Synthesizes quantitative metrics with the semantic text chunks fetched by the RAG layer. It structures a robust, grounded investment thesis tailored to the constraints of the reference data.
5. **Agent 4: Logic Supervisor (`agents/supervisor.py`)**
   Serves as the ultimate quality controller and style editor. It cross-references the interpreter's draft with the raw mathematical data to catch and fix structural hallucinations. Finally, it executes a complete style rewrite—dropping corporate AI jargon to deliver an authentic, clear report that sounds like a college student.

### Deterministic Fault-Tolerance (The Rollback Guardrail)
If any critical operation crashes or encounters missing parameters during the data or math phases, the centralized `check_for_errors` router instantly halts the path and diverts execution to the **Fallback Agent (`agents/fallback.py`)**, preventing unhandled runtime exceptions.

---

## Replication & Installation Guide

Follow these sequential steps to set up and run this pipeline on a new computer.

### Prerequisites
* **Python 3.10+** installed.
* **Ollama** installed on the local system. Download it from [ollama.com](https://ollama.com).

### Step 1: Clone and Set Up Environment
Open your terminal inside the project root folder and execute:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (Command Prompt):
venv\Scripts\activate.bat
# On Windows (PowerShell):
.\venv\Scripts\activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip and install all required dependencies from the project requirements file
pip install --upgrade pip
pip install -r requirements.txt
Step 2: Prepare Local AI Models

Ensure the local Ollama background server is running on the computer, then pull the required LLM architecture:
Bash

ollama pull qwen2.5:latest

To verify it is ready, run ollama list and ensure qwen2.5:latest appears in the registry.
Step 3: Populate Reference Material & Grounding Data
         
    Place your raw corporate metrics CSV file into the root project workspace directory and name it historical_fundamentals.csv.(run scraper.py)

    Download your preferred investment reference manual, cheat sheet, or textbook in PDF format.

    Place this PDF directly in the root project folder and rename it exactly to investing_manual.pdf.

Step 4: Initialize the Local Vector Database

Before executing the pipeline, you must process the reference manual text into vector embeddings. Run the ingestion script once:
Bash

python ingest_textbook.py

This script splits the PDF document into individual semantic chunks, initializes an all-MiniLM-L6-v2 embedding processor locally, and compiles a vector storage library inside a fresh chroma_db/ directory.
Step 5: Execute the Multi-Agent System

Once the database is constructed, trigger the state-handoff pipeline:
Bash

python main.py

The console will display real-time terminal indicators showing the execution, profiling data state, active routing coordinates, and the finalized, fact-checked strategic brief.