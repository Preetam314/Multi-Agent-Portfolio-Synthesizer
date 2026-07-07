# CAPSTONE PROJECT REPORT: Agentic AI Learners' Space 2026
**Project Title:** Autonomous Multi-Agent Portfolio Synthesizer  
**Track:** Week 4 Capstone  
**Format:** Solo Submission  

---

## Executive Summary
A fully autonomous, offline-capable multi-agent system designed to bridge the gap between quantitative financial mathematics and qualitative investment strategy. By utilizing a deterministic LangGraph orchestration pipeline, the system ingests raw corporate fundamentals, clusters them using unsupervised machine learning, retrieves foundational investment rules via a local RAG database, and synthesizes a fact-checked, human-readable portfolio strategy. 

This project goes beyond a simple LLM wrapper by enforcing a strict division of labor across specialized AI agents—ensuring that mathematical computation, semantic knowledge retrieval, and strategic reasoning never contaminate one another.

---

## 1. Problem Originality & Relevance (25%)
### The Problem: Hallucinations in Financial AI
Current single-agent AI systems fail at institutional financial analysis. Large Language Models (LLMs) are semantic engines; they cannot reliably perform complex mathematical calculations, calculate historical variances, or cluster multi-dimensional risk metrics. When a user asks a single-agent LLM to "analyze a CSV of stocks," the model often hallucinates correlations, miscalculates Return on Equity (ROE), or generates generic corporate fluff ("focus on growth") that lacks structural financial logic.

### The Multi-Agent Solution
Financial analysis genuinely requires multiple "minds." A quantitative analyst calculates the math, a researcher finds the historical precedence, and a portfolio manager drafts the strategy. The system mirrors this exact institutional hierarchy. By utilizing multiple agents, the system strictly separates **Deterministic Math** from **Probabilistic Reasoning**. The LLM is never asked to calculate a Debt-to-Equity ratio; it is only asked to interpret the mathematically proven output provided by the Quant Agent, cross-referenced against an authoritative offline textbook.

---

## 2. Orchestration Design (25%)
### The Architecture: Pipeline + Supervisor + Guardrail Router
To prevent the agents from entering infinite loops or overwriting each other's context, the system avoids decentralized "swarm" architectures. Instead, it utilizes a highly controlled **Directed Acyclic Graph (DAG)** via LangGraph, blending the **Pipeline** and **Supervisor** orchestration patterns.

#### The State Schema (The Shared Whiteboard)
The system uses a strictly typed `State` dictionary. This acts as a centralized whiteboard where each agent has read/write access to specific keys (e.g., `cluster_results_json`, `rag_context_json`). This prevents data corruption during handoffs.

#### The Agentic Workflow
1. **Pipeline Pattern:** Execution flows linearly through the primary analysis nodes (Data -> Quant -> RAG -> Interpreter). Each agent is blocked until the previous agent successfully commits its output to the State.
2. **Evaluator-Optimizer (Supervisor) Pattern:** Before the pipeline completes, the Interpreter Agent hands its draft to the Supervisor Agent. The Supervisor acts as a strict evaluator, cross-referencing the draft against the raw mathematical state to flag hallucinations, forcing a rewrite if necessary.
3. **Deterministic Routing (Fault Tolerance):** Between every major node, a custom `check_for_errors` conditional edge evaluates the State. If any agent flags a system error (e.g., missing data, Ollama server crash), the router dynamically rewires the DAG, bypassing the remaining nodes and routing directly to a **Fallback Agent** for diagnostic reporting.

---

## 3. Technical Execution & Division of Responsibility (30%)
The system utilizes at least 5 distinct specialized agents (plus a fallback diagnostic agent). The LLM engine powering the reasoning nodes is a locally hosted **Qwen 2.5** model running via Ollama.

### Agent 1: The Data Configurator
* **Role:** Data ingestion and sanitization.
* **Execution:** Reads the `historical_fundamentals.csv`, handles missing values, and dynamically calculates core financial ratios (Debt-to-Equity, Return on Equity, Current Ratios). It packages this profiled data into JSON format for the downstream agents.

### Agent 2: The Quant Engine (Machine Learning Layer)
* **Role:** Unsupervised regime identification and supervised validation.
* **Execution:** This node features a unique "LLM Critic" loop. It pings the LLM to determine the optimal `k` value for clustering based on the dataset size and asks it to isolate statistical outliers. It then executes **K-Means Clustering** to segment companies into risk/reward regimes. To validate these clusters, it trains a **Logistic Regression** classifier on the labels, achieving ~98% classification accuracy, proving the clusters are mathematically distinct before passing them forward.

### Agent 3.5: The RAG Researcher (ChromaDB Integration)
* **Role:** Grounding the strategy in authoritative financial theory.
* **Execution:** Integrates the concepts from Week 2. It takes the quantitative averages from the clusters (e.g., D/E = 4.3, ROE = 5.9%) and generates semantic queries. It searches a localized **ChromaDB** vector database populated with an offline investment manual (chunked via `langchain-huggingface` embeddings). It extracts the top matching textbook rules for each specific cluster's financial profile.

### Agent 3: The Strategy Interpreter
* **Role:** Synthesis of math and theory.
* **Execution:** Accepts two distinct inputs: the JSON math from Agent 2 and the textbook passages from Agent 3.5. It is strictly prompted to ignore its pre-trained general knowledge and draft a portfolio strategy relying *only* on the provided textbook rules applied to the provided clusters.

### Agent 4: The Logic Supervisor
* **Role:** Quality control and tone alignment.
* **Execution:** Evaluates the draft against the raw data to ensure no numbers were hallucinated. It then applies a style-transfer prompt, rewriting the output to strip away robotic AI jargon, resulting in a clean, human-readable portfolio brief suitable for a college-level presentation.

### Agent 0: The Fallback Agent (System Guardrail)
* **Role:** Error diagnosis.
* **Execution:** Only triggered if the DAG router detects a critical failure. It reads the specific error trace and outputs a diagnostic log rather than allowing the Python runtime to crash violently.

---

## 4. Clarity, Presentation & Output 
### The Tech Stack
* **Orchestration:** LangGraph, LangChain Core
* **Local LLM:** Ollama (`qwen2.5:latest`)
* **Machine Learning:** Scikit-Learn, Pandas
* **RAG / Vector Database:** ChromaDB, HuggingFace Sentence Transformers, PyPDF

### Example Pipeline Output Analysis
During testing, the system successfully identified four distinct market regimes without human intervention:
* **Cluster 2 (Safe Havens):** Low Debt/Equity (0.855), High ROE (13.81%). The RAG agent successfully retrieved textbook rules advising the maintenance of low debt and seeking growth, which the Interpreter accurately drafted.
* **Cluster 3 (High-Leverage Growth):** Massive ROE (30.51%) but dangerous Debt/Equity (5.105). The system correctly identified this as a high-risk profile, advising strict capital allocation and debt-reduction—proving the AI was relying on conservative textbook logic rather than blindly chasing high returns.

---

## 5. Replication & Installation Guide
Follow these sequential steps to set up and run this pipeline from scratch on a new computer.

### Prerequisites
* **Python 3.10+** installed.
* **Ollama** installed locally (download from [ollama.com](https://ollama.com)).

### Step 1: Set Up Environment & Dependencies
Open a terminal in the project root folder and execute:

 Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

 Upgrade pip and install all required project dependencies
pip install --upgrade pip
pip install pandas yfinance simfin scikit-learn langgraph langchain-openai langchain_community langchain-ollama plotly chromadb pypdf sentence-transformers langchain-huggingface langchain-chroma

### Step 2: Prepare Local AI Models

Ensure the Ollama background application is running, then pull the LLM:
Bash

ollama pull qwen2.5:latest

### Step 3: Populate Reference Material

    Place the raw corporate metrics CSV file into the root folder and name it historical_fundamentals.csv.

    Place your chosen investment reference PDF into the root folder and name it investing_manual.pdf.

### Step 4: Initialize the Local Vector Database

Process the reference manual into vector embeddings by running the ingestion script once:
Bash

python ingest_textbook.py

(This builds the chroma_db/ directory offline).
Step 5: Execute the Pipeline

Trigger the autonomous state-handoff graph:
Bash

python main.py