import json
# Modern Standalone Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def rag_agent(state: dict) -> dict:
    """
    Agent 3.5: Offline RAG Knowledge Fetcher.
    Queries the modernized Chroma database using cluster metrics to extract core principles.
    """
    print("\n--- [Agent RAG] Querying Local Knowledge Base for Framework Alignment... ---")
    
    if state.get("error_log"): return {}
        
    cluster_results_json = state.get("cluster_results_json")
    if not cluster_results_json:
        return {"error_log": "RAG Agent Error: No cluster math found in state."}
        
    try:
        cluster_data = json.loads(cluster_results_json)
        
        # Connect to the local database using the modern package components
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        rag_context_package = {}
        
        # Query the manual for each distinct cluster profile found
        for cluster_name, details in cluster_data.items():
            metrics = details["average_metrics"]
            roe = metrics["return_on_equity"]
            de = metrics["debt_to_equity"]
            
            # Construct a clear financial context query for the vector search
            query = f"How should an investor evaluate a company with a debt to equity ratio of {de} and a return on equity of {roe}?"
            
            # Pull the top 2 closest matching passages from the textbook chunks
            search_results = vector_db.similarity_search(query, k=2)
            passages = [doc.page_content for doc in search_results]
            
            rag_context_package[cluster_name] = passages
            
        print(" Local knowledge extraction complete. Framework rules matched to all regimes.")
        return {"rag_context_json": json.dumps(rag_context_package), "error_log": ""}
        
    except Exception as e:
        error_msg = f"Failure in RAG Agent Node: {str(e)}"
        print(error_msg)
        return {"error_log": error_msg}