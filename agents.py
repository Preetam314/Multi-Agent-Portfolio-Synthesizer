import pandas as pd
import numpy as np
import io 
from langchain_ollama import ChatOllama


def data_configurator(state: dict) -> dict:
    """
    Agent 1: The Data Configurator & Profile Engine.
    Ingests raw fundamental data, calculates financial ratios, 
    and handles accounting anomalies or missing rows.
    """
    print("\n--- [Agent 1] Profiling companies and calculating financial ratios... ---")
    
    csv_path = state.get("csv_input_path", "historical_fundamentals.csv")
    
    try:
        # Load the dataset
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return {"error_log": "Crucial Failure: Ingested CSV file is completely empty."}
            
        # Drop rows where critical fields are entirely missing
        df = df.dropna(subset=['equity', 'current_liabilities', 'ticker', 'report_date'])
        
        # Guardrails against mathematical issues (Divide by zero or negative equity anomalies)
        # Replacing zeros/negatives with NaNs so they don't break the K-Means script
        df['equity'] = df['equity'].apply(lambda x: x if x > 0 else np.nan)
        df['current_liabilities'] = df['current_liabilities'].apply(lambda x: x if x > 0 else np.nan)
        df = df.dropna(subset=['equity', 'current_liabilities'])

        # Calculate Core Financial Ratios
        df['debt_to_equity'] = df['total_debt'] / df['equity']
        df['return_on_equity'] = df['net_income'] / df['equity']
        df['current_ratio'] = df['current_assets'] / df['current_liabilities']
        
        # Keep only the features our ML clustering engine needs
        features = ['ticker', 'report_date', 'price', 'debt_to_equity', 'return_on_equity', 'current_ratio']
        profiled_df = df[features].copy()
        
        # Convert the processed DataFrame into a clean JSON string for the shared global state
        profiled_json = profiled_df.to_json(orient='records')
        
        print(f"Agent 1 Success: Profiled {len(profiled_df)} historical records.")
        return {"profiled_data_json": profiled_json, "error_log": ""}
        
    except Exception as e:
        error_msg = f"Failure in Agent 1 (Data Configurator): {str(e)}"
        print(error_msg)
        return {"error_log": error_msg}


import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# <--- Add this at the top of agents.py with your other imports

# ... (keep your Agent 1 exactly the same) ...

def quant_agent(state: dict) -> dict:
    """
    Agent 2: The Quant Clustering Engine.
    Extracts the profiled ratios, normalizes them, runs K-Means,
    and calculates statistical profiles for each discovered cluster.
    """
    print("\n--- [Agent 2] Running K-Means clustering across financial profiles... ---")
    
    if state.get("error_log"):
        print("Agent 2 skipped due to existing error in pipeline.")
        return {}
        
    profiled_json = state.get("profiled_data_json")
    if not profiled_json:
        return {"error_log": "Failure in Agent 2: No profiled data found in state."}
        
    try:
        # THE FIX: Wrap the string in io.StringIO() so Pandas knows it's text, not a file path
        df = pd.read_json(io.StringIO(profiled_json))
        
        cluster_features = ['debt_to_equity', 'return_on_equity', 'current_ratio']
        X = df[cluster_features].copy()
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['cluster_id'] = kmeans.fit_predict(X_scaled)
        
        cluster_summary = df.groupby('cluster_id')[cluster_features].mean().to_dict(orient='index')
        
        assignments = {}
        for cid in range(3):
            cluster_sample = df[df['cluster_id'] == cid].head(8)
            assignments[f"Cluster_{cid}"] = {
                "characteristic_tickers": cluster_sample['ticker'].tolist(),
                "average_metrics": cluster_summary[cid]
            }
            
        print("Agent 2 Success: Grouped data into 3 distinct financial clusters.")
        import json
        return {"cluster_results_json": json.dumps(assignments), "error_log": ""}
        
    except Exception as e:
        error_msg = f"Failure in Agent 2 (Quant Engine): {str(e)}"
        print(error_msg)
        return {"error_log": error_msg}
# --- Updated Testing Block ---
if __name__ == "__main__":
    dummy_state = {"csv_input_path": "historical_fundamentals.csv"}
    
    # Step 1 Test
    output_1 = data_configurator(dummy_state)
    dummy_state.update(output_1)
    
    # Step 2 Test
    if not dummy_state.get("error_log"):
        output_2 = quant_agent(dummy_state)
        dummy_state.update(output_2)
        
        if dummy_state.get("error_log"):
            print("Test Result: Agent 2 FAILED ->", dummy_state["error_log"])
        else:
            print("Test Result: BOTH AGENTS PASSED!")
            print("\nGenerated Cluster Map for LLM:\n", json.dumps(json.loads(dummy_state["cluster_results_json"]), indent=2))

from langchain_core.prompts import PromptTemplate


def interpreter_agent(state: dict) -> dict:
    """
    Agent 3: The LLM Interpreter.
    Takes the mathematical centroids from the Quant Agent and translates 
    them into a human-readable portfolio strategy using local hardware.
    """
    print("\n--- [Agent 3] Translating quantitative clusters into a portfolio strategy (Local Qwen2.5)... ---")
    
    if state.get("error_log"):
        print("Agent 3 skipped due to existing error in pipeline.")
        return {}
        
    cluster_results = state.get("cluster_results_json")
    if not cluster_results:
        return {"error_log": "Failure in Agent 3: No cluster data found in state."}
        
    try:
        prompt = PromptTemplate(
            input_variables=["cluster_data"],
            template="""You are an elite quantitative portfolio manager.
            Analyze the following K-Means clustering results derived from historical stock data.
            
            Cluster Centroids & Sample Tickers:
            {cluster_data}
            
            Write a concise, highly actionable investment strategy (max 3 paragraphs). 
            Identify what each cluster represents (e.g., "Cluster 0 is high-leverage growth", "Cluster 1 is stable dividend cash cows") 
            based strictly on their average Debt-to-Equity, Return on Equity (ROE), and Current Ratio.
            
            DO NOT output any python code. Output only the professional strategic report.
            """
        )
        
        # Pointing to your local Ollama model instance
        llm = ChatOllama(model="qwen2.5:latest", temperature=0.2)
        
        chain = prompt | llm
        
        response = chain.invoke({"cluster_data": cluster_results})
        
        print("Agent 3 Success: Strategy generated.")
        return {"final_portfolio_brief": response.content, "error_log": ""}
        
    except Exception as e:
        error_msg = f"Failure in Agent 3 (Interpreter): {str(e)}"
        print(error_msg)
        return {"error_log": error_msg}
    
def fallback_agent(state: dict) -> dict:
    """
    Fallback Agent: Triggers if any node fails and provides context-aware recovery.
    Fulfills the course requirement for a fallback failure mechanism.
    """
    print("\n--- [Fallback Agent] Activating emergency response... ---")
    
    error_msg = state.get("error_log", "")
    
    # Dynamically adjust the fallback strategy based on where the error occurred
    if "Agent 1" in error_msg:
        safe_strategy = f"SYSTEM FALLBACK: Critical data ingestion failure. The pipeline halted before quantitative analysis. Error details: {error_msg}"
    elif "Agent 2" in error_msg:
        safe_strategy = f"SYSTEM FALLBACK: Quantitative math engine failed. Clustering aborted. Error details: {error_msg}"
    else:
        # If Agent 3 fails, the math was already completed successfully
        safe_strategy = "SYSTEM FALLBACK: The quantitative data was processed, but the LLM interpreter is currently offline. Maintain holding patterns until natural language synthesis is restored."
    
    # We clear the error log so the graph can finish gracefully and print the brief
    return {"final_portfolio_brief": safe_strategy, "error_log": ""}