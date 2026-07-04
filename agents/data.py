import os
import pandas as pd
import json
import numpy as np


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