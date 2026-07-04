import io
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, accuracy_score
import plotly.express as px
from sklearn.manifold import TSNE
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def generate_3d_cluster_plot(df: pd.DataFrame):
    print("\n Generating interactive 3D cluster map...")
    fig = px.scatter_3d(
        df, x='debt_to_equity', y='return_on_equity', z='current_ratio',
        color='cluster_id', hover_name='ticker', hover_data=['report_date', 'price'],
        title="Portfolio Clustering Engine: 3D Market Regimes",
        labels={"debt_to_equity": "Debt-to-Equity", "return_on_equity": "ROE", "current_ratio": "Current Ratio"},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    fig.write_html("financial_clusters_3d.html")

def generate_2d_tsne_plot(df: pd.DataFrame):
    print("\n Running t-SNE for 2D projection...")
    features = df[['debt_to_equity', 'return_on_equity', 'current_ratio']]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)
    
    df['tsne_x'] = tsne_results[:, 0]
    df['tsne_y'] = tsne_results[:, 1]
    
    fig = px.scatter(
        df, x='tsne_x', y='tsne_y', color='cluster_id',
        hover_name='ticker', hover_data=['report_date', 'price'],
        title="Portfolio Clustering: 2D t-SNE Map",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.write_html("financial_clusters_tsne_2d.html")

def quant_agent(state: dict) -> dict:
    print("\n--- [Agent 2] Activating Autonomous K-Means + Logistic Regression Engine... ---")
    
    if state.get("error_log"): return {}
        
    profiled_json = state.get("profiled_data_json")
    if not profiled_json: return {"error_log": "Failure in Agent 2: No profiled data found."}
        
    try:
        df = pd.read_json(io.StringIO(profiled_json))
        features = ['debt_to_equity', 'return_on_equity', 'current_ratio']
        X = df[features].copy()
        
        # --- PHASE 1: MATH RECONNAISSANCE ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        outliers_mask = (np.abs(X_scaled) > 3).any(axis=1)
        outlier_data = df[outliers_mask][['ticker'] + features].to_dict(orient='records')
        
        scores = {}
        for k in range(2, 8):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            scores[k] = round(silhouette_score(X_scaled, labels), 3)
            
        # --- PHASE 2: LLM CRITIC LOOP ---
        print("\n Consulting LLM Critic for optimal 'k' and outlier removal...")
        prompt = PromptTemplate(
            input_variables=["scores", "outliers"],
            template="""You are the lead quant architect. Analyze this data profile.
            
            Silhouette Scores for different cluster counts (K): {scores}
            (Higher is better, but ensure we have enough groupings for diverse strategies).
            
            Mathematical Outliers: {outliers}
            
            Respond ONLY with a valid JSON object in this exact format. If an outlier heavily distorts the data, add its ticker to the exclude list:
            {{"optimal_k": 4, "exclude_tickers": ["TICKER1"]}}
            """
        )
        llm = ChatOllama(model="qwen2.5:latest", temperature=0.1)
        chain = prompt | llm
        
        response = chain.invoke({"scores": str(scores), "outliers": json.dumps(outlier_data)})
        
        raw_output = response.content.replace("```json", "").replace("```", "").strip()
        critic_decision = json.loads(raw_output)
        
        optimal_k = critic_decision.get("optimal_k", 4)
        bad_tickers = critic_decision.get("exclude_tickers", [])
        print(f" LLM Decision Executed: Using K-Means k={optimal_k}. Erasing outliers: {bad_tickers}")
        
        # --- PHASE 3: K-MEANS EXECUTION ---
        df = df[~df['ticker'].isin(bad_tickers)].copy()
        X_clean = scaler.fit_transform(df[features])
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df['cluster_id'] = kmeans.fit_predict(X_clean)
        df['cluster_id'] = df['cluster_id'].astype(str)

    # --- PHASE 4: SUPERVISED LOGISTIC REGRESSION ---
        print("\n Training Logistic Regression on K-Means labels...")
        
        # Keep it simple: let sklearn handle the multiclass logic automatically
        lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
        
        # Train it on the scaled features and the new k-means labels
        lr_model.fit(X_clean, df['cluster_id'])
        
        # Quick accuracy check
        lr_preds = lr_model.predict(X_clean)
        from sklearn.metrics import accuracy_score
        lr_acc = accuracy_score(df['cluster_id'], lr_preds)
        print(f" Logistic Regression trained successfully! Accuracy against K-Means labels: {lr_acc*100:.2f}%")
        
        # Package the math for the final Interpreter Agent
        cluster_summary = df.groupby('cluster_id')[features].mean().to_dict(orient='index')
        assignments = {}
        for cid in df['cluster_id'].unique():
            cluster_sample = df[df['cluster_id'] == cid].head(8)
            assignments[f"Cluster_{cid}"] = {
                "characteristic_tickers": cluster_sample['ticker'].tolist(),
                "average_metrics": cluster_summary[cid]
            }
            
        generate_3d_cluster_plot(df)
        generate_2d_tsne_plot(df)
        
        print(f"Agent 2 Success: Grouped data into {optimal_k} regimes & trained LR classifier.")
        return {"cluster_results_json": json.dumps(assignments), "error_log": ""}
        
    except Exception as e:
        error_msg = f"Failure in Agent 2 (Quant Engine): {str(e)}"
        print(error_msg)
        return {"error_log": error_msg}