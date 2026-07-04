import io
import json
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def interpreter_agent(state: dict) -> dict:
    """
    Agent 3: The Strategy Interpreter.
    Blends raw quantitative clusters with the fetched offline RAG textbook rules.
    """
    print("\n--- [Agent 3] Translating quantitative clusters into a portfolio strategy (Local Qwen2.5)... ---")
    
    if state.get("error_log"): 
        return {}
        
    cluster_results_json = state.get("cluster_results_json")
    rag_context_json = state.get("rag_context_json") # <-- Pulling the RAG data from main state
    
    if not cluster_results_json: 
        return {"error_log": "Failure in Agent 3: No cluster data found."}
    if not rag_context_json:
        return {"error_log": "Failure in Agent 3: No RAG context data found."}
        
    try:
        # 1. Prepare the math data for the prompt
        cluster_data = json.loads(cluster_results_json)
        formatted_math = json.dumps(cluster_data, indent=2)
        
        # 2. Prepare the textbook data for the prompt
        rag_data = json.loads(rag_context_json)
        formatted_rag = ""
        for cluster, passages in rag_data.items():
            formatted_rag += f"--- Textbook Rules for {cluster} ---\n"
            for passage in passages:
                formatted_rag += f"- {passage}\n"
            formatted_rag += "\n"

        # 3. Create the Strict Grounded Prompt
        prompt = PromptTemplate(
            input_variables=["math_data", "textbook_context"],
            template="""You are a hardheaded portfolio analyst. Your task is to write a comprehensive investment strategy report.
            
            CRITICAL INSTRUCTION: You must build your strategy entirely around the guidelines extracted from our offline investing manual. Do not invent external investment frameworks or use generic training data assumptions. If the textbook rules conflict with common wisdom, side with the textbook.

            RAW MATHEMATICAL CLUSTERS:
            {math_data}

            OFFLINE TEXTBOOK RULES (RETRIEVED VIA RAG):
            {textbook_context}

            REPORT REQUIREMENTS:
            1. Analyze each cluster individually by name. Mention its average metrics (D/E, ROE, Current Ratio).
            2. Explicitly cite the matching textbook rules when advising actions for that cluster. 
            3. Do not use corporate AI jargon. Be objective and analytical.

            Return ONLY the final report text.
            """
        )
        
        llm = ChatOllama(model="qwen2.5:latest", temperature=0.2)
        chain = prompt | llm
        
        # Run the model with both datasets fully injected
        response = chain.invoke({
            "math_data": formatted_math,
            "textbook_context": formatted_rag
        })
        
        result = response.content.strip()
        print("Agent 3 Success: Strategy draft generated using grounded textbook data.")
        return {"draft_portfolio_brief": result, "error_log": ""}
        
    except Exception as e:
        error_msg = f"Failure in Agent 3 (Interpreter): {str(e)}"
        print(error_msg)
        return {"draft_portfolio_brief": "", "error_log": error_msg}