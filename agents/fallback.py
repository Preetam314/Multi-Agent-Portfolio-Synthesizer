from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def fallback_agent(state: dict) -> dict:
    """
    Fallback Agent: Triggers if any node fails. Uses local Qwen2.5
    to analyze the raw error logs and explain the failure modes.
    """
    print("\n--- [Fallback Agent] Activating AI-Powered Error Diagnosis... ---")
    
    error_msg = state.get("error_log", "Unknown system exception encountered.")
    
    try:
        # 1. Setup a prompt forcing the AI to analyze the traceback error
        prompt = PromptTemplate(
            input_variables=["raw_error"],
            template="""You are a senior system operations engineer monitoring an AI data pipeline.
            The pipeline just crashed. Analyze the following raw error message or python traceback:
            
            "{raw_error}"
            
            Write a clear, professional diagnosis (max 2 paragraphs).
            1. Explain in simple terms what went wrong (e.g., missing file, data frame math issue, or connection timeout).
            2. Provide a brief actionable step on how the operator can fix it right now.
            
            Prefix the output with "CRITICAL FAILURE DIAGNOSIS:" and do not use markdown code blocks.
            """
        )
        
        # 2. Call the local model
        llm = ChatOllama(model="qwen2.5:latest", temperature=0.1)
        chain = prompt | llm
        
        # 3. Generate the explanation
        ai_diagnosis = chain.invoke({"raw_error": error_msg})
        safe_strategy = ai_diagnosis.content
        
    except Exception as e:
        # Emergency backup text if even Ollama/LangChain is completely broken/offline
        safe_strategy = f"SYSTEM FALLBACK: Hard failure inside the pipeline. Local AI diagnostic engine offline. Raw error log: {error_msg}"

    # Clear error_log so the graph terminates cleanly and prints out the AI's explanation
    return {"final_portfolio_brief": safe_strategy, "error_log": ""}