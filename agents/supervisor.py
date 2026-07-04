from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def supervisor_agent(state: dict) -> dict:
    """
    Agent 4: The Logic Supervisor.
    Catches LLM hallucinations by comparing the draft text against the hard math,
    and applies a humanized writing style.
    """
    print("\n--- [Agent 4] Activating Supervisor to fact-check and humanize... ---")
    
    if state.get("error_log"): 
        return {}
        
    math_data = state.get("cluster_results_json", "")
    draft_text = state.get("draft_portfolio_brief", "")
    
    if not draft_text:
        return {"error_log": "Supervisor failed: No draft text found."}

    # The prompt forces the LLM to act as a strict fact-checker and style editor
    prompt = PromptTemplate(
        input_variables=["math_data", "draft_text"],
        template="""You are a strict data supervisor. Review this draft financial report.

        RAW MATH DATA: 
        {math_data}

        DRAFT REPORT: 
        {draft_text}

        TASK 1 - FACT CHECK: Cross-reference the draft with the raw math. Fix any logical hallucinations. For example, if ROE is 30%, that is MASSIVE, not "relatively low". Correct any bad financial logic.
        
        TASK 2 - HUMANIZE: Rewrite the corrected report in 200 words. It needs to sound exactly like a college student wrote it. Use informal phrasing, the natural way students speak, and even include very minor, natural conversational mistakes. Drop the robotic AI corporate speak completely.

        Return ONLY the final rewritten text. Do not include introductory conversational filler.
        """
    )
    
    llm = ChatOllama(model="qwen2.5:latest", temperature=0.1)
    chain = prompt | llm
    
    try:
        final_text = chain.invoke({"math_data": math_data, "draft_text": draft_text}).content
        print(" Supervisor check complete. Hallucinations fixed and text humanized.")
        return {"final_portfolio_brief": final_text, "error_log": ""}
    except Exception as e:
        return {"error_log": f"Supervisor Agent crashed: {str(e)}"}