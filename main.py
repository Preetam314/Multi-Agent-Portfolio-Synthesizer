from typing import TypedDict
import json
from langgraph.graph import StateGraph, END

# Import the agent nodes we tested over the last two days
from agents.data import data_configurator
from agents.quant import quant_agent
from agents.rag import rag_agent                 # <-- ADDED RAG IMPORT
from agents.interpreter import interpreter_agent
from agents.fallback import fallback_agent
from agents.supervisor import supervisor_agent

# 1. THE STATE SCHEMA (The "Shared Whiteboard")
# This enforces strict keys so agents don't corrupt each other's data.
class State(TypedDict):
    csv_input_path: str
    profiled_data_json: str
    cluster_results_json: str
    rag_context_json: str         # <-- ADDED STATE KEY FOR OFFLINE BOOK PASSAGES
    draft_portfolio_brief: str    
    final_portfolio_brief: str
    error_log: str

# 2. THE DETERMINISTIC ROUTER (The "Rollback Guardrail")
# This is the fault-tolerance mechanism required by the rubric.
def check_for_errors(state: State) -> str:
    """
    Inspects the shared state. If any agent flags an error, 
    it cuts the power and routes execution straight to the fallback agent.
    """
    if state.get("error_log"):
        print(f"\n[Guardrail Triggered] Bypassing next nodes due to error: {state['error_log']}")
        return "rollback"
    return "continue"

# 3. ASSEMBLING THE DAG (Directed Acyclic Graph)
workflow = StateGraph(State)

# Add the specialized computational nodes
workflow.add_node("data_configurator", data_configurator)
workflow.add_node("quant_agent", quant_agent)
workflow.add_node("rag_agent", rag_agent)                 # <-- REGISTERED RAG NODE
workflow.add_node("interpreter_agent", interpreter_agent)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("fallback_agent", fallback_agent)

# Set the starting line
workflow.set_entry_point("data_configurator")

# Wire up the conditional connections
workflow.add_conditional_edges(
    "data_configurator",
    check_for_errors,
    {
        "rollback": "fallback_agent",
        "continue": "quant_agent"
    }
)

# Rerouted: Quant engine now steps into the RAG lookup layer to pull textbook rules
workflow.add_conditional_edges(
    "quant_agent",
    check_for_errors,
    {
        "rollback": "fallback_agent",
        "continue": "rag_agent"                           # <-- ROUTED TO RAG
    }
)

# Added check for RAG layer in case database connection fails or chunks are corrupted
workflow.add_conditional_edges(
    "rag_agent",
    check_for_errors,
    {
        "rollback": "fallback_agent",
        "continue": "interpreter_agent"                   # <-- ROUTED TO INTERPRETER
    }
)

# Wire up the standard connections (Nodes 3, 4, and fallback go straight through)
workflow.add_edge("interpreter_agent", "supervisor")
workflow.add_edge("supervisor", END)
workflow.add_edge("fallback_agent", END)

# Compile the blueprint into an executable system
app = workflow.compile()

# --- Run a Test Run of your Architecture ---
if __name__ == "__main__":
    print("Triggering Multi-Agent State Handoff Pipeline with Offline RAG Grounding...")
    
    # Run a SUCCESS test case (pointing to your valid CSV)
    initial_state = {
        "csv_input_path": "historical_fundamentals.csv", 
        "error_log": ""
    }
    
    final_output = app.invoke(initial_state)
    print("\n--- Pipeline Execution Complete ---")
    
    if final_output.get("final_portfolio_brief"):
        print("\nFINAL PORTFOLIO STRATEGY:\n")
        print(final_output["final_portfolio_brief"])
    elif final_output.get("error_log"):
        print(f"\nPipeline failed to generate strategy. Check error logs: {final_output['error_log']}")
    else:
        print("\nPipeline finished, but no final strategy was returned.")