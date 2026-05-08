from __future__ import annotations

from .langgraph_nodes import (
    crisis_response_node,
    intervention_planner_node,
    llm_response_node,
    memory_candidate_node,
    output_guard_node,
    retrieve_memory_node,
    retrieve_rag_node,
    risk_triage_node,
)
from .state import AgentState


def route_after_triage(state: AgentState) -> str:
    if state.get("risk", {}).get("level") == "crisis":
        return "crisis_response"
    return "retrieve_memory"


def build_graph():
    try:
      from langgraph.graph import END, StateGraph
    except ImportError as exc:
      raise RuntimeError(
          "LangGraph is not installed. Install backend/requirements.txt before building the graph."
      ) from exc

    workflow = StateGraph(AgentState)
    workflow.add_node("risk_triage", risk_triage_node)
    workflow.add_node("crisis_response", crisis_response_node)
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("retrieve_rag", retrieve_rag_node)
    workflow.add_node("intervention_planner", intervention_planner_node)
    workflow.add_node("llm_response", llm_response_node)
    workflow.add_node("output_guard", output_guard_node)
    workflow.add_node("memory_candidate", memory_candidate_node)

    workflow.set_entry_point("risk_triage")
    workflow.add_conditional_edges(
        "risk_triage",
        route_after_triage,
        {
            "crisis_response": "crisis_response",
            "retrieve_memory": "retrieve_memory",
        },
    )
    workflow.add_edge("crisis_response", END)
    workflow.add_edge("retrieve_memory", "retrieve_rag")
    workflow.add_edge("retrieve_rag", "intervention_planner")
    workflow.add_edge("intervention_planner", "llm_response")
    workflow.add_edge("llm_response", "output_guard")
    workflow.add_edge("output_guard", "memory_candidate")
    workflow.add_edge("memory_candidate", END)
    return workflow.compile()
