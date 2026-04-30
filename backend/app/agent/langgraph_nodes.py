from __future__ import annotations

from .state import AgentState


def risk_triage_node(state: AgentState) -> AgentState:
    text = state.get("input", "")
    crisis_keywords = ["自杀", "轻生", "不想活", "自残", "kill myself", "suicide"]
    elevated_keywords = ["绝望", "崩溃", "撑不住", "失眠", "panic", "hopeless"]
    diagnosis_keywords = ["诊断", "确诊", "抑郁症", "自闭症", "概率", "%"]

    tags: list[str] = []
    lowered = text.lower()
    if any(keyword in lowered for keyword in crisis_keywords):
        tags.append("self_harm_or_immediate_danger")
        state["risk"] = {
            "level": "crisis",
            "label": "高危",
            "tags": tags,
            "reason": "出现自伤、自杀或立即危险相关表达。",
        }
        return state

    if any(keyword in lowered for keyword in elevated_keywords):
        tags.append("elevated_distress")
    if any(keyword in lowered for keyword in diagnosis_keywords):
        tags.append("diagnosis_request")

    state["risk"] = {
        "level": "elevated" if "elevated_distress" in tags else "normal",
        "label": "中等风险" if "elevated_distress" in tags else "常规支持",
        "tags": tags,
        "reason": "用于选择支持路径，不作为诊断依据。",
    }
    return state


def crisis_response_node(state: AgentState) -> AgentState:
    state["reply"] = (
        "我注意到你提到了可能的危险信号。现在请优先联系当地急救电话、"
        "学校心理中心或身边可信赖的人，并尽量让自己不要独处。"
    )
    return state


def retrieve_memory_node(state: AgentState) -> AgentState:
    state["memory"] = state.get("memory", [])
    return state


def retrieve_rag_node(state: AgentState) -> AgentState:
    # Replace this placeholder with pgvector retrieval.
    state["rag_context"] = []
    return state


def intervention_planner_node(state: AgentState) -> AgentState:
    risk = state.get("risk", {})
    tags = risk.get("tags", [])
    steps = []
    if "diagnosis_request" in tags:
        steps.append("diagnosis_boundary")
    steps.extend(["emotional_validation", "cbt_or_grounding_step"])
    state["plan"] = {"steps": steps}
    return state


def llm_response_node(state: AgentState) -> AgentState:
    # Replace with Volcengine Ark call.
    state["draft"] = (
        "我不能提供诊断结论，但可以帮你把现在的感受、触发事件和可尝试的小步骤整理出来。"
    )
    return state


def output_guard_node(state: AgentState) -> AgentState:
    draft = state.get("draft", "")
    blocked = ["确诊为", "患有", "处方", "停药"]
    if any(item in draft for item in blocked):
        state["reply"] = "我不能提供诊断或处方建议，但可以继续提供自我观察和求助建议。"
    else:
        state["reply"] = draft
    return state


def memory_candidate_node(state: AgentState) -> AgentState:
    state["memory_candidate"] = None
    return state
