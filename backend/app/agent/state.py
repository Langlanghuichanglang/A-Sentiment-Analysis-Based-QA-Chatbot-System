from typing import Any, Literal, TypedDict


RiskLevel = Literal["normal", "elevated", "crisis"]


class RiskResult(TypedDict):
    level: RiskLevel
    label: str
    tags: list[str]
    reason: str


class UserProfile(TypedDict, total=False):
    mbti_type: str
    memory_enabled: bool
    communication_style: str
    preferences: dict[str, bool]


class AgentState(TypedDict, total=False):
    user_id: str
    input: str
    history: list[dict[str, Any]]
    profile: UserProfile
    risk: RiskResult
    memory: list[dict[str, Any]]
    rag_context: list[dict[str, Any]]
    plan: dict[str, Any]
    draft: str
    reply: str
    citations: list[dict[str, str]]
    memory_candidate: dict[str, Any] | None
    events: list[dict[str, Any]]
