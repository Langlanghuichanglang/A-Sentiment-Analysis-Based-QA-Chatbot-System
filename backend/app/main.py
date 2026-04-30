from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .agent.graph import build_graph


class ChatRequest(BaseModel):
    message: str
    user_id: str = "demo-user"
    profile: dict = {}
    history: list[dict] = []


app = FastAPI(title="Mental Support Agent Backend")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    graph = build_graph()
    result = graph.invoke(
        {
            "user_id": request.user_id,
            "input": request.message,
            "profile": request.profile,
            "history": request.history,
        }
    )
    return {
        "reply": result.get("reply", ""),
        "risk": result.get("risk"),
        "plan": result.get("plan"),
    }
