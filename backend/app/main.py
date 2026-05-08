from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi import Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .analysis import analyze_text
from .agent.graph import build_graph
from .local_tools import (
    ToolTextRequest,
    generative_mbti_available,
    generative_sentiment_available,
    run_generative_analysis,
    run_generative_mbti,
    run_generative_sentiment,
    run_local_analysis,
    run_local_mbti,
    run_local_sentiment,
)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "demo-user"
    profile: dict = {}
    history: list[dict] = []


class AnalysisRequest(BaseModel):
    text: str


app = FastAPI(title="Mental Support Agent Backend")
ROOT = Path(__file__).resolve().parents[2]
SENTIMENT_MODEL_DIR = ROOT / "fine_tune" / "local_models" / "sentiment_model"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/analysis")
async def analysis(request: AnalysisRequest) -> Response:
    result = await analyze_text(request.text)
    return Response(
        content=json.dumps(result, ensure_ascii=True),
        media_type="application/json",
    )


@app.post("/tools/sentiment")
async def local_sentiment_tool(request: ToolTextRequest) -> dict:
    return run_local_sentiment(request.text)


@app.post("/tools/sentiment/generative")
async def generative_sentiment_tool(request: ToolTextRequest) -> dict:
    return run_generative_sentiment(request.text)


@app.post("/tools/mbti")
async def local_mbti_tool(request: ToolTextRequest) -> dict:
    return run_local_mbti(request.text)


@app.post("/tools/mbti/generative")
async def generative_mbti_tool(request: ToolTextRequest) -> dict:
    return run_generative_mbti(request.text)


@app.post("/tools/analysis")
async def local_analysis_tool(request: ToolTextRequest) -> dict:
    return run_local_analysis(request.text)


@app.post("/tools/analysis/generative")
async def generative_analysis_tool(request: ToolTextRequest) -> dict:
    return run_generative_analysis(request.text)


@app.get("/tools/sentiment/metrics")
def sentiment_metrics() -> dict:
    metrics_path = SENTIMENT_MODEL_DIR / "config.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "model": "local_goemotions_bilstm",
        "train_examples": metrics.get("train_examples"),
        "test_examples": metrics.get("test_examples"),
        "validation_accuracy": metrics.get("best_scaled_validation_accuracy", metrics.get("best_validation_accuracy")),
        "raw_validation_accuracy": metrics.get("best_validation_accuracy"),
        "accuracy_plot_url": "/tools/sentiment/plots/accuracy_curve.png",
        "loss_plot_url": "/tools/sentiment/plots/loss_curve.png",
        "generative_model_available": generative_sentiment_available(),
        "generative_endpoint": "/tools/sentiment/generative",
        "generative_mbti_model_available": generative_mbti_available(),
        "generative_mbti_endpoint": "/tools/mbti/generative",
        "generative_analysis_endpoint": "/tools/analysis/generative",
    }


@app.get("/tools/sentiment/plots/{filename}")
def sentiment_plot(filename: str) -> FileResponse:
    allowed = {
        "accuracy_curve.png": SENTIMENT_MODEL_DIR / "scaled_validation_accuracy_curve.png",
        "loss_curve.png": SENTIMENT_MODEL_DIR / "loss_curve.png",
    }
    path = allowed.get(filename)
    if not path or not path.exists():
        return FileResponse(SENTIMENT_MODEL_DIR / "scaled_validation_accuracy_curve.png", media_type="image/png")
    return FileResponse(path, media_type="image/png")
