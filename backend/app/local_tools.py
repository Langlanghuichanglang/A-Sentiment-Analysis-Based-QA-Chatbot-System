from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from sklearn.exceptions import InconsistentVersionWarning

from .sentiment_model import load_model_bundle, predict_emotion


ROOT = Path(__file__).resolve().parents[2]
SENTIMENT_MODEL_DIR = ROOT / "fine_tune" / "local_models" / "sentiment_model"
GENERATIVE_SENTIMENT_ADAPTER = ROOT / "fine_tune" / "local_llm" / "sentiment_legacy_demo_lora"
GENERATIVE_MBTI_ADAPTER = ROOT / "fine_tune" / "local_llm" / "mbti_legacy_demo_lora"
MBTI_MODEL_DIR = ROOT / "fine_tune" / "local_models" / "kaggle_mbti_axes"


class ToolTextRequest(BaseModel):
    text: str = Field(..., min_length=1)


class SentimentResult(BaseModel):
    task: str
    predicted_emotion: str | None = None
    predicted_score: float | None = None
    emotion_scores: list[dict[str, Any]] | None = None
    primary_emotion: str | None = None
    secondary_emotions: list[str] | None = None
    valence: str | None = None
    intensity: int | None = None
    stress_level: str | None = None
    risk_hint: str | None = None
    evidence: list[str] | None = None
    support_focus: list[str] | None = None
    boundary_note: str | None = None
    provider: str | None = None


class MbtiResult(BaseModel):
    task: Literal["mbti_preference_analysis"]
    likely_type_hint: str
    axis_signals: dict[str, str]
    confidence: float | str
    communication_preferences: list[str]
    evidence: list[str]
    boundary_note: str | None = None
    provider: str | None = "local_kaggle_axis_classifiers"


class LocalAnalysisResult(BaseModel):
    sentiment: SentimentResult
    mbti: MbtiResult
    provider: str = "local_models"


def run_local_sentiment(text: str) -> dict[str, Any]:
    if sentiment_model_available():
        return run_sentiment_model(text)
    raise FileNotFoundError(f"Missing local sentiment model files under {SENTIMENT_MODEL_DIR}")


def run_generative_sentiment(text: str) -> dict[str, Any]:
    result = run_qwen_lora_json(generative_sentiment_adapter_path(), "sentiment", text)
    result["provider"] = "local_qwen_generative_sentiment"
    return SentimentResult.model_validate(result).model_dump(exclude_none=True)


def run_generative_mbti(text: str) -> dict[str, Any]:
    result = run_qwen_lora_json(generative_mbti_adapter_path(), "mbti", text)
    result["provider"] = "local_qwen_generative_mbti"
    return MbtiResult.model_validate(result).model_dump(exclude_none=True)


def run_sentiment_model(text: str) -> dict[str, Any]:
    model, vocab, _ = load_sentiment_model_bundle()
    result = predict_emotion(text, model, vocab, sentiment_model_device())
    result["emotion_scores"] = sorted(result["emotion_scores"], key=lambda item: item["score"], reverse=True)
    result["provider"] = "local_goemotions_bilstm"
    return SentimentResult.model_validate(result).model_dump(exclude_none=True)


def run_local_mbti(text: str) -> dict[str, Any]:
    clean = clean_text(text)
    axes: dict[str, str] = {}
    confidences: dict[str, float] = {}

    for axis in ["IE", "NS", "TF", "JP"]:
        model = load_axis_model(axis)
        pred = model.predict([clean])[0]
        axes[axis] = pred
        if hasattr(model[-1], "predict_proba"):
            probs = model.predict_proba([clean])[0]
            labels = list(model[-1].classes_)
            confidences[axis] = float(probs[labels.index(pred)])

    # Short chatbox inputs usually contain explicit communication signals. Let those
    # override long-text classifier priors, especially on IE.
    axes.update(infer_communication_axes(clean))
    result = {
        "task": "mbti_preference_analysis",
        "likely_type_hint": "".join([axes["IE"], axes["NS"], axes["TF"], axes["JP"]]),
        "axis_signals": axes,
        "confidence": "medium" if infer_communication_axes(clean) else min(confidences.values()) if confidences else "unknown",
        "communication_preferences": preferences_from_axes(axes),
        "evidence": extract_evidence(clean),
        "boundary_note": "This is a communication-preference signal, not a formal MBTI assessment.",
        "provider": "local_kaggle_axis_classifiers",
    }
    return MbtiResult.model_validate(result).model_dump()


def run_local_analysis(text: str) -> dict[str, Any]:
    return {
        "sentiment": run_local_sentiment(text),
        "mbti": MbtiResult.model_validate(run_local_mbti(text)).model_dump(),
        "provider": "local_models",
    }


def run_generative_analysis(text: str) -> dict[str, Any]:
    return {
        "sentiment": run_generative_sentiment(text),
        "mbti": run_generative_mbti(text),
        "provider": "local_qwen_lora_models",
    }


def sentiment_model_available() -> bool:
    return all((SENTIMENT_MODEL_DIR / name).exists() for name in ["best_model.pth", "vocab.json", "config.json"])


def generative_sentiment_available() -> bool:
    return (GENERATIVE_SENTIMENT_ADAPTER / "adapter_config.json").exists()


def generative_mbti_available() -> bool:
    return (GENERATIVE_MBTI_ADAPTER / "adapter_config.json").exists()


def generative_sentiment_adapter_path() -> Path:
    configured = os.getenv("LOCAL_GENERATIVE_SENTIMENT_ADAPTER")
    path = (ROOT / configured).resolve() if configured else GENERATIVE_SENTIMENT_ADAPTER
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing generative sentiment LoRA adapter under {path}")


def generative_mbti_adapter_path() -> Path:
    configured = os.getenv("LOCAL_GENERATIVE_MBTI_ADAPTER")
    path = (ROOT / configured).resolve() if configured else GENERATIVE_MBTI_ADAPTER
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing generative MBTI LoRA adapter under {path}")


def run_qwen_lora_json(adapter: Path, task: str, text: str) -> dict[str, Any]:
    python_exe = local_llm_python()
    script = ROOT / "scripts" / "test_qwen_lora.py"
    completed = subprocess.run(
        [
            str(python_exe),
            "-X",
            "utf8",
            str(script),
            "--adapter",
            str(adapter.relative_to(ROOT)),
            "--task",
            task,
            "--text",
            text,
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"Generative {task} model failed")
    return extract_json_object(completed.stdout)


def local_llm_python() -> Path:
    configured = os.getenv("LOCAL_LLM_PYTHON")
    if configured:
        return Path(configured)
    windows_python = ROOT / ".venv-llm" / "Scripts" / "python.exe"
    if windows_python.exists():
        return windows_python
    return Path(sys.executable)


def extract_json_object(output: str) -> dict[str, Any]:
    text = output.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object found in model output: {text[:500]}")
    return json.loads(text[start : end + 1])


@lru_cache(maxsize=1)
def sentiment_model_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_sentiment_model_bundle():
    return load_model_bundle(SENTIMENT_MODEL_DIR, sentiment_model_device())


@lru_cache(maxsize=4)
def load_axis_model(axis: str):
    with (MBTI_MODEL_DIR / f"mbti_axis_{axis}.pkl").open("rb") as file:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            return pickle.load(file)


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_communication_axes(text: str) -> dict[str, str]:
    lower = text.lower()
    axes: dict[str, str] = {}

    if any(term in lower for term in ["alone", "before talking", "time to think", "write things down"]):
        axes["IE"] = "I"
    if any(
        term in lower
        for term in [
            "with someone",
            "talking through",
            "conversation",
            "interactive",
            "out loud",
            "discussing",
            "discuss ",
            "brainstorm with others",
            "someone gives me",
            "direct feedback",
        ]
    ):
        axes["IE"] = "E"

    if any(term in lower for term in ["clear steps", "concrete steps", "timeline", "specific", "practical"]):
        axes["NS"] = "S"
    if any(term in lower for term in ["possibilities", "different possibilities", "big picture", "open-ended", "explore"]):
        axes["NS"] = "N"

    if any(term in lower for term in ["clear steps", "logic", "evidence", "options", "plan"]):
        axes["TF"] = "T"
    if any(term in lower for term in ["feelings", "warm", "encouraging", "emotional", "understand my feelings", "values"]):
        axes["TF"] = "F"

    if any(term in lower for term in ["clear steps", "timeline", "schedule", "priorities", "structured"]):
        axes["JP"] = "J"
    if any(term in lower for term in ["not like being forced", "too quickly", "open-ended", "flexibility", "possibilities", "explore"]):
        axes["JP"] = "P"

    return axes


def preferences_from_axes(axes: dict[str, str]) -> list[str]:
    return [
        "allow private reflection" if axes["IE"] == "I" else "allow interactive discussion",
        "provide concrete steps" if axes["NS"] == "S" else "explore possibilities and patterns",
        "emphasize logic and actionable options" if axes["TF"] == "T" else "validate feelings and values first",
        "clarify plans and priorities" if axes["JP"] == "J" else "preserve flexibility and room to explore",
    ]


def extract_evidence(text: str) -> list[str]:
    lower = text.lower()
    phrases: list[str] = []
    patterns = [
        (r"write things down alone", "write things down alone"),
        (r"before talking to anyone", "before talking to anyone"),
        (r"clear steps", "need clear steps"),
        (r"time to think", "time to think"),
        (r"talking through different possibilities", "talking through different possibilities"),
        (r"discussing multiple angles", "discussing multiple angles"),
        (r"out loud", "out loud"),
        (r"with someone", "with someone"),
        (r"warm, encouraging", "warm and encouraging conversation"),
        (r"open-ended", "open-ended rather than too structured"),
        (r"understand my feelings", "understand my feelings"),
        (r"not like being forced", "not like being forced"),
        (r"final decision too quickly", "final decision too quickly"),
        (r"direct feedback", "direct feedback"),
    ]
    for pattern, phrase in patterns:
        if re.search(pattern, lower) and phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= 3:
            break
    return phrases or [text[:160]]
