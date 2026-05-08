from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fine_tune" / "public" / "sentiment_legacy_demo_train.jsonl"

SYSTEM_PROMPT = (
    "You are a generative sentiment-analysis model for a mental-support assistant. "
    "Return exactly one strict JSON object. Do not output markdown, bullets, explanations, "
    "diagnosis, disease probability, or treatment decisions. "
    "Required fields: task, primary_emotion, secondary_emotions, valence, intensity, "
    "stress_level, risk_hint, evidence, support_focus."
)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    examples = build_examples()
    OUTPUT.write_text("\n".join(to_chat_line(item) for item in examples) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "examples": len(examples)}, indent=2))


def build_examples() -> list[dict[str, Any]]:
    base = [
        sleep_case(
            "我最近总是睡不着，白天很累。",
            "difficulty falling asleep and daytime fatigue",
        ),
        sleep_case(
            "I have been having trouble falling asleep lately, and I feel very tired during the day.",
            "increased daytime fatigue and difficulty falling asleep",
        ),
        sleep_case(
            "I keep waking up at night, and during the day I feel exhausted and tense.",
            "broken sleep, daytime exhaustion, and tension",
        ),
        sleep_case(
            "These days I lie awake for hours. I feel drained in the morning and anxious about my work.",
            "long sleep latency, morning fatigue, and work anxiety",
        ),
        sleep_case(
            "I cannot sleep well before exams, and the next day my body feels heavy and tired.",
            "exam-related sleep difficulty and physical fatigue",
        ),
        positive_case(
            "My advisor praised my experiment notes today. I felt happy and seen.",
            "positive_feedback",
            "joy",
            ["validation", "pride"],
            "advisor praised experiment notes, felt happy and seen",
            "positive emotion reinforcement",
        ),
        positive_case(
            "I finally finished my presentation. I was nervous before, but now I feel relieved and proud.",
            "presentation_stress",
            "relief",
            ["pride", "nervousness"],
            "finished presentation, nervous before, relieved and proud now",
            "positive emotion reinforcement",
        ),
        relationship_case(
            "My roommate and I keep arguing about our schedule. I feel annoyed, hurt, and ignored.",
            "arguing about schedule, feeling annoyed and ignored",
        ),
        relationship_case(
            "My friend has been dismissing my messages, and I feel frustrated and a little lonely.",
            "dismissed messages, frustration, and loneliness",
        ),
        academic_case(
            "I have too many deadlines this week, and I feel overwhelmed and tense.",
            "too many deadlines, overwhelm, and tension",
        ),
        crisis_case(
            "I do not want to live anymore, and I keep looking at the pills on my desk.",
            "not wanting to live and looking at pills",
        ),
    ]

    augmented: list[dict[str, Any]] = []
    for _ in range(8):
        augmented.extend(base)
    return augmented


def sleep_case(text: str, evidence: str) -> dict[str, Any]:
    return {
        "input": text,
        "output": {
            "task": "sleeping",
            "primary_emotion": "stress",
            "secondary_emotions": ["fatigue", "anxiety"],
            "valence": "low",
            "intensity": "high",
            "stress_level": "high",
            "risk_hint": "increase in sleep issues",
            "evidence": evidence,
            "support_focus": "sleep hygiene",
        },
    }


def positive_case(
    text: str,
    task: str,
    primary: str,
    secondary: list[str],
    evidence: str,
    focus: str,
) -> dict[str, Any]:
    return {
        "input": text,
        "output": {
            "task": task,
            "primary_emotion": primary,
            "secondary_emotions": secondary,
            "valence": "positive",
            "intensity": "low",
            "stress_level": "low",
            "risk_hint": "none",
            "evidence": evidence,
            "support_focus": focus,
        },
    }


def relationship_case(text: str, evidence: str) -> dict[str, Any]:
    return {
        "input": text,
        "output": {
            "task": "relationship_stress",
            "primary_emotion": "annoyance",
            "secondary_emotions": ["hurt", "feeling ignored"],
            "valence": "negative",
            "intensity": "moderate",
            "stress_level": "moderate",
            "risk_hint": "none",
            "evidence": evidence,
            "support_focus": "communication boundary clarification",
        },
    }


def academic_case(text: str, evidence: str) -> dict[str, Any]:
    return {
        "input": text,
        "output": {
            "task": "academic_stress",
            "primary_emotion": "stress",
            "secondary_emotions": ["overwhelm", "tension"],
            "valence": "negative",
            "intensity": "high",
            "stress_level": "high",
            "risk_hint": "elevated_distress",
            "evidence": evidence,
            "support_focus": "task prioritization",
        },
    }


def crisis_case(text: str, evidence: str) -> dict[str, Any]:
    return {
        "input": text,
        "output": {
            "task": "crisis_signal",
            "primary_emotion": "despair",
            "secondary_emotions": ["hopelessness", "acute distress"],
            "valence": "negative",
            "intensity": "crisis",
            "stress_level": "crisis",
            "risk_hint": "crisis",
            "evidence": evidence,
            "support_focus": "immediate safety support",
        },
    }


def to_chat_line(example: dict[str, Any]) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": json.dumps(example["output"], ensure_ascii=False)},
            ]
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
