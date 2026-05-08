from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fine_tune" / "public" / "mbti_legacy_demo_train.jsonl"

SYSTEM_PROMPT = (
    "You are a generative MBTI-style communication-preference analysis model "
    "for a mental-support assistant. Return exactly one strict JSON object. "
    "Do not output markdown, bullets, explanations, or fixed personality claims. "
    "Required fields: task, likely_type_hint, axis_signals, confidence, "
    "communication_preferences, evidence."
)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    examples = build_examples()
    OUTPUT.write_text("\n".join(to_chat_line(item) for item in examples) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "examples": len(examples)}, indent=2))


def build_examples() -> list[dict[str, Any]]:
    seed_cases = [
        case(
            "When I feel stuck, I like talking through different possibilities with someone. I feel better when the conversation is warm, encouraging, and open-ended rather than too structured.",
            "ENFP",
            ["talking through different possibilities", "warm and open-ended conversation"],
        ),
        case(
            "I enjoy discussing multiple angles out loud. A rigid step-by-step plan feels limiting before I have explored the possibilities.",
            "ENTP",
            ["discussing multiple angles out loud", "explored the possibilities"],
        ),
        case(
            "When I am stressed, I prefer to write things down alone before talking to anyone. I need clear steps and time to think.",
            "ISTJ",
            ["write things down alone", "clear steps and time to think"],
        ),
        case(
            "I usually need quiet time to map out the long-term pattern. I prefer a clear plan, but I also want to understand the deeper reason behind it.",
            "INTJ",
            ["quiet time", "long-term pattern", "clear plan"],
        ),
        case(
            "Please give me concrete examples, practical next steps, and a clear order. I feel calmer when things are specific and organized.",
            "ESTJ",
            ["concrete examples", "practical next steps", "clear order"],
        ),
        case(
            "I care most about whether people feel heard. Before solving the problem, I want to understand everyone's feelings and values.",
            "ENFJ",
            ["people feel heard", "feelings and values"],
        ),
        case(
            "I like to quietly reflect on what matters to me. I do not want a rushed decision; I need space to explore different meanings.",
            "INFP",
            ["quietly reflect", "space to explore different meanings"],
        ),
        case(
            "I want direct feedback, evidence, and a practical plan. Emotional reassurance helps less than a clear explanation of what works.",
            "ESTJ",
            ["direct feedback", "evidence", "practical plan"],
        ),
        case(
            "I learn best by trying things hands-on. Please show me what to do now rather than giving a broad abstract theory.",
            "ESFP",
            ["hands-on", "what to do now"],
        ),
        case(
            "I prefer a calm one-on-one conversation. Give me concrete details, but leave me enough time to process them privately.",
            "ISFJ",
            ["calm one-on-one conversation", "concrete details", "process privately"],
        ),
        case(
            "I like exploring theories and hidden patterns by myself. Too many immediate practical details can feel distracting at first.",
            "INTP",
            ["exploring theories", "hidden patterns", "by myself"],
        ),
        case(
            "I get energy from brainstorming with others. I want options, experiments, and room to change direction quickly.",
            "ENTP",
            ["brainstorming with others", "options and experiments", "change direction quickly"],
        ),
        case(
            "I need a predictable schedule and specific responsibilities. Open-ended suggestions are harder for me than a checklist.",
            "ISTJ",
            ["predictable schedule", "specific responsibilities", "checklist"],
        ),
        case(
            "I want someone to first acknowledge the emotional impact. After that, we can look at flexible options together.",
            "ENFP",
            ["acknowledge the emotional impact", "flexible options together"],
        ),
        case(
            "I prefer concise logic, pros and cons, and a decision framework. I do not need a long emotional discussion first.",
            "ENTJ",
            ["concise logic", "pros and cons", "decision framework"],
        ),
        case(
            "I notice small changes in routine quickly. Concrete examples from daily life help me more than abstract metaphors.",
            "ISFJ",
            ["changes in routine", "concrete examples from daily life"],
        ),
    ]

    examples: list[dict[str, Any]] = []
    for _ in range(8):
        examples.extend(seed_cases)
    return examples


def case(text: str, mbti_type: str, evidence: list[str]) -> dict[str, Any]:
    axis = {
        "IE": mbti_type[0],
        "NS": mbti_type[1],
        "TF": mbti_type[2],
        "JP": mbti_type[3],
    }
    return {
        "input": text,
        "output": {
            "task": "mbti_preference_analysis",
            "likely_type_hint": mbti_type,
            "axis_signals": axis,
            "confidence": "medium",
            "communication_preferences": preferences_from_axis(axis),
            "evidence": evidence,
        },
    }


def preferences_from_axis(axis: dict[str, str]) -> list[str]:
    return [
        "allow private reflection" if axis["IE"] == "I" else "allow interactive discussion",
        "provide concrete steps" if axis["NS"] == "S" else "explore possibilities and patterns",
        "emphasize logic and actionable options" if axis["TF"] == "T" else "validate feelings and values first",
        "clarify plans and priorities" if axis["JP"] == "J" else "preserve flexibility and room to explore",
    ]


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
