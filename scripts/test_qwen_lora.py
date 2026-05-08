from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--text", required=True)
    parser.add_argument("--task", choices=["sentiment", "mbti", "generic"], default="generic")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--raw", action="store_true", help="Print raw model output without normalization.")
    args = parser.parse_args()

    adapter_path = (ROOT / args.adapter).resolve()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    messages = [
        {"role": "system", "content": system_prompt(args.task)},
        {"role": "user", "content": args.text},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[-1] :]
    raw = strip_thinking(tokenizer.decode(generated, skip_special_tokens=True).strip())
    if args.raw:
        print(raw)
        return
    print(json.dumps(normalize_output(raw, args.task, args.text), ensure_ascii=False, indent=2))


def system_prompt(task: str) -> str:
    if task == "sentiment":
        return (
            "You are a sentiment analysis model for a mental-support system. "
            "Return exactly one strict JSON object. Do not output markdown, YAML, bullets, explanations, advice, "
            "diagnosis, or disease probability. Required fields: task, primary_emotion, secondary_emotions, "
            "valence, intensity, stress_level, risk_hint, evidence, support_focus. "
            "evidence must be an array of short phrases extracted or summarized from the user's text. "
            "Do not output [NAME], [SOURCE], source_label, or placeholders."
        )
    if task == "mbti":
        return (
            "You are an MBTI-style communication-preference analysis model for a mental-support system. "
            "Return exactly one strict JSON object. Do not output markdown, YAML, bullets, or explanations. "
            "Do not present MBTI as a fixed personality label. Required fields: task, likely_type_hint, "
            "axis_signals, confidence, communication_preferences, evidence, boundary_note. "
            "evidence must be an array of short phrases extracted or summarized from the user's text. "
            "Do not output [NAME], [SOURCE], source_label, or placeholders."
        )
    return "Return exactly one strict JSON object. Do not output <think>, markdown, YAML, bullets, or explanations."


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def normalize_output(raw: str, task: str, user_text: str) -> dict:
    parsed = try_parse_json(raw)
    if task == "sentiment":
        return normalize_sentiment(parsed if isinstance(parsed, dict) else parse_key_value_or_csv(raw), raw, user_text)
    if task == "mbti":
        return normalize_mbti(parsed if isinstance(parsed, dict) else parse_key_value_or_csv(raw), raw, user_text)
    return parsed if isinstance(parsed, dict) else {"raw": raw}


def try_parse_json(raw: str) -> dict | None:
    text = raw.strip()
    if not text:
        return None
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            fixed = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', candidate)
            fixed = fixed.replace("'", '"')
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    return None


def parse_key_value_or_csv(raw: str) -> dict:
    text = raw.strip()
    data: dict[str, object] = {}

    for line in text.splitlines():
        line = line.strip().strip(",")
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[normalize_key(key)] = clean_value(value)

    if data:
        return data

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2 and "," in lines[0] and "," in lines[1]:
        keys = [normalize_key(item) for item in lines[0].split(",")]
        values = [clean_value(item) for item in lines[1].split(",")]
        return dict(zip(keys, values))

    parts = [clean_value(item) for item in text.split(",") if clean_value(item)]
    if len(parts) >= 7:
        return {
            "task": parts[0],
            "primary_emotion": parts[1],
            "secondary_emotions": parts[2],
            "valence": parts[3],
            "intensity": parts[4],
            "stress_level": parts[5],
            "risk_hint": parts[6],
            "support_focus": parts[7] if len(parts) > 7 else "",
        }
    return {"raw": raw}


def normalize_key(key: str) -> str:
    key = key.strip().strip('"').strip("'").lower()
    aliases = {
        "emotion": "primary_emotion",
        "secondary_emotion": "secondary_emotions",
        "secondary emotion": "secondary_emotions",
        "risk": "risk_hint",
        "focus": "support_focus",
        "support": "support_focus",
    }
    return aliases.get(key, key.replace(" ", "_"))


def clean_value(value: object) -> str:
    return str(value).strip().strip(",").strip().strip('"').strip("'")


def normalize_sentiment(data: dict, raw: str, user_text: str) -> dict:
    primary = normalize_atom(data.get("primary_emotion") or data.get("emotion") or infer_primary_from_text(raw, user_text))
    valence = normalize_valence(data.get("valence") or infer_valence(primary, user_text))
    stress = normalize_stress(data.get("stress_level") or data.get("stress") or infer_stress(user_text, primary))
    risk = normalize_risk(data.get("risk_hint") or data.get("risk") or infer_risk(user_text, stress))
    intensity = normalize_intensity(data.get("intensity"), stress)
    if valence == "mixed" and primary in {"stress", "fatigue", "anxiety", "sadness", "fear", "emptiness"}:
        valence = "negative"
    if risk == "none" and stress == "high":
        risk = "elevated_distress"

    return {
        "task": "sentiment_analysis",
        "primary_emotion": primary,
        "secondary_emotions": normalize_secondary_emotions(
            data.get("secondary_emotions") or data.get("secondary_emotion"),
            user_text,
            primary,
        ),
        "valence": valence,
        "intensity": intensity,
        "stress_level": stress,
        "risk_hint": risk,
        "evidence": normalize_evidence(data.get("evidence"), user_text),
        "support_focus": normalize_support_focus(data.get("support_focus") or data.get("focus"), user_text, stress),
    }


def normalize_mbti(data: dict, raw: str, user_text: str) -> dict:
    axis = data.get("axis_signals") if isinstance(data.get("axis_signals"), dict) else {}
    inferred_axis = infer_mbti_axis(user_text)
    normalized_axis = {
        "IE": choose_axis(inferred_axis.get("IE"), normalize_axis(axis.get("IE") or data.get("IE"), ["I", "E"])),
        "NS": choose_axis(inferred_axis.get("NS"), normalize_axis(axis.get("NS") or data.get("NS"), ["N", "S"])),
        "TF": choose_axis(inferred_axis.get("TF"), normalize_axis(axis.get("TF") or data.get("TF") or data.get("FT"), ["T", "F"])),
        "JP": choose_axis(inferred_axis.get("JP"), normalize_axis(axis.get("JP") or data.get("JP"), ["J", "P"])),
    }
    normalized_axis = fill_unknown_mbti_axes(normalized_axis, user_text)
    likely_type = axes_to_type(normalized_axis) or normalize_mbti_type(data.get("likely_type_hint") or data.get("type"))
    return {
        "task": "mbti_preference_analysis",
        "likely_type_hint": likely_type,
        "axis_signals": normalized_axis,
        "confidence": normalize_confidence(data.get("confidence")),
        "communication_preferences": normalize_mbti_preferences(
            data.get("communication_preferences"),
            normalized_axis,
            user_text,
        ),
        "evidence": normalize_mbti_evidence(data.get("evidence"), user_text),
    }


def choose_axis(inferred: str | None, generated: str) -> str:
    if inferred in {"I", "E", "N", "S", "T", "F", "J", "P"}:
        return inferred
    return generated if generated in {"I", "E", "N", "S", "T", "F", "J", "P"} else "unknown"


def fill_unknown_mbti_axes(axis: dict[str, str], user_text: str) -> dict[str, str]:
    scores = score_mbti_axes(user_text)
    filled = dict(axis)
    if filled.get("IE") == "unknown":
        filled["IE"] = "E" if scores["IE"]["E"] >= scores["IE"]["I"] else "I"
    if filled.get("NS") == "unknown":
        filled["NS"] = "N" if scores["NS"]["N"] >= scores["NS"]["S"] else "S"
    if filled.get("TF") == "unknown":
        filled["TF"] = "F" if scores["TF"]["F"] > scores["TF"]["T"] else "T"
    if filled.get("JP") == "unknown":
        filled["JP"] = "P" if scores["JP"]["P"] > scores["JP"]["J"] else "J"
    return filled


def score_mbti_axes(user_text: str) -> dict[str, dict[str, int]]:
    lower = user_text.lower()
    scores = {
        "IE": {"I": 0, "E": 0},
        "NS": {"N": 0, "S": 0},
        "TF": {"T": 0, "F": 0},
        "JP": {"J": 0, "P": 0},
    }
    weighted_patterns = {
        ("IE", "I"): [
            ("alone", 2), ("privately", 2), ("private", 1), ("quiet time", 2),
            ("by myself", 2), ("before talking", 2), ("time to think", 2),
            ("write things down", 2), ("organize my notes", 2), ("reflect", 1),
            ("one-on-one", 1),
        ],
        ("IE", "E"): [
            ("someone", 1), ("people", 1), ("with me", 1), ("help me", 1),
            ("conversation", 1), ("out loud", 2), ("speak my thoughts", 2),
            ("talking through", 2), ("brainstorm", 2), ("with others", 2),
            ("group", 1), ("discuss", 2),
        ],
        ("NS", "S"): [
            ("concrete", 2), ("specific", 2), ("practical", 2), ("facts", 2),
            ("details", 2), ("examples", 1), ("daily life", 2), ("hands-on", 2),
            ("what to do now", 2), ("checklist", 2), ("routine", 1),
            ("step-by-step", 2),
        ],
        ("NS", "N"): [
            ("possibilities", 2), ("options", 1), ("bigger picture", 2),
            ("big picture", 2), ("meaning", 1), ("patterns", 2), ("theories", 2),
            ("abstract", 1), ("future", 1), ("long-term", 2), ("explore", 1),
            ("different angles", 2), ("multiple angles", 2),
        ],
        ("TF", "T"): [
            ("logic", 2), ("logical", 2), ("evidence", 2), ("facts", 2),
            ("pros and cons", 2), ("decision framework", 2), ("direct feedback", 2),
            ("efficient", 1), ("works", 1), ("compare", 1), ("comparing", 1),
            ("clear explanation", 2),
        ],
        ("TF", "F"): [
            ("feel", 1), ("feelings", 2), ("felt", 1), ("affected me", 2),
            ("understand me", 2), ("understand how this affected me", 3),
            ("values", 2), ("emotional", 2), ("warm", 2), ("encouraging", 2),
            ("heard", 2), ("cared for", 2), ("support", 1), ("reassurance", 1),
        ],
        ("JP", "J"): [
            ("clear order", 2), ("next step", 2), ("clear steps", 2),
            ("plan", 1), ("schedule", 2), ("timeline", 2), ("priorities", 2),
            ("structured", 2), ("predictable", 2), ("checklist", 2),
            ("final decision", 1), ("organized", 1), ("organize", 1),
        ],
        ("JP", "P"): [
            ("flexible", 2), ("flexibility", 2), ("options", 1), ("open-ended", 2),
            ("explore", 1), ("without forcing", 2), ("quick decision", 1),
            ("room to change", 2), ("change direction", 2), ("experiments", 2),
            ("not rushed", 2), ("before deciding", 1),
        ],
    }
    for (axis, side), patterns in weighted_patterns.items():
        for pattern, weight in patterns:
            if pattern in lower:
                scores[axis][side] += weight
    return scores


def normalize_atom(value: object) -> str:
    text = clean_value(value).lower().replace(" ", "_")
    aliases = {
        "happiness": "joy",
        "happy": "joy",
        "no_secondary_emotions": "",
        "low_stress": "low",
        "high_stress": "high",
        "sleeping": "stress",
        "fatigue": "fatigue",
        "tired": "fatigue",
        "emptiness": "emptiness",
    }
    return aliases.get(text, text or "unknown")


def normalize_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        text = clean_value(value)
        if not text or text in {"[]", "none", "no_secondary_emotions"}:
            return []
        items = re.split(r"[,;/|]", text)
    cleaned = [normalize_atom(item) for item in items]
    return [item for item in cleaned if item and item not in {"none", "no_secondary_emotions"}]


def normalize_valence(value: object) -> str:
    text = normalize_atom(value)
    if "positive" in text:
        return "positive"
    if "negative" in text:
        return "negative"
    if "neutral" in text:
        return "neutral"
    if "mixed" in text:
        return "mixed"
    return "mixed"


def normalize_stress(value: object) -> str:
    text = normalize_atom(value)
    if "crisis" in text:
        return "crisis"
    if "high" in text:
        return "high"
    if "moderate" in text or "medium" in text:
        return "moderate"
    if "low" in text:
        return "low"
    return "moderate"


def normalize_risk(value: object) -> str:
    text = normalize_atom(value)
    if "crisis" in text or "self_harm" in text:
        return "crisis"
    if "elevated" in text or "high" in text:
        return "elevated_distress"
    if "none" in text or "no_risk" in text:
        return "none"
    return "none"


def normalize_intensity(value: object, stress: str) -> int:
    text = normalize_atom(value)
    if text.isdigit():
        return max(1, min(4, int(text)))
    if "crisis" in text or stress == "crisis":
        return 4
    if "high" in text or stress == "high":
        return 3
    if "moderate" in text or "medium" in text or stress == "moderate":
        return 2
    return 1


def normalize_evidence(value: object, user_text: str) -> list[str]:
    items = evidence_items(value)
    invalid = {"[name]", "[source]", "source_label", "on_[source]"}
    items = [item for item in items if item.lower() not in invalid and "[name]" not in item.lower()]
    if items and not looks_like_full_source_sentence(items, user_text):
        return items[:3]
    extracted = extract_evidence_phrases(user_text)
    return combine_evidence(items, extracted, user_text)


def combine_evidence(items: list[str], extracted: list[str], user_text: str) -> list[str]:
    phrases: list[str] = []
    for item in items + extracted:
        if not item:
            continue
        if looks_like_full_source_sentence([item], user_text):
            continue
        add_unique_phrase(phrases, item)
        if len(phrases) >= 3:
            return phrases
    return extracted[:3] or [user_text[:160]]


def evidence_items(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        text = clean_value(value)
        if not text or text in {"[]", "none"}:
            return []
        raw_items = re.split(r"[,;/|]", text)
    return [clean_value(item) for item in raw_items if clean_value(item)]


def looks_like_full_source_sentence(items: list[str], user_text: str) -> bool:
    if len(items) != 1:
        return False
    item = items[0].strip().lower()
    source = user_text.strip().lower()
    return len(item) > 60 and (item == source[: len(item)] or item in source)


def extract_evidence_phrases(user_text: str) -> list[str]:
    text = user_text.strip()
    lower = text.lower()
    phrases: list[str] = []

    patterns = [
        (r"(trouble falling asleep|difficulty falling asleep|can't fall asleep|cannot fall asleep|sleeping badly|sleep problems?)", "difficulty falling asleep"),
        (r"(very tired during the day|tired during the day|daytime fatigue|feel exhausted|feeling exhausted)", "daytime fatigue"),
        (r"(felt happy|feel happy|happy)", "felt happy"),
        (r"(felt proud|feel proud|proud)", "felt proud"),
        (r"(felt seen|feel seen|felt cared for|feel cared for)", "felt seen or cared for"),
        (r"(felt relieved|feel relieved|relieved)", "felt relieved"),
        (r"(felt hopeful|feel hopeful|hopeful)", "felt hopeful"),
        (r"(felt anxious|feel anxious|anxious|anxiety)", "anxiety"),
        (r"(felt guilty|feel guilty|guilty)", "guilt"),
        (r"(felt annoyed|feel annoyed|annoyed|frustrated)", "annoyance or frustration"),
        (r"(felt ignored|feel ignored|ignored)", "felt ignored"),
        (r"(feel empty|felt empty|empty lately)", "emptiness"),
        (r"(do not want to live|don't want to live|kill myself|suicide|pills on my desk)", "self-harm or crisis signal"),
        (r"(overwhelmed|too much to handle)", "feeling overwhelmed"),
        (r"(nervous before|felt nervous|feel nervous|nervous)", "nervousness"),
        (r"(praised my .*?|advisor praised .*?|teacher praised .*?)(?:[,.]|$)", None),
        (r"(finished .*?|submitted .*?|completed .*?)(?:[,.]|$)", None),
    ]

    for pattern, label in patterns:
        match = re.search(pattern, lower)
        if not match:
            continue
        phrase = label or clean_matched_phrase(text, match.span())
        add_unique_phrase(phrases, phrase)
        if len(phrases) >= 3:
            return phrases

    clauses = re.split(r"[,.;]|\band\b|\bbut\b|\bbecause\b", text, flags=re.IGNORECASE)
    keyword_re = re.compile(
        r"\b(happy|proud|seen|relieved|hopeful|anxious|guilty|tired|sleep|empty|overwhelmed|ignored|frustrated|annoyed)\b",
        re.IGNORECASE,
    )
    for clause in clauses:
        clause = clause.strip()
        if 3 <= len(clause.split()) <= 12 and keyword_re.search(clause):
            add_unique_phrase(phrases, clause)
        if len(phrases) >= 3:
            break

    return phrases


def clean_matched_phrase(text: str, span: tuple[int, int]) -> str:
    start, end = span
    return text[start:end].strip(" ,.;")


def add_unique_phrase(phrases: list[str], phrase: str) -> None:
    cleaned = phrase.strip()
    if cleaned and cleaned.lower() not in {item.lower() for item in phrases}:
        phrases.append(cleaned)


def normalize_secondary_emotions(value: object, user_text: str, primary: str) -> list[str]:
    items = normalize_list(value)
    lower = user_text.lower()
    inferred: list[str] = []
    if any(term in lower for term in ["tired", "fatigue", "exhausted", "drained"]):
        add_unique_phrase(inferred, "fatigue")
    if any(term in lower for term in ["sleep", "falling asleep", "insomnia"]):
        add_unique_phrase(inferred, "fatigue")
    if any(term in lower for term in ["anxious", "anxiety", "worried", "nervous", "panic"]):
        add_unique_phrase(inferred, "anxiety")
    if any(term in lower for term in ["happy", "encouraged", "glad"]):
        add_unique_phrase(inferred, "joy")
    if any(term in lower for term in ["proud", "achievement", "accomplished"]):
        add_unique_phrase(inferred, "pride")
    if any(term in lower for term in ["seen", "validated", "recognized", "cared for"]):
        add_unique_phrase(inferred, "validation")
    if any(term in lower for term in ["relieved", "relief", "lighter"]):
        add_unique_phrase(inferred, "relief")
    if any(term in lower for term in ["guilty", "guilt"]):
        add_unique_phrase(inferred, "guilt")
    if any(term in lower for term in ["annoyed", "frustrated", "irritated"]):
        add_unique_phrase(inferred, "frustration")
    if any(term in lower for term in ["ignored", "dismissed"]):
        add_unique_phrase(inferred, "feeling ignored")
    if any(term in lower for term in ["empty", "meaningless", "numb"]):
        add_unique_phrase(inferred, "emptiness")
    if any(term in lower for term in ["do not want to live", "don't want to live", "kill myself", "pills", "suicide"]):
        add_unique_phrase(inferred, "despair")

    combined: list[str] = []
    for item in items + inferred:
        if item and item != primary:
            add_unique_phrase(combined, item)
    if combined:
        return combined[:4]

    if primary == "stress":
        return ["tension"]
    return []


def normalize_support_focus(value: object, user_text: str, stress: str) -> list[str]:
    items = evidence_items(value)
    valid = [
        item.replace("_", " ").strip()
        for item in items
        if item and item.lower() not in {"focus_on_emotion", "emotion labeling", "stress signal observation"}
    ]
    if valid:
        return valid[:3]

    lower = user_text.lower()
    focus: list[str] = []
    if any(term in lower for term in ["sleep", "falling asleep", "tired during the day"]):
        add_unique_phrase(focus, "sleep hygiene")
    if any(term in lower for term in ["anxious", "worried", "nervous"]):
        add_unique_phrase(focus, "anxiety tracking")
    if any(term in lower for term in ["happy", "proud", "encouraged", "seen"]):
        add_unique_phrase(focus, "positive emotion reinforcement")
    if any(term in lower for term in ["do not want to live", "pills", "suicide", "kill myself"]):
        add_unique_phrase(focus, "immediate safety support")
    if not focus:
        add_unique_phrase(focus, "emotion labeling")
    return focus[:3]


def normalize_confidence(value: object) -> str:
    text = normalize_atom(value)
    if text in {"high", "medium", "low"}:
        return text
    return "low"


def normalize_axis(value: object, allowed: list[str]) -> str:
    text = clean_value(value).upper()
    return text if text in allowed else "unknown"


def normalize_mbti_type(value: object) -> str:
    text = clean_value(value).upper()
    return text if re.match(r"^[IE][NS][TF][JP]$", text) else "unknown"


def infer_mbti_axis(user_text: str) -> dict[str, str]:
    lower = user_text.lower()
    scores = score_mbti_axes(user_text)
    axis: dict[str, str] = {}

    if any(term in lower for term in ["alone", "privately", "private", "before talking", "time to think", "write things down", "organize my notes"]):
        axis["IE"] = "I"
    if any(term in lower for term in ["with someone", "talking through", "conversation", "interactive", "out loud", "speak my thoughts", "brainstorming with others"]):
        axis["IE"] = "E"

    if any(term in lower for term in ["clear steps", "concrete steps", "timeline", "specific", "practical", "facts", "daily life"]):
        axis["NS"] = "S"
    if any(term in lower for term in ["possibilities", "different possibilities", "options", "bigger picture", "big picture", "open-ended", "explore", "patterns", "theories"]):
        axis["NS"] = "N"

    if any(term in lower for term in ["clear steps", "logic", "evidence", "plan", "facts", "pros and cons", "decision framework", "comparing"]):
        axis["TF"] = "T"
    if any(term in lower for term in ["feelings", "warm", "encouraging", "emotional", "understand my feelings", "understand how this affected me", "affected me", "values"]):
        axis["TF"] = "F"

    if any(term in lower for term in ["clear steps", "timeline", "final decision", "schedule", "priorities", "structured", "clear order", "next step"]):
        axis["JP"] = "J"
    if any(term in lower for term in ["not like being forced", "too quickly", "open-ended", "flexibility", "flexible", "possibilities", "options", "explore", "change direction"]):
        axis["JP"] = "P"

    for axis_name, sides in scores.items():
        if axis_name in axis:
            continue
        left, right = list(sides.keys())
        if sides[left] == 0 and sides[right] == 0:
            continue
        axis[axis_name] = left if sides[left] > sides[right] else right

    return axis


def axes_to_type(axis: dict[str, str]) -> str:
    letters = [axis.get("IE"), axis.get("NS"), axis.get("TF"), axis.get("JP")]
    if all(letter in {"I", "E", "N", "S", "T", "F", "J", "P"} for letter in letters):
        return "".join(letters)
    return "unknown"


def normalize_mbti_preferences(value: object, axis: dict[str, str], user_text: str) -> list[str]:
    raw_items = [item.replace("_", " ") for item in normalize_list(value)]
    generic = {"provide information", "validate evidence", "empathize and support", "communication preference analysis"}
    prefs = [item for item in raw_items if item not in generic]

    inferred: list[str] = []
    if axis.get("IE") == "I":
        add_unique_phrase(inferred, "allow private reflection")
    elif axis.get("IE") == "E":
        add_unique_phrase(inferred, "allow interactive discussion")
    if axis.get("NS") == "S":
        add_unique_phrase(inferred, "provide concrete steps")
    elif axis.get("NS") == "N":
        add_unique_phrase(inferred, "explore possibilities and patterns")
    if axis.get("TF") == "T":
        add_unique_phrase(inferred, "emphasize logic and actionable options")
    elif axis.get("TF") == "F":
        add_unique_phrase(inferred, "validate feelings and values first")
    if axis.get("JP") == "J":
        add_unique_phrase(inferred, "clarify plans and priorities")
    elif axis.get("JP") == "P":
        add_unique_phrase(inferred, "preserve flexibility and room to explore")

    combined: list[str] = []
    for item in inferred + prefs:
        add_unique_phrase(combined, item)
    return combined[:4] or ["communication preference analysis"]


def normalize_mbti_evidence(value: object, user_text: str) -> list[str]:
    items = evidence_items(value)
    items = [clean_evidence_text(item) for item in items]
    items = [item for item in items if item]
    if items and not looks_like_full_source_sentence(items, user_text):
        return items[:3]

    lower = user_text.lower()
    phrases: list[str] = []
    evidence_patterns = [
        (r"write things down alone", "write things down alone"),
        (r"before talking to anyone", "before talking to anyone"),
        (r"clear steps", "need clear steps"),
        (r"time to think", "time to think"),
        (r"understand my feelings", "understand my feelings"),
        (r"not like being forced", "not like being forced"),
        (r"final decision too quickly", "final decision too quickly"),
        (r"talking through different possibilities", "talking through different possibilities"),
        (r"speak my thoughts out loud", "speak thoughts out loud"),
        (r"comparing different options", "comparing different options"),
        (r"plan flexible", "keeping the plan flexible"),
        (r"bigger picture", "seeing the bigger picture"),
        (r"warm, encouraging", "warm and encouraging conversation"),
        (r"open-ended", "open-ended rather than too structured"),
    ]
    for pattern, phrase in evidence_patterns:
        if re.search(pattern, lower):
            add_unique_phrase(phrases, phrase)
        if len(phrases) >= 3:
            break
    return phrases or [user_text[:160]]


def clean_evidence_text(text: str) -> str:
    return (
        clean_value(text)
        .strip()
        .strip("[]")
        .strip()
        .strip('"')
        .strip("'")
        .replace('\\"]', "")
        .replace('"].', "")
        .replace('"]', "")
        .replace("\\", "")
        .rstrip("].")
        .strip()
    )


def infer_primary_from_text(raw: str, user_text: str = "") -> str:
    lower = f"{raw} {user_text}".lower()
    for candidate in ["joy", "happiness", "fatigue", "anxiety", "stress", "sadness", "anger", "fear", "emptiness", "neutral"]:
        if candidate in lower:
            return normalize_atom(candidate)
    if any(term in lower for term in ["sleep", "tired", "exhausted"]):
        return "stress"
    return "unknown"


def infer_valence(primary: str, user_text: str) -> str:
    lower = f"{primary} {user_text}".lower()
    if any(term in lower for term in ["happy", "joy", "proud", "relieved", "hopeful", "encouraged"]):
        return "positive"
    if any(term in lower for term in ["anxious", "sad", "empty", "angry", "stress", "tired", "guilty"]):
        return "negative"
    return "mixed"


def infer_stress(user_text: str, primary: str) -> str:
    lower = f"{primary} {user_text}".lower()
    if any(term in lower for term in ["do not want to live", "kill myself", "pills on my desk", "suicide"]):
        return "crisis"
    if any(term in lower for term in ["sleep", "tired", "empty", "overwhelmed", "anxious"]):
        return "high"
    if any(term in lower for term in ["nervous", "annoyed", "guilty", "worried"]):
        return "moderate"
    return "low"


def infer_risk(user_text: str, stress: str) -> str:
    lower = user_text.lower()
    if any(term in lower for term in ["do not want to live", "kill myself", "pills on my desk", "suicide"]):
        return "crisis"
    if stress in {"high", "crisis"}:
        return "elevated_distress" if stress == "high" else "crisis"
    return "none"


if __name__ == "__main__":
    main()
