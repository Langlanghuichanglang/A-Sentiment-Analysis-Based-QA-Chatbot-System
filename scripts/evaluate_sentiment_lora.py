from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from test_qwen_lora import normalize_output, strip_thinking, system_prompt, try_parse_json


ROOT = Path(__file__).resolve().parents[1]
CORE_FIELDS = ["primary_emotion", "valence", "stress_level", "risk_hint"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a local sentiment LoRA adapter with field-level accuracy."
    )
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", default="fine_tune/public/sentiment_public_train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    examples = load_examples(ROOT / args.data, args.offset, args.limit)
    if not examples:
        raise SystemExit("No examples found. Check --data, --offset, and --limit.")

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

    field_hits = {field: 0 for field in CORE_FIELDS}
    exact_hits = 0
    raw_json_hits = 0
    rows = []

    for index, example in enumerate(examples, start=args.offset):
        prediction, raw = predict(
            model=model,
            tokenizer=tokenizer,
            text=example["text"],
            max_new_tokens=args.max_new_tokens,
        )
        expected = normalize_output(json.dumps(example["expected"], ensure_ascii=False), "sentiment", example["text"])
        raw_json_hits += int(isinstance(try_parse_json(raw), dict))

        matches = {}
        for field in CORE_FIELDS:
            matches[field] = prediction.get(field) == expected.get(field)
            field_hits[field] += int(matches[field])
        exact = all(matches.values())
        exact_hits += int(exact)
        rows.append(
            {
                "index": index,
                "text": example["text"],
                "expected": {field: expected.get(field) for field in CORE_FIELDS},
                "predicted": {field: prediction.get(field) for field in CORE_FIELDS},
                "matches": matches,
                "exact_core_match": exact,
            }
        )

    total = len(rows)
    metrics = {
        "adapter": args.adapter,
        "data": args.data,
        "offset": args.offset,
        "examples": total,
        "raw_json_valid_rate": round(raw_json_hits / total, 4),
        "core_exact_match_accuracy": round(exact_hits / total, 4),
        "field_accuracy": {
            field: round(field_hits[field] / total, 4)
            for field in CORE_FIELDS
        },
        "note": (
            "This measures generated structured sentiment fields against JSONL labels. "
            "It is not the same as training mean_token_accuracy."
        ),
        "sample_errors": [row for row in rows if not row["exact_core_match"]][:5],
    }

    output_path = Path(args.output) if args.output else adapter_path / "sentiment_eval_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def load_examples(path: Path, offset: int, limit: int) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file):
            if line_index < offset:
                continue
            if len(examples) >= limit:
                break
            item = json.loads(line)
            messages = item["messages"]
            examples.append(
                {
                    "text": messages[1]["content"],
                    "expected": json.loads(messages[2]["content"]),
                }
            )
    return examples


def predict(model, tokenizer, text: str, max_new_tokens: int) -> tuple[dict, str]:
    messages = [
        {"role": "system", "content": system_prompt("sentiment")},
        {"role": "user", "content": text},
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[-1] :]
    raw = strip_thinking(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return normalize_output(raw, "sentiment", text), raw


if __name__ == "__main__":
    main()
