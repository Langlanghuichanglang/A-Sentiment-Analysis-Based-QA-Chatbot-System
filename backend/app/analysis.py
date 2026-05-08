from __future__ import annotations

import json
import os
from typing import Any

import httpx


ARK_BASE_URL = os.getenv(
    "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
)

SENTIMENT_SYSTEM_PROMPT = (
    "You are a sentiment-analysis model inside a mental-support assistant. "
    "Return strict JSON only. Do not diagnose, estimate disease probability, "
    "or provide treatment decisions."
)

MBTI_SYSTEM_PROMPT = (
    "You are an MBTI-style communication-preference analysis model inside a "
    "mental-support assistant. Return strict JSON only. Use MBTI only as a "
    "communication-preference signal, not as a fixed personality label."
)


async def analyze_text(text: str) -> dict[str, Any]:
    sentiment_model = os.getenv("ARK_SENTIMENT_MODEL")
    mbti_model = os.getenv("ARK_MBTI_MODEL")
    api_key = os.getenv("ARK_API_KEY")

    if not api_key or not sentiment_model or not mbti_model:
        from .local_tools import run_local_analysis

        return run_local_analysis(text)

    async with httpx.AsyncClient(timeout=30) as client:
        sentiment = await call_ark_json(
            client=client,
            api_key=api_key,
            model=sentiment_model,
            system_prompt=SENTIMENT_SYSTEM_PROMPT,
            text=text,
        )
        mbti = await call_ark_json(
            client=client,
            api_key=api_key,
            model=mbti_model,
            system_prompt=MBTI_SYSTEM_PROMPT,
            text=text,
        )

    return {
        "sentiment": sentiment,
        "mbti": mbti,
        "provider": "volcengine_ark",
    }


async def call_ark_json(
    *,
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    system_prompt: str,
    text: str,
) -> dict[str, Any]:
    response = await client.post(
        ARK_BASE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0.1,
        },
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)
