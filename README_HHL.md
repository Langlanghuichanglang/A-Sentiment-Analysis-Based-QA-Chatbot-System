# HHL Analysis Tools Package

This folder contains the sentiment analysis and MBTI analysis work owned by HHL.

## Included

- FastAPI tool endpoints under `backend/`
- Qwen3-0.6B LoRA inference script under `scripts/test_qwen_lora.py`
- Final generative sentiment LoRA adapter:
  - `fine_tune/local_llm/sentiment_legacy_demo_lora`
- Final generative MBTI LoRA adapter:
  - `fine_tune/local_llm/mbti_legacy_demo_lora`
- Small legacy training JSONL files for reproducibility:
  - `fine_tune/public/sentiment_legacy_demo_train.jsonl`
  - `fine_tune/public/mbti_legacy_demo_train.jsonl`

## Not Included

- `.venv-llm/`
- raw public datasets
- intermediate checkpoints
- optimizer states
- old local baseline model folders

## Setup

Create a local Python environment first, then install dependencies.

```powershell
python -m venv .venv-llm
.\.venv-llm\Scripts\Activate.ps1
pip install -r backend\requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers peft trl datasets accelerate
```

The base model `Qwen/Qwen3-0.6B` is not included. It will be downloaded from HuggingFace automatically on first inference.

## Start FastAPI

```powershell
.\.venv-llm\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

## Test Sentiment

```powershell
$body = @{ text = "I have been having trouble falling asleep lately, and I feel very tired during the day." } | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/tools/sentiment/generative" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 10
```

## Test MBTI

```powershell
$body = @{ text = "I usually speak my thoughts out loud while comparing different options, and I prefer keeping the plan flexible until I see the bigger picture." } | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/tools/mbti/generative" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 10
```

## Combined Tool Endpoint

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/tools/analysis/generative" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 10
```
