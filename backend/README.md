# Backend Extension Notes

The runnable demo uses `server.js` so the project can start without installing
dependencies. This folder keeps the intended FastAPI + LangGraph backend shape
for the next phase.

Suggested next steps:

1. Install backend dependencies in a virtual environment.
2. Move the node logic from `src/agent/*` into the Python graph nodes.
3. Replace the local RAG scorer with PostgreSQL + pgvector.
4. Replace the local generator with Volcengine Ark chat completions.
5. Add evaluation cases before any fine-tuning.

The report module should remain non-diagnostic. Use screening levels and tool
scores, not disease probabilities.

## Fine-Tuned Analysis Models

After training hosted analysis models, configure:

```bash
ARK_API_KEY=...
ARK_SENTIMENT_MODEL=...
ARK_MBTI_MODEL=...
```

Then call:

```bash
POST /analysis
{"text":"I have been having trouble falling asleep lately, and I feel very tired during the day."}
```

Without those environment variables, `/analysis` uses the local sentiment and
MBTI tools.

## Local Tool APIs

Local model tools are available for Agent/tool integration:

```bash
POST /tools/sentiment
{"text":"I have been having trouble falling asleep lately, and I feel very tired during the day."}
```

```bash
POST /tools/mbti
{"text":"I enjoy discussing multiple angles out loud. A rigid step-by-step plan feels limiting before I have explored the possibilities."}
```

```bash
POST /tools/analysis
{"text":"..."}
```

`/tools/sentiment` uses the local GoEmotions BiLSTM model from
`fine_tune/local_models/sentiment_model`. Its schema follows the notebook-style
predictor:

```json
{
  "task": "sentiment_model_analysis",
  "predicted_emotion": "sadness",
  "predicted_score": 0.45,
  "emotion_scores": [
    {"emotion": "sadness", "score": 0.45}
  ],
  "provider": "local_goemotions_bilstm"
}
```

`/tools/mbti` uses the local Kaggle MBTI four-axis classifiers. Both tools return
structured JSON designed for Agent tool calls.
