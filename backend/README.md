# Backend Extension Notes

The runnable demo uses `server.js` so the project can start without installing dependencies.
This folder keeps the intended FastAPI + LangGraph backend shape for the next phase.

Suggested next steps:

1. Install backend dependencies in a virtual environment.
2. Move the node logic from `src/agent/*` into the Python graph nodes.
3. Replace the local RAG scorer with PostgreSQL + pgvector.
4. Replace the local generator with Volcengine Ark chat completions.
5. Add evaluation cases before any fine-tuning.

The report module should remain non-diagnostic. Use screening levels and tool scores, not disease probabilities.
