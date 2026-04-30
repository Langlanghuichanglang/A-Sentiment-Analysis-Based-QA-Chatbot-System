# 心理支持辅助 Agent 架构

## Product Boundary

This project is a mental support and self-observation assistant for coursework. It must not claim to diagnose, prescribe, or replace clinicians.

The report is named "心理筛查与自我观察报告" instead of "诊断报告". It can include:

- Emotion themes
- Stressor clues
- Screening observations
- Risk level
- Suggested CBT or grounding exercises
- MBTI-based communication preferences

It must not include:

- Disease probability, such as "80% depression"
- Diagnosis labels, such as "you have depression"
- Medication instructions

## Runtime Flow

```text
user_input
-> risk_triage
-> crisis_response OR retrieve_user_memory
-> retrieve_psychoeducation_rag
-> intervention_planner
-> llm_response
-> output_guard
-> sentence_level_streaming
-> memory_candidate_extractor
```

## Current Implementation

- `server.js`: zero-dependency Node server for demo.
- `public/*`: chatbox UI and report panel.
- `src/agent/*`: runnable JavaScript agent nodes.
- `backend/app/*`: FastAPI + LangGraph target shape.
- `data/knowledge_base.json`: curated demo RAG content.

## Future Replacement Points

- Replace `src/llm/provider.js` with Volcengine Ark streaming calls.
- Replace `src/agent/rag.js` with PostgreSQL + pgvector.
- Replace the JavaScript graph with `backend/app/agent/graph.py` once dependencies are installed.
- Add calibrated screening questionnaires before any fine-tuning.
