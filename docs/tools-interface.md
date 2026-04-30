# Tools Interface Design

The agent uses a small whitelist of tools. Tools should be invoked by the backend only after schema validation.

## Tools

1. `search_knowledge_base`
   - Purpose: retrieve curated CBT, grounding, crisis, sleep, and psychoeducation documents.
   - Constraint: must not return diagnostic conclusions.

2. `get_crisis_resources`
   - Purpose: return locale-aware emergency and crisis support resources.
   - Constraint: should not claim that the system has contacted emergency services.

3. `suggest_grounding_exercise`
   - Purpose: suggest low-risk grounding or breathing exercises.
   - Constraint: avoid medical claims.

4. `save_memory_candidate`
   - Purpose: store user-approved redacted summaries.
   - Constraint: never store raw chat, name, phone, address, ID numbers, or diagnosis labels.

5. `generate_self_observation_report`
   - Purpose: generate a non-diagnostic screening and self-observation report.
   - Constraint: no disease probability.

The runnable JSON schemas live in `src/agent/tools.js`.
