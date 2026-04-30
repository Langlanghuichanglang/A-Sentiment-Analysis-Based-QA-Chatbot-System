# Fine-tune Plan

Fine-tuning should be phase two, after prompt, RAG, safety routing, and evaluation are stable.

## Good Fine-tune Targets

- Response tone consistency
- Refusal style for diagnosis and medication requests
- Structured report wording
- Crisis boundary phrasing
- Tool-call formatting

## Bad Fine-tune Targets

- Predicting mental disorder labels from chat
- Learning from raw user conversations
- Producing disease probabilities
- Replacing risk triage with a black-box model

## Dataset Shape

Use synthetic or consent-cleared examples only.

Each example should contain:

- user input
- risk label
- allowed support style
- desired assistant reply
- blocked behavior notes

The helper script `scripts/prepare_finetune_dataset.js` converts draft examples into JSONL chat format.
