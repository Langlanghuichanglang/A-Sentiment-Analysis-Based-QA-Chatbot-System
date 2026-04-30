const { retrieveKnowledge } = require("./rag");
const { extractMemoryCandidate, retrieveRelevantMemory } = require("./memory");
const { normalizeProfile } = require("./profile");
const { triageRisk, guardOutput, buildCrisisResponse } = require("./safety");
const { generateSupportResponse } = require("../llm/provider");

async function runAgentTurn({ message, history, memory, profile }) {
  const normalizedProfile = normalizeProfile(profile);
  const state = {
    input: message,
    history: history || [],
    memory: memory || [],
    profile: normalizedProfile,
    events: []
  };

  state.risk = triageRisk(state.input, state.history);
  state.events.push({ node: "risk_triage", output: state.risk });

  if (state.risk.level === "crisis") {
    const reply = buildCrisisResponse(state.risk);
    return {
      reply,
      risk: state.risk,
      plan: ["crisis_response_template"],
      citations: [],
      memoryCandidate: null
    };
  }

  state.relevantMemory = retrieveRelevantMemory(state.input, state.memory);
  state.events.push({ node: "retrieve_user_memory", output: state.relevantMemory });

  state.rag = retrieveKnowledge(state.input, state.profile);
  state.events.push({ node: "retrieve_psychoeducation_rag", output: state.rag });

  state.plan = planIntervention(state);
  state.events.push({ node: "intervention_planner", output: state.plan });

  const draft = await generateSupportResponse(state);
  state.events.push({ node: "llm_response", output: { length: draft.length } });

  const guard = guardOutput(draft);
  state.events.push({ node: "output_guard", output: guard });

  const reply = guard.allowed ? guard.text : guard.safeReplacement;
  const memoryCandidate = extractMemoryCandidate(state.input, reply, state.risk);
  state.events.push({ node: "memory_candidate_extractor", output: memoryCandidate });

  return {
    reply,
    risk: state.risk,
    plan: state.plan.steps,
    citations: state.rag.map((item) => ({
      id: item.id,
      title: item.title,
      source: item.source
    })),
    memoryCandidate
  };
}

function planIntervention(state) {
  const needsBoundary = state.risk.tags.includes("diagnosis_request");
  const style = state.profile.preferences;
  const steps = [];

  if (needsBoundary) {
    steps.push("diagnosis_boundary");
  }
  steps.push("emotional_validation");

  if (style.prefersStructure) {
    steps.push("structured_cbt_reframe");
  } else if (style.prefersReflection) {
    steps.push("open_reflection_question");
  } else {
    steps.push("small_next_step");
  }

  if (state.rag.some((item) => item.topic === "grounding")) {
    steps.push("grounding_exercise");
  }

  return {
    mode: needsBoundary ? "support_with_boundary" : "support",
    steps
  };
}

function segmentForStreaming(text) {
  const segments = [];
  let buffer = "";

  for (const char of text) {
    buffer += char;
    if (/[。！？!?]/.test(char) || buffer.length >= 120) {
      segments.push(buffer);
      buffer = "";
    }
  }

  if (buffer.trim()) {
    segments.push(buffer);
  }

  return segments.length ? segments : [text];
}

module.exports = {
  runAgentTurn,
  segmentForStreaming
};
