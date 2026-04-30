const TOOL_SPECS = [
  {
    name: "search_knowledge_base",
    description: "Search curated psychoeducation, CBT, grounding, and crisis-resource documents. Never returns diagnostic conclusions.",
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string" },
        topic: {
          type: "string",
          enum: ["cbt", "grounding", "crisis", "sleep", "emotion", "all"]
        },
        risk_level: {
          type: "string",
          enum: ["normal", "elevated", "crisis"]
        }
      },
      required: ["query"]
    }
  },
  {
    name: "get_crisis_resources",
    description: "Return region-aware crisis resources and emergency guidance.",
    input_schema: {
      type: "object",
      properties: {
        locale: { type: "string", description: "User locale, such as zh-CN or en-US." }
      },
      required: ["locale"]
    }
  },
  {
    name: "suggest_grounding_exercise",
    description: "Suggest a low-risk grounding or breathing exercise for acute stress.",
    input_schema: {
      type: "object",
      properties: {
        emotion: { type: "string" },
        intensity: { type: "integer", minimum: 1, maximum: 10 }
      },
      required: ["emotion"]
    }
  },
  {
    name: "save_memory_candidate",
    description: "Save a user-approved, redacted memory summary. Raw chat and PII must not be stored here.",
    input_schema: {
      type: "object",
      properties: {
        summary: { type: "string" },
        sensitivity: { type: "string", enum: ["low", "medium", "high"] },
        ttl_days: { type: "integer", minimum: 1, maximum: 90 }
      },
      required: ["summary", "sensitivity", "ttl_days"]
    }
  },
  {
    name: "generate_self_observation_report",
    description: "Generate a non-diagnostic mental health screening and self-observation report.",
    input_schema: {
      type: "object",
      properties: {
        include_screening_observations: { type: "boolean" },
        include_mbti_preferences: { type: "boolean" }
      }
    }
  }
];

module.exports = {
  TOOL_SPECS
};
