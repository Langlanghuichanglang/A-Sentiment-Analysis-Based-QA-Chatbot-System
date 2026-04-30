const knowledgeBase = require("../../data/knowledge_base.json");

function retrieveKnowledge(query, profile, limit = 3) {
  const normalized = query.toLowerCase();
  const scored = knowledgeBase.map((item) => ({
    ...item,
    score: scoreItem(item, normalized, profile)
  }));

  return scored
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

function scoreItem(item, query, profile) {
  let score = 0;
  for (const keyword of item.keywords) {
    if (query.includes(keyword.toLowerCase())) {
      score += 3;
    }
  }

  if (item.topic === "cbt" && /想法|压力|自责|失败|担心|害怕/.test(query)) {
    score += 2;
  }

  if (item.topic === "grounding" && /焦虑|慌|失眠|呼吸|紧张/.test(query)) {
    score += 2;
  }

  if (profile?.preferences?.prefersStructure && item.topic === "cbt") {
    score += 1;
  }

  return score;
}

module.exports = {
  retrieveKnowledge
};
