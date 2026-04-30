function retrieveRelevantMemory(input, memory = []) {
  if (!memory.length) return [];
  const tokens = tokenize(input);
  return memory
    .map((item) => ({
      ...item,
      score: tokenize(item.summary).filter((token) => tokens.includes(token)).length
    }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);
}

function extractMemoryCandidate(input, reply, risk) {
  if (risk.level === "crisis") return null;

  const sensitive = /(姓名|电话|地址|身份证|学校|宿舍|公司|住在)/.test(input);
  const personalTheme = /(压力|睡眠|焦虑|低落|关系|家庭|考试|工作|失业|分手|孤独)/.exec(input);

  if (!personalTheme) return null;

  return {
    summary: `用户近期提到与“${personalTheme[0]}”相关的困扰。`,
    sensitivity: sensitive ? "medium" : "low",
    ttlDays: sensitive ? 7 : 30,
    source: "memory_extractor",
    consentRequired: true
  };
}

function tokenize(text) {
  return String(text)
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter(Boolean);
}

module.exports = {
  retrieveRelevantMemory,
  extractMemoryCandidate
};
