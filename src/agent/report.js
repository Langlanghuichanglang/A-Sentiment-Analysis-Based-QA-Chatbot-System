const { normalizeProfile } = require("./profile");
const { triageRisk } = require("./safety");

function generateReport({ history = [], memory = [], profile = {} }) {
  const normalizedProfile = normalizeProfile(profile);
  const userText = history
    .filter((item) => item.role === "user")
    .map((item) => item.content)
    .join("\n");

  const risk = triageRisk(userText, []);
  const signals = detectSignals(userText);

  return {
    title: "心理筛查与自我观察报告",
    generatedAt: new Date().toISOString(),
    disclaimer:
      "本报告不是医学诊断，也不能替代心理咨询、精神科评估或紧急服务。它只整理对话中出现的线索，用于课程演示和自我观察。",
    risk,
    emotionThemes: buildEmotionThemes(signals),
    stressors: buildStressors(signals),
    screeningObservations: buildScreeningObservations(signals),
    suggestedExercises: buildExercises(signals, risk),
    mbtiNote: buildMbtiNote(normalizedProfile),
    memoryPolicy: {
      enabled: normalizedProfile.memoryEnabled,
      storedSummaryCount: memory.length,
      note: "仅保存用户允许的脱敏摘要，不保存诊断标签。"
    }
  };
}

function detectSignals(text) {
  return {
    lowMood: countMatches(text, /低落|难过|没意思|空虚|绝望|麻木/g),
    anxiety: countMatches(text, /焦虑|紧张|担心|害怕|慌|压力/g),
    sleep: countMatches(text, /失眠|睡不着|早醒|嗜睡|睡眠/g),
    selfBlame: countMatches(text, /自责|没用|失败|都是我的错|拖累/g),
    social: countMatches(text, /社交|朋友|孤独|关系|家人|室友|同学/g),
    studyWork: countMatches(text, /考试|作业|绩点|论文|工作|实习|老板|ddl|截止/g),
    sensoryRoutine: countMatches(text, /感官|噪音|固定|重复|变化|规则|兴趣/g)
  };
}

function countMatches(text, regex) {
  return (String(text).match(regex) || []).length;
}

function buildEmotionThemes(signals) {
  const themes = [];
  if (signals.lowMood > 0) themes.push("出现低落、空虚或意义感下降相关表达。");
  if (signals.anxiety > 0) themes.push("出现焦虑、紧张或压力负荷相关表达。");
  if (signals.sleep > 0) themes.push("出现睡眠节律或休息质量相关困扰。");
  if (signals.selfBlame > 0) themes.push("出现自责、失败感或负面自我评价线索。");
  return themes.length ? themes : ["目前对话中尚未出现稳定的情绪主题。"];
}

function buildStressors(signals) {
  const stressors = [];
  if (signals.studyWork > 0) stressors.push("学习、工作、考试或任务截止可能是主要压力源。");
  if (signals.social > 0) stressors.push("人际关系、家庭或孤独感可能影响当前状态。");
  if (signals.sensoryRoutine > 0) stressors.push("感官敏感、固定习惯或变化压力可以作为进一步观察点。");
  return stressors.length ? stressors : ["压力源信息不足，建议继续通过对话或量表补充。"];
}

function buildScreeningObservations(signals) {
  const observations = [];
  const depressionSignals = signals.lowMood + signals.sleep + signals.selfBlame;
  const anxietySignals = signals.anxiety + signals.sleep;
  const autismTraitSignals = signals.social + signals.sensoryRoutine;

  observations.push(
    `抑郁相关线索：${levelLabel(depressionSignals)}。建议后续可加入 PHQ-9 量表模块，但不要把结果表述为诊断概率。`
  );
  observations.push(
    `焦虑相关线索：${levelLabel(anxietySignals)}。建议后续可加入 GAD-7 量表模块，用于筛查而非确诊。`
  );
  observations.push(
    `自闭特质观察线索：${levelLabel(autismTraitSignals)}。如用于演示，可加入 AQ-10 风格问卷，但正式判断需要专业评估。`
  );
  return observations;
}

function buildExercises(signals, risk) {
  if (risk.level === "crisis") {
    return ["优先联系现实中的紧急支持资源，并把接下来 10 分钟的安全步骤写下来。"];
  }

  const exercises = ["用 0-10 分记录此刻情绪强度，并写下一件最小可行动作。"];
  if (signals.anxiety > 0 || signals.sleep > 0) {
    exercises.push("尝试 5-4-3-2-1 grounding：说出 5 个看到的东西、4 个触碰到的东西、3 个听到的声音、2 个闻到的气味、1 个味觉或呼吸感受。");
  }
  if (signals.selfBlame > 0 || signals.lowMood > 0) {
    exercises.push("写一条自动想法，再写出支持它和反驳它的证据，最后改写成更平衡的想法。");
  }
  return exercises;
}

function buildMbtiNote(profile) {
  if (profile.mbtiType === "UNKNOWN") {
    return "用户未选择 MBTI，因此报告只使用显式沟通偏好，不推断人格类型。";
  }

  const notes = [];
  if (profile.preferences.prefersStructure) notes.push("更适合清晰步骤和计划。");
  if (profile.preferences.prefersFlexibility) notes.push("更适合开放选项和低压力尝试。");
  if (profile.preferences.prefersValidation) notes.push("回应中应保留更多情绪确认。");
  if (profile.preferences.prefersLogic) notes.push("回应中可加入更多逻辑拆解。");
  if (profile.preferences.prefersConcreteSteps) notes.push("练习建议应尽量具体。");
  if (profile.preferences.prefersReflection) notes.push("可以加入价值感和意义层面的开放问题。");

  return `${profile.mbtiType} 仅作为用户自选的沟通偏好参考，不作为诊断或风险判断依据。${notes.join("")}`;
}

function levelLabel(score) {
  if (score >= 4) return "偏高";
  if (score >= 2) return "中等";
  if (score >= 1) return "较低";
  return "未明显出现";
}

module.exports = {
  generateReport
};
