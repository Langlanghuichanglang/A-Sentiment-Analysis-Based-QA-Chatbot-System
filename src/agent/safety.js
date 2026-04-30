const CRISIS_PATTERNS = [
  /自杀|轻生|结束生命|不想活|活不下去|死了算了|消失算了/,
  /割腕|跳楼|上吊|服药自杀|吞药|伤害自己|自残/,
  /suicide|kill myself|end my life|self[- ]?harm/i
];

const ELEVATED_PATTERNS = [
  /绝望|崩溃|撑不住|麻木|失控|空虚|没有意义/,
  /彻夜睡不着|连续失眠|无法起床|吃不下|暴食/,
  /panic|hopeless|worthless|breakdown/i
];

const DIAGNOSIS_PATTERNS = [
  /我是不是.*(抑郁症|焦虑症|自闭症|躁郁症|双相|强迫症)/,
  /(诊断|确诊|患有|几率|概率|百分之|%)/,
  /depression|autism|bipolar|adhd|diagnose/i
];

function triageRisk(input, history = []) {
  const text = [input, ...history.slice(-4).map((item) => item.content)].join("\n");
  const tags = [];

  if (CRISIS_PATTERNS.some((pattern) => pattern.test(text))) {
    tags.push("self_harm_or_immediate_danger");
    return {
      level: "crisis",
      label: "高危",
      tags,
      reason: "出现自伤、自杀或立即危险相关表达，需要进入危机回应流程。"
    };
  }

  if (ELEVATED_PATTERNS.some((pattern) => pattern.test(text))) {
    tags.push("elevated_distress");
  }

  if (DIAGNOSIS_PATTERNS.some((pattern) => pattern.test(input))) {
    tags.push("diagnosis_request");
  }

  if (tags.includes("elevated_distress")) {
    return {
      level: "elevated",
      label: "中等风险",
      tags,
      reason: "存在较强痛苦或功能受损线索，适合温和支持并建议进一步求助。"
    };
  }

  return {
    level: "normal",
    label: "常规支持",
    tags,
    reason: tags.includes("diagnosis_request")
      ? "用户提出诊断相关问题，需要给出非诊断边界。"
      : "未检测到立即危险信号。"
  };
}

function guardOutput(text) {
  const blockedPatterns = [
    /你(已经|就是|肯定|一定).*(抑郁症|自闭症|焦虑症|双相|躁郁症)/,
    /确诊为|诊断为|患有.*的概率是/,
    /建议.*(加量|减量|停药|换药|处方)/,
    /不要告诉任何人|别联系医生|别求助/
  ];

  const allowed = !blockedPatterns.some((pattern) => pattern.test(text));
  return {
    allowed,
    text,
    safeReplacement:
      "我不能给出诊断、处方或替代专业评估。我们可以先把你提到的感受和生活影响整理出来，并把它作为和专业人员沟通时的参考。"
  };
}

function buildCrisisResponse(risk) {
  return [
    "我很在意你刚才提到的危险信号。现在最重要的不是继续分析原因，而是先保证你处在安全的环境里。",
    "如果你有立即伤害自己的可能，请马上联系当地急救电话、学校心理中心、身边可信赖的人，或直接去最近的急诊/心理危机干预机构。",
    "你也可以先做一个很小的动作：把可能伤害自己的物品放远，离开封闭空间，给一个真实的人发消息说“我现在不安全，需要你陪我一下”。",
    "我可以继续陪你把接下来 10 分钟怎么安全度过写成步骤，但我不能替代紧急服务或专业人员。"
  ].join("\n\n");
}

module.exports = {
  triageRisk,
  guardOutput,
  buildCrisisResponse
};
