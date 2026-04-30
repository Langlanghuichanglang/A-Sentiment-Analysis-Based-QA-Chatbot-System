const { callVolcengineChat } = require("./volcengine");

async function generateSupportResponse(state) {
  if (process.env.ARK_API_KEY && process.env.ARK_MODEL) {
    try {
      return await generateWithVolcengine(state);
    } catch (error) {
      console.warn("Volcengine call failed, falling back to local generator:", error.message);
    }
  }
  return generateLocalResponse(state);
}

async function generateWithVolcengine(state) {
  const messages = [
    {
      role: "system",
      content: buildSystemPrompt(state)
    },
    ...state.history.slice(-8).map((item) => ({
      role: item.role,
      content: item.content
    })),
    {
      role: "user",
      content: state.input
    }
  ];

  const response = await callVolcengineChat({
    model: process.env.ARK_MODEL,
    messages,
    temperature: 0.4
  });

  return response;
}

function buildSystemPrompt(state) {
  return [
    "你是一个心理支持辅助 Agent，用于课程演示。",
    "你不能提供医学诊断、处方、治疗决策或疾病概率。",
    "你可以提供情绪确认、CBT 风格提问、正念/grounding 练习和求助建议。",
    `用户 MBTI/偏好仅用于沟通风格：${JSON.stringify(state.profile.preferences)}。`,
    `风险分诊结果：${state.risk.label}。`,
    `可参考知识：${state.rag.map((item) => `${item.title}: ${item.summary}`).join("\n")}`,
    "回答应温和、具体、边界清楚。"
  ].join("\n");
}

function generateLocalResponse(state) {
  const parts = [];
  const prefs = state.profile.preferences;
  const needsBoundary = state.risk.tags.includes("diagnosis_request");

  if (needsBoundary) {
    parts.push(
      "我不能根据这段对话判断你是否患有某种障碍，也不能给出百分比式诊断概率。更稳妥的做法是把它当作筛查线索和自我观察材料。"
    );
  }

  if (prefs.prefersValidation || state.risk.level === "elevated") {
    parts.push("听起来你最近真的承受了不少东西，这种状态值得被认真对待，而不是简单归结为“想太多”。");
  } else {
    parts.push("我们可以先把这件事拆开看，减少它在脑子里混成一团的感觉。");
  }

  if (state.rag.length) {
    parts.push(`我会优先参考“${state.rag[0].title}”这类低风险方法。`);
  }

  if (prefs.prefersStructure) {
    parts.push(
      "可以先按三步来：第一，写下最困扰你的具体场景；第二，标出当时自动冒出的想法；第三，给这个想法找一个更平衡的替代表述。"
    );
  } else if (prefs.prefersConcreteSteps) {
    parts.push(
      "现在可以先做一个很小的动作：用 0 到 10 分给此刻的压力打分，然后选一件 5 分钟内能完成的小事，比如喝水、洗脸、整理桌面的一角。"
    );
  } else if (prefs.prefersReflection) {
    parts.push(
      "也许可以先问自己一个问题：这件事真正刺痛你的地方，是失败感、被否定、失控，还是你很重视的某个需求没有被看见？"
    );
  } else {
    parts.push(
      "我们可以先从一个最小问题开始：这份压力里，哪一部分是今天必须处理的，哪一部分只是大脑在预演最坏情况？"
    );
  }

  parts.push("如果你愿意，可以继续告诉我：这件事最强烈的情绪是什么，强度大概是 0 到 10 的几分？");

  return parts.join("\n\n");
}

module.exports = {
  generateSupportResponse,
  buildSystemPrompt
};
