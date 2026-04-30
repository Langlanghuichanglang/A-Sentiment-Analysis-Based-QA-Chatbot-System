const SESSION_KEY = "mental-support-agent-session-id";

const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const reportButton = document.querySelector("#report-button");
const reportContent = document.querySelector("#report-content");
const resetButton = document.querySelector("#reset-button");
const crisisButton = document.querySelector("#crisis-button");

const sessionId =
  localStorage.getItem(SESSION_KEY) ||
  `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
localStorage.setItem(SESSION_KEY, sessionId);

function currentProfile() {
  return {
    mbtiType: document.querySelector("#mbti").value,
    communicationStyle: document.querySelector("#style").value,
    memoryEnabled: document.querySelector("#memory-enabled").checked
  };
}

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function addMeta(messageEl, meta) {
  const row = document.createElement("div");
  row.className = "meta-row";

  if (meta.risk) {
    const risk = document.createElement("span");
    risk.className = "chip";
    risk.textContent = `风险：${meta.risk.label}`;
    row.appendChild(risk);
  }

  for (const citation of meta.citations || []) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = citation.title;
    row.appendChild(chip);
  }

  if (row.children.length > 0) {
    messageEl.appendChild(row);
  }
}

async function parseSseStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";

    for (const part of parts) {
      const eventLine = part.split("\n").find((line) => line.startsWith("event:"));
      const dataLine = part.split("\n").find((line) => line.startsWith("data:"));
      if (!eventLine || !dataLine) continue;
      const event = eventLine.replace("event:", "").trim();
      const data = JSON.parse(dataLine.replace("data:", "").trim());
      onEvent(event, data);
    }
  }
}

async function sendMessage(message) {
  addMessage("user", message);
  const assistantEl = addMessage("assistant", "");
  let meta = null;

  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      sessionId,
      message,
      profile: currentProfile()
    })
  });

  if (!response.ok || !response.body) {
    assistantEl.textContent = "抱歉，服务暂时没有响应。";
    return;
  }

  await parseSseStream(response, (event, data) => {
    if (event === "meta") {
      meta = data;
      return;
    }
    if (event === "chunk") {
      assistantEl.textContent += data.text;
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return;
    }
    if (event === "done" && meta) {
      addMeta(assistantEl, meta);
    }
  });
}

function renderList(items) {
  if (!items || items.length === 0) return "<p class=\"empty-state\">暂无明显线索。</p>";
  return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function generateSelfObservationReport() {
  const response = await fetch("/api/report", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      sessionId,
      profile: currentProfile()
    })
  });

  const report = await response.json();
  const riskClass = `risk-${report.risk.level}`;
  reportContent.innerHTML = `
    <section class="report-section">
      <h3>结论边界</h3>
      <p class="small-text">${escapeHtml(report.disclaimer)}</p>
    </section>
    <section class="report-section">
      <h3>当前风险</h3>
      <p class="${riskClass}">${escapeHtml(report.risk.label)}：${escapeHtml(report.risk.reason)}</p>
    </section>
    <section class="report-section">
      <h3>情绪主题</h3>
      ${renderList(report.emotionThemes)}
    </section>
    <section class="report-section">
      <h3>压力源线索</h3>
      ${renderList(report.stressors)}
    </section>
    <section class="report-section">
      <h3>筛查观察</h3>
      ${renderList(report.screeningObservations)}
    </section>
    <section class="report-section">
      <h3>建议练习</h3>
      ${renderList(report.suggestedExercises)}
    </section>
    <section class="report-section">
      <h3>MBTI 沟通偏好</h3>
      <p class="small-text">${escapeHtml(report.mbtiNote)}</p>
    </section>
  `;
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;
  inputEl.value = "";
  await sendMessage(message);
});

reportButton.addEventListener("click", generateSelfObservationReport);

resetButton.addEventListener("click", async () => {
  await fetch("/api/reset", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ sessionId })
  });
  messagesEl.innerHTML = "";
  reportContent.innerHTML =
    '<p class="empty-state">会话已清空。你可以重新开始一次新的课程演示。</p>';
});

crisisButton.addEventListener("click", () => {
  addMessage(
    "system",
    "如果你或身边的人正处于立即危险中，请优先联系当地急救电话、学校心理中心、可信赖的成年人或身边的人。本项目不会替代紧急服务。"
  );
});

addMessage(
  "assistant",
  "你好，我是心理支持辅助 Agent。你可以把我当成一个用于课程演示的情绪整理工具。我可以陪你梳理压力、做 CBT 风格的小练习，并在对话后生成非诊断性质的筛查与自我观察报告。"
);
