async function callVolcengineChat({ model, messages, temperature = 0.4 }) {
  const endpoint =
    process.env.ARK_BASE_URL || "https://ark.cn-beijing.volces.com/api/v3/chat/completions";

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.ARK_API_KEY}`
    },
    body: JSON.stringify({
      model,
      messages,
      temperature
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Volcengine API error ${response.status}: ${errorText}`);
  }

  const payload = await response.json();
  const content = payload?.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error("Volcengine API returned empty content");
  }
  return content;
}

module.exports = {
  callVolcengineChat
};
