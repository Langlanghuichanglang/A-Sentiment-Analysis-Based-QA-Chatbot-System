const { readFileSync, writeFileSync } = require("node:fs");
const path = require("node:path");

const inputPath = path.join(__dirname, "..", "fine_tune", "training_examples.json");
const outputPath = path.join(__dirname, "..", "fine_tune", "ark_chat_finetune.jsonl");

const examples = JSON.parse(readFileSync(inputPath, "utf8"));

const lines = examples.map((example) =>
  JSON.stringify({
    messages: [
      {
        role: "system",
        content:
          "你是心理支持辅助 Agent。你不能诊断、处方或提供疾病概率，只能提供支持、筛查边界和自我观察建议。"
      },
      {
        role: "user",
        content: example.user
      },
      {
        role: "assistant",
        content: example.assistant
      }
    ],
    metadata: {
      risk_label: example.risk_label,
      blocked_behavior: example.blocked_behavior
    }
  })
);

writeFileSync(outputPath, `${lines.join("\n")}\n`, "utf8");
console.log(`Wrote ${lines.length} examples to ${outputPath}`);
