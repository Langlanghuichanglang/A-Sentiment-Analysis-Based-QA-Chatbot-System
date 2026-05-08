# 心理支持辅助 Agent

这是一个课程项目 MVP：以 chatbox 形式提供心理支持、自我观察、RAG 辅助、风险分诊、报告生成和后续 LangGraph/Fine-tune 扩展空间。

## 当前能跑的功能

- 网页 chatbox
- MBTI/沟通偏好设置
- 风险分诊 `risk_triage`
- 高危表达危机路径
- 诊断/疾病概率请求边界化
- RAG demo 检索
- tools 接口 schema
- 非诊断性质的“心理筛查与自我观察报告”
- 火山方舟 provider 预留
- FastAPI + LangGraph 后端骨架
- Fine-tune 数据准备脚本

## 运行

```bash
npm run dev
```

然后打开：

```text
http://localhost:3000
```

当前版本不需要安装 npm 依赖，Node.js 20+ 即可。

## 接入火山方舟

复制环境变量示例：

```bash
cp .env.example .env
```

设置：

```text
ARK_API_KEY=你的火山方舟 API Key
ARK_MODEL=你的模型 endpoint 或 model id
```

然后重新运行 `npm run dev`。没有这些环境变量时，系统会自动使用本地 mock 生成器，方便课堂演示。

## 重要边界

项目刻意不输出“你患有某疾病”或“80% 可能是某疾病”。报告里只输出筛查线索和建议进一步评估的提示。

推荐说法：

```text
抑郁相关线索：偏高。建议后续使用 PHQ-9 做筛查，并咨询专业人员。
```

不推荐说法：

```text
你 80% 可能患有抑郁症。
```

## 后续开发位置

- Agent 主流程：`src/agent/graph.js`
- 风险分诊：`src/agent/safety.js`
- RAG demo：`src/agent/rag.js`
- 知识库：`data/knowledge_base.json`
- Tools schema：`src/agent/tools.js`
- 报告生成：`src/agent/report.js`
- 火山方舟调用：`src/llm/volcengine.js`
- LangGraph 后端骨架：`backend/app/agent/graph.py`
- Fine-tune 方案：`docs/fine-tune-plan.md`

## Fine-tune 数据准备

```bash
npm run prepare:finetune
```

输出：

```text
fine_tune/ark_chat_finetune.jsonl
```

Fine-tune 目标建议只放在语气、拒答风格、工具调用格式和报告措辞，不要用于疾病诊断标签预测。
