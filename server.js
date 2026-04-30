const http = require("node:http");
const path = require("node:path");
const { readFile } = require("node:fs/promises");

const { runAgentTurn, segmentForStreaming } = require("./src/agent/graph");
const { generateReport } = require("./src/agent/report");
const { TOOL_SPECS } = require("./src/agent/tools");

const PORT = Number(process.env.PORT || 3000);
const PUBLIC_DIR = path.join(__dirname, "public");

const sessions = new Map();

function getSession(sessionId) {
  const id = sessionId || "demo-session";
  if (!sessions.has(id)) {
    sessions.set(id, {
      history: [],
      memory: [],
      profile: {
        mbtiType: "unknown",
        memoryEnabled: false,
        communicationStyle: "balanced"
      }
    });
  }
  return sessions.get(id);
}

function sendJson(res, status, payload) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store"
  });
  res.end(JSON.stringify(payload, null, 2));
}

function sendSseEvent(res, event, payload) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(payload)}\n\n`);
}

async function parseJsonBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }
  if (!chunks.length) return {};
  const text = Buffer.concat(chunks).toString("utf8");
  return JSON.parse(text || "{}");
}

async function serveStatic(req, res) {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const pathname = url.pathname === "/" ? "/index.html" : url.pathname;
  const normalized = path.normalize(pathname).replace(/^(\.\.[/\\])+/, "");
  const filePath = path.join(PUBLIC_DIR, normalized);

  if (!filePath.startsWith(PUBLIC_DIR)) {
    sendJson(res, 403, { error: "Forbidden" });
    return;
  }

  try {
    const file = await readFile(filePath);
    const ext = path.extname(filePath).toLowerCase();
    const contentTypes = {
      ".html": "text/html; charset=utf-8",
      ".css": "text/css; charset=utf-8",
      ".js": "application/javascript; charset=utf-8",
      ".json": "application/json; charset=utf-8"
    };
    res.writeHead(200, {
      "Content-Type": contentTypes[ext] || "application/octet-stream",
      "Cache-Control": "no-store"
    });
    res.end(file);
  } catch (error) {
    if (error.code === "ENOENT") {
      sendJson(res, 404, { error: "Not found" });
      return;
    }
    sendJson(res, 500, { error: "Static file error", detail: error.message });
  }
}

async function handleChatStream(req, res) {
  const body = await parseJsonBody(req);
  const session = getSession(body.sessionId);
  session.profile = {
    ...session.profile,
    ...(body.profile || {})
  };

  const userMessage = String(body.message || "").trim();
  if (!userMessage) {
    sendJson(res, 400, { error: "message is required" });
    return;
  }

  session.history.push({
    role: "user",
    content: userMessage,
    createdAt: new Date().toISOString()
  });

  const result = await runAgentTurn({
    message: userMessage,
    history: session.history,
    memory: session.memory,
    profile: session.profile
  });

  if (result.memoryCandidate && session.profile.memoryEnabled) {
    session.memory.push({
      ...result.memoryCandidate,
      createdAt: new Date().toISOString()
    });
  }

  session.history.push({
    role: "assistant",
    content: result.reply,
    risk: result.risk,
    citations: result.citations,
    createdAt: new Date().toISOString()
  });

  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no"
  });

  sendSseEvent(res, "meta", {
    risk: result.risk,
    plan: result.plan,
    citations: result.citations,
    memoryCandidate: result.memoryCandidate || null
  });

  for (const segment of segmentForStreaming(result.reply)) {
    sendSseEvent(res, "chunk", { text: segment });
    await new Promise((resolve) => setTimeout(resolve, 80));
  }

  sendSseEvent(res, "done", {
    historyLength: session.history.length,
    reportReady: session.history.filter((item) => item.role === "user").length >= 2
  });
  res.end();
}

async function handleReport(req, res) {
  const body = await parseJsonBody(req);
  const session = getSession(body.sessionId);
  const report = generateReport({
    history: session.history,
    memory: session.memory,
    profile: {
      ...session.profile,
      ...(body.profile || {})
    }
  });
  sendJson(res, 200, report);
}

async function handleReset(req, res) {
  const body = await parseJsonBody(req);
  sessions.delete(body.sessionId || "demo-session");
  sendJson(res, 200, { ok: true });
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);

    if (req.method === "POST" && url.pathname === "/api/chat/stream") {
      await handleChatStream(req, res);
      return;
    }

    if (req.method === "POST" && url.pathname === "/api/report") {
      await handleReport(req, res);
      return;
    }

    if (req.method === "POST" && url.pathname === "/api/reset") {
      await handleReset(req, res);
      return;
    }

    if (req.method === "GET" && url.pathname === "/api/tools") {
      sendJson(res, 200, { tools: TOOL_SPECS });
      return;
    }

    if (req.method === "GET") {
      await serveStatic(req, res);
      return;
    }

    sendJson(res, 405, { error: "Method not allowed" });
  } catch (error) {
    sendJson(res, 500, {
      error: "Server error",
      detail: error.message
    });
  }
});

server.listen(PORT, () => {
  console.log(`Mental support agent MVP is running at http://localhost:${PORT}`);
});
