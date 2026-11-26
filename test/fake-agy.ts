#!/usr/bin/env bun
import fs from "node:fs"
import readline from "node:readline"

interface RpcMsg {
  jsonrpc?: string
  id?: number | string
  method?: string
  params?: Record<string, unknown>
  result?: unknown
  error?: unknown
}

const argsLogPath = process.env.FAKE_AGY_ARGS_LOG
if (argsLogPath) {
  fs.appendFileSync(argsLogPath, `${JSON.stringify(process.argv.slice(2))}\n`)
}

const eventsLogPath = process.env.FAKE_AGY_EVENTS_LOG
function logEvent(evt: Record<string, unknown>) {
  if (eventsLogPath) {
    fs.appendFileSync(eventsLogPath, `${JSON.stringify(evt)}\n`)
  }
}

let nextServerId = 1
const pendingClientCalls = new Map<
  number,
  { resolve: (v: unknown) => void; reject: (e: Error) => void }
>()
const sessionMcp = new Map<string, { url: string; headers: Record<string, string> }>()
const sessionModel = new Map<string, string>()
const sessionMode = new Map<string, string>()
const sessionSystemAppend = new Map<string, string>()
const sessionsFile = process.env.FAKE_AGY_SESSIONS_FILE
function loadKnownSessions(): Set<string> {
  if (!sessionsFile || !fs.existsSync(sessionsFile)) return new Set<string>()
  try {
    const parsed = JSON.parse(fs.readFileSync(sessionsFile, "utf8")) as unknown
    return Array.isArray(parsed) ? new Set(parsed.map(String)) : new Set<string>()
  } catch {
    return new Set<string>()
  }
}
function saveKnownSession(sid: string): void {
  knownSessions.add(sid)
  if (!sessionsFile) return
  fs.writeFileSync(sessionsFile, JSON.stringify([...knownSessions]))
}
const knownSessions = loadKnownSessions()
let sessionCounter = knownSessions.size

function send(payload: Record<string, unknown>) {
  process.stdout.write(`${JSON.stringify(payload)}\n`)
}

function sendUpdate(sessionId: string, update: Record<string, unknown>) {
  send({
    jsonrpc: "2.0",
    method: "session/update",
    params: { sessionId, update },
  })
}

function requestClient(method: string, params: Record<string, unknown>): Promise<unknown> {
  const id = nextServerId++
  return new Promise((resolve, reject) => {
    pendingClientCalls.set(id, { resolve, reject })
    send({ jsonrpc: "2.0", id, method, params })
  })
}

async function callMcpTool(
  sessionId: string,
  toolName: string,
  args: Record<string, unknown>,
): Promise<unknown> {
  const mcp = sessionMcp.get(sessionId)
  if (!mcp) throw new Error("No MCP server configured for session")
  const res = await fetch(mcp.url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...mcp.headers,
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "tools/call",
      params: { name: toolName, arguments: args },
    }),
  })
  const json = (await res.json()) as { result?: unknown; error?: unknown }
  if (json.error) throw new Error(JSON.stringify(json.error))
  return json.result
}

async function listMcpTools(sessionId: string): Promise<unknown> {
  const mcp = sessionMcp.get(sessionId)
  if (!mcp) throw new Error("No MCP server configured for session")
  const res = await fetch(mcp.url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...mcp.headers,
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
      params: {},
    }),
  })
  const json = (await res.json()) as { result?: unknown; error?: unknown }
  if (json.error) throw new Error(JSON.stringify(json.error))
  return json.result
}

function extractPromptText(prompt: unknown): string {
  if (!Array.isArray(prompt)) return ""
  return prompt
    .map((b) => (b && typeof b === "object" && "text" in b ? String((b as { text: unknown }).text) : ""))
    .join("\n")
}

async function handlePrompt(id: number | string, params: Record<string, unknown>) {
  const sessionId = String(params.sessionId ?? "")
  const promptBlocks = params.prompt
  const text = extractPromptText(promptBlocks)
  logEvent({ method: "session/prompt", sessionId, text })

  if (text.includes("[SCENARIO:STALL]")) {
    sendUpdate(sessionId, {
      sessionUpdate: "agent_thought_chunk",
      content: { type: "text", text: "Thinking before stall..." },
    })
    // Intentionally never resolve!
    return
  }

  if (text.includes("[SCENARIO:NATIVE_BG_WATCHER]")) {
    sendUpdate(sessionId, {
      sessionUpdate: "tool_call",
      toolCallId: "tc_bg_1",
      name: "run_command",
      status: "in_progress",
      rawInput: { CommandLine: "gh run watch", WaitMsBeforeAsync: 500 },
    })
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: "Waiting on background command... " },
    })
    sendUpdate(sessionId, {
      sessionUpdate: "usage_update",
      used: 50_000,
      size: 1_048_576,
    })
    // Silent past short chunkTimeoutMs while native tool is still running
    await Bun.sleep(220)
    sendUpdate(sessionId, {
      sessionUpdate: "tool_call_update",
      toolCallId: "tc_bg_1",
      status: "completed",
      rawOutput: { output: "CI passed", exit_code: 0 },
    })
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: "NATIVE_BG_OK" },
    })
    send({
      jsonrpc: "2.0",
      id,
      result: { stopReason: "end_turn", usage: { inputTokens: 100, outputTokens: 20 } },
    })
    return
  }

  if (text.includes("[SCENARIO:NATIVE_BG_THEN_LLM_STALL]")) {
    sendUpdate(sessionId, {
      sessionUpdate: "tool_call",
      toolCallId: "tc_bg_then_stall",
      name: "run_command",
      status: "in_progress",
      rawInput: { CommandLine: "build" },
    })
    // Survives past chunkTimeoutMs while native tool runs...
    await Bun.sleep(160)
    sendUpdate(sessionId, {
      sessionUpdate: "tool_call_update",
      toolCallId: "tc_bg_then_stall",
      status: "completed",
      rawOutput: { output: "ok", exit_code: 0 },
    })
    // ...then hangs silently with no active native tools, so normal chunkTimeoutMs should fire!
    return
  }

  if (text.includes("[SCENARIO:NATIVE_TOOL_STALL]")) {
    sendUpdate(sessionId, {
      sessionUpdate: "tool_call",
      toolCallId: "tc_native_wedged",
      name: "run_command",
      status: "in_progress",
      rawInput: { CommandLine: "sleep forever" },
    })
    // Never completes — nativeToolTimeoutMs safety cap must still fire
    return
  }

  if (text.includes("[SCENARIO:SLOW_CHUNKS]")) {
    for (const part of ["one ", "two ", "three"]) {
      await Bun.sleep(50)
      sendUpdate(sessionId, {
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: part },
      })
    }
    send({
      jsonrpc: "2.0",
      id,
      result: { stopReason: "end_turn", usage: { inputTokens: 12, outputTokens: 3 } },
    })
    return
  }

  if (text.includes("[SCENARIO:MCP_NON_PREFIX_TOOL]")) {
    sendUpdate(sessionId, {
      sessionUpdate: "tool_call",
      toolCallId: "tc_nonprefix_1",
      name: "execute",
      status: "in_progress",
      rawInput: { code: "return 1 + 1" },
    })
    const res = (await callMcpTool(sessionId, "execute", { code: "return 1 + 1" })) as {
      content?: Array<{ text?: string }>
    }
    const mcpText = res?.content?.[0]?.text ?? ""
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: `MCP_RESULT:${mcpText}` },
    })
    send({
      jsonrpc: "2.0",
      id,
      result: { stopReason: "end_turn" },
    })
    return
  }

  if (text.includes("[SCENARIO:MULTI_STEP_USAGE_AND_QUEUED]")) {
    sendUpdate(sessionId, { sessionUpdate: "usage_update", used: 10_000, size: 1_048_576 })
    const r1 = (await callMcpTool(sessionId, "execute", { code: "step1()" })) as {
      content?: Array<{ text?: string }>
    }
    sendUpdate(sessionId, { sessionUpdate: "usage_update", used: 11_200, size: 1_048_576 })
    const r2 = (await callMcpTool(sessionId, "execute", { code: "step2()" })) as {
      content?: Array<{ text?: string }>
    }
    sendUpdate(sessionId, { sessionUpdate: "usage_update", used: 12_500, size: 1_048_576 })
    const r1Texts = (r1?.content ?? []).map((c) => c.text ?? "")
    const r2Texts = (r2?.content ?? []).map((c) => c.text ?? "")
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: {
        type: "text",
        text: `MULTI_OK:${JSON.stringify({ sessionId, r1Texts, r2Texts })}`,
      },
    })
    send({
      jsonrpc: "2.0",
      id,
      result: {
        stopReason: "end_turn",
        // Cumulative sum across 3 steps: 10000 + 11200 + 12500 = 33700
        usage: {
          inputTokens: 15_000,
          cachedReadTokens: 18_200,
          cachedWriteTokens: 0,
          outputTokens: 500,
          thoughtTokens: 100,
          totalTokens: 33_700,
        },
      },
    })
    return
  }

  if (text.includes("[SCENARIO:MCP_IMAGE_TOOL]")) {
    const res = await callMcpTool(sessionId, "execute", { code: "screenshot()" })
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: `IMAGE_RES:${JSON.stringify(res)}` },
    })
    send({
      jsonrpc: "2.0",
      id,
      result: { stopReason: "end_turn" },
    })
    return
  }

  if (text.includes("[SCENARIO:LIST_MCP_TOOLS]")) {
    const res = await listMcpTools(sessionId)
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: {
        type: "text",
        text: `MCP_TOOLS:${JSON.stringify({
          mcp: res,
          append: sessionSystemAppend.get(sessionId) ?? null,
        })}`,
      },
    })
    send({
      jsonrpc: "2.0",
      id,
      result: { stopReason: "end_turn" },
    })
    return
  }

  if (text.includes("[SCENARIO:UNCONFIRMED_NATIVE_AND_MCP]")) {
    sendUpdate(sessionId, {
      sessionUpdate: "tool_call",
      toolCallId: "tc_native_hung",
      name: "bash",
      status: "in_progress",
      rawInput: { command: "sleep" },
    })
    const res = (await callMcpTool(sessionId, "subagent", { task: "check" })) as {
      content?: Array<{ text?: string }>
    }
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: `GRACE_OK:${res?.content?.[0]?.text ?? ""}` },
    })
    send({
      jsonrpc: "2.0",
      id,
      result: { stopReason: "end_turn" },
    })
    return
  }

  if (text.includes("[SCENARIO:BG_NATIVE_ACROSS_MCP_STEPS]")) {
    // Reproduces long-running native background command that stays open across an MCP flush
    // and completes during Step 2 (while still keeping watchdog extended until completion).
    sendUpdate(sessionId, {
      sessionUpdate: "tool_call",
      toolCallId: "tc_bg_cross_step",
      name: "run_command",
      status: "in_progress",
      rawInput: { CommandLine: "watch-merge-queue", WaitMsBeforeAsync: 500, NotificationTimeoutSeconds: 900 },
    })
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: "Started background watcher. " },
    })
    const res = (await callMcpTool(sessionId, "execute", { code: "checkQueue()" })) as {
      content?: Array<{ text?: string }>
    }
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: `Queue check: ${res?.content?.[0]?.text ?? ""}. ` },
    })
    // Silent longer than chunkTimeoutMs in Step 2 while detached native background command still runs
    await Bun.sleep(180)
    sendUpdate(sessionId, {
      sessionUpdate: "tool_call_update",
      toolCallId: "tc_bg_cross_step",
      status: "completed",
      rawOutput: { output: "PR merged", exit_code: 0 },
    })
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: "CROSS_STEP_OK" },
    })
    send({
      jsonrpc: "2.0",
      id,
      result: { stopReason: "end_turn", usage: { inputTokens: 120, outputTokens: 30 } },
    })
    return
  }

  if (text.includes("[SCENARIO:MCP_PAUSE_WATCHDOG]")) {
    const res = (await callMcpTool(sessionId, "subagent", { task: "slow" })) as {
      content?: Array<{ text?: string }>
    }
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: `PAUSED_OK:${res?.content?.[0]?.text ?? ""}` },
    })
    send({
      jsonrpc: "2.0",
      id,
      result: { stopReason: "end_turn" },
    })
    return
  }

  if (text.includes("[SCENARIO:ELICIT]")) {
    sendUpdate(sessionId, {
      sessionUpdate: "tool_call",
      toolCallId: "tc_ask_1",
      name: "ask_question",
      title: "Pick target",
      status: "in_progress",
    })
    const answer = (await requestClient("elicitation/create", {
      sessionId,
      mode: "form",
      toolCallId: "tc_ask_1",
      requestedSchema: {
        type: "object",
        properties: {
          q1: {
            type: "string",
            title: "Pick target",
            oneOf: [
              { const: "mac", title: "Mac" },
              { const: "linux", title: "Linux" },
            ],
          },
        },
      },
    })) as { action?: string; content?: Record<string, unknown> }
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: `ELICIT:${JSON.stringify(answer)}` },
    })
    send({
      jsonrpc: "2.0",
      id,
      result: { stopReason: "end_turn" },
    })
    return
  }

  if (text.includes("[SCENARIO:PERMISSION]")) {
    const resp = (await requestClient("session/request_permission", {
      sessionId,
      toolCall: {
        toolCallId: "perm_1",
        name: "run_command",
        title: "rm -rf /tmp/test",
        kind: "command",
      },
      options: [
        { optionId: "allow_once", kind: "allow_once", name: "Allow" },
        { optionId: "reject_once", kind: "reject_once", name: "Deny" },
      ],
    })) as { outcome?: unknown }
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: `PERM:${JSON.stringify(resp.outcome)}` },
    })
    send({
      jsonrpc: "2.0",
      id,
      result: { stopReason: "end_turn" },
    })
    return
  }

  if (text.includes("[SCENARIO:QUOTA_429]")) {
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: "RESOURCE_EXHAUSTED (code 429) for gemini" },
    })
    return
  }

  if (text.includes("[SCENARIO:AUTH_403]")) {
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: "PERMISSION_DENIED (code 403)" },
    })
    return
  }

  if (text.includes("[SCENARIO:ECHO_HANDOFF]")) {
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: `ECHO:${JSON.stringify(promptBlocks)}` },
    })
    send({
      jsonrpc: "2.0",
      id,
      result: { stopReason: "end_turn" },
    })
    return
  }

  if (text.includes("[SCENARIO:ECHO_META]")) {
    sendUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: {
        type: "text",
        text: `META:${JSON.stringify({
          pid: process.pid,
          sessionId,
          mode: sessionMode.get(sessionId) ?? null,
          model: sessionModel.get(sessionId) ?? null,
          append: sessionSystemAppend.get(sessionId) ?? null,
        })}`,
      },
    })
    send({
      jsonrpc: "2.0",
      id,
      result: { stopReason: "end_turn" },
    })
    return
  }

  // Default: PURE_OK
  sendUpdate(sessionId, {
    sessionUpdate: "agent_thought_chunk",
    content: { type: "text", text: "Checking file" },
  })
  sendUpdate(sessionId, {
    sessionUpdate: "tool_call",
    toolCallId: "tc_read_1",
    name: "read_file",
    status: "in_progress",
    rawInput: { path: "README.md" },
  })
  sendUpdate(sessionId, {
    sessionUpdate: "tool_call_update",
    toolCallId: "tc_read_1",
    status: "completed",
    rawOutput: "ok",
  })
  sendUpdate(sessionId, {
    sessionUpdate: "usage_update",
    used: 420,
    size: 1000,
  })
  sendUpdate(sessionId, {
    sessionUpdate: "agent_message_chunk",
    content: { type: "text", text: `OK(${sessionModel.get(sessionId) ?? "default"}): ${text}` },
  })
  send({
    jsonrpc: "2.0",
    id,
    result: {
      stopReason: "end_turn",
      usage: { inputTokens: 100, outputTokens: 25, thoughtTokens: 5 },
    },
  })
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity })
rl.on("line", (line) => {
  if (!line.trim()) return
  const msg = JSON.parse(line) as RpcMsg

  if (msg.id !== undefined && msg.method === undefined) {
    const pending = pendingClientCalls.get(Number(msg.id))
    if (pending) {
      pendingClientCalls.delete(Number(msg.id))
      if (msg.error) pending.reject(new Error(JSON.stringify(msg.error)))
      else pending.resolve(msg.result)
    }
    return
  }

  const method = msg.method ?? ""
  const params = msg.params ?? {}

  if (method === "initialize") {
    send({
      jsonrpc: "2.0",
      id: msg.id,
      result: { protocolVersion: 1, agentInfo: { name: "FakeAgy", version: "1.5.16" } },
    })
    return
  }

  if (method === "authenticate") {
    send({ jsonrpc: "2.0", id: msg.id, result: { ok: true } })
    return
  }

  if (method === "session/new") {
    const sid = `fake-sess-${++sessionCounter}`
    saveKnownSession(sid)
    const meta = params._meta as { antigravity?: { systemPrompt?: { append?: unknown } } } | undefined
    const append =
      typeof meta?.antigravity?.systemPrompt?.append === "string"
        ? meta.antigravity.systemPrompt.append
        : undefined
    if (append !== undefined) sessionSystemAppend.set(sid, append)
    logEvent({ method: "session/new", sessionId: sid, ...(append !== undefined ? { append } : {}) })
    const mcpServers = Array.isArray(params.mcpServers) ? params.mcpServers : []
    const oc = mcpServers.find((s) => s && typeof s === "object" && (s as { name?: string }).name === "oc") as
      | { url?: string; headers?: Array<{ name: string; value: string }> }
      | undefined
    if (oc?.url) {
      const hdrs: Record<string, string> = {}
      for (const h of oc.headers ?? []) hdrs[h.name] = h.value
      sessionMcp.set(sid, { url: oc.url, headers: hdrs })
    }
    send({
      jsonrpc: "2.0",
      id: msg.id,
      result: {
        sessionId: sid,
        models: {
          availableModels: [
            { modelId: "gemini-4-neon-high", name: "Gemini 4 Neon" },
            { modelId: "claude-opus-5-5-medium", name: "Claude Opus 5.5" },
          ],
        },
        configOptions: [
          {
            id: "model",
            options: [
              { value: "gemini-4-neon", name: "Gemini 4 Neon" },
              { value: "claude-opus-5-5", name: "Claude Opus 5.5" },
            ],
          },
        ],
      },
    })
    return
  }

  if (method === "session/resume") {
    const sid = String(params.sessionId ?? "")
    if (!knownSessions.has(sid) || process.env.FAKE_AGY_FAIL_RESUME === "1") {
      send({
        jsonrpc: "2.0",
        id: msg.id,
        error: { code: -32002, message: "Session not found" },
      })
      return
    }
    const meta = params._meta as { antigravity?: { systemPrompt?: { append?: unknown } } } | undefined
    const append =
      typeof meta?.antigravity?.systemPrompt?.append === "string"
        ? meta.antigravity.systemPrompt.append
        : undefined
    if (append !== undefined) sessionSystemAppend.set(sid, append)
    logEvent({ method: "session/resume", sessionId: sid, ...(append !== undefined ? { append } : {}) })
    const mcpServers = Array.isArray(params.mcpServers) ? params.mcpServers : []
    const oc = mcpServers.find((s) => s && typeof s === "object" && (s as { name?: string }).name === "oc") as
      | { url?: string; headers?: Array<{ name: string; value: string }> }
      | undefined
    if (oc?.url) {
      const hdrs: Record<string, string> = {}
      for (const h of oc.headers ?? []) hdrs[h.name] = h.value
      sessionMcp.set(sid, { url: oc.url, headers: hdrs })
    }
    send({ jsonrpc: "2.0", id: msg.id, result: { sessionId: sid } })
    return
  }

  if (method === "session/set_model") {
    sessionModel.set(String(params.sessionId ?? ""), String(params.modelId ?? ""))
    send({ jsonrpc: "2.0", id: msg.id, result: {} })
    return
  }

  if (method === "session/set_mode") {
    const sid = String(params.sessionId ?? "")
    const mode = String(params.modeId ?? params.mode ?? "")
    sessionMode.set(sid, mode)
    logEvent({ method: "session/set_mode", sessionId: sid, mode })
    send({ jsonrpc: "2.0", id: msg.id, result: {} })
    return
  }

  if (method === "session/set_config_option" || method === "session/close") {
    if (method === "session/close") {
      logEvent({ method: "session/close", sessionId: params.sessionId })
    }
    send({ jsonrpc: "2.0", id: msg.id, result: {} })
    return
  }

  if (method === "session/cancel") {
    logEvent({ method: "session/cancel", sessionId: params.sessionId })
    return
  }

  if (method === "session/prompt" && msg.id !== undefined) {
    void handlePrompt(msg.id, params)
    return
  }
})
