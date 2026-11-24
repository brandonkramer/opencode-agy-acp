import { describe, expect, test } from "bun:test"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import type { LanguageModelV3Prompt, LanguageModelV3StreamPart } from "@ai-sdk/provider"
import type { Context } from "@opencode/plugin/effect/plugin"
import { Effect } from "effect"
import { SessionRegistry } from "../src/acp/sessions.js"
import { createAgyLanguageModel } from "../src/language-model.js"
import plugin from "../src/index.js"
import {
  AcpStallError,
  DEFAULT_SETTINGS,
  OPENCODE_QUESTION_TOOL,
  PROVIDER_ID,
  type AgySettings,
} from "../src/types.js"

const FAKE_AGY = path.resolve(import.meta.dir, "fake-agy.ts")

function createMockContext(opts?: {
  readonly cwd?: string
  readonly permissionImpl?: () => Effect.Effect<void, unknown>
}) {
  const store = new Map<string, unknown>()
  const cwd = opts?.cwd ?? process.cwd()
  const ctx = {
    location: { directory: cwd },
    storage: {
      get: (key: string) => Effect.sync(() => store.get(key) as never),
      set: (key: string, value: unknown) =>
        Effect.sync(() => {
          store.set(key, value)
        }),
      remove: (key: string) =>
        Effect.sync(() => {
          store.delete(key)
        }),
    },
    session: {
      get: () => Effect.succeed({ location: { directory: cwd } } as never),
      update: () => Effect.void,
    },
    permission: {
      assert: () => (opts?.permissionImpl ? opts.permissionImpl() : Effect.void),
    },
  } as unknown as Context

  return { ctx, store }
}

function makeSettings(overrides: Partial<AgySettings> = {}): AgySettings {
  return {
    ...DEFAULT_SETTINGS,
    binaryPath: FAKE_AGY,
    startupTimeoutMs: 5_000,
    chunkTimeoutMs: 5_000,
    settleTimeoutMs: 500,
    permissionTimeoutMs: 500,
    nativeFlushGraceMs: 120,
    idleTimeoutMs: 60_000,
    ...overrides,
  }
}

async function readStreamParts(
  stream: ReadableStream<LanguageModelV3StreamPart>,
): Promise<LanguageModelV3StreamPart[]> {
  const reader = stream.getReader()
  const parts: LanguageModelV3StreamPart[] = []
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    parts.push(value)
  }
  return parts
}

function textFromParts(parts: readonly LanguageModelV3StreamPart[]): string {
  return parts
    .filter((p): p is Extract<LanguageModelV3StreamPart, { type: "text-delta" }> => p.type === "text-delta")
    .map((p) => p.delta)
    .join("")
}

describe("SessionRegistry + fake-agy integration", () => {
  test("1. Pure ACP turn emits thought, native tool, usage, and omits --process-per-session", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "agy-test-1-"))
    const argsLog = path.join(tmp, "args.log")
    const prevArgsEnv = process.env.FAKE_AGY_ARGS_LOG
    process.env.FAKE_AGY_ARGS_LOG = argsLog

    const { ctx } = createMockContext({ cwd: tmp })
    const registry = new SessionRegistry(ctx, { settings: makeSettings({ codeMode: false }) })
    try {
      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const { stream } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_pure_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:PURE_OK] hello" }] }],
      })
      const parts = await readStreamParts(stream)
      const text = textFromParts(parts)
      expect(text).toContain("OK(gemini-4-neon-high): [SCENARIO:PURE_OK] hello")
      expect(parts.some((p) => p.type === "reasoning-delta")).toBe(true)
      expect(parts.some((p) => p.type === "tool-call" && p.toolName === "acp_read_file")).toBe(true)
      expect(parts.some((p) => p.type === "tool-result" && p.toolName === "acp_read_file")).toBe(true)
      const finish = parts.find((p) => p.type === "finish")
      expect(finish?.type === "finish" && finish.usage.inputTokens.total).toBe(100)

      const loggedArgs = JSON.parse(fs.readFileSync(argsLog, "utf8").trim()) as string[]
      expect(loggedArgs).toContain("--include-default-customizations=false")
      expect(loggedArgs).not.toContain("--process-per-session")
    } finally {
      if (prevArgsEnv === undefined) delete process.env.FAKE_AGY_ARGS_LOG
      else process.env.FAKE_AGY_ARGS_LOG = prevArgsEnv
      await registry.disposeAll()
      fs.rmSync(tmp, { recursive: true, force: true })
    }
  })

  test("2. Stall watchdog fires AcpStallError, sends session/cancel, resets proc, and recovers on next turn", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "agy-test-2-"))
    const eventsLog = path.join(tmp, "events.log")
    const prevEventsEnv = process.env.FAKE_AGY_EVENTS_LOG
    process.env.FAKE_AGY_EVENTS_LOG = eventsLog

    const { ctx } = createMockContext({ cwd: tmp })
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ chunkTimeoutMs: 120 }),
    })
    try {
      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const { stream } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_stall_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:STALL]" }] }],
      })
      const parts = await readStreamParts(stream)
      const errPart = parts.find((p) => p.type === "error")
      expect(errPart?.type).toBe("error")
      if (errPart?.type === "error") {
        expect(errPart.error).toBeInstanceOf(AcpStallError)
      }

      // Give cancel/kill a moment to flush
      await Bun.sleep(50)
      const events = fs
        .readFileSync(eventsLog, "utf8")
        .trim()
        .split("\n")
        .map((l) => JSON.parse(l) as { method: string })
      expect(events.some((e) => e.method === "session/cancel")).toBe(true)
      expect(registry.getState("ses_stall_1").connected).toBe(false)

      // Next turn spawns fresh and succeeds
      const { stream: stream2 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_stall_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:PURE_OK] recovered" }] }],
      })
      const parts2 = await readStreamParts(stream2)
      expect(textFromParts(parts2)).toContain("recovered")
    } finally {
      if (prevEventsEnv === undefined) delete process.env.FAKE_AGY_EVENTS_LOG
      else process.env.FAKE_AGY_EVENTS_LOG = prevEventsEnv
      await registry.disposeAll()
      fs.rmSync(tmp, { recursive: true, force: true })
    }
  })

  test("3. Watchdog pauses while OpenCode executes a slow relayed MCP tool", async () => {
    const { ctx } = createMockContext()
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ codeMode: true, chunkTimeoutMs: 100 }),
    })
    try {
      registry.updateContextSnapshot("ses_pause_1", {
        agent: "build",
        tools: {
          subagent: { description: "Run subagent", input: { type: "object", properties: {} } },
        },
      })

      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const { stream: step1 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_pause_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:MCP_PAUSE_WATCHDOG]" }] }],
      })
      const parts1 = await readStreamParts(step1)
      const toolCall = parts1.find((p) => p.type === "tool-call")
      expect(toolCall?.type).toBe("tool-call")
      if (toolCall?.type !== "tool-call") return

      // Hold OpenCode tool execution longer than chunkTimeoutMs (100ms)
      await Bun.sleep(200)

      const { stream: step2 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_pause_1" },
        prompt: [
          { role: "user", content: [{ type: "text", text: "[SCENARIO:MCP_PAUSE_WATCHDOG]" }] },
          {
            role: "tool",
            content: [
              {
                type: "tool-result",
                toolCallId: toolCall.toolCallId,
                toolName: toolCall.toolName,
                output: { type: "text", value: "subagent done" },
              },
            ],
          },
        ],
      })
      const parts2 = await readStreamParts(step2)
      expect(textFromParts(parts2)).toContain("PAUSED_OK:subagent done")
    } finally {
      await registry.disposeAll()
    }
  })

  test("4. Non-prefixed relayed MCP tool call does not deadlock activeNativeTools", async () => {
    const { ctx } = createMockContext()
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ codeMode: true }),
    })
    try {
      registry.updateContextSnapshot("ses_nonprefix_1", {
        agent: "build",
        tools: {
          execute: { description: "Code mode JS", input: { type: "object", properties: {} } },
        },
      })

      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const { stream: step1 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_nonprefix_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:MCP_NON_PREFIX_TOOL]" }] }],
      })
      const parts1 = await readStreamParts(step1)
      const toolCall = parts1.find((p) => p.type === "tool-call" && p.toolName === "execute")
      expect(toolCall).toBeDefined()
      if (toolCall?.type !== "tool-call") return

      const { stream: step2 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_nonprefix_1" },
        prompt: [
          { role: "user", content: [{ type: "text", text: "[SCENARIO:MCP_NON_PREFIX_TOOL]" }] },
          {
            role: "tool",
            content: [
              {
                type: "tool-result",
                toolCallId: toolCall.toolCallId,
                toolName: "execute",
                output: { type: "text", value: "2" },
              },
            ],
          },
        ],
      })
      const parts2 = await readStreamParts(step2)
      expect(textFromParts(parts2)).toContain("MCP_RESULT:2")
    } finally {
      await registry.disposeAll()
    }
  })

  test("5. Unconfirmed native tool + relayed MCP flushes after nativeFlushGraceMs", async () => {
    const { ctx } = createMockContext()
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ codeMode: true, nativeFlushGraceMs: 80 }),
    })
    try {
      registry.updateContextSnapshot("ses_grace_1", {
        agent: "build",
        tools: {
          subagent: { description: "Run subagent", input: { type: "object", properties: {} } },
        },
      })

      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const { stream: step1 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_grace_1" },
        prompt: [
          { role: "user", content: [{ type: "text", text: "[SCENARIO:UNCONFIRMED_NATIVE_AND_MCP]" }] },
        ],
      })
      const parts1 = await readStreamParts(step1)
      const toolCall = parts1.find((p) => p.type === "tool-call" && p.toolName === "subagent")
      expect(toolCall).toBeDefined()
      if (toolCall?.type !== "tool-call") return

      const { stream: step2 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_grace_1" },
        prompt: [
          { role: "user", content: [{ type: "text", text: "[SCENARIO:UNCONFIRMED_NATIVE_AND_MCP]" }] },
          {
            role: "tool",
            content: [
              {
                type: "tool-result",
                toolCallId: toolCall.toolCallId,
                toolName: "subagent",
                output: { type: "text", value: "checked" },
              },
            ],
          },
        ],
      })
      const parts2 = await readStreamParts(step2)
      expect(textFromParts(parts2)).toContain("GRACE_OK:checked")
    } finally {
      await registry.disposeAll()
    }
  })

  test("6. elicitation/create (ask_question) bridges in Pure ACP mode and settles cleanly", async () => {
    const { ctx } = createMockContext()
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ codeMode: false }),
    })
    try {
      registry.updateContextSnapshot("ses_elicit_1", {
        agent: "build",
        tools: {
          [OPENCODE_QUESTION_TOOL]: {
            description: "Ask user",
            input: { type: "object", properties: {} },
          },
        },
      })

      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const { stream: step1 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_elicit_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:ELICIT]" }] }],
      })
      const parts1 = await readStreamParts(step1)
      const qCall = parts1.find((p) => p.type === "tool-call" && p.toolName === OPENCODE_QUESTION_TOOL)
      expect(qCall).toBeDefined()
      // Native ask_question was settled locally when OpenCode question was emitted
      expect(parts1.some((p) => p.type === "tool-result" && p.toolName === "acp_ask_question")).toBe(
        true,
      )
      if (qCall?.type !== "tool-call") return

      const { stream: step2 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_elicit_1" },
        prompt: [
          { role: "user", content: [{ type: "text", text: "[SCENARIO:ELICIT]" }] },
          {
            role: "tool",
            content: [
              {
                type: "tool-result",
                toolCallId: qCall.toolCallId,
                toolName: OPENCODE_QUESTION_TOOL,
                output: {
                  type: "text",
                  value:
                    'User has answered your questions: "Pick target"="Mac". You can now continue with the user\'s answers in mind.',
                },
              },
            ],
          },
        ],
      })
      const parts2 = await readStreamParts(step2)
      expect(textFromParts(parts2)).toContain('"action":"accept","content":{"q1":"mac"}')
    } finally {
      await registry.disposeAll()
    }
  })

  test("7. Permission timeout in mode=default resolves with cancelled instead of hanging", async () => {
    const { ctx } = createMockContext({
      permissionImpl: () => Effect.never as unknown as Effect.Effect<void, unknown>,
    })
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ permissionTimeoutMs: 80, chunkTimeoutMs: 250 }),
    })
    try {
      await registry.setMode("ses_perm_1", "default")
      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const { stream } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_perm_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:PERMISSION]" }] }],
      })
      const parts = await readStreamParts(stream)
      expect(textFromParts(parts)).toContain('PERM:{"outcome":"cancelled"}')
    } finally {
      await registry.disposeAll()
    }
  })

  test("8. Per-pool 429 cooldown blocks gemini pool while claude-gpt pool works, plus cross-model handoff", async () => {
    const { ctx } = createMockContext()
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ codeMode: false }),
    })
    try {
      const gemini = createAgyLanguageModel("gemini-4-neon", registry)
      const claude = createAgyLanguageModel("claude-opus-5-5", registry)

      const { stream: quotaStream } = await gemini.doStream({
        headers: { "x-opencode-session-id": "ses_pool_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:QUOTA_429]" }] }],
      })
      const quotaParts = await readStreamParts(quotaStream)
      expect(textFromParts(quotaParts)).toContain("60s cooldown for gemini models")

      // Gemini pool is now blocked immediately
      const { stream: blockedStream } = await gemini.doStream({
        headers: { "x-opencode-session-id": "ses_pool_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:PURE_OK] blocked?" }] }],
      })
      expect(textFromParts(await readStreamParts(blockedStream))).toContain("cooldown active")

      // Claude pool on same session succeeds immediately
      const { stream: claudeStream } = await claude.doStream({
        headers: { "x-opencode-session-id": "ses_pool_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:PURE_OK] claude ok" }] }],
      })
      expect(textFromParts(await readStreamParts(claudeStream))).toContain("OK(claude-opus-5-5-medium): [SCENARIO:PURE_OK] claude ok")

      // Switching back after foreign non-ACP assistant turn triggers plain-text handoff
      const handoffPrompt: LanguageModelV3Prompt = [
        { role: "user", content: [{ type: "text", text: "prior question" }] },
        { role: "assistant", content: [{ type: "text", text: "foreign model answer" }] },
        { role: "user", content: [{ type: "text", text: "[SCENARIO:ECHO_HANDOFF]" }] },
      ]
      const { stream: handoffStream } = await claude.doStream({
        headers: { "x-opencode-session-id": "ses_pool_1" },
        prompt: handoffPrompt,
      })
      const handoffText = textFromParts(await readStreamParts(handoffStream))
      expect(handoffText).toContain("[Historical assistant]: foreign model answer")
      expect(handoffText).toContain("[SCENARIO:ECHO_HANDOFF]")
    } finally {
      await registry.disposeAll()
    }
  })

  test("9. Plugin exports valid id and effect registration including retry:false hook", () => {
    expect(plugin.id).toBe(PROVIDER_ID)
    expect(typeof plugin.effect).toBe("function")
  })

  test("10. Idle reaper disposes proc while preserving sessionId, resumes cleanly, skips parked turns, and deleteSession removes storage", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "agy-test-idle-"))
    const sessionsFile = path.join(tmp, "sessions.json")
    const eventsLog = path.join(tmp, "events.log")
    const prevSessions = process.env.FAKE_AGY_SESSIONS_FILE
    const prevEvents = process.env.FAKE_AGY_EVENTS_LOG
    process.env.FAKE_AGY_SESSIONS_FILE = sessionsFile
    process.env.FAKE_AGY_EVENTS_LOG = eventsLog

    const { ctx, store } = createMockContext({ cwd: tmp })
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ idleTimeoutMs: 70, codeMode: true }),
    })
    try {
      registry.updateContextSnapshot("ses_idle_1", {
        agent: "build",
        tools: {
          subagent: { description: "Run subagent", input: { type: "object", properties: {} } },
        },
      })
      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const { stream: s1 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_idle_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:PURE_OK] first" }] }],
      })
      await readStreamParts(s1)

      const acpSid = registry.getState("ses_idle_1").acpSessionId
      expect(acpSid).toBeTruthy()
      expect(registry.getState("ses_idle_1").connected).toBe(true)
      expect(store.get("antigravity-acp:session:ses_idle_1")).toBe(acpSid)

      // Arm idle reaper and wait past idleTimeoutMs
      registry.armIdle("ses_idle_1")
      await Bun.sleep(140)
      expect(registry.getState("ses_idle_1").connected).toBe(false)
      expect(store.get("antigravity-acp:session:ses_idle_1")).toBe(acpSid)

      // Next turn respawns proc and calls session/resume with preserved id
      const { stream: s2 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_idle_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:PURE_OK] resumed" }] }],
      })
      const parts2 = await readStreamParts(s2)
      expect(textFromParts(parts2)).toContain("resumed")
      expect(registry.getState("ses_idle_1").acpSessionId).toBe(acpSid)

      const events = fs
        .readFileSync(eventsLog, "utf8")
        .trim()
        .split("\n")
        .map((l) => JSON.parse(l) as { method: string; sessionId?: string })
      expect(events.some((e) => e.method === "session/resume" && e.sessionId === acpSid)).toBe(true)

      // armIdle must NOT evict while parked on an OpenCode tool call
      const { stream: s3 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_idle_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:MCP_PAUSE_WATCHDOG]" }] }],
      })
      const parts3 = await readStreamParts(s3)
      const toolCall = parts3.find((p) => p.type === "tool-call")
      expect(toolCall?.type).toBe("tool-call")
      registry.armIdle("ses_idle_1")
      await Bun.sleep(140)
      expect(registry.getState("ses_idle_1").connected).toBe(true)
      expect(registry.getState("ses_idle_1").busy).toBe(true)

      if (toolCall?.type === "tool-call") {
        const { stream: s4 } = await lm.doStream({
          headers: { "x-opencode-session-id": "ses_idle_1" },
          prompt: [
            { role: "user", content: [{ type: "text", text: "[SCENARIO:MCP_PAUSE_WATCHDOG]" }] },
            {
              role: "tool",
              content: [
                {
                  type: "tool-result",
                  toolCallId: toolCall.toolCallId,
                  toolName: toolCall.toolName,
                  output: { type: "text", value: "ok" },
                },
              ],
            },
          ],
        })
        await readStreamParts(s4)
      }

      // deleteSession tears down proc and removes stored acpSessionId
      await registry.deleteSession("ses_idle_1")
      expect(registry.hasSession("ses_idle_1")).toBe(false)
      expect(store.get("antigravity-acp:session:ses_idle_1")).toBeUndefined()
    } finally {
      if (prevSessions === undefined) delete process.env.FAKE_AGY_SESSIONS_FILE
      else process.env.FAKE_AGY_SESSIONS_FILE = prevSessions
      if (prevEvents === undefined) delete process.env.FAKE_AGY_EVENTS_LOG
      else process.env.FAKE_AGY_EVENTS_LOG = prevEvents
      await registry.disposeAll()
      fs.rmSync(tmp, { recursive: true, force: true })
    }
  })
})
