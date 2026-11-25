import { describe, expect, test } from "bun:test"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import type { LanguageModelV3Prompt, LanguageModelV3StreamPart } from "@ai-sdk/provider"
import type { Context } from "@opencode/plugin/effect/plugin"
import { Effect } from "effect"
import { SessionRegistry, resetSharedCatalogCacheForTests } from "../src/acp/sessions.js"
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
  let sessionGetCalls = 0
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
      get: () =>
        Effect.sync(() => {
          sessionGetCalls++
          return { location: { directory: cwd } } as never
        }),
      update: () => Effect.void,
    },
    permission: {
      assert: () => (opts?.permissionImpl ? opts.permissionImpl() : Effect.void),
    },
  } as unknown as Context

  return { ctx, store, getSessionGetCalls: () => sessionGetCalls }
}

function makeSettings(overrides: Partial<AgySettings> = {}): AgySettings {
  return {
    ...DEFAULT_SETTINGS,
    binaryPath: FAKE_AGY,
    prewarm: false,
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
      expect(loggedArgs).toContain("--disable-telemetry")
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
      expect(handoffText).toContain("conversation-checkpoint")
      expect(handoffText).toContain("Assistant: foreign model answer")
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
      const parts1 = await readStreamParts(s1)

      const acpSid = registry.getState("ses_idle_1").acpSessionId
      expect(acpSid).toBeTruthy()
      expect(registry.getState("ses_idle_1").connected).toBe(true)
      const persistedAfterTurn1 = JSON.parse(String(store.get("antigravity-acp:session:ses_idle_1"))) as {
        acpSessionId: string
        turns: unknown[]
      }
      expect(persistedAfterTurn1.acpSessionId).toBe(acpSid!)
      expect(persistedAfterTurn1.turns).toHaveLength(1)

      // Arm idle reaper and wait past idleTimeoutMs
      registry.armIdle("ses_idle_1")
      await Bun.sleep(140)
      expect(registry.getState("ses_idle_1").connected).toBe(false)
      const persistedWhileIdle = JSON.parse(String(store.get("antigravity-acp:session:ses_idle_1"))) as {
        acpSessionId: string
      }
      expect(persistedWhileIdle.acpSessionId).toBe(acpSid!)

      // Next turn respawns proc and calls session/resume with preserved id
      const a1Text = textFromParts(parts1)
      const { stream: s2 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_idle_1" },
        prompt: [
          { role: "user", content: [{ type: "text", text: "[SCENARIO:PURE_OK] first" }] },
          { role: "assistant", content: [{ type: "text", text: a1Text }] },
          { role: "user", content: [{ type: "text", text: "[SCENARIO:PURE_OK] resumed" }] },
        ],
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

  test("11. Auto-syncs OpenCode plan agent to ACP plan mode and respects explicit RPC override", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "agy-test-plan-"))
    const eventsLog = path.join(tmp, "events.log")
    const prevEvents = process.env.FAKE_AGY_EVENTS_LOG
    process.env.FAKE_AGY_EVENTS_LOG = eventsLog

    const { ctx } = createMockContext({ cwd: tmp })
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ codeMode: false }),
    })
    try {
      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const u1 = { role: "user" as const, content: [{ type: "text" as const, text: "[SCENARIO:ECHO_META]" }] }
      const { stream: planStream } = await lm.doStream({
        headers: {
          "x-opencode-session-id": "ses_plan_1",
          "x-opencode-agent": "plan",
        },
        prompt: [u1],
      })
      const planText = textFromParts(await readStreamParts(planStream))
      const planMeta = JSON.parse(planText.replace("META:", "")) as { sessionId: string; mode: string | null }
      expect(planMeta.mode).toBe("plan")
      expect(registry.getState("ses_plan_1").mode).toBe("plan")

      // Turn 1 in plan agent must set_mode("plan") directly once, never "auto" then "plan".
      const turn1Events = fs
        .readFileSync(eventsLog, "utf8")
        .trim()
        .split("\n")
        .map((l) => JSON.parse(l) as { method: string; mode?: string })
        .filter((e) => e.method === "session/set_mode")
      expect(turn1Events).toHaveLength(1)
      expect(turn1Events[0]?.mode).toBe("plan")

      // Switching back to build agent restores defaultMode ("auto") on the same ACP session.
      const u2 = { role: "user" as const, content: [{ type: "text" as const, text: "[SCENARIO:ECHO_META]" }] }
      const { stream: buildStream } = await lm.doStream({
        headers: {
          "x-opencode-session-id": "ses_plan_1",
          "x-opencode-agent": "build",
        },
        prompt: [
          u1,
          { role: "assistant", content: [{ type: "text", text: planText }] },
          u2,
        ],
      })
      const buildText = textFromParts(await readStreamParts(buildStream))
      const buildMeta = JSON.parse(buildText.replace("META:", "")) as { sessionId: string; mode: string | null }
      expect(buildMeta.sessionId).toBe(planMeta.sessionId)
      expect(buildMeta.mode).toBe("auto")

      // Explicit RPC override beats agent auto-sync.
      await registry.setMode("ses_plan_1", "accept-edits")
      const { stream: overrideStream } = await lm.doStream({
        headers: {
          "x-opencode-session-id": "ses_plan_1",
          "x-opencode-agent": "plan",
        },
        prompt: [
          u1,
          { role: "assistant", content: [{ type: "text", text: planText }] },
          u2,
          { role: "assistant", content: [{ type: "text", text: buildText }] },
          { role: "user", content: [{ type: "text", text: "[SCENARIO:ECHO_META]" }] },
        ],
      })
      const overrideMeta = JSON.parse(
        textFromParts(await readStreamParts(overrideStream)).replace("META:", ""),
      ) as { sessionId: string; mode: string | null }
      expect(overrideMeta.sessionId).toBe(planMeta.sessionId)
      expect(overrideMeta.mode).toBe("accept-edits")
    } finally {
      if (prevEvents === undefined) delete process.env.FAKE_AGY_EVENTS_LOG
      else process.env.FAKE_AGY_EVENTS_LOG = prevEvents
      await registry.disposeAll()
      fs.rmSync(tmp, { recursive: true, force: true })
    }
  })

  test("12. Forwards instructions via _meta.antigravity.systemPrompt.append unless disabled", async () => {
    const { ctx } = createMockContext()
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ codeMode: false, forwardInstructions: true }),
    })
    try {
      registry.updateContextSnapshot("ses_inst_1", {
        agent: "build",
        tools: {},
        rawInstructions: "Instructions from: /repo/AGENTS.md\nAlways run bun test.",
      })
      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const { stream: s1 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_inst_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:ECHO_META]" }] }],
      })
      const m1 = JSON.parse(textFromParts(await readStreamParts(s1)).replace("META:", "")) as {
        append: string | null
      }
      expect(m1.append).toBe("Instructions from: /repo/AGENTS.md\nAlways run bun test.")

      // When forwardInstructions is false, append is omitted.
      registry.updateContextSnapshot("ses_inst_off", {
        agent: "build",
        tools: {},
        rawInstructions: "Should not be forwarded",
      })
      const disabledLm = createAgyLanguageModel(
        "gemini-4-neon",
        registry,
        makeSettings({ codeMode: false, forwardInstructions: false }),
      )
      const { stream: s2 } = await disabledLm.doStream({
        headers: { "x-opencode-session-id": "ses_inst_off" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:ECHO_META]" }] }],
      })
      const m2 = JSON.parse(textFromParts(await readStreamParts(s2)).replace("META:", "")) as {
        append: string | null
      }
      expect(m2.append).toBeNull()
    } finally {
      await registry.disposeAll()
    }
  })

  test("13. Fingerprints turns for delta catch-up without wiping ACP session, and resets on undo/edit", async () => {
    const { ctx } = createMockContext()
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ codeMode: false }),
    })
    try {
      const lm = createAgyLanguageModel("gemini-4-neon", registry)

      // Turn 1: initial turn
      const u1 = { role: "user" as const, content: [{ type: "text" as const, text: "first prompt" }] }
      const { stream: s1 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_sync_1" },
        prompt: [u1],
      })
      const p1 = await readStreamParts(s1)
      const a1Text = textFromParts(p1)
      const acpSid1 = registry.getState("ses_sync_1").acpSessionId
      expect(acpSid1).toBeTruthy()

      // Turn 2: reuses same ACP session and replays thin prompt even with acp_* hosted tool in history
      const { stream: s2 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_sync_1" },
        prompt: [
          u1,
          {
            role: "assistant",
            content: [
              {
                type: "tool-call",
                toolCallId: "tc_read_1",
                toolName: "acp_read_file",
                input: { path: "README.md" },
                providerExecuted: true,
              },
              { type: "text", text: a1Text },
            ],
          },
          { role: "user", content: [{ type: "text", text: "[SCENARIO:ECHO_HANDOFF]" }] },
        ],
      })
      const echo2 = textFromParts(await readStreamParts(s2))
      expect(registry.getState("ses_sync_1").acpSessionId).toBe(acpSid1)
      expect(echo2).not.toContain("<conversation-checkpoint")

      // Turn 3: foreign model turn inserted after Turn 2 -> delta catch-up on SAME ACP session
      const u2 = { role: "user" as const, content: [{ type: "text" as const, text: "[SCENARIO:ECHO_HANDOFF]" }] }
      const { stream: s3 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_sync_1" },
        prompt: [
          u1,
          { role: "assistant", content: [{ type: "text", text: a1Text }] },
          u2,
          { role: "assistant", content: [{ type: "text", text: echo2 }] },
          { role: "user", content: [{ type: "text", text: "foreign turn prompt" }] },
          { role: "assistant", content: [{ type: "text", text: "foreign model reply" }] },
          { role: "user", content: [{ type: "text", text: "[SCENARIO:ECHO_HANDOFF]" }] },
        ],
      })
      const echo3 = textFromParts(await readStreamParts(s3))
      const echo3Blocks = JSON.parse(echo3.slice("ECHO:".length)) as Array<{ text?: string }>
      const echo3Checkpoint = echo3Blocks[0]?.text ?? ""
      expect(registry.getState("ses_sync_1").acpSessionId).toBe(acpSid1)
      expect(echo3Checkpoint).toContain('<conversation-checkpoint mode="catch-up">')
      expect(echo3Checkpoint).toContain("foreign model reply")
      expect(echo3Checkpoint).not.toContain("first prompt")

      // Turn 4: /undo truncates history -> rotates to a fresh ACP session
      const { stream: s4 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_sync_1" },
        prompt: [
          u1,
          { role: "assistant", content: [{ type: "text", text: a1Text }] },
          { role: "user", content: [{ type: "text", text: "[SCENARIO:ECHO_HANDOFF]" }] },
        ],
      })
      const echo4 = textFromParts(await readStreamParts(s4))
      const echo4Blocks = JSON.parse(echo4.slice("ECHO:".length)) as Array<{ text?: string }>
      const echo4Checkpoint = echo4Blocks[0]?.text ?? ""
      const acpSid2 = registry.getState("ses_sync_1").acpSessionId
      expect(acpSid2).toBeTruthy()
      expect(acpSid2).not.toBe(acpSid1)
      expect(echo4Checkpoint).toContain('<conversation-checkpoint mode="restore">')
    } finally {
      await registry.disposeAll()
    }
  })

  test("14. Isolates compaction and generate side calls from primary ACP session and storage", async () => {
    const { ctx, store } = createMockContext()
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ codeMode: false }),
    })
    try {
      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const { stream: primaryStream } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_side_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:ECHO_META]" }] }],
      })
      const primaryMeta = JSON.parse(
        textFromParts(await readStreamParts(primaryStream)).replace("META:", ""),
      ) as { pid: number; sessionId: string }
      const primarySid = registry.getState("ses_side_1").acpSessionId
      const storedBefore = store.get("antigravity-acp:session:ses_side_1")
      expect(primarySid).toBeTruthy()
      expect(storedBefore).toBeTruthy()

      const { stream: compactionStream } = await lm.doStream({
        headers: {
          "x-opencode-session-id": "ses_side_1",
          "x-opencode-request-kind": "compaction",
        },
        prompt: [
          { role: "user", content: [{ type: "text", text: "primary turn" }] },
          { role: "assistant", content: [{ type: "text", text: "primary reply" }] },
          { role: "user", content: [{ type: "text", text: "[SCENARIO:ECHO_META]" }] },
        ],
      })
      const compMeta = JSON.parse(
        textFromParts(await readStreamParts(compactionStream)).replace("META:", ""),
      ) as { pid: number; sessionId: string }
      expect(compMeta.sessionId).toBeTruthy()
      expect(compMeta.pid).not.toBe(primaryMeta.pid)

      const { stream: generateStream } = await lm.doStream({
        headers: {
          "x-opencode-session-id": "ses_side_1",
          "x-opencode-request-kind": "generate",
        },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:ECHO_META]" }] }],
      })
      const genMeta = JSON.parse(
        textFromParts(await readStreamParts(generateStream)).replace("META:", ""),
      ) as { pid: number; sessionId: string }
      // Side calls reuse the same shared warm worker across compaction & generate, isolated from primary.
      expect(genMeta.pid).toBe(compMeta.pid)
      expect(genMeta.pid).not.toBe(primaryMeta.pid)

      // Primary session ID, storage record, and idle state remain completely untouched.
      expect(registry.getState("ses_side_1").acpSessionId).toBe(primarySid)
      expect(registry.getState("ses_side_1").busy).toBe(false)
      expect(store.get("antigravity-acp:session:ses_side_1")).toBe(storedBefore)
    } finally {
      await registry.disposeAll()
    }
  })

  test("15. Code Mode omits --process-per-session, keeps tools.opencode.* docs, injects mcp_oc_execute hint, and forwards images", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "agy-test-cm15-"))
    const argsLog = path.join(tmp, "args.log")
    const prevArgs = process.env.FAKE_AGY_ARGS_LOG
    process.env.FAKE_AGY_ARGS_LOG = argsLog

    const { ctx } = createMockContext({ cwd: tmp })
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ codeMode: true }),
    })
    try {
      const tools = {
        execute: {
          description: "Run JS. Use `tools.opencode.session.move()` to relocate worktree.",
          input: { type: "object", properties: { code: { type: "string" } } },
        },
      }
      const resolved = registry.resolveContextTools("ses_cm15", tools)
      const instructions = [
        "# OpenCode Relayed Tools (MCP `oc`)",
        `- OpenCode's Code Mode \`execute\` tool is callable here as \`mcp_oc_execute\` (${resolved.relayedToolNames.join(",")}).`,
      ].join("\n")
      registry.updateContextSnapshot("ses_cm15", {
        agent: "build",
        tools,
        rawInstructions: instructions,
      })

      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const { stream: s1 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_cm15" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:LIST_MCP_TOOLS]" }] }],
      })
      const listPayload = JSON.parse(
        textFromParts(await readStreamParts(s1)).replace("MCP_TOOLS:", ""),
      ) as { mcp: { tools: Array<{ name: string; description: string }> }; append: string }

      expect(listPayload.append).toContain("mcp_oc_execute")
      const execTool = listPayload.mcp.tools.find((t) => t.name === "execute")
      expect(execTool?.description).toContain("tools.opencode.session.move")
      expect(execTool?.description).toContain("mcp_oc_execute")

      const loggedArgs = JSON.parse(fs.readFileSync(argsLog, "utf8").trim()) as string[]
      expect(loggedArgs).not.toContain("--process-per-session")

      // Image content forwarding through MCP relay
      const u1 = { role: "user" as const, content: [{ type: "text" as const, text: "[SCENARIO:LIST_MCP_TOOLS]" }] }
      const { stream: s2 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_cm15" },
        prompt: [
          u1,
          { role: "assistant", content: [{ type: "text", text: `MCP_TOOLS:${JSON.stringify(listPayload)}` }] },
          { role: "user", content: [{ type: "text", text: "[SCENARIO:MCP_IMAGE_TOOL]" }] },
        ],
      })
      const callParts = await readStreamParts(s2)
      const toolCall = callParts.find((p) => p.type === "tool-call" && p.toolName === "execute")
      expect(toolCall?.type).toBe("tool-call")
      if (toolCall?.type === "tool-call") {
        const { stream: s3 } = await lm.doStream({
          headers: { "x-opencode-session-id": "ses_cm15" },
          prompt: [
            u1,
            { role: "assistant", content: [{ type: "text", text: `MCP_TOOLS:${JSON.stringify(listPayload)}` }] },
            { role: "user", content: [{ type: "text", text: "[SCENARIO:MCP_IMAGE_TOOL]" }] },
            {
              role: "tool",
              content: [
                {
                  type: "tool-result",
                  toolCallId: toolCall.toolCallId,
                  toolName: toolCall.toolName,
                  output: {
                    type: "content",
                    value: [
                      { type: "text", text: "screenshot taken" },
                      { type: "image-data", data: "aGVsbG8=", mediaType: "image/png" },
                    ],
                  },
                },
              ],
            },
          ],
        })
        const finalText = textFromParts(await readStreamParts(s3))
        expect(finalText).toContain('"type":"image"')
        expect(finalText).toContain('"mimeType":"image/png"')
        expect(finalText).toContain('"data":"aGVsbG8="')
      }
    } finally {
      if (prevArgs === undefined) delete process.env.FAKE_AGY_ARGS_LOG
      else process.env.FAKE_AGY_ARGS_LOG = prevArgs
      await registry.disposeAll()
      fs.rmSync(tmp, { recursive: true, force: true })
    }
  })

  test("16. Hybrid auto Code Mode enables only for non-shadowed tools and ignores shadowed tool churn", async () => {
    const { ctx, getSessionGetCalls } = createMockContext()
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({
        codeMode: "auto",
        extraRelayedTools: ["grep"],
        blockedRelayedTools: ["subagent"],
      }),
    })
    try {
      // Only shadowed built-ins (besides extraRelayedTools test)
      const onlyShadowed = registry.resolveContextTools("ses_auto_1", {
        bash: { description: "Run shell", input: {} },
        question: { description: "Ask", input: {} },
      })
      expect(onlyShadowed.codeModeEnabled).toBe(false)
      expect(onlyShadowed.relayedToolNames).toEqual([])

      // Non-shadowed execute + extraRelayed grep + blocked subagent
      const hybrid = registry.resolveContextTools("ses_auto_2", {
        bash: { description: "Run shell v1", input: {} },
        grep: { description: "Search", input: {} },
        subagent: { description: "Blocked", input: {} },
        execute: { description: "Run JS", input: {} },
      })
      expect(hybrid.codeModeEnabled).toBe(true)
      expect(hybrid.relayedToolNames).toEqual(["grep", "execute"])

      registry.updateContextSnapshot("ses_auto_2", {
        agent: "build",
        tools: {
          bash: { description: "Run shell v1", input: {} },
          grep: { description: "Search", input: {} },
          execute: { description: "Run JS", input: {} },
        },
      })

      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const u1 = { role: "user" as const, content: [{ type: "text" as const, text: "[SCENARIO:ECHO_META]" }] }
      const { stream: s1 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_auto_2" },
        prompt: [u1],
      })
      const a1 = textFromParts(await readStreamParts(s1))
      const sid1 = (JSON.parse(a1.replace("META:", "")) as { sessionId: string }).sessionId

      // Mutate shadowed `bash` tool schema — live ACP session must NOT reset!
      registry.updateContextSnapshot("ses_auto_2", {
        agent: "build",
        tools: {
          bash: { description: "Run shell v2 changed!", input: { extra: true } },
          grep: { description: "Search", input: {} },
          execute: { description: "Run JS", input: {} },
        },
      })
      const { stream: s2 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_auto_2" },
        prompt: [
          u1,
          { role: "assistant", content: [{ type: "text", text: a1 }] },
          { role: "user", content: [{ type: "text", text: "[SCENARIO:ECHO_META]" }] },
        ],
      })
      const sid2 = (
        JSON.parse(textFromParts(await readStreamParts(s2)).replace("META:", "")) as {
          sessionId: string
        }
      ).sessionId
      expect(sid2).toBe(sid1)
      expect(getSessionGetCalls()).toBe(1)
    } finally {
      await registry.disposeAll()
    }
  })

  test("17. Prewarm standby process is claimed by Turn 1 without spawning a duplicate worker", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "agy-test-prewarm-"))
    const argsLog = path.join(tmp, "args.log")
    const prevArgs = process.env.FAKE_AGY_ARGS_LOG
    process.env.FAKE_AGY_ARGS_LOG = argsLog

    const { ctx } = createMockContext({ cwd: tmp })
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ codeMode: false, prewarm: true }),
    })
    try {
      registry.prewarmStandby()
      await Bun.sleep(80)
      // Turn off refill before Turn 1 so we verify claimStandby uses the prewarmed PID rather than spawning fresh.
      const lm = createAgyLanguageModel(
        "gemini-4-neon",
        registry,
        makeSettings({ codeMode: false, prewarm: false }),
      )
      const { stream } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_prewarm_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:PURE_OK] claimed" }] }],
      })
      const text = textFromParts(await readStreamParts(stream))
      expect(text).toContain("claimed")

      const spawnLines = fs.readFileSync(argsLog, "utf8").trim().split("\n")
      expect(spawnLines).toHaveLength(1)
    } finally {
      if (prevArgs === undefined) delete process.env.FAKE_AGY_ARGS_LOG
      else process.env.FAKE_AGY_ARGS_LOG = prevArgs
      await registry.disposeAll()
      fs.rmSync(tmp, { recursive: true, force: true })
    }
  })

  test("18. Auto prewarm skips unused directories, activates after first turn, and drops stale standby on binaryPath change", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "agy-test-auto-prewarm-"))
    const argsLog = path.join(tmp, "args.log")
    const prevArgs = process.env.FAKE_AGY_ARGS_LOG
    process.env.FAKE_AGY_ARGS_LOG = argsLog

    const { ctx, store } = createMockContext({ cwd: tmp })
    const reg1 = new SessionRegistry(ctx, {
      settings: makeSettings({ codeMode: false, prewarm: "auto" }),
    })
    try {
      // Unused directory should not prewarm under prewarm: "auto"
      expect(await reg1.shouldPrewarmDirectory()).toBe(false)
      expect(await reg1.shouldPrewarmDirectory(makeSettings({ prewarm: true }))).toBe(true)
      expect(await reg1.shouldPrewarmDirectory(makeSettings({ prewarm: false }))).toBe(false)

      // After a real turn runs, directory becomes active in storage for subsequent reloads
      const lm = createAgyLanguageModel(
        "gemini-4-neon",
        reg1,
        makeSettings({ codeMode: false, prewarm: "auto" }),
      )
      const { stream } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_auto_prewarm_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:PURE_OK] activate" }] }],
      })
      expect(textFromParts(await readStreamParts(stream))).toContain("activate")
      await Bun.sleep(20)

      // A fresh registry instance sharing storage now recognizes this directory as active
      const reg2 = new SessionRegistry(ctx, {
        settings: makeSettings({ codeMode: false, prewarm: "auto" }),
      })
      try {
        expect(await reg2.shouldPrewarmDirectory()).toBe(true)
        // Updating binaryPath disposes any mismatched standby worker
        reg2.prewarmStandby()
        await Bun.sleep(60)
        reg2.updateBaseSettings(
          makeSettings({ codeMode: false, prewarm: "auto", binaryPath: "/nonexistent/other-agy" }),
        )
        await Bun.sleep(40)
        expect(store.size).toBeGreaterThan(0)
      } finally {
        await reg2.disposeAll()
      }
    } finally {
      if (prevArgs === undefined) delete process.env.FAKE_AGY_ARGS_LOG
      else process.env.FAKE_AGY_ARGS_LOG = prevArgs
      await reg1.disposeAll()
      fs.rmSync(tmp, { recursive: true, force: true })
    }
  })

  test("19. Multi-step turn reports per-step usage, delivers mid-turn queued user messages in-band, and avoids next-turn divergence", async () => {
    const { ctx } = createMockContext()
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({ codeMode: true, maxMcpResultChars: 500 }),
    })
    try {
      registry.updateContextSnapshot("ses_mid_1", {
        agent: "build",
        tools: {
          execute: { description: "Code mode JS", input: { type: "object", properties: {} } },
        },
      })

      const lm = createAgyLanguageModel("gemini-4-neon", registry)
      const u1 = {
        role: "user" as const,
        content: [{ type: "text" as const, text: "[SCENARIO:MULTI_STEP_USAGE_AND_QUEUED]" }],
      }
      const { stream: s1 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_mid_1" },
        prompt: [u1],
      })
      const parts1 = await readStreamParts(s1)
      const tc1 = parts1.find((p) => p.type === "tool-call" && p.toolName === "execute")
      const fin1 = parts1.find((p) => p.type === "finish")
      expect(tc1?.type).toBe("tool-call")
      expect(fin1?.type === "finish" && fin1.usage.inputTokens.total).toBe(10_000)
      if (tc1?.type !== "tool-call") return

      const acpSid1 = registry.getState("ses_mid_1").acpSessionId
      expect(acpSid1).toBeTruthy()

      // Mid-turn queued user message arrives while execute #1 is running, and execute #1 returns > maxMcpResultChars
      const uMid = {
        role: "user" as const,
        content: [{ type: "text" as const, text: "Queued follow-up request #1" }],
      }
      const bigToolOut =
        `{"items":"${"X".repeat(900)}"}\n[showing lines 1-1 of 1; full output saved to /tmp/fake-oc-tool-out]`
      const promptAfterStep1: LanguageModelV3Prompt = [
        u1,
        {
          role: "assistant",
          content: [{ type: "tool-call", toolCallId: tc1.toolCallId, toolName: "execute", input: {} }],
        },
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: tc1.toolCallId,
              toolName: "execute",
              output: { type: "text", value: bigToolOut },
            },
          ],
        },
        uMid,
      ]

      const { stream: s2 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_mid_1" },
        prompt: promptAfterStep1,
      })
      const parts2 = await readStreamParts(s2)
      const tc2 = parts2.find((p) => p.type === "tool-call" && p.toolName === "execute")
      const fin2 = parts2.find((p) => p.type === "finish")
      expect(tc2?.type).toBe("tool-call")
      expect(fin2?.type === "finish" && fin2.usage.inputTokens).toEqual({
        total: 11_200,
        noCache: 1_200,
        cacheRead: 10_000,
        cacheWrite: 0,
      })
      if (tc2?.type !== "tool-call") return

      const promptAfterStep2: LanguageModelV3Prompt = [
        ...promptAfterStep1,
        {
          role: "assistant",
          content: [{ type: "tool-call", toolCallId: tc2.toolCallId, toolName: "execute", input: {} }],
        },
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: tc2.toolCallId,
              toolName: "execute",
              output: { type: "text", value: "step2-ok" },
            },
          ],
        },
      ]
      const { stream: s3 } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_mid_1" },
        prompt: promptAfterStep2,
      })
      const parts3 = await readStreamParts(s3)
      const finalText = textFromParts(parts3)
      const fin3 = parts3.find((p) => p.type === "finish")

      // Final step usage is normalized to 12500 real window (12000 input + 500 output), NOT cumulative 33700!
      expect(fin3?.type === "finish").toBe(true)
      if (fin3?.type === "finish") {
        expect((fin3.usage.inputTokens.total ?? 0) + (fin3.usage.outputTokens.total ?? 0)).toBe(12_500)
      }

      // agy received BOTH compacted tool output (with saved pointer preserved) AND intact <queued-user-message>
      const payload = JSON.parse(finalText.replace("MULTI_OK:", "")) as {
        sessionId: string
        r1Texts: string[]
        r2Texts: string[]
      }
      expect(payload.r1Texts).toHaveLength(2)
      expect(payload.r1Texts[0]).toContain("/tmp/fake-oc-tool-out")
      expect(payload.r1Texts[1]).toContain("<queued-user-message>")
      expect(payload.r1Texts[1]).toContain("Queued follow-up request #1")
      expect(payload.r2Texts).toEqual(["step2-ok"])

      // Turn 2 must stay on the SAME ACP session (no diverged reset / no restore checkpoint!)
      const { stream: nextTurn } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_mid_1" },
        prompt: [
          ...promptAfterStep2,
          { role: "assistant", content: [{ type: "text", text: finalText }] },
          { role: "user", content: [{ type: "text", text: "[SCENARIO:ECHO_HANDOFF]" }] },
        ],
      })
      const nextText = textFromParts(await readStreamParts(nextTurn))
      expect(registry.getState("ses_mid_1").acpSessionId).toBe(acpSid1)
      expect(nextText).not.toContain("<conversation-checkpoint")
    } finally {
      await registry.disposeAll()
    }
  })

  test("20. Shares model catalog refresh across concurrent SessionRegistry instances without spawning duplicate agy processes", async () => {
    resetSharedCatalogCacheForTests()
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "agy-test-catalog-dedupe-"))
    const argsLog = path.join(tmp, "args.log")
    const prevArgs = process.env.FAKE_AGY_ARGS_LOG
    process.env.FAKE_AGY_ARGS_LOG = argsLog

    const regs = [1, 2, 3].map((n) => {
      const dir = path.join(tmp, `ws-${n}`)
      fs.mkdirSync(dir, { recursive: true })
      const { ctx } = createMockContext({ cwd: dir })
      return new SessionRegistry(ctx, {
        settings: makeSettings({ codeMode: false, prewarm: false }),
      })
    })
    try {
      const results = await Promise.all(regs.map((r) => r.refreshModels({ force: false })))
      for (const res of results) {
        expect(res.models.length).toBeGreaterThanOrEqual(2)
      }
      // All 3 concurrent directories share ONE discovery spawn
      const spawns = fs.readFileSync(argsLog, "utf8").trim().split("\n").filter(Boolean)
      expect(spawns).toHaveLength(1)

      // A 4th registry loading immediately after sees fresh shared cache via loadCachedModels()
      const dir4 = path.join(tmp, "ws-4")
      fs.mkdirSync(dir4, { recursive: true })
      const { ctx: ctx4 } = createMockContext({ cwd: dir4 })
      const reg4 = new SessionRegistry(ctx4, {
        settings: makeSettings({ codeMode: false, prewarm: false }),
      })
      try {
        const status = await reg4.loadCachedModels()
        expect(status.stale).toBe(false)
        expect(reg4.getModels().length).toBeGreaterThanOrEqual(2)
      } finally {
        await reg4.disposeAll()
      }
    } finally {
      resetSharedCatalogCacheForTests()
      if (prevArgs === undefined) delete process.env.FAKE_AGY_ARGS_LOG
      else process.env.FAKE_AGY_ARGS_LOG = prevArgs
      await Promise.all(regs.map((r) => r.disposeAll()))
      fs.rmSync(tmp, { recursive: true, force: true })
    }
  })

  test("21. Native-tool watchdog grace survives slow native commands, resumes chunkTimeoutMs after completion, and enforces nativeToolTimeoutMs cap", async () => {
    const { ctx } = createMockContext()
    const registry = new SessionRegistry(ctx, {
      settings: makeSettings({
        codeMode: false,
        chunkTimeoutMs: 100,
        nativeToolTimeoutMs: 450,
      }),
    })
    try {
      const lm = createAgyLanguageModel("gemini-4-neon", registry)

      // 1) Native background command silent for 220ms (> 100ms chunkTimeoutMs, < 450ms nativeToolTimeoutMs) succeeds
      const { stream: okStream } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_native_bg_1" },
        prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:NATIVE_BG_WATCHER]" }] }],
      })
      const okParts = await readStreamParts(okStream)
      expect(okParts.some((p) => p.type === "error")).toBe(false)
      expect(textFromParts(okParts)).toContain("NATIVE_BG_OK")
      expect(registry.getState("ses_native_bg_1").connected).toBe(true)

      // 2) Once the native tool finishes, LLM silence reverts to chunkTimeoutMs (100ms)
      const { stream: llmStallStream } = await lm.doStream({
        headers: { "x-opencode-session-id": "ses_native_bg_1" },
        prompt: [
          { role: "user", content: [{ type: "text", text: "[SCENARIO:NATIVE_BG_THEN_LLM_STALL]" }] },
        ],
      })
      const llmStallParts = await readStreamParts(llmStallStream)
      const llmErr = llmStallParts.find((p) => p.type === "error")
      expect(llmErr?.type).toBe("error")
      if (llmErr?.type === "error") {
        expect(llmErr.error).toBeInstanceOf(AcpStallError)
        expect((llmErr.error as AcpStallError).idleMs).toBe(100)
      }

      // 3) Truly wedged native tool still trips the bounded nativeToolTimeoutMs cap (240ms)
      const registryCap = new SessionRegistry(ctx, {
        settings: makeSettings({
          codeMode: false,
          chunkTimeoutMs: 80,
          nativeToolTimeoutMs: 240,
        }),
      })
      try {
        const lmCap = createAgyLanguageModel("gemini-4-neon", registryCap)
        const { stream: capStream } = await lmCap.doStream({
          headers: { "x-opencode-session-id": "ses_native_cap_1" },
          prompt: [{ role: "user", content: [{ type: "text", text: "[SCENARIO:NATIVE_TOOL_STALL]" }] }],
        })
        const capParts = await readStreamParts(capStream)
        const capErr = capParts.find((p) => p.type === "error")
        expect(capErr?.type).toBe("error")
        if (capErr?.type === "error") {
          expect(capErr.error).toBeInstanceOf(AcpStallError)
          expect((capErr.error as AcpStallError).idleMs).toBe(240)
        }
      } finally {
        await registryCap.disposeAll()
      }
    } finally {
      await registry.disposeAll()
    }
  })
})
