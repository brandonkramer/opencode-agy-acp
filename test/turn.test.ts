import { describe, expect, test } from "bun:test"
import type { LanguageModelV3StreamPart } from "@ai-sdk/provider"
import {
  ActiveTurn,
  buildFinalTurnUsage,
  buildStepUsage,
  buildUsage,
  extractNativeToolTimeoutMs,
  formatNativeToolResult,
  takeSmoothSlice,
} from "../src/acp/turn.js"

function makeTurn(overrides?: {
  readonly smoothStreaming?: boolean
  readonly smoothIntervalMs?: number
}) {
  return new ActiveTurn({
    pauseWatchdog: () => {},
    resumeWatchdog: () => {},
    onApiErrorText: () => {},
    onInterruptRequested: () => {},
    onTurnSettled: () => {},
    getKnownRelayedNames: () => new Set(),
    nativeFlushGraceMs: 100,
    smoothStreaming: overrides?.smoothStreaming ?? true,
    smoothIntervalMs: overrides?.smoothIntervalMs ?? 5,
  })
}

async function drainTurn(
  turn: ActiveTurn,
  onStart: () => void,
): Promise<LanguageModelV3StreamPart[]> {
  const stream = turn.createBoundStream(undefined, onStart)
  const reader = stream.getReader()
  const parts: LanguageModelV3StreamPart[] = []
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    parts.push(value)
  }
  return parts
}

describe("smooth streaming pacer (src/acp/turn.ts)", () => {
  test("takeSmoothSlice splits on word boundaries, preserves emojis, and reconstructs losslessly", () => {
    const source =
      "Hello 🚀🎉 world, this is a reasonably long paragraph with multiple words and unicode ✨ symbols."
    let rest = source
    const pieces: string[] = []
    while (rest.length > 0) {
      const [head, tail] = takeSmoothSlice(rest, 16)
      expect(head.length).toBeGreaterThan(0)
      pieces.push(head)
      rest = tail
    }
    expect(pieces.join("")).toBe(source)
    expect(pieces.length).toBeGreaterThan(2)
    // First slice should break cleanly at whitespace around target=16
    expect(pieces[0]?.endsWith(" ")).toBe(true)
  })

  test("emits small chunks immediately in one delta without pacing", async () => {
    const turn = makeTurn({ smoothStreaming: true, smoothIntervalMs: 5 })
    const parts = await drainTurn(turn, () => {
      turn.handleUpdate({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "Short delta." },
      })
      turn.finalizeStop()
    })
    const deltas = parts.filter((p) => p.type === "text-delta")
    expect(deltas).toHaveLength(1)
    expect(deltas[0]?.type === "text-delta" && deltas[0].delta).toBe("Short delta.")
  })

  test("drips large burst chunks over ticks and reconstructs exact text", async () => {
    const turn = makeTurn({ smoothStreaming: true, smoothIntervalMs: 5 })
    const burst =
      "First sentence of a large Gemini paragraph burst with plenty of words so the pacer drips it smoothly. Second sentence keeps going to ensure multiple slices arrive cleanly."
    const parts = await drainTurn(turn, () => {
      turn.handleUpdate({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: burst },
      })
      setTimeout(() => turn.finalizeStop(), 120)
    })

    const textDeltas = parts
      .filter((p): p is Extract<LanguageModelV3StreamPart, { type: "text-delta" }> => p.type === "text-delta")
      .map((p) => p.delta)
    expect(textDeltas.length).toBeGreaterThan(3)
    expect(textDeltas.join("")).toBe(burst)
  })

  test("flushes remaining paced thought and text synchronously on transitions and finalizeStop", async () => {
    const turn = makeTurn({ smoothStreaming: true, smoothIntervalMs: 50 })
    const thought =
      "Long Gemini thought summary block that arrives all at once and would otherwise buffer when text starts immediately afterward."
    const text =
      "Long final answer block followed immediately by finalizeStop before any pacing timer fires."

    const t0 = Date.now()
    const parts = await drainTurn(turn, () => {
      turn.handleUpdate({
        sessionUpdate: "agent_thought_chunk",
        content: { type: "text", text: thought },
      })
      // Immediately transition to message chunk before the 50ms tick fires
      turn.handleUpdate({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text },
      })
      // Immediately finish before the 50ms tick fires
      turn.finalizeStop()
    })
    expect(Date.now() - t0).toBeLessThan(35)

    const types = parts.map((p) => p.type)
    const rEndIdx = types.indexOf("reasoning-end")
    const tStartIdx = types.indexOf("text-start")
    const tEndIdx = types.indexOf("text-end")
    const finishIdx = types.indexOf("finish")

    expect(rEndIdx).toBeGreaterThan(0)
    expect(tStartIdx).toBeGreaterThan(rEndIdx)
    expect(tEndIdx).toBeGreaterThan(tStartIdx)
    expect(finishIdx).toBeGreaterThan(tEndIdx)

    // No deltas may appear after their closing end markers
    for (let i = rEndIdx + 1; i < types.length; i++) {
      expect(types[i]).not.toBe("reasoning-delta")
    }
    for (let i = tEndIdx + 1; i < types.length; i++) {
      expect(types[i]).not.toBe("text-delta")
    }

    const rText = parts
      .filter((p): p is Extract<LanguageModelV3StreamPart, { type: "reasoning-delta" }> => p.type === "reasoning-delta")
      .map((p) => p.delta)
      .join("")
    const aText = parts
      .filter((p): p is Extract<LanguageModelV3StreamPart, { type: "text-delta" }> => p.type === "text-delta")
      .map((p) => p.delta)
      .join("")
    expect(rText).toBe(thought)
    expect(aText).toBe(text)
  })

  test("appends back-to-back same-block chunks continuously without mid-stream jumps", async () => {
    const turn = makeTurn({ smoothStreaming: true, smoothIntervalMs: 5 })
    const chunk1 =
      "First burst paragraph arriving from Gemini with plenty of words to stay queued when chunk two lands. "
    const chunk2 =
      "Second burst paragraph arrives ten milliseconds later and should flow seamlessly without dumping chunk one."

    const parts = await drainTurn(turn, () => {
      turn.handleUpdate({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: chunk1 },
      })
      setTimeout(() => {
        turn.handleUpdate({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: chunk2 },
        })
      }, 10)
      setTimeout(() => turn.finalizeStop(), 180)
    })

    const deltas = parts
      .filter((p): p is Extract<LanguageModelV3StreamPart, { type: "text-delta" }> => p.type === "text-delta")
      .map((p) => p.delta)
    expect(deltas.join("")).toBe(chunk1 + chunk2)
    expect(deltas.length).toBeGreaterThan(6)
    // No single delta should dump the whole remainder of chunk1 (~80+ chars) in a jump
    expect(Math.max(...deltas.map((d) => d.length))).toBeLessThan(45)
  })

  test("formatNativeToolResult strips agy command boilerplate, normalizes CRLF, preserves non-zero exit, and truncates huge outputs", () => {
    const ok = formatNativeToolResult({
      sessionUpdate: "tool_call_update",
      rawOutput: {
        output:
          "Created At: 2026-11-24T23:21:14+01:00\nCompleted At: 2026-11-24T23:21:15+01:00\nCommand exited with code 0.\n\nCommand output:\nhi\r\nbye\r\n",
        exit_code: 0,
      },
    })
    expect(ok).toBe("hi\nbye")

    const err = formatNativeToolResult({
      sessionUpdate: "tool_call_update",
      rawOutput: { output: "Created At: x\nboom\r\n", exit_code: 2 },
    })
    expect(err).toBe("[exit 2]\nboom\n")

    const huge = formatNativeToolResult({
      sessionUpdate: "tool_call_update",
      rawOutput: { output: "x".repeat(20_000), exit_code: 0 },
    })
    expect(typeof huge).toBe("string")
    expect(String(huge)).toContain("... [truncated 4000 chars]")
  })

  test("buildUsage preserves exclusive agy inputTokens + cachedReadTokens without clamping or zeroing noCache", () => {
    // Real agy Turn 1 (cache miss)
    const turn1 = buildUsage({
      inputTokens: 10_403,
      outputTokens: 29,
      cachedReadTokens: 0,
      cachedWriteTokens: 0,
      thoughtTokens: 27,
      totalTokens: 10_432,
    })
    expect(turn1.inputTokens).toEqual({
      total: 10_403,
      noCache: 10_403,
      cacheRead: 0,
      cacheWrite: 0,
    })
    expect(turn1.outputTokens).toEqual({ total: 29, text: 2, reasoning: 27 })

    // Real agy Turn 2 (cache hit: inputTokens is non-cached only; totalTokens = input + cachedRead + output)
    const turn2 = buildUsage({
      inputTokens: 2_146,
      outputTokens: 25,
      cachedReadTokens: 8_358,
      cachedWriteTokens: 0,
      thoughtTokens: 23,
      totalTokens: 10_529,
    })
    expect(turn2.inputTokens).toEqual({
      total: 10_504,
      noCache: 2_146,
      cacheRead: 8_358,
      cacheWrite: 0,
    })
    expect(turn2.outputTokens).toEqual({ total: 25, text: 2, reasoning: 23 })

    // Exclusive multi-step turn where rawInput > cachedReadTokens and totalTokens equals sum
    const multiStep = buildUsage({
      inputTokens: 50_000,
      cachedReadTokens: 40_000,
      outputTokens: 5_000,
      totalTokens: 95_000,
    })
    expect(multiStep.inputTokens).toEqual({
      total: 90_000,
      noCache: 50_000,
      cacheRead: 40_000,
      cacheWrite: undefined,
    })

    // Inclusive ACP server (inputTokens already includes cachedReadTokens)
    const inclusive = buildUsage({
      inputTokens: 10_504,
      cachedReadTokens: 8_358,
      outputTokens: 25,
      totalTokens: 10_529,
    })
    expect(inclusive.inputTokens).toEqual({
      total: 10_504,
      noCache: 2_146,
      cacheRead: 8_358,
      cacheWrite: undefined,
    })
  })

  test("buildStepUsage and buildFinalTurnUsage normalize multi-step cumulative agy usage to actual context window", () => {
    // Reproduces live 4-step agy probe: steps 10021, 10760, 11751, 13022; cumulative total = 45554
    const s1 = buildStepUsage(10_021, 0)
    const s2 = buildStepUsage(10_760, 10_021)
    const s3 = buildStepUsage(11_751, 10_760)
    expect(s1.inputTokens).toEqual({ total: 10_021, noCache: 10_021, cacheRead: 0, cacheWrite: 0 })
    expect(s2.inputTokens).toEqual({ total: 10_760, noCache: 739, cacheRead: 10_021, cacheWrite: 0 })
    expect(s3.inputTokens).toEqual({ total: 11_751, noCache: 991, cacheRead: 10_760, cacheWrite: 0 })

    const emittedInput = (s1.inputTokens.total ?? 0) + (s2.inputTokens.total ?? 0) + (s3.inputTokens.total ?? 0)
    const emittedCr = (s1.inputTokens.cacheRead ?? 0) + (s2.inputTokens.cacheRead ?? 0) + (s3.inputTokens.cacheRead ?? 0)

    const finalStep = buildFinalTurnUsage({
      raw: {
        inputTokens: 18_604,
        cachedReadTokens: 26_395,
        cachedWriteTokens: 0,
        outputTokens: 555,
        thoughtTokens: 0,
        totalTokens: 45_554,
      },
      activeContextUsed: 13_022,
      emittedIntermediateInput: emittedInput,
      emittedIntermediateCacheRead: emittedCr,
    })
    // Final assistant message matches true single-step context occupancy (12467 + 555 = 13022), NOT 45554
    expect((finalStep.inputTokens.total ?? 0) + (finalStep.outputTokens.total ?? 0)).toBe(13_022)
    // Cumulative totals across steps 1..4 reconcile exactly to agy's result.usage
    expect(emittedCr + (finalStep.inputTokens.cacheRead ?? 0)).toBe(26_395)
    expect(
      (s1.inputTokens.noCache ?? 0) +
        (s2.inputTokens.noCache ?? 0) +
        (s3.inputTokens.noCache ?? 0) +
        (finalStep.inputTokens.noCache ?? 0),
    ).toBe(18_604)

    // Native-tool-only multi-step turn (0 intermediate MCP flushes, cumulative 844K vs 81.5K real window)
    const nativeMulti = buildFinalTurnUsage({
      raw: {
        inputTokens: 82_728,
        cachedReadTokens: 761_744,
        cachedWriteTokens: 0,
        outputTokens: 16_869,
        thoughtTokens: 7_538,
        totalTokens: 861_341,
      },
      activeContextUsed: 81_508,
    })
    expect((nativeMulti.inputTokens.total ?? 0) + (nativeMulti.outputTokens.total ?? 0)).toBe(81_508)
    expect(nativeMulti.inputTokens.cacheRead).toBeGreaterThan(50_000)

    // Single-step turn (inputTotal <= contextUsed) passes through verbatim
    const singleStep = buildFinalTurnUsage({
      raw: { inputTokens: 2_146, cachedReadTokens: 8_358, outputTokens: 25, totalTokens: 10_529 },
      activeContextUsed: 10_529,
    })
    expect(singleStep.inputTokens).toEqual({
      total: 10_504,
      noCache: 2_146,
      cacheRead: 8_358,
      cacheWrite: undefined,
    })
  })

  test("extractNativeToolTimeoutMs and ActiveTurn watchdog override track in-flight native tools", () => {
    expect(extractNativeToolTimeoutMs({}, 900_000)).toBe(900_000)
    expect(extractNativeToolTimeoutMs({ NotificationTimeoutSeconds: 600 }, 900_000)).toBe(900_000)
    expect(extractNativeToolTimeoutMs({ NotificationTimeoutSeconds: 1200 }, 900_000)).toBe(1_320_000)
    expect(extractNativeToolTimeoutMs({ DurationSeconds: 10_000 }, 900_000)).toBe(3_600_000)

    const overrides: Array<number | undefined> = []
    const turn = new ActiveTurn({
      pauseWatchdog: () => {},
      resumeWatchdog: () => {},
      setWatchdogIdleOverride: (ms) => overrides.push(ms),
      nativeToolTimeoutMs: 900_000,
      onApiErrorText: () => {},
      onInterruptRequested: () => {},
      onTurnSettled: () => {},
      getKnownRelayedNames: () => new Set(),
      nativeFlushGraceMs: 100,
    })

    turn.handleUpdate({
      sessionUpdate: "tool_call",
      toolCallId: "tc_1",
      name: "run_command",
      status: "in_progress",
      rawInput: { CommandLine: "watch ci" },
    })
    turn.handleUpdate({
      sessionUpdate: "tool_call",
      toolCallId: "tc_2",
      name: "run_command",
      status: "in_progress",
      rawInput: { NotificationTimeoutSeconds: 1200 },
    })
    turn.handleUpdate({
      sessionUpdate: "tool_call_update",
      toolCallId: "tc_2",
      status: "completed",
      rawOutput: { output: "done", exit_code: 0 },
    })
    turn.handleUpdate({
      sessionUpdate: "tool_call_update",
      toolCallId: "tc_1",
      status: "completed",
      rawOutput: { output: "done", exit_code: 0 },
    })

    expect(overrides).toEqual([900_000, 1_320_000, 900_000, undefined])
  })
})
