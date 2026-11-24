import { describe, expect, test } from "bun:test"
import type { LanguageModelV3StreamPart } from "@ai-sdk/provider"
import { ActiveTurn, formatNativeToolResult, takeSmoothSlice } from "../src/acp/turn.js"

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
})
