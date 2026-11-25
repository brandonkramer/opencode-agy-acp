import { describe, expect, test } from "bun:test"
import type { LanguageModelV3Prompt } from "@ai-sdk/provider"
import {
  buildCheckpointPromptBlocks,
  buildSavedHistory,
  buildSavedHistoryFromGroups,
  buildSideCallPromptBlocks,
  decideSync,
  extractNewMidTurnUserMessages,
  fingerprintGroups,
  formatMidTurnUserMessages,
  hashAssistantTextBlocks,
  parsePersistedRecord,
} from "../src/acp/sync.js"
import { buildThinAcpPromptBlocks } from "../src/acp/history.js"
import { PROVIDER_ID } from "../src/types.js"

describe("turn fingerprinting and delta sync", () => {
  test("parsePersistedRecord handles legacy string, valid v1 JSON, and invalid inputs", () => {
    expect(parsePersistedRecord("legacy-acp-id")).toEqual({
      version: 1,
      acpSessionId: "legacy-acp-id",
      turns: [],
    })
    expect(
      parsePersistedRecord(
        JSON.stringify({
          version: 1,
          acpSessionId: "acp_1",
          lastModelId: "gemini-4-neon-high",
          turns: [{ userHash: "u1", assistantHash: "a1" }],
        }),
      ),
    ).toEqual({
      version: 1,
      acpSessionId: "acp_1",
      lastModelId: "gemini-4-neon-high",
      turns: [{ userHash: "u1", assistantHash: "a1" }],
    })
    expect(parsePersistedRecord("{broken json")).toBeUndefined()
    expect(parsePersistedRecord(JSON.stringify({ version: 2 }))).toBeUndefined()
  })

  test("fingerprintGroups skips acp_* hosted tool calls & reasoning so Turn 2 matches Turn 1", () => {
    const turn1User: LanguageModelV3Prompt[number] = {
      role: "user",
      content: [{ type: "text", text: "Inspect README" }],
    }
    const recordedTurns = buildSavedHistory({
      priorGroups: [],
      currentUserMessages: [turn1User],
      assistantTextBlocks: ["Let me check.", "Done inspecting."],
    })

    const turn2Prompt: LanguageModelV3Prompt = [
      { role: "system", content: "system" },
      turn1User,
      {
        role: "assistant",
        content: [
          { type: "reasoning", text: "internal thinking" },
          { type: "text", text: "Let me check." },
          {
            type: "tool-call",
            toolCallId: "tc_1",
            toolName: "acp_read_file",
            input: { path: "README.md" },
            providerExecuted: true,
          },
          {
            type: "tool-result",
            toolCallId: "tc_1",
            toolName: "acp_read_file",
            output: { type: "text", value: "contents" },
            providerMetadata: { [PROVIDER_ID]: { acp: true } },
          } as never,
          { type: "text", text: "Done inspecting." },
        ],
      },
      {
        role: "assistant",
        content: [{ type: "text", text: "local status note" }],
        providerMetadata: { [PROVIDER_ID]: { localStatus: true } },
      } as never,
      { role: "user", content: [{ type: "text", text: "Follow up question" }] },
    ]

    const groups = fingerprintGroups(turn2Prompt)
    expect(groups).toHaveLength(2)
    const priorGroups = groups.slice(0, -1)
    expect(priorGroups[0]?.assistantHash).toBe(hashAssistantTextBlocks(["Let me check.", "Done inspecting."]))

    const decision = decideSync({
      recorded: { version: 1, acpSessionId: "acp_1", turns: recordedTurns },
      priorGroups,
      hasLiveOrResumableAcpSession: true,
    })
    expect(decision).toEqual({ kind: "continue", matchedCount: 1 })
  })

  test("decideSync returns catch-up for unseen trailing turns, and diverged on undo/edit", () => {
    const u1: LanguageModelV3Prompt[number] = { role: "user", content: [{ type: "text", text: "turn 1" }] }
    const u2: LanguageModelV3Prompt[number] = { role: "user", content: [{ type: "text", text: "turn 2" }] }
    const u3: LanguageModelV3Prompt[number] = { role: "user", content: [{ type: "text", text: "turn 3" }] }

    const recordedTurns = buildSavedHistory({
      priorGroups: [],
      currentUserMessages: [u1],
      assistantTextBlocks: ["agy answer 1"],
    })
    const recorded = { version: 1 as const, acpSessionId: "acp_live", turns: recordedTurns }

    // Cross-model turn inserted between ACP turns
    const catchUpPrompt: LanguageModelV3Prompt = [
      u1,
      { role: "assistant", content: [{ type: "text", text: "agy answer 1" }] },
      u2,
      { role: "assistant", content: [{ type: "text", text: "gpt answer 2" }] },
      u3,
    ]
    const catchUpGroups = fingerprintGroups(catchUpPrompt)
    const catchUp = decideSync({
      recorded,
      priorGroups: catchUpGroups.slice(0, -1),
      hasLiveOrResumableAcpSession: true,
    })
    expect(catchUp.kind).toBe("catch-up")
    if (catchUp.kind === "catch-up") {
      expect(catchUp.unseenGroups).toHaveLength(1)
      expect(catchUp.matchedCount).toBe(1)
      const blocks = buildCheckpointPromptBlocks({
        groups: catchUp.unseenGroups,
        currentUserMessages: catchUpGroups.at(-1)?.userMessages,
        mode: "catch-up",
      })
      expect(blocks[0]).toMatchObject({ type: "text" })
      expect((blocks[0] as { text: string }).text).toContain('<conversation-checkpoint mode="catch-up">')
      expect((blocks[0] as { text: string }).text).toContain("gpt answer 2")
      expect((blocks[0] as { text: string }).text).not.toContain("agy answer 1")
      expect((blocks[1] as { text: string }).text).toBe("turn 3")
    }

    // Undo removes turn 1
    const undoDecision = decideSync({
      recorded,
      priorGroups: [],
      hasLiveOrResumableAcpSession: true,
    })
    expect(undoDecision.kind).toBe("diverged")

    // Edit changes turn 1 text
    const editedPrompt: LanguageModelV3Prompt = [
      { role: "user", content: [{ type: "text", text: "edited turn 1" }] },
      { role: "assistant", content: [{ type: "text", text: "agy answer 1" }] },
      u2,
    ]
    const editDecision = decideSync({
      recorded,
      priorGroups: fingerprintGroups(editedPrompt).slice(0, -1),
      hasLiveOrResumableAcpSession: true,
    })
    expect(editDecision.kind).toBe("diverged")
  })

  test("buildSideCallPromptBlocks formats compaction and generate prompts chronologically", () => {
    const blocks = buildSideCallPromptBlocks(
      [
        { role: "system", content: "Summarize concisely." },
        { role: "user", content: [{ type: "text", text: "First request" }] },
        { role: "assistant", content: [{ type: "text", text: "First response" }] },
        { role: "user", content: [{ type: "text", text: "Produce a summary." }] },
      ],
      "compaction",
    )
    expect(blocks.length).toBeGreaterThanOrEqual(2)
    expect((blocks[0] as { text: string }).text).toContain("<compaction-context>")
    expect((blocks[0] as { text: string }).text).toContain("First response")
    expect((blocks[1] as { text: string }).text).toBe("Produce a summary.")
  })

  test("merges consecutive user messages and ignores ephemeral <system-update> wrappers", () => {
    const sysUpdate: LanguageModelV3Prompt[number] = {
      role: "user",
      content: [{ type: "text", text: "<system-update>\nYou are in Plan mode.\n</system-update>" }],
    }
    const realPrompt: LanguageModelV3Prompt[number] = {
      role: "user",
      content: [{ type: "text", text: "hello" }],
    }

    // 1. Consecutive user messages merge into a single group -> priorGroups empty -> "fresh"
    const turn1Groups = fingerprintGroups([sysUpdate, realPrompt])
    expect(turn1Groups).toHaveLength(1)
    expect(
      decideSync({
        recorded: undefined,
        priorGroups: turn1Groups.slice(0, -1),
        hasLiveOrResumableAcpSession: false,
      }),
    ).toEqual({ kind: "fresh" })

    // Thin prompt blocks forward both trailing user messages in order
    const thinBlocks = buildThinAcpPromptBlocks([sysUpdate, realPrompt])
    expect(thinBlocks).toEqual([
      { type: "text", text: "<system-update>\nYou are in Plan mode.\n</system-update>" },
      { type: "text", text: "hello" },
    ])

    // 2. Step 2 omits the ephemeral <system-update> from Turn 1 and appends a trailing <system-update>
    const recordedTurns = buildSavedHistory({
      priorGroups: [],
      currentUserMessages: [sysUpdate, realPrompt],
      assistantTextBlocks: ["hi"],
    })
    const step2Groups = fingerprintGroups([
      realPrompt,
      { role: "assistant", content: [{ type: "text", text: "hi" }] },
      sysUpdate,
    ])
    expect(
      decideSync({
        recorded: { version: 1, acpSessionId: "acp_plan_1", turns: recordedTurns },
        priorGroups: step2Groups.slice(0, -1),
        hasLiveOrResumableAcpSession: true,
      }),
    ).toEqual({ kind: "continue", matchedCount: 1 })
  })

  test("extracts mid-turn queued user messages and persists multi-group turn without causing divergence", () => {
    const u1: LanguageModelV3Prompt[number] = { role: "user", content: [{ type: "text", text: "is it progressing?" }] }
    const uMid: LanguageModelV3Prompt[number] = {
      role: "user",
      content: [{ type: "text", text: "Queued follow-up message 1" }],
    }
    const uNext: LanguageModelV3Prompt[number] = {
      role: "user",
      content: [{ type: "text", text: "Next turn message" }],
    }

    const startGroups = fingerprintGroups([u1])
    const midResumePrompt: LanguageModelV3Prompt = [
      u1,
      {
        role: "assistant",
        content: [{ type: "tool-call", toolCallId: "oc_1", toolName: "execute", input: {} }],
      },
      {
        role: "tool",
        content: [{ type: "tool-result", toolCallId: "oc_1", toolName: "execute", output: { type: "text", value: "ok" } }],
      },
      uMid,
    ]
    const midGroups = fingerprintGroups(midResumePrompt)
    const added = extractNewMidTurnUserMessages(startGroups, midGroups)
    expect(added).toHaveLength(1)
    const formatted = formatMidTurnUserMessages(added)
    expect(formatted).toContain("<queued-user-message>")
    expect(formatted).toContain("Queued follow-up message 1")

    // Final persistence uses midGroups + text streamed after the mid-turn resume
    const savedTurns = buildSavedHistoryFromGroups({
      groups: midGroups,
      trailingAssistantTextBlocks: ["Handled both progress check and the queued message."],
    })
    expect(savedTurns).toHaveLength(2)

    // Next turn arrives with full history -> decideSync MUST return "continue", not "diverged"
    const nextPrompt: LanguageModelV3Prompt = [
      ...midResumePrompt,
      { role: "assistant", content: [{ type: "text", text: "Handled both progress check and the queued message." }] },
      uNext,
    ]
    const nextGroups = fingerprintGroups(nextPrompt)
    expect(
      decideSync({
        recorded: { version: 1, acpSessionId: "acp_live", turns: savedTurns },
        priorGroups: nextGroups.slice(0, -1),
        hasLiveOrResumableAcpSession: true,
      }),
    ).toEqual({ kind: "continue", matchedCount: 2 })
  })
})
