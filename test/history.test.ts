import { describe, expect, test } from "bun:test"
import type { LanguageModelV3Prompt } from "@ai-sdk/provider"
import { AssistantMessage, Media, UserMessage } from "@opencode/ai/effect"
import type { SessionRequestMessage } from "@opencode/plugin/effect/session"
import {
  buildTextOnlyHandoffPromptBlocks,
  buildThinAcpPromptBlocks,
  formatToolResultOutput,
  hasUnseenNonAcpAssistantHistory,
  isInlineBase64,
  isSyntheticRestartAfterError,
  sanitizeForAcpModel,
  sanitizeForNonAcpModel,
} from "../src/acp/history.js"
import { PROVIDER_ID } from "../src/types.js"

describe("history sanitizer & prompt builders", () => {
  test("isInlineBase64 allows JPEG /9j/ and rejects Unix file paths", () => {
    const jpegLike = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQg="
    expect(isInlineBase64(jpegLike)).toBe(true)
    expect(isInlineBase64("/workspace/assets/screenshots/abcdef123456")).toBe(false)
    expect(isInlineBase64("/Users/alice/photo.png")).toBe(false)
  })

  test("sanitizeForNonAcpModel converts ACP hosted tool calls/results to text", () => {
    const messages: SessionRequestMessage[] = [
      new AssistantMessage({
        role: "assistant",
        content: [
          {
            type: "tool-call",
            id: "c1",
            name: "acp_read_file",
            input: { path: "a.ts" },
            providerExecuted: true,
            providerMetadata: { [PROVIDER_ID]: { acp: true } },
          },
          {
            type: "tool-result",
            id: "c1",
            name: "acp_read_file",
            result: { type: "text", value: "hello" },
            providerExecuted: true,
            providerMetadata: { [PROVIDER_ID]: { acp: true } },
          },
        ],
      }),
    ]
    sanitizeForNonAcpModel(messages)
    const msg = messages[0]
    expect(msg.role).toBe("assistant")
    if (msg.role === "assistant") {
      expect(msg.content).toHaveLength(1)
      expect(msg.content[0].type).toBe("text")
    }
  })

  test("sanitizeForAcpModel rewrites provider media refs", () => {
    const messages: SessionRequestMessage[] = [
      new UserMessage({
        role: "user",
        content: [
          {
            type: "media",
            media: new Media.Asset({
              source: { type: "ref", provider: "openai" as never, id: "file_1" },
            }),
          },
        ],
      }),
    ]
    sanitizeForAcpModel(messages)
    const msg = messages[0]
    if (msg.role === "user") {
      expect(msg.content[0]).toEqual({
        type: "text",
        text: "[Provider media ref openai:file_1]",
      })
    }
  })

  test("hasUnseenNonAcpAssistantHistory ignores localStatus replies", () => {
    const prompt: LanguageModelV3Prompt = [
      { role: "user", content: [{ type: "text", text: "hi" }] },
      {
        role: "assistant",
        content: [{ type: "text", text: "cooldown" }],
        providerOptions: { [PROVIDER_ID]: { localStatus: true } },
      },
      { role: "user", content: [{ type: "text", text: "retry" }] },
    ]
    expect(hasUnseenNonAcpAssistantHistory(prompt)).toBe(false)

    const withForeign: LanguageModelV3Prompt = [
      ...prompt,
      { role: "assistant", content: [{ type: "text", text: "from another model" }] },
      { role: "user", content: [{ type: "text", text: "back to agy" }] },
    ]
    expect(hasUnseenNonAcpAssistantHistory(withForeign)).toBe(true)
  })

  test("buildThinAcpPromptBlocks & buildTextOnlyHandoffPromptBlocks respect budget", () => {
    const prompt: LanguageModelV3Prompt = [
      { role: "user", content: [{ type: "text", text: "first " + "x".repeat(200) }] },
      { role: "assistant", content: [{ type: "text", text: "reply " + "y".repeat(200) }] },
      { role: "user", content: [{ type: "text", text: "latest" }] },
    ]
    expect(buildThinAcpPromptBlocks(prompt)).toEqual([{ type: "text", text: "latest" }])
    const handoff = buildTextOnlyHandoffPromptBlocks(prompt, { maxChars: 260 })
    expect(handoff.length).toBe(2)
    expect(handoff[0].type === "text" && handoff[0].text.includes("[Historical assistant]")).toBe(
      true,
    )
    expect(handoff[1]).toEqual({ type: "text", text: "latest" })
  })

  test("isSyntheticRestartAfterError detects restart after halted error", () => {
    const prompt: LanguageModelV3Prompt = [
      { role: "user", content: [{ type: "text", text: "go" }] },
      { role: "assistant", content: [{ type: "text", text: "Antigravity ACP halted on quota" }] },
      { role: "user", content: [{ type: "text", text: "The server restarted while you were working." }] },
    ]
    expect(isSyntheticRestartAfterError(prompt)).toBe(true)
  })

  test("formatToolResultOutput preserves image-data and image file-data for MCP relay", () => {
    const imgRes = formatToolResultOutput({
      type: "content",
      value: [
        { type: "text", text: "captured screenshot" },
        { type: "image-data", data: "aGVsbG8=", mediaType: "image/png" },
        { type: "file-data", data: "d29ybGQ=", mediaType: "image/jpeg", filename: "shot.jpg" },
      ],
    })
    expect(imgRes.isError).toBe(false)
    expect(imgRes.text).toContain("captured screenshot")
    expect(imgRes.content).toEqual([
      { type: "text", text: "captured screenshot" },
      { type: "image", data: "aGVsbG8=", mimeType: "image/png" },
      { type: "image", data: "d29ybGQ=", mimeType: "image/jpeg" },
    ])
  })
})
