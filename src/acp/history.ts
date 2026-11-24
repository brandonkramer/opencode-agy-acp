import type { LanguageModelV3Prompt, LanguageModelV3ToolResultOutput } from "@ai-sdk/provider"
import { AssistantMessage, ToolMessage, UserMessage } from "@opencode/ai/effect"
import type { SessionRequestMessage } from "@opencode/plugin/effect/session"
import { PROVIDER_ID, isRecord, type AcpContentBlock, type QuotaBucket } from "../types.js"

export const UNSAFE_INTERNAL_SIGNATURE = /\bsession_move\b|tools\.opencode\./i

export const API_ERROR_ABORT_PATTERN =
  /API error \(attempt \d+\)|RESOURCE_EXHAUSTED|Individual quota reached|\(code 429\)|\(code 403\)|PERMISSION_DENIED/i

export const TEXT_ONLY_HANDOFF_HEADER =
  "Historical conversation context (plain text only; not a new request; native tool/reasoning state is not replayed):"

const HANDOFF_TRUNCATION_MARKER = "\n... [truncated]"

const KNOWN_PATH_PREFIXES = [
  "/tmp/",
  "/var/",
  "/home/",
  "/Users/",
  "/etc/",
  "/usr/",
  "/opt/",
  "./",
  "../",
  "~/",
] as const

export function quotaBucketForModel(modelID: string): QuotaBucket {
  const id = (modelID ?? "").toLowerCase()
  if (id.includes("claude") || id.includes("gpt")) return "claude-gpt"
  if (id.includes("gemini") || id.includes("orchestrator")) return "gemini"
  return "other"
}

export function isGlobalBreakerError(message: string): boolean {
  return /\(code 403\)|PERMISSION_DENIED/i.test(message)
}

function truncateText(text: string, maxLen = 6000): string {
  if (text.length <= maxLen) return text
  return `${text.slice(0, maxLen)}\n... [truncated ${text.length - maxLen} chars]`
}

function stringifyCompact(value: unknown, maxLen = 2000): string {
  if (typeof value === "string") return truncateText(value, maxLen)
  try {
    return truncateText(JSON.stringify(value), maxLen)
  } catch {
    return truncateText(String(value), maxLen)
  }
}

function summarizeToolResultValue(result: unknown): string {
  if (!isRecord(result)) return stringifyCompact(result)
  if (result.type === "content" && Array.isArray(result.value)) {
    return result.value
      .map((item) => {
        if (isRecord(item) && item.type === "text") {
          return String(item.text ?? "")
        }
        const mime = isRecord(item) ? String(item.mime ?? "") : ""
        const name = isRecord(item) ? String(item.name ?? "") : ""
        return `[file ${mime} ${name}]`
      })
      .join("\n")
  }
  if ("value" in result) return stringifyCompact(result.value, 6000)
  return stringifyCompact(result, 6000)
}

function mergeAdjacentTextParts<T extends { readonly type: string }>(parts: readonly T[]): T[] {
  const out: T[] = []
  for (const part of parts) {
    const prev = out[out.length - 1]
    if (
      part.type === "text" &&
      prev?.type === "text" &&
      !("providerMetadata" in part && (part as { providerMetadata?: unknown }).providerMetadata) &&
      !("providerMetadata" in prev && (prev as { providerMetadata?: unknown }).providerMetadata)
    ) {
      const prevText = (prev as { text?: string }).text ?? ""
      const nextText = (part as { text?: string }).text ?? ""
      out[out.length - 1] = {
        ...prev,
        text: `${prevText}\n${nextText}`,
      }
    } else {
      out.push(part)
    }
  }
  return out
}

export function sanitizeForNonAcpModel(messages: SessionRequestMessage[]): void {
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    if (msg.role === "system") continue

    let changed = false
    if (msg.role === "assistant") {
      type AssistantPart = AssistantMessage["content"][number]
      const nextContent: AssistantPart[] = []
      for (const part of msg.content) {
        if (part.type === "compaction") {
          changed = true
          nextContent.push({
            type: "text",
            text: part.text
              ? `[Prior conversation summary]\n${part.text}`
              : "[Prior conversation checkpoint]",
          })
          continue
        }
        if ((part as { readonly type: string }).type === "effort") {
          changed = true
          continue
        }
        const isAcpHosted =
          (part.type === "tool-call" || part.type === "tool-result") &&
          part.providerExecuted === true &&
          (Boolean(part.providerMetadata?.[PROVIDER_ID]) || part.name.startsWith("acp_"))
        if (isAcpHosted) {
          changed = true
          if (part.type === "tool-call") {
            nextContent.push({
              type: "text",
              text: `[Antigravity tool call \`${part.name}\`: ${stringifyCompact(part.input, 1200)}]`,
            })
          } else {
            nextContent.push({
              type: "text",
              text: `[Antigravity tool \`${part.name}\` result: ${summarizeToolResultValue(part.result)}]`,
            })
          }
          continue
        }
        nextContent.push(part)
      }
      if (changed) {
        const merged = mergeAdjacentTextParts(nextContent)
        messages[i] = new AssistantMessage({
          ...msg,
          content:
            merged.length > 0 ? merged : [{ type: "text", text: "[Antigravity turn completed]" }],
        })
      }
    } else if (msg.role === "user") {
      type UserPart = UserMessage["content"][number]
      const nextContent: UserPart[] = []
      for (const part of msg.content) {
        nextContent.push(part)
      }
      if (changed) {
        messages[i] = new UserMessage({ ...msg, content: nextContent })
      }
    } else if (msg.role === "tool") {
      type ToolPart = ToolMessage["content"][number]
      const nextContent: ToolPart[] = [...msg.content]
      if (changed) {
        messages[i] = new ToolMessage({ ...msg, content: nextContent })
      }
    }
  }
}

export function sanitizeForAcpModel(messages: SessionRequestMessage[]): void {
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    if (msg.role === "system") continue

    let changed = false
    if (msg.role === "assistant") {
      type AssistantPart = AssistantMessage["content"][number]
      const nextContent: AssistantPart[] = []
      for (const part of msg.content) {
        if (part.type === "compaction") {
          changed = true
          nextContent.push({
            type: "text",
            text: part.text
              ? `[Prior conversation summary]\n${part.text}`
              : "[Prior conversation checkpoint]",
          })
          continue
        }
        if ((part as { readonly type: string }).type === "effort") {
          changed = true
          continue
        }
        if (part.type === "media" && part.media.source.type === "ref") {
          changed = true
          nextContent.push({
            type: "text",
            text: `[Provider media ref ${part.media.source.provider}:${part.media.source.id}]`,
          })
          continue
        }
        nextContent.push(part)
      }
      if (changed) {
        const merged = mergeAdjacentTextParts(nextContent)
        messages[i] = new AssistantMessage({
          ...msg,
          content: merged.length > 0 ? merged : [{ type: "text", text: "[Prior assistant step]" }],
        })
      }
    } else if (msg.role === "user") {
      type UserPart = UserMessage["content"][number]
      const nextContent: UserPart[] = []
      for (const part of msg.content) {
        if (part.type === "media" && part.media.source.type === "ref") {
          changed = true
          nextContent.push({
            type: "text",
            text: `[Provider media ref ${part.media.source.provider}:${part.media.source.id}]`,
          })
          continue
        }
        nextContent.push(part)
      }
      if (changed) {
        const merged = mergeAdjacentTextParts(nextContent)
        messages[i] = new UserMessage({
          ...msg,
          content: merged.length > 0 ? merged : [{ type: "text", text: "[User message]" }],
        })
      }
    }
  }
}

export function formatToolResultOutput(output: LanguageModelV3ToolResultOutput): {
  readonly text: string
  readonly isError: boolean
} {
  switch (output.type) {
    case "text":
      return { text: output.value, isError: false }
    case "json":
      return { text: JSON.stringify(output.value), isError: false }
    case "error-text":
      return { text: output.value, isError: true }
    case "error-json":
      return { text: JSON.stringify(output.value), isError: true }
    case "execution-denied":
      return {
        text: output.reason ? `Tool execution denied: ${output.reason}` : "Tool execution denied",
        isError: true,
      }
    case "content": {
      const text = output.value
        .map((c) => {
          if (c.type === "text") return c.text
          if (c.type === "file-data" || c.type === "image-data") {
            const fname = "filename" in c ? c.filename : undefined
            return `[Attachment ${c.mediaType}${fname ? `: ${fname}` : ""}]`
          }
          if (c.type === "file-url" || c.type === "image-url") {
            return `[Attachment URL: ${c.url}]`
          }
          return "[Attachment]"
        })
        .join("\n")
      return { text, isError: false }
    }
  }
}

export function extractToolResultsFromPrompt(
  prompt: LanguageModelV3Prompt,
  expectedCallIds: ReadonlySet<string>,
): Map<string, { readonly text: string; readonly isError: boolean }> {
  const results = new Map<string, { readonly text: string; readonly isError: boolean }>()
  for (let i = prompt.length - 1; i >= 0; i--) {
    const msg = prompt[i]
    if (msg.role !== "tool" && msg.role !== "assistant") continue
    for (const part of msg.content) {
      if (part.type !== "tool-result") continue
      if (!expectedCallIds.has(part.toolCallId) || results.has(part.toolCallId)) continue
      results.set(part.toolCallId, formatToolResultOutput(part.output))
    }
    if (results.size >= expectedCallIds.size) break
  }
  return results
}

function readProviderMeta(holder: unknown): Record<string, unknown> | undefined {
  if (!isRecord(holder)) return undefined
  const meta = isRecord(holder.providerMetadata)
    ? holder.providerMetadata[PROVIDER_ID]
    : undefined
  if (isRecord(meta)) return meta
  const opts = isRecord(holder.providerOptions) ? holder.providerOptions[PROVIDER_ID] : undefined
  return isRecord(opts) ? opts : undefined
}

function hasAcpMarker(msg: LanguageModelV3Prompt[number]): boolean {
  if (msg.role !== "assistant") return false
  if (readProviderMeta(msg)) return true
  if (!Array.isArray(msg.content)) return false
  return msg.content.some((part) => readProviderMeta(part) !== undefined)
}

export function isLocalStatusMessage(msg: LanguageModelV3Prompt[number]): boolean {
  const check = (holder: unknown): boolean => {
    const meta = readProviderMeta(holder)
    return meta?.localStatus === true
  }
  if (check(msg)) return true
  if (Array.isArray(msg.content)) {
    return msg.content.some((part) => check(part))
  }
  return false
}

export function hasUnseenNonAcpAssistantHistory(prompt: LanguageModelV3Prompt): boolean {
  const nonSystem = prompt.filter((m) => m.role !== "system")
  if (nonSystem.length === 0) return false
  let lastAcpIdx = -1
  for (let i = nonSystem.length - 1; i >= 0; i--) {
    const msg = nonSystem[i]
    if (isLocalStatusMessage(msg)) continue
    if (hasAcpMarker(msg)) {
      lastAcpIdx = i
      break
    }
  }
  const unseen = nonSystem.slice(lastAcpIdx + 1)
  for (const msg of unseen) {
    if (isLocalStatusMessage(msg)) continue
    if (msg.role !== "assistant" || !Array.isArray(msg.content)) continue
    if (msg.content.length === 0) continue
    if (!hasAcpMarker(msg)) return true
  }
  return false
}

export function isInlineBase64(data: string): boolean {
  if (data.length < 32 || data.length % 4 !== 0) return false
  if (/\s/.test(data)) return false
  if (data.includes("://")) return false
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(data)) return false
  if (data.includes("\\")) return false
  for (const prefix of KNOWN_PATH_PREFIXES) {
    if (data.startsWith(prefix)) return false
  }
  if (data.startsWith("/") && !data.startsWith("/9j/")) {
    const rest = data.slice(1)
    if (rest.includes("/")) {
      const segments = data.split("/").filter(Boolean)
      if (
        segments.length >= 2 &&
        segments.every((seg) => /^[A-Za-z0-9_.\-]+$/.test(seg)) &&
        !/[+=]/.test(data)
      ) {
        return false
      }
    }
  }
  return /^[A-Za-z0-9+/]*={0,2}$/.test(data)
}

function fileDataToBase64(data: Uint8Array | string | URL): string | undefined {
  if (data instanceof Uint8Array) {
    return Buffer.from(data).toString("base64")
  }
  if (typeof data === "string") {
    if (data.startsWith("data:")) {
      const match = /^data:[^;]+;base64,(.*)$/s.exec(data)
      if (!match) return undefined
      const payload = match[1]
      if (!payload || payload.length === 0 || /\s/.test(payload) || payload.length % 4 !== 0) {
        return undefined
      }
      if (!/^[A-Za-z0-9+/]*={0,2}$/.test(payload)) return undefined
      return payload
    }
    if (isInlineBase64(data)) return data
  }
  return undefined
}

export function appendCurrentUserBlocks(
  currentUserMsg: LanguageModelV3Prompt[number] | undefined,
  blocks: AcpContentBlock[],
): void {
  if (!currentUserMsg || currentUserMsg.role !== "user") return
  for (const part of currentUserMsg.content) {
    if (part.type === "text") {
      if (part.text.length > 0) {
        blocks.push({ type: "text", text: part.text })
      }
    } else if (part.type === "file") {
      const b64 = fileDataToBase64(part.data)
      if (part.mediaType.startsWith("image/") && b64) {
        blocks.push({ type: "image", data: b64, mimeType: part.mediaType })
      } else {
        blocks.push({
          type: "text",
          text: `[Attached file: ${part.filename ?? "file"} (${part.mediaType})]`,
        })
      }
    }
  }
}

export function buildThinAcpPromptBlocks(prompt: LanguageModelV3Prompt): AcpContentBlock[] {
  const nonSystem = prompt.filter((m) => m.role !== "system" && !isLocalStatusMessage(m))
  if (nonSystem.length === 0) {
    return [{ type: "text", text: "Continue." }]
  }
  let startIdx = nonSystem.length
  while (startIdx > 0 && nonSystem[startIdx - 1]!.role === "user") {
    startIdx--
  }
  const trailingUsers =
    startIdx < nonSystem.length
      ? nonSystem.slice(startIdx)
      : (() => {
          let lastUserIdx = nonSystem.length - 1
          while (lastUserIdx >= 0 && nonSystem[lastUserIdx]!.role !== "user") {
            lastUserIdx--
          }
          return lastUserIdx >= 0 ? [nonSystem[lastUserIdx]!] : []
        })()
  const blocks: AcpContentBlock[] = []
  for (const userMsg of trailingUsers) {
    appendCurrentUserBlocks(userMsg, blocks)
  }
  if (blocks.length === 0) {
    blocks.push({ type: "text", text: "Continue." })
  }
  return blocks
}

function extractHistoryText(msg: LanguageModelV3Prompt[number]): string {
  if (!Array.isArray(msg.content)) return ""
  const chunks: string[] = []
  for (const part of msg.content) {
    if (part.type === "text" && typeof part.text === "string" && part.text.trim()) {
      chunks.push(part.text.trim())
    }
  }
  return chunks.join("\n").trim()
}

export function buildTextOnlyHandoffPromptBlocks(
  prompt: LanguageModelV3Prompt,
  options?: { readonly maxChars?: number },
): AcpContentBlock[] {
  const maxChars = options?.maxChars ?? 16_000
  const nonSystem = prompt.filter((m) => m.role !== "system")
  if (nonSystem.length === 0) {
    return [{ type: "text", text: "Continue." }]
  }
  let lastUserIdx = nonSystem.length - 1
  while (lastUserIdx >= 0 && nonSystem[lastUserIdx].role !== "user") {
    lastUserIdx--
  }
  const currentUserMsg = lastUserIdx >= 0 ? nonSystem[lastUserIdx] : undefined
  const priorMsgs = lastUserIdx > 0 ? nonSystem.slice(0, lastUserIdx) : []

  const lines: string[] = []
  for (const msg of priorMsgs) {
    if (msg.role !== "user" && msg.role !== "assistant") continue
    if (isLocalStatusMessage(msg)) continue
    const text = extractHistoryText(msg)
    if (!text) continue
    lines.push(msg.role === "user" ? `[Historical user]: ${text}` : `[Historical assistant]: ${text}`)
  }

  const budget = Math.max(0, maxChars - TEXT_ONLY_HANDOFF_HEADER.length)
  let total = lines.reduce((sum, l) => sum + l.length + 2, 0)
  while (lines.length > 1 && total > budget) {
    const dropped = lines.shift()!
    total -= dropped.length + 2
  }
  if (lines.length === 1 && lines[0].length + 2 > budget) {
    if (budget <= HANDOFF_TRUNCATION_MARKER.length + 2) {
      lines.length = 0
    } else {
      lines[0] = `${lines[0].slice(0, budget - HANDOFF_TRUNCATION_MARKER.length - 2)}${HANDOFF_TRUNCATION_MARKER}`
    }
  }

  const blocks: AcpContentBlock[] = []
  if (lines.length > 0) {
    blocks.push({
      type: "text",
      text: `${TEXT_ONLY_HANDOFF_HEADER}\n${lines.join("\n\n")}\n`,
    })
  }
  appendCurrentUserBlocks(currentUserMsg, blocks)
  if (blocks.length === 0) {
    blocks.push({ type: "text", text: "Continue." })
  }
  return blocks
}

export function stripUnsafeInternalLines(text: string): string {
  return text
    .split(/\r?\n/)
    .filter((line) => !UNSAFE_INTERNAL_SIGNATURE.test(line))
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
}

export function isSyntheticRestartAfterError(prompt: LanguageModelV3Prompt): boolean {
  const nonSystem = prompt.filter((m) => m.role !== "system")
  if (nonSystem.length === 0) return false
  const last = nonSystem[nonSystem.length - 1]
  if (last.role !== "user") return false

  const lastText = last.content
    .filter((p): p is { type: "text"; text: string } => p.type === "text" && typeof p.text === "string")
    .map((p) => p.text)
    .join("\n")

  if (!lastText.includes("The server restarted while you were working.")) return false

  for (let i = nonSystem.length - 2; i >= 0; i--) {
    const msg = nonSystem[i]
    if (msg.role !== "assistant") continue
    const assistantText = msg.content
      .filter((p): p is { type: "text"; text: string } => p.type === "text" && typeof p.text === "string")
      .map((p) => p.text)
      .join("\n")
    return (
      assistantText.includes("Antigravity ACP halted") ||
      API_ERROR_ABORT_PATTERN.test(assistantText)
    )
  }

  return false
}

export function summarizeTitleFromMessages(messages: readonly SessionRequestMessage[]): string | undefined {
  for (const msg of messages) {
    if (msg.role !== "user") continue
    const text = msg.content
      .filter((p) => p.type === "text")
      .map((p) => p.text)
      .join(" ")
      .replace(/\s+/g, " ")
      .trim()
    if (!text) continue
    return text.length > 60 ? `${text.slice(0, 57)}...` : text
  }
  return undefined
}
