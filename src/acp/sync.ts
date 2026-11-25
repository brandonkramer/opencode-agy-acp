import { createHash } from "node:crypto"
import type { LanguageModelV3Prompt } from "@ai-sdk/provider"
import { PROVIDER_ID, isRecord, type AcpContentBlock } from "../types.js"
import { appendCurrentUserBlocks, formatToolResultOutput, isLocalStatusMessage } from "./history.js"

const DEFAULT_CHECKPOINT_MAX_CHARS = 24_000
const CHECKPOINT_TRUNCATION_MARKER = "\n... [earlier turns truncated]"
const SYSTEM_UPDATE_BLOCK_RE = /<system-update>[\s\S]*?<\/system-update>/gi

export interface PersistedTurn {
  readonly userHash: string
  readonly assistantHash: string
  readonly userPreview?: string
  readonly assistantPreview?: string
}

export interface PersistedSessionRecord {
  readonly version: 1
  readonly acpSessionId?: string
  readonly turns: readonly PersistedTurn[]
  readonly lastModelId?: string
  readonly lastSystemPromptHash?: string
}

export interface PromptGroup {
  readonly userMessages: readonly LanguageModelV3Prompt[number][]
  readonly trailingMessages: readonly LanguageModelV3Prompt[number][]
  readonly userHash: string
  readonly assistantHash: string
  readonly userPreview: string
  readonly assistantPreview: string
  readonly assistantTranscript: string
  readonly assistantTextChunks: readonly string[]
}

export type SyncDecision =
  | { readonly kind: "fresh" }
  | { readonly kind: "fresh-with-history"; readonly priorGroups: readonly PromptGroup[] }
  | { readonly kind: "continue"; readonly matchedCount: number }
  | {
      readonly kind: "catch-up"
      readonly unseenGroups: readonly PromptGroup[]
      readonly matchedCount: number
    }
  | {
      readonly kind: "diverged"
      readonly priorGroups: readonly PromptGroup[]
      readonly matchedCount: number
    }

function sha256Short(input: string): string {
  return createHash("sha256").update(input).digest("hex").slice(0, 24)
}

function previewText(text: string, maxLen = 120): string {
  const oneLine = text.replace(/\s+/g, " ").trim()
  if (oneLine.length <= maxLen) return oneLine
  return `${oneLine.slice(0, maxLen - 3)}...`
}

function hashBinaryData(data: Uint8Array | string | URL): string {
  if (data instanceof Uint8Array) {
    return createHash("sha256").update(data).digest("hex").slice(0, 16)
  }
  if (data instanceof URL) {
    return sha256Short(`url:${data.href}`).slice(0, 16)
  }
  return sha256Short(`str:${data.length}:${data.slice(0, 128)}:${data.slice(-64)}`).slice(0, 16)
}

function stripEphemeralSystemUpdates(text: string): string {
  return text.replace(SYSTEM_UPDATE_BLOCK_RE, "").trim()
}

function isAcpHostedPart(part: unknown): boolean {
  if (!isRecord(part)) return false
  if (part.providerExecuted === true) return true
  if (typeof part.toolName === "string" && part.toolName.startsWith("acp_")) return true
  if (isRecord(part.providerMetadata) && PROVIDER_ID in part.providerMetadata) return true
  if (isRecord(part.providerOptions) && PROVIDER_ID in part.providerOptions) return true
  return false
}

export function normalizeAssistantBlocks(chunks: readonly string[]): string {
  return chunks
    .map((chunk) => chunk.trim())
    .filter((chunk) => chunk.length > 0)
    .join("\n")
}

export function hashAssistantTextBlocks(chunks: readonly string[]): string {
  const normalized = normalizeAssistantBlocks(chunks)
  if (!normalized) return "empty"
  return sha256Short(`assistant:${normalized}`)
}

export function hashUserMessages(messages: readonly LanguageModelV3Prompt[number][]): {
  readonly hash: string
  readonly preview: string
  readonly transcript: string
} {
  const strippedCanonical: unknown[] = []
  const rawCanonical: unknown[] = []
  const transcriptLines: string[] = []
  const strippedLines: string[] = []

  for (const msg of messages) {
    if (msg.role !== "user" || !Array.isArray(msg.content)) continue
    for (const part of msg.content) {
      if (part.type === "text") {
        const rawTrimmed = part.text.trim()
        if (!rawTrimmed) continue
        rawCanonical.push({ type: "text", text: rawTrimmed })
        transcriptLines.push(rawTrimmed)
        const stripped = stripEphemeralSystemUpdates(rawTrimmed)
        if (stripped) {
          strippedCanonical.push({ type: "text", text: stripped })
          strippedLines.push(stripped)
        }
      } else if (part.type === "file") {
        const entry = {
          type: "file",
          mediaType: part.mediaType,
          filename: part.filename ?? "",
          sig: hashBinaryData(part.data),
        }
        rawCanonical.push(entry)
        strippedCanonical.push(entry)
        const line = `[Attached file: ${part.filename ?? "file"} (${part.mediaType})]`
        transcriptLines.push(line)
        strippedLines.push(line)
      }
    }
  }

  const transcript = transcriptLines.join("\n").trim()
  const chosenCanonical = strippedCanonical.length > 0 ? strippedCanonical : rawCanonical
  const previewSource = strippedLines.length > 0 ? strippedLines.join("\n").trim() : transcript
  const hash = chosenCanonical.length === 0 ? "empty" : sha256Short(JSON.stringify(chosenCanonical))
  return { hash, preview: previewText(previewSource), transcript }
}

export function hashUserMessage(msg: LanguageModelV3Prompt[number]): {
  readonly hash: string
  readonly preview: string
  readonly transcript: string
} {
  return hashUserMessages([msg])
}

function extractTrailingAssistantInfo(
  trailingMessages: readonly LanguageModelV3Prompt[number][],
): {
  readonly hash: string
  readonly preview: string
  readonly transcript: string
  readonly textChunks: readonly string[]
} {
  const textChunks: string[] = []
  const transcriptLines: string[] = []
  for (const msg of trailingMessages) {
    if (isLocalStatusMessage(msg) || !Array.isArray(msg.content)) continue
    if (msg.role === "assistant") {
      for (const part of msg.content) {
        if (part.type === "text") {
          const trimmed = part.text.trim()
          if (!trimmed) continue
          textChunks.push(trimmed)
          transcriptLines.push(trimmed)
        } else if (part.type === "tool-call" && !isAcpHostedPart(part)) {
          transcriptLines.push(`[Tool call: ${part.toolName}]`)
        } else if (part.type === "tool-result" && !isAcpHostedPart(part)) {
          const formatted = formatToolResultOutput(part.output)
          if (formatted.text.trim()) {
            transcriptLines.push(`[Tool ${part.toolName} result]: ${formatted.text.trim()}`)
          }
        }
      }
    } else if (msg.role === "tool") {
      for (const part of msg.content) {
        if (part.type !== "tool-result" || isAcpHostedPart(part)) continue
        const formatted = formatToolResultOutput(part.output)
        if (formatted.text.trim()) {
          transcriptLines.push(`[Tool ${part.toolName} result]: ${formatted.text.trim()}`)
        }
      }
    }
  }
  const hash = hashAssistantTextBlocks(textChunks)
  const transcript = transcriptLines.join("\n").trim()
  return { hash, preview: previewText(transcript), transcript, textChunks }
}

/** Group a prompt into (consecutive userMessages, trailing assistant/tool messages) turns with hashes. */
export function fingerprintGroups(prompt: LanguageModelV3Prompt): PromptGroup[] {
  const groups: Array<{
    userMessages: LanguageModelV3Prompt[number][]
    trailingMessages: LanguageModelV3Prompt[number][]
  }> = []

  for (const msg of prompt) {
    if (msg.role === "system") continue
    if (isLocalStatusMessage(msg)) continue
    if (msg.role === "user") {
      const prev = groups[groups.length - 1]
      if (prev && prev.trailingMessages.length === 0) {
        prev.userMessages.push(msg)
      } else {
        groups.push({ userMessages: [msg], trailingMessages: [] })
      }
    } else if (groups.length > 0) {
      groups[groups.length - 1]!.trailingMessages.push(msg)
    }
  }

  return groups.map((group) => {
    const userInfo = hashUserMessages(group.userMessages)
    const assistantInfo = extractTrailingAssistantInfo(group.trailingMessages)
    return {
      userMessages: group.userMessages,
      trailingMessages: group.trailingMessages,
      userHash: userInfo.hash,
      assistantHash: assistantInfo.hash,
      userPreview: userInfo.preview,
      assistantPreview: assistantInfo.preview,
      assistantTranscript: assistantInfo.transcript,
      assistantTextChunks: assistantInfo.textChunks,
    }
  })
}

export function parsePersistedRecord(raw: unknown): PersistedSessionRecord | undefined {
  if (typeof raw === "string") {
    const trimmed = raw.trim()
    if (!trimmed) return undefined
    if (!trimmed.startsWith("{")) {
      return {
        version: 1,
        acpSessionId: trimmed,
        turns: [],
      }
    }
    try {
      return parsePersistedRecord(JSON.parse(trimmed) as unknown)
    } catch {
      return undefined
    }
  }
  if (!isRecord(raw) || raw.version !== 1) return undefined
  const turns: PersistedTurn[] = []
  if (Array.isArray(raw.turns)) {
    for (const item of raw.turns) {
      if (
        isRecord(item) &&
        typeof item.userHash === "string" &&
        typeof item.assistantHash === "string"
      ) {
        turns.push({
          userHash: item.userHash,
          assistantHash: item.assistantHash,
          ...(typeof item.userPreview === "string" ? { userPreview: item.userPreview } : {}),
          ...(typeof item.assistantPreview === "string"
            ? { assistantPreview: item.assistantPreview }
            : {}),
        })
      }
    }
  }
  return {
    version: 1,
    ...(typeof raw.acpSessionId === "string" && raw.acpSessionId.trim()
      ? { acpSessionId: raw.acpSessionId.trim() }
      : {}),
    turns,
    ...(typeof raw.lastModelId === "string" && raw.lastModelId.trim()
      ? { lastModelId: raw.lastModelId.trim() }
      : {}),
    ...(typeof raw.lastSystemPromptHash === "string" && raw.lastSystemPromptHash.trim()
      ? { lastSystemPromptHash: raw.lastSystemPromptHash.trim() }
      : {}),
  }
}

export function decideSync(params: {
  readonly recorded?: PersistedSessionRecord
  readonly priorGroups: readonly PromptGroup[]
  readonly hasLiveOrResumableAcpSession: boolean
}): SyncDecision {
  const { recorded, priorGroups, hasLiveOrResumableAcpSession } = params
  if (!hasLiveOrResumableAcpSession) {
    return priorGroups.length > 0
      ? { kind: "fresh-with-history", priorGroups }
      : { kind: "fresh" }
  }

  const recordedTurns = recorded?.turns ?? []
  const maxPrefix = Math.min(recordedTurns.length, priorGroups.length)
  let matched = 0
  while (matched < maxPrefix) {
    const rec = recordedTurns[matched]
    const cur = priorGroups[matched]
    if (rec.userHash !== cur.userHash || rec.assistantHash !== cur.assistantHash) {
      break
    }
    matched++
  }

  if (matched < recordedTurns.length) {
    return {
      kind: "diverged",
      priorGroups,
      matchedCount: matched,
    }
  }

  if (priorGroups.length > recordedTurns.length) {
    return {
      kind: "catch-up",
      unseenGroups: priorGroups.slice(matched),
      matchedCount: matched,
    }
  }

  return { kind: "continue", matchedCount: matched }
}

export function buildSavedHistory(params: {
  readonly priorGroups: readonly PromptGroup[]
  readonly currentUserMessages?: readonly LanguageModelV3Prompt[number][]
  readonly assistantTextBlocks: readonly string[]
}): PersistedTurn[] {
  const turns: PersistedTurn[] = params.priorGroups.map((g) => ({
    userHash: g.userHash,
    assistantHash: g.assistantHash,
    ...(g.userPreview ? { userPreview: g.userPreview } : {}),
    ...(g.assistantPreview ? { assistantPreview: g.assistantPreview } : {}),
  }))
  if (params.currentUserMessages && params.currentUserMessages.length > 0) {
    const userInfo = hashUserMessages(params.currentUserMessages)
    const assistantNormalized = normalizeAssistantBlocks(params.assistantTextBlocks)
    turns.push({
      userHash: userInfo.hash,
      assistantHash: hashAssistantTextBlocks(params.assistantTextBlocks),
      ...(userInfo.preview ? { userPreview: userInfo.preview } : {}),
      ...(assistantNormalized ? { assistantPreview: previewText(assistantNormalized) } : {}),
    })
  }
  return turns
}

/**
 * Extract newly appended user/synthetic messages that arrived in `nextGroups`
 * while an active ACP turn was parked waiting for a relayed OpenCode tool.
 */
export function extractNewMidTurnUserMessages(
  prevGroups: readonly PromptGroup[],
  nextGroups: readonly PromptGroup[],
): LanguageModelV3Prompt[number][] {
  if (nextGroups.length === 0) return []
  if (nextGroups.length > prevGroups.length) {
    const added: LanguageModelV3Prompt[number][] = []
    for (let i = prevGroups.length; i < nextGroups.length; i++) {
      added.push(...nextGroups[i]!.userMessages)
    }
    return added
  }
  if (nextGroups.length === prevGroups.length) {
    const prevLast = prevGroups[prevGroups.length - 1]
    const nextLast = nextGroups[nextGroups.length - 1]
    if (prevLast && nextLast && nextLast.userMessages.length > prevLast.userMessages.length) {
      return nextLast.userMessages.slice(prevLast.userMessages.length)
    }
  }
  return []
}

/** Format mid-turn queued user/synthetic messages for in-band delivery via MCP tool result. */
export function formatMidTurnUserMessages(
  messages: readonly LanguageModelV3Prompt[number][],
): string | undefined {
  if (messages.length === 0) return undefined
  const transcript = hashUserMessages(messages).transcript.trim()
  if (!transcript) return undefined
  return [
    "<queued-user-message>",
    "The following message(s) arrived while your tool was running. Handle them now alongside your current work:",
    transcript,
    "</queued-user-message>",
  ].join("\n")
}

/**
 * Persist turn fingerprints directly from the turn's latest prompt groups (which
 * include any mid-turn user/synthetic messages) plus assistant text streamed after
 * the last parked resumption, preventing false `diverged` session resets.
 */
export function buildSavedHistoryFromGroups(params: {
  readonly groups: readonly PromptGroup[]
  readonly trailingAssistantTextBlocks: readonly string[]
}): PersistedTurn[] {
  if (params.groups.length === 0) return []
  const turns: PersistedTurn[] = []
  for (let i = 0; i < params.groups.length - 1; i++) {
    const g = params.groups[i]!
    turns.push({
      userHash: g.userHash,
      assistantHash: g.assistantHash,
      ...(g.userPreview ? { userPreview: g.userPreview } : {}),
      ...(g.assistantPreview ? { assistantPreview: g.assistantPreview } : {}),
    })
  }
  const last = params.groups[params.groups.length - 1]!
  const combinedChunks = [...last.assistantTextChunks, ...params.trailingAssistantTextBlocks]
  const assistantNormalized = normalizeAssistantBlocks(combinedChunks)
  turns.push({
    userHash: last.userHash,
    assistantHash: hashAssistantTextBlocks(combinedChunks),
    ...(last.userPreview ? { userPreview: last.userPreview } : {}),
    ...(assistantNormalized
      ? { assistantPreview: previewText(assistantNormalized) }
      : last.assistantPreview
        ? { assistantPreview: last.assistantPreview }
        : {}),
  })
  return turns
}

export function buildCheckpointPromptBlocks(params: {
  readonly groups: readonly PromptGroup[]
  readonly currentUserMessages?: readonly LanguageModelV3Prompt[number][]
  readonly mode: "catch-up" | "restore"
  readonly maxChars?: number
}): AcpContentBlock[] {
  const maxChars = params.maxChars ?? DEFAULT_CHECKPOINT_MAX_CHARS
  const formattedTurns: string[] = []
  for (const group of params.groups) {
    const uText = hashUserMessages(group.userMessages).transcript || "[User message]"
    const lines = [`User: ${uText}`]
    if (group.assistantTranscript) {
      lines.push(`Assistant: ${group.assistantTranscript}`)
    }
    formattedTurns.push(lines.join("\n"))
  }

  const header =
    params.mode === "catch-up"
      ? `<conversation-checkpoint mode="catch-up">\nThe following turns happened on another model in this OpenCode session since your last Antigravity turn (for context only, not a new request):\n`
      : `<conversation-checkpoint mode="restore">\nThe following prior conversation turns are replayed to restore session context (not a new request):\n`
  const footer = `\n</conversation-checkpoint>\n`
  const budget = Math.max(0, maxChars - header.length - footer.length)

  let total = formattedTurns.reduce((sum, entry) => sum + entry.length + 2, 0)
  let truncated = false
  while (formattedTurns.length > 1 && total > budget) {
    const dropped = formattedTurns.shift()!
    total -= dropped.length + 2
    truncated = true
  }
  if (formattedTurns.length === 1 && formattedTurns[0].length > budget) {
    if (budget <= CHECKPOINT_TRUNCATION_MARKER.length + 4) {
      formattedTurns.length = 0
    } else {
      formattedTurns[0] =
        formattedTurns[0].slice(0, budget - CHECKPOINT_TRUNCATION_MARKER.length) +
        CHECKPOINT_TRUNCATION_MARKER
      truncated = true
    }
  }

  const blocks: AcpContentBlock[] = []
  if (formattedTurns.length > 0) {
    const prefix = truncated ? `[Earlier turns omitted to fit budget]\n\n` : ""
    blocks.push({
      type: "text",
      text: `${header}${prefix}${formattedTurns.join("\n\n")}${footer}`,
    })
  }

  for (const userMsg of params.currentUserMessages ?? []) {
    appendCurrentUserBlocks(userMsg, blocks)
  }
  if (blocks.length === 0) {
    blocks.push({ type: "text", text: "Continue." })
  }
  return blocks
}

export function buildSideCallPromptBlocks(
  prompt: LanguageModelV3Prompt,
  kind: "compaction" | "generate",
): AcpContentBlock[] {
  const blocks: AcpContentBlock[] = []
  const systemLines: string[] = []
  const conversationLines: string[] = []

  for (const msg of prompt) {
    if (msg.role === "system") {
      const content = typeof msg.content === "string" ? msg.content.trim() : ""
      if (content) systemLines.push(content)
    }
  }

  const nonSystem = prompt.filter((m) => m.role !== "system" && !isLocalStatusMessage(m))
  let trailUserStart = nonSystem.length
  while (trailUserStart > 0 && nonSystem[trailUserStart - 1]!.role === "user") {
    trailUserStart--
  }
  const hasTrailingUsers = trailUserStart < nonSystem.length

  for (let i = 0; i < nonSystem.length; i++) {
    if (hasTrailingUsers && i >= trailUserStart) continue
    const msg = nonSystem[i]!
    if (msg.role === "user") {
      const text = hashUserMessage(msg).transcript
      if (text) conversationLines.push(`User: ${text}`)
    } else if (msg.role === "assistant" && Array.isArray(msg.content)) {
      const parts: string[] = []
      for (const part of msg.content) {
        if (part.type === "text" && part.text.trim()) {
          parts.push(part.text.trim())
        } else if (part.type === "tool-call") {
          parts.push(`[Tool call: ${part.toolName}]`)
        } else if (part.type === "tool-result") {
          const formatted = formatToolResultOutput(part.output)
          if (formatted.text.trim()) {
            parts.push(`[Tool ${part.toolName} result]: ${formatted.text.trim()}`)
          }
        }
      }
      if (parts.length > 0) conversationLines.push(`Assistant: ${parts.join("\n")}`)
    } else if (msg.role === "tool" && Array.isArray(msg.content)) {
      for (const part of msg.content) {
        if (part.type !== "tool-result") continue
        const formatted = formatToolResultOutput(part.output)
        if (formatted.text.trim()) {
          conversationLines.push(`[Tool ${part.toolName} result]: ${formatted.text.trim()}`)
        }
      }
    }
  }

  const introParts: string[] = []
  if (systemLines.length > 0) {
    introParts.push(`<system-instructions>\n${systemLines.join("\n\n")}\n</system-instructions>`)
  }
  if (conversationLines.length > 0) {
    const tag = kind === "compaction" ? "compaction-context" : "side-call-context"
    introParts.push(`<${tag}>\n${conversationLines.join("\n\n")}\n</${tag}>`)
  }
  if (introParts.length > 0) {
    blocks.push({ type: "text", text: introParts.join("\n\n") })
  }
  if (hasTrailingUsers) {
    for (const userMsg of nonSystem.slice(trailUserStart)) {
      appendCurrentUserBlocks(userMsg, blocks)
    }
  }
  if (blocks.length === 0) {
    blocks.push({ type: "text", text: "Continue." })
  }
  return blocks
}
