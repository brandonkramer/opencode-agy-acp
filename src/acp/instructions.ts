import { createHash } from "node:crypto"
import { isRecord } from "../types.js"
import { stripUnsafeInternalLines } from "./history.js"

const MAX_SYSTEM_PROMPT_APPEND_CHARS = 120_000

// Antigravity supplies its own date and workspace environment; avoid duplicating OpenCode's copies.
const SUPPLIED_BY_AGY = new Set(["core/date", "core/environment"])

// Code Mode / OpenCode MCP guidance is only relevant when Code Mode MCP relay is enabled.
const CODE_MODE_ONLY_KEYS = new Set(["core/codemode", "core/mcp-guidance"])

const UNKEYED_CODE_MODE_RE = /^#\s+Code Mode\b/i

export type SystemPromptInputItem =
  | string
  | {
      readonly type?: string
      readonly text: string
      readonly metadata?: Record<string, unknown>
    }

/** Strip internal OpenCode XML namespace tags before forwarding to agy. */
export function stripInternalOpencodeNamespace(text: string): string {
  return text
    .replace(/<opencode:[^>]+>[\s\S]*?<\/opencode:[^>]+>/gi, "")
    .replace(/<opencode:[^/>]+\/>/gi, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
}

function extractInstructionChunks(
  item: SystemPromptInputItem,
  codeModeEnabled: boolean,
): {
  readonly hasInstructionsMeta: boolean
  readonly chunks: readonly string[]
} {
  if (typeof item === "string") {
    return { hasInstructionsMeta: false, chunks: [item] }
  }
  const meta = item.metadata
  const experimental = meta && isRecord(meta.experimental) ? meta.experimental : undefined
  const rawInstructions = experimental?.instructions
  if (Array.isArray(rawInstructions)) {
    const out: string[] = []
    for (const entry of rawInstructions) {
      if (!isRecord(entry) || typeof entry.text !== "string") continue
      const key = typeof entry.key === "string" ? entry.key : undefined
      if (key && SUPPLIED_BY_AGY.has(key)) continue
      if (!codeModeEnabled && key && CODE_MODE_ONLY_KEYS.has(key)) continue
      const text = entry.text.trim()
      if (text) out.push(text)
    }
    return { hasInstructionsMeta: true, chunks: out }
  }
  return {
    hasInstructionsMeta: false,
    chunks: typeof item.text === "string" ? [item.text] : [],
  }
}

/**
 * Extract user/agent/skill/Code Mode instructions from OpenCode's session context,
 * dropping OpenCode's base provider prompt (`system[0]`), built-in date/env sources,
 * and (in Pure ACP) Code Mode / `tools.opencode.*` instructions.
 */
export function extractOpenCodeInstructions(params: {
  readonly system: readonly SystemPromptInputItem[]
  readonly agentName?: string
  readonly agentSystem?: string
  readonly codeModeEnabled?: boolean
}): string | undefined {
  const codeModeEnabled = params.codeModeEnabled ?? false
  const sections: string[] = []
  const seen = new Set<string>()

  const cleanAgentSystem = params.agentSystem
    ? stripInternalOpencodeNamespace(params.agentSystem)
    : ""
  if (cleanAgentSystem) {
    seen.add(cleanAgentSystem)
    sections.push(`# Agent role (from OpenCode)\n${cleanAgentSystem}`)
  }

  for (let idx = 0; idx < params.system.length; idx++) {
    const item = params.system[idx]!
    const { hasInstructionsMeta, chunks } = extractInstructionChunks(item, codeModeEnabled)
    // OpenCode's system[0] (when not an instructions container) is its base provider prompt.
    if (!hasInstructionsMeta && idx === 0) continue
    for (const raw of chunks) {
      let cleaned = stripInternalOpencodeNamespace(raw)
      if (!codeModeEnabled) {
        if (UNKEYED_CODE_MODE_RE.test(cleaned)) continue
        cleaned = stripUnsafeInternalLines(cleaned)
      }
      if (!cleaned || seen.has(cleaned)) continue
      seen.add(cleaned)
      sections.push(cleaned)
    }
  }

  if (sections.length === 0) return undefined
  return sections.join("\n\n")
}

/** Build the `_meta.antigravity.systemPrompt.append` string when enabled. */
export function buildSystemPromptAppend(params: {
  readonly rawInstructions?: string
  readonly forwardInstructions: boolean
}): string | undefined {
  if (!params.forwardInstructions) return undefined
  const trimmed = params.rawInstructions?.trim()
  if (!trimmed) return undefined
  if (trimmed.length <= MAX_SYSTEM_PROMPT_APPEND_CHARS) return trimmed
  const suffix = "\n...[truncated]"
  return trimmed.slice(0, MAX_SYSTEM_PROMPT_APPEND_CHARS - suffix.length) + suffix
}

/** Compute a short deterministic digest for system prompt append text. */
export function hashSystemPrompt(text?: string): string {
  if (!text) return "none"
  return createHash("sha256").update(text).digest("hex").slice(0, 16)
}
