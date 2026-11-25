import { createHash } from "node:crypto"
import { isRecord } from "../types.js"
import { stripUnsafeInternalLines } from "./history.js"

const DEFAULT_SYSTEM_PROMPT_APPEND_CHARS = 120_000

// Antigravity supplies its own date and workspace environment; avoid duplicating OpenCode's copies.
const SUPPLIED_BY_AGY = new Set(["core/date", "core/environment"])

// Code Mode / OpenCode MCP guidance is only relevant when Code Mode MCP relay is enabled.
const CODE_MODE_ONLY_KEYS = new Set(["core/codemode", "core/mcp-guidance"])

const UNKEYED_CODE_MODE_RE = /^#\s+Code Mode\b/i

const CODE_MODE_EXECUTE_REF_RE =
  /\b(inside(?:\s+the)?|using|use|call(?:ing)?|via|the)(\s+)`execute`/gi

const FULL_CODE_MODE_DESC_RE = /\bBun\.(?:file|write|Glob)\b/

export type SystemPromptInputItem =
  | string
  | {
      readonly type?: string
      readonly text: string
      readonly metadata?: Record<string, unknown>
    }

/** Detect whether OpenCode's experimental `full_codemode` is active on its `execute` tool. */
export function isFullCodeModeExecuteDescription(description?: string): boolean {
  return typeof description === "string" && FULL_CODE_MODE_DESC_RE.test(description)
}

/** Strip internal OpenCode XML namespace tags before forwarding to agy. */
export function stripInternalOpencodeNamespace(text: string): string {
  return text
    .replace(/<opencode:[^>]+>[\s\S]*?<\/opencode:[^>]+>/gi, "")
    .replace(/<opencode:[^/>]+\/>/gi, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
}

/** Rewrite OpenCode's `execute` tool references to `mcp_oc_execute` for Antigravity ACP. */
export function rewriteCodeModeExecuteRefs(text: string): string {
  return text.replace(CODE_MODE_EXECUTE_REF_RE, "$1$2`mcp_oc_execute`")
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
  readonly fullCodeMode?: boolean
  readonly relayedToolNames?: readonly string[]
}): string | undefined {
  const codeModeEnabled = params.codeModeEnabled ?? false
  const hasExecuteRelay = codeModeEnabled && Boolean(params.relayedToolNames?.includes("execute"))
  const sections: string[] = []
  const seen = new Set<string>()

  if (codeModeEnabled && params.relayedToolNames && params.relayedToolNames.length > 0) {
    const lines = ["# OpenCode Relayed Tools (MCP `oc`)"]
    if (hasExecuteRelay) {
      if (params.fullCodeMode) {
        lines.push(
          "- **Full Code Mode (`mcp_oc_execute`):** OpenCode Full Code Mode is ACTIVE. Always invoke `mcp_oc_execute` (never bare `execute`).",
          "- **Default to `mcp_oc_execute` over native tools:** Prefer `mcp_oc_execute` FIRST instead of `read_file`, `write_file`, `edit_file`, `grep_search`, `find_by_name`, or `run_command` whenever creating, reading, searching, or editing files, running shell commands, calling OpenCode MCP servers, or calling `tools.opencode.*`.",
          '- **Batch & use Bun APIs:** Combine independent reads/writes/edits (`Bun.file`, `Bun.write`, `Bun.Glob` — prefer these over `tools.shell` for `ls`/`cat`/`head`/`tail`/`find`/`grep`), `fetch`, `tools.shell`, and MCP/OpenCode tools (`await search({ query: "..." })` / `await tools.<ns>.<tool>(...)`) into ONE script with `Promise.all` instead of sequential single-call turns.',
          "- **Shell & background tasks via `tools.shell`:** Run shell commands—including background watchers, long-running commands, and CLI helpers that expect OpenCode session environment variables (`OPENCODE_SESSION_ID`, etc.)—through `mcp_oc_execute` (`tools.shell`) rather than native `run_command`, so they inherit OpenCode's session environment and do not hold the ACP turn open.",
          "- **Keep return payloads compact (< 4-6 KB) — parse/project in-script:** Do NOT return raw `.output` JSON strings from CLI/JSON commands (e.g. `gh`, `git`, MCP tools) (they get double-JSON-encoded and blow past output caps, forcing extra read-back turns). Parse and slice in JavaScript (or use `gh --jq` / temp file + `Bun.file().json()`) and return only the small structured fields you need. Never `JSON.parse()` `.system_generated/steps/*/output.txt`.",
        )
      } else {
        lines.push(
          "- **Code Mode (`mcp_oc_execute`):** OpenCode's Code Mode JavaScript runtime is callable as `mcp_oc_execute` (never bare `execute`).",
          "- **When to prefer `mcp_oc_execute`:** Use `mcp_oc_execute` FIRST instead of `run_command` or sequential calls whenever you need OpenCode MCP servers, `tools.opencode.*` (sessions, worktrees, automations, models, MCP resources), batching multiple tool calls with `Promise.all`, or filtering/aggregating large tool outputs in JavaScript before returning a concise result (< 4-6 KB).",
          '- Inside `mcp_oc_execute`, discover tools with `await search({ query: "..." })` and call them via their returned `path` (e.g. `await tools.<namespace>.<tool>({ ... })`). Keep single local file reads/edits/shell on native tools unless scripting a batch, and never `JSON.parse()` truncated `.system_generated/steps/*/output.txt` or large `tools.shell` stdout directly.',
        )
      }
    }
    lines.push(
      `- Relayed OpenCode tools exposed with the \`mcp_oc_\` prefix: ${params.relayedToolNames
        .map((n) => `\`mcp_oc_${n}\``)
        .join(", ")}`,
    )
    sections.push(lines.join("\n"))
  }

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
      } else if (hasExecuteRelay) {
        cleaned = rewriteCodeModeExecuteRefs(cleaned)
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
  readonly maxChars?: number
}): string | undefined {
  if (!params.forwardInstructions) return undefined
  const trimmed = params.rawInstructions?.trim()
  if (!trimmed) return undefined
  const maxChars = params.maxChars ?? DEFAULT_SYSTEM_PROMPT_APPEND_CHARS
  if (trimmed.length <= maxChars) return trimmed
  const suffix = "\n...[truncated]"
  if (maxChars <= suffix.length) return trimmed.slice(0, Math.max(0, maxChars))
  return trimmed.slice(0, maxChars - suffix.length) + suffix
}

/** Compute a short deterministic digest for system prompt append text. */
export function hashSystemPrompt(text?: string): string {
  if (!text) return "none"
  return createHash("sha256").update(text).digest("hex").slice(0, 16)
}
