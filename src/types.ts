export const PROVIDER_ID = "antigravity-acp"
export const PACKAGE_ID = "aisdk:antigravity-acp"
export const OPENCODE_QUESTION_TOOL = "question"

export const NATIVE_SHADOWED_TOOLS = new Set<string>([
  "read",
  "write",
  "edit",
  "multiedit",
  "apply_patch",
  "bash",
  "shell",
  "grep",
  "glob",
  "ls",
  "list",
  OPENCODE_QUESTION_TOOL,
])

export type AcpMode = "auto" | "accept-edits" | "default" | "plan"

export type RequestKind = "primary" | "compaction" | "generate"

export type CodeModeSetting = boolean | "auto"

export type PrewarmSetting = boolean | "auto"

export type AcpAuthMethodId =
  | "oauth-personal"
  | "oauth-business"
  | "gemini-api-key"
  | "agent-platform"

export type QuotaBucket = "claude-gpt" | "gemini" | "other"
export type CooldownBucket = QuotaBucket | "global"

export interface AgySettings {
  readonly binaryPath?: string
  readonly defaultMode: AcpMode
  readonly forwardInstructions: boolean
  readonly codeMode: CodeModeSetting
  readonly prewarm: PrewarmSetting
  readonly smoothStreaming: boolean
  readonly maxSystemPromptChars: number
  readonly maxMcpResultChars: number
  readonly extraRelayedTools: readonly string[]
  readonly blockedRelayedTools: readonly string[]
  readonly includeDefaultCustomizations: boolean
  readonly isolateUserMcp: boolean
  readonly startupTimeoutMs: number
  readonly chunkTimeoutMs: number
  readonly nativeToolTimeoutMs: number
  readonly settleTimeoutMs: number
  readonly permissionTimeoutMs: number
  readonly nativeFlushGraceMs: number
  readonly idleTimeoutMs: number
  readonly traceLogPath?: string
  readonly extraDirs: readonly string[]
}

export const DEFAULT_SETTINGS: AgySettings = {
  defaultMode: "auto",
  forwardInstructions: true,
  codeMode: "auto",
  prewarm: "auto",
  smoothStreaming: true,
  maxSystemPromptChars: 120_000,
  maxMcpResultChars: 9_800,
  extraRelayedTools: [],
  blockedRelayedTools: [],
  includeDefaultCustomizations: false,
  isolateUserMcp: true,
  startupTimeoutMs: 30_000,
  chunkTimeoutMs: 120_000,
  nativeToolTimeoutMs: 15 * 60_000,
  settleTimeoutMs: 3_000,
  permissionTimeoutMs: 300_000,
  nativeFlushGraceMs: 400,
  idleTimeoutMs: 20 * 60_000,
  extraDirs: [],
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}

function readPositiveInt(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return Math.floor(value)
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value.trim())
    if (Number.isFinite(parsed) && parsed > 0) return Math.floor(parsed)
  }
  return undefined
}

function readNonNegativeInt(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
    return Math.floor(value)
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value.trim())
    if (Number.isFinite(parsed) && parsed >= 0) return Math.floor(parsed)
  }
  return undefined
}

function readBool(value: unknown): boolean | undefined {
  if (typeof value === "boolean") return value
  if (typeof value === "string") {
    const norm = value.trim().toLowerCase()
    if (norm === "1" || norm === "true" || norm === "yes") return true
    if (norm === "0" || norm === "false" || norm === "no") return false
  }
  return undefined
}

function readAcpMode(value: unknown): AcpMode | undefined {
  if (typeof value !== "string") return undefined
  const norm = value.trim().toLowerCase()
  if (norm === "auto" || norm === "accept-edits" || norm === "default" || norm === "plan") {
    return norm
  }
  return undefined
}

function readCodeMode(value: unknown): CodeModeSetting | undefined {
  if (typeof value === "boolean") return value
  if (typeof value === "string") {
    const norm = value.trim().toLowerCase()
    if (norm === "auto") return "auto"
    const bool = readBool(norm)
    if (bool !== undefined) return bool
  }
  return undefined
}

function readPrewarm(value: unknown): PrewarmSetting | undefined {
  if (typeof value === "boolean") return value
  if (typeof value === "string") {
    const norm = value.trim().toLowerCase()
    if (norm === "auto") return "auto"
    const bool = readBool(norm)
    if (bool !== undefined) return bool
  }
  return undefined
}

function readStringArray(value: unknown): readonly string[] | undefined {
  if (!Array.isArray(value)) return undefined
  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
}

export function resolveSettings(
  providerOptions?: Record<string, unknown>,
  env: NodeJS.ProcessEnv = process.env,
  base: AgySettings = DEFAULT_SETTINGS,
): AgySettings {
  const rawSettings = isRecord(providerOptions?.settings) ? providerOptions.settings : providerOptions ?? {}
  const binaryPath =
    (typeof rawSettings.binaryPath === "string" && rawSettings.binaryPath.trim()) ||
    (typeof env.AGY_BIN === "string" && env.AGY_BIN.trim()) ||
    base.binaryPath
  const traceLogPath =
    (typeof rawSettings.traceLogPath === "string" && rawSettings.traceLogPath.trim()) ||
    (typeof env.AGY_ACP_TRACE_LOG === "string" && env.AGY_ACP_TRACE_LOG.trim()) ||
    base.traceLogPath
  const extraDirs = readStringArray(rawSettings.extraDirs) ?? base.extraDirs
  const extraRelayedTools = readStringArray(rawSettings.extraRelayedTools) ?? base.extraRelayedTools
  const blockedRelayedTools =
    readStringArray(rawSettings.blockedRelayedTools) ?? base.blockedRelayedTools

  return {
    binaryPath,
    defaultMode: readAcpMode(rawSettings.defaultMode) ?? readAcpMode(env.AGY_ACP_DEFAULT_MODE) ?? base.defaultMode,
    forwardInstructions:
      readBool(rawSettings.forwardInstructions) ??
      readBool(env.AGY_ACP_FORWARD_INSTRUCTIONS) ??
      base.forwardInstructions,
    codeMode: readCodeMode(rawSettings.codeMode) ?? readCodeMode(env.AGY_ACP_CODE_MODE) ?? base.codeMode,
    prewarm: readPrewarm(rawSettings.prewarm) ?? readPrewarm(env.AGY_ACP_PREWARM) ?? base.prewarm,
    smoothStreaming:
      readBool(rawSettings.smoothStreaming) ??
      readBool(env.AGY_ACP_SMOOTH_STREAMING) ??
      base.smoothStreaming,
    maxSystemPromptChars:
      readPositiveInt(rawSettings.maxSystemPromptChars) ??
      readPositiveInt(env.AGY_ACP_MAX_SYSTEM_PROMPT_CHARS) ??
      base.maxSystemPromptChars,
    maxMcpResultChars:
      readNonNegativeInt(rawSettings.maxMcpResultChars) ??
      readNonNegativeInt(env.AGY_ACP_MAX_MCP_RESULT_CHARS) ??
      base.maxMcpResultChars,
    extraRelayedTools,
    blockedRelayedTools,
    includeDefaultCustomizations:
      readBool(rawSettings.includeDefaultCustomizations) ??
      readBool(env.AGY_ACP_INCLUDE_DEFAULT_CUSTOMIZATIONS) ??
      base.includeDefaultCustomizations,
    isolateUserMcp:
      readBool(rawSettings.isolateUserMcp) ??
      readBool(env.AGY_ACP_ISOLATE_USER_MCP) ??
      base.isolateUserMcp,
    startupTimeoutMs:
      readPositiveInt(rawSettings.startupTimeoutMs) ??
      readPositiveInt(env.AGY_ACP_STARTUP_TIMEOUT_MS) ??
      base.startupTimeoutMs,
    chunkTimeoutMs:
      readPositiveInt(rawSettings.chunkTimeoutMs) ??
      readPositiveInt(env.AGY_ACP_CHUNK_TIMEOUT_MS) ??
      base.chunkTimeoutMs,
    nativeToolTimeoutMs:
      readPositiveInt(rawSettings.nativeToolTimeoutMs) ??
      readPositiveInt(env.AGY_ACP_NATIVE_TOOL_TIMEOUT_MS) ??
      base.nativeToolTimeoutMs,
    settleTimeoutMs:
      readPositiveInt(rawSettings.settleTimeoutMs) ??
      readPositiveInt(env.AGY_ACP_SETTLE_TIMEOUT_MS) ??
      base.settleTimeoutMs,
    permissionTimeoutMs:
      readPositiveInt(rawSettings.permissionTimeoutMs) ??
      readPositiveInt(env.AGY_ACP_PERMISSION_TIMEOUT_MS) ??
      base.permissionTimeoutMs,
    nativeFlushGraceMs:
      readPositiveInt(rawSettings.nativeFlushGraceMs) ??
      readPositiveInt(env.AGY_ACP_NATIVE_FLUSH_GRACE_MS) ??
      base.nativeFlushGraceMs,
    idleTimeoutMs:
      readPositiveInt(rawSettings.idleTimeoutMs) ??
      readPositiveInt(env.AGY_ACP_IDLE_TIMEOUT_MS) ??
      base.idleTimeoutMs,
    traceLogPath,
    extraDirs,
  }
}

export interface WireTraceEntry {
  readonly ts: string
  readonly dir: "out" | "in" | "err" | "sys"
  readonly summary: string
}

export class AgyNotFoundError extends Error {
  constructor(binary: string) {
    super(
      `Antigravity CLI ("${binary}") was not found. Install Antigravity CLI, sign in with "agy", and ensure "agy" is in PATH (or set AGY_BIN=/path/to/agy).`,
    )
    this.name = "AgyNotFoundError"
  }
}

export class AcpRpcError extends Error {
  readonly acpCode?: number
  readonly acpData?: unknown

  constructor(message: string, acpCode?: number, acpData?: unknown) {
    super(message)
    this.name = "AcpRpcError"
    this.acpCode = acpCode
    this.acpData = acpData
  }
}

export class AcpStallError extends Error {
  readonly idleMs: number
  readonly recentTrace: readonly WireTraceEntry[]

  constructor(idleMs: number, recentTrace: readonly WireTraceEntry[]) {
    const secs = Math.round(idleMs / 1000)
    const tail = recentTrace
      .slice(-8)
      .map((entry) => `  [${entry.ts}] ${entry.dir}: ${entry.summary}`)
      .join("\n")
    super(
      `Antigravity ACP produced no updates for ${secs}s and was cancelled.${tail ? `\nRecent ACP wire trace:\n${tail}` : ""}`,
    )
    this.name = "AcpStallError"
    this.idleMs = idleMs
    this.recentTrace = recentTrace
  }
}

export interface AcpModelVariant {
  readonly id: string
  readonly acpModelId: string
  readonly thoughtLevel?: string
}

export interface AcpDiscoveredModel {
  readonly id: string
  readonly name: string
  readonly family?: string
  readonly variants: readonly AcpModelVariant[]
  readonly defaultAcpModelId: string
}

export interface AcpMcpServerHttp {
  readonly type: "http"
  readonly name: string
  readonly url: string
  readonly headers?: ReadonlyArray<{ readonly name: string; readonly value: string }>
}

export interface AcpPermissionOption {
  readonly optionId: string
  readonly name?: string
  readonly kind?: string
}

export interface AcpPermissionRequestParams {
  readonly sessionId?: string
  readonly toolCall?: {
    readonly toolCallId?: string
    readonly title?: string
    readonly name?: string
    readonly kind?: string
    readonly rawInput?: unknown
    readonly locations?: ReadonlyArray<{ readonly path?: string; readonly line?: number }>
  }
  readonly options?: ReadonlyArray<AcpPermissionOption>
}

export type AcpPermissionOutcome =
  | { readonly outcome: "selected"; readonly optionId: string }
  | { readonly outcome: "cancelled" }

export interface AcpElicitationRequestParams {
  readonly sessionId?: string
  readonly mode?: string
  readonly message?: string
  readonly requestedSchema?: unknown
  readonly toolCallId?: string
  readonly [key: string]: unknown
}

export type AcpElicitationResponse =
  | { readonly action: "accept"; readonly content: Record<string, string | string[]> }
  | { readonly action: "decline" }
  | { readonly action: "cancel" }

export type AcpContentBlock =
  | { readonly type: "text"; readonly text: string }
  | { readonly type: "image"; readonly data: string; readonly mimeType: string }

export interface AcpSessionUpdate {
  readonly sessionUpdate: string
  readonly content?: { readonly type?: string; readonly text?: string } | ReadonlyArray<unknown>
  readonly toolCallId?: string
  readonly name?: string
  readonly title?: string
  readonly kind?: string
  readonly status?: string
  readonly args?: unknown
  readonly rawInput?: unknown
  readonly rawOutput?: unknown
  readonly locations?: unknown
  readonly used?: number
  readonly size?: number
  readonly [key: string]: unknown
}

export interface AcpUsagePayload {
  readonly inputTokens?: number
  readonly outputTokens?: number
  readonly cachedReadTokens?: number
  readonly cachedWriteTokens?: number
  readonly thoughtTokens?: number
  readonly totalTokens?: number
  readonly input_tokens?: number
  readonly output_tokens?: number
  readonly cached_read_tokens?: number
  readonly cached_write_tokens?: number
  readonly thought_tokens?: number
  readonly total_tokens?: number
}

export interface AcpPromptResult {
  readonly stopReason?: string
  readonly usage?: AcpUsagePayload
}

export interface AcpSessionResult {
  readonly sessionId?: string
  readonly models?: {
    readonly availableModels?: ReadonlyArray<{ readonly modelId: string; readonly name?: string }>
  }
  readonly configOptions?: ReadonlyArray<{
    readonly id?: string
    readonly options?: ReadonlyArray<{ readonly value?: unknown; readonly name?: unknown }>
  }>
}

export interface SessionPublicState {
  readonly sessionID: string
  readonly mode: AcpMode
  readonly codeModeEnabled: boolean
  readonly starting: boolean
  readonly acpSessionId: string | null
  readonly connected: boolean
  readonly busy: boolean
  readonly modelCount: number
  readonly relayedToolCount: number
  readonly lastError: string | null
  readonly cooldownUntil: number | null
}

export interface SessionDiagnostics {
  readonly sessionID: string
  readonly acpSessionId: string | null
  readonly mode: AcpMode
  readonly codeModeEnabled: boolean
  readonly connected: boolean
  readonly busy: boolean
  readonly watchdogPaused: boolean
  readonly activeNativeTools: readonly string[]
  readonly pendingMcpCalls: number
  readonly lastError: string | null
  readonly recentTrace: readonly WireTraceEntry[]
}
