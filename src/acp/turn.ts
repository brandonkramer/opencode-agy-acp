import type {
  JSONValue,
  LanguageModelV3FinishReason,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
} from "@ai-sdk/provider"
import {
  OPENCODE_QUESTION_TOOL,
  PROVIDER_ID,
  isRecord,
  type AcpPromptResult,
  type AcpSessionUpdate,
  type AcpUsagePayload,
} from "../types.js"
import { API_ERROR_ABORT_PATTERN } from "./history.js"
import type { McpCallToolResult, McpContentItem } from "./mcp-relay.js"
import type { OpenCodeQuestion } from "./question.js"

const TERMINAL_TOOL_STATUSES = new Set([
  "completed",
  "done",
  "succeeded",
  "failed",
  "error",
  "cancelled",
  "canceled",
  "denied",
  "rejected",
])

const ERROR_TOOL_STATUSES = new Set(["failed", "error", "cancelled", "canceled", "denied", "rejected"])

const SMOOTH_MIN_CHUNK_CHARS = 32
const SMOOTH_TEXT_TARGET_TICKS = 14
const SMOOTH_TEXT_MIN_SLICE = 12
const SMOOTH_TEXT_SOFT_MAX_SLICE = 28
const SMOOTH_TEXT_SOFT_BACKLOG = 450
const SMOOTH_REASONING_TARGET_TICKS = 24
const SMOOTH_REASONING_MIN_SLICE = 14
const SMOOTH_REASONING_SOFT_MAX_SLICE = 28
const SMOOTH_REASONING_SOFT_BACKLOG = 800
const DEFAULT_SMOOTH_INTERVAL_MS = 14
const MAX_TOOL_RESULT_CHARS = 16_000
const AGY_TOOL_HEADER_RE = /^Created At:[^\n]*\n(?:Completed At:[^\n]*\n)?/i
const AGY_CMD_OK_WRAPPER_RE = /^Command exited with code 0\.\n+(?:Command output:\n)?/i

function computeSmoothSliceChars(kind: "text" | "reasoning", pendingLen: number): number {
  if (kind === "reasoning") {
    const scaled = Math.ceil(pendingLen / SMOOTH_REASONING_TARGET_TICKS)
    if (pendingLen <= SMOOTH_REASONING_SOFT_BACKLOG) {
      return Math.max(SMOOTH_REASONING_MIN_SLICE, Math.min(SMOOTH_REASONING_SOFT_MAX_SLICE, scaled))
    }
    return Math.max(SMOOTH_REASONING_MIN_SLICE, scaled)
  }
  const scaled = Math.ceil(pendingLen / SMOOTH_TEXT_TARGET_TICKS)
  if (pendingLen <= SMOOTH_TEXT_SOFT_BACKLOG) {
    return Math.max(SMOOTH_TEXT_MIN_SLICE, Math.min(SMOOTH_TEXT_SOFT_MAX_SLICE, scaled))
  }
  return Math.max(SMOOTH_TEXT_MIN_SLICE, scaled)
}

function cleanToolOutputString(raw: string, exitCode?: number): string {
  let cleaned = raw.replace(/\r\n/g, "\n").replace(/\r/g, "\n")
  cleaned = cleaned.replace(AGY_TOOL_HEADER_RE, "")
  if (exitCode === undefined || exitCode === 0) {
    const stripped = cleaned.replace(AGY_CMD_OK_WRAPPER_RE, "")
    if (stripped !== cleaned) {
      cleaned = stripped.trim().length > 0 ? stripped.replace(/\n+$/, "") : "(no output)"
    }
  } else if (!/exit(?:ed)?(?:\s+with)?\s+code/i.test(cleaned)) {
    cleaned = `[exit ${exitCode}]\n${cleaned}`
  }
  if (cleaned.length > MAX_TOOL_RESULT_CHARS) {
    return `${cleaned.slice(0, MAX_TOOL_RESULT_CHARS)}\n... [truncated ${cleaned.length - MAX_TOOL_RESULT_CHARS} chars]`
  }
  return cleaned
}

/** Split a bursty text chunk at a nearby word boundary without breaking UTF-16 surrogate pairs. */
export function takeSmoothSlice(text: string, targetChars: number): readonly [string, string] {
  if (text.length <= targetChars) return [text, ""]

  let cut = Math.min(text.length, Math.max(1, targetChars))
  const fwdLimit = Math.min(text.length, cut + 12)
  let foundBoundary = false
  for (let i = cut; i < fwdLimit; i++) {
    const ch = text.charCodeAt(i)
    if (ch === 32 || ch === 10 || ch === 9) {
      cut = i + 1
      foundBoundary = true
      break
    }
  }
  if (!foundBoundary) {
    const backLimit = Math.max(4, cut - 8)
    for (let i = cut - 1; i >= backLimit; i--) {
      const ch = text.charCodeAt(i)
      if (ch === 32 || ch === 10 || ch === 9) {
        cut = i + 1
        break
      }
    }
  }

  if (cut < text.length) {
    const prevCode = text.charCodeAt(cut - 1)
    if (prevCode >= 0xd800 && prevCode <= 0xdbff) {
      cut = Math.min(text.length, cut + 1)
    }
  }
  if (text.length - cut <= 6) {
    cut = text.length
  }

  return [text.slice(0, cut), text.slice(cut)]
}

export function buildUsage(
  raw?: AcpUsagePayload,
  activeContextUsed?: number,
): LanguageModelV3Usage {
  const r = raw as (AcpUsagePayload & Record<string, unknown>) | undefined
  const asNum = (v: unknown): number | undefined =>
    typeof v === "number" && Number.isFinite(v) ? v : undefined

  const rawInput = asNum(r?.inputTokens) ?? asNum(r?.input_tokens)
  const rawOutput = asNum(r?.outputTokens) ?? asNum(r?.output_tokens)
  const rawTotal = asNum(r?.totalTokens) ?? asNum(r?.total_tokens)
  const rawCacheRead = asNum(r?.cachedReadTokens) ?? asNum(r?.cached_read_tokens)
  const rawCacheWrite = asNum(r?.cachedWriteTokens) ?? asNum(r?.cached_write_tokens)
  const reasoning = asNum(r?.thoughtTokens) ?? asNum(r?.thought_tokens)

  const derivedInputFromTotal =
    rawTotal !== undefined && rawOutput !== undefined
      ? Math.max(0, rawTotal - rawOutput)
      : undefined
  const fallbackContext =
    typeof activeContextUsed === "number" &&
    Number.isFinite(activeContextUsed) &&
    activeContextUsed > 0
      ? activeContextUsed
      : undefined
  const cachedSum = (rawCacheRead ?? 0) + (rawCacheWrite ?? 0)

  let inputTotal: number | undefined
  let cacheRead: number | undefined
  let cacheWrite: number | undefined
  let noCache: number | undefined

  if (rawInput !== undefined && cachedSum > 0) {
    // Antigravity ACP reports `inputTokens` exclusively of `cachedReadTokens` / `cachedWriteTokens`
    // (e.g. inputTokens + cachedReadTokens + outputTokens === totalTokens). Detect exclusive vs
    // inclusive payloads so cache-hit turns preserve both non-cached input and full cacheRead.
    const exclusiveTotal = rawInput + cachedSum
    let isExclusive = false
    if (derivedInputFromTotal !== undefined) {
      isExclusive =
        Math.abs(exclusiveTotal - derivedInputFromTotal) <=
        Math.abs(rawInput - derivedInputFromTotal)
    } else if (cachedSum > rawInput) {
      isExclusive = true
    } else if (fallbackContext !== undefined) {
      isExclusive =
        Math.abs(exclusiveTotal - fallbackContext) < Math.abs(rawInput - fallbackContext)
    }

    if (isExclusive) {
      noCache = rawInput
      cacheRead = rawCacheRead
      cacheWrite = rawCacheWrite
      inputTotal = exclusiveTotal
    } else {
      inputTotal = rawInput
      cacheRead = rawCacheRead !== undefined ? Math.min(rawCacheRead, inputTotal) : undefined
      cacheWrite =
        rawCacheWrite !== undefined
          ? Math.min(rawCacheWrite, Math.max(0, inputTotal - (cacheRead ?? 0)))
          : undefined
      noCache = Math.max(0, inputTotal - (cacheRead ?? 0) - (cacheWrite ?? 0))
    }
  } else {
    inputTotal = rawInput ?? derivedInputFromTotal ?? fallbackContext
    cacheRead =
      rawCacheRead !== undefined && inputTotal !== undefined
        ? Math.min(rawCacheRead, inputTotal)
        : rawCacheRead
    cacheWrite =
      rawCacheWrite !== undefined && inputTotal !== undefined
        ? Math.min(rawCacheWrite, Math.max(0, inputTotal - (cacheRead ?? 0)))
        : rawCacheWrite
    noCache =
      inputTotal !== undefined
        ? Math.max(0, inputTotal - (cacheRead ?? 0) - (cacheWrite ?? 0))
        : undefined
  }

  const outputTotal = rawOutput
  const text =
    outputTotal !== undefined ? Math.max(0, outputTotal - (reasoning ?? 0)) : undefined

  return {
    inputTokens: {
      total: inputTotal,
      noCache,
      cacheRead,
      cacheWrite,
    },
    outputTokens: {
      total: outputTotal,
      text,
      reasoning,
    },
  }
}

/** Extract and normalize a native ACP tool result into compact, TUI-friendly output. */
export function formatNativeToolResult(update: AcpSessionUpdate): NonNullable<JSONValue> {
  const rawOutput = update.rawOutput
  if (rawOutput !== undefined && rawOutput !== null) {
    if (typeof rawOutput === "string") {
      return cleanToolOutputString(rawOutput)
    }
    if (isRecord(rawOutput)) {
      if (typeof rawOutput.output === "string") {
        const exitCode = typeof rawOutput.exit_code === "number" ? rawOutput.exit_code : undefined
        return cleanToolOutputString(rawOutput.output, exitCode)
      }
      if (typeof rawOutput.content === "string") {
        return cleanToolOutputString(rawOutput.content)
      }
      if (typeof rawOutput.text === "string") {
        return cleanToolOutputString(rawOutput.text)
      }
      try {
        const serialized = JSON.stringify(rawOutput)
        if (serialized.length > MAX_TOOL_RESULT_CHARS) {
          return `${serialized.slice(0, MAX_TOOL_RESULT_CHARS)}\n... [truncated ${serialized.length - MAX_TOOL_RESULT_CHARS} chars]`
        }
      } catch {}
    }
    return rawOutput as NonNullable<JSONValue>
  }
  if (Array.isArray(update.content)) {
    const texts: string[] = []
    for (const block of update.content) {
      if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
        texts.push(block.text)
      } else if (
        isRecord(block) &&
        isRecord(block.content) &&
        block.content.type === "text" &&
        typeof block.content.text === "string"
      ) {
        texts.push(block.content.text)
      } else {
        texts.push(JSON.stringify(block))
      }
    }
    if (texts.length > 0) return cleanToolOutputString(texts.join("\n"))
  }
  return {
    status: update.status ?? "completed",
    ...(update.title !== undefined ? { title: update.title } : {}),
  }
}

interface PendingMcpCall {
  readonly toolCallId: string
  readonly openCodeToolName: string
  readonly args: unknown
  readonly nativeToolCallId?: string
  readonly bypassNativeTools?: boolean
  emittedToOpenCode: boolean
  resolved: boolean
  readonly resolve: (res: McpCallToolResult) => void
  abortCleanup?: () => void
}

export interface TurnCallbacks {
  readonly pauseWatchdog: () => void
  readonly resumeWatchdog: () => void
  readonly onApiErrorText: (text: string) => void
  readonly onInterruptRequested: (reason: string) => void
  readonly onTurnSettled: () => void
  readonly getKnownRelayedNames: () => ReadonlySet<string>
  readonly nativeFlushGraceMs: number
  readonly smoothStreaming?: boolean
  readonly smoothIntervalMs?: number
}

export class ActiveTurn {
  readonly activeNativeTools = new Map<string, string>()
  readonly pendingMcpCalls = new Map<string, PendingMcpCall>()
  readonly dispatchedCallIds = new Set<string>()
  private readonly nativeToolText = new Map<string, string>()
  private readonly relayedUpdateIds = new Set<string>()
  private readonly bufferedParts: LanguageModelV3StreamPart[] = []
  private controller?: ReadableStreamDefaultController<LanguageModelV3StreamPart>
  private controllerCleanup?: () => void
  private openTextId?: string
  private openReasoningId?: string
  private partSeq = 0
  private flushTimer?: ReturnType<typeof setTimeout>
  private graceTimer?: ReturnType<typeof setTimeout>
  private pacedKind?: "text" | "reasoning"
  private pacedBlockId?: string
  private pacedRemainder = ""
  private pacedSliceChars = SMOOTH_TEXT_MIN_SLICE
  private pacedTimer?: ReturnType<typeof setTimeout>
  private pausedForPending = false
  private contextUsed?: number
  private readonly completedTextBlocks: string[] = []
  private currentTextBuffer = ""
  promptSettled = false
  abortedOnApiError = false
  promptResult?: AcpPromptResult
  promptError?: Error

  constructor(private readonly callbacks: TurnCallbacks) {}

  get hasController(): boolean {
    return this.controller !== undefined
  }

  get activeNativeToolNames(): readonly string[] {
    return [...this.activeNativeTools.values()]
  }

  get assistantTextBlocks(): readonly string[] {
    return this.currentTextBuffer.trim().length > 0
      ? [...this.completedTextBlocks, this.currentTextBuffer]
      : [...this.completedTextBlocks]
  }

  private syncWatchdogPause(): void {
    const shouldPause = this.pendingMcpCalls.size > 0
    if (shouldPause && !this.pausedForPending) {
      this.pausedForPending = true
      this.callbacks.pauseWatchdog()
    } else if (!shouldPause && this.pausedForPending) {
      this.pausedForPending = false
      this.callbacks.resumeWatchdog()
    }
  }

  private clearPacedTimer(): void {
    if (this.pacedTimer) {
      clearTimeout(this.pacedTimer)
      this.pacedTimer = undefined
    }
  }

  private flushPacedImmediate(): void {
    this.clearPacedTimer()
    const delta = this.pacedRemainder
    const kind = this.pacedKind
    const id = this.pacedBlockId
    this.pacedRemainder = ""
    this.pacedKind = undefined
    this.pacedBlockId = undefined
    if (delta.length > 0 && kind && id) {
      this.emitPart({
        type: kind === "reasoning" ? "reasoning-delta" : "text-delta",
        id,
        delta,
      })
    }
  }

  private scheduleNextPacedTick(): void {
    this.clearPacedTimer()
    const intervalMs = this.callbacks.smoothIntervalMs ?? DEFAULT_SMOOTH_INTERVAL_MS
    this.pacedTimer = setTimeout(() => {
      this.pacedTimer = undefined
      if (!this.pacedRemainder || !this.pacedKind || !this.pacedBlockId) {
        this.pacedRemainder = ""
        this.pacedKind = undefined
        this.pacedBlockId = undefined
        return
      }
      if (!this.controller) {
        this.flushPacedImmediate()
        return
      }
      const [head, tail] = takeSmoothSlice(this.pacedRemainder, this.pacedSliceChars)
      const kind = this.pacedKind
      const id = this.pacedBlockId
      this.pacedRemainder = tail
      if (tail.length === 0) {
        this.pacedKind = undefined
        this.pacedBlockId = undefined
      }
      this.emitPart({
        type: kind === "reasoning" ? "reasoning-delta" : "text-delta",
        id,
        delta: head,
      })
      if (tail.length > 0) {
        this.scheduleNextPacedTick()
      }
    }, intervalMs)
    this.pacedTimer.unref?.()
  }

  private emitPacedDelta(kind: "text" | "reasoning", id: string, text: string): void {
    // Continuous same-block append: queue onto active pacer without causing a mid-stream jump.
    if (
      this.callbacks.smoothStreaming &&
      this.controller &&
      this.pacedKind === kind &&
      this.pacedBlockId === id &&
      this.pacedRemainder.length > 0
    ) {
      this.pacedRemainder += text
      this.pacedSliceChars = computeSmoothSliceChars(kind, this.pacedRemainder.length)
      if (!this.pacedTimer) {
        this.scheduleNextPacedTick()
      }
      return
    }

    this.flushPacedImmediate()
    if (!this.callbacks.smoothStreaming || !this.controller || text.length <= SMOOTH_MIN_CHUNK_CHARS) {
      this.emitPart({
        type: kind === "reasoning" ? "reasoning-delta" : "text-delta",
        id,
        delta: text,
      })
      return
    }

    const sliceChars = computeSmoothSliceChars(kind, text.length)
    const [head, tail] = takeSmoothSlice(text, sliceChars)
    this.emitPart({
      type: kind === "reasoning" ? "reasoning-delta" : "text-delta",
      id,
      delta: head,
    })
    if (tail.length > 0) {
      this.pacedKind = kind
      this.pacedBlockId = id
      this.pacedRemainder = tail
      this.pacedSliceChars = sliceChars
      this.scheduleNextPacedTick()
    }
  }

  private clearTimers(): void {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer)
      this.flushTimer = undefined
    }
    if (this.graceTimer) {
      clearTimeout(this.graceTimer)
      this.graceTimer = undefined
    }
    this.clearPacedTimer()
  }

  createBoundStream(
    abortSignal: AbortSignal | undefined,
    onStart: () => void,
  ): ReadableStream<LanguageModelV3StreamPart> {
    let stepController: ReadableStreamDefaultController<LanguageModelV3StreamPart> | undefined
    return new ReadableStream<LanguageModelV3StreamPart>({
      start: (controller) => {
        stepController = controller
        this.controller = controller
        if (abortSignal) {
          const onAbort = () => {
            if (this.controller === stepController) {
              this.callbacks.onInterruptRequested("Aborted")
            }
          }
          if (abortSignal.aborted) {
            onAbort()
          } else {
            abortSignal.addEventListener("abort", onAbort, { once: true })
            this.controllerCleanup = () => abortSignal.removeEventListener("abort", onAbort)
          }
        }
        controller.enqueue({ type: "stream-start", warnings: [] })
        onStart()
      },
      cancel: () => {
        if (this.controller === stepController) {
          this.callbacks.onInterruptRequested("Stream cancelled")
        }
      },
    })
  }

  drainBufferedParts(): void {
    if (!this.controller) return
    while (this.bufferedParts.length > 0) {
      this.controller.enqueue(this.bufferedParts.shift()!)
    }
  }

  resumeWithToolResults(
    matchedResults: ReadonlyMap<
      string,
      {
        readonly text: string
        readonly content?: readonly McpContentItem[]
        readonly isError: boolean
      }
    >,
  ): void {
    this.drainBufferedParts()
    for (const callId of [...this.dispatchedCallIds]) {
      this.dispatchedCallIds.delete(callId)
      const pending = this.pendingMcpCalls.get(callId)
      if (!pending) continue
      this.pendingMcpCalls.delete(callId)
      if (!pending.resolved) {
        pending.resolved = true
        try {
          pending.abortCleanup?.()
        } catch {}
        pending.abortCleanup = undefined
        const res = matchedResults.get(callId) ?? {
          text: `Tool result missing for ${callId}`,
          isError: true,
        }
        pending.resolve({
          content:
            res.content && res.content.length > 0
              ? res.content
              : [{ type: "text", text: res.text }],
          isError: res.isError,
        })
      } else {
        try {
          pending.abortCleanup?.()
        } catch {}
        pending.abortCleanup = undefined
      }
    }
    this.syncWatchdogPause()

    if (this.promptSettled) {
      if (this.promptError) {
        this.finishWithError(this.promptError)
      } else {
        this.finalizeStop()
      }
      this.callbacks.onTurnSettled()
    } else if (!this.flushTimer && this.pendingMcpCalls.size > 0) {
      this.scheduleFlush(false)
    }
  }

  enqueueMcpCall(
    openCodeToolName: string,
    args: unknown,
    signal: AbortSignal,
  ): Promise<McpCallToolResult> {
    // If agy reported this relayed call under an unrecognized name in activeNativeTools,
    // remove that shadowing native tool entry so it can never block flushing.
    for (const [nativeId, nativeName] of this.activeNativeTools) {
      const stripped = nativeName.replace(/^acp_/, "")
      if (stripped === openCodeToolName || stripped.endsWith(`_${openCodeToolName}`)) {
        this.activeNativeTools.delete(nativeId)
        this.relayedUpdateIds.add(nativeId)
      }
    }

    const toolCallId = `oc_relay_${++this.partSeq}_${Math.random().toString(36).slice(2, 8)}`
    return new Promise<McpCallToolResult>((resolve) => {
      const entry: PendingMcpCall = {
        toolCallId,
        openCodeToolName,
        args,
        emittedToOpenCode: false,
        resolved: false,
        resolve,
      }
      this.pendingMcpCalls.set(toolCallId, entry)
      this.syncWatchdogPause()

      const onAbort = () => {
        if (!entry.resolved) {
          entry.resolved = true
          entry.abortCleanup = undefined
          this.pendingMcpCalls.delete(toolCallId)
          this.syncWatchdogPause()
          resolve({
            content: [{ type: "text", text: "MCP tool call aborted by ACP host." }],
            isError: true,
          })
        }
      }
      signal.addEventListener("abort", onAbort, { once: true })
      entry.abortCleanup = () => signal.removeEventListener("abort", onAbort)

      this.scheduleFlush(false)
    })
  }

  enqueueQuestion(
    questions: readonly OpenCodeQuestion[],
    toolCallIdHint?: string,
  ): Promise<McpCallToolResult> {
    const nativeToolCallId = this.findNativeQuestionCall(questions, toolCallIdHint)
    const toolCallId = `oc_question_${++this.partSeq}_${Math.random().toString(36).slice(2, 8)}`
    return new Promise<McpCallToolResult>((resolve) => {
      this.pendingMcpCalls.set(toolCallId, {
        toolCallId,
        openCodeToolName: OPENCODE_QUESTION_TOOL,
        args: { questions },
        nativeToolCallId,
        bypassNativeTools: nativeToolCallId === undefined,
        emittedToOpenCode: false,
        resolved: false,
        resolve,
      })
      this.syncWatchdogPause()
      this.scheduleFlush(true)
    })
  }

  private findNativeQuestionCall(
    questions: readonly OpenCodeQuestion[],
    hint?: string,
  ): string | undefined {
    if (hint && this.activeNativeTools.has(hint)) return hint
    const open = [...this.activeNativeTools.keys()]
    const firstQuestion = questions[0]?.question
    if (firstQuestion) {
      const byText = open.find((id) => this.nativeToolText.get(id)?.includes(firstQuestion))
      if (byText) return byText
    }
    return open.find((id) => /ask_question/i.test(this.nativeToolText.get(id) ?? ""))
  }

  private scheduleFlush(immediate: boolean): void {
    if (this.flushTimer) clearTimeout(this.flushTimer)
    this.flushTimer = setTimeout(() => {
      this.flushTimer = undefined
      this.tryFlush(false)
    }, immediate ? 0 : 5)
    this.flushTimer.unref?.()
  }

  private armGraceFlush(): void {
    if (this.graceTimer) return
    this.graceTimer = setTimeout(() => {
      this.graceTimer = undefined
      this.tryFlush(true)
    }, this.callbacks.nativeFlushGraceMs)
    this.graceTimer.unref?.()
  }

  tryFlush(forceBypass: boolean): void {
    if (!this.controller) return
    const unsent = [...this.pendingMcpCalls.values()].filter((c) => !c.emittedToOpenCode)
    if (unsent.length === 0) return

    const settleIds = new Set<string>()
    for (const call of unsent) {
      if (call.nativeToolCallId) settleIds.add(call.nativeToolCallId)
    }
    const bypass = forceBypass || unsent.some((c) => c.bypassNativeTools)

    if (settleIds.size > 0) {
      this.closeOpenBlocks()
      for (const id of settleIds) {
        this.settleNativeToolForUserInput(id)
      }
    }

    if (this.activeNativeTools.size > 0 && !bypass) {
      this.armGraceFlush()
      return
    }
    if (this.graceTimer) {
      clearTimeout(this.graceTimer)
      this.graceTimer = undefined
    }
    if (settleIds.size === 0) {
      this.closeOpenBlocks()
    }

    for (const call of unsent) {
      call.emittedToOpenCode = true
      this.dispatchedCallIds.add(call.toolCallId)
      const jsonInput = JSON.stringify(call.args ?? {})
      this.emitPart({
        type: "tool-input-start",
        id: call.toolCallId,
        toolName: call.openCodeToolName,
        providerExecuted: false,
      })
      this.emitPart({
        type: "tool-input-delta",
        id: call.toolCallId,
        delta: jsonInput,
      })
      this.emitPart({
        type: "tool-input-end",
        id: call.toolCallId,
      })
      this.emitPart({
        type: "tool-call",
        toolCallId: call.toolCallId,
        toolName: call.openCodeToolName,
        input: jsonInput,
        providerExecuted: false,
        providerMetadata: { [PROVIDER_ID]: { acp: true } },
      })
    }

    const finishReason: LanguageModelV3FinishReason = {
      unified: "tool-calls",
      raw: "tool_calls",
    }
    this.emitPart({
      type: "finish",
      finishReason,
      usage: buildUsage(),
      providerMetadata: { [PROVIDER_ID]: { acp: true } },
    })
    const ctrl = this.detachController()
    try {
      ctrl?.close()
    } catch {}
  }

  private settleNativeToolForUserInput(toolCallId: string): void {
    const toolName = this.activeNativeTools.get(toolCallId)
    if (toolName === undefined) return
    this.activeNativeTools.delete(toolCallId)
    this.relayedUpdateIds.add(toolCallId)
    this.emitPart({
      type: "tool-result",
      toolCallId,
      toolName,
      result: { status: "asked", message: "Asked the user with OpenCode's question tool." },
      isError: false,
      providerMetadata: { [PROVIDER_ID]: { acp: true } },
    })
  }

  private isRelayedToolName(name: string): boolean {
    if (
      name.startsWith("mcp_oc_") ||
      name.startsWith("mcp__oc__") ||
      name.startsWith("oc_") ||
      name.startsWith("oc:")
    ) {
      return true
    }
    return this.callbacks.getKnownRelayedNames().has(name)
  }

  handleUpdate(update: AcpSessionUpdate): void {
    if (this.abortedOnApiError) return
    const kind = update.sessionUpdate

    if (kind === "usage_update") {
      if (typeof update.used === "number" && Number.isFinite(update.used) && update.used >= 0) {
        this.contextUsed = update.used
      }
      return
    }

    if (kind === "agent_thought_chunk") {
      const content = isRecord(update.content) ? update.content : undefined
      const text = typeof content?.text === "string" ? content.text : undefined
      if (!text) return
      this.closeOpenText()
      if (!this.openReasoningId) {
        this.openReasoningId = `r_${++this.partSeq}`
        this.emitPart({
          type: "reasoning-start",
          id: this.openReasoningId,
          providerMetadata: { [PROVIDER_ID]: { acp: true } },
        })
      }
      this.emitPacedDelta("reasoning", this.openReasoningId, text)
      return
    }

    if (kind === "agent_message_chunk") {
      const content = isRecord(update.content) ? update.content : undefined
      const text = typeof content?.text === "string" ? content.text : undefined
      if (!text) return

      if (API_ERROR_ABORT_PATTERN.test(text)) {
        this.abortedOnApiError = true
        this.callbacks.onApiErrorText(text)
        return
      }

      this.closeOpenReasoning()
      if (!this.openTextId) {
        this.openTextId = `t_${++this.partSeq}`
        this.emitPart({
          type: "text-start",
          id: this.openTextId,
          providerMetadata: { [PROVIDER_ID]: { acp: true } },
        })
      }
      this.currentTextBuffer += text
      this.emitPacedDelta("text", this.openTextId, text)
      return
    }

    if (kind === "tool_call" || kind === "tool_call_update") {
      const callId = String(update.toolCallId ?? "")
      if (!callId || this.relayedUpdateIds.has(callId)) return

      const status = String(update.status ?? "").toLowerCase()
      let displayName = this.activeNativeTools.get(callId)
      if (displayName === undefined) {
        const name = String(update.name ?? update.kind ?? "tool")
        if (this.isRelayedToolName(name)) {
          this.relayedUpdateIds.add(callId)
          return
        }
        if (kind !== "tool_call") return

        this.closeOpenBlocks()
        displayName = `acp_${name}`
        this.activeNativeTools.set(callId, displayName)
        const rawInput = update.rawInput ?? update.args ?? {}
        const jsonInput = JSON.stringify(rawInput)
        this.nativeToolText.set(callId, `${name}\n${update.title ?? ""}\n${jsonInput}`)
        this.emitPart({
          type: "tool-input-start",
          id: callId,
          toolName: displayName,
          providerExecuted: true,
        })
        this.emitPart({
          type: "tool-input-delta",
          id: callId,
          delta: jsonInput,
        })
        this.emitPart({
          type: "tool-input-end",
          id: callId,
        })
        this.emitPart({
          type: "tool-call",
          toolCallId: callId,
          toolName: displayName,
          input: jsonInput,
          providerExecuted: true,
          providerMetadata: { [PROVIDER_ID]: { acp: true } },
        })
      }

      if (TERMINAL_TOOL_STATUSES.has(status)) {
        this.activeNativeTools.delete(callId)
        const isError = ERROR_TOOL_STATUSES.has(status)
        const resultPayload = formatNativeToolResult(update)
        this.emitPart({
          type: "tool-result",
          toolCallId: callId,
          toolName: displayName,
          result: resultPayload ?? { status },
          isError,
          providerMetadata: { [PROVIDER_ID]: { acp: true } },
        })

        if (this.pendingMcpCalls.size > 0) {
          this.tryFlush(false)
        }
      }
    }
  }

  cancelPendingCalls(reason: string): void {
    this.clearTimers()
    for (const call of this.pendingMcpCalls.values()) {
      if (!call.resolved) {
        call.resolved = true
        try {
          call.abortCleanup?.()
        } catch {}
        call.abortCleanup = undefined
        call.resolve({
          content: [{ type: "text", text: reason }],
          isError: true,
        })
      } else {
        try {
          call.abortCleanup?.()
        } catch {}
        call.abortCleanup = undefined
      }
    }
    this.pendingMcpCalls.clear()
    this.dispatchedCallIds.clear()
    this.syncWatchdogPause()
  }

  finishCleanStop(text: string, rawFinish: string, local = false): void {
    this.cancelPendingCalls(text)
    this.closeOpenBlocks()
    this.finishUnconfirmedNativeTools()

    const textId = `t_${++this.partSeq}`
    const meta = local
      ? { providerMetadata: { [PROVIDER_ID]: { localStatus: true } } }
      : { providerMetadata: { [PROVIDER_ID]: { acp: true } } }

    this.emitPart({ type: "text-start", id: textId, ...meta })
    this.emitPart({ type: "text-delta", id: textId, delta: text })
    this.emitPart({ type: "text-end", id: textId, ...meta })
    this.emitPart({
      type: "finish",
      finishReason: { unified: "stop", raw: rawFinish },
      usage: buildUsage(this.promptResult?.usage, this.contextUsed),
      ...meta,
    })

    const ctrl = this.detachController()
    try {
      ctrl?.close()
    } catch {}
  }

  finalizeStop(): void {
    this.cancelPendingCalls("Turn ended before OpenCode tool dispatch.")
    this.closeOpenBlocks()
    this.finishUnconfirmedNativeTools()

    const stopReason = this.promptResult?.stopReason
    const unified: LanguageModelV3FinishReason["unified"] =
      stopReason === "max_tokens" ? "length" : "stop"

    this.emitPart({
      type: "finish",
      finishReason: { unified, raw: stopReason ?? "end_turn" },
      usage: buildUsage(this.promptResult?.usage, this.contextUsed),
      providerMetadata: { [PROVIDER_ID]: { acp: true } },
    })

    const ctrl = this.detachController()
    try {
      ctrl?.close()
    } catch {}
  }

  finishWithError(error: Error): void {
    this.flushPacedImmediate()
    this.cancelPendingCalls(error.message)
    const ctrl = this.detachController()
    if (!ctrl) {
      this.bufferedParts.push({ type: "error", error })
      return
    }
    try {
      ctrl.enqueue({ type: "error", error })
      ctrl.close()
    } catch {}
  }

  private finishUnconfirmedNativeTools(): void {
    for (const [toolCallId, toolName] of this.activeNativeTools) {
      this.emitPart({
        type: "tool-result",
        toolCallId,
        toolName,
        result: {
          status: "unknown",
          message: "ACP did not report a final status for this tool call.",
        },
        isError: true,
        providerMetadata: { [PROVIDER_ID]: { acp: true } },
      })
    }
    this.activeNativeTools.clear()
  }

  private closeOpenText(): void {
    if (this.currentTextBuffer.trim().length > 0) {
      this.completedTextBlocks.push(this.currentTextBuffer)
    }
    this.currentTextBuffer = ""
    if (this.openTextId) {
      this.flushPacedImmediate()
      const id = this.openTextId
      this.openTextId = undefined
      this.emitPart({
        type: "text-end",
        id,
        providerMetadata: { [PROVIDER_ID]: { acp: true } },
      })
    }
  }

  private closeOpenReasoning(): void {
    if (!this.openReasoningId) return
    this.flushPacedImmediate()
    const id = this.openReasoningId
    this.openReasoningId = undefined
    this.emitPart({
      type: "reasoning-end",
      id,
      providerMetadata: { [PROVIDER_ID]: { acp: true } },
    })
  }

  private closeOpenBlocks(): void {
    this.closeOpenReasoning()
    this.closeOpenText()
  }

  private emitPart(part: LanguageModelV3StreamPart): void {
    if (this.controller) {
      try {
        this.controller.enqueue(part)
        return
      } catch {}
    }
    this.bufferedParts.push(part)
  }

  private detachController(): ReadableStreamDefaultController<LanguageModelV3StreamPart> | undefined {
    this.clearPacedTimer()
    const ctrl = this.controller
    this.controller = undefined
    const cleanup = this.controllerCleanup
    this.controllerCleanup = undefined
    cleanup?.()
    return ctrl
  }
}
