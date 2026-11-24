import type { LanguageModelV3Prompt, LanguageModelV3StreamPart } from "@ai-sdk/provider"
import type { Context } from "@opencode/plugin/effect/plugin"
import type { Provider } from "@opencode/schema/provider"
import type { Session } from "@opencode/schema/session"
import { Effect, Semaphore } from "effect"
import {
  AcpStallError,
  AgyNotFoundError,
  OPENCODE_QUESTION_TOOL,
  PROVIDER_ID,
  isRecord,
  resolveSettings,
  type AcpDiscoveredModel,
  type AcpElicitationRequestParams,
  type AcpElicitationResponse,
  type AcpMode,
  type AcpPermissionOutcome,
  type AcpPermissionRequestParams,
  type AcpSessionUpdate,
  type AgySettings,
  type CooldownBucket,
  type RequestKind,
  type SessionDiagnostics,
  type SessionPublicState,
} from "../types.js"
import {
  FALLBACK_MODELS,
  buildModelCatalog,
  catalogSignature,
  discoverAcpModels,
  isValidCatalog,
  normalizeCatalog,
  resolveAcpModelTarget,
} from "./catalog.js"
import {
  API_ERROR_ABORT_PATTERN,
  buildThinAcpPromptBlocks,
  extractToolResultsFromPrompt,
  isGlobalBreakerError,
  isLocalStatusMessage,
  isSyntheticRestartAfterError,
  quotaBucketForModel,
} from "./history.js"
import { buildSystemPromptAppend, hashSystemPrompt } from "./instructions.js"
import {
  McpRelayServer,
  buildToolDescriptors,
  type McpToolDescriptor,
} from "./mcp-relay.js"
import { AcpProcess, type AgyIO } from "./process.js"
import { elicitationToQuestions, parseQuestionToolAnswers } from "./question.js"
import {
  buildCheckpointPromptBlocks,
  buildSavedHistory,
  buildSideCallPromptBlocks,
  decideSync,
  fingerprintGroups,
  parsePersistedRecord,
  type PersistedSessionRecord,
  type SyncDecision,
} from "./sync.js"
import { ActiveTurn, buildUsage } from "./turn.js"

const STORAGE_KEY_PREFIX = "antigravity-acp:session:"
const COOLDOWN_STORAGE_KEY = "antigravity-acp:cooldown"
const MODEL_CATALOG_STORAGE_KEY = "antigravity-acp:model-catalog:v1"
const MODEL_CATALOG_TTL_MS = 6 * 60 * 60 * 1000
const COOLDOWN_BUCKETS: readonly CooldownBucket[] = ["global", "claude-gpt", "gemini", "other"]

export interface SessionContextSnapshot {
  readonly agent: string
  readonly toolSignature: string
  readonly tools: Record<string, { readonly description: string; readonly input: unknown }>
  readonly mcpSafeToOriginal: Map<string, string>
  readonly mcpDescriptors: McpToolDescriptor[]
  readonly rawInstructions?: string
  readonly systemPromptHash: string
}

interface Entry {
  readonly sessionID: string
  readonly lock: Semaphore.Semaphore
  cwd: string
  mode: AcpMode
  modeExplicit: boolean
  codeModeEnabled: boolean
  codeModeExplicit: boolean
  acpSessionId?: string
  proc?: AcpProcess
  lastModelId?: string
  lastModeApplied?: AcpMode
  lastAgent?: string
  lastError: string | null
  lastTitleSent?: string
  lastSystemPromptHash?: string
  persistedRecord?: PersistedSessionRecord
  contextSnapshot?: SessionContextSnapshot
  activeTurn?: ActiveTurn
  settlingPrompt?: Promise<void>
  acpToolsSignature?: string
  idleTimer?: ReturnType<typeof setTimeout>
  generation: number
}

export interface SessionRegistryOptions {
  readonly settings: AgySettings
  readonly io?: AgyIO
  readonly onCatalogChanged?: () => void
  readonly onStateChanged?: (sessionID: string) => void
}

export class SessionRegistry {
  readonly mcpRelay: McpRelayServer
  private baseSettings: AgySettings
  private readonly entries = new Map<string, Entry>()
  private readonly activeProcs = new Set<AcpProcess>()
  private readonly intentionallyDisposed = new WeakSet<AcpProcess>()
  private readonly cooldownsByBucket = new Map<CooldownBucket, { until: number; reason: string }>()
  private catalog: readonly AcpDiscoveredModel[] = FALLBACK_MODELS
  private catalogUpdatedAt = 0
  private catalogDiscoveryPromise?: Promise<readonly AcpDiscoveredModel[]>
  private cooldownLoadPromise?: Promise<void>
  private disposed = false

  constructor(
    private readonly ctx: Context,
    private readonly options: SessionRegistryOptions,
  ) {
    this.baseSettings = options.settings
    this.mcpRelay = new McpRelayServer({
      getTools: (sessionID) => this.entries.get(sessionID)?.contextSnapshot?.mcpDescriptors ?? [],
      callTool: (sessionID, toolName, args, signal) =>
        this.handleIncomingMcpCall(sessionID, toolName, args, signal),
    })
  }

  get settings(): AgySettings {
    return this.baseSettings
  }

  updateBaseSettings(settings: AgySettings): void {
    this.baseSettings = settings
    for (const entry of this.entries.values()) {
      if (!entry.codeModeExplicit) {
        entry.codeModeEnabled = settings.codeMode
      }
      if (!entry.modeExplicit) {
        entry.mode = entry.contextSnapshot?.agent === "plan" ? "plan" : settings.defaultMode
      }
    }
  }

  getModels(): readonly AcpDiscoveredModel[] {
    return this.catalog
  }

  hasSession(sessionID: string): boolean {
    return this.entries.has(sessionID)
  }

  private getOrCreateEntry(sessionID: string): Entry {
    let entry = this.entries.get(sessionID)
    if (!entry) {
      entry = {
        sessionID,
        lock: Semaphore.makeUnsafe(1),
        cwd: this.ctx.location.directory,
        mode: this.baseSettings.defaultMode,
        modeExplicit: false,
        codeModeEnabled: this.baseSettings.codeMode,
        codeModeExplicit: false,
        lastError: null,
        generation: 0,
      }
      this.entries.set(sessionID, entry)
    }
    return entry
  }

  private effectiveModeFor(
    entry: Entry,
    runtimeSettings: AgySettings,
    agentOverride?: string,
  ): AcpMode {
    if (entry.modeExplicit) return entry.mode
    const agent = agentOverride ?? entry.lastAgent ?? entry.contextSnapshot?.agent
    if (agent === "plan") return "plan"
    return runtimeSettings.defaultMode
  }

  private clearIdleTimer(entry: Entry): void {
    if (entry.idleTimer) {
      clearTimeout(entry.idleTimer)
      entry.idleTimer = undefined
    }
  }

  armIdle(sessionID: string): void {
    const entry = this.entries.get(sessionID)
    if (!entry || this.disposed) return
    this.clearIdleTimer(entry)
    // Never reap while a turn is active or parked waiting on OpenCode tools/questions.
    if (entry.activeTurn || !entry.proc || entry.proc.isClosed) return

    const idleMs = this.baseSettings.idleTimeoutMs
    if (idleMs <= 0) return
    entry.idleTimer = setTimeout(() => {
      entry.idleTimer = undefined
      void this.evictIdleProc(sessionID)
    }, idleMs)
    entry.idleTimer.unref?.()
  }

  private async evictIdleProc(sessionID: string): Promise<void> {
    const entry = this.entries.get(sessionID)
    if (!entry || this.disposed) return
    await Effect.runPromise(
      entry.lock.withPermit(
        Effect.promise(async () => {
          if (entry.activeTurn || !entry.proc || entry.proc.isClosed) return
          // Keep persisted acpSessionId in storage so the next turn resumes cleanly.
          await this.disposeEntryProc(entry, { clearSessionId: false })
        }),
      ),
    ).catch(() => {})
  }

  private async disposeEntryProc(
    entry: Entry,
    opts: { readonly clearSessionId: boolean },
  ): Promise<void> {
    const proc = entry.proc
    const acpSid = entry.acpSessionId
    entry.proc = undefined
    if (opts.clearSessionId) {
      entry.acpSessionId = undefined
      entry.acpToolsSignature = undefined
      entry.lastModelId = undefined
      entry.lastModeApplied = undefined
      entry.lastSystemPromptHash = undefined
    }
    entry.settlingPrompt = undefined
    if (!proc) return

    this.intentionallyDisposed.add(proc)
    this.activeProcs.delete(proc)
    if (!proc.isClosed && acpSid && opts.clearSessionId) {
      await proc.closeSession(acpSid)
    }
    if (!proc.isClosed) {
      await proc.dispose()
    }
  }

  async loadCachedModels(): Promise<{ readonly stale: boolean }> {
    try {
      const raw = await Effect.runPromise(this.ctx.storage.get(MODEL_CATALOG_STORAGE_KEY))
      if (isRecord(raw) && isValidCatalog(raw.models)) {
        const normalized = normalizeCatalog(raw.models)
        const updatedAt = typeof raw.updatedAt === "number" ? raw.updatedAt : 0
        this.catalog = normalized
        this.catalogUpdatedAt = updatedAt
        return { stale: Date.now() - updatedAt > MODEL_CATALOG_TTL_MS }
      }
    } catch {}
    return { stale: true }
  }

  private async persistCatalog(models: readonly AcpDiscoveredModel[], updatedAt: number): Promise<void> {
    try {
      const payload = JSON.parse(JSON.stringify({ version: 1, updatedAt, models }))
      await Effect.runPromise(this.ctx.storage.set(MODEL_CATALOG_STORAGE_KEY, payload))
    } catch {}
  }

  async refreshModels(options?: {
    readonly force?: boolean
  }): Promise<{ readonly models: readonly AcpDiscoveredModel[]; readonly changed: boolean }> {
    if (
      !options?.force &&
      this.catalogUpdatedAt > 0 &&
      Date.now() - this.catalogUpdatedAt <= MODEL_CATALOG_TTL_MS
    ) {
      return { models: this.catalog, changed: false }
    }

    if (!this.catalogDiscoveryPromise) {
      this.catalogDiscoveryPromise = (async () => {
        if (this.ctx.provider?.get) {
          try {
            const info = await Effect.runPromise(
              this.ctx.provider.get({ providerID: PROVIDER_ID as Provider.ID }).pipe(Effect.option),
            )
            if (info._tag === "Some") {
              this.updateBaseSettings(
                resolveSettings(
                  info.value.data as unknown as Record<string, unknown>,
                  process.env,
                  this.baseSettings,
                ),
              )
            }
          } catch {}
        }
        const discovered = await discoverAcpModels(
          this.ctx.location.directory,
          this.baseSettings,
          this.options.io,
        )
        const normalized = normalizeCatalog(discovered)
        if (normalized.length > 0) {
          const updatedAt = Date.now()
          this.catalogUpdatedAt = updatedAt
          await this.persistCatalog(normalized, updatedAt)
        }
        return normalized
      })().finally(() => {
        this.catalogDiscoveryPromise = undefined
      })
    }

    const normalized = await this.catalogDiscoveryPromise
    if (normalized.length === 0) {
      return { models: this.catalog, changed: false }
    }
    const changed = catalogSignature(this.catalog) !== catalogSignature(normalized)
    this.catalog = normalized
    return { models: normalized, changed }
  }

  async refreshModelsInBackground(): Promise<void> {
    try {
      const { changed } = await this.refreshModels({ force: false })
      if (!this.disposed && changed) {
        this.options.onCatalogChanged?.()
      }
    } catch {}
  }

  updateContextSnapshot(
    sessionID: string,
    input: {
      readonly agent: string
      readonly tools: Record<string, { readonly description: string; readonly input: unknown }>
      readonly rawInstructions?: string
    },
  ): void {
    const entry = this.getOrCreateEntry(sessionID)
    if (!entry.modeExplicit) {
      entry.mode = input.agent === "plan" ? "plan" : this.baseSettings.defaultMode
    }
    const systemPromptHash = hashSystemPrompt(input.rawInstructions)
    const toolSignature = Object.entries(input.tools)
      .map(([name, info]) => {
        let schema: string
        try {
          schema = JSON.stringify(info.input ?? null) ?? "null"
        } catch {
          schema = String(typeof info.input)
        }
        return `${name}\0${info.description ?? ""}\0${schema}`
      })
      .sort()
      .join("\0")

    const prev = entry.contextSnapshot
    if (
      prev &&
      prev.agent === input.agent &&
      prev.toolSignature === toolSignature &&
      prev.systemPromptHash === systemPromptHash
    ) {
      return
    }
    const { mcpDescriptors, mcpSafeToOriginal } = entry.codeModeEnabled
      ? buildToolDescriptors(input.tools)
      : { mcpDescriptors: [], mcpSafeToOriginal: new Map<string, string>() }

    entry.contextSnapshot = {
      agent: input.agent,
      toolSignature,
      tools: input.tools,
      mcpDescriptors,
      mcpSafeToOriginal,
      rawInstructions: input.rawInstructions,
      systemPromptHash,
    }
  }

  private acpToolsSignatureFor(entry: Entry): string {
    if (!entry.codeModeEnabled) return "pure"
    return entry.contextSnapshot ? `code:${entry.contextSnapshot.toolSignature}` : "code:__missing__"
  }

  private loadPersistedCooldown(): Promise<void> {
    if (!this.cooldownLoadPromise) {
      this.cooldownLoadPromise = (async () => {
        const read = async (key: string): Promise<unknown> => {
          try {
            return await Effect.runPromise(this.ctx.storage.get(key))
          } catch {
            return undefined
          }
        }
        const apply = (bucket: CooldownBucket, val: unknown) => {
          if (!isRecord(val)) return
          if (typeof val.until !== "number" || val.until <= Date.now()) return
          const cur = this.cooldownsByBucket.get(bucket)
          if (!cur || val.until > cur.until) {
            this.cooldownsByBucket.set(bucket, {
              until: val.until,
              reason: typeof val.reason === "string" ? val.reason : "upstream error",
            })
          }
        }
        apply("global", await read(COOLDOWN_STORAGE_KEY))
        for (const bucket of COOLDOWN_BUCKETS) {
          apply(bucket, await read(`${COOLDOWN_STORAGE_KEY}:${bucket}`))
        }
      })()
    }
    return this.cooldownLoadPromise
  }

  private tripCircuitBreaker(
    reason: string,
    bucket: CooldownBucket = "global",
  ): { until: number; reason: string } {
    const until = Date.now() + 60_000
    const trimmed = reason.trim().slice(0, 180)
    this.cooldownsByBucket.set(bucket, { until, reason: trimmed })
    try {
      Effect.runFork(
        this.ctx.storage
          .set(`${COOLDOWN_STORAGE_KEY}:${bucket}`, { until, reason: trimmed })
          .pipe(Effect.catch(() => Effect.void)),
      )
    } catch {}
    return { until, reason: trimmed }
  }

  private cooldownForModel(modelID: string): { until: number; reason: string } | undefined {
    const now = Date.now()
    const globalCooldown = this.cooldownsByBucket.get("global")
    if (globalCooldown && globalCooldown.until > now) return globalCooldown
    const bucketCooldown = this.cooldownsByBucket.get(quotaBucketForModel(modelID))
    if (bucketCooldown && bucketCooldown.until > now) return bucketCooldown
    return undefined
  }

  private activeCooldownForSession(entry?: Entry): { until: number; reason: string } | undefined {
    const now = Date.now()
    const g = this.cooldownsByBucket.get("global")
    if (g && g.until > now) return g
    if (entry?.lastModelId) {
      const b = this.cooldownsByBucket.get(quotaBucketForModel(entry.lastModelId))
      if (b && b.until > now) return b
    }
    return undefined
  }

  getState(sessionID: string): SessionPublicState {
    const entry = this.entries.get(sessionID)
    return {
      sessionID,
      mode: entry ? this.effectiveModeFor(entry, this.baseSettings) : this.baseSettings.defaultMode,
      codeModeEnabled: entry?.codeModeEnabled ?? this.baseSettings.codeMode,
      starting: false,
      acpSessionId: entry?.acpSessionId ?? null,
      connected: Boolean(entry?.proc && !entry.proc.isClosed && entry.acpSessionId),
      busy: Boolean(entry?.activeTurn),
      modelCount: this.catalog.length,
      relayedToolCount: entry?.contextSnapshot?.mcpDescriptors.length ?? 0,
      lastError: entry?.lastError ?? null,
      cooldownUntil: this.activeCooldownForSession(entry)?.until ?? null,
    }
  }

  getDiagnostics(sessionID: string): SessionDiagnostics {
    const entry = this.entries.get(sessionID)
    return {
      sessionID,
      acpSessionId: entry?.acpSessionId ?? null,
      mode: entry ? this.effectiveModeFor(entry, this.baseSettings) : this.baseSettings.defaultMode,
      codeModeEnabled: entry?.codeModeEnabled ?? this.baseSettings.codeMode,
      connected: Boolean(entry?.proc && !entry.proc.isClosed && entry.acpSessionId),
      busy: Boolean(entry?.activeTurn),
      watchdogPaused: entry?.proc?.isWatchdogPaused ?? false,
      activeNativeTools: entry?.activeTurn?.activeNativeToolNames ?? [],
      pendingMcpCalls: entry?.activeTurn?.pendingMcpCalls.size ?? 0,
      lastError: entry?.lastError ?? null,
      recentTrace: entry?.proc?.getRecentTrace() ?? [],
    }
  }

  async setMode(sessionID: string, mode: AcpMode): Promise<SessionPublicState> {
    const entry = this.getOrCreateEntry(sessionID)
    await Effect.runPromise(
      entry.lock.withPermit(
        Effect.promise(async () => {
          entry.mode = mode
          entry.modeExplicit = true
          if (entry.proc && !entry.proc.isClosed && entry.acpSessionId) {
            try {
              await entry.proc.setMode(entry.acpSessionId, mode)
              entry.lastModeApplied = mode
            } catch (err) {
              entry.lastError = err instanceof Error ? err.message : String(err)
            }
          }
        }),
      ),
    )
    this.notifyChanged(sessionID)
    return this.getState(sessionID)
  }

  async setCodeMode(sessionID: string, codeModeEnabled: boolean): Promise<SessionPublicState> {
    const entry = this.getOrCreateEntry(sessionID)
    entry.codeModeEnabled = codeModeEnabled
    entry.codeModeExplicit = true
    if (entry.contextSnapshot) {
      const { mcpDescriptors, mcpSafeToOriginal } = codeModeEnabled
        ? buildToolDescriptors(entry.contextSnapshot.tools)
        : { mcpDescriptors: [], mcpSafeToOriginal: new Map<string, string>() }
      entry.contextSnapshot = {
        ...entry.contextSnapshot,
        mcpDescriptors,
        mcpSafeToOriginal,
      }
    }
    await this.resetSession(sessionID)
    return this.getState(sessionID)
  }

  async resetSession(sessionID: string): Promise<SessionPublicState> {
    const entry = this.entries.get(sessionID)
    if (entry) {
      entry.generation++
      this.clearIdleTimer(entry)
      await Effect.runPromise(
        entry.lock.withPermit(
          Effect.promise(async () => {
            await this.interruptLocked(entry, "Session reset by user")
            await this.disposeEntryProc(entry, { clearSessionId: true })
            entry.lastError = null
            await this.runStorageRemove(sessionID)
          }),
        ),
      )
    }
    this.notifyChanged(sessionID)
    return this.getState(sessionID)
  }

  async deleteSession(sessionID: string): Promise<void> {
    await this.resetSession(sessionID)
    const entry = this.entries.get(sessionID)
    if (entry && !entry.activeTurn) {
      this.clearIdleTimer(entry)
      this.entries.delete(sessionID)
    }
    this.notifyChanged(sessionID)
  }

  async handleInterrupt(sessionID: string, reason = "Cancelled"): Promise<void> {
    const entry = this.entries.get(sessionID)
    if (!entry) return
    await Effect.runPromise(
      entry.lock.withPermit(Effect.promise(() => this.interruptLocked(entry, reason))),
    )
    this.notifyChanged(sessionID)
  }

  private async interruptLocked(entry: Entry, reason: string): Promise<void> {
    const turn = entry.activeTurn
    if (!turn) return
    if (entry.proc && !entry.proc.isClosed && entry.acpSessionId) {
      await entry.proc.cancel(entry.acpSessionId).catch(() => false)
    }
    turn.finishWithError(new Error(reason))
    if (entry.activeTurn === turn) entry.activeTurn = undefined
    await this.awaitSettlingOrKillLocked(entry)
  }

  private async awaitSettlingOrKillLocked(entry: Entry): Promise<void> {
    const settling = entry.settlingPrompt
    const settlingProc = entry.proc
    if (!settling) return
    let timer: ReturnType<typeof setTimeout> | undefined
    const hung = await Promise.race([
      settling.then(
        () => false,
        () => false,
      ),
      new Promise<true>((resolve) => {
        timer = setTimeout(() => resolve(true), this.baseSettings.settleTimeoutMs)
        timer.unref?.()
      }),
    ])
    if (timer) clearTimeout(timer)
    if (hung && settlingProc && entry.proc === settlingProc) {
      entry.lastError = "Previous Antigravity prompt did not settle in time; recycled session."
      await this.runStorageRemove(entry.sessionID)
      await this.disposeEntryProc(entry, { clearSessionId: true })
    }
  }

  private async softResetForHandoffLocked(entry: Entry): Promise<void> {
    await this.interruptLocked(entry, "Session reset for handoff")
    if (entry.proc && !entry.proc.isClosed && entry.acpSessionId) {
      await entry.proc.closeSession(entry.acpSessionId)
    }
    entry.acpSessionId = undefined
    entry.acpToolsSignature = undefined
    entry.lastModelId = undefined
    entry.lastModeApplied = undefined
    entry.lastSystemPromptHash = undefined
    entry.persistedRecord = undefined
    entry.lastError = null
    await this.runStorageRemove(entry.sessionID)
  }

  private async runStorageGetRecord(sessionID: string): Promise<PersistedSessionRecord | undefined> {
    const entry = this.entries.get(sessionID)
    if (entry?.persistedRecord) return entry.persistedRecord
    try {
      const val = await Effect.runPromise(this.ctx.storage.get(`${STORAGE_KEY_PREFIX}${sessionID}`))
      const parsed = parsePersistedRecord(val)
      if (parsed && entry) entry.persistedRecord = parsed
      return parsed
    } catch {
      return undefined
    }
  }

  private async runStorageSetRecord(entry: Entry, record: PersistedSessionRecord): Promise<void> {
    entry.persistedRecord = record
    try {
      await Effect.runPromise(
        this.ctx.storage.set(`${STORAGE_KEY_PREFIX}${entry.sessionID}`, JSON.stringify(record)),
      )
    } catch {}
  }

  private async runStorageRemove(sessionID: string): Promise<void> {
    const entry = this.entries.get(sessionID)
    if (entry) entry.persistedRecord = undefined
    try {
      await Effect.runPromise(this.ctx.storage.remove(`${STORAGE_KEY_PREFIX}${sessionID}`))
    } catch {}
  }

  private async ensureAcpSessionLocked(
    entry: Entry,
    runtimeSettings: AgySettings,
    options?: {
      readonly recorded?: PersistedSessionRecord
      readonly systemPromptAppend?: string
      readonly systemPromptHash?: string
    },
  ): Promise<{ proc: AcpProcess; acpSessionId: string; freshSession: boolean }> {
    try {
      const info = await Effect.runPromise(
        this.ctx.session.get({ sessionID: entry.sessionID as Session.ID }),
      )
      if (info?.location?.directory) {
        entry.cwd = info.location.directory
      }
    } catch {}

    let relayUrl = ""
    if (entry.codeModeEnabled) {
      await this.mcpRelay.start()
      relayUrl = this.mcpRelay.urlForSession(entry.sessionID)
    }
    const mcpServers = entry.codeModeEnabled
      ? [
          {
            type: "http" as const,
            name: "oc",
            url: relayUrl,
            headers: this.mcpRelay.authHeaders(),
          },
        ]
      : []

    const systemPromptAppend = options?.systemPromptAppend
    const systemPromptHash = options?.systemPromptHash
    const recorded = options?.recorded

    // Reuse live proc + session if alive and tool signature matches.
    if (entry.proc && !entry.proc.isClosed && entry.acpSessionId) {
      if (entry.acpToolsSignature === this.acpToolsSignatureFor(entry)) {
        return { proc: entry.proc, acpSessionId: entry.acpSessionId, freshSession: false }
      }
      const oldId = entry.acpSessionId
      try {
        await entry.proc.closeSession(oldId)
        const res = await entry.proc.newSession({
          cwd: entry.cwd,
          mcpServers,
          systemPromptAppend,
          timeoutMs: runtimeSettings.startupTimeoutMs,
        })
        if (!res.sessionId) throw new Error("Empty sessionId from refreshed ACP session")
        entry.acpSessionId = String(res.sessionId)
        entry.acpToolsSignature = this.acpToolsSignatureFor(entry)
        entry.lastModelId = undefined
        entry.lastModeApplied = undefined
        entry.lastSystemPromptHash = systemPromptHash
        const desiredMode = this.effectiveModeFor(entry, runtimeSettings)
        if (desiredMode !== "default") {
          await entry.proc.setMode(entry.acpSessionId, desiredMode).catch(() => {})
          entry.lastModeApplied = desiredMode
        }
        await this.runStorageSetRecord(entry, {
          version: 1,
          acpSessionId: entry.acpSessionId,
          turns: recorded?.turns ?? entry.persistedRecord?.turns ?? [],
          ...(entry.lastModelId ? { lastModelId: entry.lastModelId } : {}),
          ...(systemPromptHash ? { lastSystemPromptHash: systemPromptHash } : {}),
        })
        return { proc: entry.proc, acpSessionId: entry.acpSessionId, freshSession: true }
      } catch {
        await this.disposeEntryProc(entry, { clearSessionId: true })
      }
    }

    // Reuse live proc if still alive after softResetForHandoff; otherwise spawn fresh.
    let proc = entry.proc && !entry.proc.isClosed ? entry.proc : undefined
    if (!proc) {
      const spawned = new AcpProcess({
        cwd: entry.cwd,
        binaryPath: runtimeSettings.binaryPath,
        processPerSession: entry.codeModeEnabled,
        includeDefaultCustomizations: runtimeSettings.includeDefaultCustomizations,
        dangerouslySkipPermissions: false,
        extraDirs: runtimeSettings.extraDirs,
        traceLogPath: runtimeSettings.traceLogPath,
        io: this.options.io,
        onUpdate: (_sid, update) => this.handleAcpUpdate(entry, update),
        onPermissionRequest: (params) => this.handlePermissionRequest(entry, params),
        onElicitation: (params) => this.handleElicitation(entry, params),
        onExit: (code, signal, tail) => this.handleProcExit(entry, spawned, code, signal, tail),
      })
      this.activeProcs.add(spawned)
      try {
        await spawned.initialize(runtimeSettings.startupTimeoutMs)
        await spawned.authenticate(runtimeSettings.startupTimeoutMs)
        proc = spawned
      } catch (err) {
        this.intentionallyDisposed.add(spawned)
        this.activeProcs.delete(spawned)
        await spawned.dispose().catch(() => {})
        throw err
      }
    }

    const storedId = entry.acpSessionId ?? recorded?.acpSessionId
    let res: Awaited<ReturnType<AcpProcess["newSession"]>> | undefined
    let freshSession = false
    if (storedId) {
      try {
        res = await proc.resumeSession({
          sessionId: storedId,
          cwd: entry.cwd,
          mcpServers,
          systemPromptAppend,
          timeoutMs: runtimeSettings.startupTimeoutMs,
        })
      } catch {
        res = undefined
      }
    }
    if (!res?.sessionId) {
      freshSession = true
      res = await proc.newSession({
        cwd: entry.cwd,
        mcpServers,
        systemPromptAppend,
        timeoutMs: runtimeSettings.startupTimeoutMs,
      })
    }
    if (!res.sessionId) {
      this.intentionallyDisposed.add(proc)
      this.activeProcs.delete(proc)
      await proc.dispose().catch(() => {})
      throw new Error("Antigravity ACP did not return a sessionId")
    }

    entry.proc = proc
    entry.acpSessionId = String(res.sessionId)
    entry.acpToolsSignature = this.acpToolsSignatureFor(entry)
    entry.lastError = null
    entry.lastSystemPromptHash = systemPromptHash
    await this.runStorageSetRecord(entry, {
      version: 1,
      acpSessionId: entry.acpSessionId,
      turns: freshSession ? [] : (recorded?.turns ?? []),
      ...(entry.lastModelId ? { lastModelId: entry.lastModelId } : {}),
      ...(systemPromptHash ? { lastSystemPromptHash: systemPromptHash } : {}),
    })

    if (res.models?.availableModels?.length || res.configOptions?.length) {
      const live = normalizeCatalog(buildModelCatalog(res))
      if (live.length > 0 && catalogSignature(live) !== catalogSignature(this.catalog)) {
        const updatedAt = Date.now()
        this.catalog = live
        this.catalogUpdatedAt = updatedAt
        void this.persistCatalog(live, updatedAt)
        if (!this.disposed) this.options.onCatalogChanged?.()
      }
    }

    const desiredMode = this.effectiveModeFor(entry, runtimeSettings)
    if (desiredMode !== "default") {
      await proc.setMode(entry.acpSessionId, desiredMode).catch(() => {})
      entry.lastModeApplied = desiredMode
    }
    return { proc, acpSessionId: entry.acpSessionId, freshSession }
  }

  private handleAcpUpdate(entry: Entry, update: AcpSessionUpdate): void {
    if (update.sessionUpdate === "session_info_update" && typeof update.title === "string") {
      const title = update.title.trim()
      if (title && entry.lastTitleSent !== title) {
        entry.lastTitleSent = title
        Effect.runFork(
          this.ctx.session
            .update({ sessionID: entry.sessionID as Session.ID, title })
            .pipe(Effect.catch(() => Effect.void)),
        )
      }
      return
    }
    entry.activeTurn?.handleUpdate(update)
  }

  private async handlePermissionRequest(
    entry: Entry,
    params: AcpPermissionRequestParams,
  ): Promise<AcpPermissionOutcome> {
    const opts = params.options ?? []
    const allowOpt =
      opts.find(
        (o) => o.kind === "allow_once" || o.kind === "allow_always" || o.optionId.includes("allow"),
      ) ?? opts[0]
    const denyOpt = opts.find(
      (o) => o.kind === "reject_once" || o.kind === "reject_always" || o.optionId.includes("deny"),
    )

    const toolName = String(params.toolCall?.name ?? params.toolCall?.kind ?? "unknown")
    if (
      toolName.startsWith("mcp_oc_") ||
      toolName.startsWith("mcp__oc__") ||
      entry.mode === "auto"
    ) {
      return allowOpt ? { outcome: "selected", optionId: allowOpt.optionId } : { outcome: "cancelled" }
    }

    const title = String(params.toolCall?.title ?? toolName)
    const resources: string[] = []
    if (Array.isArray(params.toolCall?.locations)) {
      for (const loc of params.toolCall.locations) {
        if (loc?.path) resources.push(loc.path)
      }
    }
    if (resources.length === 0) resources.push(title)

    const proc = entry.proc
    proc?.pauseWatchdog()
    try {
      const timeoutMs = this.baseSettings.permissionTimeoutMs
      let timeoutHandle: ReturnType<typeof setTimeout> | undefined
      const timedOut = await Promise.race([
        Effect.runPromise(
          this.ctx.permission.assert({
            sessionID: entry.sessionID as Session.ID,
            agent: (entry.contextSnapshot?.agent ?? "build") as never,
            action: `antigravity.${params.toolCall?.kind ?? toolName}`,
            resources,
            metadata: {
              tool: toolName,
              title,
              rawInput: params.toolCall?.rawInput,
            },
          }),
        ).then(
          () => false,
          () => "denied" as const,
        ),
        new Promise<"timeout">((resolve) => {
          timeoutHandle = setTimeout(() => resolve("timeout"), timeoutMs)
          timeoutHandle.unref?.()
        }),
      ])
      if (timeoutHandle) clearTimeout(timeoutHandle)
      if (timedOut === false) {
        return allowOpt
          ? { outcome: "selected", optionId: allowOpt.optionId }
          : { outcome: "cancelled" }
      }
      if (timedOut === "timeout") {
        entry.lastError = `Permission prompt timed out after ${Math.round(timeoutMs / 1000)}s`
        return { outcome: "cancelled" }
      }
      return denyOpt ? { outcome: "selected", optionId: denyOpt.optionId } : { outcome: "cancelled" }
    } finally {
      proc?.resumeWatchdog()
    }
  }

  private async handleElicitation(
    entry: Entry,
    params: AcpElicitationRequestParams,
  ): Promise<AcpElicitationResponse> {
    const turn = entry.activeTurn
    if (!turn || !entry.contextSnapshot?.tools[OPENCODE_QUESTION_TOOL]) {
      return { action: "decline" }
    }
    let bridged
    try {
      bridged = elicitationToQuestions(params)
    } catch {
      return { action: "cancel" }
    }
    if (!bridged) return { action: "decline" }

    const result = await turn.enqueueQuestion(bridged.questions, params.toolCallId)
    if (result.isError) return { action: "cancel" }
    const answers = parseQuestionToolAnswers(
      result.content.map((c) => c.text).join("\n"),
      bridged.questions,
    )
    if (!answers) return { action: "cancel" }
    const content = bridged.respond(answers)
    if (!content) return { action: "cancel" }
    return { action: "accept", content }
  }

  private handleIncomingMcpCall(
    sessionID: string,
    mcpToolName: string,
    args: unknown,
    signal: AbortSignal,
  ) {
    const entry = this.entries.get(sessionID)
    const turn = entry?.activeTurn
    if (!entry || !turn) {
      return Promise.resolve({
        content: [{ type: "text" as const, text: "No active Antigravity turn to execute OpenCode tool." }],
        isError: true,
      })
    }
    const openCodeToolName = entry.contextSnapshot?.mcpSafeToOriginal.get(mcpToolName) ?? mcpToolName
    return turn.enqueueMcpCall(openCodeToolName, args, signal)
  }

  private handleProcExit(
    entry: Entry,
    proc: AcpProcess,
    code: number | null,
    signal: NodeJS.Signals | null,
    tail: string,
  ): void {
    this.activeProcs.delete(proc)
    if (this.disposed || this.intentionallyDisposed.has(proc) || entry.proc !== proc) {
      return
    }
    const lastLine = tail.split("\n").filter(Boolean).at(-1)
    entry.lastError = `agy acp exited (code=${code}, signal=${signal})${lastLine ? `: ${lastLine}` : ""}`
    entry.proc = undefined
    entry.acpSessionId = undefined
    entry.acpToolsSignature = undefined
    entry.lastModelId = undefined
    entry.lastModeApplied = undefined
    entry.lastSystemPromptHash = undefined
    entry.settlingPrompt = undefined
    if (entry.activeTurn) {
      const turn = entry.activeTurn
      entry.activeTurn = undefined
      turn.finishWithError(new Error(entry.lastError))
    }
    this.notifyChanged(entry.sessionID)
  }

  private createImmediateTextStream(
    text: string,
    rawFinish: string,
  ): ReadableStream<LanguageModelV3StreamPart> {
    const id = "t_1"
    const localMeta = { providerMetadata: { [PROVIDER_ID]: { localStatus: true } } } as const
    return new ReadableStream<LanguageModelV3StreamPart>({
      start: (controller) => {
        controller.enqueue({ type: "stream-start", warnings: [] })
        controller.enqueue({ type: "text-start", id, ...localMeta })
        controller.enqueue({ type: "text-delta", id, delta: text })
        controller.enqueue({ type: "text-end", id, ...localMeta })
        controller.enqueue({
          type: "finish",
          finishReason: { unified: "stop", raw: rawFinish },
          usage: buildUsage(),
          ...localMeta,
        })
        controller.close()
      },
    })
  }

  async startOrResumeTurn(input: {
    readonly sessionID: string
    readonly modelID: string
    readonly variant?: string
    readonly requestKind?: RequestKind
    readonly agent?: string
    readonly prompt: LanguageModelV3Prompt
    readonly abortSignal?: AbortSignal
    readonly runtimeSettings?: AgySettings
  }): Promise<ReadableStream<LanguageModelV3StreamPart>> {
    if (this.disposed) {
      return this.createImmediateTextStream(
        "Antigravity ACP plugin was disabled or reloaded. Send a new message to continue.",
        "plugin_disposed",
      )
    }

    if (input.requestKind === "compaction" || input.requestKind === "generate") {
      return this.runSideCall({
        sessionID: input.sessionID,
        modelID: input.modelID,
        variant: input.variant,
        requestKind: input.requestKind,
        prompt: input.prompt,
        abortSignal: input.abortSignal,
        runtimeSettings: input.runtimeSettings,
      })
    }

    const entry = this.getOrCreateEntry(input.sessionID)
    this.clearIdleTimer(entry)
    const runtimeSettings = input.runtimeSettings ?? this.baseSettings
    if (!entry.codeModeExplicit && runtimeSettings.codeMode !== entry.codeModeEnabled) {
      entry.codeModeEnabled = runtimeSettings.codeMode
    }

    // Immediate continuation path when OpenCode finishes executing parked MCP / question calls.
    const existingTurn = entry.activeTurn
    if (existingTurn && existingTurn.dispatchedCallIds.size > 0) {
      const matched = extractToolResultsFromPrompt(input.prompt, existingTurn.dispatchedCallIds)
      if (matched.size > 0) {
        return existingTurn.createBoundStream(input.abortSignal, () => {
          existingTurn.resumeWithToolResults(matched)
        })
      }
    }

    if (isSyntheticRestartAfterError(input.prompt)) {
      return this.createImmediateTextStream(
        "Skipped auto-resume after server restart because the previous Antigravity turn halted on an API error. Send a new message when ready.",
        "restart_skipped",
      )
    }

    await this.loadPersistedCooldown()
    const cooldown = this.cooldownForModel(input.modelID)
    if (cooldown) {
      const secs = Math.ceil((cooldown.until - Date.now()) / 1000)
      return this.createImmediateTextStream(
        `Antigravity ACP cooldown active (${secs}s remaining) after upstream API error (${cooldown.reason}) to protect your account from retry loops.`,
        "circuit_open",
      )
    }

    const groups = fingerprintGroups(input.prompt)
    const nonSystem = input.prompt.filter((m) => m.role !== "system" && !isLocalStatusMessage(m))
    const lastIsUser = nonSystem.at(-1)?.role === "user"
    const priorGroups = lastIsUser ? groups.slice(0, -1) : groups
    const currentUserMsg = lastIsUser ? groups.at(-1)?.userMessage : undefined

    let prepared: {
      readonly proc: AcpProcess
      readonly acpSessionId: string
      readonly syncDecision: SyncDecision
      readonly systemPromptHash: string
      readonly turnGeneration: number
    }
    try {
      prepared = await Effect.runPromise(
        entry.lock.withPermit(
          Effect.promise(async () => {
            if (entry.activeTurn) {
              await this.interruptLocked(entry, "Superseded by new prompt")
            } else {
              await this.awaitSettlingOrKillLocked(entry)
            }

            const postSettleCooldown = this.cooldownForModel(input.modelID)
            if (postSettleCooldown) {
              throw new Error(`COOLDOWN:${postSettleCooldown.until}:${postSettleCooldown.reason}`)
            }

            const recorded = await this.runStorageGetRecord(entry.sessionID)
            const hasLiveOrResumableAcpSession = Boolean(entry.acpSessionId ?? recorded?.acpSessionId)
            let syncDecision = decideSync({
              recorded,
              priorGroups,
              hasLiveOrResumableAcpSession,
            })

            if (syncDecision.kind === "diverged") {
              await this.softResetForHandoffLocked(entry)
              syncDecision =
                priorGroups.length > 0
                  ? { kind: "fresh-with-history", priorGroups }
                  : { kind: "fresh" }
            }

            const systemPromptAppend = buildSystemPromptAppend({
              rawInstructions: entry.contextSnapshot?.rawInstructions,
              forwardInstructions: runtimeSettings.forwardInstructions,
            })
            const systemPromptHash = hashSystemPrompt(systemPromptAppend)

            const ensured = await this.ensureAcpSessionLocked(entry, runtimeSettings, {
              recorded:
                syncDecision.kind === "fresh" || syncDecision.kind === "fresh-with-history"
                  ? undefined
                  : recorded,
              systemPromptAppend,
              systemPromptHash,
            })
            if (
              ensured.freshSession &&
              (syncDecision.kind === "continue" || syncDecision.kind === "catch-up")
            ) {
              syncDecision =
                priorGroups.length > 0
                  ? { kind: "fresh-with-history", priorGroups }
                  : { kind: "fresh" }
            }

            const target = resolveAcpModelTarget(this.catalog, input.modelID, input.variant)
            if (entry.lastModelId !== target.acpModelId) {
              try {
                await ensured.proc.setModel(ensured.acpSessionId, target.acpModelId)
              } catch {
                await ensured.proc
                  .setConfigOption(ensured.acpSessionId, "model", target.baseModelId)
                  .catch(() => {})
                if (target.thoughtLevel) {
                  await ensured.proc
                    .setConfigOption(ensured.acpSessionId, "thought_level", target.thoughtLevel)
                    .catch(() => {})
                }
              }
              entry.lastModelId = target.acpModelId
            }

            const desiredMode = this.effectiveModeFor(entry, runtimeSettings, input.agent)
            if (input.agent) entry.lastAgent = input.agent
            if (!entry.modeExplicit) {
              entry.mode = desiredMode
            }
            if (
              entry.lastModeApplied !== desiredMode &&
              (desiredMode !== "default" || entry.lastModeApplied !== undefined)
            ) {
              await ensured.proc.setMode(ensured.acpSessionId, desiredMode).catch(() => {})
              entry.lastModeApplied = desiredMode
            }

            return {
              proc: ensured.proc,
              acpSessionId: ensured.acpSessionId,
              syncDecision,
              systemPromptHash,
              turnGeneration: entry.generation,
            }
          }),
        ),
      )
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      if (msg.startsWith("COOLDOWN:")) {
        const [, untilStr, ...reasonParts] = msg.split(":")
        const secs = Math.max(1, Math.ceil((Number(untilStr) - Date.now()) / 1000))
        return this.createImmediateTextStream(
          `Antigravity ACP cooldown active (${secs}s remaining) after upstream API error (${reasonParts.join(":")}) to protect your account from retry loops.`,
          "circuit_open",
        )
      }
      if (err instanceof AgyNotFoundError || (err as NodeJS.ErrnoException)?.code === "ENOENT") {
        entry.lastError = msg
        this.notifyChanged(input.sessionID)
        return this.createImmediateTextStream(msg, "agy_unavailable")
      }
      throw err
    }

    const { proc, acpSessionId, syncDecision, systemPromptHash, turnGeneration } = prepared
    const promptBlocks =
      syncDecision.kind === "catch-up"
        ? buildCheckpointPromptBlocks({
            groups: syncDecision.unseenGroups,
            currentUserMessage: currentUserMsg,
            mode: "catch-up",
          })
        : syncDecision.kind === "fresh-with-history"
          ? buildCheckpointPromptBlocks({
              groups: syncDecision.priorGroups,
              currentUserMessage: currentUserMsg,
              mode: "restore",
            })
          : buildThinAcpPromptBlocks(input.prompt)

    let recordPersisted = false
    const persistTurnRecord = () => {
      if (recordPersisted || this.disposed || entry.generation !== turnGeneration) return
      recordPersisted = true
      const turns = buildSavedHistory({
        priorGroups,
        currentUserMessage: currentUserMsg,
        assistantTextBlocks: turn.assistantTextBlocks,
      })
      void this.runStorageSetRecord(entry, {
        version: 1,
        acpSessionId,
        turns,
        ...(entry.lastModelId ? { lastModelId: entry.lastModelId } : {}),
        ...(systemPromptHash ? { lastSystemPromptHash: systemPromptHash } : {}),
      })
    }

    const turn = new ActiveTurn({
      pauseWatchdog: () => proc.pauseWatchdog(),
      resumeWatchdog: () => proc.resumeWatchdog(),
      nativeFlushGraceMs: runtimeSettings.nativeFlushGraceMs,
      getKnownRelayedNames: () =>
        new Set(entry.contextSnapshot?.mcpDescriptors.map((d) => d.name) ?? []),
      onInterruptRequested: (reason) => {
        void this.handleInterrupt(input.sessionID, reason)
      },
      onTurnSettled: () => {
        if (!turn.promptError && !turn.abortedOnApiError) {
          persistTurnRecord()
        }
        if (entry.activeTurn === turn) entry.activeTurn = undefined
        this.notifyChanged(input.sessionID)
      },
      onApiErrorText: (text) => {
        this.handleTurnApiError(entry, turn, proc, acpSessionId, input.modelID, text)
      },
    })
    entry.activeTurn = turn
    this.notifyChanged(input.sessionID)

    return turn.createBoundStream(input.abortSignal, () => {
      const pending = proc.prompt(acpSessionId, promptBlocks, {
        idleWatchdogMs: runtimeSettings.chunkTimeoutMs,
      })
      const tracked = pending.then(
        () => {
          if (entry.settlingPrompt === tracked) entry.settlingPrompt = undefined
        },
        () => {
          if (entry.settlingPrompt === tracked) entry.settlingPrompt = undefined
        },
      )
      entry.settlingPrompt = tracked

      pending
        .then((res) => {
          if (turn.abortedOnApiError || this.disposed || entry.generation !== turnGeneration) return
          turn.promptSettled = true
          turn.promptResult = res
          entry.lastError = null
          if (turn.dispatchedCallIds.size === 0 && turn.hasController) {
            persistTurnRecord()
            turn.finalizeStop()
            if (entry.activeTurn === turn) entry.activeTurn = undefined
            this.notifyChanged(input.sessionID)
          }
        })
        .catch((err) => {
          if (turn.abortedOnApiError || this.disposed || entry.generation !== turnGeneration) return
          turn.promptSettled = true
          const error = err instanceof Error ? err : new Error(String(err))
          turn.promptError = error

          if (err instanceof AcpStallError) {
            entry.lastError = err.message
            void this.recoverStalledProc(entry, proc, acpSessionId, turn, err)
            return
          }

          if (API_ERROR_ABORT_PATTERN.test(error.message)) {
            turn.abortedOnApiError = true
            this.handleTurnApiError(entry, turn, proc, acpSessionId, input.modelID, error.message)
            return
          }

          entry.lastError = error.message
          turn.finishWithError(error)
          if (entry.activeTurn === turn) entry.activeTurn = undefined
          this.notifyChanged(input.sessionID)
        })
    })
  }

  private async runSideCall(input: {
    readonly sessionID: string
    readonly modelID: string
    readonly variant?: string
    readonly requestKind: "compaction" | "generate"
    readonly prompt: LanguageModelV3Prompt
    readonly abortSignal?: AbortSignal
    readonly runtimeSettings?: AgySettings
  }): Promise<ReadableStream<LanguageModelV3StreamPart>> {
    await this.loadPersistedCooldown()
    const cooldown = this.cooldownForModel(input.modelID)
    if (cooldown) {
      const secs = Math.ceil((cooldown.until - Date.now()) / 1000)
      return this.createImmediateTextStream(
        `Antigravity ACP cooldown active (${secs}s remaining) after upstream API error (${cooldown.reason}) to protect your account from retry loops.`,
        "circuit_open",
      )
    }

    const runtimeSettings = input.runtimeSettings ?? this.baseSettings
    const entry = this.entries.get(input.sessionID)
    let cwd = entry?.cwd ?? this.ctx.location.directory
    if (!entry) {
      try {
        const info = await Effect.runPromise(
          this.ctx.session.get({ sessionID: input.sessionID as Session.ID }),
        )
        if (info?.location?.directory) cwd = info.location.directory
      } catch {}
    }

    let proc: AcpProcess | undefined
    let sideSessionId: string | undefined
    let cleanedUp = false
    const cleanup = () => {
      if (cleanedUp) return
      cleanedUp = true
      if (!proc) return
      const targetProc = proc
      this.intentionallyDisposed.add(targetProc)
      this.activeProcs.delete(targetProc)
      if (!targetProc.isClosed && sideSessionId) {
        targetProc.notifyCancel(sideSessionId)
      }
      void targetProc.dispose().catch(() => {})
    }

    const turn = new ActiveTurn({
      pauseWatchdog: () => proc?.pauseWatchdog(),
      resumeWatchdog: () => proc?.resumeWatchdog(),
      nativeFlushGraceMs: runtimeSettings.nativeFlushGraceMs,
      getKnownRelayedNames: () => new Set<string>(),
      onInterruptRequested: (reason) => {
        cleanup()
        turn.finishWithError(new Error(reason))
      },
      onTurnSettled: () => cleanup(),
      onApiErrorText: (text) => {
        if (isGlobalBreakerError(text)) {
          this.tripCircuitBreaker(text, "global")
        } else {
          this.tripCircuitBreaker(text, quotaBucketForModel(input.modelID))
        }
        cleanup()
        turn.finishCleanStop(
          `Antigravity ACP halted side call on upstream error:\n${text.trim()}`,
          "api_error_halted",
          true,
        )
      },
    })

    proc = new AcpProcess({
      cwd,
      binaryPath: runtimeSettings.binaryPath,
      processPerSession: false,
      includeDefaultCustomizations: runtimeSettings.includeDefaultCustomizations,
      dangerouslySkipPermissions: false,
      extraDirs: runtimeSettings.extraDirs,
      traceLogPath: runtimeSettings.traceLogPath,
      io: this.options.io,
      onUpdate: (_sid, update) => turn.handleUpdate(update),
      onExit: (code, signal, tail) => {
        if (proc) this.activeProcs.delete(proc)
        if (cleanedUp || this.disposed || (proc && this.intentionallyDisposed.has(proc))) return
        const lastLine = tail.split("\n").filter(Boolean).at(-1)
        turn.finishWithError(
          new Error(`agy acp exited during ${input.requestKind} (code=${code}, signal=${signal})${lastLine ? `: ${lastLine}` : ""}`),
        )
      },
    })
    this.activeProcs.add(proc)

    try {
      await proc.initialize(runtimeSettings.startupTimeoutMs)
      await proc.authenticate(runtimeSettings.startupTimeoutMs)
      const res = await proc.newSession({
        cwd,
        mcpServers: [],
        timeoutMs: runtimeSettings.startupTimeoutMs,
      })
      if (!res.sessionId) {
        throw new Error(`Antigravity ACP did not return a sessionId for ${input.requestKind}`)
      }
      sideSessionId = String(res.sessionId)
      const target = resolveAcpModelTarget(this.catalog, input.modelID, input.variant)
      try {
        await proc.setModel(sideSessionId, target.acpModelId)
      } catch {
        await proc.setConfigOption(sideSessionId, "model", target.baseModelId).catch(() => {})
        if (target.thoughtLevel) {
          await proc
            .setConfigOption(sideSessionId, "thought_level", target.thoughtLevel)
            .catch(() => {})
        }
      }
    } catch (err) {
      cleanup()
      if (err instanceof AgyNotFoundError || (err as NodeJS.ErrnoException)?.code === "ENOENT") {
        return this.createImmediateTextStream(err instanceof Error ? err.message : String(err), "agy_unavailable")
      }
      throw err
    }

    const activeProc = proc
    const activeSid = sideSessionId
    const promptBlocks = buildSideCallPromptBlocks(input.prompt, input.requestKind)

    return turn.createBoundStream(input.abortSignal, () => {
      activeProc
        .prompt(activeSid, promptBlocks, {
          idleWatchdogMs: runtimeSettings.chunkTimeoutMs,
        })
        .then(async (res) => {
          if (turn.abortedOnApiError || this.disposed) return
          turn.promptSettled = true
          turn.promptResult = res
          await activeProc.closeSession(activeSid).catch(() => undefined)
          cleanup()
          if (turn.hasController) {
            turn.finalizeStop()
          }
        })
        .catch((err) => {
          if (turn.abortedOnApiError || this.disposed) return
          turn.promptSettled = true
          const error = err instanceof Error ? err : new Error(String(err))
          turn.promptError = error
          cleanup()
          turn.finishWithError(error)
        })
    })
  }

  private async recoverStalledProc(
    entry: Entry,
    proc: AcpProcess,
    acpSessionId: string,
    turn: ActiveTurn,
    stallError: AcpStallError,
  ): Promise<void> {
    if (!proc.isClosed) {
      await proc.cancel(acpSessionId).catch(() => false)
    }
    this.intentionallyDisposed.add(proc)
    this.activeProcs.delete(proc)
    if (entry.proc === proc) {
      entry.proc = undefined
      entry.acpSessionId = undefined
      entry.acpToolsSignature = undefined
      entry.lastModelId = undefined
      entry.lastModeApplied = undefined
      entry.settlingPrompt = undefined
    }
    await this.runStorageRemove(entry.sessionID)
    // Give the child a brief window to observe session/cancel before SIGTERM/SIGKILL.
    await new Promise<void>((resolve) => {
      const t = setTimeout(resolve, Math.min(this.baseSettings.settleTimeoutMs, 100))
      t.unref?.()
    })
    proc.killImmediate()
    turn.finishWithError(stallError)
    if (entry.activeTurn === turn) entry.activeTurn = undefined
    this.notifyChanged(entry.sessionID)
  }

  private handleTurnApiError(
    entry: Entry,
    turn: ActiveTurn,
    proc: AcpProcess,
    acpSessionId: string,
    modelID: string,
    rawText: string,
  ): void {
    if (isGlobalBreakerError(rawText)) {
      const rec = this.tripCircuitBreaker(rawText, "global")
      entry.lastError = rec.reason
      void this.runStorageRemove(entry.sessionID)
      this.intentionallyDisposed.add(proc)
      this.activeProcs.delete(proc)
      if (entry.proc === proc) {
        entry.proc = undefined
        entry.acpSessionId = undefined
        entry.acpToolsSignature = undefined
        entry.lastModelId = undefined
        entry.lastModeApplied = undefined
        entry.settlingPrompt = undefined
      }
      if (!proc.isClosed) {
        proc.notifyCancel(acpSessionId)
        proc.killImmediate()
      }
      turn.finishCleanStop(
        `Antigravity ACP halted immediately after upstream auth/permission error to prevent retry loops:\n${rawText.trim()}`,
        "api_error_halted",
        true,
      )
    } else {
      const bucket = quotaBucketForModel(entry.lastModelId ?? modelID)
      const rec = this.tripCircuitBreaker(rawText, bucket)
      entry.lastError = rec.reason
      if (!proc.isClosed) {
        proc.notifyCancel(acpSessionId)
      }
      turn.finishCleanStop(
        `Antigravity ACP halted this turn on a ${bucket} quota error (60s cooldown for ${bucket} models — switch model pools to continue immediately):\n${rawText.trim()}`,
        "quota_cooldown",
        true,
      )
    }
    if (entry.activeTurn === turn) entry.activeTurn = undefined
    this.notifyChanged(entry.sessionID)
  }

  async disposeAll(): Promise<void> {
    this.disposed = true
    const snapshot = [...this.entries.values()]
    for (const entry of snapshot) {
      this.clearIdleTimer(entry)
      if (entry.activeTurn) {
        entry.activeTurn.finishCleanStop(
          "Antigravity ACP plugin reloaded or stopped — send a new message to continue.",
          "plugin_unloaded",
          true,
        )
        entry.activeTurn = undefined
      }
    }

    const jobs: Promise<void>[] = []
    for (const entry of snapshot) {
      if (entry.proc) {
        const proc = entry.proc
        const sid = entry.acpSessionId
        this.intentionallyDisposed.add(proc)
        this.activeProcs.delete(proc)
        entry.proc = undefined
        entry.acpSessionId = undefined
        if (!proc.isClosed && sid) {
          try {
            proc.notifyCancel(sid)
          } catch {}
        }
        jobs.push(proc.dispose().catch(() => {}))
      }
    }
    await Promise.allSettled(jobs)

    const remaining = [...this.activeProcs]
    await Promise.allSettled(
      remaining.map((proc) => {
        this.intentionallyDisposed.add(proc)
        return proc.dispose().catch(() => {})
      }),
    )
    this.activeProcs.clear()
    this.entries.clear()
    await this.mcpRelay.stop().catch(() => {})
  }

  private notifyChanged(sessionID: string): void {
    try {
      this.options.onStateChanged?.(sessionID)
    } catch {}
  }
}
