import { execFile, spawn, type ChildProcessWithoutNullStreams } from "node:child_process"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { StringDecoder } from "node:string_decoder"
import {
  AcpRpcError,
  AcpStallError,
  AgyNotFoundError,
  isRecord,
  type AcpAuthMethodId,
  type AcpContentBlock,
  type AcpElicitationRequestParams,
  type AcpElicitationResponse,
  type AcpMcpServerHttp,
  type AcpPermissionOutcome,
  type AcpPermissionRequestParams,
  type AcpPromptResult,
  type AcpSessionResult,
  type AcpSessionUpdate,
  type WireTraceEntry,
} from "../types.js"

const MAX_STDOUT_BUFFER_BYTES = 16 * 1024 * 1024
const MAX_STDERR_LINES = 200
const MAX_TRACE_ENTRIES = 40

const KNOWN_AUTH_METHODS: readonly string[] = [
  "oauth-personal",
  "oauth-business",
  "gemini-api-key",
  "agent-platform",
]

const OAUTH_COLLIDING_ENV_KEYS = [
  "GEMINI_API_KEY",
  "GOOGLE_API_KEY",
  "GOOGLE_APPLICATION_CREDENTIALS",
] as const

export interface AgyIO {
  readonly spawnProcess: typeof spawn
  readonly resolveBinary: (env: NodeJS.ProcessEnv, explicitPath?: string) => string
  readonly env: () => NodeJS.ProcessEnv
}

function isExecutableFile(filePath: string): boolean {
  try {
    const stat = fs.statSync(filePath)
    if (!stat.isFile()) return false
    fs.accessSync(
      filePath,
      process.platform === "win32" ? fs.constants.F_OK : fs.constants.X_OK,
    )
    return true
  } catch {
    return false
  }
}

export function resolveAgyBinary(
  env: NodeJS.ProcessEnv = process.env,
  explicitPath?: string,
): string {
  const override = explicitPath?.trim() || env.AGY_BIN?.trim()
  if (override) return override

  const candidates = process.platform === "win32" ? ["agy.exe", "agy.cmd", "agy"] : ["agy"]
  const pathDirs = (env.PATH ?? "").split(path.delimiter).filter(Boolean)
  for (const dir of pathDirs) {
    for (const name of candidates) {
      const candidate = path.join(dir, name)
      if (isExecutableFile(candidate)) return candidate
    }
  }

  const home = env.HOME || env.USERPROFILE || os.homedir()
  const fallbackPaths =
    process.platform === "win32"
      ? [
          ...(env.LOCALAPPDATA
            ? [path.join(env.LOCALAPPDATA, "Programs", "Antigravity", "bin", "agy.exe")]
            : []),
          ...(home ? [path.join(home, ".local", "bin", "agy.exe")] : []),
        ]
      : [
          ...(home ? [path.join(home, ".local", "bin", "agy"), path.join(home, "bin", "agy")] : []),
          "/opt/homebrew/bin/agy",
          "/usr/local/bin/agy",
          "/opt/Antigravity/bin/agy",
          "/usr/bin/agy",
        ]

  for (const candidate of fallbackPaths) {
    if (isExecutableFile(candidate)) return candidate
  }

  return "agy"
}

export const defaultAgyIO: AgyIO = {
  spawnProcess: spawn,
  resolveBinary: resolveAgyBinary,
  env: () => process.env,
}

export function resolveAcpAuthMethod(env: NodeJS.ProcessEnv = process.env): AcpAuthMethodId {
  const override = env.ANTIGRAVITY_ACP_AUTH_METHOD?.trim()
  if (override && KNOWN_AUTH_METHODS.includes(override)) {
    return override as AcpAuthMethodId
  }
  if (env.GEMINI_API_KEY?.trim()) return "gemini-api-key"
  return "oauth-personal"
}

export function buildAgySpawnEnv(
  source: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
): Record<string, string | undefined> {
  const env: Record<string, string | undefined> = { ...source }
  const method = resolveAcpAuthMethod(source)
  if (method === "oauth-personal" || method === "oauth-business") {
    if (platform === "win32") {
      const doomed = new Set(OAUTH_COLLIDING_ENV_KEYS.map((k) => k.toLowerCase()))
      for (const key of Object.keys(env)) {
        if (doomed.has(key.toLowerCase())) delete env[key]
      }
    } else {
      for (const key of OAUTH_COLLIDING_ENV_KEYS) {
        delete env[key]
      }
    }
  }
  return env
}

export interface AcpProcessHandlers {
  readonly onUpdate?: (sessionId: string, update: AcpSessionUpdate) => void
  readonly onPermissionRequest?: (
    params: AcpPermissionRequestParams,
  ) => Promise<AcpPermissionOutcome>
  readonly onElicitation?: (params: AcpElicitationRequestParams) => Promise<AcpElicitationResponse>
  readonly onExit?: (code: number | null, signal: NodeJS.Signals | null, stderrTail: string) => void
  readonly onLog?: (level: "warn" | "debug", message: string, meta?: Record<string, unknown>) => void
}

export interface AcpProcessOptions extends AcpProcessHandlers {
  readonly cwd: string
  readonly binaryPath?: string
  readonly processPerSession?: boolean
  readonly includeDefaultCustomizations?: boolean
  readonly dangerouslySkipPermissions?: boolean
  readonly extraDirs?: readonly string[]
  readonly traceLogPath?: string
  readonly io?: AgyIO
}

interface PendingRpc {
  readonly method: string
  readonly resolve: (value: unknown) => void
  readonly reject: (reason?: unknown) => void
  readonly timer?: ReturnType<typeof setTimeout>
  readonly signal?: AbortSignal
  readonly onAbort?: () => void
}

interface ActiveWatchdog {
  readonly id: number
  readonly idleMs: number
  readonly onStall: () => void
  timer?: ReturnType<typeof setTimeout>
}

export class AcpProcess {
  private readonly proc: ChildProcessWithoutNullStreams
  private readonly binary: string
  private readonly trace: WireTraceEntry[] = []
  private readonly pending = new Map<number, PendingRpc>()
  private readonly decoder = new StringDecoder("utf8")
  private readonly stderrDecoder = new StringDecoder("utf8")
  private readonly stderrLines: string[] = []
  private nextId = 1
  private buffer = ""
  private closed = false
  private exitNotified = false
  private watchdog?: ActiveWatchdog
  private watchdogPauseDepth = 0
  private handlers: AcpProcessHandlers
  private readonly sessionUpdateHandlers = new Map<string, (update: AcpSessionUpdate) => void>()

  constructor(private readonly options: AcpProcessOptions) {
    this.handlers = {
      onUpdate: options.onUpdate,
      onPermissionRequest: options.onPermissionRequest,
      onElicitation: options.onElicitation,
      onExit: options.onExit,
      onLog: options.onLog,
    }
    const io = options.io ?? defaultAgyIO
    const env = io.env()
    const args = ["acp", "--stdio"]
    if (options.processPerSession) {
      args.push("--process-per-session")
    }
    if (options.includeDefaultCustomizations === false) {
      args.push("--include-default-customizations=false")
    }
    if (options.dangerouslySkipPermissions) {
      args.push("--dangerously-skip-permissions")
    }
    for (const dir of options.extraDirs ?? []) {
      if (dir) args.push("--add-dir", dir)
    }

    const binary = io.resolveBinary(env, options.binaryPath)
    this.binary = binary
    this.recordTrace("sys", `spawn ${binary} ${args.join(" ")}`)

    this.proc = io.spawnProcess(binary, args, {
      cwd: options.cwd,
      stdio: ["pipe", "pipe", "pipe"],
      env: buildAgySpawnEnv(env),
      detached: process.platform !== "win32",
    }) as ChildProcessWithoutNullStreams

    this.proc.stdin.on("error", (err) => {
      this.closed = true
      this.recordTrace("err", `stdin error: ${err.message}`)
      this.failAllPending(err)
    })

    this.proc.stdout.on("error", () => {})
    this.proc.stderr.on("error", () => {})

    this.proc.stdout.on("data", (chunk: Buffer) => {
      this.buffer += this.decoder.write(chunk)
      if (this.buffer.length > MAX_STDOUT_BUFFER_BYTES) {
        this.buffer = ""
        this.recordTrace("err", `stdout buffer exceeded ${MAX_STDOUT_BUFFER_BYTES} bytes`)
        this.failAllPending(
          new Error(
            `ACP stdout buffer overflow: exceeded ${MAX_STDOUT_BUFFER_BYTES} bytes; killing process`,
          ),
        )
        this.killImmediate()
        return
      }
      let nl: number
      while ((nl = this.buffer.indexOf("\n")) !== -1) {
        const line = this.buffer.slice(0, nl).trim()
        this.buffer = this.buffer.slice(nl + 1)
        if (line.length > 0) this.handleLine(line)
      }
    })

    this.proc.stderr.on("data", (chunk: Buffer) => {
      const text = this.stderrDecoder.write(chunk)
      for (const line of text.split(/\r?\n/)) {
        if (!line.trim()) continue
        this.stderrLines.push(line)
        if (this.stderrLines.length > MAX_STDERR_LINES) this.stderrLines.shift()
        this.recordTrace("err", line.slice(0, 240))
      }
    })

    this.proc.on("error", (err) => {
      this.closed = true
      this.recordTrace("err", `spawn/process error: ${err.message}`)
      if ((err as NodeJS.ErrnoException)?.code === "ENOENT") {
        this.failAllPending(new AgyNotFoundError(binary))
        return
      }
      this.failAllPending(err)
    })

    const handleTermination = (code: number | null, signal: NodeJS.Signals | null) => {
      this.closed = true
      this.clearWatchdog()
      const tail = this.stderrLines.join("\n")
      this.recordTrace("sys", `process exited (code=${code}, signal=${signal})`)
      this.failAllPending(
        new Error(
          `agy acp exited unexpectedly (code=${code}, signal=${signal})${tail ? `\n${tail}` : ""}`,
        ),
      )
      this.notifyExit(code, signal, tail)
    }

    this.proc.on("exit", handleTermination)
    this.proc.on("close", handleTermination)
  }

  get isClosed(): boolean {
    return this.closed
  }

  get isWatchdogPaused(): boolean {
    return this.watchdogPauseDepth > 0
  }

  getRecentTrace(): readonly WireTraceEntry[] {
    return [...this.trace]
  }

  setHandlers(handlers: AcpProcessHandlers): void {
    this.handlers = { ...handlers }
  }

  registerSessionUpdateHandler(
    sessionId: string,
    handler: (update: AcpSessionUpdate) => void,
  ): () => void {
    this.sessionUpdateHandlers.set(sessionId, handler)
    return () => {
      if (this.sessionUpdateHandlers.get(sessionId) === handler) {
        this.sessionUpdateHandlers.delete(sessionId)
      }
    }
  }

  pauseWatchdog(): void {
    this.watchdogPauseDepth++
    if (this.watchdog?.timer) {
      clearTimeout(this.watchdog.timer)
      this.watchdog.timer = undefined
    }
  }

  resumeWatchdog(): void {
    if (this.watchdogPauseDepth === 0) return
    this.watchdogPauseDepth--
    if (this.watchdogPauseDepth === 0 && this.watchdog) {
      this.armWatchdogTimer(this.watchdog)
    }
  }

  private armWatchdogTimer(dog: ActiveWatchdog): void {
    if (dog.timer) clearTimeout(dog.timer)
    if (this.watchdogPauseDepth > 0 || dog.idleMs <= 0 || this.closed) {
      dog.timer = undefined
      return
    }
    dog.timer = setTimeout(() => {
      if (this.watchdog !== dog || this.watchdogPauseDepth > 0 || this.closed) return
      this.recordTrace("sys", `watchdog expired after ${dog.idleMs}ms of ACP silence`)
      dog.onStall()
    }, dog.idleMs)
    dog.timer.unref?.()
  }

  private bumpWatchdog(): void {
    if (!this.watchdog || this.watchdogPauseDepth > 0) return
    this.armWatchdogTimer(this.watchdog)
  }

  private clearWatchdog(id?: number): void {
    if (!this.watchdog) return
    if (id !== undefined && this.watchdog.id !== id) return
    if (this.watchdog.timer) clearTimeout(this.watchdog.timer)
    this.watchdog = undefined
  }

  private recordTrace(dir: WireTraceEntry["dir"], summary: string): void {
    const entry: WireTraceEntry = {
      ts: new Date().toISOString(),
      dir,
      summary: summary.slice(0, 320),
    }
    this.trace.push(entry)
    if (this.trace.length > MAX_TRACE_ENTRIES) {
      this.trace.shift()
    }
    if (this.options.traceLogPath) {
      void fs.promises
        .appendFile(this.options.traceLogPath, `${JSON.stringify(entry)}\n`, "utf8")
        .catch(() => {})
    }
  }

  private summarizeMessage(msg: Record<string, unknown>): string {
    const method = typeof msg.method === "string" ? msg.method : undefined
    const id = msg.id !== undefined ? `#${String(msg.id)}` : ""
    if (method === "session/update" && isRecord(msg.params) && isRecord(msg.params.update)) {
      const upd = msg.params.update
      const kind = typeof upd.sessionUpdate === "string" ? upd.sessionUpdate : "update"
      const status = typeof upd.status === "string" ? ` status=${upd.status}` : ""
      const name = typeof upd.name === "string" ? ` name=${upd.name}` : ""
      return `session/update(${kind}${name}${status})`
    }
    if ((method === "session/new" || method === "session/resume") && isRecord(msg.params)) {
      const meta = isRecord(msg.params._meta) ? msg.params._meta : undefined
      const agy = meta && isRecord(meta.antigravity) ? meta.antigravity : undefined
      const sysPrompt = agy && isRecord(agy.systemPrompt) ? agy.systemPrompt : undefined
      const append = sysPrompt && typeof sysPrompt.append === "string" ? sysPrompt.append : undefined
      const suffix = append ? ` appendChars=${append.length}` : ""
      return `${method}${id ? ` ${id}` : ""}${suffix}`
    }
    if (method) return `${method}${id ? ` ${id}` : ""}`
    if (msg.error !== undefined) return `response ${id} ERROR`
    return `response ${id} OK`
  }

  private notifyExit(code: number | null, signal: NodeJS.Signals | null, tail: string): void {
    if (this.exitNotified) return
    this.exitNotified = true
    this.handlers.onExit?.(code, signal, tail)
  }

  private signalProcess(sig: NodeJS.Signals): void {
    const pid = this.proc.pid
    if (pid === undefined) {
      try {
        this.proc.kill(sig)
      } catch {}
      return
    }
    if (process.platform === "win32") {
      if (sig === "SIGKILL") {
        try {
          execFile("taskkill", ["/PID", String(pid), "/T", "/F"], { windowsHide: true }, () => {})
          return
        } catch {}
      }
      try {
        this.proc.kill(sig)
      } catch {}
      return
    }
    try {
      process.kill(-pid, sig)
    } catch {
      try {
        this.proc.kill(sig)
      } catch {}
    }
  }

  private destroyStreams(): void {
    try {
      this.proc.stdin.end?.()
    } catch {}
    for (const stream of [this.proc.stdin, this.proc.stdout, this.proc.stderr] as const) {
      try {
        ;(stream as { destroy?: () => void }).destroy?.()
      } catch {}
    }
  }

  private failPending(id: number, err: Error): void {
    const entry = this.pending.get(id)
    if (!entry) return
    if (entry.timer) clearTimeout(entry.timer)
    if (entry.signal && entry.onAbort) {
      try {
        entry.signal.removeEventListener("abort", entry.onAbort)
      } catch {}
    }
    this.pending.delete(id)
    this.clearWatchdog(id)
    entry.reject(err)
  }

  private failAllPending(err: Error): void {
    for (const id of [...this.pending.keys()]) {
      this.failPending(id, err)
    }
  }

  private handleLine(line: string): void {
    let parsed: unknown
    try {
      parsed = JSON.parse(line)
    } catch {
      this.recordTrace("in", `malformed JSON: ${line.slice(0, 120)}`)
      this.handlers.onLog?.("warn", "Ignoring malformed ACP JSON line", {
        line: line.slice(0, 300),
      })
      return
    }
    if (!isRecord(parsed)) return

    this.bumpWatchdog()
    this.recordTrace("in", this.summarizeMessage(parsed))

    if (parsed.id !== undefined && parsed.method === undefined) {
      const id = Number(parsed.id)
      const entry = this.pending.get(id)
      if (!entry) {
        this.handlers.onLog?.("debug", "Ignoring late ACP response for removed request", {
          id: parsed.id,
        })
        return
      }
      if (entry.timer) clearTimeout(entry.timer)
      if (entry.signal && entry.onAbort) {
        try {
          entry.signal.removeEventListener("abort", entry.onAbort)
        } catch {}
      }
      this.pending.delete(id)
      this.clearWatchdog(id)
      if (parsed.error !== undefined) {
        const errObj = isRecord(parsed.error) ? parsed.error : undefined
        const code = typeof errObj?.code === "number" ? errObj.code : undefined
        const msg =
          typeof errObj?.message === "string"
            ? `${errObj.message}${code !== undefined ? ` (ACP code ${code})` : ""}`
            : JSON.stringify(parsed.error)
        entry.reject(new AcpRpcError(msg, code, errObj?.data))
      } else {
        entry.resolve(parsed.result)
      }
      return
    }

    if (parsed.id !== undefined && typeof parsed.method === "string") {
      void this.handleServerRequest(parsed.id as number | string, parsed.method, parsed.params)
      return
    }

    if (parsed.method === "session/update" && isRecord(parsed.params) && isRecord(parsed.params.update)) {
      const sid = String(parsed.params.sessionId ?? "")
      const upd = parsed.params.update as AcpSessionUpdate
      const scoped = this.sessionUpdateHandlers.get(sid)
      if (scoped) {
        scoped(upd)
      } else {
        this.handlers.onUpdate?.(sid, upd)
      }
    }
  }

  private async handleServerRequest(
    id: number | string,
    method: string,
    params: unknown,
  ): Promise<void> {
    if (method === "session/request_permission") {
      try {
        const permParams = (isRecord(params) ? params : {}) as AcpPermissionRequestParams
        const outcome: AcpPermissionOutcome = this.handlers.onPermissionRequest
          ? await this.handlers.onPermissionRequest(permParams)
          : fallbackAutoSelectPermission(permParams)
        this.write({ jsonrpc: "2.0", id, result: { outcome } })
      } catch (err) {
        this.handlers.onLog?.("warn", "onPermissionRequest failed; cancelling", {
          error: err instanceof Error ? err.message : String(err),
        })
        this.write({
          jsonrpc: "2.0",
          id,
          result: { outcome: { outcome: "cancelled" } },
        })
      }
      return
    }

    if (method === "elicitation/create") {
      let result: AcpElicitationResponse
      try {
        const elicitParams = (isRecord(params) ? params : {}) as AcpElicitationRequestParams
        result = this.handlers.onElicitation
          ? await this.handlers.onElicitation(elicitParams)
          : { action: "decline" }
      } catch (err) {
        this.handlers.onLog?.("warn", "onElicitation failed; cancelling", {
          error: err instanceof Error ? err.message : String(err),
        })
        result = { action: "cancel" }
      }
      this.write({ jsonrpc: "2.0", id, result })
      return
    }

    this.write({
      jsonrpc: "2.0",
      id,
      error: { code: -32601, message: `Unsupported client method: ${method}` },
    })
  }

  private write(payload: Record<string, unknown>, onAsyncError?: (err: Error) => void): boolean {
    if (
      this.closed ||
      this.proc.exitCode !== null ||
      this.proc.signalCode !== null ||
      !this.proc.stdin.writable ||
      this.proc.stdin.destroyed
    ) {
      return false
    }
    this.recordTrace("out", this.summarizeMessage(payload))
    try {
      this.proc.stdin.write(`${JSON.stringify(payload)}\n`, (err) => {
        if (err) {
          this.closed = true
          this.recordTrace("err", `stdin write callback error: ${err.message}`)
          onAsyncError?.(err)
        }
      })
      return true
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))
      this.closed = true
      this.recordTrace("err", `stdin write threw: ${error.message}`)
      onAsyncError?.(error)
      return false
    }
  }

  request<T = unknown>(
    method: string,
    params: unknown,
    timeoutMs?: number,
    signal?: AbortSignal,
    idleWatchdogMs?: number,
  ): Promise<T> {
    if (this.closed) {
      return Promise.reject(new Error(`Cannot call ${method}: agy acp process is closed`))
    }
    if (signal?.aborted) {
      return Promise.reject(
        signal.reason instanceof Error ? signal.reason : new Error(`ACP request ${method} aborted`),
      )
    }
    const id = this.nextId++
    return new Promise<T>((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | undefined
      const onAbort = () => {
        if (!this.pending.has(id)) return
        this.failPending(
          id,
          signal?.reason instanceof Error
            ? signal.reason
            : new Error(`ACP request ${method} aborted`),
        )
      }
      if (timeoutMs && timeoutMs > 0) {
        timer = setTimeout(() => {
          const err = new Error(`ACP request ${method} timed out after ${timeoutMs}ms`)
          this.recordTrace("sys", err.message)
          this.failPending(id, err)
        }, timeoutMs)
        timer.unref?.()
      }

      this.pending.set(id, {
        method,
        resolve: (value) => resolve(value as T),
        reject,
        timer,
        signal,
        onAbort,
      })

      if (idleWatchdogMs && idleWatchdogMs > 0) {
        const dog: ActiveWatchdog = {
          id,
          idleMs: idleWatchdogMs,
          onStall: () => {
            this.failPending(id, new AcpStallError(idleWatchdogMs, this.getRecentTrace()))
          },
        }
        this.watchdog = dog
        this.armWatchdogTimer(dog)
      }

      if (signal) {
        try {
          signal.addEventListener("abort", onAbort, { once: true })
        } catch {}
      }

      const ok = this.write({ jsonrpc: "2.0", id, method, params }, (writeErr) => {
        this.failPending(id, writeErr)
      })
      if (!ok) {
        const err =
          this.proc.pid === undefined
            ? new AgyNotFoundError(this.binary)
            : new Error(`Cannot write ${method}: agy acp pipe is closed`)
        this.failPending(id, err)
      }
    })
  }

  notify(method: string, params: unknown): void {
    this.write({ jsonrpc: "2.0", method, params })
  }

  initialize(timeoutMs = 15_000): Promise<unknown> {
    return this.request(
      "initialize",
      {
        protocolVersion: 1,
        clientCapabilities: { elicitation: { form: {} } },
        clientInfo: { name: "opencode-agy-acp", version: "0.1.0" },
      },
      timeoutMs,
    )
  }

  async authenticate(timeoutMs = 10_000): Promise<unknown> {
    const methodId = resolveAcpAuthMethod(this.options.io?.env() ?? process.env)
    try {
      return await this.request("authenticate", { methodId }, timeoutMs)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      if (/Method not found|Unsupported client method|-32601/.test(msg)) return undefined
      if (/already (authenticated|logged in|signed in)/i.test(msg)) return undefined
      throw err
    }
  }

  newSession(params: {
    readonly cwd: string
    readonly mcpServers?: readonly AcpMcpServerHttp[]
    readonly systemPromptAppend?: string
    readonly timeoutMs?: number
  }): Promise<AcpSessionResult> {
    return this.request<AcpSessionResult>(
      "session/new",
      {
        cwd: params.cwd,
        mcpServers: params.mcpServers ? [...params.mcpServers] : [],
        ...(params.systemPromptAppend
          ? {
              _meta: {
                antigravity: {
                  systemPrompt: {
                    append: params.systemPromptAppend,
                  },
                },
              },
            }
          : {}),
      },
      params.timeoutMs ?? 30_000,
    )
  }

  resumeSession(params: {
    readonly sessionId: string
    readonly cwd: string
    readonly mcpServers?: readonly AcpMcpServerHttp[]
    readonly systemPromptAppend?: string
    readonly timeoutMs?: number
  }): Promise<AcpSessionResult> {
    return this.request<AcpSessionResult>(
      "session/resume",
      {
        sessionId: params.sessionId,
        cwd: params.cwd,
        mcpServers: params.mcpServers ? [...params.mcpServers] : [],
        ...(params.systemPromptAppend
          ? {
              _meta: {
                antigravity: {
                  systemPrompt: {
                    append: params.systemPromptAppend,
                  },
                },
              },
            }
          : {}),
      },
      params.timeoutMs ?? 20_000,
    )
  }

  setModel(sessionId: string, modelId: string, timeoutMs = 10_000): Promise<unknown> {
    return this.request("session/set_model", { sessionId, modelId }, timeoutMs)
  }

  setMode(sessionId: string, modeId: string, timeoutMs = 10_000): Promise<unknown> {
    return this.request("session/set_mode", { sessionId, modeId }, timeoutMs)
  }

  setConfigOption(
    sessionId: string,
    configId: string,
    value: string,
    timeoutMs = 10_000,
  ): Promise<unknown> {
    return this.request("session/set_config_option", { sessionId, configId, value }, timeoutMs)
  }

  prompt(
    sessionId: string,
    prompt: readonly AcpContentBlock[],
    options?: {
      readonly signal?: AbortSignal
      readonly timeoutMs?: number
      readonly idleWatchdogMs?: number
    },
  ): Promise<AcpPromptResult> {
    return this.request<AcpPromptResult>(
      "session/prompt",
      { sessionId, prompt },
      options?.timeoutMs ?? 0,
      options?.signal,
      options?.idleWatchdogMs,
    )
  }

  notifyCancel(sessionId: string): void {
    this.notify("session/cancel", { sessionId })
  }

  cancel(sessionId: string): Promise<boolean> {
    const ok = this.write({ jsonrpc: "2.0", method: "session/cancel", params: { sessionId } })
    return Promise.resolve(ok)
  }

  closeSession(sessionId: string): Promise<unknown> {
    return this.request("session/close", { sessionId }, 5_000).catch(() => undefined)
  }

  killImmediate(): void {
    this.clearWatchdog()
    if (this.proc.exitCode !== null || this.proc.signalCode !== null) {
      this.closed = true
      this.failAllPending(new Error("AcpProcess halted immediately"))
      this.destroyStreams()
      return
    }
    if (this.closed) return
    this.closed = true
    this.failAllPending(new Error("AcpProcess halted immediately"))
    this.signalProcess("SIGTERM")
    const timer = setTimeout(() => {
      this.signalProcess("SIGKILL")
    }, 500)
    timer.unref?.()
    this.proc.once("exit", () => {
      clearTimeout(timer)
      this.destroyStreams()
    })
  }

  async dispose(): Promise<void> {
    this.clearWatchdog()
    if (this.proc.exitCode !== null || this.proc.signalCode !== null) {
      this.closed = true
      this.failAllPending(new Error("AcpProcess disposed"))
      this.destroyStreams()
      return
    }
    if (this.closed) return
    this.closed = true
    this.failAllPending(new Error("AcpProcess disposed"))
    try {
      this.proc.stdin.end()
    } catch {}
    this.signalProcess("SIGTERM")
    await new Promise<void>((resolve) => {
      let settled = false
      const done = () => {
        if (settled) return
        settled = true
        clearTimeout(timer)
        this.destroyStreams()
        resolve()
      }
      const timer = setTimeout(() => {
        this.signalProcess("SIGKILL")
        done()
      }, 1_500)
      timer.unref?.()
      this.proc.once("exit", done)
      this.proc.once("close", done)
    })
  }
}

function fallbackAutoSelectPermission(params: AcpPermissionRequestParams): AcpPermissionOutcome {
  const opts = params.options ?? []
  const allow = opts.find((opt) => {
    const kind = (opt.kind ?? "").toLowerCase()
    const id = (opt.optionId ?? "").toLowerCase()
    return kind.includes("allow") || id.includes("allow") || id.includes("approve")
  })
  if (allow) return { outcome: "selected", optionId: allow.optionId }
  if (opts[0]) return { outcome: "selected", optionId: opts[0].optionId }
  return { outcome: "cancelled" }
}
