import crypto from "node:crypto"
import fs from "node:fs"
import http from "node:http"
import os from "node:os"
import path from "node:path"
import { NATIVE_SHADOWED_TOOLS, PROVIDER_ID, isRecord } from "../types.js"
import { isFullCodeModeExecuteDescription, rewriteCodeModeExecuteRefs } from "./instructions.js"

const DEFAULT_MAX_MCP_RESULT_CHARS = 9_800
const SPILL_MAX_AGE_MS = 24 * 60 * 60 * 1000
const SPILL_PRUNE_INTERVAL_MS = 60 * 60 * 1000
let lastSpillPruneAt = 0

const EXISTING_SAVED_FOOTER_RE =
  /\n?(\[(?:showing lines [^\]\[\n]*?; full output saved to [^\]\[\n]+|compacted [^\]\[\n]*?; full log: [^\]\[\n]+|[^\]\[\n]*(?:full output saved to|output truncated[^\]\[\n]*saved to) [^\]\[\n]+)\])\s*$/i
const MAX_EXISTING_FOOTER_CHARS = 512

function safeSliceStart(text: string, len: number): string {
  if (len <= 0) return ""
  let end = Math.min(text.length, len)
  if (end > 0 && end < text.length) {
    const prev = text.charCodeAt(end - 1)
    if (prev >= 0xd800 && prev <= 0xdbff) end--
  }
  return text.slice(0, end)
}

function safeSliceEnd(text: string, len: number): string {
  if (len <= 0) return ""
  let start = Math.max(0, text.length - len)
  if (start > 0 && start < text.length) {
    const ch = text.charCodeAt(start)
    if (ch >= 0xdc00 && ch <= 0xdfff) start++
  }
  return text.slice(start)
}

function spillOversizedOutput(raw: string, spillDir?: string): string | undefined {
  const dir = spillDir ?? path.join(os.tmpdir(), "opencode-agy-acp-outputs")
  try {
    fs.mkdirSync(dir, { recursive: true, mode: 0o700 })
    const now = Date.now()
    if (now - lastSpillPruneAt >= SPILL_PRUNE_INTERVAL_MS) {
      lastSpillPruneAt = now
      try {
        for (const name of fs.readdirSync(dir)) {
          if (!name.startsWith("mcp-")) continue
          const full = path.join(dir, name)
          try {
            const st = fs.statSync(full)
            if (now - st.mtimeMs > SPILL_MAX_AGE_MS) fs.unlinkSync(full)
          } catch {}
        }
      } catch {}
    }
    const stamp = new Date(now).toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z")
    const rand = crypto.randomBytes(4).toString("hex")
    const filePath = path.join(dir, `mcp-${stamp}-${rand}.txt`)
    fs.writeFileSync(filePath, raw, { encoding: "utf8", mode: 0o600 })
    return filePath
  } catch {
    return undefined
  }
}

/**
 * Keep relayed MCP text results safely below agy's inline truncation threshold so
 * agy never double-truncates OpenCode output into a broken steps/N/output.txt file,
 * while preserving OpenCode's saved-file pointer (or spilling full raw text to disk).
 */
export function compactMcpText(
  raw: string,
  options?: { readonly maxChars?: number; readonly spillDir?: string },
): string {
  const maxChars = options?.maxChars ?? DEFAULT_MAX_MCP_RESULT_CHARS
  if (maxChars <= 0 || raw.length <= maxChars) return raw

  const footerMatch = EXISTING_SAVED_FOOTER_RE.exec(raw)
  const existingFooter =
    footerMatch?.[1] && footerMatch[1].length <= MAX_EXISTING_FOOTER_CHARS
      ? footerMatch[1]
      : undefined
  const body = existingFooter && footerMatch ? raw.slice(0, footerMatch.index) : raw

  let footer: string
  if (existingFooter) {
    footer = `\n${existingFooter}\n[relayed output compacted from ${raw.length} chars; read the full file above with Bun.file() — do not JSON.parse .system_generated/steps/*/output.txt]`
  } else {
    const spillPath = spillOversizedOutput(raw, options?.spillDir)
    footer = spillPath
      ? `\n[relayed output compacted from ${raw.length} chars; full raw output saved to ${spillPath} — read/filter with Bun.file() if needed]`
      : `\n[relayed output compacted from ${raw.length} chars]`
  }

  const markerTemplate = (omitted: number) => `\n... [${omitted} chars omitted] ...\n`
  const reservedMarkerLen = markerTemplate(raw.length).length
  const budget = maxChars - footer.length - reservedMarkerLen
  if (budget <= 120) {
    const fallbackHead = safeSliceStart(body, Math.max(40, maxChars - footer.length - 16))
    return `${fallbackHead}\n...[compacted]${footer}`.slice(0, Math.max(maxChars, footer.length + 64))
  }

  const headBudget = Math.floor(budget * 0.7)
  const tailBudget = budget - headBudget
  const head = safeSliceStart(body, headBudget)
  const tail = safeSliceEnd(body, tailBudget)
  const omitted = Math.max(0, body.length - head.length - tail.length)
  if (omitted === 0) {
    return `${body}${footer}`
  }
  return `${head}${markerTemplate(omitted)}${tail}${footer}`
}

export function compactMcpToolResult(
  result: McpCallToolResult,
  options?: { readonly maxChars?: number; readonly spillDir?: string },
): McpCallToolResult {
  const maxChars = options?.maxChars ?? DEFAULT_MAX_MCP_RESULT_CHARS
  if (maxChars <= 0) return result
  let changed = false
  const nextContent = result.content.map((item) => {
    if (item.type !== "text" || item.text.length <= maxChars) return item
    changed = true
    return {
      type: "text" as const,
      text: compactMcpText(item.text, options),
    }
  })
  return changed ? { ...result, content: nextContent } : result
}

export interface McpToolDescriptor {
  readonly name: string
  readonly description: string
  readonly inputSchema: Record<string, unknown>
}

export type McpContentItem =
  | { readonly type: "text"; readonly text: string }
  | { readonly type: "image"; readonly data: string; readonly mimeType: string }

export interface McpCallToolResult {
  readonly content: ReadonlyArray<McpContentItem>
  readonly isError?: boolean
}

export interface McpRelayHandlers {
  readonly getTools: (sessionID: string) => McpToolDescriptor[]
  readonly getMaxResultChars?: () => number
  readonly callTool: (
    sessionID: string,
    toolName: string,
    args: unknown,
    signal: AbortSignal,
  ) => Promise<McpCallToolResult>
}

export function normalizeMcpInputSchema(schema: unknown): Record<string, unknown> {
  if (!isRecord(schema)) {
    return { type: "object", properties: {} }
  }
  const copy: Record<string, unknown> = { ...schema }
  if (!copy.type && copy.properties) {
    copy.type = "object"
  }
  if (copy.type === "object" && !copy.properties) {
    copy.properties = {}
  }
  delete copy.$schema
  return copy
}

function sanitizeMcpToolName(name: string, used: Set<string>): string {
  let base = name.replace(/[^a-zA-Z0-9_-]/g, "_")
  if (!base) base = "tool"
  let candidate = base
  let n = 2
  while (used.has(candidate)) {
    candidate = `${base}_${n++}`
  }
  used.add(candidate)
  return candidate
}

function enrichMcpToolDescription(toolName: string, rawDescription: string): string {
  const base = rawDescription.trim() || `OpenCode tool: ${toolName}`
  if (toolName === "subagent") {
    return `${base}\n\nOpenCode agents include "explore" (read-only search/review), "general", "build", and "plan". Optional \`model\` uses "<provider>/<modelID>".`
  }
  if (toolName === "execute") {
    const normalizedBase = rewriteCodeModeExecuteRefs(base)
    if (isFullCodeModeExecuteDescription(base)) {
      return [
        "PRIMARY TOOL (OpenCode Full Code Mode): PREFER `mcp_oc_execute` over native `read_file`, `write_file`, `edit_file`, `grep_search`, `find_by_name`, and `run_command` for creating, reading, searching, or editing files (especially multiple files), running shell commands (including background watchers and session-aware CLI helpers that expect `OPENCODE_SESSION_ID`), OpenCode MCP tools, and `tools.opencode.*`. Batch independent operations in ONE script with `Promise.all`, prefer `Bun.file`/`Bun.Glob`/`Bun.write` over `tools.shell` for `ls`/`cat`/`find`/`grep` (note: `tools.shell` tail-truncates stdout > ~8-10 KB, so redirect large CLI/API outputs to a temp file and read with `Bun.file()`), parse & project CLI/JSON outputs inside JS instead of returning raw `.output` strings (avoid double-JSON-encoding), return concise filtered summaries (< 4-6 KB) to avoid follow-up file reads, and never `JSON.parse()` truncated `.system_generated/steps/*/output.txt`.",
        "",
        normalizedBase,
        "",
        "Callable in Antigravity ACP as `mcp_oc_execute`. Example:",
        'await Promise.all([Bun.write("a.txt", "hello\\n"), Bun.write("b.txt", "world\\n")]);',
        'const text = await Bun.file("a.txt").text();',
        "return { written: 2, preview: text.trim() };",
      ].join("\n")
    }
    return [
      "PREFER THIS TOOL over `run_command` or sequential tool calls whenever calling OpenCode MCP tools, `tools.opencode.*`, batching multiple calls (`Promise.all`), or filtering/transforming large outputs in JavaScript before returning a concise result (< 4-6 KB). Never `JSON.parse()` truncated `.system_generated/steps/*/output.txt` or large `tools.shell` stdout directly.",
      "",
      normalizedBase,
      "",
      "Callable in Antigravity ACP as `mcp_oc_execute`. Example:",
      'const matches = await search({ query: "..." });',
      "return matches.items.map((t) => t.path);",
    ].join("\n")
  }
  return base
}

export function buildToolDescriptors(
  tools: Record<string, { readonly description: string; readonly input: unknown }>,
  options?: {
    readonly extraRelayedTools?: readonly string[]
    readonly blockedRelayedTools?: readonly string[]
  },
): { mcpDescriptors: McpToolDescriptor[]; mcpSafeToOriginal: Map<string, string> } {
  const used = new Set<string>()
  const mcpSafeToOriginal = new Map<string, string>()
  const mcpDescriptors: McpToolDescriptor[] = []
  const extra = new Set(options?.extraRelayedTools ?? [])
  const blocked = new Set(options?.blockedRelayedTools ?? [])

  for (const [toolName, info] of Object.entries(tools)) {
    if (toolName === PROVIDER_ID) continue
    if (blocked.has(toolName)) continue
    if (NATIVE_SHADOWED_TOOLS.has(toolName) && !extra.has(toolName)) continue
    const safeName = sanitizeMcpToolName(toolName, used)
    mcpSafeToOriginal.set(safeName, toolName)
    mcpDescriptors.push({
      name: safeName,
      description: enrichMcpToolDescription(toolName, info.description || ""),
      inputSchema: normalizeMcpInputSchema(info.input),
    })
  }
  return { mcpDescriptors, mcpSafeToOriginal }
}

type HttpServerWithConnections = http.Server & {
  closeIdleConnections?: () => void
  closeAllConnections?: () => void
}

export class McpRelayServer {
  readonly token: string
  private server?: HttpServerWithConnections
  private port = 0
  private started = false
  private generation = 0
  private startPromise?: Promise<void>

  constructor(private readonly handlers: McpRelayHandlers) {
    this.token = crypto.randomBytes(24).toString("hex")
  }

  async start(): Promise<void> {
    if (this.started && this.port > 0) return
    if (this.startPromise) return this.startPromise

    const gen = ++this.generation
    const server: HttpServerWithConnections = http.createServer((req, res) => {
      void this.handleHttp(req, res)
    })
    // Allow long-running OpenCode Code Mode / subagent tool calls without HTTP socket drops.
    server.timeout = 0
    server.requestTimeout = 0
    server.keepAliveTimeout = 0
    this.server = server

    this.startPromise = new Promise<void>((resolve, reject) => {
      let settled = false
      const cleanup = () => {
        server.off("error", onError)
        server.off("close", onClose)
      }
      const closeServer = () => {
        try {
          server.closeAllConnections?.()
          server.close()
        } catch {}
      }
      const onError = (err: Error) => {
        if (settled) return
        settled = true
        cleanup()
        this.startPromise = undefined
        if (this.server === server) this.server = undefined
        closeServer()
        reject(err)
      }
      const onClose = () => {
        if (settled) return
        settled = true
        cleanup()
        this.startPromise = undefined
        if (this.server === server) this.server = undefined
        reject(new Error("McpRelayServer stopped during startup"))
      }
      server.once("error", onError)
      server.once("close", onClose)
      server.listen(0, "127.0.0.1", () => {
        if (settled) return
        settled = true
        cleanup()
        this.startPromise = undefined

        if (this.generation !== gen) {
          if (this.server === server) this.server = undefined
          closeServer()
          reject(new Error("McpRelayServer stopped during startup"))
          return
        }

        const addr = server.address()
        if (!addr || typeof addr === "string") {
          if (this.server === server) this.server = undefined
          closeServer()
          reject(new Error("McpRelayServer failed to bind an ephemeral TCP port"))
          return
        }

        this.port = addr.port
        this.started = true
        resolve()
      })
    })
    return this.startPromise
  }

  urlForSession(sessionID: string): string {
    if (!this.started || !this.port) {
      throw new Error("McpRelayServer has not started")
    }
    return `http://127.0.0.1:${this.port}/mcp/${encodeURIComponent(sessionID)}`
  }

  authHeaders(): ReadonlyArray<{ readonly name: string; readonly value: string }> {
    return [{ name: "Authorization", value: `Bearer ${this.token}` }]
  }

  async stop(): Promise<void> {
    this.generation++
    const server = this.server
    this.started = false
    this.port = 0
    this.startPromise = undefined
    this.server = undefined
    if (!server) return

    try {
      server.closeIdleConnections?.()
      server.closeAllConnections?.()
    } catch {}

    await new Promise<void>((resolve) => {
      let done = false
      const finish = () => {
        if (done) return
        done = true
        clearTimeout(timer)
        resolve()
      }
      const timer = setTimeout(finish, 1_500)
      timer.unref?.()
      try {
        server.close(() => finish())
      } catch {
        finish()
      }
    })
  }

  private async handleHttp(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    try {
      const auth = req.headers.authorization
      if (auth !== `Bearer ${this.token}`) {
        res.writeHead(401, { "Content-Type": "application/json" })
        res.end(JSON.stringify({ error: "Unauthorized" }))
        return
      }

      const rawUrl = req.url ?? "/"
      const pathname = new URL(rawUrl, "http://127.0.0.1").pathname
      const match = /^\/mcp\/([^/]+)$/.exec(pathname)
      if (!match) {
        res.writeHead(404, { "Content-Type": "application/json" })
        res.end(JSON.stringify({ error: "Not found" }))
        return
      }
      const sessionID = decodeURIComponent(match[1])

      if (req.method === "DELETE" || req.method === "GET") {
        res.writeHead(req.method === "DELETE" ? 200 : 405, { "Content-Type": "application/json" })
        res.end("{}")
        return
      }

      if (req.method !== "POST") {
        res.writeHead(405, { "Content-Type": "application/json" })
        res.end(JSON.stringify({ error: "Method not allowed" }))
        return
      }

      const chunks: Buffer[] = []
      for await (const chunk of req) {
        chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : (chunk as Buffer))
      }
      const rawBody = Buffer.concat(chunks).toString("utf8")
      let parsed: unknown
      try {
        parsed = JSON.parse(rawBody)
      } catch {
        res.writeHead(400, { "Content-Type": "application/json" })
        res.end(
          JSON.stringify({
            jsonrpc: "2.0",
            id: null,
            error: { code: -32700, message: "Parse error" },
          }),
        )
        return
      }
      if (!isRecord(parsed)) {
        res.writeHead(400, { "Content-Type": "application/json" })
        res.end(
          JSON.stringify({
            jsonrpc: "2.0",
            id: null,
            error: { code: -32600, message: "Invalid Request" },
          }),
        )
        return
      }

      const id = parsed.id
      const method = String(parsed.method ?? "")
      const params = isRecord(parsed.params) ? parsed.params : {}

      if (id === undefined || id === null) {
        res.writeHead(202)
        res.end()
        return
      }

      let result: unknown
      if (method === "server/discover" || method === "ping") {
        result = {}
      } else if (method === "initialize") {
        result = {
          protocolVersion:
            typeof params.protocolVersion === "string" ? params.protocolVersion : "2025-03-26",
          capabilities: { tools: { listChanged: true } },
          serverInfo: {
            name: "opencode-relay",
            title: "OpenCode Tool Relay",
            version: "0.1.0",
          },
          instructions:
            "OpenCode workspace tool relay. Use `execute` to run JavaScript in OpenCode Code Mode (`tools.<namespace>.<tool>(...)` and `search(...)` for hidden MCP/workspace namespaces), and `subagent` to launch OpenCode subagents.",
        }
      } else if (method === "tools/list") {
        const tools = this.handlers.getTools(sessionID).map((t) => ({
          name: t.name,
          description: (t.description || "").trim() || `Tool: ${t.name}`,
          inputSchema: normalizeMcpInputSchema(t.inputSchema),
        }))
        result = { tools }
      } else if (method === "tools/call") {
        const ac = new AbortController()
        const onClose = () => {
          if (!res.writableFinished) ac.abort()
        }
        res.on("close", onClose)
        try {
          const rawResult = await this.handlers.callTool(
            sessionID,
            String(params.name ?? ""),
            params.arguments ?? {},
            ac.signal,
          )
          const maxChars = this.handlers.getMaxResultChars?.()
          result = compactMcpToolResult(rawResult, { maxChars })
        } finally {
          res.off("close", onClose)
        }
      } else {
        this.sendJsonRpc(res, {
          jsonrpc: "2.0",
          id,
          error: { code: -32601, message: `Method not found: ${method}` },
        })
        return
      }

      if (!res.writableEnded) {
        this.sendJsonRpc(res, { jsonrpc: "2.0", id, result })
      }
    } catch (err) {
      if (!res.headersSent) {
        res.writeHead(500, { "Content-Type": "application/json" })
      }
      if (!res.writableEnded) {
        res.end(
          JSON.stringify({
            jsonrpc: "2.0",
            id: null,
            error: {
              code: -32603,
              message: err instanceof Error ? err.message : String(err),
            },
          }),
        )
      }
    }
  }

  private sendJsonRpc(res: http.ServerResponse, payload: unknown): void {
    const body = Buffer.from(JSON.stringify(payload), "utf8")
    res.writeHead(200, {
      "Content-Type": "application/json",
      "Content-Length": String(body.byteLength),
    })
    res.end(body)
  }
}
