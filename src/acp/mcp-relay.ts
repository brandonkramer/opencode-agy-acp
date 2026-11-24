import crypto from "node:crypto"
import http from "node:http"
import { NATIVE_SHADOWED_TOOLS, PROVIDER_ID, isRecord } from "../types.js"
import { isFullCodeModeExecuteDescription, rewriteCodeModeExecuteRefs } from "./instructions.js"

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
        "PRIMARY TOOL (OpenCode Full Code Mode): PREFER `mcp_oc_execute` over native `read_file`, `write_file`, `edit_file`, `grep_search`, `find_by_name`, and `run_command` for creating, reading, searching, or editing files (especially multiple files), running shell commands, OpenCode MCP tools, and `tools.opencode.*`. Batch work into ONE JavaScript script and return a concise summary.",
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
      "PREFER THIS TOOL over `run_command` or sequential tool calls whenever calling OpenCode MCP tools, `tools.opencode.*`, batching multiple calls (`Promise.all`), or filtering/transforming large outputs in JavaScript before returning a concise result.",
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
          result = await this.handlers.callTool(
            sessionID,
            String(params.name ?? ""),
            params.arguments ?? {},
            ac.signal,
          )
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
