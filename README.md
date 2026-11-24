# opencode-agy-acp

A lean, deadlock-free OpenCode plugin for Google Antigravity CLI (`agy acp`), modeled on OpenCode's built-in `claude-agent` architecture.

## Design & Features

1. **Stream-idle watchdog (`chunkTimeoutMs`, default 120s):** Rejects wedged `session/prompt` calls with `AcpStallError` + the last 8 wire-trace entries, cancels and kills the hung process group, and resets the session cleanly instead of hanging forever. Automatically pauses while OpenCode runs relayed tools, `question`, or permission prompts.
2. **Deadlock-free tool & elicitation handling:**
   - **Pure ACP by default** (`codeMode: false`) so `agy` runs natively with zero MCP relay overhead and no `--process-per-session` hop.
   - When `codeMode: true`, relayed MCP calls never deadlock on `activeNativeTools` (reclassification on `mcp_oc_` / `oc_` / `oc:` / relayed names + bounded `nativeFlushGraceMs`, default 2s).
   - Full `elicitation/create` → OpenCode `question` bridge works in both Pure ACP and Code Mode.
3. **Structured Effect concurrency:** Per-session `Semaphore(1)` serialization, 5-minute idle worker reaper (`idleTimeoutMs`, preserving `acpSessionId` for `session/resume`), and `retry: false` hook to prevent double-retries.
4. **MCP isolation (`includeDefaultCustomizations: false` by default):** Prevents each `agy acp` worker from spawning duplicate copies of `~/.gemini` MCP servers.
5. **Bounded permissions & fixed `stdin.write` error propagation.**

## Configuration (`opencode.json`)

```json
{
  "plugins": ["/path/to/opencode-agy-acp"],
  "providers": {
    "antigravity-acp": {
      "settings": {
        "codeMode": false,
        "includeDefaultCustomizations": false,
        "chunkTimeoutMs": 120000,
        "permissionTimeoutMs": 300000,
        "idleTimeoutMs": 300000
      }
    }
  }
}
```

### Environment overrides

- `AGY_BIN`: explicit path to `agy`
- `AGY_ACP_CODE_MODE=true|false`
- `AGY_ACP_INCLUDE_DEFAULT_CUSTOMIZATIONS=true|false`
- `AGY_ACP_CHUNK_TIMEOUT_MS`
- `AGY_ACP_TRACE_LOG=/path/to/trace.jsonl`

## Development

```sh
bun install
bun run typecheck
bun test
```
