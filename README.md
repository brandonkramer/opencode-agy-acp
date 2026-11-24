# opencode-agy-acp

A lean, deadlock-free OpenCode plugin for Google Antigravity CLI (`agy acp`), modeled on OpenCode's built-in `claude-agent` architecture.

## Features

1. **Stream-idle watchdog (`chunkTimeoutMs`, default 120s):** Rejects wedged `session/prompt` calls with `AcpStallError` + the last 8 wire-trace entries, cancels and kills the hung process group, and resets the session cleanly instead of hanging forever. Automatically pauses while OpenCode runs relayed tools, `question`, or permission prompts.
2. **Deadlock-free tool & elicitation handling:**
   - **Pure ACP by default** (`codeMode: false`) so `agy` runs natively with zero MCP relay overhead and no `--process-per-session` hop.
   - When `codeMode: true`, relayed MCP calls never deadlock on `activeNativeTools` (reclassification on `mcp_oc_` / `oc_` / `oc:` / relayed names + bounded `nativeFlushGraceMs`, default 2s).
   - Full `elicitation/create` → OpenCode `question` bridge works in both Pure ACP and Code Mode.
3. **Structured Effect concurrency:** Per-session `Semaphore(1)` serialization, 5-minute idle worker reaper (`idleTimeoutMs`, preserving `acpSessionId` for `session/resume`), and `retry: false` hook to prevent double-retries.
4. **MCP isolation (`includeDefaultCustomizations: false` by default):** Prevents each `agy acp` worker from spawning duplicate copies of `~/.gemini` MCP servers.
5. **Bounded permissions & fixed `stdin.write` error propagation.**
6. **Plan mode auto-sync (`defaultMode`, default `"auto"`):** Automatically switches `agy` to `"plan"` when using OpenCode's `plan` agent and back when leaving `plan` (unless explicitly overridden via RPC `setMode`).
7. **Instruction & Code Mode forwarding (`forwardInstructions`, default `true`):** Forwards OpenCode agent roles, `AGENTS.md`, skills, and Code Mode instructions via ACP `_meta.antigravity.systemPrompt.append` while keeping OpenCode's core system prompt lean.
8. **Turn fingerprinting & delta catch-up:** Tracks completed turns so cross-model switches append only compact `<conversation-checkpoint>` deltas to the live ACP session, while `/undo` or edits cleanly start a restored session.
9. **Side-call isolation:** Runs compaction and auxiliary generate calls on throwaway ACP sessions without mutating primary session state.

## Configuration (`opencode.json`)

```json
{
  "plugins": ["/path/to/opencode-agy-acp"],
  "providers": {
    "antigravity-acp": {
      "settings": {
        "defaultMode": "auto",
        "forwardInstructions": true,
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
- `AGY_ACP_DEFAULT_MODE=auto|accept-edits|default|plan`
- `AGY_ACP_FORWARD_INSTRUCTIONS=true|false`
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
