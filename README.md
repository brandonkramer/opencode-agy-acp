# opencode-agy-acp

A lean, deadlock-free OpenCode plugin for Google Antigravity CLI (`agy acp`), modeled on OpenCode's built-in `claude-agent` architecture.

## Features

1. **Stream-idle watchdog (`chunkTimeoutMs`, default 120s):** Rejects wedged `session/prompt` calls with `AcpStallError` + the last 8 wire-trace entries, cancels and kills the hung process group, and resets the session cleanly instead of hanging forever. Automatically pauses while OpenCode runs relayed tools, `question`, or permission prompts.
2. **Deadlock-free tool & elicitation handling:**
   - **Hybrid Auto Code Mode by default (`codeMode: "auto"`):** Keeps native `agy` tools fast for local edits/shell (`NATIVE_SHADOWED_TOOLS`) while automatically relaying OpenCode's non-shadowed tools (`execute`, `subagent`, MCP servers) without `--process-per-session` overhead. Supports `extraRelayedTools` / `blockedRelayedTools` overrides or forcing `true` / `false`.
   - **`mcp_oc_execute` & rich media bridging:** Guides `agy` from OpenCode's `execute` docs directly to `mcp_oc_execute`, streams image/file attachments from OpenCode tool results over MCP, and avoids false deadlocks on `activeNativeTools` (`nativeFlushGraceMs`, default 400ms).
   - Full `elicitation/create` → OpenCode `question` bridge works in both Pure ACP and Code Mode.
3. **Structured Effect concurrency & active-only warm worker pooling:**
   - Active-directory standby pre-warming (`prewarm: "auto"` by default — only prewarms workspaces that recently used `antigravity-acp`, avoiding idle spawns across unused worktrees) + shared warm side-call worker for isolated `/compact` & judge (`generate`) calls (~1.6s -> ~7ms per new session).
   - Per-session `Semaphore(1)` serialization, 5-minute idle worker reaper (`idleTimeoutMs`, preserving `acpSessionId` for `session/resume`), and `retry: false` hook to prevent double-retries.
4. **Real MCP isolation (`isolateUserMcp: true`, `includeDefaultCustomizations: false` by default):** Launches `agy acp` with an isolated `--gemini-dir` (empty `config/mcp_config.json`, symlinked `antigravity-cli/` for OAuth & session resume) so workers never spawn duplicate copies of `~/.gemini` MCP servers or bloat prompts with duplicate tool schemas.
5. **Bounded permissions, MCP relay compaction (`maxMcpResultChars: 8000`), and accurate usage accounting:** Compacts oversized relayed MCP outputs cleanly below `agy`'s inline truncation threshold (preserving saved-file pointers so `agy` never double-truncates JSON), and handles `agy`'s exclusive `inputTokens` + `cachedReadTokens` accounting accurately.
6. **Plan mode auto-sync (`defaultMode`, default `"auto"`):** Automatically switches `agy` to `"plan"` when using OpenCode's `plan` agent and back when leaving `plan` (unless explicitly overridden via RPC `setMode`).
7. **Instruction & Code Mode forwarding (`forwardInstructions`, default `true`):** Forwards OpenCode agent roles, `AGENTS.md`, skills, and Code Mode instructions via ACP `_meta.antigravity.systemPrompt.append` while keeping OpenCode's core system prompt lean.
8. **Turn fingerprinting & delta catch-up:** Tracks completed turns so cross-model switches append only compact `<conversation-checkpoint>` deltas to the live ACP session, while `/undo` or edits cleanly start a restored session.
9. **Side-call isolation:** Runs compaction and auxiliary generate calls on throwaway ACP sessions without mutating primary session state.

## Configuration (`opencode.json`)

```json
{
  "plugins": ["github:brandonkramer/opencode-agy-acp"],
  "providers": {
    "antigravity-acp": {
      "settings": {
        "defaultMode": "auto",
        "forwardInstructions": true,
        "codeMode": "auto",
        "prewarm": "auto",
        "smoothStreaming": true,
        "includeDefaultCustomizations": false,
        "isolateUserMcp": true,
        "maxMcpResultChars": 8000,
        "chunkTimeoutMs": 120000,
        "permissionTimeoutMs": 300000,
        "idleTimeoutMs": 300000
      }
    }
  }
}
```

### Optional: OpenCode `full_codemode` (Experimental)

By default, OpenCode's Code Mode (`execute`, relayed here as `mcp_oc_execute`) exposes MCP servers and `tools.opencode.*`, while local file reads/edits/shell run through `agy`'s fast native tools.

If you also want scripted batch file I/O (`Bun.file`, `Bun.write`, `Bun.Glob`), `fetch`, and Code Mode `shell`/`patch` available inside `mcp_oc_execute`, you can opt into OpenCode's user-level `full_codemode` flag (note: this is an OpenCode core work-in-progress setting and applies to **all** models in your OpenCode instance):

```sh
# Try for a single isolated session:
OPENCODE_FLAGS=full_codemode opencode --standalone
```

Or enable persistently in `opencode.json` (then run `opencode reload`):

```json
{
  "experimental": {
    "full_codemode": true
  }
}
```

No extra plugin configuration is needed — `codeMode: "auto"` automatically relays `execute` as `mcp_oc_execute`.

### Environment overrides

- `AGY_BIN`: explicit path to `agy`
- `AGY_ACP_DEFAULT_MODE=auto|accept-edits|default|plan`
- `AGY_ACP_FORWARD_INSTRUCTIONS=true|false`
- `AGY_ACP_CODE_MODE=auto|true|false`
- `AGY_ACP_PREWARM=auto|true|false`
- `AGY_ACP_SMOOTH_STREAMING=true|false`
- `AGY_ACP_MAX_SYSTEM_PROMPT_CHARS`
- `AGY_ACP_MAX_MCP_RESULT_CHARS=8000` (`0` to disable)
- `AGY_ACP_INCLUDE_DEFAULT_CUSTOMIZATIONS=true|false`
- `AGY_ACP_ISOLATE_USER_MCP=true|false`
- `AGY_ACP_CHUNK_TIMEOUT_MS`
- `AGY_ACP_TRACE_LOG=/path/to/trace.jsonl`

## Development

```sh
bun install
bun run typecheck
bun test
```

## Installation

Requires OpenCode **2.4+** and the Google Antigravity CLI (`agy`) installed and signed in (`agy`).

```sh
opencode plugin add github:brandonkramer/opencode-agy-acp
opencode reload
```

Update later with `opencode plugin update`.
