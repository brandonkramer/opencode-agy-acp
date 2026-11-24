import { describe, expect, test } from "bun:test"
import {
  buildSystemPromptAppend,
  extractOpenCodeInstructions,
  hashSystemPrompt,
  isFullCodeModeExecuteDescription,
  rewriteCodeModeExecuteRefs,
  stripInternalOpencodeNamespace,
} from "../src/acp/instructions.js"
import { buildToolDescriptors } from "../src/acp/mcp-relay.js"

describe("instructions forwarding helpers", () => {
  test("drops system[0], core/date, core/environment, Code Mode, and tools.opencode.* in Pure ACP", () => {
    const out = extractOpenCodeInstructions({
      agentName: "planner",
      agentSystem: "You are a careful planner. Never edit files without approval.",
      codeModeEnabled: false,
      system: [
        {
          type: "text",
          text: "You are Gemini, an interactive CLI coding assistant running in OpenCode.",
        },
        {
          type: "text",
          text: "boilerplate wrapper",
          metadata: {
            experimental: {
              instructions: [
                { key: "core/date", text: "Today's date: Mon Nov 24 2026" },
                { key: "core/environment", text: "<env>Working directory: /repo</env>" },
                { key: "core/codemode", text: "# Code Mode\nUse `execute` tool." },
                { key: "core/mcp-guidance", text: "Use MCP tools via execute." },
                {
                  key: "/repo/AGENTS.md",
                  text: "Always use bun test instead of jest.\nConsider using `execute` to call `tools.opencode.session.move`.",
                },
              ],
            },
          },
        },
        {
          type: "text",
          text: "<available_skills>\n- example-skill: project helper\n</available_skills>",
        },
        {
          type: "text",
          text: "# Code Mode\nUnkeyed Code Mode section\n<opencode:internal>secret-tag</opencode:internal>",
        },
      ],
    })

    expect(out).toBeDefined()
    expect(out).not.toContain("You are Gemini, an interactive CLI coding assistant")
    expect(out).not.toContain("Today's date:")
    expect(out).not.toContain("<env>Working directory:")
    expect(out).not.toContain("# Code Mode")
    expect(out).not.toContain("Use MCP tools via execute.")
    expect(out).not.toContain("tools.opencode.session.move")
    expect(out).not.toContain("boilerplate wrapper")
    expect(out).not.toContain("secret-tag")
    expect(out).toContain("# Agent role (from OpenCode)")
    expect(out).toContain("Never edit files without approval.")
    expect(out).toContain("Always use bun test instead of jest.")
    expect(out).toContain("<available_skills>")
  })

  test("preserves Code Mode and tools.opencode.* instructions when codeModeEnabled is true", () => {
    const out = extractOpenCodeInstructions({
      agentName: "build",
      codeModeEnabled: true,
      system: [
        { type: "text", text: "You are Gemini." },
        {
          type: "text",
          text: "",
          metadata: {
            experimental: {
              instructions: [
                { key: "core/date", text: "Today's date: Mon Nov 24 2026" },
                { key: "core/codemode", text: "# Code Mode\nUse `execute` tool." },
                { key: "core/mcp-guidance", text: "Use MCP tools via execute." },
                { key: "core/instructions", text: "Call `tools.opencode.session.move` if needed." },
              ],
            },
          },
        },
      ],
    })

    expect(out).toBeDefined()
    expect(out).not.toContain("You are Gemini.")
    expect(out).not.toContain("Today's date:")
    expect(out).toContain("# Code Mode")
    expect(out).toContain("Use MCP tools via execute.")
    expect(out).toContain("tools.opencode.session.move")
  })

  test("returns undefined when only system[0] or core/date/core/environment entries exist", () => {
    const out = extractOpenCodeInstructions({
      agentName: "build",
      system: [
        { type: "text", text: "You are Gemini, built by Google." },
        {
          type: "text",
          text: "",
          metadata: {
            experimental: {
              instructions: [
                { key: "core/date", text: "Today's date: Mon Nov 24 2026" },
                { key: "core/environment", text: "<env>platform=darwin</env>" },
              ],
            },
          },
        },
      ],
    })
    expect(out).toBeUndefined()
  })

  test("stripInternalOpencodeNamespace cleans XML tags and normalizes whitespace", () => {
    expect(
      stripInternalOpencodeNamespace(
        "Keep this\n<opencode:meta attr='1'>drop\nmulti</opencode:meta>\n\n\n<opencode:ping/>\nAnd this",
      ),
    ).toBe("Keep this\n\nAnd this")
  })

  test("adds mcp_oc_execute bridge note when codeModeEnabled is true with relayedToolNames, omitted otherwise", () => {
    const withRelay = extractOpenCodeInstructions({
      agentName: "build",
      codeModeEnabled: true,
      relayedToolNames: ["execute", "subagent"],
      system: [{ type: "text", text: "You are Gemini." }],
    })
    expect(withRelay).toContain("# OpenCode Relayed Tools (MCP `oc`)")
    expect(withRelay).toContain("`mcp_oc_execute`")
    expect(withRelay).toContain("`mcp_oc_subagent`")

    const withoutRelay = extractOpenCodeInstructions({
      agentName: "build",
      codeModeEnabled: false,
      relayedToolNames: ["execute"],
      system: [{ type: "text", text: "You are Gemini." }],
    })
    expect(withoutRelay).toBeUndefined()
  })

  test("buildSystemPromptAppend respects forwardInstructions, maxChars truncation, and hashSystemPrompt", () => {
    expect(
      buildSystemPromptAppend({ rawInstructions: "Follow AGENTS.md", forwardInstructions: false }),
    ).toBeUndefined()
    const appended = buildSystemPromptAppend({
      rawInstructions: "  Follow AGENTS.md  ",
      forwardInstructions: true,
    })
    expect(appended).toBe("Follow AGENTS.md")
    const truncated = buildSystemPromptAppend({
      rawInstructions: "A".repeat(100),
      forwardInstructions: true,
      maxChars: 30,
    })
    expect(truncated).toHaveLength(30)
    expect(truncated?.endsWith("...[truncated]")).toBe(true)
    expect(hashSystemPrompt(appended)).toBe(hashSystemPrompt("Follow AGENTS.md"))
    expect(hashSystemPrompt(appended)).not.toBe(hashSystemPrompt("Different"))
    expect(hashSystemPrompt(undefined)).toBe("none")
  })

  test("rewrites Code Mode `execute` references to `mcp_oc_execute` without touching prose or `db.execute`", () => {
    const input = [
      "Only work inside the `execute` tool (call only inside `execute`).",
      "Consider using `execute` to call `tools.opencode.session.move`.",
      "Called inside\n`execute` as `tools.opencode.*`.",
      "You can execute the script or run `db.execute` directly.",
    ].join("\n")

    const rewritten = rewriteCodeModeExecuteRefs(input)
    expect(rewritten).toContain("inside the `mcp_oc_execute` tool")
    expect(rewritten).toContain("(call only inside `mcp_oc_execute`)")
    expect(rewritten).toContain("using `mcp_oc_execute` to call")
    expect(rewritten).toContain("Called inside\n`mcp_oc_execute` as `tools.opencode.*`")
    expect(rewritten).toContain("execute the script")
    expect(rewritten).toContain("`db.execute`")

    const forwarded = extractOpenCodeInstructions({
      agentName: "build",
      codeModeEnabled: true,
      relayedToolNames: ["execute", "subagent"],
      system: [
        { type: "text", text: "Base prompt" },
        {
          type: "text",
          text: "",
          metadata: {
            experimental: {
              instructions: [
                {
                  key: "core/codemode",
                  text: "# Code Mode\nTools only work inside the `execute` tool.\n## Code Mode tools (call only inside `execute`)",
                },
              ],
            },
          },
        },
      ],
    })
    expect(forwarded).toContain("When to prefer `mcp_oc_execute`:")
    expect(forwarded).toContain("inside the `mcp_oc_execute` tool")
    expect(forwarded).toContain("call only inside `mcp_oc_execute`")
    expect(forwarded).not.toContain("inside the `execute` tool")
  })

  test("buildToolDescriptors gives `execute` a trigger-first, `mcp_oc_execute`-normalized description", () => {
    const { mcpDescriptors } = buildToolDescriptors({
      execute: {
        description: "Run JS. Inside `execute`, the only callable tools are Code Mode tools.",
        input: { type: "object", properties: { code: { type: "string" } } },
      },
    })
    expect(mcpDescriptors).toHaveLength(1)
    const desc = mcpDescriptors[0]!.description
    expect(desc.startsWith("PREFER THIS TOOL")).toBe(true)
    expect(desc).toContain("Inside `mcp_oc_execute`")
    expect(desc).not.toContain("Inside `execute`")
    expect(desc).toContain('await search({ query: "..." })')
  })

  test("auto-detects full_codemode and steers Gemini to prefer mcp_oc_execute for file edits", () => {
    const fullDesc =
      "Run JavaScript in a confined Code Mode runtime. `fetch` is available, and Bun's file APIs (`Bun.file`, `Bun.write`, `Bun.Glob`) for files. Inside `execute`, only catalog tools are callable."
    expect(isFullCodeModeExecuteDescription(fullDesc)).toBe(true)
    expect(isFullCodeModeExecuteDescription("Run JavaScript in Code Mode.")).toBe(false)
    expect(isFullCodeModeExecuteDescription(undefined)).toBe(false)

    const forwarded = extractOpenCodeInstructions({
      agentName: "build",
      codeModeEnabled: true,
      fullCodeMode: true,
      relayedToolNames: ["execute"],
      system: [{ type: "text", text: "Base prompt" }],
    })
    expect(forwarded).toContain("OpenCode Full Code Mode is ACTIVE")
    expect(forwarded).toContain("Prefer `mcp_oc_execute` FIRST instead of `read_file`, `write_file`, `edit_file`")
    expect(forwarded).not.toContain("Keep single local file reads/edits/shell on native tools")

    const { mcpDescriptors } = buildToolDescriptors({
      execute: {
        description: fullDesc,
        input: { type: "object", properties: { code: { type: "string" } } },
      },
    })
    const desc = mcpDescriptors[0]!.description
    expect(desc.startsWith("PRIMARY TOOL (OpenCode Full Code Mode)")).toBe(true)
    expect(desc).toContain("Inside `mcp_oc_execute`")
    expect(desc).toContain('Bun.write("a.txt"')
  })
})
