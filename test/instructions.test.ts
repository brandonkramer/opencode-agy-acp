import { describe, expect, test } from "bun:test"
import {
  buildSystemPromptAppend,
  extractOpenCodeInstructions,
  hashSystemPrompt,
  stripInternalOpencodeNamespace,
} from "../src/acp/instructions.js"

describe("instructions forwarding helpers", () => {
  test("drops system[0] base prompt + core/date & core/environment while keeping role, AGENTS.md, skills, and Code Mode", () => {
    const out = extractOpenCodeInstructions({
      agentName: "planner",
      agentSystem: "You are a careful planner. Never edit files without approval.",
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
                { key: "/repo/AGENTS.md", text: "Always use bun test instead of jest." },
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
          text: "Code Mode enabled: use mcp__oc tools via typed helpers.\n<opencode:internal>secret-tag</opencode:internal>",
        },
      ],
    })

    expect(out).toBeDefined()
    expect(out).not.toContain("You are Gemini, an interactive CLI coding assistant")
    expect(out).not.toContain("Today's date:")
    expect(out).not.toContain("<env>Working directory:")
    expect(out).not.toContain("boilerplate wrapper")
    expect(out).not.toContain("secret-tag")
    expect(out).toContain("# Agent role (from OpenCode)")
    expect(out).toContain("Never edit files without approval.")
    expect(out).toContain("Always use bun test instead of jest.")
    expect(out).toContain("<available_skills>")
    expect(out).toContain("Code Mode enabled")
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

  test("buildSystemPromptAppend respects forwardInstructions and hashSystemPrompt is deterministic", () => {
    expect(
      buildSystemPromptAppend({ rawInstructions: "Follow AGENTS.md", forwardInstructions: false }),
    ).toBeUndefined()
    const appended = buildSystemPromptAppend({
      rawInstructions: "  Follow AGENTS.md  ",
      forwardInstructions: true,
    })
    expect(appended).toBe("Follow AGENTS.md")
    expect(hashSystemPrompt(appended)).toBe(hashSystemPrompt("Follow AGENTS.md"))
    expect(hashSystemPrompt(appended)).not.toBe(hashSystemPrompt("Different"))
    expect(hashSystemPrompt(undefined)).toBe("none")
  })
})
