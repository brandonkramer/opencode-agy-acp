import { Rpc } from "@opencode/plugin/rpc"
import { PROVIDER_ID } from "./types.js"

const sessionIdSchema = {
  type: "object",
  properties: {
    sessionID: { type: "string" },
  },
  required: ["sessionID"],
  additionalProperties: false,
} as const

const stateSchema = {
  type: "object",
  properties: {
    sessionID: { type: "string" },
    mode: { type: "string", enum: ["auto", "accept-edits", "default", "plan"] },
    codeModeEnabled: { type: "boolean" },
    starting: { type: "boolean" },
    acpSessionId: { anyOf: [{ type: "string" }, { type: "null" }] },
    connected: { type: "boolean" },
    busy: { type: "boolean" },
    modelCount: { type: "number" },
    relayedToolCount: { type: "number" },
    lastError: { anyOf: [{ type: "string" }, { type: "null" }] },
    cooldownUntil: { anyOf: [{ type: "number" }, { type: "null" }] },
  },
  required: [
    "sessionID",
    "mode",
    "codeModeEnabled",
    "starting",
    "acpSessionId",
    "connected",
    "busy",
    "modelCount",
    "relayedToolCount",
    "lastError",
    "cooldownUntil",
  ],
  additionalProperties: false,
} as const

const diagnosticsSchema = {
  type: "object",
  properties: {
    sessionID: { type: "string" },
    acpSessionId: { anyOf: [{ type: "string" }, { type: "null" }] },
    mode: { type: "string", enum: ["auto", "accept-edits", "default", "plan"] },
    codeModeEnabled: { type: "boolean" },
    connected: { type: "boolean" },
    busy: { type: "boolean" },
    watchdogPaused: { type: "boolean" },
    activeNativeTools: { type: "array", items: { type: "string" } },
    pendingMcpCalls: { type: "number" },
    lastError: { anyOf: [{ type: "string" }, { type: "null" }] },
    recentTrace: {
      type: "array",
      items: {
        type: "object",
        properties: {
          ts: { type: "string" },
          dir: { type: "string", enum: ["out", "in", "err", "sys"] },
          summary: { type: "string" },
        },
        required: ["ts", "dir", "summary"],
        additionalProperties: false,
      },
    },
  },
  required: [
    "sessionID",
    "acpSessionId",
    "mode",
    "codeModeEnabled",
    "connected",
    "busy",
    "watchdogPaused",
    "activeNativeTools",
    "pendingMcpCalls",
    "lastError",
    "recentTrace",
  ],
  additionalProperties: false,
} as const

export const AcpRpc = Rpc.define({
  id: PROVIDER_ID,
  methods: {
    getState: {
      input: sessionIdSchema,
      output: stateSchema,
    },
    getDiagnostics: {
      input: sessionIdSchema,
      output: diagnosticsSchema,
    },
    setMode: {
      input: {
        type: "object",
        properties: {
          sessionID: { type: "string" },
          mode: { type: "string", enum: ["auto", "accept-edits", "default", "plan"] },
        },
        required: ["sessionID", "mode"],
        additionalProperties: false,
      },
      output: stateSchema,
    },
    setCodeMode: {
      input: {
        type: "object",
        properties: {
          sessionID: { type: "string" },
          codeModeEnabled: { type: "boolean" },
        },
        required: ["sessionID", "codeModeEnabled"],
        additionalProperties: false,
      },
      output: stateSchema,
    },
    resetSession: {
      input: sessionIdSchema,
      output: stateSchema,
    },
    reloadModels: {
      input: {
        type: "object",
        properties: {},
        additionalProperties: false,
      },
      output: {
        type: "object",
        properties: {
          modelCount: { type: "number" },
        },
        required: ["modelCount"],
        additionalProperties: false,
      },
    },
  },
  events: {},
})
