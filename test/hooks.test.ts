import { afterEach, beforeEach, describe, expect, test } from "bun:test"
import { AssistantMessage, UserMessage } from "@opencode/ai/effect"
import type { Model, Provider } from "@opencode/plugin/effect"
import type { Context } from "@opencode/plugin/effect/plugin"
import type { ProviderEditor } from "@opencode/plugin/effect/provider"
import type {
  SessionContext,
  SessionModelRequest,
  SessionRetry,
  SessionTitle,
} from "@opencode/plugin/effect/session"
import { Effect, Option, Scope, Stream } from "effect"
import { resolveModelProfile } from "../src/acp/catalog.js"
import plugin from "../src/index.js"
import { OPENCODE_QUESTION_TOOL, PACKAGE_ID, PROVIDER_ID } from "../src/types.js"

describe("Plugin.define hook wiring (src/index.ts)", () => {
  let prevPrewarm: string | undefined
  beforeEach(() => {
    prevPrewarm = process.env.AGY_ACP_PREWARM
    process.env.AGY_ACP_PREWARM = "false"
  })
  afterEach(() => {
    if (prevPrewarm === undefined) delete process.env.AGY_ACP_PREWARM
    else process.env.AGY_ACP_PREWARM = prevPrewarm
  })
  test("registers provider, aisdk, context, model.request, title, retry:false, rpc, and finalizer", async () => {
    const store = new Map<string, unknown>()
    const providers = new Map<string, { info: Provider.Info; models: readonly Model.Info[] }>()
    const sessionHooks = new Map<string, Array<(input: unknown) => Effect.Effect<void>>>()
    const aisdkHooks = new Map<string, Array<(input: unknown) => Effect.Effect<void>>>()
    let registeredRpc: Record<string, (input: unknown) => Effect.Effect<unknown>> | undefined

    const mockCtx = {
      location: { directory: process.cwd() },
      storage: {
        get: (key: string) => Effect.sync(() => store.get(key) as never),
        set: (key: string, value: unknown) =>
          Effect.sync(() => {
            store.set(key, value)
          }),
        remove: (key: string) =>
          Effect.sync(() => {
            store.delete(key)
          }),
      },
      provider: {
        get: () => Effect.fail(new Error("not found") as never),
        reload: () => Effect.void,
        transform: (fn: (editor: ProviderEditor) => void) =>
          Effect.sync(() => {
            const editor: ProviderEditor = {
              list: () => [],
              get: (id) => {
                const entry = providers.get(id)
                if (!entry) return undefined
                return {
                  provider: entry.info,
                  models: new Map(entry.models.map((m) => [m.id, m])),
                }
              },
              add: (input) => {
                providers.set(input.info.id, { info: input.info, models: input.models })
              },
              update: () => {},
              remove: (id) => {
                providers.delete(id)
              },
              models: {
                set: () => {},
                update: () => {},
                remove: () => {},
              },
            }
            fn(editor)
          }),
      },
      aisdk: {
        hook: (name: string, fn: (input: unknown) => Effect.Effect<void>) =>
          Effect.sync(() => {
            const list = aisdkHooks.get(name) ?? []
            list.push(fn)
            aisdkHooks.set(name, list)
          }),
      },
      session: {
        get: () => Effect.succeed({ location: { directory: process.cwd() } } as never),
        update: () => Effect.void,
        hook: (name: string, fn: (input: unknown) => Effect.Effect<void>) =>
          Effect.sync(() => {
            const list = sessionHooks.get(name) ?? []
            list.push(fn)
            sessionHooks.set(name, list)
          }),
      },
      permission: {
        assert: () => Effect.void,
      },
      rpc: {
        register: (
          _def: unknown,
          handlers: Record<string, (input: unknown) => Effect.Effect<unknown>>,
        ) =>
          Effect.sync(() => {
            registeredRpc = handlers
            return { events: { emit: () => Effect.void } } as never
          }),
      },
      event: {
        subscribe: () => Stream.empty,
      },
    } as unknown as Context

    // Pre-populate model catalog cache so background discovery doesn't spawn real agy
    store.set("antigravity-acp:model-catalog:v1", {
      version: 1,
      updatedAt: Date.now(),
      models: [
        {
          id: "gemini-4-neon",
          name: "Gemini 4 Neon",
          family: "gemini-4-neon",
          defaultAcpModelId: "gemini-4-neon-high",
          variants: [{ id: "high", acpModelId: "gemini-4-neon-high", thoughtLevel: "high" }],
        },
      ],
    })

    const scope = Effect.runSync(Scope.make())
    try {
      await Effect.runPromise(plugin.effect(mockCtx).pipe(Scope.provide(scope)))

      // 1. Provider transform registered antigravity-acp
      const reg = providers.get(PROVIDER_ID)
      expect(reg).toBeDefined()
      expect(reg!.info.package).toBe(PACKAGE_ID)
      expect(reg!.models[0]?.settings?.antigravityBridgeGeneration).toBeTruthy()

      // 2. Retry hook sets retry: false
      const retryHook = sessionHooks.get("retry")?.[0]
      expect(retryHook).toBeDefined()
      const retryEvent: SessionRetry = {
        sessionID: "ses_hook_1" as never,
        agent: "build" as never,
        model: { providerID: PROVIDER_ID as never, id: "gemini-4-neon" as never },
        error: {} as never,
        attempt: 1,
        decision: { retry: true, delay: 1000 },
      }
      await Effect.runPromise(retryHook!(retryEvent))
      expect(retryEvent.decision).toEqual({ retry: false })

      // 3. Context hook: codeMode="auto" stays Pure ACP when only shadowed tools exist (retains only `question`)
      const contextHook = sessionHooks.get("context")?.[0]
      expect(contextHook).toBeDefined()
      const acpContext: SessionContext = {
        sessionID: "ses_hook_1" as never,
        agent: "build" as never,
        model: { providerID: PROVIDER_ID as never, id: "gemini-4-neon" as never },
        system: [{ type: "system", text: "outer system prompt" } as never],
        messages: [
          new UserMessage({ role: "user", content: [{ type: "text", text: "hi" }] }),
        ],
        options: {},
        tools: {
          bash: { description: "Run shell", input: {} },
          [OPENCODE_QUESTION_TOOL]: { description: "Ask", input: {} },
        },
      }
      await Effect.runPromise(contextHook!(acpContext))
      expect(acpContext.system).toEqual([])
      expect(Object.keys(acpContext.tools)).toEqual([OPENCODE_QUESTION_TOOL])

      // 3b. codeMode="auto" auto-enables Code Mode when non-shadowed tools like `subagent` are present
      const hybridAutoCtx: SessionContext = {
        ...acpContext,
        sessionID: "ses_hook_hybrid" as never,
        system: [{ type: "system", text: "outer" } as never],
        tools: {
          bash: { description: "Run shell", input: {} },
          [OPENCODE_QUESTION_TOOL]: { description: "Ask", input: {} },
          subagent: { description: "Subagent", input: {} },
        },
      }
      await Effect.runPromise(contextHook!(hybridAutoCtx))
      expect(Object.keys(hybridAutoCtx.tools)).toEqual(["bash", OPENCODE_QUESTION_TOOL, "subagent"])

      // 4. Context hook: non-ACP provider sanitizes acp_* hosted tools
      const nonAcpContext: SessionContext = {
        sessionID: "ses_hook_1" as never,
        agent: "build" as never,
        model: { providerID: "openai" as never, id: "gpt-5" as never },
        system: [],
        messages: [
          new AssistantMessage({
            role: "assistant",
            content: [
              {
                type: "tool-call",
                id: "t1",
                name: "acp_read_file",
                input: { path: "a.ts" },
                providerExecuted: true,
                providerMetadata: { [PROVIDER_ID]: { acp: true } },
              },
            ],
          }),
        ],
        options: {},
        tools: {},
      }
      await Effect.runPromise(contextHook!(nonAcpContext))
      expect(nonAcpContext.messages[0].role).toBe("assistant")
      if (nonAcpContext.messages[0].role === "assistant") {
        expect(nonAcpContext.messages[0].content[0].type).toBe("text")
      }

      // 5. model.request & title hooks
      const reqHook = sessionHooks.get("model.request")?.[0]
      const reqEvt: SessionModelRequest = {
        sessionID: "ses_hook_1" as never,
        agent: "build" as never,
        model: {
          providerID: PROVIDER_ID as never,
          id: "gemini-4-neon" as never,
          variant: "high" as never,
        },
        kind: "primary",
        headers: {},
      }
      await Effect.runPromise(reqHook!(reqEvt))
      expect(reqEvt.headers).toEqual({
        "x-opencode-session-id": "ses_hook_1",
        "x-opencode-request-kind": "primary",
        "x-opencode-agent": "build",
        "x-opencode-variant": "high",
      })

      const titleHook = sessionHooks.get("title")?.[0]
      const titleEvt: SessionTitle = {
        sessionID: "ses_hook_1" as never,
        model: { providerID: PROVIDER_ID as never, id: "gemini-4-neon" as never },
        system: [],
        messages: [
          new UserMessage({ role: "user", content: [{ type: "text", text: "Fix ACP stall bug" }] }),
        ],
        options: {},
      }
      await Effect.runPromise(titleHook!(titleEvt))
      expect(titleEvt.result).toBe("Fix ACP stall bug")

      // 6. RPC handlers & Code Mode toggle preserves tools in context hook
      expect(registeredRpc).toBeDefined()
      await Effect.runPromise(
        registeredRpc!.setCodeMode({ sessionID: "ses_hook_1", codeModeEnabled: true }),
      )
      const diag = (await Effect.runPromise(
        registeredRpc!.getDiagnostics({ sessionID: "ses_hook_1" }),
      )) as { codeModeEnabled: boolean; watchdogPaused: boolean }
      expect(diag.codeModeEnabled).toBe(true)
      expect(diag.watchdogPaused).toBe(false)

      const codeModeCtx: SessionContext = {
        ...acpContext,
        system: [{ type: "system", text: "outer" } as never],
        tools: {
          bash: { description: "Run shell", input: {} },
          subagent: { description: "Subagent", input: {} },
        },
      }
      await Effect.runPromise(contextHook!(codeModeCtx))
      expect(Object.keys(codeModeCtx.tools)).toEqual(["bash", "subagent"])

      // 7. aisdk sdk hook binds languageModel
      const sdkHook = aisdkHooks.get("sdk")?.[0]
      const sdkEvt = { model: reg!.models[0], package: PACKAGE_ID, options: {}, sdk: undefined as unknown }
      await Effect.runPromise(sdkHook!(sdkEvt))
      expect(typeof (sdkEvt.sdk as { languageModel: unknown }).languageModel).toBe("function")

      // 8. compaction and generate hooks sanitize messages, clear tools, and preserve system prompts
      const compactionHook = sessionHooks.get("compaction")?.[0]
      const generateHook = sessionHooks.get("generate")?.[0]
      expect(compactionHook).toBeDefined()
      expect(generateHook).toBeDefined()

      const sideCtx: SessionContext = {
        ...acpContext,
        system: [{ type: "text", text: "Keep summary short" } as never],
        tools: { bash: { description: "Run shell", input: {} } },
      }
      await Effect.runPromise(compactionHook!(sideCtx))
      expect(sideCtx.system).toEqual([{ type: "text", text: "Keep summary short" } as never])
      expect(sideCtx.tools).toEqual({})
    } finally {
      await Effect.runPromise(Scope.close(scope, Option.none() as never))
    }
  })

  test("preserves existing providers.antigravity-acp.settings across transform and reads effective settings", async () => {
    const store = new Map<string, unknown>()
    const providers = new Map<string, { info: Provider.Info; models: readonly Model.Info[] }>()
    const sessionHooks = new Map<string, Array<(input: unknown) => Effect.Effect<void>>>()
    let registeredRpc: Record<string, (input: unknown) => Effect.Effect<unknown>> | undefined

    // Pre-seed provider entry as OpenCode config loader does before plugin transform runs.
    providers.set(PROVIDER_ID, {
      info: {
        id: PROVIDER_ID as Provider.ID,
        name: "Antigravity (ACP)",
        package: PACKAGE_ID,
        activation: "enabled",
        settings: {
          defaultMode: "accept-edits",
          forwardInstructions: false,
          codeMode: true,
        } as never,
      } as Provider.Info,
      models: [],
    })

    store.set("antigravity-acp:model-catalog:v1", {
      version: 1,
      updatedAt: Date.now(),
      models: [
        {
          id: "gemini-4-neon",
          name: "Gemini 4 Neon",
          family: "gemini-4-neon",
          defaultAcpModelId: "gemini-4-neon-high",
          variants: [{ id: "high", acpModelId: "gemini-4-neon-high", thoughtLevel: "high" }],
        },
      ],
    })

    const mockCtx = {
      location: { directory: process.cwd() },
      storage: {
        get: (key: string) => Effect.sync(() => store.get(key) as never),
        set: (key: string, value: unknown) =>
          Effect.sync(() => {
            store.set(key, value)
          }),
        remove: (key: string) =>
          Effect.sync(() => {
            store.delete(key)
          }),
      },
      agent: {
        get: () => Effect.succeed({ data: { system: "Role instructions from agent" } } as never),
      },
      provider: {
        get: () => {
          const entry = providers.get(PROVIDER_ID)
          return entry
            ? Effect.succeed({ data: entry.info } as never)
            : Effect.fail(new Error("not found") as never)
        },
        reload: () => Effect.void,
        transform: (fn: (editor: ProviderEditor) => void) =>
          Effect.sync(() => {
            const editor: ProviderEditor = {
              list: () => [],
              get: (id) => {
                const entry = providers.get(id)
                if (!entry) return undefined
                return {
                  provider: entry.info,
                  models: new Map(entry.models.map((m) => [m.id, m])),
                }
              },
              add: (input) => {
                providers.set(input.info.id, { info: input.info, models: input.models })
              },
              update: () => {},
              remove: (id) => {
                providers.delete(id)
              },
              models: {
                set: () => {},
                update: () => {},
                remove: () => {},
              },
            }
            fn(editor)
          }),
      },
      aisdk: { hook: () => Effect.void },
      session: {
        get: () => Effect.succeed({ location: { directory: process.cwd() } } as never),
        update: () => Effect.void,
        hook: (name: string, fn: (input: unknown) => Effect.Effect<void>) =>
          Effect.sync(() => {
            const list = sessionHooks.get(name) ?? []
            list.push(fn)
            sessionHooks.set(name, list)
          }),
      },
      permission: { assert: () => Effect.void },
      rpc: {
        register: (
          _def: unknown,
          handlers: Record<string, (input: unknown) => Effect.Effect<unknown>>,
        ) =>
          Effect.sync(() => {
            registeredRpc = handlers
            return { events: { emit: () => Effect.void } } as never
          }),
      },
      event: { subscribe: () => Stream.empty },
    } as unknown as Context

    const scope = Effect.runSync(Scope.make())
    try {
      await Effect.runPromise(plugin.effect(mockCtx).pipe(Scope.provide(scope)))
      const transformed = providers.get(PROVIDER_ID)
      expect(transformed?.info.settings).toEqual({
        defaultMode: "accept-edits",
        forwardInstructions: false,
        codeMode: true,
      })

      const state = (await Effect.runPromise(
        registeredRpc!.getState({ sessionID: "ses_preserved" }),
      )) as { mode: string; codeModeEnabled: boolean }
      expect(state.mode).toBe("accept-edits")
      expect(state.codeModeEnabled).toBe(true)
    } finally {
      await Effect.runPromise(Scope.close(scope, Option.none() as never))
    }
  })

  test("refreshes delayed providers.antigravity-acp.settings during context hook before filtering tools/Code Mode", async () => {
    const store = new Map<string, unknown>()
    const providers = new Map<string, { info: Provider.Info; models: readonly Model.Info[] }>()
    const sessionHooks = new Map<string, Array<(input: unknown) => Effect.Effect<void>>>()
    let registeredRpc: Record<string, (input: unknown) => Effect.Effect<unknown>> | undefined

    store.set("antigravity-acp:model-catalog:v1", {
      version: 1,
      updatedAt: Date.now(),
      models: [
        {
          id: "gemini-4-neon",
          name: "Gemini 4 Neon",
          family: "gemini-4-neon",
          defaultAcpModelId: "gemini-4-neon-high",
          variants: [{ id: "high", acpModelId: "gemini-4-neon-high", thoughtLevel: "high" }],
        },
      ],
    })

    const mockCtx = {
      location: { directory: process.cwd() },
      storage: {
        get: (key: string) => Effect.sync(() => store.get(key) as never),
        set: (key: string, value: unknown) =>
          Effect.sync(() => {
            store.set(key, value)
          }),
        remove: (key: string) =>
          Effect.sync(() => {
            store.delete(key)
          }),
      },
      agent: {
        get: () => Effect.succeed({ data: { system: undefined } } as never),
      },
      provider: {
        get: () => {
          const entry = providers.get(PROVIDER_ID)
          return entry
            ? Effect.succeed({ data: entry.info } as never)
            : Effect.fail(new Error("not found") as never)
        },
        reload: () => Effect.void,
        transform: (fn: (editor: ProviderEditor) => void) =>
          Effect.sync(() => {
            const editor: ProviderEditor = {
              list: () => [],
              get: (id) => {
                const entry = providers.get(id)
                if (!entry) return undefined
                return {
                  provider: entry.info,
                  models: new Map(entry.models.map((m) => [m.id, m])),
                }
              },
              add: (input) => {
                providers.set(input.info.id, { info: input.info, models: input.models })
              },
              update: () => {},
              remove: (id) => {
                providers.delete(id)
              },
              models: { set: () => {}, update: () => {}, remove: () => {} },
            }
            fn(editor)
          }),
      },
      aisdk: { hook: () => Effect.void },
      session: {
        get: () => Effect.succeed({ location: { directory: process.cwd() } } as never),
        update: () => Effect.void,
        hook: (name: string, fn: (input: unknown) => Effect.Effect<void>) =>
          Effect.sync(() => {
            const list = sessionHooks.get(name) ?? []
            list.push(fn)
            sessionHooks.set(name, list)
          }),
      },
      permission: { assert: () => Effect.void },
      rpc: {
        register: (
          _def: unknown,
          handlers: Record<string, (input: unknown) => Effect.Effect<unknown>>,
        ) =>
          Effect.sync(() => {
            registeredRpc = handlers
            return { events: { emit: () => Effect.void } } as never
          }),
      },
      event: { subscribe: () => Stream.empty },
    } as unknown as Context

    const scope = Effect.runSync(Scope.make())
    try {
      await Effect.runPromise(plugin.effect(mockCtx).pipe(Scope.provide(scope)))

      // Simulate OpenCode overlaying user settings after plugin effect/transform finished.
      const current = providers.get(PROVIDER_ID)!
      providers.set(PROVIDER_ID, {
        ...current,
        info: {
          ...current.info,
          settings: { codeMode: true } as never,
        },
      })

      const contextHook = sessionHooks.get("context")?.[0]
      expect(contextHook).toBeDefined()
      const event: SessionContext = {
        sessionID: "ses_delayed_cm" as never,
        agent: "build" as never,
        model: { providerID: PROVIDER_ID as never, id: "gemini-4-neon" as never },
        system: [
          { type: "text", text: "You are Gemini." } as never,
          {
            type: "text",
            text: "",
            metadata: {
              experimental: {
                instructions: [{ key: "core/codemode", text: "# Code Mode\nUse `execute`." }],
              },
            },
          } as never,
        ],
        messages: [new UserMessage({ role: "user", content: [{ type: "text", text: "hi" }] })],
        options: {},
        tools: {
          bash: { description: "Run shell", input: {} },
          execute: { description: "Execute code", input: {} },
        },
      }
      await Effect.runPromise(contextHook!(event))
      // Tools are preserved (not trimmed down to { question }) on Turn 1!
      expect(Object.keys(event.tools)).toEqual(["bash", "execute"])
      const state = (await Effect.runPromise(
        registeredRpc!.getState({ sessionID: "ses_delayed_cm" }),
      )) as { codeModeEnabled: boolean; relayedToolCount: number }
      expect(state.codeModeEnabled).toBe(true)
      expect(state.relayedToolCount).toBe(1)
    } finally {
      await Effect.runPromise(Scope.close(scope, Option.none() as never))
    }
  })

  test("sanitizes stripped-metadata executed:true acp_* tool parts across compaction, generate, and title hooks for non-ACP OpenAI-Chat models", async () => {
    const store = new Map<string, unknown>()
    const providers = new Map<string, { info: Provider.Info; models: readonly Model.Info[] }>()
    const sessionHooks = new Map<string, Array<(input: unknown) => Effect.Effect<void>>>()

    store.set("antigravity-acp:model-catalog:v1", {
      version: 1,
      updatedAt: Date.now(),
      models: [
        {
          id: "gemini-4-neon",
          name: "Gemini 4 Neon",
          family: "gemini-4-neon",
          defaultAcpModelId: "gemini-4-neon-high",
          variants: [{ id: "high", acpModelId: "gemini-4-neon-high", thoughtLevel: "high" }],
        },
      ],
    })

    const mockCtx = {
      location: { directory: process.cwd() },
      storage: {
        get: (key: string) => Effect.sync(() => store.get(key) as never),
        set: (key: string, value: unknown) =>
          Effect.sync(() => {
            store.set(key, value)
          }),
        remove: (key: string) =>
          Effect.sync(() => {
            store.delete(key)
          }),
      },
      provider: {
        get: () => Effect.fail(new Error("not found") as never),
        reload: () => Effect.void,
        transform: (fn: (editor: ProviderEditor) => void) =>
          Effect.sync(() => {
            const editor: ProviderEditor = {
              list: () => [],
              get: (id) => {
                const entry = providers.get(id)
                if (!entry) return undefined
                return { provider: entry.info, models: new Map(entry.models.map((m) => [m.id, m])) }
              },
              add: (input) => {
                providers.set(input.info.id, { info: input.info, models: input.models })
              },
              update: () => {},
              remove: (id) => {
                providers.delete(id)
              },
              models: { set: () => {}, update: () => {}, remove: () => {} },
            }
            fn(editor)
          }),
      },
      aisdk: { hook: () => Effect.void },
      session: {
        get: () => Effect.succeed({ location: { directory: process.cwd() } } as never),
        update: () => Effect.void,
        hook: (name: string, fn: (input: unknown) => Effect.Effect<void>) =>
          Effect.sync(() => {
            const list = sessionHooks.get(name) ?? []
            list.push(fn)
            sessionHooks.set(name, list)
          }),
      },
      permission: { assert: () => Effect.void },
      rpc: {
        register: () => Effect.sync(() => ({ events: { emit: () => Effect.void } }) as never),
      },
      event: { subscribe: () => Stream.empty },
    } as unknown as Context

    const makeStrippedReplayMessages = () => [
      new AssistantMessage({
        role: "assistant",
        content: [
          {
            type: "tool-call",
            id: "tc_1",
            name: "acp_run_command",
            input: { cmd: "git status" },
            providerExecuted: true,
          },
          {
            type: "tool-result",
            id: "tc_1",
            name: "acp_run_command",
            result: { type: "text", value: "clean tree" },
            providerExecuted: true,
          },
        ],
      }),
    ]

    const scope = Effect.runSync(Scope.make())
    try {
      await Effect.runPromise(plugin.effect(mockCtx).pipe(Scope.provide(scope)))

      for (const hookName of ["compaction", "generate"] as const) {
        const hook = sessionHooks.get(hookName)?.[0]
        expect(hook).toBeDefined()
        const evt = {
          sessionID: "ses_cross_1" as never,
          agent: "compaction" as never,
          model: { providerID: "ollama-cloud" as never, id: "deepseek-v4.1-flash:1017" as never },
          system: [],
          messages: makeStrippedReplayMessages(),
          options: {},
          tools: {},
        }
        await Effect.runPromise(hook!(evt))
        const msg = evt.messages[0]!
        expect(msg.role).toBe("assistant")
        if (msg.role === "assistant") {
          expect(msg.content.every((p) => p.type === "text")).toBe(true)
          expect(msg.content.some((p) => p.type === "tool-call" || p.type === "tool-result")).toBe(
            false,
          )
        }
      }

      const titleHook = sessionHooks.get("title")?.[0]
      expect(titleHook).toBeDefined()
      const titleEvt: SessionTitle = {
        sessionID: "ses_cross_1" as never,
        model: { providerID: "ollama-cloud" as never, id: "deepseek-v4.1-flash:1017" as never },
        system: [],
        messages: makeStrippedReplayMessages(),
        options: {},
      }
      await Effect.runPromise(titleHook!(titleEvt))
      expect(titleEvt.result).toBeUndefined()
      const tMsg = titleEvt.messages[0]!
      if (tMsg.role === "assistant") {
        expect(tMsg.content.every((p) => p.type === "text")).toBe(true)
      }
    } finally {
      await Effect.runPromise(Scope.close(scope, Option.none() as never))
    }
  })

  test("populates per-family model limits and input modalities in provider transform", async () => {
    expect(resolveModelProfile("gemini-4-neon")).toEqual({
      limit: { context: 1_048_576, output: 65_536 },
      capabilities: { tools: true, input: ["text", "image"], output: ["text"] },
    })
    expect(resolveModelProfile("claude-opus-5-5")).toEqual({
      limit: { context: 1_000_000, output: 128_000 },
      capabilities: { tools: true, input: ["text", "image"], output: ["text"] },
    })
    expect(resolveModelProfile("claude-sonnet-4-6")).toEqual({
      limit: { context: 1_000_000, output: 64_000 },
      capabilities: { tools: true, input: ["text", "image"], output: ["text"] },
    })
    expect(resolveModelProfile("gpt-oss-120b-medium")).toEqual({
      limit: { context: 131_072, output: 32_768 },
      capabilities: { tools: true, input: ["text"], output: ["text"] },
    })
    expect(resolveModelProfile("unknown-future-model")).toEqual({
      limit: { context: 200_000, output: 65_536 },
      capabilities: { tools: true, input: ["text", "image"], output: ["text"] },
    })

    const store = new Map<string, unknown>()
    const providers = new Map<string, { info: Provider.Info; models: readonly Model.Info[] }>()
    store.set("antigravity-acp:model-catalog:v1", {
      version: 1,
      updatedAt: Date.now(),
      models: [
        {
          id: "gemini-4-neon",
          name: "Gemini 4 Neon",
          family: "gemini-4-neon",
          defaultAcpModelId: "gemini-4-neon-high",
          variants: [{ id: "high", acpModelId: "gemini-4-neon-high", thoughtLevel: "high" }],
        },
        {
          id: "claude-opus-5-5",
          name: "Claude Opus 5.5",
          family: "claude-opus-5-5",
          defaultAcpModelId: "claude-opus-5-5-medium",
          variants: [{ id: "medium", acpModelId: "claude-opus-5-5-medium", thoughtLevel: "medium" }],
        },
        {
          id: "gpt-oss-120b-medium",
          name: "GPT-OSS 120B (Medium)",
          family: "gpt-oss-120b-medium",
          defaultAcpModelId: "gpt-oss-120b-medium",
          variants: [],
        },
      ],
    })

    const mockCtx = {
      location: { directory: process.cwd() },
      storage: {
        get: (key: string) => Effect.sync(() => store.get(key) as never),
        set: (key: string, value: unknown) =>
          Effect.sync(() => {
            store.set(key, value)
          }),
        remove: (key: string) =>
          Effect.sync(() => {
            store.delete(key)
          }),
      },
      provider: {
        get: () => Effect.fail(new Error("not found") as never),
        reload: () => Effect.void,
        transform: (fn: (editor: ProviderEditor) => void) =>
          Effect.sync(() => {
            const editor: ProviderEditor = {
              list: () => [],
              get: (id) => {
                const entry = providers.get(id)
                if (!entry) return undefined
                return {
                  provider: entry.info,
                  models: new Map(entry.models.map((m) => [m.id, m])),
                }
              },
              add: (input) => {
                providers.set(input.info.id, { info: input.info, models: input.models })
              },
              update: () => {},
              remove: (id) => {
                providers.delete(id)
              },
              models: { set: () => {}, update: () => {}, remove: () => {} },
            }
            fn(editor)
          }),
      },
      aisdk: { hook: () => Effect.void },
      session: {
        get: () => Effect.succeed({ location: { directory: process.cwd() } } as never),
        update: () => Effect.void,
        hook: () => Effect.void,
      },
      permission: { assert: () => Effect.void },
      rpc: {
        register: () => Effect.sync(() => ({ events: { emit: () => Effect.void } }) as never),
      },
      event: { subscribe: () => Stream.empty },
    } as unknown as Context

    const scope = Effect.runSync(Scope.make())
    try {
      await Effect.runPromise(plugin.effect(mockCtx).pipe(Scope.provide(scope)))
      const models = providers.get(PROVIDER_ID)?.models ?? []
      const byId = new Map(models.map((m) => [String(m.id), m]))
      expect(byId.get("gemini-4-neon")?.limit).toEqual({ context: 1_048_576, output: 65_536 })
      expect(byId.get("claude-opus-5-5")?.limit).toEqual({ context: 1_000_000, output: 128_000 })
      expect(byId.get("gpt-oss-120b-medium")?.limit).toEqual({ context: 131_072, output: 32_768 })
      expect(byId.get("gpt-oss-120b-medium")?.capabilities.input).toEqual(["text"])
    } finally {
      await Effect.runPromise(Scope.close(scope, Option.none() as never))
    }
  })
})
