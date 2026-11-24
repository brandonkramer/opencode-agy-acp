import { describe, expect, test } from "bun:test"
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
import plugin from "../src/index.js"
import { OPENCODE_QUESTION_TOOL, PACKAGE_ID, PROVIDER_ID } from "../src/types.js"

describe("Plugin.define hook wiring (src/index.ts)", () => {
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

      // 3. Context hook: Pure ACP clears system and retains only `question` tool
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
          subagent: { description: "Subagent", input: {} },
        },
      }
      await Effect.runPromise(contextHook!(acpContext))
      expect(acpContext.system).toEqual([])
      expect(Object.keys(acpContext.tools)).toEqual([OPENCODE_QUESTION_TOOL])

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
    } finally {
      await Effect.runPromise(Scope.close(scope, Option.none() as never))
    }
  })
})
