import { Model, Plugin, Provider } from "@opencode/plugin/effect"
import { Effect, Stream } from "effect"
import {
  sanitizeForAcpModel,
  sanitizeForNonAcpModel,
  summarizeTitleFromMessages,
} from "./acp/history.js"
import { SessionRegistry } from "./acp/sessions.js"
import { createAgyLanguageModel } from "./language-model.js"
import { AcpRpc } from "./rpc.js"
import {
  OPENCODE_QUESTION_TOOL,
  PACKAGE_ID,
  PROVIDER_ID,
  resolveSettings,
  type AcpDiscoveredModel,
} from "./types.js"

function buildModelEntries(
  providerID: Provider.ID,
  models: readonly AcpDiscoveredModel[],
  bridgeGeneration: string,
): Model.Info[] {
  return models.map((m) => {
    const defaults = Model.Info.default(providerID, Model.ID.make(m.id))
    return {
      ...defaults,
      family: Model.Family.make(m.family ?? m.id),
      name: m.name,
      // Inert generation marker ensures OpenCode refreshes LanguageModelV3 bindings on plugin reload.
      settings: { antigravityBridgeGeneration: bridgeGeneration },
      capabilities: {
        tools: true,
        input: ["text", "image"],
        output: ["text"],
      },
      variants: m.variants.map((v) => ({ id: Model.VariantID.make(v.id) })),
      limit: {
        context: 1_048_576,
        output: 65_536,
      },
    }
  })
}

export default Plugin.define({
  id: PROVIDER_ID,
  effect: (ctx) =>
    Effect.gen(function* () {
      const providerID = Provider.ID.make(PROVIDER_ID)
      const bridgeGeneration = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`

      const providerInfo = yield* ctx.provider.get({ providerID }).pipe(Effect.option)
      const initialSettings = resolveSettings(
        providerInfo._tag === "Some"
          ? (providerInfo.value.data as unknown as Record<string, unknown>)
          : undefined,
      )

      const registry = new SessionRegistry(ctx, {
        settings: initialSettings,
        onCatalogChanged: () => {
          Effect.runFork(ctx.provider.reload().pipe(Effect.catch(() => Effect.void)))
        },
      })

      yield* Effect.addFinalizer(() => Effect.promise(() => registry.disposeAll()))

      const cacheStatus = yield* Effect.promise(() => registry.loadCachedModels())

      yield* ctx.provider.transform((editor) => {
        if (editor.get(providerID)) {
          editor.remove(providerID)
        }
        const info = Provider.Info.empty(providerID)
        editor.add({
          info: {
            ...info,
            name: "Antigravity (ACP)",
            activation: "enabled",
            package: PACKAGE_ID,
          },
          models: buildModelEntries(providerID, registry.getModels(), bridgeGeneration),
        })
      })

      yield* ctx.aisdk.hook(
        "sdk",
        (input) =>
          Effect.sync(() => {
            const runtimeSettings = resolveSettings(input.options)
            input.sdk = {
              languageModel: (id: string) =>
                createAgyLanguageModel(id, registry, runtimeSettings),
            }
          }),
        { providerID },
      )

      yield* ctx.aisdk.hook(
        "language",
        (input) =>
          Effect.sync(() => {
            const runtimeSettings = resolveSettings(input.options)
            const id = String(input.model.modelID ?? input.model.id)
            input.language = createAgyLanguageModel(id, registry, runtimeSettings)
          }),
        { providerID },
      )

      yield* ctx.session.hook("context", (input) =>
        Effect.sync(() => {
          if (input.model.providerID === providerID) {
            sanitizeForAcpModel(input.messages)
            registry.updateContextSnapshot(input.sessionID, {
              agent: input.agent,
              tools: input.tools,
            })
            // Thin ACP: clear outer system instructions so agy owns its system prompt.
            input.system = []
            const state = registry.getState(input.sessionID)
            if (!state.codeModeEnabled) {
              // In Pure ACP keep OpenCode's native `question` tool if present so elicitation/create works.
              const questionTool = input.tools[OPENCODE_QUESTION_TOOL]
              input.tools = questionTool ? { [OPENCODE_QUESTION_TOOL]: questionTool } : {}
            }
          } else {
            sanitizeForNonAcpModel(input.messages)
          }
        }),
      )

      yield* ctx.session.hook(
        "model.request",
        (input) =>
          Effect.sync(() => {
            input.headers["x-opencode-session-id"] = input.sessionID
            input.headers["x-opencode-request-kind"] = input.kind
            input.headers["x-opencode-agent"] = input.agent
            input.headers["x-opencode-variant"] = input.model.variant ?? ""
          }),
        { providerID },
      )

      yield* ctx.session.hook(
        "title",
        (input) =>
          Effect.sync(() => {
            if (input.result) return
            const derived = summarizeTitleFromMessages(input.messages)
            if (derived) input.result = derived
          }),
        { providerID },
      )

      // Match claude-agent: agy handles upstream retries itself; outer OpenCode retries cause double-loops.
      yield* ctx.session.hook(
        "retry",
        (input) =>
          Effect.sync(() => {
            input.decision = { retry: false }
          }),
        { providerID },
      )

      yield* ctx.rpc.register(AcpRpc, {
        getState: (input) => {
          const { sessionID } = input as { readonly sessionID: string }
          return Effect.sync(() => registry.getState(sessionID))
        },
        getDiagnostics: (input) => {
          const { sessionID } = input as { readonly sessionID: string }
          return Effect.sync(() => registry.getDiagnostics(sessionID))
        },
        setMode: (input) => {
          const { sessionID, mode } = input as {
            readonly sessionID: string
            readonly mode: "auto" | "accept-edits" | "default" | "plan"
          }
          return Effect.promise(() => registry.setMode(sessionID, mode))
        },
        setCodeMode: (input) => {
          const { sessionID, codeModeEnabled } = input as {
            readonly sessionID: string
            readonly codeModeEnabled: boolean
          }
          return Effect.promise(() => registry.setCodeMode(sessionID, codeModeEnabled))
        },
        resetSession: (input) => {
          const { sessionID } = input as { readonly sessionID: string }
          return Effect.promise(() => registry.resetSession(sessionID))
        },
        reloadModels: () =>
          Effect.gen(function* () {
            const { models } = yield* Effect.promise(() => registry.refreshModels({ force: true }))
            yield* ctx.provider.reload()
            return { modelCount: models.length }
          }).pipe(Effect.orDie),
      })

      if (cacheStatus.stale) {
        yield* Effect.forkScoped(
          Effect.promise(() => registry.refreshModelsInBackground()).pipe(
            Effect.catch(() => Effect.void),
          ),
        )
      }

      yield* ctx.event
        .subscribe({
          types: [
            "session.execution.succeeded",
            "session.execution.failed",
            "session.execution.interrupted",
            "session.deleted",
          ],
        })
        .pipe(
          Stream.runForEach((evt) => {
            const sid = evt.data.sessionID
            if (!sid || !registry.hasSession(sid)) return Effect.void
            if (evt.type === "session.deleted") {
              return Effect.promise(() => registry.deleteSession(sid))
            }
            if (evt.type === "session.execution.succeeded") {
              return Effect.sync(() => registry.armIdle(sid))
            }
            return Effect.promise(async () => {
              await registry.handleInterrupt(
                sid,
                evt.type === "session.execution.interrupted"
                  ? "Interrupted by user"
                  : "Session execution failed",
              )
              registry.armIdle(sid)
            })
          }),
          Effect.catch(() => Effect.void),
          Effect.forkScoped,
        )

      yield* Effect.logInfo("opencode-agy-acp loaded", {
        directory: ctx.location.directory,
        models: registry.getModels().length,
        codeMode: initialSettings.codeMode,
        includeDefaultCustomizations: initialSettings.includeDefaultCustomizations,
      })
    }).pipe(Effect.orDie),
})
