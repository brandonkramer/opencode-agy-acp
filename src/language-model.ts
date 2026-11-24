import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3FinishReason,
  LanguageModelV3Usage,
} from "@ai-sdk/provider"
import type { SessionRegistry } from "./acp/sessions.js"
import { buildUsage } from "./acp/turn.js"
import { PROVIDER_ID, type AgySettings } from "./types.js"

export function createAgyLanguageModel(
  modelID: string,
  registry: SessionRegistry,
  runtimeSettings?: AgySettings,
): LanguageModelV3 {
  return {
    specificationVersion: "v3",
    provider: PROVIDER_ID,
    modelId: modelID,
    supportedUrls: {},

    async doGenerate(options: LanguageModelV3CallOptions) {
      const { stream } = await this.doStream(options)
      const reader = stream.getReader()
      const texts: string[] = []
      const reasonings: string[] = []
      let finishReason: LanguageModelV3FinishReason = { unified: "stop", raw: "stop" }
      let usage: LanguageModelV3Usage = buildUsage()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        if (value.type === "text-delta") texts.push(value.delta)
        else if (value.type === "reasoning-delta") reasonings.push(value.delta)
        else if (value.type === "finish") {
          finishReason = value.finishReason
          usage = value.usage
        } else if (value.type === "error") {
          throw value.error
        }
      }

      const content: Array<
        { readonly type: "reasoning"; readonly text: string } | { readonly type: "text"; readonly text: string }
      > = []
      if (reasonings.length > 0) content.push({ type: "reasoning", text: reasonings.join("") })
      if (texts.length > 0) content.push({ type: "text", text: texts.join("") })

      return {
        content,
        finishReason,
        usage,
        warnings: [],
      }
    },

    async doStream(options: LanguageModelV3CallOptions) {
      const headers = options.headers ?? {}
      const sessionID = String(headers["x-opencode-session-id"] ?? "__default__")
      const variant = String(headers["x-opencode-variant"] ?? "") || undefined
      const stream = await registry.startOrResumeTurn({
        sessionID,
        modelID,
        variant,
        prompt: options.prompt,
        abortSignal: options.abortSignal,
        runtimeSettings,
      })
      return { stream }
    },
  }
}
