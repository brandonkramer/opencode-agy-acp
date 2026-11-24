import { AcpProcess, type AgyIO } from "./process.js"
import {
  isRecord,
  type AcpDiscoveredModel,
  type AcpModelVariant,
  type AcpSessionResult,
  type AgySettings,
} from "../types.js"

export const FALLBACK_MODELS: readonly AcpDiscoveredModel[] = [
  {
    id: "gemini-4-neon",
    name: "Gemini 4 Neon",
    family: "gemini-4-neon",
    defaultAcpModelId: "gemini-4-neon-high",
    variants: [
      { id: "low", acpModelId: "gemini-4-neon-low", thoughtLevel: "low" },
      { id: "medium", acpModelId: "gemini-4-neon-medium", thoughtLevel: "medium" },
      { id: "high", acpModelId: "gemini-4-neon-high", thoughtLevel: "high" },
      { id: "xhigh", acpModelId: "gemini-4-neon-xhigh", thoughtLevel: "xhigh" },
    ],
  },
  {
    id: "gemini-4-argon",
    name: "Gemini 4 Argon",
    family: "gemini-4-argon",
    defaultAcpModelId: "gemini-4-argon-medium",
    variants: [
      { id: "low", acpModelId: "gemini-4-argon-low", thoughtLevel: "low" },
      { id: "medium", acpModelId: "gemini-4-argon-medium", thoughtLevel: "medium" },
      { id: "high", acpModelId: "gemini-4-argon-high", thoughtLevel: "high" },
    ],
  },
  {
    id: "gemini-4-hydrogen",
    name: "Gemini 4 Hydrogen",
    family: "gemini-4-hydrogen",
    defaultAcpModelId: "gemini-4-hydrogen-high",
    variants: [
      { id: "low", acpModelId: "gemini-4-hydrogen-low", thoughtLevel: "low" },
      { id: "medium", acpModelId: "gemini-4-hydrogen-medium", thoughtLevel: "medium" },
      { id: "high", acpModelId: "gemini-4-hydrogen-high", thoughtLevel: "high" },
      { id: "xhigh", acpModelId: "gemini-4-hydrogen-xhigh", thoughtLevel: "xhigh" },
    ],
  },
  {
    id: "gemini-4-carbon",
    name: "Gemini 4 Carbon",
    family: "gemini-4-carbon",
    defaultAcpModelId: "gemini-4-carbon-medium",
    variants: [
      { id: "low", acpModelId: "gemini-4-carbon-low", thoughtLevel: "low" },
      { id: "medium", acpModelId: "gemini-4-carbon-medium", thoughtLevel: "medium" },
      { id: "high", acpModelId: "gemini-4-carbon-high", thoughtLevel: "high" },
    ],
  },
  {
    id: "gemini-pro-orchestrator-neon",
    name: "Orchestrator (Neon)",
    family: "gemini-pro-orchestrator-neon",
    defaultAcpModelId: "gemini-pro-orchestrator-neon",
    variants: [],
  },
  {
    id: "gemini-pro-orchestrator-g1-pro",
    name: "Orchestrator (Argon)",
    family: "gemini-pro-orchestrator-g1-pro",
    defaultAcpModelId: "gemini-pro-orchestrator-g1-pro",
    variants: [],
  },
  {
    id: "gemini-3.8-flash",
    name: "Gemini 3.8 Flash",
    family: "gemini-3.8-flash",
    defaultAcpModelId: "gemini-3.8-flash-medium",
    variants: [
      { id: "low", acpModelId: "gemini-3.8-flash-low", thoughtLevel: "low" },
      { id: "medium", acpModelId: "gemini-3.8-flash-medium", thoughtLevel: "medium" },
      { id: "high", acpModelId: "gemini-3.8-flash-high", thoughtLevel: "high" },
    ],
  },
  {
    id: "claude-opus-5-5",
    name: "Claude Opus 5.5",
    family: "claude-opus-5-5",
    defaultAcpModelId: "claude-opus-5-5-medium",
    variants: [
      { id: "low", acpModelId: "claude-opus-5-5-low", thoughtLevel: "low" },
      { id: "medium", acpModelId: "claude-opus-5-5-medium", thoughtLevel: "medium" },
      { id: "high", acpModelId: "claude-opus-5-5-high", thoughtLevel: "high" },
    ],
  },
  {
    id: "claude-sonnet-5-5",
    name: "Claude Sonnet 5.5",
    family: "claude-sonnet-5-5",
    defaultAcpModelId: "claude-sonnet-5-5-medium",
    variants: [
      { id: "low", acpModelId: "claude-sonnet-5-5-low", thoughtLevel: "low" },
      { id: "medium", acpModelId: "claude-sonnet-5-5-medium", thoughtLevel: "medium" },
      { id: "high", acpModelId: "claude-sonnet-5-5-high", thoughtLevel: "high" },
    ],
  },
  {
    id: "gpt-oss-120b-medium",
    name: "GPT-OSS 120B (Medium)",
    family: "gpt-oss-120b-medium",
    defaultAcpModelId: "gpt-oss-120b-medium",
    variants: [],
  },
]

function cleanDisplayName(raw: string | undefined, fallback: string): string {
  return (raw ?? fallback).replace(/\s*\(ACP\)\s*$/i, "").trim() || fallback
}

export function buildModelCatalog(sessionResult: AcpSessionResult | undefined): readonly AcpDiscoveredModel[] {
  const availableModels = sessionResult?.models?.availableModels ?? []
  const configOptions = sessionResult?.configOptions ?? []
  const modelOpt = configOptions.find((c) => c.id === "model")
  const thoughtOpt = configOptions.find((c) => c.id === "thought_level")
  const thoughtOrder: string[] = Array.isArray(thoughtOpt?.options)
    ? thoughtOpt.options.map((o) => String(o.value ?? ""))
    : ["low", "medium", "high", "xhigh"]

  const baseList: Array<{ value: string; name: string }> = Array.isArray(modelOpt?.options)
    ? modelOpt.options.map((o) => ({
        value: String(o.value ?? ""),
        name: String(o.name ?? o.value ?? ""),
      }))
    : []

  if (baseList.length === 0 && availableModels.length === 0) {
    return FALLBACK_MODELS
  }

  if (baseList.length === 0) {
    return availableModels.map((m) => ({
      id: m.modelId,
      name: cleanDisplayName(m.name, m.modelId),
      family: m.modelId,
      defaultAcpModelId: m.modelId,
      variants: [],
    }))
  }

  const availIds = new Set(availableModels.map((m) => m.modelId))
  const baseIds = new Set(baseList.map((b) => b.value))
  const matchedAvailIds = new Set<string>()

  const catalog: AcpDiscoveredModel[] = baseList.map((base) => {
    const variants: AcpModelVariant[] = []
    if (availIds.has(base.value)) {
      matchedAvailIds.add(base.value)
    }

    for (const level of thoughtOrder) {
      if (!level) continue
      const candidate = `${base.value}-${level}`
      if (availIds.has(candidate) && !baseIds.has(candidate)) {
        variants.push({ id: level, acpModelId: candidate, thoughtLevel: level })
        matchedAvailIds.add(candidate)
      }
    }

    const prefix = `${base.value}-`
    for (const m of availableModels) {
      if (!m.modelId.startsWith(prefix) || matchedAvailIds.has(m.modelId) || baseIds.has(m.modelId)) {
        continue
      }
      const suffix = m.modelId.slice(prefix.length)
      if (!suffix) continue
      variants.push({ id: suffix, acpModelId: m.modelId, thoughtLevel: suffix })
      matchedAvailIds.add(m.modelId)
    }

    const defaultAcpModelId = availIds.has(base.value)
      ? base.value
      : variants.find((v) => v.id === "medium")?.acpModelId ??
        variants.find((v) => v.id === "high")?.acpModelId ??
        variants[0]?.acpModelId ??
        base.value

    return {
      id: base.value,
      name: cleanDisplayName(base.name, base.value),
      family: base.value,
      variants,
      defaultAcpModelId,
    }
  })

  for (const m of availableModels) {
    if (matchedAvailIds.has(m.modelId) || baseIds.has(m.modelId)) continue
    catalog.push({
      id: m.modelId,
      name: cleanDisplayName(m.name, m.modelId),
      family: m.modelId,
      defaultAcpModelId: m.modelId,
      variants: [],
    })
  }

  return catalog
}

export function resolveAcpModelTarget(
  models: readonly AcpDiscoveredModel[],
  baseModelId: string,
  variant?: string,
): { readonly acpModelId: string; readonly baseModelId: string; readonly thoughtLevel?: string } {
  const found = models.find((m) => m.id === baseModelId)
  if (!found) {
    return {
      baseModelId,
      acpModelId: variant ? `${baseModelId}-${variant}` : baseModelId,
      thoughtLevel: variant || undefined,
    }
  }
  if (variant) {
    const matched = found.variants.find((v) => v.id === variant)
    if (matched) {
      return {
        baseModelId: found.id,
        acpModelId: matched.acpModelId,
        thoughtLevel: matched.thoughtLevel,
      }
    }
  }
  return {
    baseModelId: found.id,
    acpModelId: found.defaultAcpModelId,
    thoughtLevel: found.variants.find((v) => v.acpModelId === found.defaultAcpModelId)?.thoughtLevel,
  }
}

export function isValidCatalog(raw: unknown): raw is readonly AcpDiscoveredModel[] {
  if (!Array.isArray(raw) || raw.length === 0) return false
  return raw.every(
    (item) =>
      isRecord(item) &&
      typeof item.id === "string" &&
      typeof item.name === "string" &&
      typeof item.defaultAcpModelId === "string" &&
      Array.isArray(item.variants),
  )
}

export function normalizeCatalog(models: readonly AcpDiscoveredModel[]): readonly AcpDiscoveredModel[] {
  return models.map((m) => ({
    ...m,
    family: m.family || m.id,
  }))
}

export function catalogSignature(models: readonly AcpDiscoveredModel[]): string {
  return JSON.stringify(models.map((m) => [m.id, m.name, m.family, m.defaultAcpModelId, m.variants]))
}

export async function discoverAcpModels(
  cwd: string,
  settings: Pick<AgySettings, "binaryPath" | "includeDefaultCustomizations" | "startupTimeoutMs" | "traceLogPath">,
  io?: AgyIO,
): Promise<readonly AcpDiscoveredModel[]> {
  const proc = new AcpProcess({
    cwd,
    binaryPath: settings.binaryPath,
    processPerSession: false,
    includeDefaultCustomizations: settings.includeDefaultCustomizations,
    traceLogPath: settings.traceLogPath,
    io,
  })
  try {
    await proc.initialize(settings.startupTimeoutMs)
    await proc.authenticate(settings.startupTimeoutMs).catch(() => undefined)
    const sess = await proc.newSession({
      cwd,
      mcpServers: [],
      timeoutMs: settings.startupTimeoutMs,
    })
    const catalog = buildModelCatalog(sess)
    if (sess?.sessionId) {
      await proc.closeSession(String(sess.sessionId))
    }
    return catalog
  } finally {
    await proc.dispose()
  }
}
