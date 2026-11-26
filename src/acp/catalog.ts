import fs from "node:fs"
import os from "node:os"
import path from "node:path"
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
    id: "gemini-pro-orchestrator",
    name: "Orchestrator (Carbon)",
    family: "gemini-pro-orchestrator",
    defaultAcpModelId: "gemini-pro-orchestrator",
    variants: [],
  },
  {
    id: "gemini-pro-orchestrator-argon",
    name: "Orchestrator (Argon)",
    family: "gemini-pro-orchestrator-argon",
    defaultAcpModelId: "gemini-pro-orchestrator-argon",
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

export interface AcpModelProfile {
  readonly limit: {
    readonly context: number
    readonly output: number
  }
  readonly capabilities: {
    readonly tools: boolean
    readonly input: readonly string[]
    readonly output: readonly string[]
  }
}

/** Resolve per-family context/output limits and input modalities for discovered ACP models. */
export function resolveModelProfile(modelId: string): AcpModelProfile {
  const id = modelId.trim().toLowerCase()
  if (id.startsWith("gpt-oss")) {
    return {
      limit: { context: 131_072, output: 32_768 },
      capabilities: { tools: true, input: ["text"], output: ["text"] },
    }
  }
  if (id.startsWith("claude-")) {
    const output = /^claude-(?:opus|sonnet|fable)-5\b/.test(id) ? 128_000 : 64_000
    return {
      limit: { context: 1_000_000, output },
      capabilities: { tools: true, input: ["text", "image"], output: ["text"] },
    }
  }
  if (id.startsWith("gemini-")) {
    return {
      limit: { context: 1_048_576, output: 65_536 },
      capabilities: { tools: true, input: ["text", "image"], output: ["text"] },
    }
  }
  if (id.startsWith("gpt-")) {
    return {
      limit: { context: 400_000, output: 128_000 },
      capabilities: { tools: true, input: ["text", "image"], output: ["text"] },
    }
  }
  return {
    limit: { context: 200_000, output: 65_536 },
    capabilities: { tools: true, input: ["text", "image"], output: ["text"] },
  }
}

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

const DISK_CATALOG_MAX_ENTRY_AGE_MS = 7 * 24 * 60 * 60 * 1000
const DISK_CATALOG_LOCK_STALE_MS = 15_000

export function resolveDiskCatalogCachePath(
  env: NodeJS.ProcessEnv = process.env,
): string | undefined {
  if (env.AGY_ACP_CATALOG_CACHE_PATH !== undefined) {
    const raw = env.AGY_ACP_CATALOG_CACHE_PATH.trim()
    if (!raw || raw === "none" || raw === "off" || raw === "0") return undefined
    return raw
  }
  const home = env.HOME?.trim() || env.USERPROFILE?.trim() || os.homedir()
  if (!home) return undefined
  const cacheRoot = env.XDG_CACHE_HOME?.trim() || path.join(home, ".cache")
  return path.join(cacheRoot, "opencode-agy-acp", "model-catalog-v1.json")
}

export function readDiskCachedCatalog(
  cacheKey: string,
  ttlMs: number,
  cachePath?: string,
): { readonly models: readonly AcpDiscoveredModel[]; readonly updatedAt: number } | undefined {
  const filePath = cachePath ?? resolveDiskCatalogCachePath()
  if (!filePath) return undefined
  try {
    const raw = JSON.parse(fs.readFileSync(filePath, "utf8")) as unknown
    if (!isRecord(raw) || raw.version !== 1 || !isRecord(raw.entries)) return undefined
    const entry = raw.entries[cacheKey]
    if (!isRecord(entry) || typeof entry.updatedAt !== "number" || !isValidCatalog(entry.models)) {
      return undefined
    }
    if (Date.now() - entry.updatedAt > ttlMs) return undefined
    return {
      models: normalizeCatalog(entry.models),
      updatedAt: entry.updatedAt,
    }
  } catch {
    return undefined
  }
}

export function writeDiskCachedCatalog(
  cacheKey: string,
  models: readonly AcpDiscoveredModel[],
  updatedAt: number,
  cachePath?: string,
): void {
  const filePath = cachePath ?? resolveDiskCatalogCachePath()
  if (!filePath || models.length === 0) return
  try {
    const dir = path.dirname(filePath)
    fs.mkdirSync(dir, { recursive: true, mode: 0o700 })
    const entries: Record<string, { readonly updatedAt: number; readonly models: readonly AcpDiscoveredModel[] }> = {}
    const now = Date.now()
    try {
      const existing = JSON.parse(fs.readFileSync(filePath, "utf8")) as unknown
      if (isRecord(existing) && existing.version === 1 && isRecord(existing.entries)) {
        for (const [k, v] of Object.entries(existing.entries)) {
          if (
            isRecord(v) &&
            typeof v.updatedAt === "number" &&
            now - v.updatedAt <= DISK_CATALOG_MAX_ENTRY_AGE_MS &&
            isValidCatalog(v.models)
          ) {
            entries[k] = { updatedAt: v.updatedAt, models: normalizeCatalog(v.models) }
          }
        }
      }
    } catch {}
    entries[cacheKey] = { updatedAt, models: normalizeCatalog(models) }
    const tmpPath = `${filePath}.${process.pid}.${Math.random().toString(36).slice(2, 8)}.tmp`
    try {
      fs.writeFileSync(tmpPath, JSON.stringify({ version: 1, entries }), { encoding: "utf8", mode: 0o600 })
      fs.renameSync(tmpPath, filePath)
    } catch {
      try {
        fs.unlinkSync(tmpPath)
      } catch {}
    }
  } catch {}
}

/**
 * Best-effort cooperative cross-process lock around cold model-catalog discovery so
 * concurrent OpenCode processes/worktrees do not launch duplicate `agy acp` spawns.
 */
export function tryAcquireDiskCatalogLock(
  cachePath?: string,
): { readonly acquired: boolean; readonly release: () => void } {
  const filePath = cachePath ?? resolveDiskCatalogCachePath()
  if (!filePath) return { acquired: true, release: () => {} }
  const lockPath = `${filePath}.lock`
  try {
    fs.mkdirSync(path.dirname(filePath), { recursive: true, mode: 0o700 })
    const tryCreate = (): boolean => {
      try {
        const fd = fs.openSync(lockPath, "wx", 0o600)
        try {
          fs.writeSync(fd, `${process.pid}\n`)
        } finally {
          fs.closeSync(fd)
        }
        return true
      } catch (err) {
        if ((err as NodeJS.ErrnoException)?.code !== "EEXIST") return true
        return false
      }
    }
    if (!tryCreate()) {
      try {
        const st = fs.statSync(lockPath)
        if (Date.now() - st.mtimeMs > DISK_CATALOG_LOCK_STALE_MS) {
          fs.unlinkSync(lockPath)
          if (!tryCreate()) return { acquired: false, release: () => {} }
        } else {
          return { acquired: false, release: () => {} }
        }
      } catch {
        return { acquired: true, release: () => {} }
      }
    }
    let released = false
    return {
      acquired: true,
      release: () => {
        if (released) return
        released = true
        try {
          fs.unlinkSync(lockPath)
        } catch {}
      },
    }
  } catch {
    return { acquired: true, release: () => {} }
  }
}

export async function waitForDiskCachedCatalog(
  cacheKey: string,
  ttlMs: number,
  cachePath?: string,
  maxWaitMs = 4_000,
): Promise<{ readonly models: readonly AcpDiscoveredModel[]; readonly updatedAt: number } | undefined> {
  const filePath = cachePath ?? resolveDiskCatalogCachePath()
  if (!filePath) return undefined
  const lockPath = `${filePath}.lock`
  const deadline = Date.now() + maxWaitMs
  while (Date.now() < deadline) {
    const found = readDiskCachedCatalog(cacheKey, ttlMs, filePath)
    if (found) return found
    if (!fs.existsSync(lockPath)) break
    await new Promise<void>((resolve) => {
      const t = setTimeout(resolve, 120)
      t.unref?.()
    })
  }
  return readDiskCachedCatalog(cacheKey, ttlMs, filePath)
}

export async function discoverAcpModels(
  cwd: string,
  settings: Pick<
    AgySettings,
    "binaryPath" | "includeDefaultCustomizations" | "isolateUserMcp" | "startupTimeoutMs" | "traceLogPath"
  >,
  io?: AgyIO,
): Promise<readonly AcpDiscoveredModel[]> {
  const proc = new AcpProcess({
    cwd,
    binaryPath: settings.binaryPath,
    processPerSession: false,
    includeDefaultCustomizations: settings.includeDefaultCustomizations,
    isolateUserMcp: settings.isolateUserMcp,
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
