import { describe, expect, test } from "bun:test"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import {
  AcpProcess,
  buildAgySpawnEnv,
  ensureIsolatedGeminiDir,
  resolveAcpAuthMethod,
} from "../src/acp/process.js"
import { AcpStallError } from "../src/types.js"

const FAKE_AGY = path.resolve(import.meta.dir, "fake-agy.ts")

describe("AcpProcess & env", () => {
  test("buildAgySpawnEnv scrubs colliding API keys for oauth-personal", () => {
    const env = buildAgySpawnEnv(
      {
        PATH: "/usr/bin",
        GEMINI_API_KEY: "secret",
        GOOGLE_API_KEY: "secret2",
        GOOGLE_APPLICATION_CREDENTIALS: "/tmp/creds.json",
        ANTIGRAVITY_ACP_AUTH_METHOD: "oauth-personal",
      },
      "darwin",
    )
    expect(resolveAcpAuthMethod({ GEMINI_API_KEY: "x" })).toBe("gemini-api-key")
    expect(env.GEMINI_API_KEY).toBeUndefined()
    expect(env.GOOGLE_API_KEY).toBeUndefined()
    expect(env.GOOGLE_APPLICATION_CREDENTIALS).toBeUndefined()
  })

  test("watchdog fires AcpStallError with trace when prompt hangs, and pauses/resumes cleanly", async () => {
    const proc = new AcpProcess({
      cwd: process.cwd(),
      binaryPath: FAKE_AGY,
      processPerSession: false,
      includeDefaultCustomizations: false,
    })
    try {
      await proc.initialize(5_000)
      await proc.authenticate(5_000)
      const sess = await proc.newSession({ cwd: process.cwd() })
      const sid = String(sess.sessionId)

      // Pausing keeps watchdog from firing while paused
      proc.pauseWatchdog()
      expect(proc.isWatchdogPaused).toBe(true)
      proc.resumeWatchdog()
      expect(proc.isWatchdogPaused).toBe(false)

      // Activity-based watchdog survives slow chunks whose total > idleWatchdogMs
      const okRes = await proc.prompt(
        sid,
        [{ type: "text", text: "[SCENARIO:SLOW_CHUNKS]" }],
        { idleWatchdogMs: 90 },
      )
      expect(okRes.stopReason).toBe("end_turn")

      // Silence triggers AcpStallError with wire trace
      let caught: unknown
      try {
        await proc.prompt(
          sid,
          [{ type: "text", text: "[SCENARIO:STALL]" }],
          { idleWatchdogMs: 100 },
        )
      } catch (err) {
        caught = err
      }
      expect(caught).toBeInstanceOf(AcpStallError)
      const stall = caught as AcpStallError
      expect(stall.idleMs).toBe(100)
      expect(stall.recentTrace.length).toBeGreaterThan(0)
      expect(stall.message).toContain("Recent ACP wire trace")
    } finally {
      await proc.dispose()
    }
  })

  test("rejects pending RPC immediately if stdin is destroyed before write", async () => {
    const proc = new AcpProcess({
      cwd: process.cwd(),
      binaryPath: FAKE_AGY,
    })
    await proc.dispose()
    await expect(proc.initialize(500)).rejects.toThrow(/closed/i)
  })

  test("ensureIsolatedGeminiDir creates empty mcp_config.json with symlinked antigravity-cli and fails open on conflict", () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "agy-iso-test-"))
    try {
      const fakeHome = path.join(tmp, "home")
      const fakeCache = path.join(tmp, "cache")
      fs.mkdirSync(path.join(fakeHome, ".gemini", "antigravity-cli"), { recursive: true })
      fs.writeFileSync(path.join(fakeHome, ".gemini", "antigravity-cli", "settings.json"), "{}\n")

      const iso = ensureIsolatedGeminiDir({ HOME: fakeHome, XDG_CACHE_HOME: fakeCache })
      expect(iso).toBe(path.join(fakeCache, "opencode-agy-acp", "isolated-gemini"))
      expect(fs.readFileSync(path.join(iso!, "config", "mcp_config.json"), "utf8")).toBe(
        '{"mcpServers":{}}\n',
      )
      expect(fs.existsSync(path.join(iso!, "config", ".migrated"))).toBe(true)
      const linkPath = path.join(iso!, "antigravity-cli")
      expect(fs.lstatSync(linkPath).isSymbolicLink()).toBe(true)
      expect(fs.existsSync(path.join(linkPath, "settings.json"))).toBe(true)

      // Calling a second time is idempotent
      expect(ensureIsolatedGeminiDir({ HOME: fakeHome, XDG_CACHE_HOME: fakeCache })).toBe(iso)

      // Refuses to overwrite a non-symlink directory placed at antigravity-cli
      fs.unlinkSync(linkPath)
      fs.mkdirSync(linkPath)
      expect(ensureIsolatedGeminiDir({ HOME: fakeHome, XDG_CACHE_HOME: fakeCache })).toBeUndefined()
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true })
    }
  })

  test("AcpProcess passes --gemini-dir only when isolateUserMcp is true", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "agy-iso-proc-"))
    const argsLog = path.join(tmp, "args.log")
    const prevArgs = process.env.FAKE_AGY_ARGS_LOG
    process.env.FAKE_AGY_ARGS_LOG = argsLog
    const fakeHome = path.join(tmp, "home")
    const fakeCache = path.join(tmp, "cache")
    const io = {
      spawnProcess: (await import("node:child_process")).spawn,
      resolveBinary: () => FAKE_AGY,
      env: () => ({ ...process.env, HOME: fakeHome, XDG_CACHE_HOME: fakeCache }),
    }
    const p1 = new AcpProcess({ cwd: tmp, binaryPath: FAKE_AGY, isolateUserMcp: true, io })
    let p2: AcpProcess | undefined
    try {
      await p1.initialize(5_000)
      p2 = new AcpProcess({ cwd: tmp, binaryPath: FAKE_AGY, isolateUserMcp: false, io })
      await p2.initialize(5_000)
      const lines = fs
        .readFileSync(argsLog, "utf8")
        .trim()
        .split("\n")
        .map((l) => JSON.parse(l) as string[])
      expect(lines).toHaveLength(2)
      expect(lines[0]).toContain("--gemini-dir")
      expect(lines[1]).not.toContain("--gemini-dir")
    } finally {
      if (prevArgs === undefined) delete process.env.FAKE_AGY_ARGS_LOG
      else process.env.FAKE_AGY_ARGS_LOG = prevArgs
      await Promise.allSettled([p1.dispose(), ...(p2 ? [p2.dispose()] : [])])
      fs.rmSync(tmp, { recursive: true, force: true })
    }
  })
})
