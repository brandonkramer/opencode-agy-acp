import { describe, expect, test } from "bun:test"
import path from "node:path"
import { AcpProcess, buildAgySpawnEnv, resolveAcpAuthMethod } from "../src/acp/process.js"
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
})
