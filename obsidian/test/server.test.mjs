// `slim chat`'s lifecycle: the command line, the vault it is told, and a stale server said once.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { notices, setRequestUrl } from "./fakes.mjs";
import { plugin } from "./recorder-harness.mjs";

const require = createRequire(import.meta.url);
const { serverArgv, serverEnv } = require("../src/server.js");

test("plugin-started servers are tied to the current Obsidian process", () => {
  // The plugin never asks for --reload: a restart cuts requests in flight, so a save during
  // a recording loses that recording's filing. `slim chat --reload` is a terminal thing.
  assert.deepEqual(serverArgv({ repoPath: "/repo", port: 7546 }, 4321),
    ["--directory", "/repo", "run", "slim", "chat", "--port", "7546",
     "--parent-pid", "4321"]);
  assert.ok(!serverArgv({ repoPath: "/repo", port: 7546, devReload: true }, 4321)
    .includes("--reload"));
});

test("a plugin-started server is told which vault this plugin is in", () => {
  // The plugin sends vault-RELATIVE paths and the server now DISCOVERS its own vault. With
  // two vaults registered, a server left to discover would resolve those paths against the
  // wrong one — the audio it looks for is in the vault the plugin wrote to, not that one.
  const env = serverEnv("/Users/x/Vaults/Work", { PATH: "/usr/bin", SLIM_VAULT: "/stale" });
  assert.equal(env.SLIM_VAULT, "/Users/x/Vaults/Work");
  assert.equal(env.PATH, "/usr/bin");
});

// --- the stale-server guard ------------------------------------------------------------------
// Reuse is deliberate; a server OLDER THAN THE CODE being silently preferred is not. A server
// from the previous afternoon survived a Cmd+Q, was reused, and failed two recordings against
// a bug already fixed on disk (2026-08-23).

test("a stale server produces a warning naming the file and how to restart", async () => {
  setRequestUrl(async () => ({
    status: 200,
    json: { stale: true, started_at: "2026-08-22T18:05:38Z",
            newest_source: "2026-08-23T14:00:00Z", newest_file: "transcribe.py" },
  }));
  await plugin().warnIfStale();
  assert.equal(notices.length, 1);
  assert.match(notices[0], /transcribe\.py/);
  assert.match(notices[0], /lsof -tiTCP:7546/);
});

test("one stale server is said ONCE, not on every request that reuses it", async () => {
  // 2026-09-29: every copilot call re-ran the check and stacked another 30 s notice.
  const health = { pid: 40828, stale: true, started_at: "2026-09-28T11:19:26Z",
                   newest_source: "2026-09-29T13:14:20Z", newest_file: "skills.py" };
  setRequestUrl(async () => ({ status: 200, json: health }));
  const p = plugin();
  await p.warnIfStale();
  await p.warnIfStale();
  await p.warnIfStale();
  assert.equal(notices.length, 1);
  health.pid = 51234;                 // a DIFFERENT stale server is news again
  health.started_at = "2026-09-29T14:00:00Z";
  await p.warnIfStale();
  assert.equal(notices.length, 2);
});

test("a fresh server says nothing at all", async () => {
  setRequestUrl(async () => ({ status: 200, json: { stale: false } }));
  await plugin().warnIfStale();
  assert.equal(notices.length, 0);
});

test("a server too old to have /api/health is itself the proof", async () => {
  // ⚠ 404 is not an error to swallow: only a server predating the endpoint can answer it.
  setRequestUrl(async () => ({ status: 404, json: null }));
  await plugin().warnIfStale();
  assert.equal(notices.length, 1);
  assert.match(notices[0], /OLD code/);
});
