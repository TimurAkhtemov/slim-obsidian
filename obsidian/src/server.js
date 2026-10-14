// `slim chat`'s lifecycle: started by the plugin when nothing serves the port, warned about when
// a reused one is stale. The plugin's server methods live here and are attached in main.js.
"use strict";

const { Notice, requestUrl } = require("obsidian");
const { spawn } = require("child_process");
const net = require("net");
const nodePath = require("path");

function serverArgv(settings, parentPid = process.pid) {
  return ["--directory", settings.repoPath, "run", "slim", "chat",
          "--port", String(settings.port), "--parent-pid", String(parentPid)];
}

// ⚠ The server DISCOVERS its own vault (slim/vaultpath.py). This plugin sends vault-RELATIVE
// paths, so a server that discovered a different vault than this one would look for the audio
// in a vault nobody wrote to, and file the note there. The plugin knows which vault it is in;
// it says so, and the override wins over discovery.
function serverEnv(vaultBase, env = process.env) {
  return Object.assign({}, env, { SLIM_VAULT: vaultBase });
}

// A TCP connect is the only honest test of "is
// something serving this port" — a pid file lies after a crash.
function portOpen(port) {
  return new Promise((resolve) => {
    const sock = net.connect({ port, host: "127.0.0.1" }, () => {
      sock.end();
      resolve(true);
    });
    sock.on("error", () => resolve(false));
    sock.setTimeout(1000, () => {
      sock.destroy();
      resolve(false);
    });
  });
}

async function waitForPort(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await portOpen(port)) return true;
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}

const serverMethods = {
  // ---- the server's lifecycle ------------------------------------------------------------

  /* Nothing else starts `slim chat`: there is no launchd agent, and the Electron shell that
   * used to do it was cut on 2026-08-27. Without this, recording with the server down fails
   * at the POST — the audio survives in the staging folder but recovery is by hand.
   *
   * Not a memory decision: measured 2026-08-06, the server is 31 MB and holds NO model. Ollama
   * loads weights on demand and evicts on its own keep_alive.
   */
  ensureServer() {
    // One spawn at a time. Both `post()` calls of a single recording can arrive here at once,
    // and two spawns would race for one port.
    if (!this._ensuring) {
      this._ensuring = this._ensureServer().finally(() => { this._ensuring = null; });
    }
    return this._ensuring;
  },

  async _ensureServer() {
    const port = this.settings.port;
    // A CLI-started server is REUSED, not duplicated — the second bind would fail anyway, and
    // killing their own `uv run slim chat` on unload would be worse than not managing it at all.
    // ⚠ waitForPort, not a single portOpen: `--reload` closes the port for a beat on every
    // restart, and a one-shot probe there reads "nothing serving" and spawns a SECOND server
    // that then fights the reloader's respawn for the bind. 3s covers a restart. It probes
    // once immediately, so a live server still returns at once; the 3s is paid only when
    // nothing is listening at all, and that path is the un-awaited one at onload().
    if (await waitForPort(port, 3000)) {
      this.warnIfStale();          // not awaited: a reused server must not delay a recording
      return "reused";
    }

    try {
      // ⚠ detached: its own process GROUP. `uv run` wraps
      // the real python server, so a plain child.kill() reaches only uv — measured there, the
      // server outlived the app. See stopServer().
      const argv = serverArgv(this.settings);
      const base = this.vaultBasePath();
      // ⚠ stdio is "ignore", so the server's own output goes nowhere from here. Run
      // `uv run slim chat` in a terminal when you want to watch it work — and `--reload`
      // there too, which is the only place it ever belonged (a restart cuts requests in
      // flight, so a save during a recording loses that recording's filing).
      this.server = spawn(this.settings.uvPath, argv,
                          { stdio: "ignore", detached: true,
                            env: base ? serverEnv(base) : process.env });
    } catch (e) {
      this.server = null;
      new Notice(`SLIM: could not launch ${this.settings.uvPath} — ${(e && e.message) || e}`,
                 20000);
      return "failed";
    }
    const child = this.server;
    child.on("error", (e) => {
      // A bad uv path fails here, asynchronously, not at spawn().
      if (this.server === child) this.server = null;
      console.error("[slim] server spawn", e);
      new Notice(`SLIM: the server did not start — check the repo and uv paths in settings. ` +
                 `(${(e && e.message) || e})`, 20000);
    });
    child.on("exit", () => { if (this.server === child) this.server = null; });

    if (await waitForPort(port, 30000)) return "started";
    new Notice(`SLIM: the server did not come up on 127.0.0.1:${port} within 30s. Check that ` +
               "Ollama is running and that `uv run slim chat` works from the repo.", 20000);
    return "timeout";
  },

  /* ⚠ A REUSED SERVER CAN BE OLDER THAN THE CODE, and nothing about it looks wrong.
   *
   * Reuse is deliberate — it is what stops this plugin killing a terminal `slim chat --reload`.
   * But `uv run` reads Python from disk at spawn time and never hot-reloads, and "the plugin
   * owns the lifecycle" turned out not to hold: Obsidian does not reliably call `onunload` when
   * the whole APP quits, so a server started the previous afternoon survived a Cmd+Q and a
   * relaunch, was reused, and failed two recordings against a bug already fixed on disk
   * (2026-08-23). In July the same trap cost a full day of debugging.
   *
   * It WARNS rather than restarting: the server on the port might be their, with a reloader
   * attached, and killing that out from under them is its own surprise. ONCE PER SERVER: the
   * check runs on every request that reuses the server, and saying it each time stacked a
   * column of 30 s notices over the sidebar (2026-09-29).
   */
  /* This vault's own base path, or null on a mobile adapter that has none. `openDiskFile()`
   * already resolves audio against it; the server has to agree on the same directory. */
  vaultBasePath() {
    try {
      const adapter = this.app.vault.adapter;
      if (typeof adapter.getBasePath !== "function") return null;
      return nodePath.resolve(adapter.getBasePath());
    } catch (e) {
      console.error("[slim] no vault base path", e);
      return null;
    }
  },

  async warnIfStale() {
    // ⚠ Deliberately NOT this.post(): that calls ensureServer() and would recurse.
    const res = await requestUrl({
      url: `http://127.0.0.1:${this.settings.port}/api/health`,
      method: "GET",
      throw: false,
    }).catch(() => null);
    // A server old enough to predate this endpoint answers 404 — which is itself proof that it
    // is stale, and worth saying so rather than swallowing.
    if (!res || res.status === 404) {
      this.sayOnce("predates-health",
        "SLIM: the running server predates /api/health, so it is serving OLD code. " +
        `Restart it:\n\nlsof -tiTCP:${this.settings.port} -sTCP:LISTEN | xargs kill`);
      return;
    }
    const health = res.json;
    if (!health) return;
    // A REUSED server resolved its own vault, possibly before this vault was opened or moved.
    // Wrong vault is worse than old code: the recording is filed somewhere nobody is looking.
    // Python's Path.resolve() follows symlinks; Node's nodePath.resolve() does not. Use
    // fs.realpathSync to match, or fall back to the lexical resolve if it fails.
    const base = this.vaultBasePath();
    const fs = require("fs");
    let resolvedBase = base;
    try { if (base) resolvedBase = fs.realpathSync(base); } catch {}
    const killCmd = `lsof -tiTCP:${this.settings.port} -sTCP:LISTEN | xargs kill`;
    if (resolvedBase && health.vault && health.vault !== resolvedBase) {
      this.sayOnce(`vault:${health.pid}:${health.started_at}`,
        `SLIM: the server on port ${this.settings.port} is using the vault ${health.vault}, ` +
        `but this vault is ${base}. Recordings would be filed in the wrong one. Restart it:` +
        `\n\n${killCmd}`);
      return;
    }
    if (!health.stale) return;
    this.sayOnce(`stale:${health.pid}:${health.started_at}`,
      `SLIM: the running server started ${health.started_at} but ${health.newest_file} changed ` +
      `${health.newest_source}. It is serving OLD code. Restart it:\n\n` + killCmd);
  },

  sayOnce(key, text) {
    if (this.staleSaid === key) return;
    this.staleSaid = key;
    new Notice(text, 30000);
  },

  stopServer() {
    const child = this.server;
    this.server = null;
    if (!child || child.pid == null) return;
    try {
      process.kill(-child.pid, "SIGTERM");   // the whole group: uv AND the python server
    } catch (e) {
      try { child.kill(); } catch (e2) { /* already gone */ }
    }
  },
};

module.exports = { serverArgv, serverEnv, portOpen, waitForPort, serverMethods };
