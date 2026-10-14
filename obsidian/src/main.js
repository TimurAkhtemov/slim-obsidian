/*
 * SLIM Recorder — record, take notes, file the result.
 *
 * The plugin's entry. Obsidian loads ONE file, `obsidian/main.js`, and evals it with a shim
 * `require` that serves `obsidian` and CodeMirror and cannot resolve a relative path, so the
 * modules in `src/` are bundled into that file by esbuild (`npm run build`, options in
 * `build.mjs`) and the bundle is COMMITTED: the vault only ever copies a file that already
 * built, and `test/bundle.test.mjs` fails when it is stale.
 *
 * THE DIVISION OF LABOUR IS THE ARCHITECTURE (CLAUDE.md, "Capture"):
 * this plugin owns the UI and writes files through the Vault API. Every judgement — transcribe,
 * summarize, suggest a path — belongs to the Python server on 127.0.0.1.
 * A judgement that migrates into this plugin is a bug, because it becomes untestable by the
 * suite that protects the rest of SLIM.
 *
 * ⚠ AUDIO GOES INTO THE VAULT, NOT OVER HTTP. The server caps request bodies far below what a
 * recording weighs, so the file is written to disk here and only its PATH is posted.
 *
 * ⚠ AUDIO IS FLUSHED TO DISK AS IT ARRIVES, through Node's `fs` rather than the Vault API,
 * because Obsidian can only create or overwrite a binary file and never append. `isDesktopOnly`
 * is what licenses that. The alternative — holding every chunk in browser memory until Stop —
 * makes a crash during a two-hour meeting cost the entire meeting, and long meetings are the
 * use case.
 *
 * ⚠ THIS PLUGIN OWNS `slim chat`'s LIFECYCLE. Nothing else starts it, so recording with the
 * server down used to fail at the POST. It spawns on load if the port is free and kills the
 * process GROUP on unload — see `stopServer` in server.js.
 */
"use strict";

// The bundle's entry and nothing else. esbuild leaves an entry's own names at the bundle's top
// level, where every module's same-named import collides with them and is renamed (Notice ->
// Notice2); behind this one require, each module keeps its own names.
module.exports = require("./plugin");
