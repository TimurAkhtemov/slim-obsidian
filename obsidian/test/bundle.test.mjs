// The committed bundle: built from src/, and loadable by Obsidian's shim require.
import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { builtinModules, createRequire } from "node:module";
import "./fakes.mjs";

const OBSIDIAN = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const BUNDLE = path.join(OBSIDIAN, "main.js");
// build.mjs imports esbuild, a devDependency: missing until `npm install` has run here.
const build = await import("../build.mjs").catch((error) => error);

test("main.js is exactly what src/ builds to", async () => {
  if (build instanceof Error) assert.fail(`esbuild is missing: run npm install in obsidian/ (${build.message})`);
  const { build: esbuild } = await import("esbuild");
  const result = await esbuild({ ...build.options, write: false });
  const built = result.outputFiles[0].text;
  assert.ok(built === fs.readFileSync(BUNDLE, "utf8"), "main.js is stale: run npm run build in obsidian/");
});

test("the bundle requires only what Obsidian's loader serves", () => {
  // Obsidian's shim `require` serves `obsidian`, CodeMirror and Electron, and hands the rest to
  // the renderer's global require, which knows Node's builtins and nothing in the plugin folder.
  // A relative `require("./chat/view")` throws before the plugin class exists, which takes the
  // RECORDER down with it; Node resolves relative paths, so only this assertion sees it.
  const source = fs.readFileSync(BUNDLE, "utf8");
  assert.doesNotMatch(source, /require\(\s*["']\.\.?\//);
  const required = [...source.matchAll(/\brequire\(\s*["']([^"']+)["']\s*\)/g)].map((m) => m[1]);
  assert.ok(required.length > 0);
  const served = (name) => name === "obsidian" || name === "electron" || name === "@electron/remote"
    || name.startsWith("@codemirror/") || builtinModules.includes(name.replace(/^node:/, ""));
  assert.deepEqual([...new Set(required.filter((name) => !served(name)))], []);
});

test("the bundle loads as the plugin class, and it is the only plugin code Obsidian sees", () => {
  const Plugin = createRequire(import.meta.url)(BUNDLE);
  assert.equal(typeof Plugin, "function");
  assert.equal(typeof Plugin.prototype.onload, "function");
  const scripts = fs.readdirSync(OBSIDIAN).filter((name) => name.endsWith(".js")).sort();
  assert.deepEqual(scripts, ["main.js"]);
});
