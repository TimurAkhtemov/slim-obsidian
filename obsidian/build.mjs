/*
 * Bundles src/ into main.js, the one file Obsidian loads. Obsidian evals main.js with a shim
 * `require` that serves `obsidian` and CodeMirror and cannot resolve a relative path, so the
 * plugin's own modules are joined here and main.js is COMMITTED: the vault only ever copies a
 * file that already built. test/bundle.test.mjs rebuilds with these options and fails when
 * main.js is stale.
 */
import { build } from "esbuild";
import { fileURLToPath } from "node:url";
import path from "node:path";

const HERE = path.dirname(fileURLToPath(import.meta.url));

export const options = {
  absWorkingDir: HERE,
  entryPoints: ["src/main.js"],
  bundle: true,
  format: "cjs",
  platform: "node",
  target: "es2022",
  minify: false,
  sourcemap: false,
  outfile: "main.js",
  external: ["obsidian", "electron", "@electron/remote", "@codemirror/*"],
  banner: { js: "/* generated from src/ by npm run build; do not edit */" },
  logLevel: "warning",
};

if (process.argv[1] === fileURLToPath(import.meta.url)) await build(options);
