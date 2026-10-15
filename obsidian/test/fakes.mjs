/*
 * The stubs the plugin's tests run against. src/ requires `obsidian` and `@electron/remote` at
 * load, and neither exists outside Obsidian, so importing this file patches `Module._load` to
 * serve the fakes below. A harness imports it before anything requires src/.
 *
 * Two fake DOMs, kept apart on purpose: the recorder's reads text back through `.text` and
 * `setText`, the copilot's through `textContent` and `children`. Each harness picks how
 * MarkdownRenderer draws into its own.
 */
import Module from "node:module";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

export const SRC = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "src");

// The plugin's source text, for the few tests that assert on how the code is written: one
// src file ("recorder/view.js"), or all of them, in every folder, joined.
export function srcText(file) {
  const files = file ? [file]
    : readdirSync(SRC, { recursive: true }).filter((name) => name.endsWith(".js")).sort();
  return files.map((name) => readFileSync(path.join(SRC, name), "utf8")).join("\n");
}

// Live bindings: importers read them; only the functions below replace them.
export let notices = [];
export let noticeInstances = [];
export function resetNotices() { notices = []; }
export function resetNoticeInstances() { noticeInstances = []; }

// --- the recorder's DOM ---------------------------------------------------------------------

export function fakeContentEl() {
  const el = {
    classes: new Set(), children: [], isConnected: true, style: {}, events: {}, text: "",
    addClass(name) { this.classes.add(name); },
    removeClass(name) { this.classes.delete(name); },
    toggleClass(name, on) { if (on) this.classes.add(name); else this.classes.delete(name); },
    createDiv(options = {}) { const child = fakeContentEl(); child.options = options; if (options.cls) options.cls.split(" ").forEach((c) => child.classes.add(c)); this.children.push(child); return child; },
    createEl(tag, options = {}) { const child = fakeContentEl(); child.tag = tag; child.options = options; if (options.cls) options.cls.split(" ").forEach((c) => child.classes.add(c)); this.children.push(child); return child; },
    createSpan(options = {}) { const child = fakeContentEl(); child.tag = "span"; child.options = options; if (options.cls) options.cls.split(" ").forEach((c) => child.classes.add(c)); this.children.push(child); return child; },
    prepend(child) { this.children = [child, ...this.children.filter((x) => x !== child)]; },
    remove() { this.isConnected = false; },
    empty() { this.children = []; },
    setAttr() {}, setText(value) { this.text = value; },
    addEventListener(event, callback) { this.events[event] = callback; },
    querySelector() { return null; },
  };
  return el;
}

export function findClass(root, name) {
  if (root.classes && root.classes.has(name)) return root;
  for (const child of root.children || []) {
    const found = findClass(child, name);
    if (found) return found;
  }
  return null;
}

// --- the copilot's DOM ----------------------------------------------------------------------

// A DOM just large enough for render(): Obsidian's createEl/empty plus the handful of
// properties the copilot reads and writes. Tests walk `children`; nothing is queried by CSS.
export function fakeEl(tag = "div", options = {}) {
  const el = {
    tag, cls: options.cls || "", textContent: options.text ?? "", children: [], attrs: {},
    listeners: {}, value: "", open: false, scrollTop: 0, scrollHeight: 0, clientHeight: 100,
    createEl(childTag, childOptions = {}) {
      const child = fakeEl(childTag, childOptions);
      child.parent = el;
      el.children.push(child);
      return child;
    },
    createDiv(childOptions = {}) { return el.createEl("div", childOptions); },
    empty() { el.children = []; },
    addClass() {}, removeClass() {},
    setAttribute(name, value) { el.attrs[name] = value; },
    addEventListener(type, fn) { (el.listeners[type] ||= []).push(fn); },
    focus() { el.focused = true; },
    select() {},
    remove() { if (el.parent) el.parent.children = el.parent.children.filter((child) => child !== el); },
  };
  return el;
}

export function find(el, predicate) {
  if (predicate(el)) return el;
  for (const child of el.children) {
    const hit = find(child, predicate);
    if (hit) return hit;
  }
  return null;
}

// --- obsidian --------------------------------------------------------------------------------

// Indirections because src/ destructures `requestUrl` and `MarkdownRenderer` at load — tests
// need to swap them after.
let requestUrlImpl = async () => ({ status: 200, json: {} });
export function setRequestUrl(fn) { requestUrlImpl = fn; }
let renderImpl = async () => {};
export function setMarkdownRender(fn) { renderImpl = fn; }

export const obsidianStub = {
  Plugin: class {},
  ItemView: class {
    constructor(leaf) { this.leaf = leaf; this.app = leaf && leaf.app; this.contentEl = leaf && leaf.contentEl; }
    async onOpen() {}
    async onClose() {}
  },
  MarkdownView: class {
    constructor(leaf) {
      this.leaf = leaf;
      this.app = leaf && leaf.app;
      this.file = null;
      this.contentEl = fakeContentEl();
      this.loadedStates = [];
      this.saved = 0;
      this.editor = {
        inserted: [],
        cursor: null,
        focused: 0,
        replaceSelection: (text) => this.editor.inserted.push(text),
        lastLine: () => Math.max(0, String(this.file && this.file.text || "").split("\n").length - 1),
        setCursor: (cursor) => { this.editor.cursor = cursor; },
        focus: () => { this.editor.focused += 1; },
      };
    }
    async onOpen() {}
    async onClose() {}
    async save() { this.saved += 1; }
    async setState(state) {
      this.loadedStates.push(state);
      this.file = this.app && this.app.vault.getAbstractFileByPath(state.file);
    }
  },
  Modal: class {
    constructor(app) { this.app = app; this.contentEl = fakeEl("div"); }
    open() { this.onOpen(); }
    close() { this.onClose(); }
  },
  MarkdownRenderer: { render: (...args) => renderImpl(...args) },
  PluginSettingTab: class {},
  Setting: class {},
  Notice: class {
    constructor(msg) { notices.push(msg); this.noticeEl = fakeContentEl(); noticeInstances.push(this); }
    hide() { this.hidden = true; }
  },
  requestUrl: (...args) => requestUrlImpl(...args),
};

// --- @electron/remote -----------------------------------------------------------------------

// The SHARED session the handler is installed on. Every test asserts it is left at null.
export const installed = { handler: undefined };
const remoteStub = {
  session: {
    defaultSession: { setDisplayMediaRequestHandler(h) { installed.handler = h; } },
  },
};

const origLoad = Module._load;
Module._load = function (request) {
  if (request === "obsidian") return obsidianStub;
  if (request === "@electron/remote") return remoteStub;
  return origLoad.apply(this, arguments);
};
