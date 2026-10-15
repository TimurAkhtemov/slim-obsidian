// The live editor's widgets: removed words, ↺, a folded removal, See before, and a draft. `ui`
// calls back into the editor (hover, undo, reveal, fold, See before) at call time.
"use strict";

const { excerpt, beforeBlocks } = require("./diff");

const DRAFT_RENDER_MS = 250;              // a draft's Markdown is drawn again at most this often

function makeWidgets(WidgetType, host, ui) {
  // Words a change removed, struck, where they were.
  class Gone extends WidgetType {
    constructor(text, raw, path) { super(); this.text = text; this.raw = raw; this.path = path; }
    eq(other) { return other.text === this.text && other.raw === this.raw; }
    toDOM() {
      const span = document.createElement("span");
      span.className = "slim-del";
      span.textContent = this.text;
      if (!this.raw) {
        const holder = document.createElement("div");
        Promise.resolve(host.render(holder, this.text, this.path)).then(() => {
          const p = holder.querySelector("p") || holder;
          if (p.childNodes.length) span.replaceChildren(...p.childNodes);
        }).catch(() => {});
      }
      return span;
    }
  }

  const undoButton = (view, runId, id, tip) => {
    const button = document.createElement("button");
    button.className = "slim-undo";
    button.setAttribute("aria-label", `Undo: ${tip}`);
    if (host.icon) host.icon(button, "undo-2");
    if (!button.childNodes.length) button.textContent = "↺";
    const label = button.appendChild(document.createElement("span"));
    label.className = "slim-undo-tip";
    label.textContent = `Undo · ${tip}`;
    button.onmousedown = (event) => event.preventDefault();         // keep the caret where it is
    // The editor ignores events inside a widget: the button holds its change's hover itself.
    button.onmouseenter = () => ui.showHover(view, runId, id);
    button.onmouseleave = () => ui.leaveHover(view);
    button.onclick = (event) => {
      event.preventDefault();
      const done = ui.undo(view, runId, [id]);
      if (done === "refused") host.refused?.(view, runId, id);
    };
    return button;
  };

  // ↺ at the left of a change's first line, shown while the change is hovered.
  class Undo extends WidgetType {
    constructor(runId, id, tip, shown) { super(); this.runId = runId; this.id = id; this.tip = tip; this.shown = shown; }
    eq(other) { return other.runId === this.runId && other.id === this.id && other.tip === this.tip && other.shown === this.shown; }
    toDOM(view) {
      const holder = document.createElement("span");
      holder.className = `slim-undo-holder${this.shown ? " is-shown" : ""}`;
      holder.dataset.key = `${this.runId}\n${this.id}\n${this.tip}`;
      holder.appendChild(undoButton(view, this.runId, this.id, this.tip));
      return holder;
    }
    // Hovering only shows or hides it: the button under the pointer stays the same element.
    updateDOM(dom) {
      if (dom.dataset.key !== `${this.runId}\n${this.id}\n${this.tip}`) return false;
      dom.classList.toggle("is-shown", this.shown);
      return true;
    }
    ignoreEvent() { return true; }
  }

  // Removed text, folded into one struck line that opens on a click; a moved block's line says
  // so and a click goes to where it went.
  class Folded extends WidgetType {
    constructor(opts) { super(); Object.assign(this, opts); }
    eq(other) { return other.key === this.key && other.hovered === this.hovered; }
    get key() { return JSON.stringify(["runId", "id", "text", "moved", "open", "undo", "path"].map((k) => this[k])); }
    toDOM(view) {
      const box = document.createElement("div");
      box.className = `slim-removed slim-change${this.hovered ? " is-hovered" : ""}`;
      box.dataset.slimRun = this.runId;
      box.dataset.slimChange = this.id;
      box.dataset.key = this.key;
      box.onmouseenter = () => ui.showHover(view, this.runId, this.id);
      box.onmouseleave = () => ui.leaveHover(view);
      const line = box.appendChild(document.createElement("button"));
      line.className = "slim-removed-line";
      line.setAttribute("aria-expanded", String(Boolean(this.open)));
      const struck = line.appendChild(document.createElement("s"));
      struck.textContent = excerpt(this.text) || "(blank lines)";
      if (this.moved) line.appendChild(document.createTextNode(" · moved"));
      line.title = this.moved ? "Show where it went" : this.open ? "Fold" : "Show what was removed";
      line.onmousedown = (event) => event.preventDefault();
      line.onclick = (event) => {
        event.preventDefault();
        if (this.moved) ui.reveal(view, this.runId, this.id);
        else ui.toggleOpen(view, this.runId, this.id);
      };
      if (this.open && !this.moved) {
        const full = box.appendChild(document.createElement("div"));
        full.className = "slim-removed-full";
        Promise.resolve(host.render(full, this.text, this.path)).catch(() => { full.textContent = this.text; });
      }
      if (this.undo) {
        const holder = box.appendChild(document.createElement("span"));
        holder.className = `slim-undo-holder${this.hovered ? " is-shown" : ""}`;
        holder.appendChild(undoButton(view, this.runId, this.id, this.undo));
      }
      return box;
    }
    // Hovering only marks it: the box under the pointer stays the same element.
    updateDOM(dom) {
      if (dom.dataset.key !== this.key) return false;
      dom.classList.toggle("is-hovered", Boolean(this.hovered));
      dom.querySelector(".slim-undo-holder")?.classList.toggle("is-shown", Boolean(this.hovered));
      return true;
    }
    ignoreEvent() { return true; }
  }

  // See before: the note as it was, in place, read-only, with what changed since marked on it.
  class Before extends WidgetType {
    constructor(runId, text, path) { super(); this.runId = runId; this.text = text; this.path = path; }
    eq(other) { return other.runId === this.runId && other.text === this.text; }
    toDOM(view) {
      const root = document.createElement("div");
      root.className = "slim-before";
      const bar = root.appendChild(document.createElement("div"));
      bar.className = "slim-before-bar";
      const title = bar.appendChild(document.createElement("b"));
      title.textContent = "Before SLIM's edits";
      const note = bar.appendChild(document.createElement("span"));
      note.textContent = "read-only";
      const back = bar.appendChild(document.createElement("button"));
      back.className = "slim-link";
      back.textContent = "Back to now";
      back.onmousedown = (event) => event.preventDefault();
      back.onclick = (event) => { event.preventDefault(); ui.showBefore(view, this.runId, this.text, false); };
      for (const block of beforeBlocks(this.text, view.state.doc.toString())) {
        const el = root.appendChild(document.createElement("div"));
        if (block.kind === "added") {
          el.className = "slim-before-marker is-added";
          el.textContent = `+ added here: ${block.markdown}${block.count > 1 ? ` (and ${block.count - 1} more)` : ""}`;
          continue;
        }
        el.className = `slim-before-block${block.kind === "same" ? "" : ` slim-change is-${block.kind}`}`;
        Promise.resolve(host.render(el, block.markdown, this.path)).catch(() => { el.textContent = block.markdown; });
        if (block.kind === "moved") {
          const marker = root.appendChild(document.createElement("div"));
          marker.className = "slim-before-marker is-moved";
          marker.textContent = "moved";
        }
      }
      return root;
    }
    ignoreEvent() { return true; }
  }

  // The draft's Markdown, drawn again at most every DRAFT_RENDER_MS with the latest text; plain
  // text until the first drawing is in. Drawings can finish out of order: one replaces what is
  // shown only if it began later, and each is dropped (and its component) with the next.
  const paintDraft = (view, box, text, path) => {
    box.slimText = text;
    if (box.slimTimer) return;
    box.slimTimer = setTimeout(() => {
      box.slimTimer = null;
      box.slimAt = Date.now();
      const order = box.slimBegun = (box.slimBegun || 0) + 1;
      const shown = box.slimText, holder = document.createElement("div");
      const drawn = host.renderFor ? host.renderFor(holder, shown, path)
        : Promise.resolve(host.render(holder, shown, path)).then(() => null);
      Promise.resolve(drawn).then((drop) => {
        if (box.slimGone || order < (box.slimShown || 0)) { drop?.(); return; }
        box.slimShown = order;
        box.slimDrop?.();
        box.slimDrop = drop;
        box.replaceChildren(...holder.childNodes);
        view.requestMeasure();
      }).catch(() => { if (!box.slimGone && order >= (box.slimShown || 0)) box.textContent = shown; });
    }, Math.max(0, DRAFT_RENDER_MS - (Date.now() - (box.slimAt || 0))));
  };

  class Draft extends WidgetType {
    constructor(id, text, path) { super(); this.id = id; this.text = text; this.path = path; }
    eq(other) { return other.id === this.id && other.text === this.text; }
    toDOM(view) {
      const box = document.createElement("div");
      box.className = "slim-draft";
      box.dataset.draft = this.id;
      box.textContent = this.text;
      paintDraft(view, box, this.text, this.path);
      return box;
    }
    updateDOM(dom, view) {
      if (dom.dataset.draft !== this.id) return false;
      paintDraft(view, dom, this.text, this.path);
      return true;
    }
    destroy(dom) {
      dom.slimGone = true;
      clearTimeout(dom.slimTimer);
      dom.slimDrop?.();
    }
    ignoreEvent() { return true; }
  }

  return { Gone, Undo, Folded, Before, Draft };
}

module.exports = { DRAFT_RENDER_MS, makeWidgets };
