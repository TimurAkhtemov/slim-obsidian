// RecorderView: the review card and the tabbed note it adjusts.
"use strict";

const { MarkdownRenderer, Notice } = require("obsidian");
const { markdownBlockShortcut, stampSourceAtoms, markdownFromEditable } = require("./editable-markdown");
const { TYPES, renderMeetingTabs } = require("./meeting-note");

module.exports = {
  /* Returns the render promise when there is one, so a caller can act when the DOM is
   * actually there; a synchronous fallback returns undefined. */
  renderMarkdown(el, markdown) {
    if (!markdown) {
      el.createEl("p", { cls: "slim-empty", text: "Nothing was captured here." });
      return undefined;
    }
    if (MarkdownRenderer && typeof MarkdownRenderer.render === "function") {
      return MarkdownRenderer.render(this.app, markdown, el,
                                     this.filedNotePath || this.draftRel || "",
                                     this.plugin).catch((e) => {
        console.error("[slim] render markdown", e);
        el.setText(markdown);
      });
    }
    el.setText(markdown);
    return undefined;
  },

  renderLiveMarkdownEditor(parent, markdown, onChange, cls = "") {
    const editor = parent.createDiv({
      cls: `slim-live-markdown-editor ${cls}`.trim(),
      attr: {
        contenteditable: "true", spellcheck: "true", role: "textbox",
        "aria-multiline": "true", "data-placeholder": "Write Markdown…",
      },
    });
    if (markdown) {
      const rendered = this.renderMarkdown(editor, markdown);
      // After the async render lands, mark every node whose source the serializer cannot
      // reconstruct — otherwise the first keystroke deletes it.
      const stamp = () => stampSourceAtoms(editor, markdown);
      stamp();
      if (typeof window !== "undefined" && typeof window.setTimeout === "function") {
        window.setTimeout(stamp, 0);
      }
      // ⚠ The timer is a guess at when the render lands. Stamp when it actually resolves: an
      // embed rendered after the last pass has no source and is deleted by the first keystroke.
      if (rendered && typeof rendered.then === "function") rendered.then(stamp);
    }
    else {
      const p = editor.createEl("p");
      p.createEl("br");
    }
    const save = (event) => onChange(event && event.target && typeof event.target.value === "string"
      ? event.target.value : markdownFromEditable(editor));
    editor.addEventListener("input", save);
    editor.addEventListener("keyup", (event) => {
      if (event.key !== " ") return;
      const selection = typeof window !== "undefined" && window.getSelection
        ? window.getSelection() : null;
      let current = selection && selection.anchorNode;
      if (current && current.nodeType === 3) current = current.parentElement;
      while (current && current.parentElement !== editor) current = current.parentElement;
      if (!current) return;
      const shortcut = markdownBlockShortcut(current.textContent || "");
      if (!shortcut || typeof document === "undefined") return;
      let target;
      if (shortcut.kind === "list") {
        target = document.createElement(shortcut.tag);
        const item = document.createElement("li");
        item.appendChild(document.createElement("br"));
        target.appendChild(item);
        current.replaceWith(target);
        target = item;
      } else {
        target = document.createElement(shortcut.tag);
        target.appendChild(document.createElement("br"));
        current.replaceWith(target);
      }
      const range = document.createRange();
      range.selectNodeContents(target);
      range.collapse(true);
      selection.removeAllRanges();
      selection.addRange(range);
      save();
    });
    return editor;
  },

  /* The three tabs, drawn by the shared `renderMeetingTabs` reader. What the recorder adds
   * over a passively rendered block is the editing: a summary they can click into, and a notes
   * editor. `editableSummary` gates only the DERIVED half — their notes are editable in every
   * state, because the meeting block hides them from Obsidian's own editor, so read-only here
   * meant not editable anywhere and reaching them meant reopening the review.
   */
  renderSource(root, editableSummary) {
    const parts = {
      summary: this.result ? (this.result.summary || "") : (this.noteParts.summary || ""),
      notes: this.result && this.result.notes_md != null
        ? this.result.notes_md : this.noteParts.notes,
      transcript: this.noteParts.transcript,
    };
    return renderMeetingTabs(
      this.app, root, parts, this.filedNotePath || this.draftRel || "", {
        editable: true,
        component: this.plugin,
        activeTab: this.activeTab,
        onTab: (value) => {
          this.summaryEditing = false;
          this.activeTab = value;
          this.render();
        },
        summaryPane: (body) => this.renderSummaryPane(body, parts.summary, editableSummary),
        notesPane: (body) => this.renderNotesPane(body, parts.notes),
      });
  },

  renderSummaryPane(generated, summary, editableSummary) {
    if (editableSummary && this.summaryEditing) {
      const edit = generated.createDiv({ cls: "slim-summary-editing" });
      const toolbar = edit.createDiv({ cls: "slim-summary-editing-head" });
      toolbar.createSpan({ text: "Live Markdown" });
      const done = toolbar.createEl("button", { text: "Done" });
      const live = this.renderLiveMarkdownEditor(edit, summary, (value) => {
        this.result.summary = value;
        this.edited = true;
      }, "slim-summary slim-summary-editor");
      const finish = () => {
        if (!this.summaryEditing) return;
        this.summaryEditing = false;
        this.render();
      };
      done.addEventListener("click", finish);
      live.addEventListener("keydown", (e) => {
        if (e.key === "Escape") finish();
      });
      // Defer until the click that moved focus has reached its destination; an immediate
      // rerender on mousedown would swallow the user's intended filing/tab click.
      live.addEventListener("blur", () => {
        if (typeof window !== "undefined" && typeof window.setTimeout === "function") {
          window.setTimeout(finish, 0);
        } else finish();
      });
      if (typeof window !== "undefined" && typeof window.setTimeout === "function") {
        window.setTimeout(() => { if (typeof live.focus === "function") live.focus(); }, 0);
      }
      return;
    }
    const preview = generated.createDiv({
      cls: `slim-summary-preview${editableSummary ? " is-editable" : ""}`,
      attr: editableSummary ? { role: "button", tabindex: "0", "aria-label": "Edit AI summary" } : {},
    });
    this.renderMarkdown(preview.createDiv({ cls: "slim-rendered-markdown" }), summary);
    if (editableSummary) {
      preview.createSpan({ cls: "slim-summary-edit-cue", text: "Edit" });
      const begin = (e) => {
        if (e && e.target && typeof e.target.closest === "function" && e.target.closest("a")) return;
        this.summaryEditing = true;
        this.render();
      };
      preview.addEventListener("click", begin);
      preview.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); begin(e); }
      });
    }
  },

  // ⚠ SAVED ON BLUR, NOT ON EVERY KEYSTROKE, and that is a constraint rather than a
  // preference: their notes live INSIDE the meeting block, so writing them re-renders the very
  // object they are typing into. On blur the caret has already left, so nothing is lost.
  renderNotesPane(parent, notes) {
    // A BOX, not a value: `saveNotes` moves it on success. Captured by value, every later
    // click away re-posted the same notes — a full note rewrite and an index pass each time.
    const baseline = { text: notes || "" };
    const editor = this.renderLiveMarkdownEditor(parent, baseline.text, (value) => {
      this.result.notes_md = value;
      this.edited = true;
    }, "slim-notes-editor is-full");
    editor.setAttr("data-placeholder", "Add the notes you want to keep with this recording…");
    // In EVERY state that has a filed note behind it. Review was the gap: `Approve & finish`
    // was the only writer there, so typing in review and walking away lost the keystrokes —
    // and review is the state they are most likely to walk away from.
    if (this.filedNotePath) {
      this.notesBaseline = baseline;
      editor.addEventListener("blur", () => this.saveNotes(baseline));
    }
    return editor;
  },

  renderCard(root) {
    const c = this.result.card;
    this.renderObjectHeader(root, "AI review", this.result.title || "Review this meeting note",
                            "Ready for approval");
    this.renderSavedLocation(root);
    this.renderProperties(root);
    this.renderSource(root, true);
    this.renderRetryRow(root);
    const moreRow = root.createDiv({ cls: "slim-row slim-resume-row" });
    const moreBtn = moreRow.createEl("button", { text: "Resume recording" });
    moreBtn.addEventListener("click", () => this.startAppendRecording());
    const details = root.createEl("details", { cls: "slim-review-details" });
    details.createEl("summary", { text: "Review filing details" });
    root = details.createDiv({ cls: "slim-review-controls" });
    root.createEl("p", { cls: "slim-hint",
      text: "The note is already saved. Adjust any suggestion, then approve the final result." });

    // The title the server settled on — their if they typed one, otherwise written from the
    // transcript. Editable, because a generated title is a suggestion like everything else
    // here, and this is the one field that also names the FILE.
    if (this.cardTitle == null) this.cardTitle = this.result.title || "";
    const ti = root.createDiv({ cls: "slim-field" });
    ti.createEl("label", { text: "Title" });
    const tin = ti.createEl("input", { type: "text" });
    tin.value = this.cardTitle;
    tin.addEventListener("input", (e) => {
      this.cardTitle = e.target.value;
      this.edited = true;
    });

    // --- destination. EVERY segment is a control: confidence sets the default, never a lock.
    const dest = root.createDiv({ cls: "slim-field" });
    dest.createEl("label", { text: "Filing to" });
    const pathRow = dest.createDiv({ cls: "slim-path" });
    const renderPath = () => {
      pathRow.empty();
      const segs = c.dest_dir.split("/");
      segs.forEach((seg, i) => {
        if (i) pathRow.createSpan({ cls: "slim-sep", text: "/" });
        const cls = i >= (c.confident_depth || 0) ? "slim-seg slim-seg-open" : "slim-seg";
        pathRow.createSpan({ cls, text: seg });
      });
    };
    renderPath();

    const det = dest.createEl("details", { cls: "slim-tree-wrap" });
    det.createEl("summary", { text: "browse the vault" });
    const tree = det.createDiv({ cls: "slim-tree" });

    // The tree is the WHOLE vault now, so it has to collapse. A flat list was tolerable only
    // while a depth cap was hiding most of it — which was the bug, not the reason it was flat.
    // Everything starts collapsed except the ancestors of the suggested destination, so the
    // first thing they see is where SLIM wants to put it, in context.
    const roots = [];
    const byPath = new Map();
    (c.folders || []).slice().sort((a, b) => a.path.localeCompare(b.path)).forEach((f) => {
      const node = { path: f.path, count: f.count, name: f.path.split("/").pop(), children: [] };
      byPath.set(f.path, node);
      const parentPath = f.path.split("/").slice(0, -1).join("/");
      const parent = parentPath && byPath.get(parentPath);
      (parent ? parent.children : roots).push(node);
    });

    const expanded = new Set();
    (c.dest_dir || "").split("/").reduce((acc, seg) => {
      const p = acc ? `${acc}/${seg}` : seg;
      expanded.add(p);
      return p;
    }, "");

    const select = (path, isNew) => {
      c.dest_dir = path;
      this.edited = true;
      renderPath();
      drawTree();
      if (mi) mi.value = path;
      if (isNew) new Notice(`SLIM: will create ${path}`);
    };

    const drawNode = (node, depth, parentEl) => {
      const row = parentEl.createDiv({
        cls: "slim-tree-row" + (node.path === c.dest_dir ? " is-sel" : ""),
      });
      row.style.paddingLeft = `${6 + depth * 14}px`;
      const hasKids = node.children.length > 0;
      const tw = row.createSpan({ cls: "slim-tw" + (hasKids ? "" : " is-leaf") });
      tw.setText(hasKids ? (expanded.has(node.path) ? "▾" : "▸") : "·");
      if (hasKids) {
        tw.addEventListener("click", (e) => {
          e.stopPropagation();
          if (expanded.has(node.path)) expanded.delete(node.path);
          else expanded.add(node.path);
          drawTree();
        });
      }
      row.createSpan({ cls: "slim-tree-name", text: node.name });
      if (node.isNew) row.createSpan({ cls: "slim-tree-new", text: "new" });
      else row.createSpan({ cls: "slim-tree-count", text: String(node.count) });

      const plus = row.createSpan({ cls: "slim-tree-plus", text: "+" });
      plus.setAttr("aria-label", `New folder in ${node.path}`);
      plus.addEventListener("click", (e) => {
        e.stopPropagation();
        newFolderUnder(node);
      });

      row.addEventListener("click", () => select(node.path, !!node.isNew));

      if (hasKids && expanded.has(node.path)) {
        node.children.forEach((k) => drawNode(k, depth + 1, parentEl));
      }
    };

    const newFolderUnder = (node) => {
      const row = tree.createDiv({ cls: "slim-tree-row is-newrow" });
      row.createSpan({ cls: "slim-tw is-leaf", text: "+" });
      const input = row.createEl("input", { type: "text", cls: "slim-newfolder",
                                            placeholder: `new folder in ${node.path}` });
      input.focus();
      const commit = () => {
        const name = input.value.trim().replace(/^\/+|\/+$/g, "");
        if (!name) { drawTree(); return; }
        const path = `${node.path}/${name}`;
        // Not created on disk here: the apply endpoint makes parents when the note moves, so
        // abandoning the card leaves no empty folder behind.
        const fresh = { path, count: 0, name, children: [], isNew: true };
        node.children.push(fresh);
        byPath.set(path, fresh);
        expanded.add(node.path);
        select(path, true);
      };
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") drawTree();
      });
      input.addEventListener("blur", commit);
    };

    const drawTree = () => {
      tree.empty();
      roots.forEach((r) => drawNode(r, 0, tree));
    };
    drawTree();

    const manual = det.createDiv({ cls: "slim-field" });
    manual.createEl("label", { text: "or type a path" });
    const mi = manual.createEl("input", { type: "text" });
    mi.value = c.dest_dir;
    mi.addEventListener("change", (e) => {
      c.dest_dir = e.target.value.replace(/^\/+|\/+$/g, "");
      this.edited = true;
      renderPath();
      drawTree();
    });
    manual.createEl("p", { cls: "slim-hint",
      text: "A folder that does not exist yet is created when the note is filed." });

    // --- type
    const ty = root.createDiv({ cls: "slim-field" });
    ty.createEl("label", { text: "Type" });
    const tsel = ty.createEl("select");
    TYPES.forEach(([value, label]) => {
      const o = tsel.createEl("option", { text: label });
      o.value = value;
      if (value === c.type_tag) o.selected = true;
    });
    tsel.addEventListener("change", (e) => { c.type_tag = e.target.value; this.edited = true; });

    // --- topics. Creating is visibly heavier than reusing: an existing topic shows how often
    //     it has been used, a brand-new one says so. Informational, never blocking — and it
    //     applies to one THEY type too, since they are the likeliest source of a one-off.
    //
    // ⚠ THIS SAID "Tags" AND WROTE `topics:`, and `tags:` in the note is a DIFFERENT FIELD —
    // it carries the TYPE, for Obsidian's tag pane. So the card's "Tags" and the note's `tags:`
    // named two unrelated things and they had to hold both in their head (2026-08-23). The label
    // now matches the field it edits. The `#` went with it: `topics:` are not Obsidian tags,
    // and a `#` promises a clickable one.
    const tg = root.createDiv({ cls: "slim-field" });
    tg.createEl("label", { text: "Topics" });
    const chips = tg.createDiv({ cls: "slim-chips" });
    const counts = c.tag_counts || {};
    const drawChips = () => {
      chips.empty();
      (c.topics || []).forEach((t, i) => {
        const n = counts[String(t).toLowerCase()] || 0;
        const chip = chips.createSpan({ cls: "slim-chip" + (n ? "" : " is-new") });
        chip.createSpan({ text: String(t) });
        chip.createSpan({ cls: "slim-chip-count", text: n ? String(n) : "new" });
        const x = chip.createSpan({ cls: "slim-chip-x", text: "×" });
        x.addEventListener("click", () => {
          c.topics.splice(i, 1); this.edited = true; drawChips();
        });
      });
      const add = chips.createEl("input", { type: "text", cls: "slim-chip-add",
                                            placeholder: "+ add tag" });
      add.addEventListener("keydown", (e) => {
        if (e.key !== "Enter") return;
        const v = add.value.trim().replace(/^#/, "");
        if (!v) return;
        c.topics = c.topics || [];
        c.topics.push(v);
        this.edited = true;
        drawChips();
      });
    };
    drawChips();

    const row = root.createDiv({ cls: "slim-row" });
    const file = row.createEl("button", { cls: "mod-cta", text: "Approve & finish" });
    file.addEventListener("click", async () => {
      file.disabled = true;
      // ⚠ `filing` covers the whole request. A move drops the old file from its leaf, and
      // that layout change ran the pending-review notice while Approve was still out — state
      // `card`, pane empty, the path from before the move (2026-09-09).
      this.filing = true;
      try {
        if (this.edited) await this.applyCard();
        else await this.plugin.completeReview({ note: this.filedNotePath });
        await this.loadNoteParts();
        new Notice("SLIM: meeting note finished.");
        this.state = "finished";
        this.activeTab = "summary";
        this.edited = false;
        this.freezeEditor(false);
        this.render();
      } catch (e) {
        console.error("[slim] file", e);
        new Notice(`SLIM: ${e.message || e}`);
        file.disabled = false;
      } finally {
        this.filing = false;
      }
    });
  },
};
