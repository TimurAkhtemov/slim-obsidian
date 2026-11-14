// CopilotView: drawing the sidebar, its threads, its transcript and the streaming answer.
// The layout is a modern copilot's: a slim header (the chat's title, history, new chat), the
// conversation, and one rounded composer pinned to the bottom with its context chips, the
// textarea and a toolbar (@, image, effort, send/stop).
"use strict";

const { MarkdownRenderer, MarkdownRenderChild } = require("obsidian");
const { threadRows, relativeDate, folderOf, pathStem, groupCitations, slashQuery, matchSkills, inputsText,
        normalizeMath, LIVE_RENDER_MS, EFFORTS, EFFORT_HINTS, effortLabel, metaText, stageLine, toolIcon,
        toolSummary, starters } = require("./text");
const { clear, create, icon, iconButton, autoGrow, atBottom } = require("./dom");

// The composer grows to about ten lines, then scrolls (the CSS max-height says the same).
const INPUT_MAX_PX = 240;

module.exports = {
  renderSlash() {
    const menu = this.el?.slash;
    if (!menu) return;
    clear(menu);
    this.el.slashSelected = null;
    const hits = this.slashDismissed ? [] : matchSkills(this.skills, slashQuery(this.draft));
    menu.hidden = !hits.length;
    if (!hits.length) { this.slashIndex = 0; return; }
    const idx = Math.min(Math.max(0, this.slashIndex || 0), hits.length - 1);
    this.slashIndex = idx;
    hits.forEach((skill, i) => {
      const cls = `slim-slash-item${skill.error ? " is-broken" : ""}${i === idx ? " is-selected" : ""}`;
      const item = create(menu, "button", cls);
      if (i === idx) this.el.slashSelected = item;
      create(item, "span", "slim-slash-name", `/${skill.name}`);
      create(item, "small", "slim-slash-about", skill.error ? `Broken: ${skill.error}` : skill.description || "");
      item.onmousedown = (event) => event.preventDefault();     // keep the composer focused
      item.onclick = () => { if (!skill.error) this.pickSkill(skill); };
    });
  },

  // The `@` picker: one line per note or folder, its title (a folder's name, with the folder icon)
  // on the left and its parent folder faint on the right, the chosen row highlighted.
  renderMention() {
    const menu = this.el?.mention;
    if (!menu) return;
    clear(menu);
    this.el.mentionSelected = null;
    const hits = this.mentionDismissed ? [] : this.mentionHits();
    menu.hidden = !hits.length;
    if (!hits.length) { this.mentionIndex = 0; return; }
    const idx = Math.min(Math.max(0, this.mentionIndex || 0), hits.length - 1);
    this.mentionIndex = idx;
    hits.forEach((hit, i) => {
      const item = create(menu, "button", `slim-mention-item${i === idx ? " is-selected" : ""}`);
      if (i === idx) this.el.mentionSelected = item;
      item.setAttribute("title", hit.path);
      const parts = hit.path.split("/");
      const name = create(item, "span", "slim-mention-label");
      if (hit.kind === "folder") icon(create(name, "span", "slim-mention-icon"), "folder");
      create(name, "span", "slim-mention-name", hit.kind === "folder" ? parts.at(-1) : pathStem(hit.path));
      if (parts.length > 1) create(item, "span", "slim-mention-path", parts.slice(0, -1).join("/"));
      item.onmousedown = (event) => event.preventDefault();
      item.onclick = () => this.pin(hit);
    });
  },

  imageURL(threadId, imageId) {
    const port = this.plugin.settings?.port || 7546;
    return `http://127.0.0.1:${port}/api/copilot/image?thread=${encodeURIComponent(threadId)}`
      + `&id=${encodeURIComponent(imageId)}`;
  },

  // The transcript sticks to its bottom while anything grows it (tokens, lookups, steps, Markdown
  // landing late), unless the owner scrolled up: only their scrolling unsticks it (`onscroll`),
  // and sending or opening a chat sticks it again. Scrolled up, the ↓ button offers the way back.
  // Deciding by "was it at the bottom before this growth" failed: one growth that skipped the
  // check (lookup rows) left it 166px up, and from then on nothing followed (2026-11-14).
  follow(transcript = this.el?.transcript) {
    if (!transcript) return;
    if (this.stuck) {
      transcript.scrollTop = transcript.scrollHeight;
      this.el.lastTop = transcript.scrollTop;
    }
    if (this.el?.jump) this.el.jump.hidden = this.stuck;
  },

  // Per-token updates touch two live nodes. A full render() per delta tore the whole view
  // down ~40 times a second: the transcript's scroll position reset to the top on every
  // token, the composer lost focus per keystroke, and a hand-expanded group snapped shut.
  renderDelta() {
    if (!this.el) { this.render(); return; }
    if (!this.el.liveText) return;          // the chat list is showing; the next render catches up
    this.el.liveText.textContent = this.streamingText;
    this.follow();
    this.scheduleLiveRender();
  },

  // Markdown while it streams: plain text shows until the first rendered swap, then a detached
  // render replaces the visible node at most every LIVE_RENDER_MS. The version guard drops a
  // slow render that resolves after a newer one; the turn event's full render is the final word.
  scheduleLiveRender() {
    if (this.liveRenderTimer || !this.el?.liveText) return;
    const wait = Math.max(0, LIVE_RENDER_MS - (Date.now() - this.liveRenderedAt));
    this.liveRenderTimer = setTimeout(() => { this.liveRenderTimer = null; this.renderLive(); }, wait);
  },

  stopLiveRender() {
    if (this.liveRenderTimer) clearTimeout(this.liveRenderTimer);
    this.liveRenderTimer = null;
    this.liveRenderVersion += 1;            // an in-flight render must not land on the next view
    for (const host of this.liveHosts || []) this.removeChild?.(host);
    this.liveHosts = new Set();
  },

  // Each live pass renders under its own child component, dropped with its buffer: a long
  // answer is ~170 passes, and post-processor children registered on the shared mdHost would
  // otherwise pile up until the turn's full render.
  liveHost(buffer) {
    if (!MarkdownRenderChild || typeof this.addChild !== "function") return undefined;
    const host = this.addChild(new MarkdownRenderChild(buffer));
    (this.liveHosts ||= new Set()).add(host);
    return host;
  },

  dropLive(buffer, host) {
    buffer?.remove?.();
    if (host) { this.removeChild?.(host); this.liveHosts?.delete(host); }
  },

  renderLive() {
    const el = this.el;
    if (!el?.live || !this.sending) return;
    const version = ++this.liveRenderVersion;
    this.liveRenderedAt = Date.now();
    const buffer = create(el.live, "div", "slim-message-text slim-rendered-markdown");
    buffer.hidden = true;
    const host = this.liveHost(buffer);
    return this.renderMarkdown(buffer, this.streamingText, host).then(() => {
      if (version !== this.liveRenderVersion || this.el !== el || !this.sending) { this.dropLive(buffer, host); return; }
      this.dropLive(el.liveRendered, el.liveHost);
      el.liveRendered = buffer;
      el.liveHost = host;
      buffer.hidden = false;
      el.liveText.hidden = true;
      this.follow(el.transcript);
    }).catch((e) => console.error("[slim] live render", e));
  },

  // The turn's lookups, a compact row each (an icon for what it did, its label, one line): drawn
  // live as each one ran (`liveTools`), and folded under one summary row once the turn is saved.
  fillTools(list, tools) {
    clear(list);
    list.hidden = !tools?.length;
    for (const tool of tools || []) {
      const label = String(tool.label || tool.name || "");
      const item = create(list, "li", `slim-step slim-tool${tool.refused ? " is-refused" : ""}`);
      item.setAttribute("title", label);
      icon(create(item, "span", "slim-step-icon"), toolIcon(tool));
      create(item, "span", "slim-step-text", label);
    }
    return list;
  },

  // A row that folds a list away under it, open or shut per turn across renders. The chevron
  // turns with aria-expanded.
  foldable(toggle, list, opened, key) {
    const show = (open) => {
      list.hidden = !open;
      toggle.setAttribute("aria-expanded", String(open));
    };
    show(opened.has(key));
    toggle.onclick = () => {
      if (opened.has(key)) opened.delete(key);
      else opened.add(key);
      show(opened.has(key));
    };
  },

  // A finished turn's lookups: one summary row ("Searched 2 · Read 1"), the rows under it.
  renderToolGroup(message, turn, tools) {
    const group = create(message, "div", "slim-tool-group");
    const toggle = create(group, "button", "slim-tool-summary");
    // One kind of lookup shows its icon; a mix shows a list.
    const kinds = new Set(tools.map((tool) => toolIcon(tool)));
    icon(create(toggle, "span", "slim-step-icon"), kinds.size === 1 ? [...kinds][0] : "list");
    create(toggle, "span", "slim-tool-summary-text", toolSummary(tools));
    icon(create(toggle, "span", "slim-chevron"), "chevron-right");
    const list = this.fillTools(create(group, "ul", "slim-steps slim-tools"), tools);
    this.foldable(toggle, list, this.toolsOpen, turn);
    return group;
  },

  // The notes an answer drew on: "N sources" with stacked file icons; open, a title per note,
  // its folder faint on the right, each opening the note.
  renderSources(message, turn, sources) {
    const box = create(message, "div", "slim-sources");
    const toggle = create(box, "button", "slim-sources-toggle");
    const stack = create(toggle, "span", "slim-sources-stack");
    for (let i = 0; i < Math.min(3, sources.length); i++) icon(create(stack, "span", "slim-sources-file"), "file-text");
    create(toggle, "span", "slim-sources-count", `${sources.length} source${sources.length === 1 ? "" : "s"}`);
    icon(create(toggle, "span", "slim-chevron"), "chevron-right");
    const list = create(box, "div", "slim-sources-list");
    for (const source of sources) {
      const row = create(list, "button", "slim-source");
      row.setAttribute("title", source.path);
      create(row, "span", "slim-source-title", source.title);
      const folder = folderOf(source.path);
      if (folder) create(row, "span", "slim-source-path", folder);
      row.onclick = () => this.openFile(source.path);
    }
    this.foldable(toggle, list, this.sourcesOpen, turn);
  },

  renderTools() {
    if (this.el?.liveEdit) this.renderEdit();
    else if (this.el?.liveTools) { this.fillTools(this.el.liveTools, this.liveTools); this.follow(); }
    else this.renderSoon();
  },

  renderStage() {
    if (!this.el) { this.render(); return; }
    if (!this.el.stage) return;
    this.el.stage.textContent = stageLine(this.stage, this.stageSince, Date.now());
  },

  renderMarkdown(el, markdown, host = this.mdHost || this) {
    const text = normalizeMath(markdown);
    if (MarkdownRenderer && typeof MarkdownRenderer.render === "function") {
      return MarkdownRenderer.render(this.app, text, el, this.openNote?.path || "",
                                     host).catch((e) => {
        console.error("[slim] render answer", e);
        el.setText?.(text);
      });
    }
    el.textContent = text;
    return Promise.resolve();
  },

  render() {
    const root = this.contentEl;
    if (!root || !root.createEl) return;
    const focused = this.composerFocused;
    // Scrolled up, a redraw (an effort picked, a pin) keeps their place; at the bottom it stays there.
    const kept = !this.stuck && this.el?.transcript ? this.el.transcript.scrollTop : null;
    clear(root);
    this.el = {};
    // Rendered Markdown attaches child components (embeds, callouts) to the component it is
    // given. One host per render, replaced by the next, so they do not pile up on the view.
    if (this.mdHost) { this.removeChild(this.mdHost); this.mdHost = null; }
    if (MarkdownRenderChild && typeof this.addChild === "function") {
      this.mdHost = this.addChild(new MarkdownRenderChild(root));
    }
    root.addClass?.("slim-copilot");
    this.renderHeader(root);
    if (this.showThreads) this.renderThreadList(root);
    else this.renderTranscript(root, kept);
    this.renderComposer(root, focused);
  },

  // One slim row: the chat's title, then history and new chat.
  renderHeader(root) {
    const header = create(root, "header", "slim-copilot-head");
    const name = this.showThreads ? "Chats" : this.activeThread?.title || "New chat";
    const title = create(header, "div", "slim-copilot-title", name);
    title.setAttribute("title", name);
    const tools = create(header, "div", "slim-copilot-tools");
    const history = iconButton(tools, "history", this.showThreads ? "Back to the chat" : "Chats",
      this.showThreads ? "clickable-icon slim-history is-active" : "clickable-icon slim-history");
    history.onclick = () => {
      this.showThreads = !this.showThreads;
      this.render();
      if (this.showThreads) this.loadThreads();
    };
    const add = iconButton(tools, "square-pen", "New chat", "clickable-icon slim-new-chat");
    add.onclick = () => this.newThread();
  },

  // The notes near the open one, a popover over the conversation opened from the composer's
  // "N nearby" chip: context to reach for, not the reason the pane is open.
  renderNearby(composer) {
    const path = this.openNote?.path;
    const related = path && this.context?.source?.path === path ? this.context.related || [] : [];
    if (!related.length) return;
    const list = create(composer, "div", "slim-copilot-nearby");
    list.hidden = !this.nearbyOpen;
    this.el.nearby = list;
    create(list, "div", "slim-popover-label", "Nearby notes");
    for (const item of related) {
      const button = create(list, "button", "slim-related-note");
      button.setAttribute("title", item.path);
      create(button, "span", "", item.title || pathStem(item.path));
      create(button, "small", "", item.scope || "");
      button.onclick = () => this.openFile(item.path);
    }
  },

  // How hard the model thinks, as Pi's levels; the last one chosen is kept in the settings.
  // A menu as the <select> it replaced was: opening it puts focus on the level in use; the arrows
  // move, Enter or Space picks, Escape closes back to the button, Tab closes and moves on from it.
  renderEffortMenu(composer) {
    const menu = create(composer, "div", "slim-effort-menu");
    menu.setAttribute("role", "menu");
    menu.hidden = !this.effortOpen;
    this.el.effortMenu = menu;
    this.el.effortItems = EFFORTS.map((value, i) => {
      const item = create(menu, "button", `slim-effort-item${value === this.effort ? " is-selected" : ""}`);
      item.setAttribute("role", "menuitemradio");
      item.setAttribute("aria-checked", String(value === this.effort));
      item.setAttribute("tabindex", "-1");              // out of the Tab order: the arrows move within
      create(item, "span", "slim-effort-name", effortLabel(value));
      create(item, "small", "slim-effort-hint", EFFORT_HINTS[value]);
      item.onclick = () => this.pickEffort(value);
      item.onkeydown = (event) => this.effortKey(event, i);
      return item;
    });
  },

  effortKey(event, index) {
    const items = this.el?.effortItems || [];
    const move = (to) => {
      event.preventDefault();
      items[(to + items.length) % items.length]?.focus?.();
    };
    switch (event.key) {
      case "ArrowDown": return move(index + 1);
      case "ArrowUp": return move(index - 1);
      case "Home": return move(0);
      case "End": return move(items.length - 1);
      case "Enter":
      case " ":
        event.preventDefault();
        return this.pickEffort(EFFORTS[index]);
      case "Escape":
        event.preventDefault();
        event.stopPropagation?.();
        this.showEffortMenu(false);
        return this.el.effortButton?.focus?.();
      case "Tab":
        // Not prevented: with focus back on the button, the Tab moves on from there.
        this.showEffortMenu(false);
        return this.el.effortButton?.focus?.();
      default:
        return undefined;
    }
  },

  pickEffort(value) {
    this.effortOpen = false;
    this.setEffort(value);
    this.render();
    this.el?.effortButton?.focus?.();
  },

  showEffortMenu(open) {
    this.effortOpen = open;
    if (this.el?.effortMenu) this.el.effortMenu.hidden = !open;
    this.el?.effortButton?.setAttribute("aria-expanded", String(open));
    if (open && this.nearbyOpen) this.showNearby(false);
    if (open) this.el?.effortItems?.[Math.max(0, EFFORTS.indexOf(this.effort))]?.focus?.();
  },

  showNearby(open) {
    this.nearbyOpen = open;
    if (this.el?.nearby) this.el.nearby.hidden = !open;
    this.el?.nearbyChip?.setAttribute("aria-expanded", String(open));
    if (open && this.effortOpen) this.showEffortMenu(false);
  },

  // The context the next question carries, as chips inside the composer: the open note (✕ leaves
  // it out of this chat), "N nearby", and the `@` pins. Small, one line each, wrapping.
  renderChips(composer) {
    const sourcePath = this.openNote?.path || "";
    const pins = this.activeThread?.pins || [];
    if (!sourcePath && !pins.length) return;
    const chips = create(composer, "div", "slim-composer-chips");
    if (sourcePath) {
      const nearby = this.context?.source?.path === sourcePath ? this.context : null;
      const off = Boolean(this.activeThread?.openNoteOff);
      const chip = create(chips, "span", `slim-ctx-chip slim-copilot-note-chip${off ? " is-off" : ""}`);
      icon(create(chip, "span", "slim-ctx-icon"), "file-text");
      const note = create(chip, "span", `slim-copilot-note${off ? " is-off" : ""}`, nearby?.source?.title || pathStem(sourcePath));
      note.setAttribute("title", sourcePath);
      const label = off ? "Include the open note again" : "Leave the open note out of this chat";
      const x = create(chip, "button", `slim-chip-x slim-copilot-note-off${off ? " is-off" : ""}`, off ? "↩" : "✕");
      x.setAttribute("title", label);
      x.setAttribute("aria-label", label);
      x.onclick = (event) => { event.preventDefault(); event.stopPropagation(); this.toggleOpenNote(); };
      const related = nearby?.related || [];
      if (related.length) {
        const more = create(chips, "button", "slim-ctx-chip slim-copilot-nearby-count", `${related.length} nearby`);
        more.setAttribute("aria-expanded", String(Boolean(this.nearbyOpen)));
        more.setAttribute("title", "Notes near the open one");
        this.el.nearbyChip = more;
        more.onclick = () => this.showNearby(!this.nearbyOpen);
      }
    }
    for (const pin of pins) {
      const chip = create(chips, "span", `slim-pin is-${pin.kind}`);
      chip.setAttribute("title", pin.path);
      icon(create(chip, "span", "slim-ctx-icon"), pin.kind === "folder" ? "folder" : "file-text");
      create(chip, "span", "slim-pin-name", pin.kind === "folder" ? `${pathStem(pin.path)}/` : pathStem(pin.path));
      const x = create(chip, "button", "slim-chip-x", "✕");
      x.setAttribute("aria-label", `Unpin ${pin.path}`);
      x.onclick = () => this.unpin(pin.path);
    }
  },

  renderComposer(root, focused) {
    const composer = create(root, "footer", "slim-copilot-composer");
    // The menus and popovers open above the composer, over the conversation.
    this.el.slash = create(composer, "div", "slim-slash-menu");
    this.el.mention = create(composer, "div", "slim-mention-menu");
    this.renderNearby(composer);
    this.renderEffortMenu(composer);
    this.renderChips(composer);
    if (this.pendingImages.length) {
      const previews = create(composer, "div", "slim-copilot-image-previews");
      this.pendingImages.forEach((attachment, index) => {
        const item = create(previews, "div", "slim-copilot-image-preview");
        const image = create(item, "img", "");
        image.src = attachment.preview
          || (attachment.id && this.activeThread ? this.imageURL(this.activeThread.id, attachment.id) : "");
        image.alt = attachment.name;
        const remove = iconButton(item, "x", `Remove ${attachment.name}`);
        remove.onclick = () => { this.pendingImages.splice(index, 1); this.render(); };
      });
    }
    const input = create(composer, "textarea", "slim-copilot-input");
    input.placeholder = "Ask anything · @ to add context";
    input.value = this.draft;
    input.rows = 1;
    autoGrow(input, INPUT_MAX_PX);
    const syncSend = () => this.el.send?.classList?.toggle("is-empty",
      !this.sending && !String(input.value || "").trim() && !this.pendingImages.length);
    input.oninput = () => {
      this.draft = input.value;
      this.caret = input.selectionStart;
      this.slashDismissed = false;
      this.slashIndex = 0;
      this.mentionDismissed = false;
      this.mentionIndex = 0;
      autoGrow(input, INPUT_MAX_PX);
      syncSend();
      if (input.value === "/") this.loadSkills();
      else this.renderSlash();
      this.renderMention();
    };
    input.onclick = input.onkeyup = () => {
      if (input.selectionStart !== this.caret) {
        this.caret = input.selectionStart;
        this.renderMention();
      }
    };
    input.onfocus = () => { this.composerFocused = true; };
    input.onblur = () => { this.composerFocused = false; };
    this.el.input = input;
    input.onkeydown = (event) => {
      // A half-typed command completes before it sends: "/con" + Enter is "/consolidate… ".
      const picks = this.mentionDismissed ? [] : this.mentionHits();
      if (picks.length) {
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          this.mentionIndex = ((this.mentionIndex || 0) + (event.key === "ArrowDown" ? 1 : -1) + picks.length) % picks.length;
          this.renderMention();
          this.el.mentionSelected?.scrollIntoView?.({ block: "nearest" });
          return;
        }
        if (event.key === "Escape") {
          event.preventDefault();
          this.mentionDismissed = true;
          this.renderMention();
          return;
        }
        if (event.key === "Tab" || (event.key === "Enter" && !event.shiftKey)) {
          event.preventDefault();
          this.pin(picks[Math.min(this.mentionIndex || 0, picks.length - 1)]);
          return;
        }
      }
      const hits = this.slashDismissed ? [] : matchSkills(this.skills, slashQuery(this.draft));
      if (hits.length && (event.key === "ArrowDown" || event.key === "ArrowUp")) {
        event.preventDefault();
        const delta = event.key === "ArrowDown" ? 1 : -1;
        this.slashIndex = ((this.slashIndex || 0) + delta + hits.length) % hits.length;
        this.renderSlash();
        this.el.slashSelected?.scrollIntoView?.({ block: "nearest" });
        return;
      }
      if (hits.length && event.key === "Escape") {
        event.preventDefault();
        this.slashDismissed = true;
        this.renderSlash();
        return;
      }
      const chosen = hits[Math.min(this.slashIndex || 0, hits.length - 1)];
      const valid = chosen && !chosen.error ? chosen : hits.find((h) => !h.error);
      if (valid && (event.key === "Tab"
          || (event.key === "Enter" && !event.shiftKey && `/${valid.name}` !== this.draft
              && (this.draft.length > 1 || this.slashIndex > 0)))) {
        event.preventDefault();
        this.pickSkill(valid);
        return;
      }
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); this.send(); }
      if ((event.key === "Backspace" || event.key === "Delete") &&
          !event.metaKey && !event.altKey && !event.ctrlKey) {
        // Obsidian's sidebar keymap can consume deletion before Electron applies the
        // textarea's native edit. Do the ordinary character/selection deletion here; keep
        // modified word/line deletion native so macOS retains its platform conventions.
        let start = Number.isInteger(input.selectionStart) ? input.selectionStart : input.value.length;
        let end = Number.isInteger(input.selectionEnd) ? input.selectionEnd : start;
        if (start === end && event.key === "Backspace" && start > 0) start -= 1;
        if (start === end && event.key === "Delete" && end < input.value.length) end += 1;
        if (start !== end) {
          event.preventDefault();
          input.value = `${input.value.slice(0, start)}${input.value.slice(end)}`;
          input.selectionStart = input.selectionEnd = start;
          this.draft = input.value;
          this.slashDismissed = false;
          autoGrow(input, INPUT_MAX_PX);
          syncSend();
          this.renderSlash();
        }
      }
    };
    input.onpaste = (event) => {
      const files = [...(event.clipboardData?.files || [])];
      if (files.some((file) => String(file.type || "").startsWith("image/"))) {
        event.preventDefault();
        this.addImages(files);
        return;
      }
      // Electron does not consistently perform the textarea's native insertion after a
      // plugin paste listener has inspected clipboardData. Insert plain text ourselves while
      // preserving the browser selection: paste must replace highlighted text, not append to
      // the draft or trigger a full render that moves the caret.
      const text = event.clipboardData?.getData?.("text/plain");
      if (typeof text !== "string") return;
      event.preventDefault();
      const start = Number.isInteger(input.selectionStart) ? input.selectionStart : input.value.length;
      const end = Number.isInteger(input.selectionEnd) ? input.selectionEnd : start;
      input.value = `${input.value.slice(0, start)}${text}${input.value.slice(end)}`;
      input.selectionStart = input.selectionEnd = start + text.length;
      this.draft = input.value;
      autoGrow(input, INPUT_MAX_PX);
      syncSend();
      this.renderSlash();
    };

    // The toolbar: @ and image on the left; effort and send (stop while a turn runs) on the right.
    const bar = create(composer, "div", "slim-composer-bar");
    const left = create(bar, "div", "slim-composer-left");
    const at = iconButton(left, "at-sign", "Add a note or folder", "clickable-icon slim-at");
    at.onmousedown = (event) => event?.preventDefault?.();     // the textarea keeps its caret
    at.onclick = () => this.openMention();
    const picker = create(left, "input", "slim-copilot-image-picker");
    picker.type = "file";
    picker.accept = "image/png,image/jpeg,image/webp";
    picker.multiple = true;
    picker.onchange = () => { this.addImages(picker.files); picker.value = ""; };
    const attach = iconButton(left, "image-plus", "Attach images", "clickable-icon slim-attach");
    attach.onclick = () => picker.click();
    const right = create(bar, "div", "slim-composer-right");
    const effort = create(right, "button", "slim-effort");
    effort.setAttribute("aria-label", `Effort: ${effortLabel(this.effort)}`);
    effort.setAttribute("title", "Effort: how much the model thinks before it answers");
    effort.setAttribute("aria-haspopup", "menu");
    effort.setAttribute("aria-expanded", String(Boolean(this.effortOpen)));
    icon(create(effort, "span", "slim-effort-icon"), "brain");
    create(effort, "span", "slim-effort-label", effortLabel(this.effort));
    effort.onclick = () => this.showEffortMenu(!this.effortOpen);
    effort.onkeydown = (event) => {
      if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
      event.preventDefault();
      this.showEffortMenu(true);
    };
    this.el.effortButton = effort;
    const empty = !this.sending && !String(this.draft || "").trim() && !this.pendingImages.length;
    const send = iconButton(right, this.sending ? "square" : "arrow-up", this.sending ? "Stop response" : "Send",
      `slim-send${this.sending ? " is-stop" : ""}${empty ? " is-empty" : ""}`);
    send.onclick = () => this.sending ? this.cancel() : this.send();
    this.el.send = send;
    composer.ondragover = (event) => {
      if ([...(event.dataTransfer?.items || [])].some((item) => item.kind === "file")) event.preventDefault();
    };
    composer.ondrop = (event) => {
      const files = [...(event.dataTransfer?.files || [])];
      if (files.some((file) => String(file.type || "").startsWith("image/"))) {
        event.preventDefault();
        this.addImages(files);
      }
    };
    this.renderSlash();
    this.renderMention();
    // The caret goes where the last action left it (an @ inserted mid-text, a starter's end),
    // else to the end.
    const caretAt = this.caretAt;
    this.caretAt = null;
    if (focused && typeof input.focus === "function") {
      input.focus();
      const at = Number.isInteger(caretAt) && caretAt <= input.value.length ? caretAt : input.value.length;
      input.selectionStart = input.selectionEnd = at;
    }
  },

  // The search box and the rows under it. Only the rows are redrawn when results land, so
  // typing never loses the box.
  renderThreadList(root) {
    const list = create(root, "nav", "slim-copilot-threads");
    const search = create(list, "input", "slim-thread-search");
    search.type = "search";
    search.placeholder = "Search chats";
    search.value = this.threadQuery;
    search.oninput = () => this.searchThreads(search.value);
    this.el.threadRows = create(list, "div", "slim-thread-rows");
    this.drawThreadRows(this.el.threadRows);
  },

  renderThreadRows() {
    if (!this.showThreads || !this.el?.threadRows) return;
    clear(this.el.threadRows);
    this.drawThreadRows(this.el.threadRows);
  },

  // A quiet list: the title, and under it a faint date and the note it was started on; rename
  // and delete show on hover.
  drawThreadRows(list) {
    if (!this.threads.length) {
      create(list, "p", "slim-copilot-status", this.threadQuery.trim() ? "No chats match." : "No chats yet.");
      return;
    }
    for (const thread of threadRows(this.threads)) {
      const row = create(list, "div", `slim-chat-row${thread.id === this.activeThread?.id ? " is-active" : ""}`);
      const open = create(row, "button", "slim-chat-open");
      open.setAttribute("title", thread.title || "Conversation");
      open.onclick = () => this.openThread(thread.id);
      create(open, "span", "slim-chat-title", thread.title || "Conversation");
      const sub = create(open, "span", "slim-chat-sub");
      const date = relativeDate(thread.updated_at);
      if (date) create(sub, "small", "slim-chat-date", date);
      if (thread.source_path) {
        const label = create(sub, "small", "slim-chat-note", pathStem(thread.source_path));
        label.setAttribute("title", thread.source_path);
      }
      const rename = iconButton(row, "pencil", `Rename ${thread.title}`);
      rename.onclick = () => this.renameThread(thread.id);
      const armed = this.deleteArmed === thread.id;
      const remove = iconButton(
        row, armed ? "check" : "trash-2",
        armed ? `Confirm delete ${thread.title}` : `Delete ${thread.title}`,
        armed ? "clickable-icon is-armed" : "clickable-icon");
      remove.onclick = () => this.deleteThread(thread.id);
    }
  },

  // A new chat: SLIM's name, one line, and three starters that fill the composer.
  renderEmpty(transcript) {
    const empty = create(transcript, "div", "slim-chat-empty");
    icon(create(empty, "div", "slim-chat-empty-icon"), "message-square-text");
    create(empty, "div", "slim-chat-empty-name", "SLIM");
    create(empty, "p", "slim-chat-empty-line", "Ask about your notes, or have SLIM edit the open one.");
    const list = create(empty, "div", "slim-suggestions");
    const noteOpen = Boolean(this.openNote?.path) && !this.activeThread?.openNoteOff;
    for (const starter of starters(noteOpen)) {
      const button = create(list, "button", "slim-suggest", starter.label);
      button.onclick = () => this.useStarter(starter.text);
    }
  },

  renderTranscript(root, kept = null) {
    // The wrap holds the ↓ button over the scrolling transcript.
    const wrap = create(root, "div", "slim-transcript-wrap");
    const transcript = create(wrap, "section", "slim-copilot-transcript");
    this.el.transcript = transcript;
    const pending = [];
    let lastNote = null;
    for (const turn of this.turns) {
      // A turn drawn for the first time fades in; a redraw of one already shown does not.
      const fresh = !this.seenTurns.has(turn);
      this.seenTurns.add(turn);
      if (turn.role === "you") {
        // The note a question was asked on, faint above it, when it is not the one before's.
        if (turn.note && turn.note !== lastNote) {
          const on = create(transcript, "div", "slim-message-context");
          on.setAttribute("title", turn.note);
          if (fresh) on.setAttribute("data-new", "");
          icon(create(on, "span", "slim-ctx-icon"), "file-text");
          create(on, "span", "slim-message-context-name", pathStem(turn.note));
        }
        lastNote = turn.note || null;
        const message = create(transcript, "article", "slim-message is-you");
        if (fresh) message.setAttribute("data-new", "");
        if (turn.attachments?.length) this.renderAttachments(message, turn.attachments);
        create(message, "div", "slim-message-text", turn.text);
        continue;
      }
      const message = create(transcript, "article", `slim-message is-${turn.role}`);
      if (fresh) message.setAttribute("data-new", "");
      const run = turn.turn?.edit_run;
      if (turn.turn?.tools?.length) this.renderToolGroup(message, turn, turn.turn.tools);
      if (run) {
        this.renderEditRun(message, run, pending);
      } else {
        const body = create(message, "div", "slim-message-text slim-rendered-markdown");
        pending.push(this.renderMarkdown(body, turn.text)
          .catch((e) => console.error("[slim] render answer", e)));
      }
      if (turn.attachments?.length) this.renderAttachments(message, turn.attachments);
      // Sources fold into one row, only when the answer drew on notes.
      const sources = groupCitations(turn.turn?.citations);
      if (sources.length) this.renderSources(message, turn, sources);
      const read = inputsText(turn.turn?.inputs);
      if (read) create(message, "small", "slim-inputs", read);
      if (turn.turn?.verification?.truncated && !run) {
        create(message, "small", "slim-verification", "Cut off at the length limit — ask for the rest.");
      }
      if (run) this.renderEditMeta(message, run, turn);
      else if (turn.turn) this.renderActions(message, turn);
    }
    const streamingHere = this.sending && this.turns === this.streamingTurns;
    if (streamingHere) {
      const live = create(transcript, "article", "slim-message is-slim is-streaming");
      if (this.liveFresh) { live.setAttribute("data-new", ""); this.liveFresh = false; }
      this.el.live = live;
      if (this.editRun) {
        this.el.liveEdit = live;
        this.drawLiveEdit(live);
      } else {
        this.el.liveTools = this.fillTools(create(live, "ul", "slim-steps slim-tools"), this.liveTools);
        this.el.liveText = create(live, "div", "slim-message-text", this.streamingText);
      }
      // One shimmering status line: what the model is doing, and for how long.
      this.el.stage = create(transcript, "div", "slim-copilot-stage", stageLine(this.stage, this.stageSince, Date.now()));
    }
    if (this.error) {
      const error = create(transcript, "div", "slim-copilot-error", this.error);
      if (this.retryQuestion) {
        const retry = iconButton(error, "rotate-ccw", "Retry response");
        retry.onclick = () => this.retry();
      }
    }
    if (!this.turns.length && !streamingHere && !this.error) this.renderEmpty(transcript);
    const jump = iconButton(wrap, "arrow-down", "Jump to the latest", "clickable-icon slim-jump");
    jump.onclick = () => { this.stuck = true; this.follow(transcript); };
    this.el.jump = jump;
    // Only scrolling UP unsticks: growth never moves scrollTop, so the event a programmatic
    // scroll queues cannot unstick it however much lands before it fires.
    transcript.onscroll = () => {
      const top = transcript.scrollTop;
      if (atBottom(transcript)) this.stuck = true;
      else if (top < (this.el.lastTop ?? 0)) this.stuck = false;
      this.el.lastTop = top;
      jump.hidden = this.stuck;
    };
    // A full render is a new transcript element at scrollTop 0: the newest turn — the
    // reason for the render — would sit below the fold. Scrolled up, they keep their place.
    if (kept === null) this.stuck = true;
    else transcript.scrollTop = this.el.lastTop = kept;
    this.follow(transcript);
    if (pending.length) {
      // The Markdown lands after that scroll: follow it down, or put their place back once —
      // unless they scrolled meanwhile.
      const anchor = transcript.scrollTop;
      Promise.all(pending).then(() => {
        if (this.el.transcript !== transcript) return;
        if (this.stuck) this.follow(transcript);
        else if (transcript.scrollTop === anchor) transcript.scrollTop = this.el.lastTop = kept;
      }).catch((e) => console.error("[slim] render turns", e));
    }
  },

  // A sent image stays a thumbnail: the server serves the stored bytes back. A pending image
  // has its own preview; a ref that fails to load falls back to the name chip.
  renderAttachments(message, attachments) {
    const gallery = create(message, "div", "slim-message-images");
    const threadId = this.activeThread?.id;
    for (const attachment of attachments) {
      const name = attachment.name || "Attached image";
      const src = attachment.preview || (attachment.id && threadId ? this.imageURL(threadId, attachment.id) : "");
      if (!src) { create(gallery, "span", "slim-image-chip", name); continue; }
      const image = create(gallery, "img", "slim-message-image");
      image.src = src;
      image.alt = name;
      image.onload = () => this.follow();
      image.onerror = () => { image.remove?.(); create(gallery, "span", "slim-image-chip", name); };
    }
  },

  // Under an answer: copy, regenerate (last answer only, while idle), and how it was made.
  renderActions(message, turn) {
    const actions = create(message, "div", "slim-message-actions");
    const copy = iconButton(actions, "copy", "Copy answer");
    copy.onclick = () => {
      if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) navigator.clipboard.writeText(turn.text);
    };
    if (turn === this.turns.at(-1) && !this.sending) {
      const again = iconButton(actions, "refresh-cw", "Regenerate");
      again.onclick = () => this.regenerate();
    }
    create(actions, "small", "slim-message-meta", metaText(turn.turn));
  },

  renderSoon() {
    if (this.renderQueued) return;
    this.renderQueued = true;
    queueMicrotask(() => { this.renderQueued = false; this.render(); });
  },
};
