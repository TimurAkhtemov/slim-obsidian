// CopilotView: drawing the sidebar, its threads, its transcript and the streaming answer.
"use strict";

const { MarkdownRenderer, MarkdownRenderChild } = require("obsidian");
const { threadRows, shortDate, pathStem, groupCitations, slashQuery, matchSkills, mentionQuery, inputsText,
        normalizeMath, LIVE_RENDER_MS, metaText, stageLine, SUGGESTED_PROMPTS } = require("./text");
const { clear, create, iconButton, autoGrow, atBottom } = require("./dom");

module.exports = {
  renderSlash() {
    const menu = this.el?.slash;
    if (!menu) return;
    clear(menu);
    const hits = this.slashDismissed ? [] : matchSkills(this.skills, slashQuery(this.draft));
    menu.hidden = !hits.length;
    if (!hits.length) { this.slashIndex = 0; return; }
    const idx = Math.min(Math.max(0, this.slashIndex || 0), hits.length - 1);
    this.slashIndex = idx;
    hits.forEach((skill, i) => {
      const cls = `slim-slash-item${skill.error ? " is-broken" : ""}${i === idx ? " is-selected" : ""}`;
      const item = create(menu, "button", cls);
      create(item, "span", "slim-slash-name", `/${skill.name}`);
      create(item, "small", "slim-slash-about", skill.error ? `Broken: ${skill.error}` : skill.description || "");
      item.onmousedown = (event) => event.preventDefault();     // keep the composer focused
      item.onclick = () => { if (!skill.error) this.pickSkill(skill); };
    });
  },

  // The `@` picker, built like the slash menu: notes and folders, the chosen row highlighted.
  renderMention() {
    const menu = this.el?.mention;
    if (!menu) return;
    clear(menu);
    const hits = this.mentionDismissed ? [] : this.mentionHits();
    menu.hidden = !hits.length;
    if (!hits.length) { this.mentionIndex = 0; return; }
    const idx = Math.min(Math.max(0, this.mentionIndex || 0), hits.length - 1);
    this.mentionIndex = idx;
    hits.forEach((hit, i) => {
      const item = create(menu, "button", `slim-slash-item${i === idx ? " is-selected" : ""}`);
      create(item, "span", "slim-slash-name", hit.kind === "folder" ? `${hit.path}/` : pathStem(hit.path));
      create(item, "small", "slim-slash-about", hit.kind === "folder" ? "Folder" : hit.path);
      item.onmousedown = (event) => event.preventDefault();
      item.onclick = () => this.pin(hit);
    });
  },

  imageURL(threadId, imageId) {
    const port = this.plugin.settings?.port || 7546;
    return `http://127.0.0.1:${port}/api/copilot/image?thread=${encodeURIComponent(threadId)}`
      + `&id=${encodeURIComponent(imageId)}`;
  },

  // Per-token updates touch two live nodes. A full render() per delta tore the whole view
  // down ~40 times a second: the transcript's scroll position reset to the top on every
  // token, the composer lost focus per keystroke, and a hand-expanded group snapped shut.
  renderDelta() {
    if (!this.el) { this.render(); return; }
    if (!this.el.liveText) return;          // the chat list is showing; the next render catches up
    const transcript = this.el.transcript;
    const follow = atBottom(transcript);      // decided BEFORE the text grows the pane
    this.el.liveText.textContent = this.streamingText;
    if (follow) transcript.scrollTop = transcript.scrollHeight;
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

  // Each live pass renders under its own child component, dropped with its buffer: a Deep
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
      const follow = atBottom(el.transcript);
      this.dropLive(el.liveRendered, el.liveHost);
      el.liveRendered = buffer;
      el.liveHost = host;
      buffer.hidden = false;
      el.liveText.hidden = true;
      if (follow) el.transcript.scrollTop = el.transcript.scrollHeight;
    }).catch((e) => console.error("[slim] live render", e));
  },

  // The turn's lookups, a muted line each, as Pi shows each tool call: drawn live as each one ran
  // (`liveTools`), and again from the saved turn. The edit run's step style.
  fillTools(list, tools) {
    clear(list);
    list.hidden = !tools?.length;
    for (const tool of tools || []) {
      const item = create(list, "li", `slim-step slim-tool${tool.refused ? " is-refused" : ""}`);
      create(item, "span", "slim-step-icon", tool.refused ? "–" : "✓");
      create(item, "span", "slim-step-text", String(tool.label || tool.name || ""));
    }
    return list;
  },

  renderTools() {
    if (this.el?.liveEdit) this.renderEdit();
    else if (this.el?.liveTools) this.fillTools(this.el.liveTools, this.liveTools);
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
    clear(root);
    this.el = {};
    // Rendered Markdown attaches child components (embeds, callouts) to the component it is
    // given. One host per render, replaced by the next, so they do not pile up on the view.
    if (this.mdHost) { this.removeChild(this.mdHost); this.mdHost = null; }
    if (MarkdownRenderChild && typeof this.addChild === "function") {
      this.mdHost = this.addChild(new MarkdownRenderChild(root));
    }
    root.addClass?.("slim-copilot");
    const header = create(root, "header", "slim-copilot-head");
    const headRow = create(header, "div", "slim-copilot-head-row");
    create(headRow, "div", "slim-copilot-eyebrow", "SLIM");
    const tools = create(headRow, "div", "slim-copilot-tools");
    const history = iconButton(tools, "history", this.showThreads ? "Back to the chat" : "Chats",
      this.showThreads ? "clickable-icon slim-history is-active" : "clickable-icon slim-history");
    history.onclick = () => {
      this.showThreads = !this.showThreads;
      this.render();
      if (this.showThreads) this.loadThreads();
    };
    const add = iconButton(tools, "plus", "New chat");
    add.onclick = () => this.newThread();

    // The open note is a chip, not a gate: the chat works with none.
    const sourcePath = this.openNote?.path || "";
    const nearby = this.context?.source?.path === sourcePath ? this.context : null;
    const sourceTitle = nearby?.source?.title || (sourcePath ? pathStem(sourcePath) : "No note open");
    const related = nearby?.related || [];
    const off = Boolean(this.activeThread?.openNoteOff);
    // ✕ on the chip that names the open note: this chat leaves it out until clicked again.
    const toggle = (parent) => {
      if (!sourcePath) return;
      const button = create(parent, "button", `slim-chip-x slim-copilot-note-off${off ? " is-off" : ""}`,
        off ? "\u21A9" : "\u2715");
      const label = off ? "Include the open note again" : "Leave the open note out of this chat";
      button.setAttribute("title", label);
      button.setAttribute("aria-label", label);
      button.onclick = (event) => { event.preventDefault(); event.stopPropagation(); this.toggleOpenNote(); };
    };
    const noteCls = `slim-copilot-note${off ? " is-off" : ""}`;
    if (related.length) {
      // The title is the chip; Nearby folds under it. Closed by default: it is context for
      // them to reach for, not the reason the pane is open.
      const context = create(header, "details", "slim-copilot-context");
      context.open = this.nearbyOpen;
      const summary = create(context, "summary", "slim-copilot-context-summary");
      summary.onclick = () => { this.nearbyOpen = !context.open; };   // pre-toggle state, as with groups
      const note = create(summary, "span", noteCls, sourceTitle);
      note.setAttribute("title", sourcePath);
      toggle(summary);
      create(summary, "small", "slim-copilot-nearby-count", `${related.length} nearby`);
      for (const item of related) {
        const button = create(context, "button", "slim-related-note");
        button.setAttribute("title", item.path);
        create(button, "span", "", item.title || pathStem(item.path));
        create(button, "small", "", item.scope || "");
        button.onclick = () => this.openFile(item.path);
      }
    } else {
      const row = create(header, "div", "slim-copilot-note-row");
      const note = create(row, "div", noteCls, sourceTitle);
      note.setAttribute("title", sourcePath);
      toggle(row);
    }

    if (this.showThreads) this.renderThreadList(root);
    else this.renderTranscript(root);

    const composer = create(root, "footer", "slim-copilot-composer");
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
    const pins = this.activeThread?.pins || [];
    if (pins.length) {
      const chips = create(composer, "div", "slim-pins");
      for (const pin of pins) {
        const chip = create(chips, "span", `slim-pin is-${pin.kind}`);
        chip.setAttribute("title", pin.path);
        create(chip, "span", "slim-pin-name", pin.kind === "folder" ? `${pathStem(pin.path)}/` : pathStem(pin.path));
        const x = create(chip, "button", "slim-chip-x", "\u2715");
        x.setAttribute("aria-label", `Unpin ${pin.path}`);
        x.onclick = () => this.unpin(pin.path);
      }
    }
    const slash = create(composer, "div", "slim-slash-menu");
    this.el.slash = slash;
    this.el.mention = create(composer, "div", "slim-slash-menu slim-mention-menu");
    const modes = create(composer, "select", "slim-mode-select");
    for (const value of ["quick", "deep"]) {
      const option = create(modes, "option", "", value[0].toUpperCase() + value.slice(1));
      option.value = value;
      option.selected = value === this.mode;
    }
    modes.onchange = () => { this.mode = modes.value; };
    const input = create(composer, "textarea", "slim-copilot-input");
    input.placeholder = "Ask SLIM, or / for skills";
    input.value = this.draft;
    input.rows = 1;
    autoGrow(input);
    input.oninput = () => {
      this.draft = input.value;
      this.caret = input.selectionStart;
      this.slashDismissed = false;
      this.slashIndex = 0;
      this.mentionDismissed = false;
      this.mentionIndex = 0;
      autoGrow(input);
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
          autoGrow(input);
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
      autoGrow(input);
      this.renderSlash();
    };
    const picker = create(composer, "input", "slim-copilot-image-picker");
    picker.type = "file";
    picker.accept = "image/png,image/jpeg,image/webp";
    picker.multiple = true;
    picker.onchange = () => { this.addImages(picker.files); picker.value = ""; };
    const attach = iconButton(composer, "image-plus", "Attach images", "clickable-icon slim-attach");
    attach.onclick = () => picker.click();
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
    const send = iconButton(
      composer, this.sending ? "square" : "send", this.sending ? "Stop response" : "Send",
      "mod-cta slim-send");
    send.onclick = () => this.sending ? this.cancel() : this.send();
    this.renderSlash();
    this.renderMention();
    if (focused && typeof input.focus === "function") {
      input.focus();
      input.selectionStart = input.selectionEnd = input.value.length;
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

  drawThreadRows(list) {
    if (!this.threads.length) {
      create(list, "p", "slim-copilot-status", this.threadQuery.trim() ? "No chats match." : "No chats yet.");
      return;
    }
    for (const thread of threadRows(this.threads)) {
      const row = create(list, "div", `slim-chat-row${thread.id === this.activeThread?.id ? " is-active" : ""}`);
      const open = create(row, "button", "slim-chat-open", thread.title || "Conversation");
      open.onclick = () => this.openThread(thread.id);
      const date = shortDate(thread.updated_at);
      if (date) create(open, "small", "slim-chat-date", ` ${date}`);
      if (thread.source_path) {
        const label = create(open, "small", "slim-chat-note", ` ${pathStem(thread.source_path)}`);
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

  renderTranscript(root) {
    const transcript = create(root, "section", "slim-copilot-transcript");
    this.el.transcript = transcript;
    const pending = [];
    for (const turn of this.turns) {
      const message = create(transcript, "article", `slim-message is-${turn.role}`);
      const run = turn.role === "slim" ? turn.turn?.edit_run : null;
      if (turn.role === "slim" && turn.turn?.tools?.length) {
        this.fillTools(create(message, "ul", "slim-steps slim-tools"), turn.turn.tools);
      }
      if (run) {
        this.renderEditRun(message, run, pending);
      } else if (turn.role === "slim") {
        const body = create(message, "div", "slim-message-text slim-rendered-markdown");
        pending.push(this.renderMarkdown(body, turn.text)
          .catch((e) => console.error("[slim] render answer", e)));
      } else {
        create(message, "div", "slim-message-text", turn.text);
      }
      if (turn.attachments?.length) this.renderAttachments(message, turn.attachments);
      // Sources are a footnote row, one chip per note, only when the answer drew on notes.
      const sources = groupCitations(turn.turn?.citations);
      if (sources.length) {
        const list = create(message, "div", "slim-sources");
        create(list, "span", "slim-copilot-label", "From your notes");
        for (const source of sources) {
          const chip = create(list, "button", "slim-source", source.title);
          chip.setAttribute("title", source.path);
          chip.onclick = () => this.openFile(source.path);
        }
      }
      const read = inputsText(turn.turn?.inputs);
      if (read) create(message, "small", "slim-inputs", read);
      if (turn.turn?.verification?.truncated && !run) {
        create(message, "small", "slim-verification", "Cut off at the length limit — ask for the rest.");
      }
      if (run) this.renderEditMeta(message, run, turn);
      else if (turn.role === "slim" && turn.turn) this.renderActions(message, turn);
    }
    const streamingHere = this.sending && this.turns === this.streamingTurns;
    if (streamingHere) {
      const live = create(transcript, "article", "slim-message is-slim is-streaming");
      this.el.live = live;
      if (this.editRun) {
        this.el.liveEdit = live;
        this.drawLiveEdit(live);
      } else {
        this.el.liveTools = this.fillTools(create(live, "ul", "slim-steps slim-tools"), this.liveTools);
        this.el.liveText = create(live, "div", "slim-message-text", this.streamingText);
      }
      this.el.stage = create(transcript, "div", "slim-copilot-stage", stageLine(this.stage, this.stageSince, Date.now()));
    }
    if (this.error) {
      const error = create(transcript, "div", "slim-copilot-error", this.error);
      if (this.retryQuestion) {
        const retry = iconButton(error, "rotate-ccw", "Retry response");
        retry.onclick = () => this.retry();
      }
    }
    if (!this.turns.length && !streamingHere && !this.error) {
      const suggestions = create(transcript, "div", "slim-suggestions");
      for (const prompt of SUGGESTED_PROMPTS) {
        const button = create(suggestions, "button", "slim-suggest", prompt);
        button.onclick = () => this.send(prompt);
      }
    }
    // A full render is a new transcript element at scrollTop 0: the newest turn — the
    // reason for the render — would sit below the fold.
    transcript.scrollTop = transcript.scrollHeight;
    if (pending.length) {
      // The Markdown lands after that scroll. Re-apply it once — unless they scrolled meanwhile.
      const anchor = transcript.scrollTop;
      Promise.all(pending).then(() => {
        if (this.el.transcript === transcript && transcript.scrollTop === anchor) {
          transcript.scrollTop = transcript.scrollHeight;
        }
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
