// CopilotView: drawing the sidebar, its threads, its transcript and the streaming answer.
"use strict";

const { MarkdownRenderer, MarkdownRenderChild } = require("obsidian");
const { groupThreads, pathStem, noteTitle, groupCitations, slashQuery, matchSkills, inputsText,
        normalizeMath, LIVE_RENDER_MS, metaText, stageLine, SUGGESTED_PROMPTS } = require("./text");
const { clear, create, iconButton, autoGrow, atBottom } = require("./dom");

module.exports = {
  renderSlash() {
    const menu = this.el?.slash;
    if (!menu) return;
    clear(menu);
    const hits = matchSkills(this.skills, slashQuery(this.draft));
    menu.hidden = !hits.length;
    for (const skill of hits) {
      const item = create(menu, "button", `slim-slash-item${skill.error ? " is-broken" : ""}`);
      create(item, "span", "slim-slash-name", `/${skill.name}`);
      create(item, "small", "slim-slash-about", skill.error ? `Broken: ${skill.error}` : skill.description || "");
      item.onmousedown = (event) => event.preventDefault();     // keep the composer focused
      item.onclick = () => this.pickSkill(skill);
    }
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

  renderStage() {
    if (!this.el) { this.render(); return; }
    if (!this.el.stage) return;
    this.el.stage.textContent = stageLine(this.stage, this.stageSince, Date.now());
  },

  renderMarkdown(el, markdown, host = this.mdHost || this) {
    const text = normalizeMath(markdown);
    if (MarkdownRenderer && typeof MarkdownRenderer.render === "function") {
      return MarkdownRenderer.render(this.app, text, el, this.context?.source?.path || "",
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
    history.onclick = () => { this.showThreads = !this.showThreads; this.render(); };
    const add = iconButton(tools, "plus", "New chat");
    add.onclick = () => this.newThread();

    if (this.loading) { create(root, "p", "slim-copilot-status", "Indexing this note..."); return; }
    if (this.error && !this.context) { create(root, "p", "slim-copilot-error", this.error); return; }
    if (!this.context) { create(root, "p", "slim-copilot-status", "Open a Markdown note."); return; }

    const sourcePath = this.context.source.path;
    const sourceTitle = this.context.source.title || pathStem(sourcePath);
    const related = this.context.related || [];
    if (related.length) {
      // The title is the chip; Nearby folds under it. Closed by default: it is context for
      // them to reach for, not the reason the pane is open.
      const context = create(header, "details", "slim-copilot-context");
      context.open = this.nearbyOpen;
      const summary = create(context, "summary", "slim-copilot-context-summary");
      summary.onclick = () => { this.nearbyOpen = !context.open; };   // pre-toggle state, as with groups
      const note = create(summary, "span", "slim-copilot-note", sourceTitle);
      note.setAttribute("title", sourcePath);
      create(summary, "small", "slim-copilot-nearby-count", `${related.length} nearby`);
      for (const item of related) {
        const button = create(context, "button", "slim-related-note");
        button.setAttribute("title", item.path);
        create(button, "span", "", item.title || pathStem(item.path));
        create(button, "small", "", item.scope || "");
        button.onclick = () => this.openNote(item.path);
      }
    } else {
      const note = create(header, "div", "slim-copilot-note", sourceTitle);
      note.setAttribute("title", sourcePath);
    }

    if (this.showThreads) this.renderThreadList(root, sourceTitle);
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
    const slash = create(composer, "div", "slim-slash-menu");
    this.el.slash = slash;
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
      autoGrow(input);
      if (input.value === "/") this.loadSkills();
      else this.renderSlash();
    };
    input.onfocus = () => { this.composerFocused = true; };
    input.onblur = () => { this.composerFocused = false; };
    this.el.input = input;
    input.onkeydown = (event) => {
      // A half-typed command completes before it sends: "/con" + Enter is "/consolidate… ".
      const hits = matchSkills(this.skills, slashQuery(this.draft));
      if (hits.length && (event.key === "Tab"
          || (event.key === "Enter" && !event.shiftKey && this.draft.length > 1
              && !hits.some((h) => `/${h.name}` === this.draft)))) {
        event.preventDefault();
        this.pickSkill(hits[0]);
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
    if (focused && typeof input.focus === "function") {
      input.focus();
      input.selectionStart = input.selectionEnd = input.value.length;
    }
  },

  renderThreadList(root, sourceTitle) {
    const list = create(root, "nav", "slim-copilot-threads");
    if (!this.threads.length) {
      create(list, "p", "slim-copilot-status", "No chats yet.");
      return;
    }
    for (const group of groupThreads(this.threads, this.context.source.id)) {
      const details = create(list, "details", "slim-note-group");
      details.open = this.groupOpen.has(group.sourceId)
        ? this.groupOpen.get(group.sourceId) : group.expanded;
      const label = group.sourceId === this.context.source.id
        ? sourceTitle : noteTitle(this.app, group.path);
      const summary = create(details, "summary", "", label);
      summary.setAttribute("title", group.path);
      // At click time `open` is still the pre-toggle state; the browser flips it afterwards.
      summary.onclick = () => { this.groupOpen.set(group.sourceId, !details.open); };
      for (const thread of group.threads) {
        const row = create(details, "div", `slim-chat-row${thread.id === this.activeThread?.id ? " is-active" : ""}`);
        const open = create(row, "button", "slim-chat-open", thread.title || "Conversation");
        open.onclick = () => this.openThread(thread.id);
        const rename = iconButton(row, "pencil", `Rename ${thread.title}`);
        rename.onclick = () => this.renameThread(thread.id);
        const armed = this.deleteArmed === thread.id;
        const remove = iconButton(
          row, armed ? "check" : "trash-2",
          armed ? `Confirm delete ${thread.title}` : `Delete ${thread.title}`,
          armed ? "clickable-icon is-armed" : "clickable-icon");
        remove.onclick = () => this.deleteThread(thread.id);
      }
    }
  },

  renderTranscript(root) {
    const transcript = create(root, "section", "slim-copilot-transcript");
    this.el.transcript = transcript;
    const pending = [];
    for (const turn of this.turns) {
      const message = create(transcript, "article", `slim-message is-${turn.role}`);
      const run = turn.role === "slim" ? turn.turn?.edit_run : null;
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
          chip.onclick = () => this.openNote(source.path);
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
