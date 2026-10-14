// Meeting sessions: one RecorderView per recording, and the one capture slot.
"use strict";

const { DRAFT_SUFFIX, localDay, recordingId } = require("./meeting-note");
const { RecorderView } = require("./recorder-view");

/* One durable meeting object per recording/note, with one reserved capture slot.
 *
 * RecorderView deliberately remains the session controller: it already owns the draft, staged
 * segments, processing promise, review edits and recovery behavior. This manager owns only the
 * facts that are global across those controllers — lookup, path changes, and the exclusive
 * microphone/system-audio lease. Processing and review sessions never occupy that lease.
 */
class MeetingSessionManager {
  constructor(plugin, sessionFactory = null) {
    this.plugin = plugin;
    this.sessionFactory = sessionFactory || ((leaf) => new RecorderView(leaf, plugin));
    this.sessions = new Map();
    this.paths = new Map();
    this.activeCaptureId = null;
  }

  createSession(leaf = null) {
    const session = this.sessionFactory(leaf);
    session.sessionId = session.sessionId || recordingId();
    if (!session.recordingId) session.recordingId = session.sessionId;
    session.manager = this;
    this.sessions.set(session.sessionId, session);
    this.indexSession(session);
    return session;
  }

  indexSession(session) {
    if (!session) return null;
    for (const [path, owner] of this.paths) {
      if (owner === session && path !== session.draftRel && path !== session.filedNotePath) {
        this.paths.delete(path);
      }
    }
    for (const path of [session.draftRel, session.filedNotePath]) {
      if (path) this.paths.set(path, session);
    }
    return session;
  }

  sessionForPath(path) { return (path && this.paths.get(path)) || null; }

  sessionForLeaf(leaf) {
    if (!leaf) return null;
    for (const session of this.sessions.values()) {
      if (session.editorLeaf === leaf) return session;
    }
    return null;
  }

  claimCapture(session) {
    if (!session || !session.sessionId) throw new Error("meeting session has no identity");
    // Only a session that is recording holds the microphone. The ribbon's setup draft takes
    // the lease so a second click returns to it, but it captures nothing — a Resume elsewhere
    // takes over rather than reporting a recording they could not see (2026-09-05).
    const holder = this.activeCapture();
    if (holder && holder !== session && holder.state === "recording") {
      throw new Error("another meeting capture is already active");
    }
    this.activeCaptureId = session.sessionId;
    return session;
  }

  /* A setup draft that was never started keeps the lease for as long as Obsidian stays open, so
   * a click days later reopened it under the name of the day it was made (2026-09-17). An
   * untouched one from an earlier day is deleted and replaced; one with their input is kept. */
  async releaseStaleSetupDraft(session) {
    if (session.state !== "idle" || (session.segments && session.segments.length)) return false;
    if (String(session.recordingId || "").slice(0, 8) === localDay()) return false;
    if (typeof session.removeUntouchedDraft !== "function") return false;
    if (!(await session.removeUntouchedDraft())) return false;
    this.removeSession(session);
    return true;
  }

  releaseCapture(session) {
    if (session && this.activeCaptureId === session.sessionId) this.activeCaptureId = null;
  }

  activeCapture() {
    const session = this.activeCaptureId && this.sessions.get(this.activeCaptureId);
    if (!session && this.activeCaptureId) this.activeCaptureId = null;
    return session || null;
  }

  async activate(leaf = null) {
    let session = this.activeCapture();
    if (session && await this.releaseStaleSetupDraft(session)) session = null;
    if (!session) {
      // A processing session may navigate its leaf to the filed note at any moment. Starting B
      // in that same leaf lets A's completion replace B's live recorder. Give the new capture a
      // fresh pane whenever the current one already belongs to another durable meeting object.
      const availableLeaf = this.sessionForLeaf(leaf) ? null : leaf;
      session = this.createSession(availableLeaf);
      this.claimCapture(session);
    }
    await session.open();
    this.indexSession(session);
    return session;
  }

  async openPath(path, leaf = null) {
    const session = this.sessionForPath(path);
    if (!session) return null;
    if (leaf) session.editorLeaf = leaf;
    await session.open();
    this.indexSession(session);
    return session;
  }

  async adoptFile(file, leaf = null) {
    if (!file || !file.path) return null;
    const existing = this.sessionForPath(file.path);
    if (existing) {
      if (leaf) existing.editorLeaf = leaf;
      return existing;
    }
    const session = this.createSession(leaf);
    let adopted = false;
    if (file.path.endsWith(DRAFT_SUFFIX) && typeof session.adoptPausedDraft === "function") {
      adopted = await session.adoptPausedDraft(file.path);
      if (adopted) session.editorLeaf = leaf;
    } else if (typeof session.openExisting === "function") {
      adopted = await session.openExisting(file, leaf);
    }
    if (!adopted) {
      this.removeSession(session);
      return null;
    }
    this.indexSession(session);
    if (typeof session.attachEditorObject === "function") session.attachEditorObject();
    if (typeof session.render === "function") session.render();
    return session;
  }

  removeSession(session) {
    if (!session) return;
    this.releaseCapture(session);
    for (const [path, owner] of this.paths) {
      if (owner === session) this.paths.delete(path);
    }
    this.sessions.delete(session.sessionId);
  }

  ownsEditor(info) {
    for (const session of this.sessions.values()) {
      if (typeof session.ownsEditor === "function" && session.ownsEditor(info)) return session;
    }
    return null;
  }

  handleVaultDelete(file) {
    const session = file && this.sessionForPath(file.path);
    if (!session) return;
    // The session knows which deletions are the server's own work (the draft during filing,
    // the old path during a move). Every other deletion is the user's instruction to release
    // this one object, including the setup draft that reserves the microphone action.
    if (session.expectsDeletion(file.path)) return;
    if (typeof session.handleVaultDelete === "function") session.handleVaultDelete(file);
    this.removeSession(session);
  }

  handleVaultRename(file, oldPath) {
    const session = this.sessionForPath(oldPath);
    if (!session || !file || !file.path) return;
    if (session.draftRel === oldPath) session.draftRel = file.path;
    if (session.filedNotePath === oldPath) {
      session.filedNotePath = file.path;
      if (session.result) session.result.note = file.path;
    }
    this.indexSession(session);
  }

  flushNotes() {
    return Promise.all([...this.sessions.values()].map((session) =>
      typeof session.flushNotes === "function" ? session.flushNotes() : null));
  }

  onClose() {
    for (const session of this.sessions.values()) {
      if (typeof session.onClose === "function") session.onClose();
    }
    this.sessions.clear();
    this.paths.clear();
    this.activeCaptureId = null;
  }
}

module.exports = { MeetingSessionManager };
