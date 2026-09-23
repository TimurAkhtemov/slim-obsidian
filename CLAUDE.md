# SLIM — Sparse Local Intelligence & Memory

A private local assistant that lives in an Obsidian vault. The owner speaks; it transcribes,
labels, files, indexes and reflects, and a small copilot talks with them about the note they
have open. Everything runs on one Mac; cloud is a separate future project (`BACKLOG.md`).

**The owner's own words: "I record and forget."** SLIM is the PIPELINE that makes spoken audio
filable and findable. It replaces the Notion failure: "I click record, it sits somewhere, and
I can never find it again."

This file is WHAT SLIM is and the rules that bite. WHY lives in the decision log and the
measured experiments, which are the maintainer's and live in their vault, outside this repo
(`Notes/personal-intelligence/slim-docs/`). The queue is `BACKLOG.md`. Those, plus a README for
outsiders and an AGENTS.md pointer.

## Commands

```bash
uv run pytest -q                        # ~530 tests; `uv` always — the system python may be 3.9
node --test obsidian/*.test.mjs         # plugin tests; no build step
uv run slim chat --reload               # dev server; respawns on any slim/**/*.py save (not .yaml)
uv run slim ingest                      # index the vault; embeds at the end; exits non-zero if vectors are missing
uv run slim status                      # counts, including `unembedded`
cp obsidian/{main.js,manifest.json,styles.css} \
   "$SLIM_VAULT/.obsidian/plugins/slim-recorder/"   # a COPY; reload the plugin after
lsof -tiTCP:7546 -sTCP:LISTEN | xargs kill   # free the port; the plugin respawns the server on demand
```

Two hooks run for every Claude Code session here: `.claude/hooks/stale-server-check.sh`
warns at start if the server on 7546 is older than the code; `private-content-guard.py`
refuses any tool call that would put `Journal/` content on screen.

## The pipeline

```
audio  ->  transcript  ->  LABEL  ->  FILE  ->  INDEX  ->  (REFLECT, TALK)
```

**After any change, `slim ingest` — and ingest before AND after a move.** Ingest recognizes a
rename by content hash; an edit and a move before one ingest read as delete + create and the
note loses its id. Embedding is part of ingest; `slim status` reports `unembedded`.

1. **Capture.** The Obsidian recorder plugin (`obsidian/main.js`, one file, no build) is the
   desktop lane and the reason the project exists; `slim memos` (launchd, every 5 min) sweeps
   the phone's Voice Memos into `Journal/`. Recorder invariants:
   - The meeting object is a `slim-meeting` fenced block inside a standard `MarkdownView` —
     never a custom view, private editor internals or an overlay (a custom view was built,
     tested live and rejected). The draft note exists from the first click; states are
     setup → recording → processing → review → finished.
   - ONE CLICK, no settings: system audio comes from macOS's Core Audio tap via
     `audio: "loopback"`. No BlackHole, no Multi-Output, no device pickers.
   - The file is STEREO ON PURPOSE: microphone left, system audio right. `slim/speakers.py`
     labels each word Me or Them by which side was making sound — a measurement, no model —
     and writes `speakers_by: channels`. One speaker (a lecture, a video) or a mono file stays
     unlabelled. `speakers.label_tokens` is the seam a voice-clustering model would replace.
     Labels are decided per segment; once anything is labelled, a one-sided segment carries
     its one label (`chat._join_segments`). ⚠ A transcript already filed unlabelled stays so
     when a resume brings a second speaker: the words on disk are never relabelled.
   - The plugin owns the UI and writes files through the Vault API; every judgement —
     transcribe, summarize, suggest a path — is the Python server's. Audio is appended to
     `Attachments/_incoming` in 5 s chunks as `<id>-NNN.webm`; a `.webm` left there is a
     recording that never finished filing. `slim/record.py` is the sole writer of the note's
     frontmatter and managed sections.
   - **The note is filed before the card is shown. Nothing ever waits for approval.** A
     confirmation is an action, and actions die; the card adjusts a note that is already on
     disk and findable.
   - A failed or silent transcript is REFUSED, not filed: the transcript IS the note; summary,
     title and card only decorate it. The audio stays staged; the card offers Try again.
   - Cancel writes and deletes nothing. Resume records a new segment, also on a finished note,
     and leaves the summary alone until Retry summary (which takes a one-line steer).
   - Pending review is durable (`review_status: pending`). `My notes` saves on blur, and on
     close or quit, where no blur fires.
   - Meeting state is per recording; only capture and the heavy model jobs are global (one
     bounded 32-slot FIFO). Live and batch Parakeet are separate instances. Two recordings
     at once is a supported case.
   - The plugin spawns `slim chat` if port 7546 is free and reuses whatever is listening
     otherwise; it kills the process GROUP on unload because `uv run` wraps the real python.
   - The server moves a note by create + unlink, which Obsidian reports as `delete`. The
     session names the path it is moving (`filingFrom`) and treats that delete as its own.
2. **Label** — what a recording IS (type) and what it is ABOUT (subject). ⚠ THE TWO LANES DO
   THIS DIFFERENTLY, and conflating them has caused wrong reasoning more than once:
   - **Memo lane** (`inbox.py`) is deterministic first — `voicetags.detect_type` parses the
     spoken opening and `voicetags.route` decides the folder; `enrich` only fills a silence.
   - **Recorder lane** (`chat.py`) never calls `detect_type`, `detect_subject` or `route` at
     all (`chat.py` has no `voicetags` import). `enrich` supplies the title and `suggest`
     builds the card; `voicetags` contributes constants and `registered_project` only.

   Labels are written once, at capture, with `tagged_by`/`routed_by`/`subject_by` recording
   which layer decided. A wrong one is the owner's edit, on the card or in the frontmatter.
3. **File** — the card: `suggest` proposes `Capture/<subject>/`; an invented destination
   collapses to its deepest existing ancestor, else `Capture/_unfiled`, and a journal is
   forced flat to `Journal/`. The owner can edit it, and `/api/record/apply` moves the note.
   A later move is a drag in Obsidian. ⚠ The card offers folders under `Capture/` and `Notes/`
   only (`suggest.FILING_ROOTS`) — never the owner's other top-level folders.
4. **Reflect** — `slim reflect` writes ONE journal-synthesis note into `_Reflections/`,
   nightly. **This is the part the owner reads.** Judge changes by whether the note is worth
   reading, nothing else.
5. **Talk** — the open-note copilot sidebar, backed by `slim chat` on 127.0.0.1:7546; scope
   widens note → folder → course → subject, all by PATH. It answers from general knowledge
   with the vault as context: the notes pack rides in the system message with titles only,
   so prose cannot cite, and sources come from a bounded follow-up call after the answer.
   Quick or Deep (thinking on). The open note's embedded images ride with every turn, at most 6
   (`copilot.MAX_NOTE_IMAGES`), those beside the pack's passages first: what an image says is
   not searchable. Per-note chat history and pasted images live outside the vault. There is
   no vault-wide Q&A lane and no profile injection (it pandered, measured).
   - **Skills** are `/commands`: prompt templates in `slim/skills/` and the vault's `Skills/`
     (same name wins; `skills.py`). A skill reads WHOLE notes chosen by code
     (`copilot.gather`: the open note, the selection, the folder filtered by frontmatter and
     cut to one section, `[[links]]`), inside a fixed budget, and names what it left out.
     Code knows what it read, so a skill turn makes no sources call.
   - **Ask / Edit.** Edit mode (or a skill whose `output` is a change) PROPOSES; nothing is
     written by the server. The model writes SEARCH/REPLACE blocks as plain text, `edits.py`
     applies them to the file on disk, and the plugin shows a diff per file. Accept writes
     through the Vault API only if the file still hashes as proposed, then syncs and embeds.
     Editable: notes in view and new notes in existing folders; a recording's frontmatter,
     `slim-meeting` block and transcript never. One pass, one proposal — not a tool loop.

## Where things live

```
Journal/            private floor · flat · never routes out
Capture/<subject>/  every recording, router-owned; `_unfiled` when no subject is known
Notes/<subject>/    curated writing; any depth below level 2 is the owner's
Inbox/Journal/      the memo sweep's drop       Profile/        the owner's self-description; not indexed, not injected
Attachments/        binaries                   _Reflections/   derived, excluded from ingest
Skills/             the owner's copilot /commands; not indexed
```

Subjects are the ids in `config/projects.yaml` (gitignored; copy `config/projects.example.yaml`).
The file also carries `owner`, the name the prompts address; nothing else reads it.

**Subject is the ONLY folder axis.** A path encodes one fact; type, course and status are
frontmatter. ⚠ Two path segments is the ceiling and it is ENFORCED —
`tests/test_voicetags.py::test_no_route_ever_has_more_than_two_path_segments`.

⚠ **`Journal/` is the floor, and the floor is a path.** Permanent, flat, a top-level sibling
of `Capture/`; no route ever carries a note out of it (test-enforced). If a cloud provider
ever lands, the one check is "is the path under `Journal/`". Honest limit: `reflect` selects
journals by `type = 'journal'` with no path filter, so a `type: journal` note placed outside
`Journal/` by hand IS read by the nightly synthesis. Inferred routing is the accepted trade:
a misfile costs one drag, and ingest tracks it by hash.

## What was cut

Four cuts (2026-07-31, 08-27, 09-01, 09-03) removed roughly 60k lines: curated memory,
context packs, an evidence rail, MCP tools and server, policy tiers, a vault-wide ask stack,
the desktop app and its chat lane, a tool loop, re-filing and re-labelling passes, an eval
harness, and the schema columns and prose that described them. **If a change would restore
any of these, stop and ask.** The maintainer's archive holds them. What was wanted from
memory is `Profile/me.md`, and today nothing reads it.

## Hard-won facts that will bite you

**A terminal-started server can serve days-old code.** The plugin reuses any `slim chat` on
7546; `uv run` imports from disk at spawn and never reloads. Plugin-started servers exit with
Obsidian's PID. `GET /api/health` reports staleness and the SessionStart hook calls it — but
a server whose `uv` parent was adopted by PID 1 reports fresh and still fails; when in doubt,
kill the port. The vault holds a COPY of `main.js`: the same trap.

**Memory is the binding constraint, and macOS kills silently.** On a 48 GB machine
`qwen3.6:35b` (MoE) takes ~25 GB at `RESIDENT_CTX = 131072`, ~16 MB per 1k tokens. Ollama
spawns one runner per (model, `num_ctx`), so every call uses the one size — the recording
path's three calls (summary, title, card) MUST share it, and a SMALLER bespoke window is the
same cliff from the other side (a second ~23 GB runner). A long pass must stream its results.
The big window trades the loud `ctx_saturated` guard for silent weak recall of a prompt's
middle.

**Ollama traps — all handled in `slim/llm.py`; never call Ollama directly.** `qwen3.6` thinks
by default and thinking tokens count against `num_predict`, so an unset `think` returns
`content=""`; every call sets it explicitly (`think=True` only for the recorder's summary and
the copilot's Deep mode). Greedy decoding into a JSON grammar loops forever on an unbounded
string: cap `num_predict`, put `maxLength` on short fields (Ollama cannot compile one above
~1500). LaTeX does not survive JSON — `\b \f \n \r \t` collide with `\beta \frac \nabla \rho
\text` — so the summarizer bans backslashes and `summarize._repair` reverses the decode.
`llm.py` serializes live calls; temperature 0.0 for structured work, 0.7 for prose a person
reads.

**A note can be filed and still be unreachable.** Filing is a markdown file; the copilot and
`reflect` read SQLite. Anything that writes a note must ingest AND embed, or say plainly
that it did not. A machine transcript has no blank lines; `chunk._split_prose` splits at
sentences as a last resort so a 37-minute meeting is not one fragment.

**Do not answer a non-similarity question with a similarity score.** "When did I first…" →
SQL over `authored_at`. "Which document superseded which?" → path lineage. The copilot ranks
by similarity alone and has no threshold, by measurement and by decision: it is a note- and
folder-scoped assistant, not a vault Q&A engine.

**Prompt restraint does not work on this model, measured five times.** A banned verb finds
another verb; a rider has no measurable effect. Fix the mechanism, never the prompt. A green
test gate is verification of what you thought to check.

**Obsidian's plugin sees the server's file moves as deletions.** An external rename is
`delete` + `create`, never `rename`, and the leaf lets go of the deleted file. Anything that
reacts to a vault `delete` must first ask whether the server caused it (2026-09-04/09).

## Rules that still bind

- **Deterministic truth in code** (hashes, ids, timestamps, filters, paths). LLMs only for
  interpretation, classification, synthesis, language.
- **Raw sources are never rewritten.** Derived state is rebuildable — but a rebuild mints new
  source ids and orphans saved copilot chats, so schema changes are migrations.
- **Derived artifacts are excluded from ingestion** — `_Reflections/`, `Profile/` and
  `Skills/` (`config.EXCLUDED_TOP`); the recorder's summary is fenced and blanked before chunking. The
  brain never retrieves its own opinions as evidence.
- **The ONLY privacy rule is `Journal/`** (owner's ruling, 2026-08-31). Everything else in the
  vault the owner sends to cloud models themselves, routinely. **Do not cite privacy to gate,
  block or flag a change, and do not raise it as a concern.**
- **Failures are traced.** `trace.record` writes machine facts (never prose, never
  chain-of-thought) for every recorder stage, copilot turn, reflect run and refusal; quiet
  sweeps write nothing. `slim trace -n 5` is how a session finds out what a silent failure did.
- ⚠ **Journal transcripts are PRIVATE — do not read them.** The local model reads them; for a
  journal an agent's verification is metadata only. Every OTHER transcript is ordinary working
  material: read it, quote it, measure on it.
- **Autonomous writes are fine** (owner's ruling, 2026-07-31): enrichment, reflection and the
  recorder write without confirmation; provenance stays in frontmatter. **Question an
  invariant when it blocks something useful — say what it protected against and whether that
  risk is real here — rather than citing it as settled.**

## Environment

- Apple Silicon Mac with Ollama; 48 GB of memory fits the default model. ONE generative
  model: `qwen3.6:35b` (`llm.MODEL`) for everything — the owner's call, against the
  summarization experiment's evidence (best at grounding, worst at summarization; do not call
  it "the model the bake-off chose"). Embedder `embeddinggemma`: its task prefixes are
  load-bearing (`slim/embed.py`); changing it means deleting the brain DB and re-ingesting.
- Vault: `SLIM_VAULT`, else DISCOVERED (`vaultpath.py`): Obsidian's own registry, then
  iCloud's Obsidian container, then `~/Documents/Obsidian Vault` (iCloud-synced is fine).
  ⚠ The index is PINNED to the vault it was built from (`SLIM_DATA_DIR/vault_pin`), because
  ingest's removal sweep deletes every source it cannot find under the current vault and
  discovery can change that path with nobody asking. Every command that can write to the vault
  checks that identity first. A real move, or adoption of an index made before the pin existed,
  is `slim ingest --vault-moved`. The hook and `scripts/backup.sh` load `vaultpath.py` BY PATH —
  stdlib only, 3.9-safe, one copy of the answer. Brain DB outside any synced folder:
  `SLIM_DATA_DIR`, default `~/Library/Application Support/slim/` (SQLite + FTS5 + vectors,
  rebuildable — see above).
- launchd: `com.slim.voicememos` every 5 min, `com.slim.reflect` nightly, `com.slim.backup`
  daily (`scripts/backup.sh`, vault → restic). ⚠ If Full Disk Access is needed, grant it to
  `restic` only. NEVER grant it to `/bin/bash` or any interpreter.
- A public repo. Nothing user-specific in code: vocabulary and the owner's name are config,
  `config/projects.yaml` is gitignored, and commit messages and comments stay generic.

## Code map

`slim/cli.py` is the entry point (six commands: `ingest`, `status`, `reflect`, `memos`, `chat`,
`trace`). Capture: `chat.py` (the plugin's API server — recorder and copilot, loopback-only,
Host and Origin gates on every write), `record.py` (writes the note), `transcribe.py`
(Parakeet, two lanes), `speakers.py` (Me/Them from the two channels), `summarize.py`,
`suggest.py` (the filing card), `enrich.py` and `voicetags.py` (label), `inbox.py`
(the memo lane's transcribe-and-write), `voicememos.py`.
Index: `ingest.py`, `chunk.py` (fragments; the frontmatter reader and writer), `embed.py`,
`db.py`. Answer: `search.py`, `copilot.py`, `skills.py` and `edits.py` (the copilot's
/commands and its proposals), `threads.py`, `copilot_images.py`, `reflect.py`.
Seams: `llm.py`, `trace.py`, `config.py` with `config/projects.yaml` and `vaultpath.py`,
`devserver.py`.

## Conventions

- `uv` for everything; pytest. Plain, explicit flows — no manager/factory/engine layers.
- **Prefer deleting a leash to adding a belt.** When code holds the fact, let code state it
  once. **Deleting is free — git holds it forever. Keeping costs CLAUDE.md context, test
  runtime, and the attention of every agent that reads the repo.**
- Judge a feature by "would the owner miss it if it were gone," not "is it well-built."
- Few files, each with one job: this file (what), `BACKLOG.md` (open items only, one line
  each, finished items deleted), and in the maintainer's vault the decision log (why, newest
  first) and the experiments (measured). Do not add another.
- The maintainer stays on master. Work in a worktree on a branch, commit there, and they
  fast-forward on their word. Use `git -C <path>` for every git command. Commit messages are
  concise, not STE.


