# SLIM — Sparse Local Intelligence & Memory

A private local assistant that lives in an Obsidian vault. The owner speaks; it transcribes,
labels, files, indexes and reflects, and a small copilot talks with them about the note they
have open. Everything runs on one Mac; cloud is a separate future project (`BACKLOG.md`).

**The owner's own words: "I record and forget."** SLIM is the PIPELINE that makes spoken audio
filable and findable. It replaces the Notion failure: "I click record, it sits somewhere, and
I can never find it again."

This file is what SLIM is and the rules that bind, **each with its reason beside it.** A rule
whose reason no longer holds is deleted, not obeyed; a rule without a reason is not a rule.
Rules marked **(owner)** are the owner's value calls: they change only on the owner's word. The
maintainer's vault (`Notes/personal-intelligence/slim-docs/`) keeps the older decision log and
the measured experiments as history; it stopped at 2026-09-09 and is no longer where reasons
live. The queue is `BACKLOG.md`; the README is for outsiders.

## Commands

```bash
uv run pytest -q                        # ~875 tests; `uv` always — the system python is 3.9
node --test "obsidian/test/**/*.test.mjs"   # plugin tests; they load obsidian/src/ directly
(cd obsidian && npm install && npm run build)   # src/ -> the committed main.js; bundle.test.mjs fails if stale
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

After a change to notes, `slim ingest`; embedding is part of it, and `slim status` reports
`unembedded`. Ingest recognizes a rename by content hash, so **a hand edit plus a hand move
between two ingests reads as delete + create and the note loses its id** (and the chats saved
against it). The server's own edit-and-move passes the old path, so only hand moves can hit this.

1. **Capture.** The Obsidian recorder plugin (`obsidian/src/`, bundled by esbuild into the
   committed `obsidian/main.js`, the one file Obsidian loads) is the desktop lane and the reason
   the project exists; `slim memos` (launchd, every 5 min) sweeps the phone's Voice Memos into
   `Journal/`. How the recorder works, and why:
   - The meeting object is a `slim-meeting` fenced block inside a standard `MarkdownView`,
     never a custom view, private editor internals or an overlay: a custom view was built,
     tested live and rejected (2026-08-23). The draft note exists from the first click; states
     are setup → recording → processing → review → finished.
   - **(owner)** ONE CLICK, no settings: system audio comes from macOS's Core Audio tap via
     `audio: "loopback"`. No BlackHole, no Multi-Output, no device pickers.
   - The file is STEREO ON PURPOSE: microphone left, system audio right, so `capture/speakers.py`
     labels each word Me or Them by which side was making sound — a measurement, no model — and
     writes `speakers_by: channels`. One speaker (a lecture, a video) or a mono file stays
     unlabelled. `speakers.label_tokens` is the seam a voice-clustering model would replace.
     Labels are decided per segment; once anything is labelled, a one-sided segment carries
     its one label (`recorder.pipeline._join_segments`). A transcript already filed unlabelled
     stays so when a resume brings a second speaker: the words on disk are never relabelled.
     Live, the card raises an alarm when BOTH sides are digital zeros (`captureSilence`); one
     side alone never warns, because a zero mic beside playing audio is echo cancellation working.
   - The plugin owns the UI and writes files through the Vault API; every judgement —
     transcribe, summarize, suggest a path — is the Python server's, where the tests are. Audio
     is appended to `Attachments/_incoming` in 5 s chunks as `<id>-NNN.webm`, so a crash costs
     seconds, not the meeting; a `.webm` left there is a recording that never finished filing.
     `recorder/note.py` is the sole writer of the note's frontmatter and managed sections.
   - **The note is filed before the card is shown; nothing waits for approval.** A confirmation
     is an action, and actions die (2026-08-03/04); the card adjusts a note that is already on
     disk and findable.
   - **A failed or silent transcript is REFUSED, not filed.** Three recordings were lost by
     filing empty transcripts (2026-08-05, 08-22). The transcript IS the note; the audio stays
     staged and the card offers Try again.
   - Cancel writes and deletes nothing. Resume records a new segment, also on a finished note,
     and leaves the summary alone until Retry summary (which takes a one-line steer).
   - Pending review is durable (`review_status: pending`). `My notes` saves on blur, and on
     close or quit, where no blur fires.
   - Meeting state is per recording; only capture and the heavy model jobs are global (one
     bounded 32-slot FIFO). Live and batch Parakeet are separate instances. Two recordings
     at once is a supported case.
   - The plugin spawns `slim chat` if port 7546 is free and reuses whatever is listening
     otherwise (that keeps `--reload` usable, and is the stale-server trap below); it kills the
     process GROUP on unload because `uv run` wraps the real python.
   - The server moves a note by create + unlink, which Obsidian reports as `delete`. The
     session names the path it is moving (`filingFrom`) and treats that delete as its own.
2. **Label** — what a recording IS (type) and what it is ABOUT (subject). The two lanes do
   this differently, and conflating them has caused wrong reasoning more than once:
   - **Memo lane** (`memos/inbox.py`) is deterministic first — `voicetags.detect_type` parses the
     spoken opening and `voicetags.route` decides the folder; `enrich` only fills a silence.
   - **Recorder lane** (`recorder/`, `server.py`) never calls `detect_type`, `detect_subject` or
     `route` (neither imports `voicetags`). `enrich` supplies the title and `suggest` builds the
     card; `voicetags` contributes constants and `registered_project` only.

   Labels are written once, at capture, with `tagged_by`/`routed_by`/`subject_by` recording
   which layer decided. A wrong one is the owner's edit, on the card or in the frontmatter.
3. **File** — the card: `suggest` proposes `Capture/<subject>/`; an invented destination
   collapses to its deepest existing ancestor, else `Capture/_unfiled`, and a journal is
   forced flat to `Journal/`. The owner can edit it, and `/api/record/apply` moves the note.
   A later move is a drag in Obsidian. **(owner)** The card offers folders under `Capture/` and
   `Notes/` only (`recorder.suggest.FILING_ROOTS`); `Inbox/` is staging and derived folders are
   never offered either way.
4. **Reflect** — `slim reflect` writes ONE journal-synthesis note into `_Reflections/`,
   nightly. **This is the part the owner reads.** Judge changes by whether the note is worth
   reading, nothing else.
5. **Talk** — the open-note copilot sidebar, backed by `slim chat` on 127.0.0.1:7546; scope
   widens note → folder → course → subject, all by PATH. It answers from general knowledge
   with the vault as context: the notes pack rides in the system message with titles only,
   so prose cannot cite (measured 2026-09-01), and sources come from a bounded follow-up call
   after the answer. It ranks by similarity alone, with no threshold, by design. Quick or Deep
   (thinking on). The open note's embedded images ride with every turn, at most 6
   (`copilot.context.MAX_NOTE_IMAGES`), those beside the pack's passages first: what an image
   says is not searchable. Per-note chat history and pasted images live outside the vault.
   **(owner)** There is no vault-wide Q&A lane, and no profile injection (it pandered under two
   wordings, measured 2026-09-01).
   - **Skills** are `/commands`: prompt templates in `slim/skills/` and the vault's `Skills/`
     (same name wins; `copilot/skills.py`). A skill reads WHOLE notes chosen by code
     (`copilot.context.gather`: the open note, the selection, the folder filtered by frontmatter
     and cut to one section, `[[links]]`), inside a fixed budget, and names what it left out.
     Code knows what it read, so a skill turn makes no sources call.
   - **Answer or edit** — one chat, no mode. A question first meets the intent gate
     (`copilot/turn.py`, `_intent`): one small JSON call on its own short prompt (SLIM's last
     answer and the question, never the note) asks whether the owner wants the note changed.
     No: the Ask path above. Yes, or it can't tell: the model edits the open note through tools
     (`copilot.edits.TOOLS`: `replace`, `edit_text`, `insert`, `move`, `plan`) that point at
     block handles — [b12] a block, s7 the section under heading b7, b11-b15 a range — at
     temperature 0.0. A small change is one call; a larger one is a plan carried out one call
     per step, and nothing after the steps (fix-ups for the plan's outline were cut: every time
     they fired they wasted calls or undid work). Code holds the plan and the limits
     (`copilot/run.py`: 8 steps, 30 calls, 3 refusals in a row, 12k output tokens) and refuses
     what a call must not do, with a reason the model reads back: among them, a later step
     rewriting what an earlier step wrote, and new HTML blocks, which Obsidian renders no
     Markdown inside (a new `<details>` is written as a folded callout instead,
     `copilot.edits.fold`). A change that reworks only the run's own text is `part_of` the
     change it grew from, so the owner never sees SLIM's drafts as removals. What the owner
     reads (steps, labels, summary) names text, never a handle (`copilot.blocks.Note.plain`,
     `copilot.blocks.quote`). `llm.hold()` keeps the model lock for the run. An `edit` skill
     runs the same loop (tidy: the selection's blocks only); a `new-note` skill only
     `create_note`s, in existing folders, never `Journal/`.
     **The server never writes the note** (2026-10-12): the note is open in the owner's editor, and
     writing the file behind it would fight their typing and the editor's undo.
     It works on the editor's text (`doc` + its hash) and streams each change as UTF-16 splices;
     the plugin applies them through the editor as they land (`live/editor.js`,
     `makeLiveEditor`). While a call writes an insert or a replace, its text shows where it will
     go as a draft, never the note's text (`copilot.run._Drafts`), and the change that lands
     takes its place; the chat counts each call's seconds. The plugin skips a change whose text
     the owner typed into meanwhile, draws the marks, then syncs and embeds the note. ↺ undoes a
     change and those built on it; Undo all and Redo; Cmd-Z is the editor's own. The note as it
     was before the run is kept beside the chat, outside the vault
     (`copilot.threads.save_before`), so See before and Restore work after the note is closed;
     Restore warns when the note has changed since, and is refused when a recording's own parts
     have (`/api/copilot/restore`). A recording's frontmatter, `slim-meeting` block and
     transcript are [locked] blocks no call may touch, and no change's splices may reach them.
     New text put above a plain note's properties goes right after them.

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

**Subject is the only folder axis SLIM uses:** a path encodes one fact; type, course and status
are frontmatter, where they can be queried. Folders SLIM invents are at most two segments
(`Capture/<subject>`; `tests/capture/test_voicetags.py` checks every memo route), so folders
don't multiply across every axis; depth the owner creates is theirs, and the card files into it.

**(owner)** **`Journal/` is the floor, and the floor is a path.** Permanent, flat, a top-level
sibling of `Capture/`; no route ever carries a note out of it (test-enforced). If a cloud
provider ever lands, the one check is "is the path under `Journal/`". Known gap: `reflect`
selects journals by `type = 'journal'` with no path filter, so a `type: journal` note placed
outside `Journal/` by hand IS read by the nightly synthesis.

The code, `slim/`: subpackages are feature folders. Imports run one way:
`vault <- index <- copilot`, `vault <- capture <- memos`, `capture <- recorder` (which also uses
`index`); `server` joins `recorder` and `copilot`; the seams (config, db, trace, llm) import no package.

```
slim/            entry points and seams: cli, server (`slim chat`'s HTTP), config, db, trace,
                 llm, llm_render, reflect, devserver; vaultpath (stdlib only, loaded BY PATH)
slim/vault/      reading and writing notes: frontmatter, notefile, note_images, identity (the pin)
slim/capture/    audio to labelled text, both lanes: audio, transcribe, speakers, summarize,
                 enrich, voicetags
slim/memos/      the phone lane: voicememos (the sweep), inbox (files a memo)
slim/recorder/   the desktop lane: routes, pipeline, review, state, note (the note's sole writer),
                 suggest (the filing card)
slim/index/      ingest, chunk, embed, search
slim/copilot/    Talk: turn (the intent gate is `_intent`), context, run (the editing run), edits,
                 blocks (the note as handled blocks), threads, images, skills; the built-in
                 skill templates are `slim/skills/`
```

The plugin, `obsidian/src/`, has the same shape: `recorder/`, `chat/` and `live/` feature folders;
the entry, `plugin.js`, `server.js`, `settings-tab.js` and `sse.js` at the top. The live editor
(drafts, change marks, ↺) is `live/editor.js` (`makeLiveEditor`). Tests mirror both trees.

## What was cut (owner)

Four cuts (2026-07-31, 08-27, 09-01, 09-03) removed roughly 60k lines of code nobody used:
curated memory, context packs, an evidence rail, MCP tools and server, policy tiers, a
vault-wide ask stack, the desktop app and its chat lane, a tool loop for search (the editing
loop in Talk is the owner's call of 2026-10-12), re-filing and re-labelling passes, and the
schema columns and prose that described them. **Restoring any of these is the owner's call:
stop and ask.** The maintainer's archive holds them. Model evaluations run from harnesses kept
outside the repo.

## Traps that will bite you, and what they cost when they did

**A terminal-started server can serve days-old code.** The plugin reuses any `slim chat` on
7546; `uv run` imports from disk at spawn and never reloads. Plugin-started servers exit with
Obsidian's PID. `GET /api/health` reports staleness and the SessionStart hook calls it — but
a server whose `uv` parent was adopted by PID 1 reports fresh and still fails; when in doubt,
kill the port. The vault holds a COPY of `main.js`: the same trap. And `main.js` is built from
`obsidian/src/`: an edit there without `npm run build` is not in it (`bundle.test.mjs` catches
that). Obsidian's loader cannot resolve a relative `require` (it broke the plugin on 2026-08-25
while node's tests stayed green), which is why src/ is bundled.

**Memory is the binding constraint, and macOS kills silently.** On a 48 GB machine
`qwen3.6:35b` (MoE) takes ~25 GB at `RESIDENT_CTX = 131072`, ~16 MB per 1k tokens. Ollama
spawns one runner per (model, `num_ctx`), so every call uses the one size — the recording
path's three calls (summary, title, card) MUST share it, and a SMALLER bespoke window is the
same cliff from the other side (a second ~23 GB runner; measured 2026-08-26). A long pass must
stream its results. The big window trades the loud `ctx_saturated` guard for silent weak
recall of a prompt's middle.

**One model job at a time, across every session on this Mac.** On 2026-09-29 parallel jobs
loading models (Ollama plus MLX benches) drove the machine to ~70 GB and it had to be
hard-reset: GPU-wired memory can't swap, so macOS hangs instead of killing. Before a model job
(regress, an eval, a live server run), check `ps` and Ollama's `server.log` for another one; a
client at a different `num_ctx` makes Ollama reload the model on every switch.

**Ollama traps — all handled in `slim/llm.py`; never call Ollama directly.** `qwen3.6` thinks
by default and thinking tokens count against `num_predict`, so an unset `think` returns
`content=""`; every call sets it explicitly (`think=True` only for the recorder's summary and
the copilot's Deep mode). Greedy decoding into a JSON grammar loops forever on an unbounded
string: cap `num_predict`, put `maxLength` on short fields (Ollama cannot compile one above
~1500). LaTeX does not survive JSON — `\b \f \n \r \t` collide with `\beta \frac \nabla \rho
\text` — so the summarizer bans backslashes and `capture.summarize._repair` reverses the decode.
`llm.py` serializes live calls; temperature 0.0 for structured work, 0.7 for prose a person
reads. /api/chat parses a tool call itself and holds it back until it is whole, tools= or not,
so a call with tools goes RAW to /api/generate, its prompt built by `llm_render.render` byte for
byte as Ollama 0.35's `qwen3.5` renderer builds it (`tests/test_llm_render.py` holds Ollama's
own rendering; after an Ollama upgrade, check it still matches) and its reply read by
`llm_render._Reader`. A model with another renderer needs its own. A tool call's text is XML,
taken as written: the JSON trap above never touches it, and unlike Ollama's parser `_Reader`
keeps a text parameter written `null` as the word. Even at temperature 0, a run can differ
between two identical calls (seen 2026-10-14 on unchanged code; likely Ollama's prompt cache): compare regress results case by case, not
by a single run.

**A note can be filed and still be unreachable.** Filing is a markdown file; the copilot and
`reflect` read SQLite. Anything that writes a note must ingest AND embed, or say plainly
that it did not (2,802 fragments once sat without vectors, 2026-08-04). A machine transcript has
no blank lines; `index.chunk._split_prose` splits at sentences as a last resort so a 37-minute
meeting is not one fragment.

**A prompt rule alone is not a fix.** On this model a banned verb finds another verb and a
rider often has no measurable effect (the Restraint Rule suppressed key points, 2026-08-05; the
profile pandered under two wordings, 09-01). Back a prompt rule with a mechanism, as the
summarizer's backslash ban is backed by `_repair`, and measure it. A green test gate is
verification of what you thought to check.

**Obsidian's plugin sees the server's file moves as deletions.** An external rename is
`delete` + `create`, never `rename`, and the leaf lets go of the deleted file. Anything that
reacts to a vault `delete` must first ask whether the server caused it (2026-09-04/09).

## Rules, with their reasons

- **Deterministic truth in code** (hashes, ids, timestamps, filters, paths); LLMs only for
  interpretation, classification, synthesis, language. A model guessing a fact the machine
  already knows is a bug that varies run to run.
- **Raw sources are never rewritten.** Derived state is rebuildable — but a rebuild mints new
  source ids and orphans saved copilot chats, so schema changes are migrations.
- **Derived artifacts are excluded from ingestion** — `_Reflections/`, `Profile/` and
  `Skills/` (`config.EXCLUDED_TOP`); the recorder's summary is fenced and blanked before
  chunking — so the brain never retrieves its own opinions as evidence.
- **Failures are traced.** `trace.record` writes machine facts (never prose, never
  chain-of-thought) for every recorder stage, copilot turn, reflect run and refusal; quiet
  sweeps write nothing. `slim trace -n 5` is how a session finds out what a silent failure did.
- **The index is PINNED to the vault it was built from** (`SLIM_DATA_DIR/vault_pin`,
  `vault/identity.py`): ingest's removal sweep deletes every source it cannot find under the
  current vault, and discovery can change that path with nobody asking. Every command that can
  write to the vault checks the pin first. A real move, or an index made before the pin, is
  `slim ingest --vault-moved`.
- **`slim/vaultpath.py` stays where it is, stdlib-only and 3.9-safe:** the backup script, the
  privacy hook and `scripts/pdf2md.py` load it BY PATH with the system python (3.9).
- **Full Disk Access goes to `restic` only, never `/bin/bash` or an interpreter:** that would
  hand the whole disk to anything that interpreter runs.
- **A public repo: nothing user-specific in code, comments or commit messages.** Vocabulary and
  the owner's name are config; `config/projects.yaml` is gitignored.
- **Question a rule when it blocks something useful** — say what it protected against and
  whether that risk is real here — rather than citing it as settled.

## The owner's rulings

These are value calls, not technical constraints; they change only on the owner's word.

- **The ONLY privacy rule is `Journal/`** (2026-08-31). Everything else in the vault the owner
  sends to cloud models themselves, routinely. Do not cite privacy to gate, block or flag a
  change, and do not raise it as a concern.
- **Journal transcripts are PRIVATE — do not read them** (2026-08-01). The local model reads
  them; for a journal an agent's verification is metadata only. Every OTHER transcript is
  ordinary working material: read it, quote it, measure on it.
- **Autonomous writes are fine** (2026-07-31): enrichment, reflection and the recorder write
  without confirmation; provenance stays in frontmatter.
- **ONE generative model, `qwen3.6:35b` (`llm.MODEL`), for everything** (2026-09-01) — chosen
  against the summarization experiment's evidence (best at grounding, worst at summarization;
  do not call it "the model the bake-off chose"). Embedder `embeddinggemma`: its task prefixes
  are load-bearing (`index/embed.py`); changing it means deleting the brain DB and re-ingesting.
- Also marked **(owner)** above: one click with no settings, the card's filing roots, no
  vault-wide Q&A and no profile injection, the `Journal/` floor, and the cut list.

## Environment

- Apple Silicon Mac with Ollama; 48 GB of memory fits the default model. macOS only: MLX
  (Parakeet) and the Core Audio tap need it.
- Vault: `SLIM_VAULT`, else DISCOVERED (`vaultpath.py`): Obsidian's own registry, then
  iCloud's Obsidian container, then `~/Documents/Obsidian Vault` (iCloud-synced is fine). Brain
  DB outside any synced folder: `SLIM_DATA_DIR`, default `~/Library/Application Support/slim/`
  (SQLite + FTS5 + vectors, rebuildable — see above).
- launchd: `com.slim.voicememos` every 5 min, `com.slim.reflect` nightly, `com.slim.backup`
  daily (`scripts/backup.sh`, vault → restic).

## Conventions

- `uv` for everything (the system python is 3.9); pytest.
- Plain, explicit flows: no indirection (manager, factory, engine, wrapper) without a second
  caller that needs it. Folders and files are not indirection: group code by feature, one job
  per file.
- **Prefer deleting a leash to adding a belt.** When code holds the fact, let code state it
  once. Deleting is free — git holds it forever. Keeping costs this file's context, test
  runtime, and the attention of every agent that reads the repo.
- Judge a feature by "would the owner miss it if it were gone," not "is it well-built."
- Documentation files, each with one job: this file (what, and why), `BACKLOG.md` (open items
  only, one line each, finished items deleted), the README (for outsiders). A reason goes beside
  its rule here, not in another file.
- Tests check behaviour, not spelling: a test that reads source text only for something no
  behaviour test can reach (the bundle's requires, one helper for one rule).
- The maintainer stays on master. Work in a worktree on a branch, commit there, and they
  fast-forward on their word. Use `git -C <path>` for every git command. Commit messages are
  concise, not STE.
