# SLIM — Sparse Local Intelligence & Memory

A private local assistant that lives in an Obsidian vault. The owner speaks; it transcribes,
labels, files, indexes and reflects, and a copilot talks with them about their notes. Everything
runs on one Mac. **The owner's own words: "I record and forget."** SLIM is the pipeline that
makes spoken audio filable and findable.

Each rule here carries its reason. A rule whose reason no longer holds is deleted, not obeyed.
Rules marked **(owner)** change only on the owner's word. The queue is `BACKLOG.md`; the README
is for outsiders.

## Commands

```bash
uv run pytest -q                        # ~1000 tests; `uv` always — the system python is 3.9
node --test "obsidian/test/**/*.test.mjs"   # plugin tests; they load obsidian/src/ directly
(cd obsidian && npm install && npm run build)   # src/ -> the committed main.js; bundle.test.mjs fails if stale
uv run slim chat --reload               # dev server; respawns on any slim/**/*.py save (not .yaml)
uv run slim ingest                      # index the vault, embedding included
uv run slim status                      # counts, including `unembedded`
uv run slim trace -n 5                  # what the last silent failures did
cp obsidian/{main.js,manifest.json,styles.css} \
   "$SLIM_VAULT/.obsidian/plugins/slim-recorder/"   # a COPY; reload the plugin after
lsof -tiTCP:7546 -sTCP:LISTEN | xargs kill   # free the port; the plugin respawns the server on demand
```

## The pipeline

```
audio  ->  transcript  ->  LABEL  ->  FILE  ->  INDEX  ->  (REFLECT, TALK)
```

1. **Capture** — the Obsidian plugin (`obsidian/src/`, bundled into the committed `main.js`).
   The plugin owns the UI and writes files through the Vault API; every judgement (transcribe,
   summarize, suggest a path) is the Python server's, where the tests are.
   - The meeting is a `slim-meeting` fenced block in a standard `MarkdownView`: a custom view
     was built and rejected.
   - The file is stereo on purpose: mic left, system audio right, so `capture/speakers.py`
     labels words Me/Them by channel, with no model. Words already on disk are never relabelled.
   - Audio lands in `Attachments/_incoming` in 5 s chunks, so a crash costs seconds. A `.webm`
     left there is a recording that never finished filing.
   - **The note is filed before the card shows; nothing waits for approval**: confirmations
     were abandoned and the recordings lost.
   - **A failed or silent transcript is refused, not filed** (empty transcripts lost three
     recordings). The audio stays staged and the card offers Try again.
   - `recorder/note.py` is the sole writer of a recording's frontmatter and managed sections.
2. **Label** — type and subject, written once at capture; `tagged_by`/`subject_by` record
   which layer decided. A wrong label is the owner's edit.
3. **File** — `suggest` proposes `Notes/<subject>/`; an invented folder collapses to its
   deepest existing ancestor, else `Notes/_unfiled`. A journal goes flat to `Journal/`
   (test-enforced). The card offers folders under `Notes/` only (`suggest.FILING_ROOTS`): a
   separate `Capture/` tree put every subject in two places and gave lookups two trees.
4. **Reflect** — `slim reflect` writes one synthesis note into `_Reflections/` nightly. This is
   the part the owner reads: judge changes by whether that note is worth reading.
5. **Talk** — the copilot sidebar, served by `slim chat` on 127.0.0.1:7546.

After any change to notes, `slim ingest`. Ingest detects renames by content hash, so a hand
edit plus a hand move between two ingests reads as delete + create and the note loses its id.

## Talk: the decisions

The copilot is a vault assistant shaped like the Pi coding agent, on the same model and server.

- **(owner) It answers from the whole vault**: the earlier scoped stack over-cited and was too
  restrictive. An assistant, not a citation engine; no prompt asks for citations, and sources
  are what code saw the turn read.
- **One loop, no intent gate** (`turn._loop`): every call gets the lookups, the edit tools and
  `create_note`, and the model decides. A gate measured worse: it said "no record" on 5 of 7
  questions answered elsewhere in the vault. The loop is slower; the owner accepted that.
- **The conversation is the transcript, as Pi keeps a session**: replaying only the answer
  text made follow-ups skip lookups and invent facts. Replays leave out thinking and the
  attached notes (`turn.without_attachment`); the window drops the oldest whole turns.
- **One system message and tool list for every turn**, the open note riding with the question,
  so llama-server reuses the cached prefix.
- **Chats stand on their own** (`copilot/threads.py`, outside the vault): switching notes never
  ends a chat.
- **Edits are tool calls on block handles** (`copilot/edits.py`, `copilot/run.py`, which holds
  the limits). **The server never writes the open note**: it streams splices the plugin applies
  through the editor, so it never fights the owner's typing or undo. A recording's frontmatter,
  `slim-meeting` block and transcript are locked blocks.
- **A claimed edit is checked by code** (`run.claims_edit`): the model sometimes says "I've
  replaced…" with nothing changed, and a prompt rule alone did not stop it.
- Lookups see only what the index sees, plus the open note and `@` pins, so SLIM never finds
  its own reflections as evidence.

## Where things live

```
Journal/            flat · never routes out
Notes/<subject>/    recordings and the owner's writing; `_unfiled` when no subject is known;
                    depth below level 2 is the owner's
Profile/            the owner's self-description; not indexed
Attachments/        binaries; hidden from Obsidian's file explorer by the plugin
_Reflections/       derived, excluded from ingest
Skills/             the owner's copilot /commands; not indexed
```

Subjects are the ids in `config/projects.yaml` (gitignored; copy the example). Subject is the
only folder axis: type, course and status are frontmatter, where they can be queried.

Imports run one way: `vault <- index <- copilot`, `vault <- capture <- recorder` (which also
uses `index`); `server` joins `recorder` and `copilot`; the seams (config, db, trace, llm)
import no package. The plugin's `obsidian/src/` mirrors this with `recorder/`, `chat/` and
`live/`; tests mirror both trees.

## Traps

- **A stale server or plugin.** The plugin reuses any `slim chat` already on 7546, and
  `uv run` never reloads, so a terminal-started server can serve days-old code. The vault holds
  a COPY of `main.js`, and `main.js` is only what `npm run build` made of `src/`. When in doubt,
  kill the port, rebuild and copy. Obsidian's loader cannot resolve a relative `require`,
  which is why `src/` is bundled.
- **One model job at a time, across every session on this Mac.** Parallel model loads once
  drove the 48 GB machine to a hard reset: GPU-wired memory can't swap. Before any model job,
  check `ps`, `ollama ps` and llama-server's log. Never load the generative weights in Ollama
  beside llama-server.
- **Never call the model server directly; go through `slim/llm.py`.** It handles: explicit
  `enable_thinking` on every call (unset, thinking eats the cap and returns ""); capped
  `num_predict` and `maxLength` under a JSON grammar (greedy decoding loops forever);
  LaTeX backslashes, which JSON mangles (`capture.summarize._repair`); tool calls sent raw to
  `/completion` and read by `llm_render._Reader`. An overlong prompt is refused (HTTP 400),
  never cut. Runs differ even at temperature 0: compare evals case by case.
- **A filed note can be unreachable.** The copilot and `reflect` read SQLite, so anything that
  writes a note must ingest and embed, or say plainly that it did not.
- **A prompt rule alone is not a fix.** On this model a banned verb finds another verb. Back
  a prompt rule with a mechanism and measure it.
- **Obsidian sees the server's file moves as deletions** (create + unlink, never rename), and
  the leaf lets go of the file. Anything reacting to a vault `delete` must first ask whether
  the server caused it (`filingFrom`, `isEmptiedLeaf`).

## Rules

- **Deterministic truth in code** (hashes, ids, timestamps, paths); LLMs only for
  interpretation and language. A model guessing a fact the machine knows varies run to run.
- **SLIM never rewrites a transcript on its own**; a change the owner asks for (masking, an
  edit) is fine. Derived state is rebuildable, but a rebuild mints new source ids and strands
  saved chats, so schema changes are migrations.
- **Derived artifacts are excluded from ingest** (`config.EXCLUDED_TOP`; the recorder's summary
  is blanked before chunking), so SLIM never retrieves its own opinions as evidence.
- **Failures are traced** with `trace.record`: machine facts, never prose or chain-of-thought.
- **The index is pinned to its vault** (`vault/identity.py`): ingest's sweep deletes every
  source it cannot find, and vault discovery can change with nobody asking. A real move is
  `slim ingest --vault-moved`.
- **`slim/vaultpath.py` stays stdlib-only and 3.9-safe**: `scripts/backup.sh` and
  `scripts/pdf2md.py` load it by path with the system python.
- **Full Disk Access goes to `restic` only**, never a shell or interpreter, which would hand the
  whole disk to anything it runs.
- **Public repo**: nothing user-specific in code, comments or commits; names and vocabulary
  are config.
- **(owner) One generative model, `qwen3.6:35b` (`llm.MODEL`)**, served by llama-server;
  Ollama keeps the embedder `embeddinggemma`, whose task prefixes are load-bearing. Changing
  the embedder means deleting the brain DB and re-ingesting.
- **(owner) Autonomous writes are fine**: enrichment, reflection and the recorder write without
  confirmation; provenance stays in frontmatter.
- **(owner) No privacy gates**, `Journal/` included; don't cite privacy to block a change. A
  privacy feature the owner asks for (PII masking) is their call.
- Question a rule when it blocks something useful: say what it protected against and whether
  that risk is real here.

## Environment

Apple Silicon Mac, 48 GB, macOS only (MLX Parakeet and the Core Audio tap). llama-server runs
the generative model; Ollama runs the embedder. Vault: `SLIM_VAULT`, else discovered by
`vaultpath.py`. Brain DB: `SLIM_DATA_DIR`, default `~/Library/Application Support/slim/`.
launchd: `com.slim.reflect` nightly, `com.slim.backup` daily (`scripts/backup.sh`, restic).

## Conventions

- `uv` for everything; pytest.
- Plain, explicit flows: no manager, factory or wrapper without a second caller. Group code by
  feature, one job per file.
- Prefer deleting a leash to adding a belt; git keeps what is deleted.
- Judge a feature by "would the owner miss it if it were gone".
- Tests check behaviour, not spelling.
- Docs each have one job: this file (what and why), `BACKLOG.md` (open items, one line each),
  the README (outsiders).
- The maintainer stays on master. Work in a worktree on a branch, commit there; they
  fast-forward on their word. Use `git -C <path>`. Commit messages are concise.
