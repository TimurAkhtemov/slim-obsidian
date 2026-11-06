# SLIM — Sparse Local Intelligence & Memory

A private local assistant that lives in an Obsidian vault. You speak; it transcribes,
organizes, reflects, and can talk with you about any of it. Everything runs on one Mac —
local models over llama.cpp and Ollama, a SQLite index, and Markdown files you can read without it.

Built to replace a paid Notion subscription whose only load-bearing feature was AI meeting
notes, and whose real failure was simpler: *I click record, it sits somewhere, and I can
never find it again.* So SLIM is not a recorder with a model bolted on. It is the pipeline
that makes spoken audio **filable and findable**, and every design decision below follows
from that.

```
audio  ->  transcript  ->  LABEL  ->  FILE  ->  INDEX  ->  (REFLECT, TALK)
```

---

## Contents

- [How it fits together](#how-it-fits-together)
- [Your vault stays yours](#your-vault-stays-yours)
- [How a recording is filed](#how-a-recording-is-filed)
- [What happens when you record](#what-happens-when-you-record)
- [How answers work](#how-answers-work)
- [Where state lives](#where-state-lives)
- [Requirements](#requirements)
- [Getting started](#getting-started)
- [Code map](#code-map)
- [The rules that shaped it](#the-rules-that-shaped-it)
- [Tests](#tests)

---

## How it fits together

Everything you touch is either Obsidian or the terminal. The Obsidian plugin talks to a
local HTTP server on `127.0.0.1:7546`; the CLI and the scheduled jobs import the same Python
modules directly, with no server in between. The pipeline ends in exactly two places: Markdown
you own, and a SQLite index that can be deleted and rebuilt.

```mermaid
flowchart TB
    subgraph S["Surfaces"]
        direction LR
        REC["Obsidian recorder<br/>one click"]
        COP["Copilot sidebar<br/>the open note"]
        CLI["slim CLI<br/>ingest / reflect / status"]
    end

    SRV["slim chat — 127.0.0.1:7546<br/>22 endpoints, loopback only, Host + Origin gated"]

    subgraph P["Pipeline"]
        direction LR
        CAP["CAPTURE<br/>transcribe.py"] --> LAB["LABEL<br/>enrich"]
        LAB --> FIL["FILE<br/>suggest / recorder/note.py"]
        FIL --> IDX["INDEX<br/>ingest / chunk / embed"]
        IDX --> ANS["ANSWER<br/>search / copilot / reflect"]
    end

    VAULT["The vault<br/>your Markdown"]
    BRAIN["The brain<br/>SQLite + FTS5 + vectors"]

    REC --> SRV
    COP --> SRV
    SRV --> CAP
    SRV --> ANS
    CLI -- "no HTTP" --> IDX
    FIL --> VAULT
    IDX --> BRAIN
    BRAIN -. "retrieval reads" .-> ANS
```

That the CLI bypasses HTTP is not an accident, and it has a consequence worth knowing early:
a `slim chat` server started in a terminal keeps running the code it was launched with, so it
can serve days-old behaviour while your CLI is perfectly current. `GET /api/health` reports
staleness; when in doubt, kill the port.

## Your vault stays yours

**SLIM does not ask you to restructure anything.** Indexing walks every Markdown file in the
vault — `rglob("*.md")` — and skips only four top-level names (`Attachments/`, `Profile/`,
`Skills/`, `_Reflections/`), a few dot-folders, and the recorder's own drafts. There is no allow-list of
folders anywhere. A vault organised any way at all is searchable and copilot-ready with no
setup at all.

The folder names SLIM uses are an **output, not a prerequisite**. They appear only when it
writes something, and it creates them itself with their parents:

| Folder | When it appears | What it is |
|---|---|---|
| `Journal/` | first journal entry | the private floor — permanent, flat, and no route ever carries a note out of it |
| `Capture/<subject>/` | first filed recording | every recording, organised by subject and nothing else |
| `Notes/<subject>/` | you make it, if you want it | curated writing; any depth below level 2 is yours |
| `Attachments/` | first recording | archived audio, named by content hash |
| `_Reflections/` | first nightly run | derived synthesis, excluded from indexing |

Subjects are yours — they are the ids you define in `config/projects.yaml`.

**One real friction, named rather than hidden:** the filing card proposes destinations under
`Capture/` and `Notes/` only. It will not offer you folders you already use. A new recording
therefore lands in `Capture/<subject>/`, and you drag it wherever you like afterwards —
because identity is the content hash, not the path, a move keeps the note's history and its
saved copilot chats.

Two degradations, both graceful. Without registered subjects, the copilot's scope ladder
stops at the note's own folder instead of widening further. And the nightly reflection reads
journals under `Journal/` only, so a note with `type: journal` moved anywhere else drops out of it.

## How a recording is filed

```mermaid
flowchart LR
    B1["one click in Obsidian"] --> B2["batch transcript<br/>silence is REFUSED, not filed"]
    B2 --> B3["summarize() + enrich() title"]
    B3 --> B4["suggest.build_card()<br/>asks the model for a folder"]
    B4 --> B5["invented folder collapses to the<br/>deepest real ancestor, else _unfiled"]
```

The recorder asks the model for a filing card, and then code constrains the answer to folders
that actually exist. The model proposes and code disposes, which is why a hallucinated path
cannot create a folder.

## What happens when you record

The single most load-bearing decision in the product: **the note is filed before the card is
shown.** A confirmation is an action, and actions get abandoned — so nothing waits for
approval. By the time you see a filing card, the note is written, moved and indexed. The card
only adjusts a note that is already on disk and findable.

```mermaid
sequenceDiagram
    participant O as Obsidian plugin
    participant S as slim chat
    participant D as Disk

    O->>D: draft note exists from click one
    O->>D: audio appended in 5s chunks to Attachments/_incoming
    O->>S: PCM every ~2s — live captions (never saved)
    Note over O,S: Stop — one synchronous POST /api/record
    S->>S: batch transcription (120s windows, 15s overlap)
    S->>S: summary (thinking on), title, filing card
    S->>D: archive audio by hash, write the note, delete the draft
    S->>D: index + embed
    S-->>O: card appears, on an already-filed note
```

Live captions and the authoritative transcript come from two separate speech-model instances.
The captions are for your eyes while you speak and are discarded; the note always gets the
batch transcript, because **the transcript is the note** — summary, title and card only
decorate it. A failed or silent transcript is refused rather than filed, and your audio stays
staged so you can retry.

The three model calls in that sequence share one server and its one context window: the
window is fixed when the server starts, so a call cannot ask for a second, smaller one.

## How answers work

Identity is the content hash; the path is an attribute. That is what lets you reorganise
freely. It is also why an edit *and* a move between two indexing runs read as a delete plus a
create — the bytes changed and the old path vanished at the same moment, so nothing links them.
Index before and after a move, and a rename is free.

```mermaid
flowchart LR
    Q["your question"] --> N["1 · sync the open note<br/>index + embed on demand"]
    N --> W["2 · widen by PATH<br/>note → folder → course → subject"]
    W --> H["3 · hybrid retrieval<br/>BM25 + cosine, fused by reciprocal rank<br/>vector weighted 4:1, per-source cap"]
    H --> G["4 · answer<br/>notes ride in the SYSTEM message, titles only"]
    G --> C["5 · sources<br/>a SECOND call, after the answer"]
```

Two deliberate details. The notes pack carries **titles and no numbers**, so the prose is
structurally unable to fabricate a `[1]`-style citation; attribution is a separate bounded
call that is shown the finished answer and asked which notes it drew on. And scope is always
a *path*, never a similarity score — the copilot is a note- and folder-scoped assistant, not a
vault-wide question answering engine. "When did I first…" is a SQL question about
`authored_at`, and answering it with a similarity score would be wrong.

Quick mode disables model thinking; Deep enables it with a larger budget. The copilot has no
profile in its prompt, edits only the open note (a skill may also create a new one), with every
change applied in the editor and undoable, and its chat history lives outside the vault and is
never treated as evidence.

Separately, `slim reflect` runs nightly and writes **one** synthesis note into `_Reflections/`
from your recent journals. It is the part of the system that is actually read for pleasure,
and it is create-only — it never overwrites an existing reflection.

## Where state lives

| Where | What | If you deleted it |
|---|---|---|
| the vault | Every note, its frontmatter, its full transcript, archived audio, the reflections. Plain Markdown, readable with SLIM uninstalled. | Everything is gone — this is the only copy that matters. |
| `slim.db` | `sources`, `fragments`, `fragments_fts`, `embeddings`. In Application Support, never in a synced folder. | `slim ingest` rebuilds it — but new source ids orphan saved chats, so schema changes are migrations. |
| outside both | Copilot threads, pasted images, JSONL traces, and the vault pin that stops ingest sweeping the wrong vault. | Chat history and the audit trail are lost; your notes are untouched. |

Raw sources are never rewritten, and derived artifacts are never re-ingested — `_Reflections/`
and `Profile/` are excluded, and the recorder's own summary is blanked before chunking. The
index never retrieves its own opinions as evidence.

**Filed is not the same as findable.** Filing writes a Markdown file; the copilot and the
reflection read SQLite. Anything that writes a note must index *and* embed it, or say plainly
that it did not.

## Requirements

macOS on Apple Silicon. Transcription runs on MLX and the recorder takes system audio from
Core Audio's process tap, so there is no Intel or Linux path. The generative model is the
binding constraint: `qwen3.6:35b` holds roughly 32 GB resident with the two 262k slots below,
which a 48 GB machine fits and a 16 GB machine does not.

- **[uv](https://docs.astral.sh/uv/)** — every command is `uv run`, which builds the Python
  3.12+ environment on first use. There is no separate install step.
- **ffmpeg** — `brew install ffmpeg`. `ffprobe` reads the recorder's container and `ffmpeg`
  decodes it; a recording dies without them.
- **[Ollama](https://ollama.com)**, running, with both models pulled. Ollama runs the embedder;
  the generative model's weights are its `qwen3.6:35b` files, served by llama.cpp (next item):

  ```bash
  ollama pull qwen3.6:35b      # the weights llama-server loads: the model and its projector
  ollama pull embeddinggemma   # the embedder; its task prefixes are load-bearing
  ```

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**'s `llama-server` on `127.0.0.1:8080`
  (`brew install llama.cpp`), for every generative call: summary, title, card, reflect, copilot.
  It renders the prompt with the model's own chat template, which Ollama does not. Point it at
  the two GGUF files Ollama keeps for `qwen3.6:35b` (the `model` and `projector` layers of its
  manifest, under `~/.ollama/models/blobs/`), and give each slot at least `llm.RESIDENT_CTX` tokens:

  ```bash
  llama-server -m <model.gguf> --mmproj <projector.gguf> --jinja -ngl 999 \
    -np 2 -c 524288 --spec-type draft-mtp --sleep-idle-seconds 600 --port 8080
  ```

  `-np 2 -c 524288` is two slots of 262,144 tokens, so a second client (a coding agent) keeps
  its own prompt cache; `--spec-type draft-mtp` uses the model's own draft head, about twice the
  decode speed.

- **Obsidian** 1.5 or newer, desktop. It is the only interactive surface.

The speech model's weights (`mlx-community/parakeet-tdt-0.6b-v3`) download themselves on the
first transcription, so the first one is slow.

## Getting started

**1. Open a vault in Obsidian**, or create one. SLIM installs into the vault it discovers, and
discovery falls back to `~/Documents/Obsidian Vault` whether or not anything is there. An
existing vault of any shape is fine — see [Your vault stays yours](#your-vault-stays-yours).

**2. Name your subjects.** Copy the example and edit it: the subject ids are your folder names
under `Notes/` and `Capture/`, which the copilot widens across, and `owner` is how the prompts
address you.

```bash
cp config/projects.example.yaml config/projects.yaml
```

This file is gitignored and optional — without it SLIM still runs, but the copilot's scope stops
at the note's own folder. Filing does not read it: the card proposes folders that exist, else
`Capture/_unfiled`.

**3. Install the plugin** into that same vault:

```bash
VAULT="${SLIM_VAULT:-$(python3 slim/vaultpath.py)}"
[ -d "$VAULT" ] || { echo "no vault at $VAULT — open one in Obsidian, or set SLIM_VAULT"; exit 1; }
PLUGIN="$VAULT/.obsidian/plugins/slim-recorder"
mkdir -p "$PLUGIN"
cp obsidian/main.js obsidian/manifest.json obsidian/styles.css "$PLUGIN/"
```

Then enable **SLIM** in Community Plugins.

**4. Index what is already there**, and check SLIM resolved the vault you meant:

```bash
uv run slim ingest    # index the vault; embeds at the end
uv run slim status    # prints the vault, the counts, and anything still unembedded
```

If it picked the wrong vault: `export SLIM_VAULT=/path/to/vault`.

**5. Use it.** The microphone icon records; the message icon is the open-note copilot. The
plugin starts `slim chat` when it needs to. A reused server does not hot-reload, so restart a
stale one before testing changes:

```bash
lsof -tiTCP:7546 -sTCP:LISTEN | xargs kill
```

The full command set:

```bash
uv run slim ingest                        # index the vault; embeds at the end
uv run slim status                        # index health, incl. fragments still unembedded
uv run slim reflect                       # journal window -> ONE reflection note
uv run slim chat                          # the Obsidian plugin's server, 127.0.0.1:7546
uv run slim trace -n 1                    # replay the last traced operation
```

Optional background jobs, each installed by its own script in `scripts/`: the reflection
nightly, and a restic backup of the vault daily.

## Code map

- `slim/` — the entry points and seams: `cli.py` (five commands), `server.py` (the plugin's API
  server, run as `slim chat`), `reflect.py`, `db.py`, `llm.py` (never call the model server directly) and
  `llm_render.py` (the raw tool-call reply format), `trace.py`, `config.py`,
  `vaultpath.py`. Each subpackage below is a feature folder, not a layer.
- `slim/vault/` — `frontmatter.py` (the frontmatter reader and writer), `notefile.py` (slugs,
  safe writes), `note_images.py` (a note's embedded images), `identity.py` (the vault pin).
- `slim/capture/` — audio to labelled text: `transcribe.py` (two speech lanes),
  `audio.py` (ffmpeg, hashing, archive), `speakers.py`, `summarize.py`, `enrich.py` and
  `voicetags.py` (labelling).
- `slim/recorder/` — the recorder: `routes.py` (the server's record routes), `pipeline.py`
  (audio to a filed note), `review.py` (edits to a filed note), `state.py` (the job queue,
  progress, locks), `note.py` (the sole writer of a recorded note), `suggest.py` (the filing
  card).
- `slim/index/` — `ingest.py`, `chunk.py` (fragments), `embed.py`, `search.py`.
- `slim/copilot/` — `turn.py` (one turn: the gate and the Ask path), `context.py` (what it
  reads), `run.py` (the editing run), `edits.py` (the tools) and `blocks.py` (the note as
  blocks with handles); `threads.py`, `images.py` (chat attachments), `skills.py`; the built-in
  skill templates are `slim/skills/`.
- `obsidian/src/` — the plugin: `recorder/`, `chat/` and `live/` feature folders, with the entry,
  `plugin.js`, `server.js`, `settings-tab.js` and `sse.js` at the top; one job per file, bundled by
  esbuild (`npm run build`) into the committed `obsidian/main.js` that Obsidian loads. `obsidian/test/`
  mirrors it.
- `CLAUDE.md` — the operating brief, including the traps that cost real debugging time.
- `BACKLOG.md` — what is next. Open items only.

## The rules that shaped it

- **Deterministic truth in code.** Hashes, ids, timestamps, filters and paths are code. The
  model is for interpretation, classification, synthesis and language — never for a fact the
  machine already knows.
- **`Journal/` is a floor, and the floor is a path.** No route carries a note out of it, and a
  test enforces that.
- **Subject is the only folder axis.** A path encodes one fact; type, course and status live in
  frontmatter where they can be queried. Folders SLIM invents are at most two segments deep;
  depth you create is yours.
- **The note is filed before the card is shown.** Nothing waits for approval.
- **Autonomous writes are fine, and they are traced.** Provenance stays in frontmatter;
  `trace.record` logs machine facts — never prose, never chain-of-thought — for every stage,
  turn and refusal.
- **Prefer deleting a leash to adding a belt.** Roughly 60k lines have been cut from this
  project on purpose. Judge a feature by "would the owner miss it", not "is it well-built".

## Tests

Both suites run without a model server, a vault, or any of the model weights — ffmpeg is the one
binary the recorder tests genuinely need, because they exercise real container handling:

```bash
brew install ffmpeg               # ffprobe is a real dependency of the recorder tests
uv run pytest -q                  # the Python side
node --test "obsidian/test/**/*.test.mjs"   # the plugin (esbuild builds main.js: npm install, npm run build)
```

That property is deliberate — it is what keeps the system inspectable, and it is what CI runs
on every push. The plugin suite needs Node and, for the test that the committed `main.js` is
what `obsidian/src/` builds to, esbuild (`npm ci --prefix obsidian`).

## License

MIT. See [LICENSE](LICENSE).
