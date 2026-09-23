# Backlog

**Open items only, one line each.** A finished item is DELETED. Git and the decision log (in the vault)
are the history. Past ~20 items is a signal about scope, not a reason for headings.

- [ ] Live-test copilot skills and Edit mode: `/consolidate-my-notes` in a meetings folder, a stale Accept, a transcript edit refused.
- [ ] Verify the recorder end to end on the cut code (record → summary → card → file → copilot on the note).
- [ ] Cloud providers — ratified 2026-09-01 as DEFERRED guidance; `Journal/` stays local.
- [ ] Tell remote speakers apart (Speaker 1, Speaker 2): a voice-clustering model on the system channel, replacing `speakers.label_tokens`. Test with a panel podcast played as system audio.
- [ ] In-room meetings on one microphone: the same model on the mic channel; "which cluster is me" needs an enrollment or a rename on the card. Test with a podcast played out of the speakers into the mic.
- [ ] `chunk.parse_frontmatter` reads a block list (`tags:` then `  - item` lines) as `''`, so those tags/topics are invisible to everything that reads the index; 9 notes in `Capture/` and `Notes/` are written that way. Parse block lists; `set_frontmatter` also writes flow lists unquoted, so a `#tag` item would become a YAML comment.
- [ ] The recorder summary resolves a bare image name with no note path, so an ambiguous one (`paste-001.png` ×5 in the vault) can be another note's; pass the note's path to `copilot_images.note_images`.
