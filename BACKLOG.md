# Backlog

**Open items only, one line each.** A finished item is DELETED. Git and the decision log (in the vault)
are the history. Past ~20 items is a signal about scope, not a reason for headings.

- [ ] Verify the recorder end to end on the cut code (record → summary → card → file → copilot on the note).
- [ ] Notes autosave fires only on blur; quitting Obsidian with the caret in the field loses the last edit. Flush on view close.
- [ ] Send pasted images to the summarizer (`qwen3.6:35b` is vision-capable; a screenshotted formula reaches it as a filename). Needs a cap, a resize, and a measured RSS run.
- [ ] `filed_by: slim` is never cleared when they accept a card unedited — "SLIM guessed" and "they agreed" look the same.
- [ ] Copilot slash-commands: a sidebar invocation that runs a deterministic tool or skill on the open note.
- [ ] Copilot markdown edits: small reads/writes on the open note (the persona still says "You cannot edit files yet").
- [ ] A `[[wikilink]]` in a rendered copilot answer has no click handler — verify, then delegate clicks on `a.internal-link`.
- [ ] Move `Export-*/` (376 transcripts) out of the working tree — gitignored and never committed, but it does not belong beside the code.
- [ ] Cloud providers — ratified 2026-09-01 as DEFERRED guidance; `Journal/` stays local.
- [ ] `lecture` vs `learning` is an unstable distinction; collapse when the vocabulary is next touched.
- [ ] Stale vault path: `scripts/pdf2md.py:465` (and its usage line 8) defaults `--vault` to `~/Documents/Obsidian Vault`; the vault is now the iCloud container. Route through `vaultpath.discover_vault()`.
- [ ] `scripts/backup.sh:5` header comment still names `~/Documents/Obsidian Vault`.
- [ ] `slim status` and `slim trace` never call `require_vault_identity()`, so the one command you'd run to notice a vault/pin mismatch cannot report one.
- [ ] `slim/llm.py:44` says "Tests pin this" about the explicit `think` parameter, but no test decodes the HTTP body — deleting `"think": think` (`llm.py:118`, `:190`) passes the whole suite and breaks the recorder. Add a wire-level test.
- [ ] Speaker labels are decided per segment: a resumed segment with one speaker is appended unlabelled under a labelled transcript and reads as the last speaker continuing. Decide at the recording level (render one-speaker stereo segments with their single label when any segment is labelled).
- [ ] Tell remote speakers apart (Speaker 1, Speaker 2): a voice-clustering model on the system channel, replacing `speakers.label_tokens`. Test with a panel podcast played as system audio.
- [ ] In-room meetings on one microphone: the same model on the mic channel; "which cluster is me" needs an enrollment or a rename on the card. Test with a podcast played out of the speakers into the mic.
