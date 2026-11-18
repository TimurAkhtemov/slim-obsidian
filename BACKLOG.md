# Backlog

**Open items only, one line each.** A finished item is DELETED. Git and the decision log (in the vault)
are the history. Past ~20 items is a signal about scope, not a reason for headings.

- [ ] Live-test copilot skills and editing in Obsidian: `/consolidate-my-notes` in a meetings folder, `/brief`, `/actions folder`, `/tidy` on a selection, typing during a run, ↺ and Restore, a transcript edit refused.
- [ ] Verify the recorder in Obsidian on the cut code: live capture (mic + Core Audio tap), the card UI, Resume, the silent-capture alarm (both sides zeros for 5 s). The server half passed through the API on 2026-09-22 (stereo Me/Them, summary, apply-move, append, retry, silent refusal, journal, copilot).
- [ ] Cloud providers — ratified 2026-09-01 as DEFERRED guidance; `Journal/` stays local.
- [ ] Tell remote speakers apart (Speaker 1, Speaker 2): a voice-clustering model on the system channel, replacing `speakers.label_tokens`. Test with a panel podcast played as system audio.
- [ ] In-room meetings on one microphone: the same model on the mic channel; "which cluster is me" needs an enrollment or a rename on the card. Test with a podcast played out of the speakers into the mic.
- [ ] The recorder summary resolves a bare image name with no note path, so an ambiguous one (`paste-001.png` ×5 in the vault) can be another note's; pass the note's path to `note_images.load`.
- [ ] 11 orphaned transcript caches in `Attachments/_incoming` (`slim status`); two hold words no note has (2026-08-27, 2026-09-10). Owner decides: file or delete.
- [ ] "Record this?" when a call starts, as Notion does: poll Core Audio for another process running mic input (`kAudioProcessPropertyIsRunningInput` + `BundleID`, macOS 14.4+; probed OK 2026-09-29) and offer one click. No model, no listening; needs a tiny native helper and must ignore SLIM's own capture.
- [ ] `slim mask --audio`: re-transcribe old recordings for word times and silence their secret spans; today `slim mask` is text only, so `pii_partial: [audio]` stays on a note.
- [ ] The plugin's meeting-note parser takes the FIRST `## Transcript` heading and the server the LAST; make them agree.
- [ ] `note.replace_review_sections` (the review card's apply) rebuilds a recording's head and drops owner text between sections of a legacy or hand-edited note; splice like `slim mask`.
- [ ] `slim mask <path>` typed in another Unicode normalisation form than the disk's reports FileNotFoundError (fails safe; `--all` still reaches the note).
- [ ] Masking leaves alone the titles and file names of recordings filed before it, and live captions (on screen only, never saved).
- [ ] Owner to decide after living with it: keep or switch off the cue-free spaced 3-2-4 SSN shape (`pii_detect.SSN_SPACED_CUE_FREE`).
- [ ] `slim mask` does not reach saved chats or the copilot's pre-edit copies under the data dir; mask or purge them.
- [ ] A listed answer given in another order than asked binds by position: "zip and account number?" ⏎ "48213377 and 10001" masks the zip and leaves the account.
- [ ] A cue noun followed by balance/type/holder is read as the cue: "What's your account balance?" ⏎ "1234567." is masked as an account.
- [ ] A card's 3-digit security code is not a masked class.
- [ ] Masking's context rules (`pii_detect.py`, cue binding) keep missing rarer phrasings, most of all in answers to a question that asks for two things. Measure them against the local model and a PII tagger on the synthetic calls and the reviews' hard cases; if a model matches them, delete the binding rules and keep the shapes.
