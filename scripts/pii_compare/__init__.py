"""Can a model replace the PII detector's context rules? A measured comparison, synthetic data only.

Arms: A rules (the shipped `pii.mask`), B shapes (A with every cue-binding entry point patched
off), C shapes + the local model on each standing number, D shapes + a GLiNER PII tagger, E rules +
the model on the numbers A left alone. Sets: `hard` (330 reviewer cases, hand-labelled in
`data/hard.json` by `label_hard.py`), `realistic`, `calls` (30 transcribed synthetic calls) and
`ordinary` (`gen_ordinary.py`). Run: `uv run python -m scripts.pii_compare <arm> <set> --out DIR`;
model arms are model jobs. The tagger runs in its own virtualenv through `tagger.py`.
"""
