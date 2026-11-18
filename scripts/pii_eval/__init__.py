"""Measure PII masking on synthetic calls and public synthetic sets, in a throwaway vault.

    uv run python -m scripts.pii_eval text|calls|public --out DIR ...

One job per module: `gen` (scripts), `score` (matching and scorers), `calls`, `public`,
`sandbox`, and the CLI in `__main__`. Importing any of them imports nothing from `slim`.
"""
