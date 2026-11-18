"""Measure PII masking on synthetic calls and public synthetic sets, in a throwaway vault.

    uv run python -m scripts.pii_eval text   --out DIR [--no-model] [--n 30] [--seed 7]
    uv run python -m scripts.pii_eval calls  --out DIR [--full 5] [--only s03] [--redo]
    uv run python -m scripts.pii_eval public --out DIR [--rows 2000]

Nothing here touches the owner's data. Every value is generated (valid-looking, never real) or
comes from a public synthetic dataset whose licence allows evaluation. `sandbox()` points
SLIM_VAULT, SLIM_DATA_DIR and SLIM_PROJECTS into DIR BEFORE anything from `slim` is imported
(config resolves its paths at import) and refuses to go on unless they resolved there.

Importing this module has no side effects: the generator and scorers are pure, and each mode's
heavy steps (say, Parakeet, the model, ffmpeg) arrive as callables so tests can fake them.
`text` and `calls` with the model are model jobs: one at a time on this Mac (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .calls import _full_runner, detector_sha, rms_dbfs, run_calls, synthesize
from .gen import CLASSES, generate
from .public import PUBLIC_CLASSES, http_json, print_public_report, run_public
from .sandbox import sandbox
from .score import print_calls_report, print_text_report, run_text


def _write(out: Path, mode: str, report: dict) -> None:
    (out / f"report-{mode}.json").write_text(json.dumps(report, indent=1, default=str))


def _classes(arg: str | None):
    if arg is None:
        return None
    if arg == "all":
        return CLASSES
    got = tuple(c.strip() for c in arg.split(",") if c.strip())
    bad = [c for c in got if c not in CLASSES]
    if bad:
        raise SystemExit(f"unknown classes: {bad}")
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("text", "calls", "public"))
    ap.add_argument("--out", required=True, help="a new or pii_eval-made directory, outside any "
                    "git tree or vault")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--only", action="append", help="script id (repeatable)")
    ap.add_argument("--no-model", action="store_true", help="patterns only: no password pass")
    ap.add_argument("--classes", help="comma list or 'all' (default: the shipped default)")
    ap.add_argument("--full", type=int, default=5, help="calls: run handle_record on the first N")
    ap.add_argument("--redo", action="store_true", help="calls: rescore scripts already scored")
    ap.add_argument("--rows", type=int, default=2000, help="public: English rows per dataset")
    a = ap.parse_args(argv)

    out, config = sandbox(a.out, classes=_classes(a.classes))
    from slim.capture import pii

    classes = tuple(config.PII_MASK)
    if a.mode == "public":
        report = run_public(out, rows=a.rows, fetch=http_json,
                            find=lambda t: pii.find(t, PUBLIC_CLASSES))
        print_public_report(report)
        _write(out, "public", report)
        return 0

    scripts = generate(a.n, a.seed)
    if a.only:
        scripts = [s for s in scripts if s.sid in set(a.only)]
    (out / "scripts.json").write_text(json.dumps([asdict(s) for s in scripts], indent=1))
    model = not a.no_model
    if model:
        print("model job: check nothing else is loading a model on this Mac (ps, ollama ps, "
              "the llama-server log) before this runs")
    if a.mode == "text":
        find = pii.ask_model if model else None
        report = run_text(scripts, lambda t: pii.mask(t, classes, find_password=find), classes,
                          model=model)
        print_text_report(report)
    else:
        from slim.capture import audio, transcribe as tmod
        from slim.recorder import pipeline

        report = run_calls(
            scripts, out,
            synth=lambda s, dest: synthesize(s, dest, a.seed),
            transcribe=lambda p: pipeline._transcribe_segment(Path(p), tmod)[0],
            mask_segments=lambda gots: pipeline._mask_segments(gots, model=model),
            silence=audio.silence, measure=rms_dbfs, classes=classes, model=model,
            seed=a.seed, detector=detector_sha(),
            full=_full_runner(config) if a.full and model else None, full_n=a.full,
            redo=a.redo)
        print_calls_report(report)
    _write(out, a.mode, report)
    return 0 if report["gate"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
