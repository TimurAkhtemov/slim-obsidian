"""The calls mode: speak a script with `say`, run the pipeline's steps, keep results by cache key."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

from .gen import Script
from .sandbox import REPO
from .score import aggregate_calls, check_full, locate, score_call


VOICE = {"Me": "Samantha", "Them": "Daniel"}
LEAD_SECONDS = 0.5


def _bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
        if Path(prefix, name).exists():
            return str(Path(prefix, name))
    raise SystemExit(f"{name} not found")


def _seconds(path: Path) -> float:
    out = subprocess.run([_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def audio_key(script: Script, seed: int) -> str:
    """What the synthesized audio depends on: the lines, the voices, the seed (gaps)."""
    blob = json.dumps([script.lines, VOICE, seed, LEAD_SECONDS, "synth-1"])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def synthesize(script: Script, dest: Path, seed: int = 7) -> dict:
    """Speak the script into a stereo 48 kHz opus `.webm`: Me (Samantha) LEFT, Them (Daniel)
    RIGHT, one line after another with a 0.3-0.6 s gap, never overlapping. Returns each line's
    time span. Line audio is cached beside `dest`."""
    parts = dest.parent / f"{script.sid}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"{seed}:{script.sid}:gaps")
    args, graph, lines, t = [], [], [], LEAD_SECONDS
    for k, (speaker, text) in enumerate(script.lines):
        txt, aiff = parts / f"{k:02d}.txt", parts / f"{k:02d}.aiff"
        if not aiff.exists() or not txt.exists() or txt.read_text() != text:
            txt.write_text(text)
            subprocess.run(["say", "-v", VOICE[speaker], "-o", str(aiff), "-f", str(txt)], check=True)
        dur = _seconds(aiff)
        gap = round(rng.uniform(0.3, 0.6), 3)
        lines.append({"speaker": speaker, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur + gap
        args += ["-i", str(aiff)]
        pan = "c0=c0|c1=0*c0" if speaker == "Me" else "c0=0*c0|c1=c0"
        delay = f",adelay={int(LEAD_SECONDS * 1000)}:all=1" if k == 0 else ""
        graph.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=mono,"
                     f"pan=stereo|{pan}{delay},apad=pad_dur={gap}[a{k}]")
    graph.append("".join(f"[a{k}]" for k in range(len(lines))) + f"concat=n={len(lines)}:v=0:a=1[out]")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([_bin("ffmpeg"), "-y", "-nostdin", "-loglevel", "error", *args,
                    "-filter_complex", ";".join(graph), "-map", "[out]",
                    "-c:a", "libopus", "-b:a", "96k", "-ar", "48000", str(dest)], check=True)
    return {"lines": lines, "seconds": round(t, 3)}


def rms_dbfs(path: Path, a: float, b: float) -> float:
    """Loudest channel's RMS over [a, b] seconds, in dBFS. Raises when no audio could be read
    there: an unreadable span must fail a silence check, not pass it."""
    import numpy as np

    out = subprocess.run([_bin("ffmpeg"), "-nostdin", "-loglevel", "error", "-ss", f"{a:.3f}",
                          "-t", f"{max(b - a, 0.02):.3f}", "-i", str(path), "-f", "f32le", "-ac", "2",
                          "-ar", "48000", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(out[:len(out) - len(out) % 8], dtype="<f4").reshape(-1, 2)
    if not len(x):
        raise ValueError(f"no samples in {a:.2f}-{b:.2f} s")
    rms = float(np.sqrt((x.astype(np.float64) ** 2).mean(axis=0)).max())
    return 20 * float(np.log10(max(rms, 1e-10)))


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detector_sha() -> str:
    """Results depend on the detector: a change to it rescores (but does not re-speak or
    re-transcribe) every script."""
    return _sha_file(REPO / "slim" / "capture" / "pii.py")[:16]


def run_calls(scripts: list[Script], out: Path, *, synth, transcribe, mask_segments, silence,
              measure, classes, model: bool = True, seed: int = 7, detector: str = "",
              full=None, full_n: int = 0, redo: bool = False, log=print) -> dict:
    """Every script: speak it, transcribe, mask, mute, re-transcribe, score.

    Caches, each keyed by what it depends on: audio by `audio_key` (lines, voices, seed); the
    first transcription by the audio's sha256; the result by seed, classes, model, detector and
    audio key. A result whose keys differ is recomputed, reusing the audio and the first
    transcription; `redo` drops all three."""
    out = Path(out)
    for sub in ("audio", "work", "results"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    results = []
    for n, s in enumerate(scripts):
        key = audio_key(s, seed)
        meta = {"seed": seed, "classes": list(classes), "model": model, "detector": detector,
                "audio_key": key, "full": bool(full is not None and n < full_n)}
        res_path = out / "results" / f"{s.sid}.json"
        audio_path = out / "audio" / f"{s.sid}.webm"
        manifest_path = audio_path.with_suffix(".json")
        if redo:
            for old in [res_path, audio_path, manifest_path, *(out / "work").glob(f"{s.sid}.*")]:
                old.unlink(missing_ok=True)
            shutil.rmtree(out / "audio" / f"{s.sid}.parts", ignore_errors=True)
        if res_path.exists():
            prior = json.loads(res_path.read_text())
            if prior.get("meta") == meta:
                results.append(prior)
                continue
            log(f"  {s.sid}: cached result is for {_diff(prior.get('meta') or {}, meta)}; rescoring")
        t0 = time.perf_counter()
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if manifest.get("key") != key or not audio_path.exists():
            manifest = {**synth(s, audio_path), "key": key}
            manifest_path.write_text(json.dumps(manifest))
        audio_sha = _sha_file(audio_path)
        work = out / "work" / f"{s.sid}.webm"
        if not work.exists() or _sha_file(work) != audio_sha:
            shutil.copyfile(audio_path, work)
        first = out / "work" / f"{s.sid}.first.json"
        cached = json.loads(first.read_text()) if first.exists() else {}
        if cached.get("audio_sha") == audio_sha:
            got = cached["got"]
        else:
            got = transcribe(work)
            first.write_text(json.dumps({"audio_sha": audio_sha, "got": got}))
        raw = str(got.get("text") or "")
        masked_gots, counts, reasons, seg_spans = mask_segments([dict(got)])
        masked = str(masked_gots[0].get("text") or "")
        spans = [tuple(x) for x in seg_spans[0]]
        silenced = out / "work" / f"{s.sid}.silenced.webm"
        shutil.copyfile(work, silenced)
        silence(silenced, spans)
        heard_after = str(transcribe(silenced).get("text") or "")
        res = score_call(s, raw, masked, heard_after, classes, spans=spans, reasons=reasons,
                         manifest=manifest, measure=measure, silenced=silenced, model=model)
        res["meta"] = meta
        res["counts"] = counts
        res["masked"] = masked
        if meta["full"]:
            transcribed = {i.token for i in s.items if locate(i.token, raw)}
            res["full"] = full(s, audio_path, work, transcribed)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        (out / "results" / f"{s.sid}.raw.md").write_text(raw)
        (out / "results" / f"{s.sid}.masked.md").write_text(masked)
        (out / "results" / f"{s.sid}.silenced.md").write_text(heard_after)
        res_path.write_text(json.dumps(res, indent=1))
        log(f"  {s.sid}  {res['seconds']:.1f}s  items={len(s.items)} spans={len(spans)} "
            f"audio_fail={len(res['audio_failures'])}")
        results.append(res)
    return aggregate_calls(results, classes, model=model)


def _diff(old: dict, new: dict) -> str:
    keys = [k for k in new if old.get(k) != new[k]]
    return "other " + ", ".join(keys) if keys else "another run"


def _full_runner(config, *, pipeline=None, measure=rms_dbfs):
    """`handle_record` on a staged copy, then `check_full`, including the archived audio's
    level inside the spans the pipeline itself muted (seen through `_silence_segments`)."""
    import sqlite3

    def run(script: Script, audio_path: Path, work: Path, transcribed: set[str]) -> dict:
        pl = pipeline
        if pl is None:
            from slim.recorder import pipeline as pl
        rel = f"Attachments/_incoming/eval{script.sid}-001.webm"
        staged = config.VAULT / rel
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(audio_path, staged)
        cache = work.with_suffix(work.suffix + ".transcript.json")
        if cache.exists():               # same bytes, so the pipeline reuses this transcript
            shutil.copyfile(cache, staged.with_suffix(staged.suffix + ".transcript.json"))
        muted: list = []
        real = pl._silence_segments

        def spy(segments, seg_spans):
            muted.append([[tuple(x) for x in sp] for sp in seg_spans])
            return real(segments, seg_spans)

        pl._silence_segments = spy
        try:
            result = pl.handle_record(audio_rel=rel, notes_md="", declared_type="meeting",
                                      title="", vault=config.VAULT)
        except Exception as exc:                       # noqa: BLE001 - scored as a failure
            return {"script": script.sid, "ok": False, "title": None, "note": None,
                    "problems": [f"handle_record raised {type(exc).__name__}"]}
        finally:
            pl._silence_segments = real
        note_text = (config.VAULT / result["note"]).read_text(encoding="utf-8")
        con = sqlite3.connect(config.DB_PATH)
        try:
            fragments = [r[0] for r in con.execute(
                "SELECT f.text FROM fragments f JOIN sources s ON s.id = f.source_id "
                "WHERE s.path = ? AND s.deleted = 0", (result["note"],))]
        finally:
            con.close()
        archived = config.VAULT / result["audio"]
        levels = []
        for a, b in (muted[-1][0] if muted and muted[-1] else []):
            try:
                levels.append(([a, b], measure(archived, a, b)))
            except Exception:                          # noqa: BLE001 - unreadable fails the check
                levels.append(([a, b], None))
        return check_full(script, note_text, result, fragments, transcribed,
                          config.PII_MASK, archived_levels=levels)
    return run
