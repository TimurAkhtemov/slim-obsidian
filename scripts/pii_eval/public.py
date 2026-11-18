"""The public mode: licence-checked public synthetic PII sets, scored against `pii.find`."""

from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path

from .gen import _aba_ok
from .score import _covered


PERMISSIVE = {"apache-2.0", "mit", "cc-by-4.0", "cc0-1.0"}
DATASETS = (
    ("gretelai/synthetic_pii_finance_multilingual", "test"),
    ("gretelai/gretel-pii-masking-en-v1", "test"),
    ("ai4privacy/pii-masking-300k", "validation"),
)
LABELS = {
    "ssn": "ssn", "credit_card_number": "card", "iban": "account", "bban": "account",
    "bank_routing_number": "account", "account_number": "account", "email": "email",
    "street_address": "address", "address": "address", "phone_number": "phone",
}
PUBLIC_CLASSES = ("ssn", "card", "account", "email", "address", "phone")


def _finance(row: dict):
    if row.get("language") != "English":
        return None
    text = row["generated_text"]
    gold = [(int(s["start"]), int(s["end"]), LABELS.get(s["label"]), s["label"])
            for s in json.loads(row.get("pii_spans") or "[]")]
    # a span past the end of its text is the dataset's error, not a value we could find
    return text, [g for g in gold if 0 <= g[0] < g[1] <= len(text)]


def _en_v1(row: dict):
    text = row["text"]
    gold = []
    for e in ast.literal_eval(row.get("entities") or "[]"):
        value, label = str(e.get("entity") or ""), (e.get("types") or [""])[0]
        if not value:
            continue
        for m in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", text):
            gold.append((m.start(), m.end(), LABELS.get(label), label))
    gold.sort()
    return text, gold


ADAPTERS = {
    "gretelai/synthetic_pii_finance_multilingual": _finance,
    "gretelai/gretel-pii-masking-en-v1": _en_v1,
}


def _invalid(cls: str, label: str, value: str) -> bool:
    """The gold value fails our own validator: rejecting it is right, not a miss."""
    from slim.capture import pii

    d = re.sub(r"\D", "", value)
    if cls == "ssn":
        return len(d) != 9 or not pii._ssn_ok(d)
    if cls == "card":
        return not 13 <= len(d) <= 19 or not pii._luhn(d)
    if label == "iban":
        s = re.sub(r"\s", "", value)
        return not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{8,30}", s) or not pii._iban_ok(s)
    if label == "bank_routing_number":
        return len(d) != 9 or not _aba_ok(d)
    if cls == "email":
        return "@" not in value            # a template slot such as "[Shipper's Email]"
    if cls == "phone":
        return len(d) < 7
    return False


def score_public(docs, find) -> dict:
    """Recall (every letter and digit of a gold value under some hit) and precision (a hit of class
    C overlapping a gold span mapped to C), per class; misses split into invalid gold and true."""
    stats = {c: {"gold": 0, "found": 0, "found_valid": 0, "touched": 0, "invalid": 0,
                 "true_miss": 0, "hits": 0,
                 "tp_hits": 0, "misses": [], "false_positives": []} for c in PUBLIC_CLASSES}
    for text, gold in docs:
        hits = find(text)
        spans = [(h.start, h.end) for h in hits]
        for a, b, cls, label in gold:
            if cls not in stats:
                continue
            st = stats[cls]
            st["gold"] += 1
            touched = any(x < b and a < y for x, y in spans)
            st["touched"] += touched
            bad = _invalid(cls, label, text[a:b])
            st["invalid"] += bad
            if touched and _covered(a, b, text, spans):
                st["found"] += 1
                st["found_valid"] += not bad
                continue
            st["true_miss"] += not bad
            if not bad and len(st["misses"]) < 10:
                st["misses"].append({"label": label, "value": text[a:b],
                                     "context": text[max(0, a - 40):b + 40].replace("\n", " ")})
        for h in hits:
            if h.cls not in stats:
                continue
            st = stats[h.cls]
            st["hits"] += 1
            over = [g for g in gold if g[0] < h.end and h.start < g[1]]
            if any(g[2] == h.cls for g in over):
                st["tp_hits"] += 1
            elif len(st["false_positives"]) < 10:
                st["false_positives"].append({"text": text[h.start:h.end],
                                              "gold_label": over[0][3] if over else None})
    for st in stats.values():
        st["recall"] = round(st["found"] / st["gold"], 3) if st["gold"] else None
        valid = st["gold"] - st["invalid"]
        st["recall_valid"] = round(st["found_valid"] / valid, 3) if valid else None
        st["precision"] = round(st["tp_hits"] / st["hits"], 3) if st["hits"] else None
    return {"by_class": stats}


def licence(dataset: str, fetch) -> tuple[str, bool]:
    info = fetch(f"https://huggingface.co/api/datasets/{dataset}")
    lic = (info.get("cardData") or {}).get("license")
    if not lic:
        tags = [t.split(":", 1)[1] for t in info.get("tags") or [] if t.startswith("license:")]
        lic = tags[0] if len(tags) == 1 else tags or None
    names = lic if isinstance(lic, list) else [lic] if lic else []
    ok = bool(names) and all(str(n).lower() in PERMISSIVE for n in names)
    return (",".join(map(str, names)) or "unknown"), ok


_LAST_REQUEST = [0.0]


def http_json(url: str):
    """GET JSON with a generic User-Agent, one request a second, backing off on 429 and 5xx
    (Retry-After when given)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "pii-eval/1"})
    for attempt in range(6):
        time.sleep(max(0.0, 1.0 - (time.monotonic() - _LAST_REQUEST[0])))
        _LAST_REQUEST[0] = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as exc:                       # noqa: BLE001 - retried, then raised
            code = getattr(exc, "code", None)
            if attempt == 5 or (code is not None and code < 500 and code != 429):
                raise
            after = (getattr(exc, "headers", None) or {}).get("Retry-After") if code else None
            wait = float(after) if after and str(after).isdigit() else min(60.0, 5.0 * 2 ** attempt)
            print(f"  HTTP {code or type(exc).__name__}: retrying in {wait:.0f}s")
            time.sleep(wait)


def _fetch_failed(report, entry, exc, where: str, log) -> None:
    code = getattr(exc, "code", None)
    if code is not None:
        entry["error"] = f"HTTP {code} at {where}"
        log(f"HTTP {code} at {where}: stopped this dataset (cached pages are kept; rerun resumes)")
    else:
        report["network"] = False
        entry["error"] = f"network unavailable ({type(exc).__name__}) at {where}"
        log(entry["error"])


def run_public(out: Path, *, rows: int, fetch, find, log=print) -> dict:
    """Each candidate dataset whose licence is permissive: up to `rows` English rows through
    `find`. Pages are cached under `<out>/public/`."""
    report = {"mode": "public", "network": True, "rows": rows, "datasets": {}}
    for dataset, split in DATASETS:
        entry = report["datasets"].setdefault(dataset, {"used": False})
        try:
            lic, ok = licence(dataset, fetch)
        except OSError as exc:
            _fetch_failed(report, entry, exc, f"{dataset} licence", log)
            if not report["network"]:
                log("network unavailable: public mode skipped")
                return report
            continue
        entry["licence"] = lic
        if not ok:
            entry["reason"] = f"licence {lic!r} is not one of {sorted(PERMISSIVE)}"
            log(f"skip {dataset}: {entry['reason']}")
            continue
        adapter = ADAPTERS.get(dataset)
        if adapter is None:
            entry["reason"] = "no adapter for its format"
            log(f"skip {dataset}: {entry['reason']}")
            continue
        cache = out / "public" / dataset.replace("/", "__")
        cache.mkdir(parents=True, exist_ok=True)
        docs, offset = [], 0
        while len(docs) < rows:
            page_path = cache / f"{split}-{offset}.json"
            if page_path.exists():
                page = json.loads(page_path.read_text())
            else:
                url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}"
                       f"&config=default&split={split}&offset={offset}&length=100")
                try:
                    page = fetch(url)
                except OSError as exc:
                    _fetch_failed(report, entry, exc, f"{dataset} offset {offset}", log)
                    break
                if not isinstance(page, dict) or "rows" not in page:
                    err = page.get("error") if isinstance(page, dict) else type(page).__name__
                    entry["error"] = f"no rows at {dataset} offset {offset}: {str(err)[:120]}"
                    log(entry["error"] + " (not cached)")
                    break
                page_path.write_text(json.dumps(page))
            got = page.get("rows") or []
            for r in got:
                doc = adapter(r.get("row") or {})
                if doc and len(docs) < rows:
                    docs.append(doc)
            offset += 100
            total = page.get("num_rows_total")
            if len(got) < 100 or (total is not None and offset >= total):
                break
        entry.update({"used": True, "split": split, "rows": len(docs), **score_public(docs, find)})
        log(f"{dataset} ({lic}, {split}): {len(docs)} English rows")
    return report


def print_public_report(r: dict) -> None:
    print("\n== public (written text, patterns only; no speech)")
    print("recall: every letter and digit of a gold value under a hit of ANY class (an account "
          "masked as [CARD] counts as found); valid-rec: found among values that pass our "
          "validators; invalid: gold that fails them (counted whether found or not); "
          "precision: hits of class C over a gold span mapped to C")
    for name, d in r["datasets"].items():
        if not d.get("used"):
            print(f"{name}: skipped — {d.get('reason', 'not reached')}")
            continue
        print(f"{name}: licence {d['licence']}, {d['rows']} rows")
        if d.get("error"):
            print(f"  incomplete: {d['error']}")
        print(f"  {'class':<9}{'gold':>6}{'found':>7}{'touched':>8}{'recall':>8}{'valid-rec':>10}"
              f"{'invalid':>9}{'true-miss':>10}{'hits':>6}{'precision':>11}")
        for c, st in d["by_class"].items():
            f = lambda v: "-" if v is None else f"{v:.2f}"
            print(f"  {c:<9}{st['gold']:>6}{st['found']:>7}{st['touched']:>8}{f(st['recall']):>8}"
                  f"{f(st['recall_valid']):>10}{st['invalid']:>9}{st['true_miss']:>10}"
                  f"{st['hits']:>6}{f(st['precision']):>11}")
        for c, st in d["by_class"].items():
            for m in st["misses"]:
                print(f"    miss {c} ({m['label']}): {m['value']!r} … {m['context']!r}")
            for fp in st["false_positives"]:
                print(f"    false positive {c}: {fp['text']!r} (gold label: {fp['gold_label']})")
