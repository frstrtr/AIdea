"""Extract real (topic, entropy, themes) pairs from transcripts.jsonl.

Walks the file, groups events by run_id, emits one paired example per
completed run where BOTH a request_started (topic) and a themes
(themes list) event are present. Output is OmniLang's paired-data JSONL
shape:

    {"baseline": {"prompt": ..., "completion": ...},
     "typed":    {"prompt": ..., "completion": ...}}

The two views encode the same underlying (topic → themes) mapping but
the typed view uses the schema tokens from schema.yaml so the ablation
harness can compare typed-token finetuning vs. plain-text finetuning.

Usage::

    python -m distill.extract --in /opt/aidea/transcripts.jsonl \\
                              --out distill/real_pairs.jsonl

Idempotent — overwrite the output safely.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


# Default theme-entropy bucket when a run's transcript doesn't carry one.
# Most pre-anti-fatigue runs used 0.5, which maps to 'wild'.
_DEFAULT_ENTROPY_BUCKET = "wild"

# Mapping from numeric theme_entropy [0,1] to the 5 schema buckets.
def _bucket_for_entropy(value: float | str | None) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("sane", "wild", "insane", "crazy", "mad"):
            return v
        # Some transcripts stored entropy as a numeric string.
        try:
            value = float(v)
        except ValueError:
            return _DEFAULT_ENTROPY_BUCKET
    if not isinstance(value, (int, float)):
        return _DEFAULT_ENTROPY_BUCKET
    v = float(value)
    if v < 0.20:
        return "sane"
    if v < 0.50:
        return "wild"
    if v < 0.75:
        return "insane"
    if v < 0.92:
        return "crazy"
    return "mad"


def _format_baseline(topic: str, bucket: str, themes: list[str]) -> dict:
    """Plain-text view — newline-separated themes, entropy as a bare word."""
    prompt = (
        f"Topic: {topic.strip()}\n"
        f"Theme entropy: {bucket}\n"
        f"Themes:\n"
    )
    completion = "\n".join(t.strip() for t in themes if t and t.strip())
    return {"prompt": prompt, "completion": completion}


def _format_typed(topic: str, bucket: str, themes: list[str]) -> dict:
    """Typed view — entropy is a schema token, themes are <T:>-separated."""
    prompt = f"<E:{bucket}> Topic: {topic.strip()} Themes:"
    parts = [t.strip() for t in themes if t and t.strip()]
    completion = " " + " ".join(f"<T:> {t}" for t in parts)
    return {"prompt": prompt, "completion": completion}


def _group_events_by_run(path: Path) -> dict[str, dict]:
    """Pass 1 over transcripts.jsonl. Returns {run_id: {topic, theme_entropy,
    themes, source, ts}}. Skips runs missing topic or themes."""
    by_run: dict[str, dict] = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = ev.get("run_id")
            if not rid:
                continue
            kind = ev.get("kind")
            r = by_run[rid]
            if kind == "request_started":
                r["topic"] = (ev.get("topic") or "").strip()
                r["theme_entropy"] = ev.get("theme_entropy")
                r["entropy"] = ev.get("entropy")
                r["source"] = ev.get("source", "")
                r["ts"] = ev.get("ts", 0)
            elif kind == "themes":
                # Some runs have ev['themes'] as a list, others (rare) as a
                # comma-joined string.
                t = ev.get("themes")
                if isinstance(t, list):
                    r["themes"] = [str(x).strip() for x in t if str(x).strip()]
                elif isinstance(t, str):
                    r["themes"] = [x.strip() for x in t.split(",") if x.strip()]
    return dict(by_run)


def extract_pairs(
    transcripts: Path,
    out_path: Path,
    *,
    seed: int = 0,
) -> int:
    """Read transcripts, write paired JSONL, return the count written."""
    by_run = _group_events_by_run(transcripts)
    # Pick out runs that have BOTH a topic and a themes list.
    candidates: list[tuple[str, dict]] = []
    for rid, r in by_run.items():
        if not r.get("topic"):
            continue
        themes = r.get("themes")
        if not themes:
            continue
        candidates.append((rid, r))
    # Sort newest-first so the held-out tail (taken from the END of the
    # file by a downstream split script) doesn't bias toward old runs.
    candidates.sort(key=lambda kv: kv[1].get("ts", 0))

    rng = random.Random(seed)
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rid, r in candidates:
            topic = r["topic"]
            themes = r["themes"]
            # Some real runs didn't log theme_entropy — fall back to the
            # synth-entropy if present, else default 'wild'.
            te = r.get("theme_entropy")
            if te is None:
                te = r.get("entropy")
            bucket = _bucket_for_entropy(te)
            record = {
                "source": "real",
                "run_id": rid,
                "topic": topic,
                "entropy_bucket": bucket,
                "themes": themes,
                "baseline": _format_baseline(topic, bucket, themes),
                "typed": _format_typed(topic, bucket, themes),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--in", dest="src",
        default="transcripts.jsonl",
        help="Path to transcripts.jsonl (default: ./transcripts.jsonl)",
    )
    ap.add_argument(
        "--out", dest="dst",
        default="distill/real_pairs.jsonl",
        help="Output JSONL (default: distill/real_pairs.jsonl)",
    )
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"Transcripts not found: {src}", file=sys.stderr)
        return 1
    n = extract_pairs(src, Path(args.dst))
    print(f"Wrote {n} pairs to {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
