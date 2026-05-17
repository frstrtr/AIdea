"""Cap each donor-theme to ≤N global appearances across the training set.

Walks the real + synth paired JSONL files, counts theme frequencies
(case-insensitive), and emits a new ``train_pairs.jsonl`` where each
theme appears at most ``--cap`` times globally. Records that end up
with fewer than ``--min-themes`` themes after filtering are dropped
entirely.

Why: the teacher (Claude Max's raw generate_themes) inherits the LLM's
favourite-domain bias — harbor pilotage, monastic rules, mycorrhizal
symbiosis, falconry, lacquer restoration, sourdough — so a student
trained on the raw teacher labels will become MORE biased than today's
pipeline, not less. Capping turns the bias into a finite resource:
the first N records to use a domain keep it; later occurrences drop
it from those records and force the model to learn other donors for
similar topics.

Order matters because the cap is consumed greedily. We RANDOMIZE the
iteration order (seeded) so the cap budget is spread fairly across
real and synthetic records, not biased toward whichever block was
read first.

Usage::

    python -m distill.dedupe \\
        --in distill/real_pairs.jsonl distill/synth_pairs.jsonl \\
        --out distill/train_pairs.jsonl \\
        --cap 3 --min-themes 5 --seed 17
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def _normalize(theme: str) -> str:
    """Case- and whitespace-insensitive key for dedup counting."""
    return " ".join(theme.strip().lower().split())


def _format_baseline(topic: str, bucket: str, themes: list[str]) -> dict:
    prompt = (
        f"Topic: {topic.strip()}\n"
        f"Theme entropy: {bucket}\n"
        f"Themes:\n"
    )
    completion = "\n".join(t.strip() for t in themes if t and t.strip())
    return {"prompt": prompt, "completion": completion}


def _format_typed(topic: str, bucket: str, themes: list[str]) -> dict:
    prompt = f"<E:{bucket}> Topic: {topic.strip()} Themes:"
    parts = [t.strip() for t in themes if t and t.strip()]
    completion = " " + " ".join(f"<T:> {t}" for t in parts)
    return {"prompt": prompt, "completion": completion}


def dedupe(
    inputs: list[Path],
    out_path: Path,
    *,
    cap: int = 3,
    min_themes: int = 5,
    seed: int = 17,
) -> dict:
    """Read inputs, cap themes, write deduped JSONL. Returns a stats dict."""
    rng = random.Random(seed)

    # Load all records preserving order info (so we can re-randomise).
    records: list[dict] = []
    for p in inputs:
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(r)

    rng.shuffle(records)

    # Greedy cap consumption — count as we walk; once a normalized theme
    # has been written `cap` times globally, drop it from later records.
    written = 0
    dropped_records = 0
    dropped_themes = 0
    kept_themes = 0
    counts: Counter[str] = Counter()
    in_themes: list[int] = []
    kept_per_record: list[int] = []
    by_bucket_kept: Counter[str] = Counter()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for r in records:
            topic = r.get("topic", "")
            bucket = r.get("entropy_bucket", "wild")
            themes_in = r.get("themes", []) or []
            in_themes.append(len(themes_in))

            kept_for_record: list[str] = []
            for t in themes_in:
                key = _normalize(t)
                if not key:
                    continue
                if counts[key] >= cap:
                    dropped_themes += 1
                    continue
                counts[key] += 1
                kept_for_record.append(t.strip())
                kept_themes += 1

            kept_per_record.append(len(kept_for_record))
            if len(kept_for_record) < min_themes:
                dropped_records += 1
                continue

            new_record = {
                "source": r.get("source", "unknown"),
                "run_id": r.get("run_id"),
                "topic": topic,
                "entropy_bucket": bucket,
                "themes": kept_for_record,
                "baseline": _format_baseline(topic, bucket, kept_for_record),
                "typed": _format_typed(topic, bucket, kept_for_record),
            }
            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            written += 1
            by_bucket_kept[bucket] += 1

    stats = {
        "input_records": len(records),
        "output_records": written,
        "dropped_records": dropped_records,
        "input_themes_total": sum(in_themes),
        "kept_themes_total": kept_themes,
        "dropped_themes_total": dropped_themes,
        "unique_themes_kept": len([k for k, c in counts.items() if c > 0]),
        "by_bucket_kept": dict(by_bucket_kept),
        "saturated_themes": [k for k, c in counts.items() if c >= cap][:20],
    }
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--in", dest="inputs", nargs="+", required=True,
        help="Input paired JSONL files (one or more — usually real + synth).",
    )
    ap.add_argument(
        "--out", default="distill/train_pairs.jsonl",
        help="Output deduped JSONL.",
    )
    ap.add_argument(
        "--cap", type=int, default=3,
        help="Max appearances per (normalized) theme. Default 3.",
    )
    ap.add_argument(
        "--min-themes", type=int, default=5,
        help="Drop records that end up with fewer than this many themes.",
    )
    ap.add_argument(
        "--seed", type=int, default=17,
        help="RNG seed for fair shuffling.",
    )
    args = ap.parse_args()

    stats = dedupe(
        [Path(p) for p in args.inputs],
        Path(args.out),
        cap=args.cap,
        min_themes=args.min_themes,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
