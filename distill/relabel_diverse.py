"""Path B — re-label every topic with a bias-fighting teacher prompt.

The Claude teacher gravitates toward a small set of favourite donor
domains (harbor pilotage, monastic rules, mycorrhizal symbiosis,
falconry, lacquer restoration, sourdough …) regardless of topic. A
distill student trained on those labels would amplify the bias the
production anti-fatigue layer was deployed to fight.

This script re-labels each topic via a parallel teacher call whose
prompt explicitly forbids those over-used domains plus their common
near-rephrasings. Output is appended to ``distill/diverse_pairs.jsonl``
in the same OmniLang paired-data shape as build_dataset.py, resumable
on restart.

Usage::

    # Re-label every topic from extract+synth at a fresh random bucket,
    # forbidding the top-20 saturated domains from the dedup stats:
    python -m distill.relabel_diverse \\
        --topics distill/real_pairs.jsonl distill/synth_pairs.jsonl \\
        --avoid distill/avoid.txt \\
        --out distill/diverse_pairs.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aidea import (  # noqa: E402
    THEMES_SYSTEM,
    _query_text,
    _theme_guidance,
)


_BUCKET_SPREAD = {
    "sane":   0.10,
    "wild":   0.40,
    "insane": 0.65,
    "crazy":  0.85,
    "mad":    0.98,
}

_DEFAULT_MODEL = "claude-opus-4-7"
_DEFAULT_N_THEMES = 8


def _load_avoid(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _avoid_block(avoid: list[str]) -> str:
    if not avoid:
        return ""
    lines = "\n".join(f"  - {a}" for a in avoid)
    return (
        "\n\nAVOID THESE OVERUSED DOMAINS — do NOT include any of these, "
        "and do NOT include near-rephrasings of them (different word "
        "order, RU/EN swaps, sub-categories of the same craft):\n"
        f"{lines}\n\n"
        "If one of these would have been your pick, find an alternative "
        "from a STRUCTURALLY DIFFERENT lateral field — try industries / "
        "crafts / sciences / sub-cultures that this list does not touch."
    )


async def diverse_themes(
    topic: str,
    bucket: str,
    avoid: list[str],
    *,
    n_themes: int,
    model: str,
) -> list[str]:
    """Local variant of aidea.generate_themes that injects an AVOID block.
    Same THEMES_SYSTEM, same THEMES_PROMPT shape — only the user prompt
    is augmented."""
    spread = _BUCKET_SPREAD[bucket]
    prompt = (
        f"Pick {n_themes} donor domains for cross-pollination with this user "
        f"problem / question / project:\n\n  {topic.strip()}\n\n"
        f"Entropy of theme selection: {spread*100:.0f}% — see guidance below.\n\n"
        f"Guidance:\n{_theme_guidance(spread)}"
        f"{_avoid_block(avoid)}\n\n"
        f"Output exactly {n_themes} domain names, one per line, no numbering, "
        f"no bullets, no commentary. The names must be specific enough to "
        f"suggest mechanisms — not bare category labels.\n\nBegin:"
    )
    raw = await _query_text(prompt, THEMES_SYSTEM, model, kind="distill_diverse")
    out: list[str] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-•*0123456789).: ").strip()
        if not line or len(line) > 200:
            continue
        out.append(line)
        if len(out) >= n_themes:
            break
    return out


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


def _load_topics(paths: list[Path]) -> list[tuple[str, str]]:
    """Collect (topic, src_tag) pairs from one or more paired JSONL inputs,
    de-duped by topic."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for p in paths:
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
                t = (r.get("topic") or "").strip()
                if not t:
                    continue
                key = t.lower()[:120]
                if key in seen:
                    continue
                seen.add(key)
                out.append((t, r.get("source", "unknown")))
    return out


def _already_done(out_path: Path) -> set[str]:
    seen: set[str] = set()
    if not out_path.exists():
        return seen
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = (r.get("topic") or "").strip().lower()[:120]
            if t:
                seen.add(t)
    return seen


async def main_async(args: argparse.Namespace) -> int:
    inputs = [Path(p) for p in args.topics]
    avoid = _load_avoid(Path(args.avoid)) if args.avoid else []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    topics = _load_topics(inputs)
    done = _already_done(out_path)
    pending = [(t, s) for t, s in topics if t.lower()[:120] not in done]
    print(f"Total topics: {len(topics)}  "
          f"already done: {len(done)}  "
          f"pending: {len(pending)}", file=sys.stderr, flush=True)
    print(f"Avoid list: {len(avoid)} domains", file=sys.stderr, flush=True)

    if args.limit:
        pending = pending[: args.limit]

    rng = random.Random(time.time_ns() & 0xffffffff)
    buckets = list(_BUCKET_SPREAD.keys())

    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i, (topic, src_tag) in enumerate(pending, 1):
            bucket = rng.choice(buckets)
            try:
                themes = await diverse_themes(
                    topic=topic, bucket=bucket, avoid=avoid,
                    n_themes=args.n_themes, model=args.model,
                )
            except Exception as e:
                print(f"  [{i}/{len(pending)}] failed for «{topic[:50]}»: {e}",
                      file=sys.stderr, flush=True)
                continue
            if not themes:
                continue
            record = {
                "source": f"{src_tag}-diverse",
                "topic": topic,
                "entropy_bucket": bucket,
                "themes": themes,
                "baseline": _format_baseline(topic, bucket, themes),
                "typed": _format_typed(topic, bucket, themes),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if i % 10 == 0:
                print(f"  [{i}/{len(pending)}] wrote {written} records",
                      file=sys.stderr, flush=True)

    print(f"\nWrote {written} diverse records to {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topics", nargs="+", required=True,
                    help="Paired JSONL inputs to extract topics from.")
    ap.add_argument("--avoid", default=None,
                    help="Text file with one domain per line — these and their "
                         "rephrasings will be explicitly forbidden in the prompt.")
    ap.add_argument("--out", default="distill/diverse_pairs.jsonl",
                    help="Output JSONL — appended, resumable.")
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--n-themes", type=int, default=_DEFAULT_N_THEMES)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, label only this many topics (smoke test).")
    args = ap.parse_args()

    os.environ.setdefault("AIDEA_SOURCE", "distill")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
