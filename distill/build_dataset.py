"""A3 — augment the real pairs with synthetic AIdea-style topics labeled by
the teacher (Claude Max via the same agent SDK AIdea uses for synthesis).

Pipeline:

  1. Generate N synthetic topics in batches of ~20, asking the teacher
     to span business / lifestyle / technical / creative / personal /
     learning / travel / relationships / hobbies, and mix RU + EN to
     match AIdea's real traffic distribution.
  2. For each synthetic topic — and (optionally) each real topic that
     extract.py already produced — call the teacher with AIdea's own
     `generate_themes` to produce a gold themes list for a randomly
     sampled entropy bucket.
  3. Write paired OmniLang JSONL — same shape as extract.py.

Resumable: every successful (topic, themes) pair is appended to the
output file as it lands, so a CTRL-C-and-restart picks up from where
it left off (de-dupes by topic). Crash-safe by virtue of being
append-only.

Usage::

    # Generate 500 synthetic topics + label them at random entropy buckets:
    python -m distill.build_dataset --synth 500 --out distill/synth_pairs.jsonl

    # Also re-label the REAL pairs at a fresh random bucket so the model
    # sees each topic at multiple entropy levels:
    python -m distill.build_dataset --relabel-real distill/real_pairs.jsonl \\
                                    --out distill/synth_pairs.jsonl

    # Quick smoke test — generate and label 3 topics:
    python -m distill.build_dataset --synth 3 --out /tmp/smoke.jsonl
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

# Reuse AIdea's wired agent-SDK plumbing (retry layer, transcripts logging
# scope is fine; this script just uses the model client).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aidea import (  # noqa: E402
    generate_themes,
    THEMES_SYSTEM,
)
from aidea import _query_text  # noqa: E402 — private but stable enough for tooling


# Map entropy buckets to numeric theme_entropy values used by
# generate_themes. Mirrors aidea.ENTROPY_LEVELS.spread.
_BUCKET_SPREAD = {
    "sane":   0.10,
    "wild":   0.40,
    "insane": 0.65,
    "crazy":  0.85,
    "mad":    0.98,
}

_DEFAULT_MODEL = "claude-opus-4-7"
_DEFAULT_N_THEMES = 8


_TOPIC_GEN_SYSTEM = (
    "You generate diverse, realistic problem/question/project topics that "
    "real users send to an idea-generation tool. Output ONE topic per line. "
    "Topics should be 1-2 sentences each, no numbering, no preamble, no "
    "commentary. Mix categories (business, lifestyle, technical, creative, "
    "personal, learning, travel, relationships, hobbies, civic). Mix languages "
    "roughly 50/50 English and Russian to reflect actual traffic. Avoid "
    "duplicates — every topic should sit in a different niche."
)


def _topic_gen_prompt(n: int, seed_hint: str) -> str:
    return (
        f"Generate exactly {n} diverse topics. Seed flavour: {seed_hint}.\n"
        f"One topic per line, 1-2 sentences each. No numbering, no bullets.\n"
        f"Begin:"
    )


_SEED_HINTS = [
    "founders and operators chasing growth",
    "lifestyle micro-optimisations",
    "frustrated specialists with one stuck constraint",
    "creative writing and craft problems",
    "post-corporate sabbatical questions",
    "students learning a new technical skill",
    "retirees designing their week",
    "couples planning shared projects",
    "side-hustle prototyping",
    "weekend hackers and tinkerers",
    "policy and civic improvement questions",
    "expat / relocation logistics",
    "marathon-training-style personal projects",
    "underemployed creatives with one big bet",
    "small-business owners squeezed by economics",
]


async def generate_synthetic_topics(n: int, *, model: str) -> list[str]:
    """Generate roughly N topics in batches of ~20, varying the seed hint
    each batch for diversity. Returns a de-duplicated list."""
    out: list[str] = []
    seen: set[str] = set()
    rng = random.Random(42 + n)
    batch_size = 20
    n_batches = (n + batch_size - 1) // batch_size
    for i in range(n_batches):
        hint = _SEED_HINTS[rng.randrange(len(_SEED_HINTS))]
        prompt = _topic_gen_prompt(batch_size, hint)
        try:
            raw = await _query_text(
                prompt, _TOPIC_GEN_SYSTEM, model, kind="distill_topic_gen",
            )
        except Exception as e:
            print(f"  [batch {i+1}/{n_batches}] topic-gen failed: {e}",
                  file=sys.stderr, flush=True)
            continue
        for line in raw.splitlines():
            line = line.strip().lstrip("-•*0123456789).: ").strip()
            if len(line) < 12 or len(line) > 600:
                continue
            key = line.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            out.append(line)
            if len(out) >= n:
                return out
        print(f"  [batch {i+1}/{n_batches}] now have {len(out)}/{n} topics",
              file=sys.stderr, flush=True)
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


def _load_existing_topics(path: Path) -> set[str]:
    """Read whatever pairs are already in the output JSONL so we don't
    re-label them. Resume-safe."""
    seen: set[str] = set()
    if not path.exists():
        return seen
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = rec.get("topic")
                if t:
                    seen.add(t.strip().lower()[:120])
    except OSError:
        pass
    return seen


async def label_one(
    topic: str,
    bucket: str,
    *,
    n_themes: int,
    model: str,
) -> list[str] | None:
    """Call the teacher to produce themes for one (topic, bucket) pair."""
    try:
        themes = await generate_themes(
            topic=topic,
            n_themes=n_themes,
            theme_entropy=_BUCKET_SPREAD[bucket],
            model=model,
        )
    except Exception as e:
        print(f"  [label] failed for «{topic[:50]}…»: {e}",
              file=sys.stderr, flush=True)
        return None
    themes = [t.strip() for t in themes if t and t.strip()]
    if not themes:
        return None
    return themes


async def build_dataset(
    *,
    synth_n: int,
    out_path: Path,
    relabel_real: Path | None = None,
    model: str = _DEFAULT_MODEL,
    n_themes: int = _DEFAULT_N_THEMES,
    buckets_per_topic: int = 1,
) -> int:
    """Main driver. Generates synth_n new topics, optionally re-labels
    each topic in `relabel_real` at a fresh random bucket, and appends
    paired records to out_path. Returns total records written this run.

    Each record is labeled at `buckets_per_topic` different entropy
    buckets — 1 by default (one row per topic) but bump to 2-3 to get
    the model used to seeing the same topic in different conditioning
    contexts."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already = _load_existing_topics(out_path)

    # Topic queue: synthetic first, then optional real re-labels.
    topics: list[tuple[str, str]] = []  # (topic, source_tag)
    if synth_n > 0:
        print(f"Generating {synth_n} synthetic topics via teacher...",
              file=sys.stderr, flush=True)
        synth_topics = await generate_synthetic_topics(synth_n, model=model)
        for t in synth_topics:
            if t.strip().lower()[:120] not in already:
                topics.append((t, "synth"))
        print(f"  → {len(topics)} new synthetic topics queued",
              file=sys.stderr, flush=True)

    if relabel_real and relabel_real.exists():
        with relabel_real.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = rec.get("topic")
                if t and t.strip().lower()[:120] not in already:
                    topics.append((t, "real-relabel"))
        print(f"Total queue size: {len(topics)} (synth + real-relabel)",
              file=sys.stderr, flush=True)

    rng = random.Random(time.time_ns() & 0xffffffff)
    buckets = list(_BUCKET_SPREAD.keys())

    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i, (topic, src_tag) in enumerate(topics, 1):
            picks = rng.sample(buckets, min(buckets_per_topic, len(buckets)))
            for bucket in picks:
                themes = await label_one(
                    topic, bucket, n_themes=n_themes, model=model,
                )
                if not themes:
                    continue
                record = {
                    "source": src_tag,
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
                print(f"  [{i}/{len(topics)}] {written} records written so far",
                      file=sys.stderr, flush=True)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--synth", type=int, default=0,
        help="Generate this many synthetic topics via the teacher.",
    )
    ap.add_argument(
        "--relabel-real", default=None,
        help="Path to extract.py's real_pairs.jsonl — each real topic "
             "gets re-labeled at a fresh random bucket too.",
    )
    ap.add_argument(
        "--out", default="distill/synth_pairs.jsonl",
        help="Output JSONL — appended, resumable.",
    )
    ap.add_argument(
        "--model", default=_DEFAULT_MODEL,
        help="Teacher model id (defaults to AIdea's own).",
    )
    ap.add_argument(
        "--n-themes", type=int, default=_DEFAULT_N_THEMES,
        help="Themes per label call (default 8).",
    )
    ap.add_argument(
        "--buckets-per-topic", type=int, default=1,
        help="Label each topic at this many distinct entropy buckets "
             "(default 1). Bump to 2-3 for more conditioning coverage.",
    )
    args = ap.parse_args()

    # AIDEA_SOURCE keeps these LLM calls properly tagged in transcripts/usage.
    os.environ.setdefault("AIDEA_SOURCE", "distill")

    n = asyncio.run(build_dataset(
        synth_n=args.synth,
        out_path=Path(args.out),
        relabel_real=Path(args.relabel_real) if args.relabel_real else None,
        model=args.model,
        n_themes=args.n_themes,
        buckets_per_topic=args.buckets_per_topic,
    ))
    print(f"\nWrote {n} new records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
