"""Backfill quality labels for past runs.

Walks ``transcripts.jsonl``, finds every completed run, asks the existing
critic to score the produced idea against the original topic, and writes a
``winner`` outcome record so the deck-gen RAG can boost contributing cards
on future retrievals.

Idempotent — run_ids that already have a winner outcome are skipped. Safe
to re-run; ``--force`` recomputes them anyway.

The hot loop reuses the same critic that runs in live ``refine=true`` mode,
so the resulting scores are directly comparable to organically-produced
labels.

Usage::

    python3 rerank.py                 # score all unscored completed runs
    python3 rerank.py --limit 3       # score at most 3 (smoke test)
    python3 rerank.py --dry-run       # show what would be scored
    python3 rerank.py --force         # recompute even already-scored runs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import rag
from aidea import critic_score, total_score


def load_runs(transcripts_path: Path) -> dict[str, dict]:
    """Group transcript events by run_id. Returns a dict keyed by run_id with
    {topic, source, mode, completed, errored, ideas: [{text, cards}, ...]}."""
    runs: dict[str, dict] = defaultdict(lambda: {"ideas": []})
    if not transcripts_path.exists():
        return {}
    with transcripts_path.open(encoding="utf-8") as f:
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
            r = runs[rid]
            if kind == "request_started":
                r["topic"] = ev.get("topic", "") or ""
                r["mode"] = ev.get("mode", "default") or "default"
                r["source"] = ev.get("source", "") or ""
            elif kind == "idea":
                r["ideas"].append({
                    "text": ev.get("text", "") or "",
                    "cards": ev.get("cards", []) or [],
                })
                if not r.get("source"):
                    r["source"] = ev.get("source", "") or ""
            elif kind == "request_completed":
                r["completed"] = True
            elif kind == "request_errored":
                r["errored"] = True
    return runs


def already_scored() -> set[str]:
    """Run_ids that already have a winner outcome — these are skipped."""
    out: set[str] = set()
    p = rag.OUTCOMES_LOG
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind", "winner") == "winner" and rec.get("run_id"):
                out.add(rec["run_id"])
    return out


async def score_one(rid: str, r: dict, model: str) -> tuple[int, dict]:
    """Score the run's idea (max over multi-idea modes). Returns (total, score_dict)."""
    topic = r["topic"]
    best_total = -1
    best_score: dict = {}
    for idea in r["ideas"]:
        s = await critic_score(topic, idea["text"], model)
        t = total_score(s)
        if t > best_total:
            best_total = t
            best_score = s
    return best_total, best_score


def best_idea_cards(r: dict) -> list[str]:
    """Card names from the first idea — all ideas in a run share the deck."""
    if not r["ideas"]:
        return []
    return [c.get("name", "") for c in r["ideas"][0].get("cards", []) if c.get("name")]


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--transcripts", default="transcripts.jsonl",
                    help="Path to transcripts.jsonl (default: ./transcripts.jsonl)")
    ap.add_argument("--model", default="claude-opus-4-7",
                    help="Model for the critic call")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process this many candidates")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be scored, don't call the model")
    ap.add_argument("--force", action="store_true",
                    help="Re-score run_ids that already have a winner outcome")
    args = ap.parse_args()

    runs = load_runs(Path(args.transcripts))
    if not runs:
        print(f"No runs found in {args.transcripts}", file=sys.stderr)
        return 1

    already = set() if args.force else already_scored()

    candidates: list[tuple[str, dict]] = []
    skipped_already = 0
    skipped_no_idea = 0
    skipped_no_topic = 0
    for rid, r in runs.items():
        if rid in already:
            skipped_already += 1
            continue
        if not r["ideas"]:
            skipped_no_idea += 1
            continue
        if not r.get("topic", "").strip():
            skipped_no_topic += 1
            continue
        candidates.append((rid, r))

    if args.limit:
        candidates = candidates[:args.limit]

    print(f"Runs found:        {len(runs)}")
    print(f"Already scored:    {skipped_already}")
    print(f"No idea (errored): {skipped_no_idea}")
    print(f"No topic:          {skipped_no_topic}")
    print(f"To score:          {len(candidates)}")
    if args.dry_run:
        for rid, r in candidates:
            print(f"  · {rid}  [{r.get('source','?')}]  "
                  f"{r['topic'][:70]}")
        return 0

    if not candidates:
        print("Nothing to do.")
        return 0

    totals: list[int] = []
    for i, (rid, r) in enumerate(candidates, 1):
        topic_preview = r["topic"][:60].replace("\n", " ")
        print(f"[{i}/{len(candidates)}] {rid}  [{r.get('source','?')}]")
        print(f"    topic: {topic_preview}")
        try:
            total, score = await score_one(rid, r, args.model)
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            continue
        card_names = best_idea_cards(r)
        rag.record_winner(rid, card_names, total, source=r.get("source", ""))
        totals.append(total)
        print(f"    score={total}/400  feas={score.get('feasibility')} "
              f"unex={score.get('unexpectedness')} "
              f"uniq={score.get('uniqueness')} "
              f"fit={score.get('topic_fit')}  "
              f"({len(card_names)} cards labeled)")

    if totals:
        avg = sum(totals) / len(totals)
        print()
        print(f"Scored {len(totals)} runs. Mean total: {avg:.1f}/400 "
              f"(min {min(totals)}, max {max(totals)}).")

    s = rag.stats()
    print()
    print(f"Corpus now: total_cards={s['total_cards']}, "
          f"winner_runs_recorded={s['winner_runs_recorded']}, "
          f"cards_with_quality_label={s['cards_with_quality_label']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
