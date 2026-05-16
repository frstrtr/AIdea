"""Per-user anti-fatigue layer for AIdea's sampling.

Three retention mechanics, all data-driven from ``transcripts.jsonl``:

  1. Per-user donor cooldown — sample_cards downweights any card whose
     ``domain`` appeared in the user's last N runs (recent_card_domains).
  2. Mood-tag rotation — sample_cards prefers mood tags the user has NOT
     seen recently (recent_mood_tags).
  3. Novelty-saturation alarm — when more than ``threshold`` of the user's
     last-week donors are also in the prior week, donor_repeat_rate() fires
     and the pipeline forces a fresh deck-gen + wider wander.

This module owns *only* per-user history queries. The mechanics that
apply the results (downweighting, rotation, cache bypass, prompt
wildcards) live in ``aidea.py`` where the sampling and prompt logic is.

All scans walk transcripts.jsonl directly — no index, no extra storage.
Cheap because we only read recent runs / a 7-day window. Transcripts
file is forward-compatible: cards in past idea-events that lack a
mood_tag field simply don't contribute to mood rotation, the cooldown
still works on domain alone.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

TRANSCRIPTS = Path(__file__).parent / "transcripts.jsonl"


def _read_idea_events_for_source(
    source: str,
    since_ts: float = 0.0,
) -> list[dict]:
    """Every 'idea' transcript event for ``source`` after ``since_ts``.
    Returns list of {ts, run_id, cards} dicts in file order."""
    if not source or not TRANSCRIPTS.exists():
        return []
    out: list[dict] = []
    try:
        with TRANSCRIPTS.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("source") != source:
                    continue
                if ev.get("kind") != "idea":
                    continue
                if ev.get("ts", 0) < since_ts:
                    continue
                out.append({
                    "ts": ev.get("ts", 0),
                    "run_id": ev.get("run_id", ""),
                    "cards": ev.get("cards", []) or [],
                })
    except OSError:
        return []
    return out


def _group_runs(events: list[dict]) -> list[dict]:
    """Group raw idea-events by run_id, return list of {ts, run_id, cards}
    where cards is the union of all idea events for that run. Sorted
    newest-first by max event ts."""
    by_run: dict[str, dict] = {}
    for e in events:
        rid = e.get("run_id") or ""
        if not rid:
            continue
        if rid not in by_run:
            by_run[rid] = {"run_id": rid, "ts": e["ts"], "cards": list(e["cards"])}
        else:
            by_run[rid]["cards"].extend(e["cards"])
            if e["ts"] > by_run[rid]["ts"]:
                by_run[rid]["ts"] = e["ts"]
    runs = list(by_run.values())
    runs.sort(key=lambda r: r["ts"], reverse=True)
    return runs


def recent_card_domains(source: str, n_runs: int = 3) -> set[str]:
    """Donor card domains the user has seen in their last N runs.

    Used by sample_cards as a soft cooldown — Layer 1 of the anti-fatigue
    stack. Empty set if the user has never run AIdea before."""
    runs = _group_runs(_read_idea_events_for_source(source))
    domains: set[str] = set()
    for r in runs[:n_runs]:
        for c in r["cards"]:
            d = (c or {}).get("domain")
            if d:
                domains.add(d)
    return domains


def recent_mood_tags(source: str, n_runs: int = 3) -> dict[str, int]:
    """Counter {mood_tag: count} for the user's last N runs.

    Empty when no cards have been tagged yet (Layer 2 only kicks in
    organically after the alignment+mood scorer has populated mood_tag
    on enough runs). Caller picks LOWEST-count tags to bias sampling
    toward — that's the rotation mechanic."""
    runs = _group_runs(_read_idea_events_for_source(source))
    counter: dict[str, int] = {}
    for r in runs[:n_runs]:
        for c in r["cards"]:
            t = (c or {}).get("mood_tag")
            if t:
                counter[t] = counter.get(t, 0) + 1
    return counter


def donor_repeat_rate(source: str, days: int = 7) -> float:
    """Rolling N-day donor-domain repeat rate for this source.

    Returns ratio in [0, 1]: fraction of donor domains in the most
    recent half-window (last ``days/2`` days) that ALSO appeared in the
    prior half-window (``days`` to ``days/2`` days ago).

    >0.4 is the saturation threshold — the user is seeing the same
    themes too often and the pipeline should force a fresh deck-gen +
    wider wander on the next run. This is the 'BeReal Day 200' defense
    (Layer 3)."""
    now = time.time()
    half = (days * 86400) / 2
    events = _read_idea_events_for_source(source, since_ts=now - 2 * half)
    recent: set[str] = set()
    prior: set[str] = set()
    for e in events:
        ts = e["ts"]
        for c in e["cards"]:
            d = (c or {}).get("domain")
            if not d:
                continue
            if ts >= now - half:
                recent.add(d)
            else:
                prior.add(d)
    if not recent:
        return 0.0
    overlap = recent & prior
    return len(overlap) / max(1, len(recent))


# Saturation threshold — exposed as a module-level constant so callers
# (and tests) can reference the same value the alarm uses.
SATURATION_THRESHOLD: float = 0.4
