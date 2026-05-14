"""Card-level RAG for AIdea — self-evolution at the deck-gen layer only.

Compounds quality across runs by retrieving donor concepts that have proven
useful for similar past topics from the SAME source, then injecting them
as "warm-start examples" into the deck-generation prompt for the next run.

Architectural rule (load-bearing): retrieval happens at the deck-gen layer
ONLY, never at the synthesis layer. Injecting retrieved past ideas into the
synthesizer collapses the entropy mechanism the whole system was built to
inject. Cards seed; ideas remain fresh.

Privacy: retrieval is scoped per-source by default. A telegram chat_id's
corpus is isolated from other chats and from web users. There is no
shared global pool — each source has its own RAG memory.

Quality signal: each ingested card carries provenance (topic / run_id /
mode / source). When a refine pass picks a winner, ``record_winner()``
links the winning idea's contributing cards to their ``critic_total``
score. At retrieval time, candidates are ranked by
``bm25_score * (1 + critic_total / 300)`` so proven winners surface
preferentially. Unscored cards rank by similarity alone — the corpus
self-evolves as more refine runs accumulate.

Backends:
- BM25 (default; ``rank_bm25``; works from day 1 with zero corpus, no
  model download). Token-level lexical retrieval. Right tool for a
  small / growing corpus.
- (planned) Local CPU embeddings via ``fastembed`` or
  ``sentence-transformers`` for when lexical retrieval starts missing
  semantic overlap. Drop in behind the same ``retrieve_similar`` API.
  Switch via ``AIDEA_RAG_BACKEND=embed`` once the corpus is large
  enough to justify it.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None  # type: ignore[assignment]
    _BM25_AVAILABLE = False


CARDS_CORPUS = Path(__file__).parent / "cards_corpus.jsonl"
OUTCOMES_LOG = Path(__file__).parent / "card_outcomes.jsonl"

# Cap retrieval cost — past N records is plenty for BM25 at scale.
_CORPUS_LOOKBACK = int(os.environ.get("AIDEA_RAG_LOOKBACK", "5000"))


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Cheap, language-agnostic word tokenizer. Lowercased, single-char dropped."""
    if not text:
        return []
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) > 1]


# ---------------------------------------------------------------------------
# Ingest: append every card from a freshly-generated deck.
# ---------------------------------------------------------------------------


def ingest_deck(
    source: str,
    run_id: str,
    topic: str,
    cards: list[Any],
    mode: str = "default",
) -> int:
    """Append one record per card to ``cards_corpus.jsonl``. Returns count.

    ``cards`` is a list of ``Card`` dataclass instances (or anything with
    a ``__dict__`` carrying ``name`` / ``domain`` and optional depth fields)."""
    if not cards:
        return 0
    try:
        CARDS_CORPUS.parent.mkdir(parents=True, exist_ok=True)
        ts = time.time()
        with CARDS_CORPUS.open("a", encoding="utf-8") as f:
            for c in cards:
                body = {
                    k: v
                    for k, v in (
                        c.__dict__ if hasattr(c, "__dict__") else c
                    ).items()
                    if v is not None
                }
                rec = {
                    "ts": ts,
                    "run_id": run_id or "unknown",
                    "source": source or "unknown",
                    "topic": topic or "",
                    "mode": mode or "default",
                    "card": body,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(cards)
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Outcomes: link winning cards to a critic score.
# ---------------------------------------------------------------------------


def record_winner(
    run_id: str,
    winning_card_names: list[str],
    critic_total: int,
    source: str = "",
) -> None:
    """Mark the cards that contributed to a refine-winner with their score.
    Outcomes live in a separate JSONL so the corpus stays append-only."""
    if not run_id or critic_total is None:
        return
    try:
        OUTCOMES_LOG.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.time(),
            "run_id": run_id,
            "source": source,
            "kind": "winner",
            "winning_card_names": [str(n) for n in winning_card_names],
            "critic_total": int(critic_total),
        }
        with OUTCOMES_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def record_feedback(
    run_id: str,
    useful: bool,
    comment: str = "",
    source: str = "",
) -> None:
    """Append a user-feedback signal targeting a prior run.

    Translates to an additive critic_total boost/penalty when retrieval ranks
    cards from that run: +50 for "useful", -100 for "not useful". The
    feedback record is also written to the transcript log so future analysis
    can correlate explicit feedback with idea content.
    """
    if not run_id:
        return
    boost = 50 if bool(useful) else -100
    try:
        OUTCOMES_LOG.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.time(),
            "run_id": run_id,
            "source": source,
            "kind": "feedback",
            "useful": bool(useful),
            "comment": (comment or "")[:1000],
            "critic_total": int(boost),
        }
        with OUTCOMES_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass
    # Also write to the structured transcript log for richer analysis later.
    try:
        from transcripts import log_event as _tlog, set_source
        if source:
            set_source(source)
        _tlog("feedback", run_id_target=run_id, useful=bool(useful),
              comment=(comment or "")[:1000])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Loaders.
# ---------------------------------------------------------------------------


def _load_corpus(source: str | None = None) -> list[dict[str, Any]]:
    """Load corpus records, optionally filtered by source."""
    if not CARDS_CORPUS.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with CARDS_CORPUS.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if source is not None and rec.get("source") != source:
                    continue
                out.append(rec)
    except OSError:
        return []
    # Most-recent-first, cap to lookback window.
    out.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return out[:_CORPUS_LOOKBACK]


def _load_outcomes() -> dict[str, int]:
    """Map run_id -> combined quality signal for that run.

    Combines the critic's winner score (0..300, kept as max if duplicated)
    with explicit user feedback (additive: +50 "useful", -100 "not useful").
    Clamped to [-300, 300] so a single noisy signal can't dominate. Feedback
    overrides the critic on negative ("user says it's broken" wins over
    "critic gave it a 220"), reinforces on positive.
    """
    if not OUTCOMES_LOG.exists():
        return {}
    winner_max: dict[str, int] = {}
    feedback_sum: dict[str, int] = {}
    try:
        with OUTCOMES_LOG.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = rec.get("run_id")
                total = rec.get("critic_total")
                kind = rec.get("kind", "winner")
                if not (rid and isinstance(total, (int, float))):
                    continue
                if kind == "feedback":
                    feedback_sum[rid] = feedback_sum.get(rid, 0) + int(total)
                else:
                    if rid not in winner_max or winner_max[rid] < int(total):
                        winner_max[rid] = int(total)
    except OSError:
        return {}
    combined: dict[str, int] = {}
    for rid in set(winner_max) | set(feedback_sum):
        v = winner_max.get(rid, 0) + feedback_sum.get(rid, 0)
        combined[rid] = max(-300, min(300, v))
    return combined


# ---------------------------------------------------------------------------
# Retrieve: top-K past cards by BM25 over (topic + card body), boosted by
# critic_total. Source-scoped.
# ---------------------------------------------------------------------------


def retrieve_similar(
    source: str,
    topic: str,
    k: int = 8,
) -> list[dict[str, Any]]:
    """Top-K past cards from ``source`` ranked by relevance to ``topic``.

    Falls back to an empty list when:
      - ``rank_bm25`` is not installed
      - corpus is empty for this source
      - query tokenizes to zero terms
    """
    if not _BM25_AVAILABLE:
        return []
    corpus = _load_corpus(source=source)
    if not corpus:
        return []
    outcomes = _load_outcomes()

    query_tokens = _tokenize(topic)
    if not query_tokens:
        return []

    docs: list[list[str]] = []
    for rec in corpus:
        card = rec.get("card", {}) or {}
        # Concatenate topic + every text field on the card for lexical match.
        parts = [
            rec.get("topic", ""),
            str(card.get("name", "")),
            str(card.get("domain", "")),
            str(card.get("mechanism", "")),
            str(card.get("why", "")),
            str(card.get("transfer", "")),
            str(card.get("invariants", "")),
            str(card.get("prior_application", "")),
        ]
        docs.append(_tokenize(" ".join(p for p in parts if p)))

    # If every doc is empty, BM25 will raise.
    if not any(docs):
        return []

    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(query_tokens)

    boosted: list[tuple[float, dict[str, Any]]] = []
    for rec, s in zip(corpus, scores):
        if s <= 0:
            continue
        # Combined critic + feedback signal in [-300, +300].
        total = outcomes.get(rec.get("run_id"))
        if isinstance(total, (int, float)):
            # total=-300 -> 0.0 (effectively filtered), 0 -> 1.0, +300 -> 2.0
            boost = max(0.0, 1.0 + float(total) / 300.0)
        else:
            boost = 1.0
        boosted.append((s * boost, rec))

    boosted.sort(key=lambda x: -x[0])
    # Dedupe by card name (same card may appear from multiple past runs).
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for _, rec in boosted:
        name = (rec.get("card") or {}).get("name", "")
        key = str(name).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(rec)
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# Render retrieved cards for the deck-gen prompt.
# ---------------------------------------------------------------------------


def render_warm_start(retrieved: list[dict[str, Any]]) -> str:
    """Format top-K retrieved cards as a prompt-friendly bullet list.
    Returns the empty string when retrieved is empty so the deck-gen prompt
    can omit the warm-start section cleanly."""
    if not retrieved:
        return ""
    lines: list[str] = []
    for rec in retrieved:
        c = rec.get("card") or {}
        name = c.get("name", "?")
        domain = c.get("domain", "?")
        mech = c.get("mechanism", "") or ""
        header = f"- {name} ({domain})"
        if mech:
            # Single-line mechanism so the prompt stays readable.
            mech = mech.replace("\n", " ").strip()
            header += f"\n    mechanism: {mech}"
        lines.append(header)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stats for /corpus, /api/corpus.
# ---------------------------------------------------------------------------


def stats(source: str | None = None) -> dict[str, Any]:
    """Return corpus health metrics. ``source=None`` aggregates across all
    sources (admin view); a specific source returns a tenant-scoped view."""
    corpus = _load_corpus(source=source)
    outcomes = _load_outcomes()
    by_source: dict[str, int] = {}
    by_mode: dict[str, int] = {}
    topics: set[str] = set()
    for rec in corpus:
        by_source[rec.get("source", "?")] = by_source.get(rec.get("source", "?"), 0) + 1
        by_mode[rec.get("mode", "?")] = by_mode.get(rec.get("mode", "?"), 0) + 1
        t = (rec.get("topic") or "").strip()
        if t:
            topics.add(t.lower())
    # Scored cards: those whose run_id has an outcome entry.
    scored = sum(
        1 for rec in corpus if outcomes.get(rec.get("run_id"))
    )
    return {
        "backend": "bm25" if _BM25_AVAILABLE else "unavailable",
        "available": _BM25_AVAILABLE,
        "scope": source or "(all sources)",
        "total_cards": len(corpus),
        "unique_topics": len(topics),
        "by_source": by_source,
        "by_mode": by_mode,
        "winner_runs_recorded": len(outcomes),
        "cards_with_quality_label": scored,
    }


def format_stats_text(s: dict[str, Any]) -> str:
    """Plain-text summary for terminal / Telegram /corpus output."""
    lines = [
        f"Backend: {s['backend']}",
        f"Scope:   {s['scope']}",
        f"Cards:   {s['total_cards']}  ({s['cards_with_quality_label']} with quality label)",
        f"Topics:  {s['unique_topics']} distinct",
        f"Refine winners on record: {s['winner_runs_recorded']}",
    ]
    if s["by_source"]:
        lines.append("By source:")
        for k, v in sorted(s["by_source"].items(), key=lambda kv: -kv[1])[:10]:
            lines.append(f"  {k}: {v}")
    if s["by_mode"]:
        lines.append("By mode:")
        for k, v in sorted(s["by_mode"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)
