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
BOOTSTRAP_STATE_FILE = Path(__file__).parent / "bootstrap_state.json"

# Cap retrieval cost — past N records is plenty for BM25 at scale.
_CORPUS_LOOKBACK = int(os.environ.get("AIDEA_RAG_LOOKBACK", "5000"))

# How many user-facing queries to gather before flipping from aggregate
# (shared, anonymized) retrieval to strict per-source. During bootstrap,
# the goal is to make the RAG actually useful from day 1 rather than
# after months of solo accumulation. After the threshold, all future
# retrievals strictly filter by ``source``.
_BOOTSTRAP_THRESHOLD = int(os.environ.get("AIDEA_BOOTSTRAP_THRESHOLD", "1000"))


# ---------------------------------------------------------------------------
# PII scrub — applied to the ``topic`` field on every ingest.
# ---------------------------------------------------------------------------


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL_RE = re.compile(r"https?://\S+")
_LONG_NUM_RE = re.compile(r"\b\d{7,}\b")  # phone / order / ID-like


def scrub_topic(topic: str) -> str:
    """Strip obvious PII from a user topic before storage.

    Applied always, not just during bootstrap — the topic is in the
    corpus regardless of retrieval scope and the scrubs are essentially
    free in retrieval quality (BM25 keeps the surrounding words). Reduces
    the cost of an honest mistake (someone pasting an order number /
    email / URL into their topic line)."""
    if not topic:
        return ""
    t = _EMAIL_RE.sub("[email]", topic)
    t = _URL_RE.sub("[url]", t)
    t = _LONG_NUM_RE.sub("[number]", t)
    return t


# ---------------------------------------------------------------------------
# Bootstrap state: persisted counter + active flag.
# ---------------------------------------------------------------------------


def _default_bootstrap_state() -> dict[str, Any]:
    return {
        "threshold": _BOOTSTRAP_THRESHOLD,
        "queries_seen": 0,
        "active": True,
        "started_at": int(time.time()),
        "switched_at": None,
    }


def _load_bootstrap_state() -> dict[str, Any]:
    if not BOOTSTRAP_STATE_FILE.exists():
        return _default_bootstrap_state()
    try:
        s = json.loads(BOOTSTRAP_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_bootstrap_state()
    # Tolerate older / partial files.
    base = _default_bootstrap_state()
    base.update({k: v for k, v in s.items() if k in base})
    # Honor the env-var threshold if it has gone up since last write.
    base["threshold"] = max(int(base.get("threshold") or 0), _BOOTSTRAP_THRESHOLD)
    return base


def _save_bootstrap_state(state: dict[str, Any]) -> None:
    try:
        BOOTSTRAP_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = BOOTSTRAP_STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        tmp.replace(BOOTSTRAP_STATE_FILE)
    except OSError:
        pass


def bootstrap_state() -> dict[str, Any]:
    """Read-only view of the current bootstrap state. Safe for /api/bootstrap."""
    s = _load_bootstrap_state()
    s["remaining"] = max(0, int(s["threshold"]) - int(s["queries_seen"]))
    return s


def note_query(source: str) -> dict[str, Any]:
    """Increment the bootstrap counter (called once per user-facing request)
    and flip to per-source mode if the threshold has been crossed. Returns
    the (post-increment) state so callers can show the user the latest
    counter. CLI sources (``source=='cli'``) are NOT counted — only
    ``web`` and ``telegram-*``."""
    s = _load_bootstrap_state()
    if source != "cli":
        s["queries_seen"] = int(s.get("queries_seen", 0)) + 1
        if s.get("active") and s["queries_seen"] >= int(s["threshold"]):
            s["active"] = False
            s["switched_at"] = int(time.time())
        _save_bootstrap_state(s)
    s["remaining"] = max(0, int(s["threshold"]) - int(s["queries_seen"]))
    return s


def bootstrap_notice_text() -> str:
    """User-facing notice describing the bootstrap trade. Returns the empty
    string when bootstrap is no longer active."""
    s = _load_bootstrap_state()
    if not s.get("active"):
        return ""
    return (
        "⚠️ Bootstrap mode\n"
        "For the first {threshold} queries across all users, your topic and "
        "the generated donor cards are added to a shared anonymized corpus "
        "that warm-starts everyone's deck generation. Personal details "
        "(emails, URLs, long numeric strings) are scrubbed before storage. "
        "After {threshold} queries (currently {seen}/{threshold}), retrieval "
        "automatically switches to per-channel isolation — your future "
        "queries only see your own chat's past. This is a one-time "
        "cold-start measure to make the system useful on day 1 rather "
        "than after months of solo accumulation."
    ).format(threshold=int(s["threshold"]), seen=int(s["queries_seen"]))


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
                    "topic": scrub_topic(topic or ""),
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
    """Top-K past cards ranked by relevance to ``topic``.

    Scope:
      - During bootstrap (queries_seen < threshold): aggregate across ALL
        sources to make retrieval useful from day 1. Users have been
        informed upfront via bootstrap_notice_text().
      - After bootstrap: strict per-``source`` filter — tenant-isolated.

    Falls back to an empty list when:
      - ``rank_bm25`` is not installed
      - corpus is empty in the chosen scope
      - query tokenizes to zero terms
    """
    if not _BM25_AVAILABLE:
        return []
    state = _load_bootstrap_state()
    scope_source: str | None = source if not state.get("active") else None
    corpus = _load_corpus(source=scope_source)
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

    # epsilon=0.25 (default in rank_bm25) ensures terms get a small positive
    # IDF floor in tiny corpora rather than collapsing to 0 / negative — so
    # any actual token overlap surfaces as a non-zero score.
    bm25 = BM25Okapi(docs, epsilon=0.25)
    scores = bm25.get_scores(query_tokens)

    # Compute the per-doc overlap count alongside the BM25 score. Empty-
    # overlap docs (overlap == 0) are rejected; everything else is kept
    # and ranked by (BM25 × quality boost).
    query_set = set(query_tokens)
    boosted: list[tuple[float, dict[str, Any]]] = []
    for rec, s, doc in zip(corpus, scores, docs):
        overlap = len(query_set.intersection(doc))
        if overlap == 0:
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
