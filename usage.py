"""Usage tracking for the AIdea pipeline.

Captures token / duration / cost from every LLM call (via the agent SDK's
ResultMessage), persists per-call records to a JSONL log, and summarizes
over time windows. Also records RateLimitEvent payloads when they arrive,
since those carry the real subscription-window state (e.g. the 5-hour
reset boundary).

Pipeline integration:
  - usage.start_run() at the top of each request (web / CLI / bot) returns
    a run_id and sets a context-var so every nested LLM call attaches to it.
  - The query-loop wrapper in aidea.py calls record_call(...) per response.
  - usage.summarize() reads the log and returns this-run / 7d / 30d totals
    plus the most-recently-observed rate-limit state.
"""

from __future__ import annotations

import contextvars
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

USAGE_LOG = Path(__file__).parent / "usage.jsonl"


# ---------------------------------------------------------------------------
# Run scoping
# ---------------------------------------------------------------------------


_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aidea_run_id", default=None,
)


def start_run(prefix: str = "run") -> str:
    """Start a new run; subsequent record_call() inherits the id via contextvar."""
    rid = f"{prefix}-{uuid.uuid4().hex[:10]}"
    _current_run_id.set(rid)
    return rid


def current_run_id() -> str | None:
    return _current_run_id.get()


# ---------------------------------------------------------------------------
# Per-call record
# ---------------------------------------------------------------------------


@dataclass
class CallUsage:
    ts: float
    run_id: str
    kind: str               # deck | synth | critic | refine | evolve
    duration_ms: int
    duration_api_ms: int
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    total_cost_usd: float
    model: str
    # Most-recent rate-limit info observed during this call (RateLimitEvent).
    rate_limit_status: str | None = None
    rate_limit_type: str | None = None
    rate_limit_resets_at: int | None = None
    rate_limit_utilization: float | None = None


def _record_path() -> Path:
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    return USAGE_LOG


def record_call(c: CallUsage) -> None:
    """Append one usage record to the JSONL log."""
    try:
        with _record_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    except OSError:
        # Logging is non-essential; never let it break the pipeline.
        pass


def _load_records() -> list[dict]:
    if not USAGE_LOG.exists():
        return []
    out: list[dict] = []
    try:
        with USAGE_LOG.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


# ---------------------------------------------------------------------------
# Extracting usage from an SDK response stream
# ---------------------------------------------------------------------------


def build_call_record(
    *,
    run_id: str,
    kind: str,
    result_message: Any,
    rate_limit_info: Any | None,
) -> CallUsage:
    """Synthesize a CallUsage from a ResultMessage + (optional) RateLimitInfo."""
    u = getattr(result_message, "usage", None) or {}
    model_usage = getattr(result_message, "model_usage", None) or {}
    primary_model = next(iter(model_usage.keys()), "")

    rl_status = getattr(rate_limit_info, "status", None) if rate_limit_info else None
    rl_type = getattr(rate_limit_info, "rate_limit_type", None) if rate_limit_info else None
    rl_resets_at = getattr(rate_limit_info, "resets_at", None) if rate_limit_info else None
    rl_util = getattr(rate_limit_info, "utilization", None) if rate_limit_info else None

    return CallUsage(
        ts=time.time(),
        run_id=run_id,
        kind=kind,
        duration_ms=int(getattr(result_message, "duration_ms", 0) or 0),
        duration_api_ms=int(getattr(result_message, "duration_api_ms", 0) or 0),
        input_tokens=int(u.get("input_tokens", 0) or 0),
        output_tokens=int(u.get("output_tokens", 0) or 0),
        cache_creation_input_tokens=int(u.get("cache_creation_input_tokens", 0) or 0),
        cache_read_input_tokens=int(u.get("cache_read_input_tokens", 0) or 0),
        total_cost_usd=float(getattr(result_message, "total_cost_usd", 0.0) or 0.0),
        model=primary_model or "",
        rate_limit_status=rl_status,
        rate_limit_type=rl_type,
        rate_limit_resets_at=rl_resets_at,
        rate_limit_utilization=rl_util,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _totals(items: list[dict]) -> dict:
    return {
        "calls": len(items),
        "input_tokens": sum(r.get("input_tokens", 0) for r in items),
        "output_tokens": sum(r.get("output_tokens", 0) for r in items),
        "cache_creation_input_tokens": sum(
            r.get("cache_creation_input_tokens", 0) for r in items
        ),
        "cache_read_input_tokens": sum(
            r.get("cache_read_input_tokens", 0) for r in items
        ),
        "total_cost_usd": round(
            sum(r.get("total_cost_usd", 0.0) for r in items), 4,
        ),
        "duration_ms": sum(r.get("duration_ms", 0) for r in items),
        "duration_api_ms": sum(r.get("duration_api_ms", 0) for r in items),
    }


def summarize(run_id: str | None = None) -> dict:
    """Return totals over this-run / last-7d / last-30d / all-time, plus the
    most-recently observed rate-limit window. The 5h-windows-this-week count
    is a LOCAL heuristic — Anthropic does not currently expose the real
    weekly session ceiling through the SDK, so this should be read as
    'distinct 5-hour windows we touched locally in the last 7 days'."""
    records = _load_records()
    now = time.time()
    day = 86_400
    five_h = 5 * 3600

    this_run = [
        r for r in records if run_id and r.get("run_id") == run_id
    ] if run_id else []
    last_7d = [r for r in records if r.get("ts", 0) >= now - 7 * day]
    last_30d = [r for r in records if r.get("ts", 0) >= now - 30 * day]

    # Most recent rate-limit observation
    last_rl: dict | None = None
    for r in reversed(records):
        if r.get("rate_limit_status") or r.get("rate_limit_resets_at"):
            last_rl = {
                "status": r.get("rate_limit_status"),
                "type": r.get("rate_limit_type"),
                "resets_at": r.get("rate_limit_resets_at"),
                "utilization": r.get("rate_limit_utilization"),
                "observed_at": r.get("ts"),
            }
            break

    # Distinct 5-hour wall-clock windows touched in the last 7 days
    windows_7d: set[int] = set()
    for r in last_7d:
        ts = r.get("ts")
        if ts:
            windows_7d.add(int(ts // five_h))

    return {
        "this_run": _totals(this_run),
        "last_7d": _totals(last_7d),
        "last_30d": _totals(last_30d),
        "total": _totals(records),
        "five_h_windows_last_7d": len(windows_7d),
        "rate_limit": last_rl,
        "note": (
            "Tokens / duration / cost come from the agent SDK's ResultMessage. "
            "five_h_windows_last_7d is a local heuristic — the real subscription "
            "ceiling is not exposed through the SDK. rate_limit reflects the most "
            "recently observed RateLimitEvent and is authoritative until the "
            "window resets."
        ),
    }
