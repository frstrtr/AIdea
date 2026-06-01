"""Per-Telegram-user free-tier quota.

Each Telegram user gets a fixed number of free *idea generations* (default
10, override with AIDEA_FREE_LIMIT). Scheduled/interactive brews and admin
users don't count — gating lives in bot.run_pipeline_for_telegram.

State is a tiny JSON file next to the bot so the count survives restarts
(the in-memory ChatState does not). Records are keyed by str(user_id):

    {"count": int, "name": str, "username": str,
     "first_ts": float, "last_ts": float, "reset_ts": float}
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_PATH = Path(__file__).parent / "quota_state.json"
# A threading lock is cheap insurance — the bot is single-threaded asyncio,
# but the read-modify-write below must stay atomic regardless.
_LOCK = threading.Lock()


def free_limit() -> int:
    """Number of free idea generations per user. AIDEA_FREE_LIMIT or 10."""
    try:
        return max(0, int(os.environ.get("AIDEA_FREE_LIMIT", "10")))
    except ValueError:
        return 10


def _load() -> dict:
    try:
        with _PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    # Atomic write — never leave a half-written quota file behind a crash.
    tmp = _PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(_PATH)


def count(user_id: int | str) -> int:
    with _LOCK:
        rec = _load().get(str(user_id))
    return int(rec.get("count", 0)) if rec else 0


def is_over(user_id: int | str, limit: int | None = None) -> bool:
    lim = free_limit() if limit is None else limit
    return count(user_id) >= lim


def remaining(user_id: int | str, limit: int | None = None) -> int:
    lim = free_limit() if limit is None else limit
    return max(0, lim - count(user_id))


def increment(
    user_id: int | str,
    *,
    name: str = "",
    username: str = "",
    ts: float | None = None,
) -> int:
    """Add one to the user's generation count and persist. Returns new count."""
    now = time.time() if ts is None else ts
    key = str(user_id)
    with _LOCK:
        data = _load()
        rec = data.get(key) or {"count": 0, "first_ts": now}
        rec["count"] = int(rec.get("count", 0)) + 1
        rec["last_ts"] = now
        rec.setdefault("first_ts", now)
        if name:
            rec["name"] = name
        if username:
            rec["username"] = username
        data[key] = rec
        _save(data)
        return rec["count"]


def refund(user_id: int | str) -> int:
    """Undo one increment (e.g. the run errored or was cancelled before any
    ideas were delivered). Never goes below zero. Returns the new count."""
    key = str(user_id)
    with _LOCK:
        data = _load()
        rec = data.get(key)
        if rec and int(rec.get("count", 0)) > 0:
            rec["count"] = int(rec["count"]) - 1
            _save(data)
            return rec["count"]
        return int(rec.get("count", 0)) if rec else 0


def reset(user_id: int | str) -> int:
    """Admin: clear a user's count (e.g. after they subscribe). Returns the
    count they had before the reset."""
    key = str(user_id)
    with _LOCK:
        data = _load()
        prior = int((data.get(key) or {}).get("count", 0))
        if key in data:
            data[key]["count"] = 0
            data[key]["reset_ts"] = time.time()
            _save(data)
    return prior
