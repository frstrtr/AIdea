"""One-time backfill of the monitoring log group from existing data.

Reads the on-disk DBs (transcripts.jsonl + usage.jsonl + quota_state.json)
and the RAG corpus (via rag.stats), then posts a per-day digest into the
already-created forum topics of the AIDEA_LOG_CHAT_ID group:

  🆕 New users    — chunked list of historical bot users (first-seen + count)
  📥 Generations  — one message per active day (topics/modes/tokens/cost)
  💌 Inquiries    — seed note (no historical data; live tracking started)
  🚧 Quota hits   — seed note (ditto)
  📊 Daily summary— per-day rollup + RAG corpus snapshot + quota state

Idempotent: writes log_backfill_done.json on success and refuses to re-run
without --force. Every message is prefixed ⏪ to mark it as backfill. Posting
is throttled to stay under Telegram's group rate limit.

Usage:
  python backfill_log.py --dry-run     # print what would be posted
  python backfill_log.py               # post for real
  python backfill_log.py --force       # re-run even if the marker exists
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

import os  # noqa: E402  (after load_dotenv so .env wins)

TRANSCRIPTS = HERE / "transcripts.jsonl"
USAGE = HERE / "usage.jsonl"
QUOTA = HERE / "quota_state.json"
LOG_TOPICS = HERE / "log_topics.json"
MARKER = HERE / "log_backfill_done.json"
REQ_MARKER = HERE / "log_backfill_requests_done.json"

DAY_BULLET_CAP = 15  # max per-day generation bullets before "+N more"
USERS_PER_MSG = 15   # new-user list chunk size


# --------------------------------------------------------------------------
# small formatters (kept local so this script has no bot.py side effects)
# --------------------------------------------------------------------------
def fmt_tokens(n: int) -> str:
    n = int(n or 0)
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


def fmt_usd(x: float) -> str:
    return f"${float(x or 0):,.2f}"


def day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts)) if ts else "?"


def loadl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


# --------------------------------------------------------------------------
# build the digest
# --------------------------------------------------------------------------
def build():
    tx = loadl(TRANSCRIPTS)
    usage = loadl(USAGE)

    # usage rollup keyed by run_id
    u_by_run: dict[str, dict] = collections.defaultdict(
        lambda: {"in": 0, "out": 0, "usd": 0.0}
    )
    for r in usage:
        rid = r.get("run_id")
        if not rid:
            continue
        u_by_run[rid]["in"] += int(r.get("input_tokens", 0) or 0)
        u_by_run[rid]["out"] += int(r.get("output_tokens", 0) or 0)
        u_by_run[rid]["usd"] += float(r.get("total_cost_usd", 0) or 0)

    gens = []  # telegram generations
    for r in tx:
        if r.get("kind") != "request_started":
            continue
        src = str(r.get("source", ""))
        if not src.startswith("telegram"):
            continue
        rid = r.get("run_id", "")
        u = u_by_run.get(rid, {"in": 0, "out": 0, "usd": 0.0})
        chat = r.get("chat_id") or src.replace("telegram-", "")
        gens.append({
            "ts": r.get("ts", 0),
            "chat": str(chat),
            "topic": (r.get("topic") or "").strip(),
            "mode": r.get("mode") or "default",
            "in": u["in"], "out": u["out"], "usd": u["usd"],
        })
    gens.sort(key=lambda g: g["ts"])

    # first-seen + count per user
    users: dict[str, dict] = {}
    for g in gens:
        u = users.setdefault(g["chat"], {"first": g["ts"], "count": 0})
        u["first"] = min(u["first"], g["ts"])
        u["count"] += 1

    # group by day
    by_day: dict[str, list] = collections.defaultdict(list)
    for g in gens:
        by_day[day(g["ts"])].append(g)

    return gens, users, by_day


def compose_request_messages() -> list[str]:
    """One message per individual telegram request (faithful replay, nothing
    truncated), oldest→newest — for --all-requests."""
    gens, _users, _by_day = build()
    out = []
    for g in gens:
        topic = (g["topic"] or "(no topic)")[:1500]
        out.append(
            f"⏪ chat {g['chat']}\n"
            f"topic: {topic}\n"
            f"mode: {g['mode']} · {day(g['ts'])}\n"
            f"tokens: {fmt_tokens(g['in'])} in / {fmt_tokens(g['out'])} out · "
            f"{fmt_usd(g['usd'])}"
        )
    return out


def corpus_snapshot() -> str:
    try:
        import rag
        return rag.format_stats_text(rag.stats())
    except Exception as e:
        return f"(corpus snapshot unavailable: {e})"


def quota_snapshot() -> str:
    try:
        q = json.loads(QUOTA.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return ("no users tracked yet — quota tracking went live 2026-06-01.")
    limit = int(os.environ.get("AIDEA_FREE_LIMIT", "10") or 10)
    walled = sum(1 for r in q.values() if int(r.get("count", 0)) >= limit)
    used = sum(int(r.get("count", 0)) for r in q.values())
    return (f"users tracked: {len(q)} · walled (≥{limit}): {walled} · "
            f"free gens used: {used}")


def compose_messages():
    """Return {topic_key: [text, ...]} of everything to post."""
    gens, users, by_day = build()
    msgs: dict[str, list[str]] = collections.defaultdict(list)

    # 🆕 New users — chunked list, oldest first
    ordered = sorted(users.items(), key=lambda kv: kv[1]["first"])
    total = len(ordered)
    for i in range(0, total, USERS_PER_MSG):
        chunk = ordered[i:i + USERS_PER_MSG]
        lines = [f"⏪ Historical bot users ({i+1}–{i+len(chunk)} of {total})"]
        for chat, info in chunk:
            lines.append(
                f"• chat {chat} — first seen {day(info['first'])} · "
                f"{info['count']} gen(s)"
            )
        msgs["new_users"].append("\n".join(lines))

    # 📥 Generations — one message per active day
    for d in sorted(by_day):
        items = by_day[d]
        out_sum = sum(g["out"] for g in items)
        usd_sum = sum(g["usd"] for g in items)
        nusers = len({g["chat"] for g in items})
        lines = [f"⏪ {d} — {len(items)} generation(s)"]
        for g in items[:DAY_BULLET_CAP]:
            topic = g["topic"][:60] or "(no topic)"
            lines.append(
                f"• {topic} ({g['mode']}) — {fmt_tokens(g['out'])} out · "
                f"{fmt_usd(g['usd'])}"
            )
        if len(items) > DAY_BULLET_CAP:
            lines.append(f"  …+{len(items) - DAY_BULLET_CAP} more")
        lines.append(
            f"day total: {fmt_tokens(out_sum)} out · {fmt_usd(usd_sum)} · "
            f"{nusers} user(s)"
        )
        msgs["generations"].append("\n".join(lines))

    # 💌 Inquiries / 🚧 Quota hits — seed notes (no historical data)
    msgs["inquiries"].append(
        "ℹ️ Subscription-inquiry capture went live 2026-06-01 — no historical "
        "inquiries. Messages from users who hit the free limit will appear here."
    )
    msgs["quota_hits"].append(
        "ℹ️ Quota tracking went live 2026-06-01 — no historical quota hits. "
        "Users crossing the free-generation limit will appear here."
    )

    # 📊 Daily summary — per-day rollup, corpus snapshot, quota state
    roll = ["⏪ Per-day activity (telegram), May 14 – Jun 1"]
    for d in sorted(by_day):
        items = by_day[d]
        roll.append(
            f"{d}: {len(items)} gen(s) · {len({g['chat'] for g in items})} "
            f"user(s) · {fmt_tokens(sum(g['out'] for g in items))} out · "
            f"{fmt_usd(sum(g['usd'] for g in items))}"
        )
    g_out = sum(g["out"] for g in gens)
    g_usd = sum(g["usd"] for g in gens)
    roll.append(
        f"— total: {len(gens)} gens · {len(users)} users · "
        f"{fmt_tokens(g_out)} out · {fmt_usd(g_usd)}"
    )
    msgs["summary"].append("\n".join(roll))
    msgs["summary"].append("⏪ RAG corpus snapshot\n" + corpus_snapshot())
    msgs["summary"].append("⏪ Quota state\n" + quota_snapshot())

    return msgs


# --------------------------------------------------------------------------
# posting
# --------------------------------------------------------------------------
def send(token, chat_id, thread_id, text, dry, throttle):
    if dry:
        print(f"\n──[ thread {thread_id} ]──\n{text}")
        return True
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if thread_id is not None:
        payload["message_thread_id"] = thread_id
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=30):
                time.sleep(throttle)
                return True
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 429:
                try:
                    wait = json.loads(body)["parameters"]["retry_after"]
                except Exception:
                    wait = 5
                print(f"  429 — sleeping {wait}s")
                time.sleep(wait + 1)
                continue
            print(f"  send failed {e.code}: {body}")
            return False
        except Exception as e:
            print(f"  send error: {e}")
            return False
    return False


# order topics are posted in
POST_ORDER = ["new_users", "generations", "inquiries", "quota_hits", "summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--throttle", type=float, default=1.5)
    ap.add_argument(
        "--all-requests", action="store_true",
        help="post one message per individual request (faithful, untruncated) "
             "into 📥 Generations instead of the per-day digest",
    )
    args = ap.parse_args()

    marker = REQ_MARKER if args.all_requests else MARKER
    if marker.exists() and not args.force and not args.dry_run:
        print(f"Already done ({marker.read_text().strip()}). Use --force.")
        return 1

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id_raw = os.environ.get("AIDEA_LOG_CHAT_ID", "").strip()
    if not token or not chat_id_raw:
        print("TELEGRAM_BOT_TOKEN and AIDEA_LOG_CHAT_ID must be set.")
        return 1
    chat_id = int(chat_id_raw)

    try:
        topics = json.loads(LOG_TOPICS.read_text()).get(str(chat_id), {})
    except (FileNotFoundError, json.JSONDecodeError):
        topics = {}
    if not topics and not args.dry_run:
        print("log_topics.json has no thread-ids for this chat — start the "
              "bot once so it creates the topics, then re-run.")
        return 1

    # --all-requests: faithful per-request replay, all into 📥 Generations.
    if args.all_requests:
        reqs = compose_request_messages()
        # default to a gentler throttle for the bigger burst unless overridden
        throttle = args.throttle if args.throttle != 1.5 else 3.0
        print(f"{'DRY-RUN: ' if args.dry_run else ''}posting {len(reqs)} "
              f"per-request messages → generations (throttle {throttle}s)\n")
        sent = 0
        thread = topics.get("generations")
        for text in reqs:
            ok = send(token, chat_id, thread, text, args.dry_run, throttle)
            sent += 1 if ok else 0
            print(f"  {'(dry)' if args.dry_run else ('ok' if ok else 'FAIL')}"
                  f" — {text.splitlines()[1][:60]}")
        print(f"\n{'would post' if args.dry_run else 'posted'} {sent}/{len(reqs)}")
        if not args.dry_run and sent:
            REQ_MARKER.write_text(
                json.dumps({"done_ts": time.time(), "messages": sent}) + "\n"
            )
            print(f"wrote {REQ_MARKER.name}")
        return 0

    msgs = compose_messages()
    total = sum(len(v) for v in msgs.values())
    print(f"{'DRY-RUN: ' if args.dry_run else ''}posting {total} messages "
          f"across {len(msgs)} topics (throttle {args.throttle}s)\n")

    sent = 0
    for key in POST_ORDER:
        for text in msgs.get(key, []):
            thread = topics.get(key)
            ok = send(token, chat_id, thread, text, args.dry_run, args.throttle)
            sent += 1 if ok else 0
            print(f"  [{key}] {'(dry)' if args.dry_run else ('ok' if ok else 'FAIL')}"
                  f" — {text.splitlines()[0][:60]}")

    print(f"\n{'would post' if args.dry_run else 'posted'} {sent}/{total}")
    if not args.dry_run and sent:
        MARKER.write_text(
            json.dumps({"done_ts": time.time(), "messages": sent}) + "\n"
        )
        print(f"wrote {MARKER.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
