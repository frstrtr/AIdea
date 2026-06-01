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
IDEAS_MARKER = HERE / "log_backfill_ideas_done.json"
IDEAS_COLOR = 0xFFD67E  # must match bot._LOG_TOPIC_DEFS["ideas"]

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
_DRY_ID = [1000]  # fake, increasing message_ids for --dry-run reply threading


def send(token, chat_id, thread_id, text, dry, throttle, reply_to=None):
    """Post one message. Returns the sent message_id (int) on success, or None
    on failure. In dry-run returns a fake increasing id so reply threading can
    still be exercised."""
    if dry:
        tag = f" ↳reply {reply_to}" if reply_to else ""
        print(f"\n──[ thread {thread_id}{tag} ]──\n{text}")
        _DRY_ID[0] += 1
        return _DRY_ID[0]
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if thread_id is not None:
        payload["message_thread_id"] = thread_id
    if reply_to is not None:
        payload["reply_to_message_id"] = reply_to
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
                time.sleep(throttle)
                return body.get("result", {}).get("message_id")
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
            return None
        except Exception as e:
            print(f"  send error: {e}")
            return None
    return None


def create_topic(token, chat_id, name, color):
    """Create a forum topic; return its message_thread_id (or None)."""
    payload = {"chat_id": chat_id, "name": name, "icon_color": color}
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/createForumTopic",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            return body.get("result", {}).get("message_thread_id")
    except Exception as e:
        print(f"  create_topic failed: {e}")
        return None


def split_text(text, limit=4000):
    """Split on paragraph / line boundaries under the Telegram cap."""
    text = (text or "").strip()
    if not text:
        return ["(empty)"]
    chunks, rem = [], text
    while len(rem) > limit:
        cut = max(rem[:limit].rfind("\n\n"), rem[:limit].rfind("\n"))
        if cut < limit // 2:
            cut = limit
        chunks.append(rem[:cut].rstrip())
        rem = rem[cut:].lstrip()
    if rem:
        chunks.append(rem)
    return chunks


def idea_pairs():
    """(query_text, final_idea_or_None) per telegram request, oldest first."""
    tx = loadl(TRANSCRIPTS)
    by_run = collections.defaultdict(list)
    for r in tx:
        by_run[r.get("run_id")].append(r)
    started = sorted(
        (r for r in tx if r.get("kind") == "request_started"
         and str(r.get("source", "")).startswith("telegram")),
        key=lambda r: r.get("ts", 0),
    )
    pairs = []
    for r in started:
        ev = by_run.get(r.get("run_id"), [])
        refined = [x for x in ev if x.get("kind") == "refined" and x.get("text")]
        ideas = [x for x in ev if x.get("kind") == "idea" and x.get("text")]
        winners = [x for x in ev if x.get("kind") == "winner"]
        final = None
        if refined:
            final = refined[-1]["text"]
        elif winners and ideas:
            i = winners[-1].get("i")
            m = [x for x in ideas if x.get("i") == i]
            final = (m[0] if m else ideas[0])["text"]
        elif ideas:
            final = ideas[0]["text"]
        chat = r.get("chat_id") or str(r.get("source", "")).replace("telegram-", "")
        topic = (r.get("topic") or "(no topic)")[:1500]
        q = (f"📥 chat {chat} · {day(r.get('ts', 0))}\n"
             f"topic: {topic}\nmode: {r.get('mode', 'default')}")
        pairs.append((q, final))
    return pairs


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
    ap.add_argument(
        "--ideas", action="store_true",
        help="post each query into 💡 Ideas with its final idea threaded as a "
             "reply",
    )
    args = ap.parse_args()

    marker = (IDEAS_MARKER if args.ideas
              else REQ_MARKER if args.all_requests else MARKER)
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

    # --ideas: query → final-idea reply pairs into 💡 Ideas.
    if args.ideas:
        thread = topics.get("ideas")
        if thread is None and not args.dry_run:
            print("💡 Ideas topic missing — creating it…")
            thread = create_topic(token, chat_id, "💡 Ideas", IDEAS_COLOR)
            if thread is None:
                print("could not create the Ideas topic — aborting.")
                return 1
            # persist so the live bot reuses the same thread
            try:
                allc = json.loads(LOG_TOPICS.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                allc = {}
            allc.setdefault(str(chat_id), {})["ideas"] = thread
            LOG_TOPICS.write_text(json.dumps(allc, ensure_ascii=False, indent=2))
            print(f"created 💡 Ideas (thread {thread}), cached in log_topics.json")

        pairs = idea_pairs()
        throttle = args.throttle if args.throttle != 1.5 else 3.0
        with_idea = sum(1 for _q, i in pairs if i)
        print(f"{'DRY-RUN: ' if args.dry_run else ''}{len(pairs)} queries "
              f"({with_idea} with a recoverable idea) → 💡 Ideas "
              f"(throttle {throttle}s)\n")
        posted = 0
        for q, idea in pairs:
            qid = send(token, chat_id, thread, q, args.dry_run, throttle)
            posted += 1 if qid else 0
            if not qid:
                print(f"  query FAIL — {q.splitlines()[0][:50]}")
                continue
            if idea:
                chunks = split_text(idea)
                for n, ch in enumerate(chunks):
                    head = "💡 " if n == 0 else ""
                    tail = f"  ({n+1}/{len(chunks)})" if len(chunks) > 1 else ""
                    if send(token, chat_id, thread, head + ch + tail,
                            args.dry_run, throttle, reply_to=qid):
                        posted += 1
            else:
                if send(token, chat_id, thread,
                        "💡 (no idea recorded — run errored or was cancelled)",
                        args.dry_run, throttle, reply_to=qid):
                    posted += 1
            print(f"  {'(dry)' if args.dry_run else 'ok'} — "
                  f"{q.splitlines()[1][:50]}")
        print(f"\n{'would post' if args.dry_run else 'posted'} {posted} messages")
        if not args.dry_run and posted:
            IDEAS_MARKER.write_text(
                json.dumps({"done_ts": time.time(), "messages": posted}) + "\n"
            )
            print(f"wrote {IDEAS_MARKER.name}")
        return 0

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
