"""Telegram bot for AIdea.

Wires the same pipeline as the CLI and web UI to a Telegram chat:

  /idea     <topic>     default mode (one idea at the request entropy)
  /einstein <topic>     four mechanism-specific ideas (Adjacent Possible,
                        Exaptation, Slow Hunch, Productive Error)
  /lsd      <topic>     prior dissolution — predictive-processing framing
  /futures  <topic>     temporal projection (+1y / +3y / +10y / +30y)
  /usage                local LLM usage summary + observed subscription
                        window state
  /settings             show + tune entropy / cards / card-depth / etc.
  /help                 command list
  /cancel               abort an in-flight task in this chat

Progress updates are delivered by editing a single "working..." message
every ~1.5s so chats stay readable. Long ideas are split across multiple
messages on paragraph boundaries (Telegram caps at 4096 chars/message).

Auth: reads TELEGRAM_BOT_TOKEN from the environment. The bot inherits
its model access from whatever the agent CLI is already authenticated
against — same auth model as the web app.
"""

from __future__ import annotations

import asyncio
import json
import logging
import html
import json
import os
import random
import time
from datetime import time as dtime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from the project directory before anything else reads env vars.
load_dotenv(Path(__file__).parent / ".env")

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import NetworkError, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from aidea import (
    CARD_DEPTH_BY_NAME,
    DREAM_LABEL,
    EINSTEIN_MECHANISMS,
    FUTURES_HORIZONS,
    LSD_LABEL,
    LSD_VALIDATION_LABEL,
    LUCID_LABEL,
    build_dream_prompt,
    build_einstein_prompt,
    build_futures_prompt,
    build_lsd_prompt,
    lsd_validate,
    build_lucid_prompt,
    build_prompt,
    critic_score,
    load_or_generate_deck,
    parse_entropy,
    refine_idea,
    sample_cards,
    synthesize,
    total_score,
)
from transcripts import log_event as transcript_log, set_source
from usage import (
    start_run,
    summarize,
    summarize_for_chat,
    summarize_per_chat,
)
import quota


def _admin_chat_ids() -> set[int]:
    """Comma-separated AIDEA_ADMIN_CHAT_IDS env var → set of chat_ids.
    Empty / unset means no admins (everyone gets the user view)."""
    raw = os.environ.get("AIDEA_ADMIN_CHAT_IDS", "")
    out: set[int] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(int(tok))
        except ValueError:
            continue
    return out


def _is_admin(chat_id: int) -> bool:
    return chat_id in _admin_chat_ids()


# ---------------------------------------------------------------------------
# Activity-feed log group
#
# A monitoring Telegram supergroup (forum) the bot mirrors activity into:
# every idea generation (who / topic / mode / tokens / cost / quota), every
# subscription inquiry from a walled user, and pipeline errors. Set its id in
# AIDEA_LOG_CHAT_ID. The group must be a *forum* and the bot needs the
# "Manage Topics" right so it can file each kind of event under its own topic;
# if either is missing we silently fall back to the group's General topic.
# ---------------------------------------------------------------------------

_LOG_TOPICS_PATH = Path(__file__).parent / "log_topics.json"
# logical channel -> (topic title, icon color). Colors must be one of the six
# Telegram-permitted forum-topic colors. Insertion order is the order topics
# are pre-created at startup, so the group reads top-to-bottom as listed.
_LOG_TOPIC_DEFS: dict[str, tuple[str, int]] = {
    "new_users": ("🆕 New users", 0x6FB9F0),       # blue
    "generations": ("📥 Generations", 0x8EEE98),   # green
    "ideas": ("💡 Ideas", 0xFFD67E),               # yellow — query→idea pairs
    "inquiries": ("💌 Subscription inquiries", 0xFFD67E),  # yellow
    "quota_hits": ("🚧 Quota hits", 0xCB86DB),     # purple
    "errors": ("⚠️ Errors", 0xFB6F5F),             # red
    "summary": ("📊 Daily summary", 0xFF93B2),     # rose
}
# Serialize first-use topic creation so two concurrent events don't each
# create a duplicate forum topic for the same channel.
_LOG_TOPIC_LOCK = asyncio.Lock()


def _log_chat_id() -> int | None:
    raw = os.environ.get("AIDEA_LOG_CHAT_ID", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _load_log_topics() -> dict:
    try:
        with _LOG_TOPICS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_log_topics(data: dict) -> None:
    try:
        with _LOG_TOPICS_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        log.exception("could not persist log_topics.json")


async def _log_thread_id(context: Any, key: str) -> int | None:
    """message_thread_id for a logical log channel, creating the forum topic
    on first use and caching it. None means 'post to the General topic'
    (group isn't a forum, or the bot lacks Manage-Topics)."""
    chat_id = _log_chat_id()
    if chat_id is None:
        return None
    cache = _load_log_topics().get(str(chat_id), {})
    if key in cache:
        return cache[key]
    async with _LOG_TOPIC_LOCK:
        # Re-check under the lock — another coroutine may have just made it.
        cache = _load_log_topics().get(str(chat_id), {})
        if key in cache:
            return cache[key]
        title, color = _LOG_TOPIC_DEFS.get(key, (key.title(), 0x6FB9F0))
        try:
            topic = await context.bot.create_forum_topic(
                chat_id=chat_id, name=title, icon_color=color,
            )
            tid = int(topic.message_thread_id)
        except Exception:
            # Not a forum / no rights — caller posts to General.
            return None
        allcache = _load_log_topics()
        allcache.setdefault(str(chat_id), {})[key] = tid
        _save_log_topics(allcache)
        return tid


async def _log_to_group(
    context: Any, key: str, text: str, reply_to: int | None = None,
    parse_mode: str | None = None,
) -> int | None:
    """Mirror an event into the monitoring group. Best-effort: any failure is
    swallowed so logging can never break a user's pipeline. Plain text by
    default (no parse_mode) because user topics carry arbitrary characters;
    pass parse_mode=ParseMode.HTML for cards built with escaped fields. Returns
    the sent message_id (so callers can thread replies under it), or None on
    failure. Pass reply_to to post as a reply to an earlier message."""
    chat_id = _log_chat_id()
    if chat_id is None:
        return None
    try:
        thread_id = await _log_thread_id(context, key)
        kwargs: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if thread_id is not None:
            kwargs["message_thread_id"] = thread_id
        if reply_to is not None:
            kwargs["reply_to_message_id"] = reply_to
        if parse_mode is not None:
            kwargs["parse_mode"] = parse_mode
        msg = await context.bot.send_message(**kwargs)
        return msg.message_id if msg else None
    except Exception:
        log.exception("log-group post failed (key=%s)", key)
        return None


# lols spam/ban check — the @oLolsBot deeplink: opens the bot and checks the
# user-id passed via start. {id} is the Telegram user-id. Override via env.
_LOLS_URL = os.environ.get(
    "AIDEA_LOLS_URL", "https://t.me/oLolsBot?start={id}",
)


def _new_user_card(update: Update, topic: str) -> str:
    """Rich HTML profile card for the 🆕 New users topic: clickable id-based
    profile mention (name+surname), the raw fields, Android/iOS deeplinks, and
    a lols spam-check link. All dynamic fields HTML-escaped."""
    u = update.effective_user
    if u is None:
        return f"🆕 New user\nfirst topic: {html.escape(topic[:300])}"
    uid = u.id
    full = html.escape(u.full_name or u.first_name or "user")
    first = html.escape(u.first_name or "—")
    last = html.escape(u.last_name or "—")
    esc_topic = html.escape(topic[:300])
    if u.username:
        un = html.escape(u.username)
        uname = f'<a href="https://t.me/{un}">@{un}</a>'
        via_username = f'  • via @{un}: https://t.me/{un}\n'
    else:
        uname = "— (no username)"
        via_username = ""
    lols = _LOLS_URL.format(id=uid)
    return (
        "🆕 <b>New user</b>\n"
        f'👤 <a href="tg://user?id={uid}">{full}</a>\n'
        f"   name: {first} · surname: {last}\n"
        f"📛 username: {uname}\n"
        f"🆔 id: <code>{uid}</code>\n"
        f"💬 first topic: {esc_topic}\n"
        "\n"
        "🔗 <b>Open profile</b>\n"
        f'  • Android/Desktop: <a href="tg://user?id={uid}">tg://user?id={uid}</a>\n'
        f"  • iOS: <code>tg://openmessage?user_id={uid}</code>\n"
        f"{via_username}"
        f'🛡 lols check: <a href="{lols}">@oLolsBot ?start={uid}</a>'
    )


def _user_label(update: Update) -> str:
    """Human-readable identifier for the log feed: 'Name @handle (id 123)'."""
    u = update.effective_user
    if u is None:
        return f"chat {update.effective_chat.id}"
    name = u.full_name or u.first_name or "user"
    handle = f" @{u.username}" if u.username else ""
    return f"{name}{handle} (id {u.id})"


async def ensure_log_topics(context: Any) -> None:
    """Pre-create every log topic at startup, in _LOG_TOPIC_DEFS order, so the
    monitoring group is laid out top-to-bottom as designed instead of in
    whatever order events happen to fire. No-op once they're all cached, and
    silently skipped if AIDEA_LOG_CHAT_ID is unset / the group isn't a forum."""
    if _log_chat_id() is None:
        return
    for key in _LOG_TOPIC_DEFS:
        await _log_thread_id(context, key)


async def daily_summary_callback(context: Any) -> None:
    """Once-a-day rollup posted to the 📊 Daily summary topic: how many users
    we've seen, how many are walled, total free generations consumed, plus
    token/cost figures from usage.jsonl."""
    if _log_chat_id() is None:
        return
    limit = quota.free_limit()
    table = quota.table()
    users = len(table)
    walled = sum(1 for r in table.values() if int(r.get("count", 0)) >= limit)
    used = sum(int(r.get("count", 0)) for r in table.values())
    subs = sum(1 for uid in table if quota.is_subscriber(uid))
    try:
        u = summarize(run_id=None)
        wk = u.get("last_7d", {}) or {}
        tot = u.get("total", {}) or {}
        cost_7d = wk.get("total_cost_usd", 0)
        out_7d = wk.get("output_tokens", 0)
        cost_all = tot.get("total_cost_usd", 0)
    except Exception:
        cost_7d = out_7d = cost_all = 0
    await _log_to_group(
        context, "summary",
        "📊 Daily summary\n"
        f"users seen: {users} · walled (≥{limit}): {walled} · "
        f"subscribers 💎: {subs}\n"
        f"free generations used: {used}\n"
        f"last 7d: {fmt_tokens(out_7d)} out · {fmt_usd(cost_7d)}\n"
        f"all-time cost: {fmt_usd(cost_all)}",
    )


def _sub_lapse_days() -> int:
    try:
        return max(1, int(os.environ.get("AIDEA_SUB_LAPSE_DAYS", "3") or 3))
    except ValueError:
        return 3


async def subscription_lapse_callback(context: Any) -> None:
    """Daily subscription watch → 💌 Inquiries topic. Flags subscribers whose
    plan lapses within AIDEA_SUB_LAPSE_DAYS (default 3) and any that expired in
    the last day, each with a ready-to-paste renew command. Stays silent when
    there's nothing to report."""
    if _log_chat_id() is None:
        return
    now = time.time()
    window = _sub_lapse_days() * 86400.0
    lapsing, expired = [], []
    for uid, rec in quota.table().items():
        try:
            until = float(rec.get("subscribed_until") or 0)
        except (TypeError, ValueError):
            continue
        if until <= 0:
            continue
        delta = until - now
        who = rec.get("name") or rec.get("username") or uid
        if 0 <= delta <= window:
            lapsing.append((delta, f"• {who} ({uid}) — {delta / 86400:.1f}d "
                                   f"left  →  /subscribe {uid} 30"))
        elif -86400 <= delta < 0:  # expired within the last day
            expired.append(f"• {who} ({uid}) — EXPIRED  →  /subscribe {uid} 30")
    if not lapsing and not expired:
        return
    lines = ["⏳ Subscription watch"]
    lines += [t for _, t in sorted(lapsing)]
    lines += expired
    await _log_to_group(context, "inquiries", "\n".join(lines))


def _format_user_usage(chat_id: int) -> str:
    """Compact per-user view — what the user sees in /usage."""
    u = summarize_for_chat(chat_id)
    tot = u.get("total", {}) or {}
    week = u.get("last_7d", {}) or {}
    requests = int(tot.get("requests", 0) or 0)
    if requests == 0:
        return (
            "📈 *Your usage*\n\n"
            "No completed runs yet from this chat. Send a topic and I'll "
            "start tracking your requests / tokens / cost."
        )
    return (
        "📈 *Your usage*\n"
        f"  last 7 days: {int(week.get('requests', 0) or 0)} requests · "
        f"{int(week.get('calls', 0) or 0)} LLM calls · "
        f"{fmt_usd(week.get('total_cost_usd', 0))}\n"
        f"  all time:    {requests} requests · "
        f"{int(tot.get('calls', 0) or 0)} LLM calls · "
        f"{fmt_tokens(tot.get('output_tokens', 0))} out · "
        f"{fmt_usd(tot.get('total_cost_usd', 0))}\n\n"
        "_Only your own runs are counted. One request = one user message; "
        "each generates 7–10 LLM calls (deck + ideas + scoring)._"
    )


def _format_admin_usage() -> str:
    """Admin view — globals plus top users by cost. Shows both 'requests'
    (unique run_ids = user-visible asks) and 'calls' (LLM API invocations
    = ~7-10 per request) so the admin sees both real demand and total
    subscription burn."""
    u = summarize(run_id=None)
    tot = u.get("total", {}) or {}
    week = u.get("last_7d", {}) or {}
    per_chat = summarize_per_chat()
    lines = [
        "📈 *Usage — global (admin)*",
        f"  last 7 days: {int(week.get('requests', 0) or 0)} requests · "
        f"{int(week.get('calls', 0) or 0)} LLM calls · "
        f"{fmt_usd(week.get('total_cost_usd', 0))}",
        f"  all time:    {int(tot.get('requests', 0) or 0)} requests · "
        f"{int(tot.get('calls', 0) or 0)} LLM calls · "
        f"{fmt_tokens(tot.get('output_tokens', 0))} out · "
        f"{fmt_usd(tot.get('total_cost_usd', 0))}",
        "",
        "*Top chats by cost (all-time)*",
        "  `      chat_id   reqs  calls    cost`",
    ]
    for row in per_chat[:15]:
        t = row["totals"]
        reqs = int(t.get("requests", 0) or 0)
        calls = int(t.get("calls", 0) or 0)
        cost = fmt_usd(t.get("total_cost_usd", 0))
        lines.append(
            f"  `{row['chat_id']:>14}  {reqs:>4}  {calls:>5}  {cost:>7}`"
        )
    if not per_chat:
        lines.append("  (no records yet)")
    rl = u.get("rate_limit")
    if rl and rl.get("resets_at"):
        dt = rl["resets_at"] - time.time()
        if dt > 0:
            h, m = int(dt // 3600), int((dt % 3600) // 60)
            lines.append("")
            lines.append(
                f"Subscription window: status={rl.get('status', '?')}, "
                f"resets in {h}h {m}m"
            )
    return "\n".join(lines)

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    level=logging.INFO,
)
# httpx logs every request URL at INFO, which includes the bot token in the
# Telegram API path. Silence it (and the lower-level httpcore) so the token
# never reaches the journal.
for _noisy in ("httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger("aidea.bot")


# ---------------------------------------------------------------------------
# Per-chat settings + cancellation
# ---------------------------------------------------------------------------


@dataclass
class ChatSettings:
    # The mode plain-text messages (and the /menu Yes-confirm flow) run.
    # /einstein /lsd /futures /dream /lucid slash-commands still bypass this
    # and use their own mode regardless. Lucid needs a prior so it's NOT
    # available via menu — only via the slash command.
    mode: str = "default"
    entropy: str = "wild"
    cards: int = 30
    card_depth: str = "medium"
    n_concepts: int = 3
    n_ideas: int = 3
    model: str = "claude-opus-4-7"
    refine: bool = False
    evolve_deck: bool = False
    seed: int | None = None


@dataclass
class ChatState:
    settings: ChatSettings = field(default_factory=ChatSettings)
    busy: bool = False
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    last_run_id: str | None = None  # target of /feedback when no id is passed
    bootstrap_notice_sent: bool = False  # one-shot upfront notice during bootstrap
    # Wall-clock seconds (time.time()) when the current run started. Used to
    # show "elapsed Xs, ~Ys to go" in the busy-rejection message so users
    # know the bot isn't stuck.
    run_started_at: float = 0.0
    run_eta: float = 0.0  # the ETA we promised when the run started
    # Topic awaiting yes/no confirmation from the user before we burn an
    # idea-generation pipeline on it. Plain-text messages don't kick off
    # work immediately — they sit here until the user clicks the inline
    # "Yes, generate" button. Cleared on Yes (after launch) or No.
    pending_topic: str | None = None
    # Set when the user taps the 🍵 Brew button on the bottom keyboard:
    # the next plain-text message becomes the pending_topic, the user
    # gets the same Yes/No confirmation card as the regular flow, and
    # pending_brew tells the confirm callback to render a Pillow card
    # at the end of the pipeline. Cleared on submit or cancel.
    awaiting_brew_topic: bool = False
    pending_brew: bool = False
    # Set once a user crosses their free-generation limit: their subsequent
    # plain messages are captured as subscription inquiries (forwarded to the
    # log group) rather than echoed back with a Yes/No card. Cleared when an
    # admin resets their quota via /resetquota.
    awaiting_inquiry: bool = False


# chat_id -> ChatState
_STATES: dict[int, ChatState] = {}


def state_for(chat_id: int) -> ChatState:
    s = _STATES.get(chat_id)
    if s is None:
        s = ChatState()
        _STATES[chat_id] = s
    return s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


TELEGRAM_LIMIT = 4000  # safe margin under the 4096-char Telegram cap


def split_for_telegram(text: str) -> list[str]:
    """Split a long message at paragraph / line boundaries under the cap."""
    text = text.strip()
    if not text:
        return [""]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > TELEGRAM_LIMIT:
        slice_ = remaining[:TELEGRAM_LIMIT]
        # Prefer last \n\n, then last \n, then last space
        cut = max(
            slice_.rfind("\n\n"),
            slice_.rfind("\n"),
            slice_.rfind(". "),
        )
        if cut < TELEGRAM_LIMIT // 2:
            cut = TELEGRAM_LIMIT
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def fmt_ms(n: int) -> str:
    if not n:
        return "0s"
    if n < 1000:
        return f"{n}ms"
    if n < 60_000:
        return f"{n/1000:.1f}s"
    return f"{n // 60_000}m {round((n % 60_000) / 1000)}s"


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def fmt_usd(n: float) -> str:
    if not n:
        return "$0.00"
    if n < 0.01:
        return f"${n:.4f}"
    return f"${n:.2f}"


# Fallback ETAs (seconds) when transcripts.jsonl has no matching history yet.
# Rough order-of-magnitude — refined as soon as a few real runs of that mode
# land in the history.
_ETA_FALLBACK: dict[str, float] = {
    "default": 180.0,  # 3 syntheses + 3 critic passes + deck
    "einstein": 220.0,  # 4 mechanism passes + 4 critic + deck
    "lsd": 200.0,  # 3 × (anarchic + sober) + 3 critic + deck
    "futures": 220.0,  # 4 horizon passes + 4 critic + deck
    "dream": 110.0,  # 3 dreams, no critic (feasibility-off mode)
    "lucid": 110.0,  # 3 lucid dreams, no critic
}


def estimate_runtime_seconds(mode: str, refine: bool = False) -> float:
    """Heuristic ETA for a request in ``mode`` based on recent transcripts.

    Pairs (request_started, request_completed) by run_id over the last ~2000
    transcript lines (cheap even for a long history), filters by matching
    mode + refine flag, averages the last 10 durations. Falls back to a
    hard-coded table when there is no matching history yet.
    """
    try:
        from pathlib import Path
        path = Path("transcripts.jsonl")
        if not path.exists():
            raise FileNotFoundError
        with path.open(encoding="utf-8") as f:
            lines = f.readlines()[-2000:]
    except (OSError, FileNotFoundError):
        base = _ETA_FALLBACK.get(mode, 45.0)
        return base * (1.6 if refine else 1.0)

    starts: dict[str, tuple[float, str, bool]] = {}
    durations: list[float] = []
    for line in lines:
        try:
            ev = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        rid = ev.get("run_id")
        if not rid:
            continue
        k = ev.get("kind")
        if k == "request_started":
            starts[rid] = (
                float(ev.get("ts", 0) or 0),
                str(ev.get("mode", "default")),
                bool(ev.get("refine")),
            )
        elif k == "request_completed" and rid in starts:
            ts_start, m, r = starts.pop(rid)
            if m == mode and r == refine and ts_start > 0:
                d = float(ev.get("ts", 0) or 0) - ts_start
                if 1.0 < d < 600.0:  # ignore obviously broken entries
                    durations.append(d)

    if not durations:
        base = _ETA_FALLBACK.get(mode, 45.0)
        return base * (1.6 if refine else 1.0)
    recent = durations[-10:]
    return sum(recent) / len(recent)


async def _send_with_retry(
    bot: Any,
    *,
    chat_id: int,
    text: str,
    attempts: int = 2,
    **kwargs: Any,
) -> None:
    """bot.send_message wrapped with a single retry on TimedOut / NetworkError.

    Telegram occasionally returns 'Timed out' on a perfectly good request when
    the bot is sending bursts (e.g. settings-update confirmation + keyboard
    refresh + idea content all back-to-back). The default read_timeout is 5s
    and a single sub-second blip from api.telegram.org surfaces as a cryptic
    'usage error: Timed out' to the user. One retry with a short backoff
    masks the blip without bloating the call site."""
    last_err: BaseException | None = None
    for attempt in range(attempts):
        try:
            await bot.send_message(chat_id=chat_id, text=text, **kwargs)
            return
        except (TimedOut, NetworkError) as e:
            last_err = e
            if attempt + 1 >= attempts:
                break
            await asyncio.sleep(0.8)
    if last_err is not None:
        raise last_err


def fmt_eta(seconds: float) -> str:
    """Human-friendly ETA. Sub-minute → 'Xs'; otherwise 'Xm Ys'."""
    s = max(1, int(round(seconds)))
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60:02d}s"


# ---------------------------------------------------------------------------
# Wrapping an async LLM call with a Telegram "editing message" progress UI
# ---------------------------------------------------------------------------


async def run_with_progress(
    coro,
    *,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    cancel: asyncio.Event,
    headline: str,
) -> Any:
    """Drive an awaitable, editing the user-visible status message every ~1.5s
    with elapsed time. Aborts the task if cancel fires."""
    chat_id = update.effective_chat.id
    status = await context.bot.send_message(
        chat_id=chat_id, text=f"⏳ {headline} (0s)",
    )
    task = asyncio.create_task(coro)
    start = time.monotonic()
    last_edit = 0.0
    try:
        while not task.done():
            done, _ = await asyncio.wait(
                [task],
                timeout=1.5,
                return_when=asyncio.FIRST_COMPLETED,
            )
            now = time.monotonic()
            elapsed = now - start
            if cancel.is_set():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                await status.edit_text(f"🛑 cancelled ({elapsed:.1f}s) — {headline}")
                raise asyncio.CancelledError()
            if not done and now - last_edit > 1.2:
                last_edit = now
                try:
                    await status.edit_text(f"⏳ {headline} ({elapsed:.1f}s)")
                except Exception:
                    pass  # ignore "message not modified" / network blips
        result = task.result()
        elapsed = time.monotonic() - start
        try:
            await status.edit_text(f"✅ {headline} done in {elapsed:.1f}s")
        except Exception:
            pass
        return result
    except asyncio.CancelledError:
        raise


async def send_idea(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    header: str,
    body: str,
) -> None:
    """Send an idea as one or more Telegram messages, with a header line."""
    chat_id = update.effective_chat.id
    full = f"*{header}*\n\n{body}".strip()
    for chunk in split_for_telegram(full):
        await context.bot.send_message(
            chat_id=chat_id,
            text=chunk,
            disable_web_page_preview=True,
        )


# ---------------------------------------------------------------------------
# Pipeline runner — one shape used by /idea, /einstein, /lsd, /futures
# ---------------------------------------------------------------------------


async def run_pipeline_for_telegram(
    *,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    topic: str,
    mode: str,  # 'default' | 'einstein' | 'lsd' | 'futures' | 'dream' | 'lucid'
    extra: dict | None = None,
) -> None:
    chat_id = update.effective_chat.id
    state = state_for(chat_id)

    # Free-tier quota. Counts interactive idea generations per Telegram
    # *user* (not chat). Brews are free, admins are exempt — those bypass
    # both the wall and the counter. Scheduled brews never reach this
    # function (separate worker), so they're free automatically.
    tg_user = update.effective_user
    user_id = tg_user.id if tg_user is not None else chat_id
    is_brew = bool(extra and extra.get("brew_render"))
    # Interactive brews count too — they run the same (costlier) pipeline.
    # Admins and paid subscribers are exempt from the free-tier wall.
    # Scheduled brews never reach this function (separate worker), so they
    # stay free automatically.
    is_sub = quota.is_subscriber(user_id)
    quota_exempt = _is_admin(user_id) or _is_admin(chat_id) or is_sub
    free_limit = quota.free_limit()
    if not quota_exempt and quota.is_over(user_id, free_limit):
        await context.bot.send_message(
            chat_id=chat_id,
            text=_quota_prompt(_looks_russian(topic), free_limit),
        )
        # Capture whatever they send next as a subscription inquiry.
        state.awaiting_inquiry = True
        return

    if state.busy:
        elapsed = max(0.0, time.time() - (state.run_started_at or time.time()))
        remaining = max(1.0, (state.run_eta or 45.0) - elapsed)
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"⏳ Your previous request is still running "
                f"(elapsed {fmt_eta(elapsed)}, ~{fmt_eta(remaining)} to go). "
                "Send /cancel to abort, or wait."
            ),
        )
        return

    state.busy = True
    state.cancel.clear()
    state.run_started_at = time.time()
    run_id = start_run(f"tg-{chat_id}")
    state.last_run_id = run_id  # /feedback targets this run by default
    set_source(f"telegram-{chat_id}")

    # Count this generation against the user's free tier (brews / admins are
    # exempt). Refunded in the error / cancel paths so a failed run is free.
    quota_count: int | None = None
    if not quota_exempt:
        quota_count = quota.increment(
            user_id,
            name=(tg_user.full_name if tg_user else ""),
            username=(tg_user.username if tg_user else ""),
        )
    try:
        from rag import note_query, bootstrap_state, bootstrap_notice_text
        bs = note_query(f"telegram-{chat_id}")
        if bs.get("active") and not state.bootstrap_notice_sent:
            notice = bootstrap_notice_text()
            if notice:
                await context.bot.send_message(chat_id=chat_id, text=notice)
            state.bootstrap_notice_sent = True
    except Exception:
        pass
    s = state.settings
    spread, level = parse_entropy(s.entropy)
    depth = CARD_DEPTH_BY_NAME[s.card_depth]

    transcript_log(
        "request_started",
        topic=topic, mode=mode,
        chat_id=chat_id,
        entropy=s.entropy, cards=s.cards, card_depth=s.card_depth,
        n_concepts=s.n_concepts, n_ideas=s.n_ideas,
        seed=s.seed, model=s.model,
        refine=bool(s.refine), evolve_deck=bool(s.evolve_deck),
    )

    eta = estimate_runtime_seconds(mode, refine=bool(s.refine))
    state.run_eta = eta
    await context.bot.send_message(
        chat_id=chat_id,
        text=(
            f"📥 Got it — estimated reply in ~{fmt_eta(eta)}.\n"
            f"Topic: {topic}\n"
            f"mode={mode} · entropy={level.name} ({spread:.2f}) · "
            f"deck={s.cards}@{depth.name} · "
            f"draw={s.n_concepts}/idea · refine={s.refine} · evolve={s.evolve_deck}"
        ),
    )

    try:
        # Stage 0: deck
        deck = await run_with_progress(
            load_or_generate_deck(
                topic=topic,
                n=s.cards,
                depth=depth,
                model=s.model,
                force_regen=False,
                verbose=False,
            ),
            update=update, context=context, cancel=state.cancel,
            headline="generating donor deck",
        )
        transcript_log(
            "deck",
            size=len(deck), depth=depth.name,
            cards=[{k: v for k, v in c.__dict__.items() if v is not None} for c in deck],
            source_kind="generated",
        )

        # Decide passes
        lucid_prior = (extra or {}).get("lucid_prior", "")
        if mode == "einstein":
            passes = [
                (("einstein", k), m["label"]) for k, m in EINSTEIN_MECHANISMS.items()
            ]
        elif mode == "lsd":
            passes = [(("lsd", None), LSD_LABEL) for _ in range(s.n_ideas)]
        elif mode == "futures":
            passes = [
                (("futures", h["key"]), h["label"]) for h in FUTURES_HORIZONS
            ]
        elif mode == "dream":
            passes = [(("dream", None), DREAM_LABEL) for _ in range(s.n_ideas)]
        elif mode == "lucid":
            passes = [
                (("lucid", lucid_prior), LUCID_LABEL) for _ in range(s.n_ideas)
            ]
        else:
            passes = [((None, None), None) for _ in range(s.n_ideas)]

        rng = random.Random(s.seed)
        ideas: list[str] = []
        cards_per_idea: list[list] = []
        mech_labels: list[str | None] = []

        for i, ((m, mk), label) in enumerate(passes):
            drawn = sample_cards(deck=deck, n=s.n_concepts, spread=spread, rng=rng)
            if m == "einstein":
                prompt = build_einstein_prompt(topic, drawn, mk)
            elif m == "lsd":
                prompt = build_lsd_prompt(topic, drawn)
            elif m == "futures":
                prompt = build_futures_prompt(topic, drawn, mk)
            elif m == "dream":
                prompt = build_dream_prompt(topic, drawn)
            elif m == "lucid":
                prompt = build_lucid_prompt(topic, drawn, mk or "")
            else:
                prompt = build_prompt(topic, drawn, level)

            label_for_user = label or f"idea {i + 1}"
            idea = await run_with_progress(
                synthesize(prompt=prompt, model=s.model, stream_to_stdout=False),
                update=update, context=context, cancel=state.cancel,
                headline=f"synthesizing {label_for_user}",
            )
            transcript_log(
                "idea", i=i, mechanism=label, text=idea,
                cards=[{k: v for k, v in c.__dict__.items() if v is not None} for c in drawn],
            )

            header = (
                f"Idea {i + 1}" +
                (f" — {label}" if label else "")
            )
            await send_idea(update, context, header, idea)

            if m == "lsd":
                # Sober validation pass: priors back online, separate
                # insight from hallucination. The sober version is what
                # we keep for downstream critic / refine.
                sober = await run_with_progress(
                    lsd_validate(topic, idea, s.model),
                    update=update, context=context, cancel=state.cancel,
                    headline=f"sober validation {i + 1}",
                )
                transcript_log(
                    "lsd_validation", i=i, anarchic=idea, sober=sober,
                )
                await send_idea(
                    update, context,
                    f"Sober validation {i + 1} (priors online)",
                    sober,
                )
                ideas.append(sober)
                cards_per_idea.append(drawn)
                mech_labels.append(LSD_VALIDATION_LABEL)
            else:
                ideas.append(idea)
                cards_per_idea.append(drawn)
                mech_labels.append(label)

        # Stage 3: critic — score every idea by default so the user sees a
        # 4-axis breakdown (feasibility / unexpectedness / uniqueness /
        # topic_fit) per idea. Skipped only for feasibility-off modes
        # (dream / lucid) where the axes don't map cleanly.
        score_enabled = mode not in ("dream", "lucid")
        scored: list[dict] = []
        if score_enabled and ideas:
            for i, idea in enumerate(ideas):
                score = await run_with_progress(
                    # Panel for the refine path (better winner selection) and
                    # for paid subscribers on every generation; single critic
                    # otherwise. Global AIDEA_CRITIC_PANEL still forces it on.
                    critic_score(
                        topic, idea, s.model,
                        force_panel=bool(s.refine) or is_sub,
                    ),
                    update=update, context=context, cancel=state.cancel,
                    headline=f"scoring idea {i + 1}",
                )
                score["i"] = i
                score["total"] = total_score(score)
                scored.append(score)
                transcript_log(
                    "score", i=i,
                    mechanism=mech_labels[i] if i < len(mech_labels) else None,
                    feasibility=score.get("feasibility"),
                    unexpectedness=score.get("unexpectedness"),
                    uniqueness=score.get("uniqueness"),
                    topic_fit=score.get("topic_fit"),
                    total=score.get("total"),
                    notes=score.get("notes", ""),
                )
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"Score idea {i + 1}"
                        + (f" [{mech_labels[i]}]" if mech_labels[i] else "")
                        + f": feasibility={score['feasibility']}, "
                        f"unexpectedness={score['unexpectedness']}, "
                        f"uniqueness={score['uniqueness']}, "
                        f"topic-fit={score['topic_fit']}, "
                        f"total={score['total']}/400\n"
                        f"notes: {score.get('notes', '')}"
                    ),
                )

        # Record the best-scoring idea as a RAG winner regardless of refine
        # flag — the score data is already paid for; capturing it costs
        # nothing and keeps the corpus self-evolving from every run.
        if scored:
            winner = max(scored, key=lambda x: x["total"])
            wmech = mech_labels[winner["i"]] if mech_labels[winner["i"]] else ""
            transcript_log(
                "winner",
                i=winner["i"], total=winner["total"],
                mechanism=mech_labels[winner["i"]],
                notes=winner.get("notes", ""),
            )
            try:
                from rag import record_winner as _rag_record_winner
                _rag_record_winner(
                    run_id=run_id,
                    winning_card_names=[
                        c.name for c in cards_per_idea[winner["i"]]
                    ],
                    critic_total=int(winner["total"]),
                    source=f"telegram-{chat_id}",
                )
            except Exception:
                pass

            if s.refine:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"🏆 Best so far: idea {winner['i'] + 1}"
                        + (f" [{wmech}]" if wmech else "")
                        + f" — {winner['total']}/400. Refining at low entropy..."
                    ),
                )
                refined = await run_with_progress(
                    refine_idea(
                        topic=topic,
                        idea=ideas[winner["i"]],
                        notes=winner.get("notes", ""),
                        model=s.model,
                    ),
                    update=update, context=context, cancel=state.cancel,
                    headline="refining winner",
                )
                transcript_log("refined", i=winner["i"], text=refined)
                await send_idea(update, context, "Refined winner", refined)

        # Brew rendering hook — when extra.brew_render is set (cmd_brew /
        # 🍵 Brew button path), pick the best idea (refined winner if
        # available, else max-scored, else first), parse it into title /
        # pitch / mechanism / first_step, render a 1080×1920 Pillow card,
        # and deliver as a shareable photo. The text version of the idea
        # has already been sent earlier in the loop, so this is purely an
        # additive visual artifact for Stories sharing.
        if extra and extra.get("brew_render") and ideas:
            try:
                from brew_render import (
                    render_card,
                    render_card_with_image,
                    parse_idea_fields,
                    default_output_path,
                )
                import brew_image
                # Pick the best text: refined > critic-winner > first.
                refined_text = locals().get("refined")
                if refined_text:
                    best_text = refined_text
                elif scored:
                    best_text = ideas[winner["i"]]
                else:
                    best_text = ideas[0]
                fields = parse_idea_fields(best_text)
                png_path: Path | None = None
                # Illustrated path — Claude scene prompt + remote SDXL render
                # + composite. Any failure (GPU host down, model not loaded,
                # timeout) falls through to the existing text-only card so
                # the user still gets something shareable.
                if brew_image.is_enabled():
                    try:
                        img_path = default_output_path(
                            brew_id=f"{run_id}-img",
                        )
                        png_img, scene = await brew_image.illustrate_idea(
                            fields, img_path,
                        )
                        transcript_log("brew_image", scene=scene)
                        png_path = render_card_with_image(
                            image_path=png_img,
                            title=fields.get("title") or topic[:60],
                            pitch=fields.get("pitch", ""),
                            first_step=fields.get("first_step", ""),
                            output_path=default_output_path(brew_id=run_id),
                        )
                    except Exception:
                        log.exception(
                            "brew illustrated render failed — "
                            "falling back to text-only card",
                        )
                if png_path is None:
                    png_path = render_card(
                        title=fields.get("title") or topic[:60],
                        pitch=fields.get("pitch", ""),
                        mechanism=fields.get("mechanism", ""),
                        first_step=fields.get("first_step", ""),
                        output_path=default_output_path(brew_id=run_id),
                    )
                with png_path.open("rb") as f:
                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=f,
                        caption="☕ Your brew is ready — tap & hold the image to share to Stories.",
                    )
            except Exception as e:
                log.exception("brew render failed")
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"(brew render failed — text version above still works: {e})",
                )

        # Final usage summary
        u = summarize(run_id=run_id)
        run = u.get("this_run", {})
        text = (
            "Usage this run: "
            f"{fmt_tokens(run.get('input_tokens', 0))} in / "
            f"{fmt_tokens(run.get('output_tokens', 0))} out · "
            f"{fmt_ms(run.get('duration_ms', 0))} · "
            f"{fmt_usd(run.get('total_cost_usd', 0))}"
        )
        await context.bot.send_message(chat_id=chat_id, text=text)
        transcript_log(
            "request_completed",
            n_ideas=len(ideas), refined=bool(s.refine) and bool(ideas),
        )

        # Activity feed → monitoring group. Emitted only on success (the
        # error/cancel paths refund the slot, so new-user / quota-hit signals
        # below would be wrong if fired earlier).
        if quota_count is not None:
            quota_line = f"{quota_count}/{free_limit}" + (" · brew" if is_brew else "")
        elif is_sub:
            quota_line = "subscriber 💎"
        else:
            quota_line = "exempt (admin)"
        await _log_to_group(
            context, "generations",
            f"📥 {_user_label(update)}\n"
            f"topic: {topic}\n"
            f"mode: {mode} · quota: {quota_line}\n"
            f"tokens: {fmt_tokens(run.get('input_tokens', 0))} in / "
            f"{fmt_tokens(run.get('output_tokens', 0))} out · "
            f"{fmt_usd(run.get('total_cost_usd', 0))} · "
            f"{fmt_ms(run.get('duration_ms', 0))}",
        )
        # First-ever generation from this user → onboarding signal.
        if quota_count == 1:
            mid = await _log_to_group(
                context, "new_users", _new_user_card(update, topic),
                parse_mode=ParseMode.HTML,
            )
            if mid is None and _log_chat_id() is not None:
                # HTML may have been rejected — never silently drop the event.
                await _log_to_group(
                    context, "new_users",
                    f"🆕 {_user_label(update)}\nfirst topic: {topic}",
                )
        # Just consumed their last free generation → the wall is now up.
        if quota_count is not None and quota_count == free_limit:
            await _log_to_group(
                context, "quota_hits",
                f"🚧 {_user_label(update)} used their last free generation "
                f"({quota_count}/{free_limit}) — wall is now up.",
            )

        # 💡 Ideas topic: the query, with the final idea threaded as a reply.
        # Final = refined winner if refined, else the critic-winner, else the
        # first idea. Long ideas split across reply messages.
        final_idea = locals().get("refined")
        if not final_idea:
            if scored and ideas:
                final_idea = ideas[winner["i"]]
            elif ideas:
                final_idea = ideas[0]
        if final_idea:
            qid = await _log_to_group(
                context, "ideas",
                f"📥 {_user_label(update)}\ntopic: {topic}\nmode: {mode}",
            )
            if qid:
                chunks = split_for_telegram(final_idea)
                for n, ch in enumerate(chunks):
                    head = "💡 " if n == 0 else ""
                    tail = f"  ({n + 1}/{len(chunks)})" if len(chunks) > 1 else ""
                    await _log_to_group(
                        context, "ideas", head + ch + tail, reply_to=qid,
                    )

    except asyncio.CancelledError:
        transcript_log("request_errored", error="cancelled")
        # The run was aborted before delivering ideas — give the slot back.
        if quota_count is not None:
            quota.refund(user_id)
        return
    except Exception as e:
        log.exception("pipeline failed")
        transcript_log(
            "request_errored", error_type=type(e).__name__, error=str(e),
        )
        # Failed run shouldn't cost the user a free generation.
        if quota_count is not None:
            quota.refund(user_id)
        await _log_to_group(
            context, "errors",
            f"⚠️ {_user_label(update)}\n"
            f"topic: {topic}\nmode: {mode}\n"
            f"error: {type(e).__name__}: {e}",
        )
        # Localized, friendly message. The raw SDK string ("Claude Code
        # returned an error result: success", "Command failed with exit
        # code N") is meaningless to a user — show them something they
        # can act on instead. Topic-language detection reuses the same
        # helper as the greeting-filter welcome.
        if _looks_russian(topic):
            msg = (
                "⚠️ Не получилось завершить генерацию идей.\n\n"
                "Попробуйте, пожалуйста, ещё раз через минуту — иногда "
                "это временная ошибка фонового сервиса. Если не помогает, "
                "попробуйте переформулировать запрос (особенно если он про "
                "медицинскую/чувствительную тему — модель может отказывать "
                "в ответе)."
            )
        else:
            msg = (
                "⚠️ Couldn't finish generating ideas.\n\n"
                "Please try again in a minute — sometimes the upstream "
                "service has a transient error. If it keeps failing, try "
                "rephrasing the request (especially if it's medical / "
                "sensitive — the model can decline some topics)."
            )
        await context.bot.send_message(chat_id=chat_id, text=msg)
    finally:
        state.busy = False
        state.cancel.clear()
        state.run_started_at = 0.0
        state.run_eta = 0.0


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


HELP_TEXT = (
    "AIdea — applied-ideas synthesizer with entropy controls.\n\n"
    "Pipeline: topic → donor-deck (cards from many domains) → "
    "stochastic sample at entropy → synthesis → optional critic + refine.\n\n"
    "🧭 /menu — open the button-driven control panel (modes, entropy, depth, refine).\n"
    "Tip: just send a plain message — bot will echo it back and ask Yes/No "
    "before running the pipeline.\n\n"
    "Modes (pick one):\n"
    "/idea <topic>      default — n_ideas at the current entropy, feasibility "
    "required (or just send the topic without any command)\n"
    "/einstein <topic>  four mechanism-specific ideas:\n"
    "     Adjacent Possible — step through a just-unlocked door\n"
    "     Exaptation — transplant a mechanism from another field\n"
    "     Slow Hunch — articulate a latent tension in the field\n"
    "     Productive Error — invert a load-bearing assumption\n"
    "/lsd <topic>       Prior Dissolution — predictive-processing framing: "
    "suspend the field's interpretive prior and re-perceive\n"
    "/futures <topic>   Temporal projection (+1y / +3y / +10y / +30y). "
    "Identify what is obvious from each future, ship the v0.1 today\n"
    "/dream <topic>     Unconstrained dream — prediction-error OFFLINE, no "
    "feasibility check. Output is a dream image + post-wake interpretation\n"
    "/lucid <prior> | <topic>   Lucid dream — dream that resolves toward "
    "your injected prior. Example: /lucid energy-based | reducing churn\n\n"
    "Settings:\n"
    "/settings              show current entropy / deck / depth / refine knobs\n"
    "/set <k>=<v>           tune one knob (entropy, cards, card_depth, "
    "n_concepts, n_ideas, refine, evolve_deck, seed)\n"
    "/set themes=…          comma-separated donor domains (overrides "
    "auto-generation)\n"
    "/set theme_entropy=X   0..1 — how far the auto-themes wander\n\n"
    "Flags (set via /set):\n"
    "  refine=true       score every idea and refine the winner\n"
    "  evolve_deck=true  after refine, sharpen winning cards into the cache\n\n"
    "Other:\n"
    "/usage   local LLM-API usage + observed subscription-window state\n"
    "/cancel  abort the current task in this chat\n"
)


async def cmd_start(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """First-touch entry point. Send the full help text AND attach the
    persistent bottom button menu in one message so new users see every
    option immediately without typing /menu."""
    state = state_for(update.effective_chat.id)
    await update.message.reply_text(
        HELP_TEXT,
        reply_markup=_main_menu_kb(state.settings),
    )


async def cmd_help(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Same as /start — help text plus the bottom button menu, so the
    user always has the keyboard attached after asking for help."""
    state = state_for(update.effective_chat.id)
    await update.message.reply_text(
        HELP_TEXT,
        reply_markup=_main_menu_kb(state.settings),
    )


def _topic_from(update: Update) -> str:
    """Extract everything after the command word."""
    text = (update.message.text or "").strip()
    # Strip the slash-command + optional @botname
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


async def cmd_idea(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    topic = _topic_from(update)
    if not topic:
        await update.message.reply_text(
            "usage: /idea <one or two sentences describing what you're working on>",
        )
        return
    await run_pipeline_for_telegram(
        update=update, context=ctx, topic=topic, mode="default",
    )


async def cmd_brew(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Brew = pipeline + Pillow card. Same pipeline as /idea but the
    winning idea is also rendered as a 1080×1920 PNG and sent as a
    shareable photo (Stories-ready). Caller uses the chat's current
    mode setting — Default unless menu-picked otherwise."""
    topic = _topic_from(update)
    chat_id = update.effective_chat.id
    if not topic:
        await update.message.reply_text(
            "Send: `/brew <your topic>` — or tap 🍵 *Brew* on the menu "
            "and send your topic as the next message.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    state = state_for(chat_id)
    mode = state.settings.mode or "default"
    await run_pipeline_for_telegram(
        update=update, context=ctx, topic=topic, mode=mode,
        extra={"brew_render": True},
    )


async def cmd_einstein(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    topic = _topic_from(update)
    if not topic:
        await update.message.reply_text("usage: /einstein <topic>")
        return
    await run_pipeline_for_telegram(
        update=update, context=ctx, topic=topic, mode="einstein",
    )


async def cmd_lsd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    topic = _topic_from(update)
    if not topic:
        await update.message.reply_text("usage: /lsd <topic>")
        return
    await run_pipeline_for_telegram(
        update=update, context=ctx, topic=topic, mode="lsd",
    )


async def cmd_futures(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    topic = _topic_from(update)
    if not topic:
        await update.message.reply_text("usage: /futures <topic>")
        return
    await run_pipeline_for_telegram(
        update=update, context=ctx, topic=topic, mode="futures",
    )


async def cmd_dream(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    topic = _topic_from(update)
    if not topic:
        await update.message.reply_text(
            "usage: /dream <topic>\n"
            "Dreaming mode: prediction-error offline. The output ignores "
            "feasibility constraints and names what survives waking."
        )
        return
    await run_pipeline_for_telegram(
        update=update, context=ctx, topic=topic, mode="dream",
    )


async def cmd_lucid(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    # /lucid <prior> :: <topic>   — prior and topic separated by " :: "
    # If no separator is present, treat everything as topic with empty prior
    # (degrades to dream mode behavior).
    raw = _topic_from(update)
    if not raw:
        await update.message.reply_text(
            "usage:\n"
            "  /lucid <directional prior> :: <topic>\n"
            "Example:\n"
            "  /lucid the team must stay solo-founder-sized :: "
            "growing my consultancy without hiring\n\n"
            "The dream biases toward your injected prior; one reality "
            "check fires at the end."
        )
        return
    # Accept '::' or '|' as the prior/topic separator.
    sep = "::" if "::" in raw else ("|" if "|" in raw else None)
    if sep is not None:
        prior, _, topic = raw.partition(sep)
        prior = prior.strip()
        topic = topic.strip()
    else:
        prior, topic = "", raw
    if not topic:
        await update.message.reply_text(
            "usage: /lucid <prior> :: <topic> (topic was empty)",
        )
        return
    await run_pipeline_for_telegram(
        update=update, context=ctx, topic=topic, mode="lucid",
        extra={"lucid_prior": prior},
    )


MODES_TEXT = (
    "AIdea modes — each implements a named cognitive process from the literature.\n"
    "Pick at most one per request.\n\n"

    "/idea <topic>\n"
    "  Default. Applied conceptual blending — picks 1+ donor concepts and\n"
    "  finds the structural mechanism that transfers onto your topic.\n"
    "  Feasibility required.\n"
    "  Source: Fauconnier & Turner, The Way We Think (2002); Kauffman,\n"
    "  Investigations (2000), the 'adjacent possible' framing.\n\n"

    "/einstein <topic>\n"
    "  Four passes, one per generative mechanism Steven Johnson catalogues\n"
    "  in Where Good Ideas Come From (2010):\n"
    "    • Adjacent Possible — name a capability unlocked in last 1-3 years,\n"
    "      walk through that door (Kauffman 2000 / Johnson 2010)\n"
    "    • Exaptation — transplant a mechanism from a far-distant field\n"
    "      (Gould & Vrba, Paleobiology 1982; Gutenberg / wine press)\n"
    "    • Slow Hunch — articulate a latent tension (Johnson 2010)\n"
    "    • Productive Error — invert a load-bearing assumption\n"
    "      (Fleming 1928 / mutation-as-discovery)\n"
    "  Pair with /set refine=true to rank mechanisms and harden the winner.\n\n"

    "/lsd <topic>\n"
    "  TWO-PASS structure (most expensive mode):\n"
    "    Pass 1 — Anarchic generation: priors relaxed, error-detection OFF.\n"
    "      Flatten the hierarchy, treat 2-3 load-bearing assumptions as\n"
    "      noise, force cross-module connections between distant donors,\n"
    "      propose an 'uphill move' against the current optimization.\n"
    "    Pass 2 — Sober validation: priors back online. Separates the\n"
    "      structural insight from the hallucination, proposes a buildable\n"
    "      v0.1.\n"
    "  Source: Carhart-Harris & Friston, 'REBUS and the Anarchic Brain'\n"
    "  (Pharmacological Reviews 2019); Anil Seth, Being You (2021); Friston,\n"
    "  Free Energy Principle (Nature Reviews Neuroscience 2010). The two\n"
    "  passes mirror the REBUS pro-tip: anarchic brain generates, sober\n"
    "  brain validates 24h later.\n\n"

    "/futures <topic>\n"
    "  Four temporal horizons (+1y / +3y / +10y / +30y). At each: name three\n"
    "  concrete shifts that are likely by then, identify what's obvious from\n"
    "  there but invisible today, translate to a v0.1 you ship this year.\n"
    "  Source: Anil Seth, Being You (2021) — perceptual forward-modeling\n"
    "  compensates for the ~100ms delay between event and conscious\n"
    "  experience. Scenario planning lineage: Pierre Wack (Shell 1970s);\n"
    "  Stewart Brand, The Clock of the Long Now (1999).\n\n"

    "/dream <topic>\n"
    "  Prediction-error signal offline. Generative model runs free, NO\n"
    "  feasibility floor. Output: vivid dream image (may violate physics,\n"
    "  economics, regulation) + 'what survives waking' line naming the\n"
    "  salvageable fragment. For harvesting, not shipping.\n"
    "  Source: Friston, Free Energy Principle (2010) — dreams as complexity\n"
    "  reduction / synaptic garbage collection. Hobson & Friston (Progress\n"
    "  in Neurobiology 2012) on waking vs dreaming consciousness.\n\n"

    "/lucid <prior> :: <topic>\n"
    "  Lucid dream — dream-state generation + a directional prior YOU\n"
    "  inject. The hallucination biases toward your belief; one reality\n"
    "  check fires before waking. More salvage than pure dream because the\n"
    "  prior anchors the hallucination.\n"
    "  Example: /lucid solo-founder only :: how do I monetize my AI tool\n"
    "  Source: Stephen LaBerge, Exploring the World of Lucid Dreaming\n"
    "  (1990); Voss et al., 'Lucid dreaming: a state of consciousness with\n"
    "  features of both waking and non-lucid dreaming' (Sleep 2009).\n\n"

    "Knobs (set via /set k=v):\n"
    "  entropy=sane|wild|insane|crazy|mad   parametrises Carhart-Harris's\n"
    "                                       Entropic Brain spectrum (2014)\n"
    "  cards=N                              deck size (shuffle pool)\n"
    "  card_depth=shallow|medium|deep|max   per-card detail\n"
    "  n_concepts=K                         cards drawn per idea\n"
    "  n_ideas=N                            ideas per run\n"
    "  theme_entropy=X                      0..1, how distant the auto-\n"
    "                                       generated donor domains are\n"
    "  themes=A,B,C                         override theme auto-pick\n"
    "  refine=true|false                    Bayesian model selection +\n"
    "                                       harden the winner\n"
    "  evolve_deck=true|false               Loftus-style memory consolidation:\n"
    "                                       rewrite winning cards into the cache\n"
    "  seed=N                               reproducible sampling\n\n"

    "Inspection:\n"
    "  /settings   show current knob values\n"
    "  /usage      LLM usage + observed subscription window\n"
    "  /cancel     abort the in-flight request in this chat\n"
)


async def cmd_modes(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(MODES_TEXT)


async def cmd_settings(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state_for(update.effective_chat.id).settings
    await update.message.reply_text(
        f"entropy={s.entropy}\n"
        f"cards={s.cards}\n"
        f"card_depth={s.card_depth}\n"
        f"n_concepts={s.n_concepts}\n"
        f"n_ideas={s.n_ideas}\n"
        f"refine={s.refine}\n"
        f"evolve_deck={s.evolve_deck}\n"
        f"seed={s.seed}\n"
        f"model={s.model}\n\n"
        "Change one with /set <key>=<value>"
    )


SETTABLE = {
    "entropy", "cards", "card_depth", "n_concepts", "n_ideas",
    "refine", "evolve_deck", "seed", "model",
}


async def cmd_set(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    txt = _topic_from(update)
    if "=" not in txt:
        await update.message.reply_text("usage: /set <key>=<value>")
        return
    key, _, val = txt.partition("=")
    key, val = key.strip(), val.strip()
    if key not in SETTABLE:
        await update.message.reply_text(
            f"unknown key {key!r}; settable: {sorted(SETTABLE)}",
        )
        return
    s = state_for(update.effective_chat.id).settings
    try:
        if key in ("cards", "n_concepts", "n_ideas"):
            setattr(s, key, int(val))
        elif key == "seed":
            s.seed = int(val) if val.lower() not in ("", "none", "null") else None
        elif key in ("refine", "evolve_deck"):
            setattr(s, key, val.lower() in ("1", "true", "yes", "on"))
        else:
            setattr(s, key, val)
    except ValueError as e:
        await update.message.reply_text(f"value error: {e}")
        return
    await update.message.reply_text(f"ok, {key}={getattr(s, key)}")


async def cmd_usage(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Per-user view by default. Admin (AIDEA_ADMIN_CHAT_IDS env var)
    gets globals + top-chat breakdown instead."""
    chat_id = update.effective_chat.id
    if _is_admin(chat_id):
        text = _format_admin_usage()
    else:
        text = _format_user_usage(chat_id)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_whoami(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Tell the user their Telegram chat_id. Useful for finding the value
    to put into AIDEA_ADMIN_CHAT_IDS on the LXC."""
    chat_id = update.effective_chat.id
    is_admin = _is_admin(chat_id)
    await update.message.reply_text(
        f"Your chat_id: `{chat_id}`\n"
        f"Admin: *{'yes' if is_admin else 'no'}*",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_quota(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the caller their remaining free generations. Admins see the
    whole roster (who's used what)."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id if update.effective_user else chat_id
    limit = quota.free_limit()
    if _is_admin(user_id) or _is_admin(chat_id):
        rows = quota.table()  # admin-only snapshot of the quota table
        if not rows:
            await update.message.reply_text("No users have generated yet.")
            return
        ordered = sorted(
            rows.items(), key=lambda kv: int(kv[1].get("count", 0)), reverse=True,
        )
        subs = sum(1 for uid in rows if quota.is_subscriber(uid))
        lines = [f"📊 *Free-tier usage* (limit {limit}/user · "
                 f"{subs} subscriber(s))\n"]
        for uid, rec in ordered[:30]:
            c = int(rec.get("count", 0))
            who = rec.get("name") or rec.get("username") or uid
            if quota.is_subscriber(uid):
                tag = " 💎"
            elif c >= limit:
                tag = " 🔒"
            else:
                tag = ""
            lines.append(f"  {c}/{limit}{tag} — {who} (`{uid}`)")
        await update.message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.MARKDOWN,
        )
        return
    if quota.is_subscriber(user_id):
        await update.message.reply_text(
            "💎 You're a subscriber — unlimited generations, with the diverse "
            "critic panel on every one. Thank you!"
        )
        return
    used = quota.count(user_id)
    left = max(0, limit - used)
    await update.message.reply_text(
        f"You've used {used}/{limit} free idea generations — {left} left."
    )


async def cmd_resetquota(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin: reset a user's free-tier count after they subscribe.
    Usage: /resetquota <user_id>  (the id shown in the inquiry log)."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id if update.effective_user else chat_id
    if not (_is_admin(user_id) or _is_admin(chat_id)):
        await update.message.reply_text("Admins only.")
        return
    arg = _topic_from(update).strip().lstrip("@")
    if not arg:
        await update.message.reply_text("usage: /resetquota <user_id>")
        return
    try:
        target = int(arg)
    except ValueError:
        await update.message.reply_text(
            "user_id must be numeric — copy it from the inquiry log entry."
        )
        return
    prior = quota.reset(target)
    # Clear their inquiry-wait flag so the next message runs the pipeline
    # again (only reachable for 1:1 chats where chat_id == user_id).
    st = _STATES.get(target)
    if st is not None:
        st.awaiting_inquiry = False
    await update.message.reply_text(
        f"✅ Reset quota for `{target}` (was {prior}). "
        f"They now have {quota.free_limit()} fresh generations.",
        parse_mode=ParseMode.MARKDOWN,
    )


def _admin_target(update: Update) -> tuple[bool, int | None, str]:
    """Shared guard for admin user-id commands. Returns
    (is_admin, target_user_id_or_None, raw_args)."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id if update.effective_user else chat_id
    if not (_is_admin(user_id) or _is_admin(chat_id)):
        return False, None, ""
    parts = _topic_from(update).split()
    if not parts:
        return True, None, ""
    try:
        return True, int(parts[0].lstrip("@")), " ".join(parts[1:])
    except ValueError:
        return True, None, " ".join(parts[1:])


async def cmd_subscribe(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin: activate a paid subscription (the loop-closer after payment).
    Usage: /subscribe <user_id> [days]   (days default 30, stacks if re-run).
    A subscriber bypasses the free-tier wall and gets the critic panel on
    every generation."""
    ok, target, rest = _admin_target(update)
    if not ok:
        await update.message.reply_text("Admins only.")
        return
    if target is None:
        await update.message.reply_text("usage: /subscribe <user_id> [days]")
        return
    days = 30
    if rest.strip():
        try:
            days = max(1, int(rest.split()[0]))
        except ValueError:
            pass
    until = quota.set_subscriber(target, days=days)
    st = _STATES.get(target)
    if st is not None:
        st.awaiting_inquiry = False  # let them generate immediately
    when = time.strftime("%Y-%m-%d", time.gmtime(until))
    await update.message.reply_text(
        f"💎 `{target}` is now a subscriber for {days} day(s) — until {when}. "
        "Unlimited generations + critic panel on every run.",
        parse_mode=ParseMode.MARKDOWN,
    )
    await _log_to_group(
        _ctx, "inquiries",
        f"💎 subscription activated: user {target} · {days}d · until {when}",
    )


async def cmd_unsubscribe(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin: revoke a subscription immediately. Usage: /unsubscribe <user_id>."""
    ok, target, _rest = _admin_target(update)
    if not ok:
        await update.message.reply_text("Admins only.")
        return
    if target is None:
        await update.message.reply_text("usage: /unsubscribe <user_id>")
        return
    quota.clear_subscriber(target)
    await update.message.reply_text(
        f"Revoked subscription for `{target}`.", parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_feedback(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Explicit user-feedback signal targeting the last completed run.

    Usage:
      /feedback useful [optional comment]
      /feedback bad    [optional comment]

    Stored in card_outcomes.jsonl as a synthetic critic_total (+50 for
    useful, -100 for bad) so the next deck-gen retrieval re-ranks
    accordingly. Source-scoped — only affects this chat's RAG memory.
    """
    chat_id = update.effective_chat.id
    state = state_for(chat_id)
    parts = (update.message.text or "").split(maxsplit=2)
    if len(parts) < 2:
        await update.message.reply_text(
            "Usage:\n"
            "  /feedback useful [comment]\n"
            "  /feedback bad [comment]\n"
            "Targets the last completed run in this chat."
        )
        return
    verdict = parts[1].strip().lower()
    comment = parts[2].strip() if len(parts) > 2 else ""
    if verdict not in {"useful", "good", "yes", "+", "👍",
                        "bad", "useless", "no", "-", "👎"}:
        await update.message.reply_text(
            "First arg must be 'useful' or 'bad'."
        )
        return
    useful = verdict in {"useful", "good", "yes", "+", "👍"}
    if not state.last_run_id:
        await update.message.reply_text(
            "No run in this chat to attach feedback to yet."
        )
        return
    from rag import record_feedback
    record_feedback(
        run_id=state.last_run_id,
        useful=useful,
        comment=comment,
        source=f"telegram-{chat_id}",
    )
    label = "👍 useful" if useful else "👎 not useful"
    await update.message.reply_text(
        f"{label} recorded for run {state.last_run_id}. "
        "This will reweight future retrievals in this chat."
    )


async def cmd_bootstrap(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the cold-start bootstrap counter and whether per-channel
    isolation has kicked in yet."""
    from rag import bootstrap_state, bootstrap_notice_text
    s = bootstrap_state()
    if s.get("active"):
        text = (
            f"Bootstrap: ACTIVE  ·  {int(s['queries_seen'])}/{int(s['threshold'])} "
            f"queries  ·  {int(s['remaining'])} remaining\n"
            f"Retrieval is aggregate across all sources right now.\n\n"
            + bootstrap_notice_text()
        )
    else:
        sw = s.get("switched_at")
        text = (
            f"Bootstrap: COMPLETE  ·  switched at unix {sw}\n"
            "Retrieval is now strict per-channel — your future queries "
            "only see your own chat's past cards."
        )
    await update.message.reply_text(text)


async def cmd_corpus(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show RAG-corpus stats for this chat (per-source-scoped)."""
    from rag import stats as rag_stats, format_stats_text
    chat_id = update.effective_chat.id
    source = f"telegram-{chat_id}"
    s = rag_stats(source=source)
    text = format_stats_text(s)
    note = (
        "\n\nRAG retrieves donor concepts that worked for your past topics in "
        "THIS chat and uses them to warm-start the next deck. Privacy-scoped: "
        "no cross-chat leakage. Quality boost activates after at least one "
        "refine-winner is recorded."
    )
    await update.message.reply_text(text + note)


async def cmd_cancel(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    state = state_for(update.effective_chat.id)
    if not state.busy:
        await update.message.reply_text("Nothing to cancel — no active task.")
        return
    state.cancel.set()
    await update.message.reply_text("🛑 Cancel signalled.")


# Plain greetings and ack-fillers that should NOT trigger a full ideation
# pipeline — users who send these usually don't know what the bot does and
# are paying us (and themselves) ~3 min + ~$0.05 of subscription tokens
# for a forced "idea about saying hi" answer. Catch them at the door and
# explain instead.
_GREETING_WORDS = {
    # ru
    "привет", "приветик", "здравствуй", "здравствуйте", "хай", "ку",
    "хей", "доброе утро", "добрый день", "добрый вечер", "ку-ку",
    "здарова", "здаров", "хелло", "йо",
    # en
    "hi", "hello", "hey", "yo", "sup", "howdy", "good morning",
    "good evening", "good afternoon",
    # other common
    "salut", "bonjour", "hola", "ciao",
}


def _looks_like_idle_chat(text: str) -> bool:
    """True if the message is plainly a greeting / ack / too-short to be a
    real ideation task. Conservative — false-negatives (let through) are
    cheaper than false-positives (lecture a real user)."""
    t = text.strip().lower().rstrip(".!?…,)(:;")
    if not t:
        return True
    if t in _GREETING_WORDS:
        return True
    # Very short non-questions are almost certainly chat, not a task.
    # Keep the bar low so things like "куда поехать?" or "what now?" still
    # go through.
    if len(t) < 8 and "?" not in text:
        return True
    return False


def _looks_russian(text: str) -> bool:
    """Cheap script-based language detection. We only need RU vs anything-else
    for the welcome message — the synthesizer's own LANGUAGE_RULE handles the
    rest of the conversation."""
    cyr = sum(1 for ch in text if "Ѐ" <= ch <= "ӿ")
    lat = sum(1 for ch in text if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))
    return cyr > lat


_WELCOME_RU = (
    "Привет 👋\n\n"
    "Я генерирую *неожиданные, но выполнимые* идеи под вашу конкретную задачу.\n\n"
    "Просто напишите мне свою задачу, проблему или вопрос одной-двумя "
    "фразами — я попрошу подтверждения (Да/Нет) и через 2–3 минуты "
    "пришлю 3 разных идеи с конкретным первым шагом и примером, где "
    "похожий механизм уже сработал.\n\n"
    "Примеры:\n"
    "  • Как развивать персональный сайт эксперта по управлению бизнесом?\n"
    "  • Что приготовить на ужин за 15 минут?\n"
    "  • Придумай новую технологию кроссовок\n\n"
    "Команды:\n"
    "  🧭 /menu — панель управления с кнопками (режим, энтропия, глубина)\n"
    "  /help — все команды и настройки\n"
    "  /einstein /lsd /futures /dream /lucid — другие режимы генерации"
)

_WELCOME_EN = (
    "Hi 👋\n\n"
    "I generate *unexpected-but-doable* ideas for your specific problem.\n\n"
    "Just send your task, problem, or question in a sentence or two — "
    "I'll ask you to confirm (Yes/No) and in 2–3 minutes I'll reply with "
    "3 different ideas, each with a concrete first step and a real "
    "example where a similar mechanism has worked elsewhere.\n\n"
    "Examples:\n"
    "  • How to develop a personal site for a business consultant?\n"
    "  • What can I cook for dinner in 15 minutes?\n"
    "  • Invent a new sneaker technology\n\n"
    "Commands:\n"
    "  🧭 /menu — button-driven control panel (mode, entropy, depth)\n"
    "  /help — all commands and settings\n"
    "  /einstein /lsd /futures /dream /lucid — other generation modes"
)


_UNKNOWN_CMD_RU = (
    "⚠️ Такая команда не распознана.\n\n"
    "Доступные команды:\n"
    "  /idea <запрос> — основная генерация идей\n"
    "  /einstein /lsd /futures /dream /lucid — другие режимы\n"
    "  /help — полная справка по командам и настройкам\n"
    "  /settings /set — конфигурация\n"
    "  /usage /corpus /feedback /cancel\n\n"
    "Или просто напишите задачу/проблему/вопрос обычным сообщением — без слэша."
)

_UNKNOWN_CMD_EN = (
    "⚠️ That command isn't recognised.\n\n"
    "Available commands:\n"
    "  /idea <topic> — main idea generation\n"
    "  /einstein /lsd /futures /dream /lucid — alternate modes\n"
    "  /help — full help with commands and settings\n"
    "  /settings /set — configuration\n"
    "  /usage /corpus /feedback /cancel\n\n"
    "Or just send your task / problem / question as a plain message — no slash needed."
)


def _quota_prompt(ru: bool, limit: int) -> str:
    """Shown when a user hits their free-generation cap. Invites them to
    leave a subscription inquiry right in the chat (captured + forwarded)."""
    if ru:
        return (
            "✨ Понравились идеи?\n\n"
            f"Вы использовали все {limit} бесплатных генераций идей.\n\n"
            "Чтобы продолжить — оставьте заявку на подписку прямо здесь: "
            "просто напишите сообщение (имя, контакт и что вам нужно), "
            "и мы свяжемся с вами."
        )
    return (
        "✨ Enjoying the ideas?\n\n"
        f"You've used all {limit} free idea generations.\n\n"
        "To keep going, leave your subscription inquiry right here — just "
        "send a message (your name, contact, and what you need) and we'll "
        "get back to you."
    )


_INQUIRY_ACK_RU = "✅ Спасибо! Заявка передана — мы свяжемся с вами."
_INQUIRY_ACK_EN = (
    "✅ Thanks! Your inquiry has been passed along — we'll be in touch."
)


def _effective_n_ideas(s: "ChatSettings") -> tuple[int, str]:
    """Return (n_ideas, why) — modes that force a specific count get a
    short suffix explaining why. Einstein/Futures are 4-pass by design;
    LSD/Dream/Lucid honour the user's n_ideas; Default uses n_ideas as-is."""
    mode = s.mode or "default"
    if mode == "einstein":
        return 4, "one per Johnson mechanism (Adjacent / Exapt / Hunch / Err)"
    if mode == "futures":
        return 4, "one per horizon (+1y / +3y / +10y / +30y)"
    return int(s.n_ideas or 1), ""


def _depth_detail(depth_name: str) -> str:
    """Return '(N fields/card, ~T tok)' for the named depth. Pulled live
    from aidea.CARD_DEPTH_BY_NAME so it stays in sync with the source."""
    try:
        from aidea import CARD_DEPTH_BY_NAME
        d = CARD_DEPTH_BY_NAME.get(depth_name)
        if d is None:
            return ""
        return f"({len(d.fields)} fields/card, ~{d.target_tokens} tok)"
    except Exception:
        return ""


def _confirm_prompt(
    topic: str,
    ru: bool,
    s: "ChatSettings",
    brew: bool = False,
) -> tuple[str, InlineKeyboardMarkup]:
    """Build the localized 'are you sure?' confirmation message + buttons.

    The topic is echoed back (clipped) so the user can sanity-check what
    got received, and the active settings are surfaced as a parameter
    block so they're not committing to a 3-minute pipeline blindly.

    When ``brew`` is True the header is tagged with 🍵 so the user sees
    they're about to fire a Brew (text + Pillow card) rather than the
    regular text-only flow.

    Yes is styled `success` (green), No is styled `danger` (red) — Bot API
    9.4 button colors, supported by python-telegram-bot ≥ 22.7."""
    preview = topic[:300] + ("…" if len(topic) > 300 else "")
    mode = s.mode or "default"
    n_ideas, n_reason = _effective_n_ideas(s)
    eta = estimate_runtime_seconds(mode, refine=bool(s.refine))
    refine_on = bool(s.refine)
    header_ru = "🍵 Получено для Brew:" if brew else "📥 Получено сообщение:"
    header_en = "🍵 Got your topic for Brew:" if brew else "📥 Got your message:"
    confirm_tail_ru = (
        " Запустить Brew (текст + карточка для Stories)?"
        if brew else " Отправить в генерацию идей?"
    )
    confirm_tail_en = (
        " Run the Brew (text idea + shareable Stories card)?"
        if brew else " Submit as an idea-generation request?"
    )

    depth_extra = _depth_detail(s.card_depth)

    if ru:
        params = (
            f"⚙️ Параметры:\n"
            f"  • Режим:    *{_mode_label(mode)}*\n"
            f"  • Энтропия: *{s.entropy}*\n"
            f"  • Карты:    колода *{s.cards}* × глубина *{s.card_depth}* "
            f"{depth_extra} · *{s.n_concepts}* на идею\n"
            f"  • Идей:     *{n_ideas}*"
            + (f"  _({n_reason})_" if n_reason else "")
            + "\n"
            f"  • Доработка победителя: *{'вкл' if refine_on else 'выкл'}*\n"
            f"  • Ожидаемое время: *~{fmt_eta(eta)}*"
        )
        text = (
            f"{header_ru}\n\n«{preview}»\n\n"
            f"{params}\n\n{confirm_tail_ru}"
        )
        yes, no = "✅ Да, генерировать", "❌ Нет, отменить"
    else:
        params = (
            f"⚙️ Settings:\n"
            f"  • Mode:    *{_mode_label(mode)}*\n"
            f"  • Entropy: *{s.entropy}*\n"
            f"  • Cards:   deck *{s.cards}* × depth *{s.card_depth}* "
            f"{depth_extra} · *{s.n_concepts}* drawn per idea\n"
            f"  • Ideas:   *{n_ideas}*"
            + (f"  _({n_reason})_" if n_reason else "")
            + "\n"
            f"  • Refine winner: *{'on' if refine_on else 'off'}*\n"
            f"  • ETA: *~{fmt_eta(eta)}*"
        )
        text = (
            f"{header_en}\n\n«{preview}»\n\n"
            f"{params}\n\n{confirm_tail_en}"
        )
        yes, no = "✅ Yes, generate", "❌ No, cancel"

    # Brew rides the same confirmation card but has THREE choices:
    # run it now, or schedule for +3h / +8h ("appointment hook" — the
    # core retention lever from idea 5). Regular ideation keeps the
    # plain Yes/No.
    if brew:
        if ru:
            now_lab, later_lab, morning_lab, cancel_lab = (
                "⚡ Сейчас", "☕ Через 3ч", "🌅 Через 8ч", "❌ Отмена",
            )
        else:
            now_lab, later_lab, morning_lab, cancel_lab = (
                "⚡ Now", "☕ In 3h", "🌅 In 8h", "❌ Cancel",
            )
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(now_lab,    callback_data="brew:now",    style="success"),
                InlineKeyboardButton(later_lab,  callback_data="brew:3h",     style="primary"),
                InlineKeyboardButton(morning_lab, callback_data="brew:8h",    style="primary"),
            ],
            [
                InlineKeyboardButton(cancel_lab, callback_data="brew:cancel", style="danger"),
            ],
        ])
    else:
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton(yes, callback_data="idea:confirm", style="success"),
            InlineKeyboardButton(no, callback_data="idea:cancel", style="danger"),
        ]])
    return text, kb


async def on_plain_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle non-command text messages. Branches:
    1. Bottom-keyboard label — open the matching sub-picker / toggle.
    2. Starts with '/' — an unknown / typo'd command. Send help.
    3. Plain greeting / empty / too short — send the welcome explainer.
    4. Substantive — stash as pending_topic and ask Yes/No before we
       burn a ~3-min pipeline + tokens. The actual launch happens in
       on_callback when the user clicks Yes."""
    topic = (update.message.text or "").strip()

    # 1. Persistent bottom menu — tap routes to the matching action.
    #    Prefix match because labels carry trailing live-state suffixes
    #    (e.g. "🎲 Mode: einstein"). Tapping any menu button also CLEARS
    #    a pending Brew prompt — the user changed their mind.
    action = _menu_action_for(topic)
    if action is not None:
        state = state_for(update.effective_chat.id)
        if action != "start_brew":
            state.awaiting_brew_topic = False
        await _handle_menu_action(action, update, ctx)
        return

    if topic.startswith("/"):
        msg = _UNKNOWN_CMD_RU if _looks_russian(topic) else _UNKNOWN_CMD_EN
        await update.message.reply_text(msg)
        return

    state = state_for(update.effective_chat.id)

    # If 🍵 Brew was just tapped, this message IS the brew topic. Run
    # it through the SAME Yes/No confirmation card as the regular flow,
    # but flag pending_brew so the confirm callback adds the Pillow
    # card render. Tapping the button alone is not enough opt-in —
    # users still want to sanity-check the topic before paying ~3 min.
    is_brew = state.awaiting_brew_topic
    state.awaiting_brew_topic = False

    # Free-tier wall. A user past their limit can't start a new generation
    # (interactive brews included); only admins are never walled. Their plain
    # messages become subscription inquiries: forwarded to the log group,
    # acknowledged here, never turned into a Yes/No card.
    tg_user = update.effective_user
    user_id = tg_user.id if tg_user is not None else update.effective_chat.id
    walled = (
        not (_is_admin(user_id) or _is_admin(update.effective_chat.id))
        and not quota.is_subscriber(user_id)
        and quota.is_over(user_id)
    )
    if walled:
        ru = _looks_russian(topic)
        await _log_to_group(
            ctx, "inquiries",
            f"💌 {_user_label(update)}\n{topic}",
        )
        if not state.awaiting_inquiry:
            state.awaiting_inquiry = True
            await update.message.reply_text(_quota_prompt(ru, quota.free_limit()))
        else:
            await update.message.reply_text(
                _INQUIRY_ACK_RU if ru else _INQUIRY_ACK_EN,
            )
        return

    if _looks_like_idle_chat(topic):
        if is_brew:
            await update.message.reply_text(
                "🍵 Brew cancelled — that didn't look like a topic. "
                "Tap 🍵 Brew again when you have a real question."
            )
            return
        welcome = _WELCOME_RU if _looks_russian(topic) else _WELCOME_EN
        await update.message.reply_text(welcome)
        return

    state.pending_topic = topic
    state.pending_brew = is_brew
    text, kb = _confirm_prompt(
        topic, ru=_looks_russian(topic), s=state.settings, brew=is_brew,
    )
    await update.message.reply_text(
        text, reply_markup=kb, parse_mode=ParseMode.MARKDOWN,
    )


async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Yes/No inline-button handler for the plain-text confirmation flow."""
    query = update.callback_query
    if query is None:
        return
    await query.answer()  # stops the spinner on the button

    chat_id = update.effective_chat.id
    state = state_for(chat_id)
    data = query.data or ""

    if data == "idea:confirm":
        topic = state.pending_topic
        is_brew = state.pending_brew
        state.pending_topic = None
        state.pending_brew = False
        if not topic:
            # Stale button click (bot restarted, or already confirmed).
            try:
                await query.edit_message_text("⚠️ Запрос уже не активен / no pending topic.")
            except Exception:
                pass
            return
        ru = _looks_russian(topic)
        mode = state.settings.mode or "default"

        # Lucid needs a 'prior | topic' (or 'prior :: topic') format because
        # the prior anchors the hallucination. Parse here so the user can
        # pick Lucid from the menu and then just type 'prior | topic' as
        # their next message — no slash command needed.
        extra: dict = {"brew_render": True} if is_brew else {}
        if mode == "lucid":
            sep = None
            if "|" in topic:
                sep = "|"
            elif "::" in topic:
                sep = "::"
            if sep:
                prior, _, lucid_topic = topic.partition(sep)
                topic = lucid_topic.strip()
                extra["lucid_prior"] = prior.strip()
            else:
                # No prior supplied — explain the format and abort gracefully.
                msg = (
                    "⚠️ Lucid-режим требует формат `prior | topic`.\n\n"
                    "Например:\n"
                    "  `solo-founder only | как монетизировать AI-бота`\n\n"
                    "Либо выберите другой режим через 🎲 *Mode*."
                    if ru else
                    "⚠️ Lucid mode needs a `prior | topic` format.\n\n"
                    "Example:\n"
                    "  `solo-founder only | how do I monetize my AI tool`\n\n"
                    "Or pick a different mode via 🎲 *Mode*."
                )
                try:
                    await query.edit_message_text(msg, parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    pass
                return

        mode_lab = _mode_label(mode)
        brew_tag = " · 🍵 Brew" if is_brew else ""
        ack = (
            f"✅ Запускаю генерацию ({mode_lab}{brew_tag}) для: «{topic[:200]}»"
            if ru
            else f"✅ Generating ideas ({mode_lab}{brew_tag}) for: «{topic[:200]}»"
        )
        try:
            await query.edit_message_text(ack)
        except Exception:
            pass
        await run_pipeline_for_telegram(
            update=update, context=ctx, topic=topic, mode=mode,
            extra=extra or None,
        )
    elif data == "idea:cancel":
        state.pending_topic = None
        state.pending_brew = False
        ru = _looks_russian(query.message.text if query.message else "")
        msg = (
            "❌ Отменено — сообщение не отправлено в генерацию."
            if ru
            else "❌ Cancelled — your message was not submitted."
        )
        try:
            await query.edit_message_text(msg)
        except Exception:
            pass

    elif data.startswith("brew:drop_"):
        # /brews → ❌ Cancel #N
        try:
            bid = int(data.split("_", 1)[1])
        except (ValueError, IndexError):
            return
        import brew_queue
        ok = brew_queue.cancel(bid, chat_id)
        msg = (
            f"❌ Brew #{bid} cancelled." if ok
            else f"⚠️ Brew #{bid} can't be cancelled (already running or delivered)."
        )
        try:
            await query.edit_message_text(msg)
        except Exception:
            pass

    elif data in ("brew:now", "brew:3h", "brew:8h", "brew:cancel"):
        topic = state.pending_topic
        state.pending_topic = None
        state.pending_brew = False
        ru = _looks_russian(topic or "")
        if data == "brew:cancel" or not topic:
            msg = (
                "❌ Brew отменён — сообщение не отправлено."
                if ru else "❌ Brew cancelled — your message was not submitted."
            )
            try:
                await query.edit_message_text(msg)
            except Exception:
                pass
            return

        mode = state.settings.mode or "default"
        if data == "brew:now":
            # Existing flow: run immediately with card render.
            ack = (
                f"✅ Запускаю Brew ({_mode_label(mode)}) для: «{topic[:200]}»"
                if ru else
                f"✅ Brewing now ({_mode_label(mode)}) for: «{topic[:200]}»"
            )
            try:
                await query.edit_message_text(ack)
            except Exception:
                pass
            await run_pipeline_for_telegram(
                update=update, context=ctx, topic=topic, mode=mode,
                extra={"brew_render": True},
            )
            return

        # Scheduled paths: write to brew_queue, return immediately.
        import brew_queue
        delay = 3 * 3600 if data == "brew:3h" else 8 * 3600
        reveal_at = time.time() + delay
        try:
            brew_id = brew_queue.schedule(
                chat_id=chat_id, topic=topic, mode=mode,
                reveal_at=reveal_at,
            )
        except Exception as e:
            log.exception("brew_queue.schedule failed")
            try:
                await query.edit_message_text(
                    f"⚠️ Couldn't schedule the Brew: {e}\nFalling back to /brew run now.",
                )
            except Exception:
                pass
            return

        when_local = time.strftime("%H:%M", time.localtime(reveal_at))
        delay_lab = "3 hours" if data == "brew:3h" else "8 hours"
        delay_lab_ru = "3 часа" if data == "brew:3h" else "8 часов"
        ack = (
            f"🍵 *Brew поставлен в очередь* (#{brew_id})\n\n"
            f"Тема: «{topic[:200]}»\n"
            f"Дойдёт через {delay_lab_ru} (~{when_local}).\n\n"
            f"_Отменить: `/brews` → нажмите ❌ рядом с этим брю._"
            if ru else
            f"🍵 *Brew queued* (#{brew_id})\n\n"
            f"Topic: «{topic[:200]}»\n"
            f"Ready in {delay_lab} (~{when_local}).\n\n"
            f"_Cancel: `/brews` → tap ❌ next to this brew._"
        )
        try:
            await query.edit_message_text(ack, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Button-driven control panel — /menu opens a single message with inline
# keyboards for picking mode, entropy, depth, and toggles. Discoverable
# alternative to the textual /set syntax. Plain-text + /idea keep working
# unchanged for users who already know what they want.
# ---------------------------------------------------------------------------


_MODE_INFO: list[tuple[str, str, str]] = [
    # (key, label, one-line description shown in the picker)
    ("default",  "💡 Default",  "applied ideas · feasibility required"),
    ("einstein", "🧠 Einstein", "4 mechanism passes (adjacent / exapt / hunch / err)"),
    ("lsd",      "🌀 LSD",      "prior dissolution + sober validation"),
    ("futures",  "🔮 Futures",  "4 temporal horizons (+1y / +3y / +10y / +30y)"),
    ("dream",    "💤 Dream",    "feasibility OFF — dream image + interpretation"),
    ("lucid",    "🧘 Lucid",    "feasibility ON — directional dream toward a prior"),
]


# Rich per-mode descriptions shown in the picker-confirm message, so a user
# who taps a mode also learns *what it does and why*, not just "set to X".
# Distilled from MODES_TEXT (which is the long /help block); each entry is
# kept under ~700 chars so it fits in a single Telegram message comfortably.
_MODE_DESCRIPTION: dict[str, str] = {
    "default": (
        "💡 *Default mode* — applied conceptual blending.\n\n"
        "Picks 1+ donor concepts and finds the structural mechanism that "
        "transfers onto your topic. Feasibility required at every entropy "
        "level — even 'mad' must ship a v0.1 in six months.\n\n"
        "*When to reach for it:*\n"
        "  • You have a concrete problem and want a buildable answer with "
        "a clear first step.\n"
        "  • You want three different angles to compare side-by-side, not "
        "one verdict to take or leave.\n"
        "  • 'What's the right next move' matters more than 'what's the "
        "most surprising move'.\n\n"
        "_Sources: Fauconnier & Turner, The Way We Think (2002); "
        "Kauffman, Investigations (2000) — the 'adjacent possible' framing._"
    ),
    "einstein": (
        "🧠 *Einstein mode* — four passes, one per generative mechanism "
        "from Steven Johnson's _Where Good Ideas Come From_ (2010):\n\n"
        "  • *Adjacent Possible* — name a capability unlocked in the "
        "last 1–3 years, walk through that door.\n"
        "  • *Exaptation* — transplant a mechanism from a far-distant "
        "field (Gould & Vrba, 1982).\n"
        "  • *Slow Hunch* — articulate a latent tension the field hasn't "
        "named.\n"
        "  • *Productive Error* — invert a load-bearing assumption "
        "(Fleming 1928).\n\n"
        "*When to reach for it:*\n"
        "  • You want *structural* variety, not entropy variety — four "
        "genuinely different routes to the same problem.\n"
        "  • You suspect different mechanisms fit your topic differently "
        "and want to find out which one carries the load.\n"
        "  • Big-bet questions where comparing 'what just unlocked' vs "
        "'what to invert' side-by-side beats a single answer.\n"
        "  • Pair with Refine=ON to rank the four mechanisms and harden "
        "the winner."
    ),
    "lsd": (
        "🌀 *LSD mode* — two-pass structure (the most expensive mode):\n\n"
        "  *Pass 1 — Anarchic generation:* priors relaxed, error-detection "
        "OFF. Flatten the hierarchy, treat 2–3 load-bearing assumptions "
        "as noise, force cross-module connections between distant donors.\n\n"
        "  *Pass 2 — Sober validation:* priors back online. Separates "
        "the structural insight from the hallucination, proposes a "
        "buildable v0.1.\n\n"
        "*When to reach for it:*\n"
        "  • The field's assumptions feel like the blocker, not the "
        "technology — you want to dissolve them and see what survives.\n"
        "  • You've tried the obvious moves and they all rhyme — you need "
        "something further out, with a safety net to filter hallucination.\n"
        "  • You want one disruptive answer rather than four executable "
        "variants, and can afford the extra pipeline cost.\n\n"
        "_Source: Carhart-Harris & Friston, 'REBUS and the Anarchic Brain' "
        "(Pharmacol Reviews 2019)._"
    ),
    "futures": (
        "🔮 *Futures mode* — four temporal horizons (+1y / +3y / +10y / "
        "+30y).\n\n"
        "At each horizon: name three concrete shifts that are likely by "
        "then, identify what's obvious from there but invisible today, "
        "translate to a v0.1 you can ship *this year*.\n\n"
        "*When to reach for it:*\n"
        "  • Strategy questions — where to bet given where the field is "
        "heading, not where it sits today.\n"
        "  • You can articulate 'I want to choose between options that pay "
        "off at different timescales' — futures gives you the comparison.\n"
        "  • Product / career roadmaps where the current move depends on "
        "what's obvious from a future vantage point.\n"
        "  • 'What's obvious from 2035 that's invisible today?'\n\n"
        "_Lineage: Pierre Wack scenario planning (Shell, 1970s); "
        "Stewart Brand, The Clock of the Long Now (1999); Anil Seth on "
        "perceptual forward-modeling (Being You, 2021)._"
    ),
    "dream": (
        "💤 *Dream mode* — prediction-error signal OFFLINE. Generative "
        "model runs free, NO feasibility floor.\n\n"
        "Output: a vivid dream image (may violate physics, economics, "
        "regulation) plus a 'what survives waking' line naming the "
        "salvageable fragment.\n\n"
        "*When to reach for it:*\n"
        "  • You want fragments to mine, not solutions to ship — the "
        "dream is a metaphor source, not a plan.\n"
        "  • You're stuck in the current frame and need to be jolted out "
        "before re-engaging.\n"
        "  • Use the 'what survives waking' salvage line as the seed for "
        "a subsequent Default or Einstein run.\n"
        "  • Mood-boarding, world-building, copywriting kick-starters.\n\n"
        "_Source: Friston, Free Energy Principle (2010) — dreams as "
        "complexity reduction / synaptic garbage collection. Hobson & "
        "Friston (Progress in Neurobiology, 2012)._"
    ),
    "lucid": (
        "🧘 *Lucid mode* — dream-state generation + a directional prior "
        "you inject.\n\n"
        "The hallucination biases toward your belief; one reality-check "
        "fires before waking. More salvageable than pure dream because "
        "the prior anchors what the engine imagines.\n\n"
        "*Format:* send your message as `prior | topic` (or "
        "`prior :: topic`). Example:\n"
        "  `solo-founder only | how do I monetize my AI tool`\n"
        "Or use the slash command directly:\n"
        "  `/lucid <prior> | <topic>`\n\n"
        "*When to reach for it:*\n"
        "  • You suspect your problem assumptions are blocking the obvious "
        "answer — Lucid lets you write the assumption as the prior and "
        "watch the engine work around it.\n"
        "  • You want one weird mood-board output to harvest fragments "
        "from, rather than three executable ideas.\n"
        "  • The prior acts as a salvage anchor — without it (Dream mode) "
        "the output is harder to translate into anything actionable.\n\n"
        "_Source: LaBerge, Exploring the World of Lucid Dreaming (1990); "
        "Voss et al., Sleep (2009)._"
    ),
}

_ENTROPY_INFO: list[tuple[str, str, str]] = [
    ("sane",   "😴 Sane",   "stay within established practice — 'we should just do that'"),
    ("wild",   "✨ Wild",    "combine familiar in uncommon ways — 'huh, didn't think of that combo'"),
    ("insane", "🤯 Insane", "transplant mechanism from unrelated domain — 'wait, can we?'"),
    ("crazy",  "🚨 Crazy",  "challenge a load-bearing assumption — sounds reckless, survives reread"),
    ("mad",    "🎭 Mad",    "reinterpret the problem itself — must still ship v0.1 in 6 months"),
]

_DEPTH_INFO: list[tuple[str, str, str]] = [
    ("shallow", "🪶 Shallow", "name + domain only — fastest, most stochastic"),
    ("medium",  "📘 Medium",  "+ mechanism, transfer hint — default"),
    ("deep",    "📗 Deep",    "+ invariants, prior application — slower, richer"),
    ("max",     "📚 Max",     "every field populated — slowest, most context"),
]


def _mode_label(key: str) -> str:
    for k, lab, _ in _MODE_INFO:
        if k == key:
            return lab
    return key


def _settings_summary(s: "ChatSettings") -> str:
    mode_lab = _mode_label("default")  # main menu always assumes default unless user picks
    return (
        f"Mode: default · "
        f"Entropy: {s.entropy} · "
        f"Depth: {s.card_depth} · "
        f"Ideas: {s.n_ideas} · "
        f"Refine: {'on' if s.refine else 'off'}"
    )


def _main_menu_text(s: "ChatSettings") -> str:
    mode_lab = _mode_label(s.mode or "default")
    return (
        "🧭 *AIdea control panel*\n\n"
        "Current settings:\n"
        f"  · Mode: *{mode_lab}*\n"
        f"  · Entropy: *{s.entropy}*\n"
        f"  · Card depth: *{s.card_depth}*\n"
        f"  · Ideas per run: *{s.n_ideas}*\n"
        f"  · Refine winner: *{'on' if s.refine else 'off'}*\n\n"
        "Use the bottom buttons to change settings. To generate, just send "
        "your topic as a plain message — bot will ask Yes/No before running."
    )


# Bottom-attached persistent ReplyKeyboard.
#
# Button text embeds the current value of each setting (e.g. "🎲 Mode:
# einstein", "🔧 Refine: ON") so the user can read live state straight off
# the keyboard without opening anything. Because the trailing value changes,
# tap-routing in on_plain_message uses PREFIX matching — see
# _MAIN_MENU_PREFIXES + _menu_action_for.
#
# Telegram does not support per-button colors on either KeyboardButton or
# InlineKeyboardButton beyond Bot API 9.4's three predefined styles
# (primary / success / danger) — emoji groupings carry the rest.
_MAIN_MENU_PREFIXES: list[tuple[str, str]] = [
    ("🍵 Brew",     "start_brew"),
    ("🎲 Mode",     "open_mode"),
    ("🌪 Entropy",  "open_entropy"),
    ("📘 Depth",    "open_depth"),
    ("🔧 Refine",   "toggle_refine"),
    ("📈 Usage",    "show_usage"),
    ("❓ Help",     "show_help"),
    ("🧭 Status",   "show_status"),
    ("✖ Hide menu", "hide_menu"),
]


def _menu_action_for(text: str) -> str | None:
    """Map a tap on the bottom keyboard (whose label may carry a
    trailing live-state suffix) back to a stable action key. Prefix
    match so "🎲 Mode: einstein" still routes to open_mode."""
    for prefix, action in _MAIN_MENU_PREFIXES:
        if text.startswith(prefix):
            return action
    return None


def _main_menu_kb(s: "ChatSettings") -> ReplyKeyboardMarkup:
    """Persistent bottom-attached menu. Two rows of three + one row of two.
    Button text shows current values so the user reads live state at a
    glance.

    Color scheme (Bot API 9.4):
      primary (blue)  — opens a sub-picker / tunable setting
      success (green) — neutral utility (status / usage / help)
      danger  (red)   — destructive (Hide menu)"""
    refine_txt = "ON" if s.refine else "off"
    return ReplyKeyboardMarkup(
        [
            # Top row — the primary action gets its own wide button.
            [KeyboardButton("🍵 Brew", style="success")],
            [
                KeyboardButton(f"🎲 Mode: {s.mode or 'default'}", style="primary"),
                KeyboardButton(f"🌪 Entropy: {s.entropy}",        style="primary"),
                KeyboardButton(f"📘 Depth: {s.card_depth}",       style="primary"),
            ],
            [
                KeyboardButton(f"🔧 Refine: {refine_txt}", style="primary"),
                KeyboardButton("📈 Usage",                 style="success"),
                KeyboardButton("❓ Help",                  style="success"),
            ],
            [
                KeyboardButton("🧭 Status",     style="success"),
                KeyboardButton("✖ Hide menu",  style="danger"),
            ],
        ],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Send a topic or tap a button…",
    )


def _picker_kb(items: list[tuple[str, str, str]], prefix: str, current: str) -> InlineKeyboardMarkup:
    """Build a sub-picker keyboard. One row per option, marker on the
    currently-selected value. No Back button — the persistent bottom
    keyboard is always visible, so navigation is handled there.

    Color: currently-selected option is styled `success` (green) so the
    user can spot their current pick at a glance."""
    rows: list[list[InlineKeyboardButton]] = []
    for key, label, desc in items:
        is_current = (key == current)
        marker = "✅ " if is_current else "   "
        rows.append([InlineKeyboardButton(
            f"{marker}{label} — {desc}",
            callback_data=f"{prefix}:{key}",
            style="success" if is_current else None,
        )])
    return InlineKeyboardMarkup(rows)


async def cmd_menu(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Open the button-driven control panel. Attaches the persistent
    bottom keyboard and prints the current settings summary."""
    state = state_for(update.effective_chat.id)
    await update.message.reply_text(
        _main_menu_text(state.settings),
        reply_markup=_main_menu_kb(state.settings),
        parse_mode=ParseMode.MARKDOWN,
    )


async def _handle_menu_action(
    action: str,
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Dispatch from a bottom-keyboard label tap to the right sub-picker.

    Sub-pickers are sent as fresh InlineKeyboard messages — one-shot,
    transient — so the persistent bottom keyboard stays untouched and
    multiple pickers don't pile up at the bottom."""
    chat_id = update.effective_chat.id
    state = state_for(chat_id)
    s = state.settings

    if action == "open_mode":
        await ctx.bot.send_message(
            chat_id=chat_id,
            text=(
                "*Mode* — how the synthesizer assembles each idea.\n\n"
                f"Current: *{_mode_label(s.mode or 'default')}*\n\n"
                "_Note: Lucid needs a 'prior | topic' format — see the "
                "description after picking it._"
            ),
            reply_markup=_picker_kb(_MODE_INFO, "mode", s.mode or "default"),
            parse_mode=ParseMode.MARKDOWN,
        )

    elif action == "open_entropy":
        await ctx.bot.send_message(
            chat_id=chat_id,
            text=(
                "*Entropy* — how far the cross-domain sampler is allowed to wander.\n\n"
                f"Current: *{s.entropy}*"
            ),
            reply_markup=_picker_kb(_ENTROPY_INFO, "entropy", s.entropy),
            parse_mode=ParseMode.MARKDOWN,
        )

    elif action == "open_depth":
        await ctx.bot.send_message(
            chat_id=chat_id,
            text=(
                "*Card depth* — how many fields each donor card carries.\n\n"
                f"Current: *{s.card_depth}*"
            ),
            reply_markup=_picker_kb(_DEPTH_INFO, "depth", s.card_depth),
            parse_mode=ParseMode.MARKDOWN,
        )

    elif action == "toggle_refine":
        s.refine = not s.refine
        await ctx.bot.send_message(
            chat_id=chat_id,
            text=f"🔧 Refine: *{'ON' if s.refine else 'off'}*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=_main_menu_kb(s),  # refresh button labels with new value
        )

    elif action == "show_usage":
        try:
            text = (
                _format_admin_usage()
                if _is_admin(chat_id)
                else _format_user_usage(chat_id)
            )
            await _send_with_retry(
                ctx.bot,
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            try:
                await _send_with_retry(
                    ctx.bot, chat_id=chat_id, text=f"usage error: {e}",
                )
            except Exception:
                # Even the error-reply send failed; just log and move on.
                log.exception("show_usage: failed to deliver error message")

    elif action == "show_help":
        await ctx.bot.send_message(chat_id=chat_id, text=HELP_TEXT)

    elif action == "show_status":
        await ctx.bot.send_message(
            chat_id=chat_id,
            text=_main_menu_text(s),
            parse_mode=ParseMode.MARKDOWN,
        )

    elif action == "hide_menu":
        await ctx.bot.send_message(
            chat_id=chat_id,
            text="Menu hidden. Send /menu to bring it back.",
            reply_markup=ReplyKeyboardRemove(),
        )

    elif action == "start_brew":
        # Two-step Brew flow: this button only PROMPTS for a topic.
        # The next plain-text message routes straight into the Brew
        # pipeline via on_plain_message's awaiting_brew_topic check —
        # no Yes/No confirmation step, since tapping 🍵 already counts
        # as the explicit opt-in.
        state.awaiting_brew_topic = True
        await ctx.bot.send_message(
            chat_id=chat_id,
            text=(
                "🍵 *Brew*\n\n"
                "Send your topic / problem / question as your next message. "
                "I'll run the full pipeline and deliver one *card* — "
                "1080×1920, ready to share to Stories.\n\n"
                "_Tip: any other menu button cancels the Brew prompt._"
            ),
            parse_mode=ParseMode.MARKDOWN,
        )


async def on_menu_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Inline-keyboard handler for the sub-pickers opened from the bottom
    keyboard. Each picker is a transient one-shot message — selecting a
    value mutates ChatSettings and edits the picker message to confirm,
    no back-navigation needed (the persistent bottom keyboard is always
    there).

    Pattern routes:
      mode:<name>     → set mode
      entropy:<name>  → set entropy
      depth:<name>    → set card_depth
    """
    query = update.callback_query
    if query is None:
        return
    await query.answer()

    chat_id = update.effective_chat.id
    state = state_for(chat_id)
    s = state.settings
    data = (query.data or "")

    def _confirm_edit(label: str, value: str) -> str:
        return f"✅ *{label}* set to *{value}*"

    async def _refresh_bottom_keyboard():
        """Send a tiny status message whose only job is to push the
        new ReplyKeyboardMarkup down — so the bottom row's live-state
        labels reflect the change the user just made."""
        try:
            await ctx.bot.send_message(
                chat_id=chat_id,
                text="⚙️ updated",
                reply_markup=_main_menu_kb(s),
            )
        except Exception:
            pass

    if data.startswith("mode:"):
        key = data.split(":", 1)[1]
        valid = {k for k, _, _ in _MODE_INFO}
        if key in valid:
            s.mode = key
            lab = _mode_label(key)
            desc = _MODE_DESCRIPTION.get(key, "")
            body = f"✅ *Mode set to {lab}*\n\n{desc}" if desc else _confirm_edit("Mode", lab)
            try:
                await query.edit_message_text(body, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                pass
            await _refresh_bottom_keyboard()

    elif data.startswith("entropy:"):
        key = data.split(":", 1)[1]
        valid = {k for k, _, _ in _ENTROPY_INFO}
        if key in valid:
            s.entropy = key
            try:
                await query.edit_message_text(
                    _confirm_edit("Entropy", key),
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                pass
            await _refresh_bottom_keyboard()

    elif data.startswith("depth:"):
        key = data.split(":", 1)[1]
        valid = {k for k, _, _ in _DEPTH_INFO}
        if key in valid:
            s.card_depth = key
            try:
                await query.edit_message_text(
                    _confirm_edit("Card depth", key),
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                pass
            await _refresh_bottom_keyboard()


# ---------------------------------------------------------------------------
# Scheduled-Brew worker — pulled from brew_queue, runs the minimal
# headless pipeline (deck-gen → sample → synth → critic winner pick →
# Pillow render), delivers as one photo with a single header line.
# Wired into PTB JobQueue in build_app(); polls every 60s.
# ---------------------------------------------------------------------------


async def _brew_worker_run(bot: Any, row: dict) -> None:
    """Process one due Brew end-to-end. Caller has already called
    brew_queue.claim() on this row (so status is 'brewing'), so on
    success / failure we update status and stop."""
    import brew_queue
    from brew_render import (
        render_card,
        parse_idea_fields,
        default_output_path,
    )
    from aidea import (
        load_or_generate_deck,
        sample_cards,
        build_prompt,
        synthesize,
        critic_score,
        total_score,
        parse_entropy,
        CARD_DEPTH_BY_NAME,
    )
    from usage import start_run
    from transcripts import set_source, log_event as transcript_log

    brew_id = row["id"]
    chat_id = row["chat_id"]
    topic = row["topic"]
    mode = (row.get("mode") or "default")

    run_id = start_run(f"tg-{chat_id}")
    set_source(f"telegram-{chat_id}")
    transcript_log(
        "request_started",
        topic=topic, mode=mode, brew_id=brew_id,
        scheduled=True,
    )

    try:
        # Minimal pipeline — single idea, default knobs. Refine off (a
        # scheduled brew shouldn't double its runtime budget). Mode is
        # honoured for the synthesis prompt but multi-idea modes
        # (einstein/futures) deliver the first idea only — the Brew
        # card carries ONE idea, by design.
        spread, level = parse_entropy("wild")
        depth = CARD_DEPTH_BY_NAME["medium"]
        deck = await load_or_generate_deck(
            topic=topic, n=30, depth=depth,
            model="claude-opus-4-7",
            force_regen=False, verbose=False,
        )
        import random
        rng = random.Random()
        cards = sample_cards(deck=deck, n=3, spread=spread, rng=rng)
        prompt = build_prompt(topic, cards, level)
        idea_text = await synthesize(
            prompt=prompt, model="claude-opus-4-7", stream_to_stdout=False,
        )
        transcript_log(
            "idea", i=0, mechanism=None, text=idea_text,
            cards=[{k: v for k, v in c.__dict__.items() if v is not None}
                   for c in cards],
        )

        fields = parse_idea_fields(idea_text)
        png_path = render_card(
            title=fields.get("title") or topic[:60],
            pitch=fields.get("pitch", ""),
            mechanism=fields.get("mechanism", ""),
            first_step=fields.get("first_step", ""),
            output_path=default_output_path(brew_id=brew_id),
        )

        # Deliver as one photo with the topic as a small header caption.
        # The card itself carries the substance; caption is just context.
        with png_path.open("rb") as f:
            msg = await bot.send_photo(
                chat_id=chat_id,
                photo=f,
                caption=(
                    f"☕ Your Brew is ready.\n"
                    f"Topic: «{topic[:200]}»"
                ),
            )
        brew_queue.mark_delivered(
            brew_id=brew_id,
            message_id=msg.message_id,
            result_path=str(png_path),
        )
        transcript_log("request_completed", n_ideas=1, brew_id=brew_id)
    except Exception as e:
        log.exception("brew worker failed for brew_id=%s", brew_id)
        transcript_log("request_errored", error_type=type(e).__name__, error=str(e))
        brew_queue.mark_failed(brew_id, str(e))
        # Tell the user something went wrong so they don't wait forever.
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"⚠️ Your Brew (#{brew_id}) couldn't finish. "
                    f"Try sending the topic again, or run /brew now."
                ),
            )
        except Exception:
            pass


async def brew_poll_callback(context: ContextTypes.DEFAULT_TYPE) -> None:
    """PTB JobQueue callback — runs every 60 s. Pulls due brews from
    SQLite, processes them serially so one stuck pipeline can't block
    a queue or burn the rate-limit by parallel claude subprocess fan-
    out."""
    import brew_queue
    try:
        due = brew_queue.due_brews(limit=5)
    except Exception:
        log.exception("brew_poll_callback: due_brews failed")
        return
    for row in due:
        # Atomic pending → brewing. Skip if another worker beat us.
        if not brew_queue.claim(row["id"]):
            continue
        await _brew_worker_run(context.bot, row)


async def cmd_brews(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """List the user's pending + last-5 historical brews, with inline
    ❌ buttons to cancel any still-pending entry."""
    import brew_queue
    chat_id = update.effective_chat.id
    pending = brew_queue.pending_for_chat(chat_id, limit=10)
    history = brew_queue.history_for_chat(chat_id, limit=5)
    lines = ["🍵 *Your brews*"]
    if pending:
        lines.append("\n*Pending:*")
        for b in pending:
            when = time.strftime("%H:%M", time.localtime(b["reveal_at"]))
            dt = b["reveal_at"] - time.time()
            mins = max(0, int(dt / 60))
            lines.append(
                f"  • #{b['id']}  ({when}, in {mins}m)  «{b['topic'][:60]}»"
            )
    else:
        lines.append("\n_No pending brews. Tap 🍵 Brew on the menu to schedule one._")
    if history:
        lines.append("\n*Recent:*")
        for b in history:
            stat = b["status"]
            icon = {"delivered": "✅", "failed": "⚠️", "pending": "⏳", "brewing": "🔄"}.get(stat, "·")
            when = time.strftime("%m-%d %H:%M", time.localtime(b["created_at"]))
            lines.append(f"  {icon} {when}  «{b['topic'][:50]}»")

    kb_rows: list[list[InlineKeyboardButton]] = []
    for b in pending[:5]:
        kb_rows.append([InlineKeyboardButton(
            f"❌ Cancel #{b['id']}",
            callback_data=f"brew:drop_{b['id']}",
            style="danger",
        )])
    kb = InlineKeyboardMarkup(kb_rows) if kb_rows else None
    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb,
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def build_app() -> Application:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set. Get a token from @BotFather and "
            "export it before running this bot."
        )
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("idea", cmd_idea))
    app.add_handler(CommandHandler("brew", cmd_brew))
    app.add_handler(CommandHandler("brews", cmd_brews))
    app.add_handler(CommandHandler("einstein", cmd_einstein))
    app.add_handler(CommandHandler("lsd", cmd_lsd))
    app.add_handler(CommandHandler("futures", cmd_futures))
    app.add_handler(CommandHandler("dream", cmd_dream))
    app.add_handler(CommandHandler("lucid", cmd_lucid))
    app.add_handler(CommandHandler("modes", cmd_modes))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("set", cmd_set))
    app.add_handler(CommandHandler("usage", cmd_usage))
    app.add_handler(CommandHandler("quota", cmd_quota))
    app.add_handler(CommandHandler("resetquota", cmd_resetquota))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("whoami", cmd_whoami))
    app.add_handler(CommandHandler("corpus", cmd_corpus))
    app.add_handler(CommandHandler("feedback", cmd_feedback))
    app.add_handler(CommandHandler("bootstrap", cmd_bootstrap))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_plain_message))
    # Yes/No buttons for the plain-text confirmation flow.
    app.add_handler(CallbackQueryHandler(
        on_callback,
        pattern=r"^(idea:(confirm|cancel)|brew:(now|3h|8h|cancel|drop_\d+))$",
    ))
    # Sub-pickers opened from the bottom-attached menu (Mode/Entropy/Depth).
    app.add_handler(CallbackQueryHandler(
        on_menu_callback,
        pattern=r"^(mode:|entropy:|depth:)",
    ))
    # Scheduled-Brew queue: init the SQLite store and start the poller.
    # Runs every 60s, processes up to 5 due brews per tick serially.
    try:
        import brew_queue
        brew_queue.init_db()
        if app.job_queue is not None:
            app.job_queue.run_repeating(
                brew_poll_callback,
                interval=60.0,
                first=15.0,
                name="brew_poll",
            )
            log.info("brew queue + poller registered")
        else:
            log.warning("PTB JobQueue unavailable — scheduled brews disabled")
    except Exception:
        log.exception("brew queue init failed — scheduled brews disabled")

    # Log group: pre-create the topics shortly after startup, post a daily
    # rollup at 18:00 UTC, and a subscription-lapse watch at 09:00 UTC. All
    # no-op when AIDEA_LOG_CHAT_ID is unset.
    if app.job_queue is not None and _log_chat_id() is not None:
        app.job_queue.run_once(ensure_log_topics, when=5, name="ensure_log_topics")
        app.job_queue.run_daily(
            daily_summary_callback,
            time=dtime(hour=18, minute=0, tzinfo=timezone.utc),
            name="log_daily_summary",
        )
        app.job_queue.run_daily(
            subscription_lapse_callback,
            time=dtime(hour=9, minute=0, tzinfo=timezone.utc),
            name="subscription_lapse_watch",
        )
        log.info("log-group topics + daily summary + lapse watch scheduled")
    return app


def main() -> None:
    app = build_app()
    log.info("AIdea Telegram bot starting — polling Telegram for updates.")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
