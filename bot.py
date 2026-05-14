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
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from the project directory before anything else reads env vars.
load_dotenv(Path(__file__).parent / ".env")

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
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
from usage import start_run, summarize

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
    entropy: str = "wild"
    cards: int = 30
    card_depth: str = "medium"
    n_concepts: int = 3
    n_ideas: int = 1
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
    "default": 35.0,
    "einstein": 90.0,
    "lsd": 65.0,
    "futures": 90.0,
    "dream": 40.0,
    "lucid": 40.0,
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

        # Stage 3: critic + refine
        if s.refine and ideas:
            scored: list[dict] = []
            for i, idea in enumerate(ideas):
                score = await run_with_progress(
                    critic_score(topic, idea, s.model),
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
                        f"topic-fit={score['topic_fit']}, "
                        f"total={score['total']}/300\n"
                        f"notes: {score.get('notes', '')}"
                    ),
                )

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
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🏆 Winner: idea {winner['i'] + 1}"
                    + (f" [{wmech}]" if wmech else "")
                    + f" — {winner['total']}/300. Refining at low entropy..."
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

    except asyncio.CancelledError:
        transcript_log("request_errored", error="cancelled")
        return
    except Exception as e:
        log.exception("pipeline failed")
        transcript_log(
            "request_errored", error_type=type(e).__name__, error=str(e),
        )
        await context.bot.send_message(
            chat_id=chat_id, text=f"❌ pipeline failed: {e}",
        )
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
    "Tip: just send a plain message — it's treated as /idea <your text>.\n\n"
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
    await update.message.reply_text(HELP_TEXT)


async def cmd_help(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT)


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
    u = summarize(run_id=None)
    lines = []
    for label, key in (
        ("This week", "last_7d"),
        ("This month", "last_30d"),
        ("All time", "total"),
    ):
        b = u.get(key, {})
        lines.append(
            f"{label}: {b.get('calls', 0)} calls · "
            f"{fmt_tokens(b.get('input_tokens', 0))} in / "
            f"{fmt_tokens(b.get('output_tokens', 0))} out · "
            f"{fmt_ms(b.get('duration_ms', 0))} · "
            f"{fmt_usd(b.get('total_cost_usd', 0))}"
        )
    lines.append(
        f"5h windows touched (last 7d): {u.get('five_h_windows_last_7d', 0)} "
        "(local heuristic; not subscription truth)"
    )
    rl = u.get("rate_limit")
    if rl:
        status = rl.get("status", "?")
        util = rl.get("utilization")
        util_s = f"{round(util * 100)}%" if isinstance(util, (int, float)) else "—"
        resets = ""
        if rl.get("resets_at"):
            dt = rl["resets_at"] - time.time()
            if dt > 0:
                h, m = int(dt // 3600), int((dt % 3600) // 60)
                resets = (
                    f", resets in {h}h {m}m"
                    if h else f", resets in {m}m"
                )
        lines.append(
            f"Subscription window ({rl.get('type', '?')}): "
            f"status={status}, utilization={util_s}{resets}"
        )
    else:
        lines.append("Subscription window: no rate-limit event observed yet")
    lines.append("")
    lines.append(u.get("note", ""))
    await update.message.reply_text("\n".join(lines))


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


async def on_plain_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Treat any non-command text message as a default /idea request."""
    topic = (update.message.text or "").strip()
    if not topic:
        await update.message.reply_text(HELP_TEXT)
        return
    await run_pipeline_for_telegram(
        update=update, context=ctx, topic=topic, mode="default",
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
    app.add_handler(CommandHandler("einstein", cmd_einstein))
    app.add_handler(CommandHandler("lsd", cmd_lsd))
    app.add_handler(CommandHandler("futures", cmd_futures))
    app.add_handler(CommandHandler("dream", cmd_dream))
    app.add_handler(CommandHandler("lucid", cmd_lucid))
    app.add_handler(CommandHandler("modes", cmd_modes))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("set", cmd_set))
    app.add_handler(CommandHandler("usage", cmd_usage))
    app.add_handler(CommandHandler("corpus", cmd_corpus))
    app.add_handler(CommandHandler("feedback", cmd_feedback))
    app.add_handler(CommandHandler("bootstrap", cmd_bootstrap))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_plain_message))
    return app


def main() -> None:
    app = build_app()
    log.info("AIdea Telegram bot starting — polling Telegram for updates.")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
