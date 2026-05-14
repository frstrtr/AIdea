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
    EINSTEIN_MECHANISMS,
    FUTURES_HORIZONS,
    LSD_LABEL,
    build_einstein_prompt,
    build_futures_prompt,
    build_lsd_prompt,
    build_prompt,
    critic_score,
    load_or_generate_deck,
    parse_entropy,
    refine_idea,
    sample_cards,
    synthesize,
    total_score,
)
from usage import start_run, summarize

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    level=logging.INFO,
)
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
    mode: str,  # 'default' | 'einstein' | 'lsd' | 'futures'
) -> None:
    chat_id = update.effective_chat.id
    state = state_for(chat_id)
    if state.busy:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Already running a task in this chat. Use /cancel to abort it.",
        )
        return

    state.busy = True
    state.cancel.clear()
    run_id = start_run(f"tg-{chat_id}")
    s = state.settings
    spread, level = parse_entropy(s.entropy)
    depth = CARD_DEPTH_BY_NAME[s.card_depth]

    await context.bot.send_message(
        chat_id=chat_id,
        text=(
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

        # Decide passes
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
            else:
                prompt = build_prompt(topic, drawn, level)

            label_for_user = label or f"idea {i + 1}"
            idea = await run_with_progress(
                synthesize(prompt=prompt, model=s.model, stream_to_stdout=False),
                update=update, context=context, cancel=state.cancel,
                headline=f"synthesizing {label_for_user}",
            )
            ideas.append(idea)
            cards_per_idea.append(drawn)
            mech_labels.append(label)

            header = (
                f"Idea {i + 1}" +
                (f" — {label}" if label else "")
            )
            await send_idea(update, context, header, idea)

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

    except asyncio.CancelledError:
        return
    except Exception as e:
        log.exception("pipeline failed")
        await context.bot.send_message(
            chat_id=chat_id, text=f"❌ pipeline failed: {e}",
        )
    finally:
        state.busy = False
        state.cancel.clear()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


HELP_TEXT = (
    "AIdea — applied-ideas synthesizer with entropy controls.\n\n"
    "Commands:\n"
    "/idea <topic>      one idea at the current entropy\n"
    "/einstein <topic>  four mechanism-specific ideas (Adjacent Possible, "
    "Exaptation, Slow Hunch, Productive Error)\n"
    "/lsd <topic>       prior dissolution — re-perceive under a different frame\n"
    "/futures <topic>   temporal projection (+1y/+3y/+10y/+30y)\n"
    "/settings          show current entropy/deck/depth/refine knobs\n"
    "/set <k>=<v>       tune one knob (entropy, cards, card_depth, n_concepts, "
    "n_ideas, refine, evolve_deck, seed)\n"
    "/usage             LLM usage + subscription-window state\n"
    "/cancel            abort the current task in this chat\n"
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


async def cmd_cancel(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    state = state_for(update.effective_chat.id)
    if not state.busy:
        await update.message.reply_text("Nothing to cancel — no active task.")
        return
    state.cancel.set()
    await update.message.reply_text("🛑 Cancel signalled.")


async def cmd_fallback(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "I respond to slash commands. Try /help to see what's available.",
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
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("set", cmd_set))
    app.add_handler(CommandHandler("usage", cmd_usage))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, cmd_fallback))
    return app


def main() -> None:
    app = build_app()
    log.info("AIdea Telegram bot starting — polling Telegram for updates.")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
