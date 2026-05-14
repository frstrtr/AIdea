"""Local web interface for AIdea.

Run:
  .venv/bin/uvicorn server:app --reload --port 8000
Then open http://localhost:8000

Streaming uses Server-Sent Events over a POST body (fetch + ReadableStream
in the browser) so the page can show progress through the three pipeline
stages without blocking on a single long wait.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable

from dotenv import load_dotenv

# Load .env from the project directory before anything else reads env vars.
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from aidea import (
    CARD_DEPTH_BY_NAME,
    CARD_DEPTHS,
    DREAM_LABEL,
    EINSTEIN_MECHANISMS,
    ENTROPY_LEVELS,
    FUTURES_HORIZONS,
    FUTURES_HORIZONS_BY_KEY,
    LSD_LABEL,
    LSD_VALIDATION_LABEL,
    LUCID_LABEL,
    Card,
    build_dream_prompt,
    build_einstein_prompt,
    build_futures_prompt,
    build_lsd_prompt,
    build_lucid_prompt,
    build_prompt,
    cards_from_static_bank,
    critic_score,
    evolve_cards,
    load_bank,
    load_or_generate_deck,
    lsd_validate,
    merge_evolved_into_deck,
    parse_entropy,
    refine_idea,
    sample_cards,
    save_deck_to_cache,
    synthesize,
    total_score,
)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    entropy: str = "wild"
    cards: int = Field(30, ge=4, le=500)
    card_depth: str = "medium"
    n_concepts: int = Field(3, ge=2, le=10)
    n_ideas: int = Field(1, ge=1, le=5)
    seed: int | None = None
    model: str = "claude-opus-4-7"
    regen_deck: bool = False
    bank: str | None = None  # path to static JSON bank; bypasses deck-gen
    bank_data: dict[str, list[str]] | None = None  # inline static bank
    refine: bool = False  # score ideas + refine the winner at low entropy
    einstein: bool = False  # four mechanism-specific passes instead of n_ideas
    lsd: bool = False  # prior-dissolution passes (mutually exclusive with einstein)
    futures: bool = False  # four temporal-horizon passes (mut. excl. w/ einstein/lsd)
    dream: bool = False  # dreaming mode: feasibility-off, dream-image output
    lucid: str | None = None  # lucid mode: directional prior the dream biases toward
    themes: list[str] | None = None  # explicit theme override for deck-gen
    theme_entropy: float | None = None  # 0..1; defaults to spread of entropy field
    evolve_deck: bool = False  # rewrite winning cards back into the deck cache


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


HEARTBEAT_INTERVAL = 1.5  # seconds


async def _watched(
    awaitable: Awaitable[Any], phase: str
) -> AsyncIterator[tuple[str, Any]]:
    """Drive an awaitable while yielding ('progress', elapsed) ticks.

    Final yield is ('result', value). Lets the SSE generator emit live
    elapsed-time updates during the long LLM calls so the UI knows the
    request is alive even when the underlying SDK delivers its response
    as a single chunk (no token streaming through the agent SDK).
    """
    task = asyncio.create_task(awaitable)
    start = time.monotonic()
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=HEARTBEAT_INTERVAL)
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            yield "progress", {"phase": phase, "elapsed": round(elapsed, 1)}
    result = await task
    yield "result", result


def _card_to_dict(c: Card) -> dict:
    return {k: v for k, v in c.__dict__.items() if v is not None}


# ---------------------------------------------------------------------------
# The pipeline as an SSE stream
# ---------------------------------------------------------------------------


async def event_stream(req: GenerateRequest) -> AsyncIterator[bytes]:
    from usage import start_run, summarize as usage_summarize
    from transcripts import set_source, log_event as transcript_log
    run_id = start_run("web")
    set_source("web")
    yield _sse("run", {"run_id": run_id})

    mode = (
        "einstein" if req.einstein else
        "lsd" if req.lsd else
        "futures" if req.futures else
        "dream" if req.dream else
        "lucid" if req.lucid else
        "default"
    )
    transcript_log(
        "request_started",
        topic=req.topic, mode=mode,
        entropy=req.entropy, cards=req.cards, card_depth=req.card_depth,
        n_concepts=req.n_concepts, n_ideas=req.n_ideas,
        seed=req.seed, model=req.model,
        refine=bool(req.refine), evolve_deck=bool(req.evolve_deck),
        bank=req.bank, has_bank_data=req.bank_data is not None,
        regen_deck=bool(req.regen_deck),
    )

    modes_on = sum(
        1 for f in (req.einstein, req.lsd, req.futures, req.dream, bool(req.lucid)) if f
    )
    if modes_on > 1:
        yield _sse("error", {
            "message": (
                "einstein / lsd / futures / dream / lucid modes are "
                "mutually exclusive; pick at most one."
            ),
        })
        return
    if req.evolve_deck and req.bank_data is not None:
        yield _sse("error", {
            "message": "evolve_deck requires a generated deck; not compatible with bank_data.",
        })
        return
    if req.evolve_deck and req.bank:
        yield _sse("error", {
            "message": "evolve_deck requires a generated deck; not compatible with bank (static).",
        })
        return

    try:
        spread, level = parse_entropy(req.entropy)
    except SystemExit as e:
        yield _sse("error", {"message": str(e)})
        return

    if req.card_depth not in CARD_DEPTH_BY_NAME:
        yield _sse("error", {"message": f"unknown card_depth {req.card_depth!r}"})
        return
    depth = CARD_DEPTH_BY_NAME[req.card_depth]

    # --- Stage 0: deck -----------------------------------------------------
    if req.bank_data is not None:
        yield _sse("status", {
            "phase": "deck",
            "message": "Loading user-supplied inline bank",
        })
        try:
            deck = cards_from_static_bank(req.bank_data)
        except Exception as e:
            yield _sse("error", {"message": f"inline bank invalid: {e}"})
            return
    elif req.bank:
        yield _sse("status", {
            "phase": "deck",
            "message": f"Loading static bank: {req.bank}",
        })
        try:
            deck = cards_from_static_bank(load_bank(req.bank))
        except Exception as e:
            yield _sse("error", {"message": f"deck stage failed: {e}"})
            return
    else:
        yield _sse("status", {
            "phase": "deck",
            "message": (
                f"Generating donor deck "
                f"(n={req.cards}, depth={depth.name}). "
                f"At depth={depth.name} expect 30-120s on first run; "
                f"cached after."
            ),
        })
        deck = None
        _theme_entropy = (
            req.theme_entropy
            if req.theme_entropy is not None
            else spread
        )
        try:
            async for kind, payload in _watched(
                load_or_generate_deck(
                    topic=req.topic,
                    n=req.cards,
                    depth=depth,
                    model=req.model,
                    force_regen=req.regen_deck,
                    verbose=False,
                    theme_entropy=_theme_entropy,
                    themes=req.themes,
                ),
                phase="deck",
            ):
                if kind == "progress":
                    yield _sse("progress", payload)
                else:
                    deck = payload
        except Exception as e:
            yield _sse("error", {"message": f"deck stage failed: {e}"})
            return
        assert deck is not None

    yield _sse("deck", {
        "size": len(deck),
        "depth": depth.name,
        "cards": [_card_to_dict(c) for c in deck],
    })
    transcript_log(
        "deck",
        size=len(deck),
        depth=depth.name,
        cards=[_card_to_dict(c) for c in deck],
    )

    # --- Stage 1 & 2: sample + synthesize, per idea -----------------------
    rng = random.Random(req.seed)
    ideas: list[str] = []
    mechanism_labels: list[str | None] = []
    cards_per_idea: list[list[Card]] = []

    if req.einstein:
        passes = [
            (("einstein", key), mech["label"], mech["blurb"])
            for key, mech in EINSTEIN_MECHANISMS.items()
        ]
    elif req.dream:
        dream_blurb = (
            "Dreaming: prediction-error offline. Generative model runs "
            "free, no feasibility constraint. Output is a dream image + "
            "what survives waking."
        )
        passes = [
            (("dream", None), DREAM_LABEL, dream_blurb)
            for _ in range(req.n_ideas)
        ]
    elif req.lucid:
        lucid_blurb = (
            "Lucid Dreaming: relaxed feasibility + a directional prior "
            "the dream biases toward, with one reality check."
        )
        passes = [
            (("lucid", req.lucid), LUCID_LABEL, lucid_blurb)
            for _ in range(req.n_ideas)
        ]
    elif req.lsd:
        lsd_blurb = (
            "Predictive processing: perception is constructed from priors. "
            "Loosen the field's interpretive prior and re-perceive."
        )
        passes = [
            (("lsd", None), LSD_LABEL, lsd_blurb)
            for _ in range(req.n_ideas)
        ]
    elif req.futures:
        passes = [
            (("futures", h["key"]), h["label"], h["framing"])
            for h in FUTURES_HORIZONS
        ]
    else:
        passes = [((None, None), None, None) for _ in range(req.n_ideas)]

    for i, ((mode, mech_key), mech_label, mech_blurb) in enumerate(passes):
        cards = sample_cards(deck=deck, n=req.n_concepts, spread=spread, rng=rng)
        yield _sse("sample", {
            "i": i,
            "level": level.name,
            "spread": spread,
            "mechanism": mech_label,
            "mechanism_blurb": mech_blurb,
            "cards": [_card_to_dict(c) for c in cards],
        })

        synth_phase = f"synth-{i}"
        synth_msg = (
            f"Synthesizing idea {i + 1}/{len(passes)} "
            f"via {mech_label} mechanism..."
            if mech_label
            else f"Synthesizing idea {i + 1}/{len(passes)}..."
        )
        yield _sse("status", {
            "phase": synth_phase,
            "i": i,
            "message": synth_msg,
        })

        if mode == "einstein":
            prompt = build_einstein_prompt(req.topic, cards, mech_key)
        elif mode == "lsd":
            prompt = build_lsd_prompt(req.topic, cards)
        elif mode == "futures":
            prompt = build_futures_prompt(req.topic, cards, mech_key)
        elif mode == "dream":
            prompt = build_dream_prompt(req.topic, cards)
        elif mode == "lucid":
            prompt = build_lucid_prompt(req.topic, cards, mech_key or "")
        else:
            prompt = build_prompt(req.topic, cards, level)

        idea: str | None = None
        try:
            async for kind, payload in _watched(
                synthesize(prompt=prompt, model=req.model, stream_to_stdout=False),
                phase=synth_phase,
            ):
                if kind == "progress":
                    yield _sse("progress", payload)
                else:
                    idea = payload
        except Exception as e:
            yield _sse("error", {"message": f"synthesis failed: {e}"})
            return
        assert idea is not None

        # Emit the anarchic idea (visible to the user even before sober pass).
        yield _sse("idea", {
            "i": i,
            "text": idea,
            "mechanism": mech_label,
        })
        transcript_log(
            "idea", i=i,
            mechanism=mech_label,
            text=idea,
            cards=[_card_to_dict(c) for c in cards],
        )

        # LSD has a mandatory second pass: sober validation. Anything else
        # (default / einstein / futures / dream / lucid) emits the idea as-is.
        if mode == "lsd":
            validate_phase = f"lsd-validate-{i}"
            yield _sse("status", {
                "phase": validate_phase,
                "i": i,
                "message": (
                    f"Sober validation pass {i + 1}/{len(passes)} — priors "
                    "back online; filtering what survives..."
                ),
            })
            sober: str | None = None
            try:
                async for kind, payload in _watched(
                    lsd_validate(req.topic, idea, req.model),
                    phase=validate_phase,
                ):
                    if kind == "progress":
                        yield _sse("progress", payload)
                    else:
                        sober = payload
            except Exception as e:
                yield _sse("error", {"message": f"lsd validation failed: {e}"})
                return
            assert sober is not None
            yield _sse("lsd_validation", {
                "i": i,
                "anarchic": idea,
                "sober": sober,
            })
            transcript_log(
                "lsd_validation", i=i, anarchic=idea, sober=sober,
            )
            # Use the sober version downstream (critic/refine sees this).
            ideas.append(sober)
            mechanism_labels.append(LSD_VALIDATION_LABEL)
        else:
            ideas.append(idea)
            mechanism_labels.append(mech_label)
        cards_per_idea.append(cards)

    # --- Stage 3: critic + refinement (optional) --------------------------
    if req.refine and ideas:
        yield _sse("status", {
            "phase": "critic",
            "message": (
                f"Scoring {len(ideas)} idea(s) on feasibility / "
                f"unexpectedness / topic fit..."
            ),
        })
        scored: list[dict] = []
        for i, idea in enumerate(ideas):
            score: dict | None = None
            try:
                async for kind, payload in _watched(
                    critic_score(req.topic, idea, req.model),
                    phase=f"critic-{i}",
                ):
                    if kind == "progress":
                        yield _sse("progress", payload)
                    else:
                        score = payload
            except Exception as e:
                yield _sse("error", {"message": f"critic failed: {e}"})
                return
            assert score is not None
            score["i"] = i
            score["total"] = total_score(score)
            score["mechanism"] = (
                mechanism_labels[i]
                if i < len(mechanism_labels) else None
            )
            scored.append(score)
            yield _sse("score", score)
            transcript_log(
                "score", i=i,
                mechanism=score.get("mechanism"),
                feasibility=score.get("feasibility"),
                unexpectedness=score.get("unexpectedness"),
                topic_fit=score.get("topic_fit"),
                total=score.get("total"),
                notes=score.get("notes", ""),
            )

        winner = max(scored, key=lambda s: s["total"])
        yield _sse("winner", {
            "i": winner["i"],
            "total": winner["total"],
            "notes": winner.get("notes", ""),
            "mechanism": winner.get("mechanism"),
        })
        transcript_log(
            "winner",
            i=winner["i"], total=winner["total"],
            mechanism=winner.get("mechanism"),
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
                source="web",
            )
        except Exception:
            pass

        yield _sse("status", {
            "phase": "refine",
            "message": (
                f"Refining idea {winner['i'] + 1} at low entropy "
                "to harden the first-step and risks..."
            ),
        })
        refined: str | None = None
        try:
            async for kind, payload in _watched(
                refine_idea(
                    topic=req.topic,
                    idea=ideas[winner["i"]],
                    notes=winner.get("notes", ""),
                    model=req.model,
                ),
                phase="refine",
            ):
                if kind == "progress":
                    yield _sse("progress", payload)
                else:
                    refined = payload
        except Exception as e:
            yield _sse("error", {"message": f"refine failed: {e}"})
            return
        assert refined is not None

        yield _sse("refined", {"i": winner["i"], "text": refined})
        transcript_log("refined", i=winner["i"], text=refined)

        # --- Stage 4: deck evolution (opt-in) ----------------------------
        if req.evolve_deck and req.bank is None and req.bank_data is None:
            winning_cards = cards_per_idea[winner["i"]]
            yield _sse("status", {
                "phase": "evolve",
                "message": (
                    f"Sharpening {len(winning_cards)} winning card(s) "
                    "and writing them back to the deck cache..."
                ),
            })
            evolved: list[Card] | None = None
            try:
                async for kind, payload in _watched(
                    evolve_cards(
                        topic=req.topic,
                        idea=refined,
                        cards=winning_cards,
                        depth=depth,
                        model=req.model,
                    ),
                    phase="evolve",
                ):
                    if kind == "progress":
                        yield _sse("progress", payload)
                    else:
                        evolved = payload
            except Exception as e:
                yield _sse("error", {"message": f"evolve failed: {e}"})
                return
            if evolved:
                deck = merge_evolved_into_deck(deck, evolved)
                try:
                    save_deck_to_cache(
                        req.topic, req.cards, depth, req.model, deck,
                    )
                except Exception as e:
                    yield _sse("error", {
                        "message": f"deck cache write failed: {e}",
                    })
                    return
                pairs = [
                    {
                        "before": _card_to_dict(o),
                        "after": _card_to_dict(n),
                    }
                    for o, n in zip(winning_cards, evolved)
                ]
                yield _sse("evolved", {"pairs": pairs})
                transcript_log("evolved", pairs=pairs)

    try:
        yield _sse("usage", usage_summarize(run_id=run_id))
    except Exception:
        pass

    transcript_log("request_completed", n_ideas=len(ideas), refined=bool(req.refine))
    yield _sse("done", {})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(title="AIdea")


@app.get("/api/meta")
async def meta():
    return {
        "levels": [
            {"name": l.name, "spread": l.spread, "instruction": l.instruction}
            for l in ENTROPY_LEVELS
        ],
        "depths": [
            {
                "name": d.name,
                "target_tokens": d.target_tokens,
                "fields": list(d.fields),
                "description": d.description,
            }
            for d in CARD_DEPTHS
        ],
    }


@app.get("/api/usage")
async def usage_endpoint() -> dict:
    """LLM usage summary: this-run is empty (no run scope on a bare GET),
    last-7d / last-30d / total reflect the local usage.jsonl log, and
    rate_limit reflects the most recently observed RateLimitEvent."""
    from usage import summarize as usage_summarize
    return usage_summarize(run_id=None)


@app.get("/api/corpus")
async def corpus_endpoint(source: str | None = None) -> dict:
    """RAG corpus health: card count, sources, modes, quality-labelled
    fraction. ``?source=...`` returns the tenant-scoped view; omitting
    returns the aggregate."""
    from rag import stats as rag_stats
    return rag_stats(source=source)


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if req.cards < req.n_concepts:
        raise HTTPException(400, "cards must be >= n_concepts")
    return StreamingResponse(event_stream(req), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


# ---------------------------------------------------------------------------
# UI (single-file: HTML + CSS + JS inline)
# ---------------------------------------------------------------------------


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AIdea</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    max-width: 920px; margin: 2rem auto; padding: 0 1rem;
    line-height: 1.45; color: #1a1a1a;
  }
  h1 { font-weight: 600; margin-bottom: 0.2rem; }
  .sub { color: #666; margin-top: 0; margin-bottom: 1.5rem; font-size: 0.95rem; }
  form { display: grid; gap: 0.75rem; }
  label { display: flex; flex-direction: column; font-size: 0.85rem; color: #555; gap: 0.25rem; }
  label.inline { flex-direction: row; align-items: center; gap: 0.5rem; }
  .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; }
  input, select, textarea, button {
    font: inherit; padding: 0.45rem 0.6rem;
    border: 1px solid #ccc; border-radius: 4px; background: #fff; color: inherit;
  }
  textarea { min-height: 4.5rem; resize: vertical; font-family: inherit; }
  button {
    background: #1a1a1a; color: #fff; cursor: pointer; padding: 0.55rem 1.4rem;
    border: none; font-weight: 500;
  }
  button:disabled { background: #999; cursor: progress; }
  .panel {
    border: 1px solid #e2e2e2; border-radius: 6px;
    padding: 0.9rem 1.1rem; margin-top: 0.9rem; background: #fafafa;
  }
  .status { color: #555; font-size: 0.92rem; display: flex;
            align-items: baseline; gap: 0.5rem; flex-wrap: wrap; }
  .status .dot {
    width: 0.55rem; height: 0.55rem; border-radius: 50%;
    background: #1a8a4f; flex: 0 0 auto;
    animation: pulse 1.1s ease-in-out infinite;
    transform: translateY(0.05rem);
  }
  .status.done .dot { background: #888; animation: none; }
  .status.error .dot { background: #b00020; animation: none; }
  .status .msg { font-style: italic; }
  .status .elapsed { color: #888; font-family: ui-monospace, "SF Mono",
                     Menlo, monospace; font-size: 0.85rem; }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: translateY(0.05rem) scale(1); }
    50%      { opacity: 0.35; transform: translateY(0.05rem) scale(0.7); }
  }
  details { margin: 0; }
  summary { cursor: pointer; color: #444; font-size: 0.9rem; font-weight: 500; }
  details[open] summary { margin-bottom: 0.6rem; }
  .card { font-size: 0.92rem; margin: 0.45rem 0; padding-left: 0.4rem; border-left: 2px solid #ddd; }
  .card .name { font-weight: 600; }
  .card .domain { color: #888; font-size: 0.85rem; margin-left: 0.3rem; }
  .card .body { color: #444; font-size: 0.85rem; margin-left: 1rem; }
  .card .body i { color: #888; font-style: normal; font-variant: small-caps; }
  .idea {
    background: #fff; border-left: 3px solid #1a1a1a;
    padding: 0.9rem 1.1rem; margin-top: 0.5rem; border-radius: 0 4px 4px 0;
  }
  .idea pre {
    white-space: pre-wrap; word-break: break-word;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 0.92rem; margin: 0;
  }
  .meta { color: #777; font-size: 0.8rem; margin-bottom: 0.3rem; }
  .meta.refined-meta { color: #1a8a4f; font-weight: 600; letter-spacing: 0.05em;
                       text-transform: uppercase; }
  .mech-tag { display: inline-block; margin-left: 0.4rem; padding: 0.05rem 0.45rem;
              background: #2c3e75; color: #fff; border-radius: 999px;
              font-size: 0.7rem; letter-spacing: 0.04em; text-transform: uppercase;
              font-weight: 600; }
  .mech-tag.lsd {
    background: linear-gradient(90deg, #6a2c75, #c9396a, #f0a544);
    color: #fff; text-shadow: 0 0 4px rgba(0,0,0,0.4);
  }
  .mech-tag.futures {
    background: linear-gradient(90deg, #1e6b6b, #2e8c63, #b59500);
    color: #fff; text-shadow: 0 0 4px rgba(0,0,0,0.4);
  }
  .mech-tag.dream {
    background: linear-gradient(90deg, #102050, #43306e, #6b3ea3);
    color: #fff; text-shadow: 0 0 4px rgba(0,0,0,0.5);
  }
  .mech-tag.lucid {
    background: linear-gradient(90deg, #4a2a8c, #8c3aa6, #d36ab9);
    color: #fff; text-shadow: 0 0 4px rgba(0,0,0,0.4);
  }
  .mech-blurb { color: #666; font-size: 0.85rem; font-style: italic;
                margin: 0.15rem 0 0.45rem; }
  .panel.usage { position: sticky; bottom: 0.6rem; margin-top: 1.2rem;
                 background: #fbfbf7; }
  .usage-refresh { cursor: pointer; color: #888; margin-left: 0.4rem;
                   font-size: 0.85rem; user-select: none; }
  .usage-refresh:hover { color: #1a1a1a; }
  .usage-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                gap: 0.5rem; margin-top: 0.4rem; }
  .usage-card { background: #fff; border: 1px solid #eee; border-radius: 6px;
                padding: 0.45rem 0.6rem; font-size: 0.85rem;
                font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  .usage-card .label { color: #888; text-transform: uppercase;
                       letter-spacing: 0.05em; font-size: 0.68rem;
                       font-weight: 600; font-family: ui-sans-serif, system-ui; }
  .usage-card .val { font-size: 1rem; margin-top: 0.15rem; }
  .usage-card .sub { color: #888; font-size: 0.75rem; margin-top: 0.1rem; }
  .usage-note { color: #888; font-size: 0.75rem; font-style: italic;
                margin-top: 0.4rem; }
  .usage-rate-ok { color: #1a8a4f; }
  .usage-rate-warn { color: #b59500; }
  .usage-rate-bad { color: #b03a2e; }
  @media (prefers-color-scheme: dark) {
    .panel.usage { background: #161616; }
    .usage-card { background: #1a1a1a; border-color: #333; }
    .usage-card .label, .usage-card .sub, .usage-note { color: #aaa; }
  }
  .evolve-card { margin: 0.6rem 0; padding: 0.5rem 0.7rem;
                 border-left: 3px solid #1a8a4f; background: #f4faf6;
                 border-radius: 0 4px 4px 0; }
  .evolve-name { font-weight: 600; font-size: 0.95rem; }
  .evolve-domain { color: #888; font-size: 0.82rem; font-weight: 400; }
  .evolve-field { margin-top: 0.3rem; font-size: 0.86rem; }
  .evolve-field b { color: #888; font-weight: 600; text-transform: uppercase;
                    letter-spacing: 0.04em; font-size: 0.72rem;
                    margin-right: 0.3rem; }
  .evolve-before { color: #a33; font-family: ui-monospace, monospace;
                   font-size: 0.85rem; margin: 0.1rem 0 0.1rem 1rem; }
  .evolve-after { color: #1a8a4f; font-family: ui-monospace, monospace;
                  font-size: 0.85rem; margin: 0 0 0 1rem; }
  @media (prefers-color-scheme: dark) {
    .evolve-card { background: #0f1a13; border-left-color: #3acb78; }
    .evolve-before { color: #f08080; }
    .evolve-after { color: #3acb78; }
  }
  .error { color: #b00020; border-left-color: #b00020; }
  .score-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.2rem 0 0.4rem; }
  .score-pill { background: #eee; color: #333; border-radius: 999px;
                padding: 0.15rem 0.6rem; font-size: 0.82rem;
                font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  .score-pill.total { background: #1a1a1a; color: #fff; }
  .critic-notes { color: #555; font-size: 0.88rem; font-style: italic;
                  margin-top: 0.2rem; }
  .winner-banner { background: #fff8c2; border-left: 3px solid #b59500;
                   padding: 0.45rem 0.7rem; font-weight: 600;
                   font-size: 0.95rem; border-radius: 0 4px 4px 0; }
  .meta.lsd-sober-meta { color: #5a6ad6; font-weight: 600; letter-spacing: 0.05em;
                         text-transform: uppercase; }
  .idea.lsd-sober { border-left: 4px solid #5a6ad6; background: #f5f6fb; }
  @media (prefers-color-scheme: dark) {
    .meta.lsd-sober-meta { color: #93a4ff; }
    .idea.lsd-sober { background: #0f1129; border-left-color: #93a4ff; }
  }
  .idea.refined { border-left-width: 4px; border-left-color: #1a8a4f;
                  background: #f4faf6; }
  @media (prefers-color-scheme: dark) {
    body { color: #e5e5e5; }
    input, select, textarea { background: #1e1e1e; border-color: #444; color: #e5e5e5; }
    button { background: #e5e5e5; color: #1a1a1a; }
    .panel { background: #1a1a1a; border-color: #333; }
    .idea { background: #111; border-left-color: #e5e5e5; }
    .idea.refined { background: #0f1a13; border-left-color: #3acb78; }
    .meta.refined-meta { color: #3acb78; }
  }
  details.modes-legend { margin-bottom: 1.2rem; }
  details.modes-legend > summary {
    cursor: pointer; font-weight: 600; padding: 0.4rem 0;
    color: var(--fg, #1a1a1a);
  }
  details.modes-legend table {
    width: 100%; border-collapse: collapse; margin-top: 0.6rem;
    font-size: 0.92rem;
  }
  details.modes-legend th,
  details.modes-legend td {
    text-align: left; vertical-align: top;
    padding: 0.55rem 0.7rem;
    border-top: 1px solid #eaeaea;
  }
  details.modes-legend th {
    font-size: 0.78rem; letter-spacing: 0.05em;
    text-transform: uppercase; color: #888;
    border-top: none;
  }
  details.modes-legend td:first-child { width: 8rem; }
  details.modes-legend code { font-size: 0.92em; }
  .legend-cite {
    margin-top: 0.4rem; padding-top: 0.35rem;
    border-top: 1px dashed #eaeaea;
    color: #888; font-size: 0.82rem; font-style: italic;
  }
  @media (prefers-color-scheme: dark) {
    details.modes-legend th,
    details.modes-legend td { border-top-color: #2a2a2a; }
    .legend-cite { border-top-color: #2a2a2a; color: #999; }
    .sub, .status, summary, .card .domain, .critic-notes { color: #aaa; }
    .score-pill { background: #2a2a2a; color: #ddd; }
    .score-pill.total { background: #e5e5e5; color: #1a1a1a; }
    .winner-banner { background: #1a1a0a; border-left-color: #b59500;
                     color: #f0e0a0; }
  }
</style>
</head>
<body>
<h1>AIdea</h1>
<p class="sub">
  Inference-driven entropy idea generator. Three orthogonal knobs:
  <b>deck size</b> (raw material pool), <b>card depth</b> (pre-seeded detail),
  <b>entropy</b> (how cross-domain the shuffle and how far the synthesis may
  depart from current practice). Feasibility is enforced at every level
  except Dream / Lucid.
</p>

<details class="modes-legend">
  <summary>Modes — pick at most one</summary>
  <table>
    <tr><th>Mode</th><th>What it does</th></tr>
    <tr><td><code>default</code></td><td>
      Applied conceptual blending. Generate <code>n_ideas</code> ideas at
      the requested entropy. Picks 1+ donor concepts whose structural
      mechanism transfers onto your topic. Feasibility required.
      <div class="legend-cite">Fauconnier &amp; Turner, <i>The Way We Think</i> (2002); Kauffman, <i>Investigations</i> (2000).</div>
    </td></tr>
    <tr><td><code>Einstein</code></td><td>
      Four passes, one per generative mechanism Steven Johnson catalogues
      in <i>Where Good Ideas Come From</i> (2010):
      <b>Adjacent Possible</b> (Kauffman 2000 — walk through a just-unlocked door),
      <b>Exaptation</b> (Gould &amp; Vrba, <i>Paleobiology</i> 1982 — transplant a mechanism from a far-distant field),
      <b>Slow Hunch</b> (Johnson 2010 — articulate a latent tension),
      <b>Productive Error</b> (Fleming 1928 — invert a load-bearing assumption).
      <div class="legend-cite">Johnson, <i>Where Good Ideas Come From</i> (2010).</div>
    </td></tr>
    <tr><td><code>LSD</code></td><td>
      TWO passes per idea: <b>anarchic generation</b> (priors relaxed,
      error-detection OFF — flatten hierarchy, force cross-module
      connections, propose an "uphill move"), then <b>sober validation</b>
      (priors back online; separate insight from hallucination, propose a
      buildable v0.1). The sober output is what enters downstream refine.
      <div class="legend-cite">Carhart-Harris &amp; Friston, "REBUS and the Anarchic Brain", <i>Pharmacological Reviews</i> (2019); Seth, <i>Being You</i> (2021); Friston, "Free-energy principle", <i>Nature Reviews Neuroscience</i> (2010).</div>
    </td></tr>
    <tr><td><code>Futures</code></td><td>
      Four temporal horizons (+1y / +3y / +10y / +30y). Each names three
      concrete shifts at that horizon, identifies what is obvious from
      there but invisible today, then translates to a v0.1 shippable
      this year. The Wright-brothers vantage point.
      <div class="legend-cite">Seth, <i>Being You</i> (2021) — the brain's ~100ms perceptual forward-modeling. Foresight lineage: Wack at Shell (1970s); Brand, <i>The Clock of the Long Now</i> (1999).</div>
    </td></tr>
    <tr><td><code>Dream</code></td><td>
      Prediction-error signal offline. Generative model runs free with
      <b>NO feasibility constraint</b>. Output: vivid dream image that
      may violate physics / economics / regulation, plus an explicit
      "what survives waking" line naming the salvageable fragment.
      <div class="legend-cite">Friston, "Free-energy principle" (2010) — dreams as complexity reduction / synaptic garbage collection. Hobson &amp; Friston, "Waking and dreaming consciousness", <i>Progress in Neurobiology</i> (2012).</div>
    </td></tr>
    <tr><td><code>Lucid</code></td><td>
      Hybrid: dream-state generation + a directional prior you inject.
      The hallucination biases toward your prior; one reality check fires
      before waking. More salvage than pure Dream because the prior
      anchors the hallucination.
      <div class="legend-cite">LaBerge, <i>Exploring the World of Lucid Dreaming</i> (1990); Voss et al., "Lucid dreaming: a state of consciousness with features of both waking and non-lucid dreaming", <i>Sleep</i> (2009).</div>
    </td></tr>
  </table>
  <p><b>Flags:</b> <code>refine</code> = score every idea, pick winner, harden it. <code>evolve_deck</code> = after refine, sharpen winning cards back into the deck cache. <code>theme_entropy</code> = how distant the auto-generated donor domains should be (0 = in-field; 1 = wildly distant).</p>
</details>

<form id="form">
  <label>
    Topic — your problem / project / question
    <textarea name="topic" required
      placeholder="e.g. Reducing churn in a B2B SaaS for civil engineers"></textarea>
  </label>

  <div class="row">
    <label>Entropy
      <select name="entropy">
        <option value="sane">sane — established practice</option>
        <option value="wild" selected>wild — uncommon combos</option>
        <option value="insane">insane — cross-domain transplant</option>
        <option value="crazy">crazy — challenge an assumption</option>
        <option value="mad">mad — reinterpret the problem</option>
      </select>
    </label>
    <label>Cards in deck
      <input type="number" name="cards" value="30" min="4" max="500">
    </label>
    <label>Card depth
      <select name="card_depth">
        <option value="shallow">shallow (~15 tok)</option>
        <option value="medium" selected>medium (~60 tok)</option>
        <option value="deep">deep (~200 tok)</option>
        <option value="max">max (~500 tok)</option>
      </select>
    </label>
  </div>

  <div class="row">
    <label>Cards drawn / idea
      <input type="number" name="n_concepts" value="3" min="2" max="10">
    </label>
    <label>Ideas
      <input type="number" name="n_ideas" value="1" min="1" max="5">
    </label>
    <label>Seed (optional)
      <input type="number" name="seed" placeholder="empty = random">
    </label>
    <label>Model
      <input type="text" name="model" value="claude-opus-4-7">
    </label>
  </div>

  <label class="inline">
    <input type="checkbox" name="regen_deck">
    Force regenerate deck (otherwise use cache if present)
  </label>

  <label class="inline">
    <input type="checkbox" name="refine">
    Critic + refine winner (extra calls: 1 score per idea, 1 refinement)
  </label>

  <label class="inline">
    <input type="checkbox" name="einstein">
    Einstein mode — four mechanisms (Adjacent Possible · Exaptation · Slow Hunch · Productive Error). Overrides n_ideas to 4.
  </label>

  <label class="inline">
    <input type="checkbox" name="lsd">
    LSD mode — Prior Dissolution. Loosen the field's interpretive prior and re-perceive (Friston / Seth / REBUS). Uses n_ideas.
  </label>

  <label class="inline">
    <input type="checkbox" name="futures">
    Futures mode — temporal projection (+1y · +3y · +10y · +30y). Identify what is obvious from each future, ship the v0.1 today. Overrides n_ideas to 4.
  </label>

  <label class="inline">
    <input type="checkbox" name="dream">
    Dream mode — prediction-error offline. Generative model runs free, no feasibility constraint. Output is a dream image + "what survives waking". Uses n_ideas.
  </label>

  <label>
    Lucid prior (optional — enables Lucid Dreaming mode)
    <textarea name="lucid" rows="2"
      placeholder="A directional belief the dream should bias toward, e.g. 'the team must stay solo-founder-sized'. Leave empty to skip."></textarea>
  </label>

  <label class="inline">
    <input type="checkbox" name="evolve_deck">
    Evolve deck — when refine produces a winner, sharpen the cards that contributed and write them back to the deck cache. Requires refine. Not compatible with bring-your-own-deck.
  </label>

  <details class="advanced">
    <summary>Theme generator (advanced)</summary>
    <label>
      Theme entropy override (0..1) — how distant from your field the
      auto-generated donor domains should be. Empty = use the run's main
      entropy. 0 = stay in-field; 1 = wildly distant.
      <input type="text" name="theme_entropy" placeholder="(empty = follow entropy)">
    </label>
    <label>
      Explicit themes (comma-separated). Overrides the auto-generated theme
      list. Example: <code>harbor pilotage, mycology, monastic rules</code>.
      <textarea name="themes" rows="2"
        placeholder="domain, domain, domain"></textarea>
    </label>
  </details>

  <details class="advanced">
    <summary>Bring your own deck (optional)</summary>
    <label>
      Inline donor bank as JSON — <code>{"domain": ["concept", ...], ...}</code>.
      Replaces topic-aware deck generation. Shallow depth (no mechanism field).
      <textarea name="bank_data" rows="4"
        placeholder='{"my-field": ["concept1", "concept2"], "adjacent": ["..."]}'></textarea>
    </label>
  </details>

  <button id="submit" type="submit">Generate</button>
</form>

<div id="output"></div>

<div id="usage-strip" class="panel usage" hidden>
  <div class="meta">LLM usage <span id="usage-refresh" class="usage-refresh" title="Reload usage from log">↻</span></div>
  <div id="usage-body"></div>
</div>

<script>
const form = document.getElementById('form');
const out = document.getElementById('output');
const btn = document.getElementById('submit');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  out.innerHTML = '';
  for (const k of Object.keys(statusByPhase)) delete statusByPhase[k];
  for (const k of Object.keys(startByPhase)) delete startByPhase[k];

  const fd = new FormData(form);
  const modesOn = ['einstein', 'lsd', 'futures', 'dream'].filter(k => fd.get(k)).length
                + (fd.get('lucid') ? 1 : 0);
  if (modesOn > 1) {
    addPanel('<div class="error">Einstein / LSD / Futures / Dream / Lucid modes are ' +
             'mutually exclusive. Pick at most one.</div>');
    return;
  }
  if (fd.get('evolve_deck') && fd.get('bank_data')) {
    addPanel('<div class="error">evolve_deck requires a generated deck; ' +
             'not compatible with bring-your-own bank_data.</div>');
    return;
  }
  if (fd.get('evolve_deck') && !fd.get('refine')) {
    addPanel('<div class="error">evolve_deck has no effect without refine; ' +
             'enable Critic + refine winner first.</div>');
    return;
  }
  const payload = {};
  for (const [k, v] of fd.entries()) {
    if (v === '') continue;
    if (['cards','n_concepts','n_ideas','seed'].includes(k)) payload[k] = Number(v);
    else if (k === 'theme_entropy') {
      const t = Number(v);
      if (!isNaN(t)) payload.theme_entropy = t;
    }
    else if (['regen_deck','refine','einstein','lsd','futures','dream','evolve_deck'].includes(k)) payload[k] = true;
    else if (k === 'themes') {
      // comma- or newline-separated; empty => omit
      const parts = String(v).split(/[,\n]+/).map(s => s.trim()).filter(Boolean);
      if (parts.length) payload.themes = parts;
    }
    else if (k === 'bank_data') {
      try {
        payload.bank_data = JSON.parse(v);
      } catch (err) {
        addPanel('<div class="error">bank_data is not valid JSON: ' +
                 escapeHtml(err.message) + '</div>');
        return;
      }
    } else payload[k] = v;
  }
  if (!('regen_deck' in payload)) payload.regen_deck = false;
  if (!('refine' in payload)) payload.refine = false;
  btn.disabled = true;
  btn.textContent = 'Generating…';
  try {
    const resp = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error('HTTP ' + resp.status + ': ' + text);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        handleEvent(chunk);
      }
    }
  } catch (err) {
    addPanel('<div class="error">Error: ' + escapeHtml(err.message) + '</div>');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate';
  }
});

// Tracks the live status panel per phase so status/progress events can
// update the same DOM node in place rather than appending new ones.
const statusByPhase = {};
const startByPhase = {};

function handleEvent(chunk) {
  let event = 'message', data = '';
  for (const line of chunk.split('\n')) {
    if (line.startsWith('event: ')) event = line.slice(7);
    else if (line.startsWith('data: ')) data += line.slice(6);
  }
  if (!data) return;
  let obj;
  try { obj = JSON.parse(data); } catch (e) { return; }

  if (event === 'status') {
    const phase = obj.phase || 'global';
    startByPhase[phase] = performance.now();
    upsertStatus(phase, obj.message);
  } else if (event === 'progress') {
    const phase = obj.phase || 'global';
    updateElapsed(phase, obj.elapsed);
  } else if (event === 'deck') {
    finishStatus('deck', 'deck ready: ' + obj.size + ' cards (depth=' +
                 escapeHtml(obj.depth) + ')');
    const cardHtml = obj.cards.map(renderCard).join('');
    addPanel(
      '<details><summary>Donor deck — ' + obj.size +
      ' cards, depth=' + escapeHtml(obj.depth) +
      '</summary>' + cardHtml + '</details>'
    );
  } else if (event === 'sample') {
    const cardHtml = obj.cards.map(renderCard).join('');
    const mechTag = obj.mechanism
      ? '<span class="mech-tag' + (
          obj.mechanism && obj.mechanism.indexOf('LSD') === 0 ? ' lsd' :
          obj.mechanism === 'Dreaming' ? ' dream' :
          obj.mechanism === 'Lucid Dreaming' ? ' lucid' :
          obj.mechanism && obj.mechanism.indexOf('Futures') === 0 ? ' futures' :
          ''
        ) + '">' + escapeHtml(obj.mechanism) + '</span>'
      : '';
    const blurb = obj.mechanism_blurb
      ? '<div class="mech-blurb">' + escapeHtml(obj.mechanism_blurb) + '</div>'
      : '';
    addPanel(
      '<div class="meta">Drawn for idea ' + (obj.i + 1) +
      ' — entropy=' + escapeHtml(obj.level) +
      ' (spread=' + obj.spread.toFixed(2) + ')' +
      mechTag + '</div>' +
      blurb +
      cardHtml
    );
  } else if (event === 'idea') {
    finishStatus('synth-' + obj.i, 'idea ' + (obj.i + 1) + ' ready');
    const mechTag = obj.mechanism
      ? '<span class="mech-tag' + (
          obj.mechanism && obj.mechanism.indexOf('LSD') === 0 ? ' lsd' :
          obj.mechanism === 'Dreaming' ? ' dream' :
          obj.mechanism === 'Lucid Dreaming' ? ' lucid' :
          obj.mechanism && obj.mechanism.indexOf('Futures') === 0 ? ' futures' :
          ''
        ) + '">' + escapeHtml(obj.mechanism) + '</span>'
      : '';
    addPanel(
      '<div class="meta">Idea ' + (obj.i + 1) + mechTag + '</div>' +
      '<div class="idea"><pre>' + escapeHtml(obj.text) + '</pre></div>'
    );
  } else if (event === 'lsd_validation') {
    finishStatus('lsd-validate-' + obj.i, 'sober validation ' + (obj.i + 1) + ' done');
    addPanel(
      '<div class="meta lsd-sober-meta">LSD · sober validation ' + (obj.i + 1) + '</div>' +
      '<div class="idea lsd-sober"><pre>' + escapeHtml(obj.sober) + '</pre></div>'
    );
  } else if (event === 'score') {
    finishStatus('critic-' + obj.i,
                 'idea ' + (obj.i + 1) + ' scored ' + obj.total + '/300');
    const mechTag = obj.mechanism
      ? '<span class="mech-tag' + (
          obj.mechanism && obj.mechanism.indexOf('LSD') === 0 ? ' lsd' :
          obj.mechanism === 'Dreaming' ? ' dream' :
          obj.mechanism === 'Lucid Dreaming' ? ' lucid' :
          obj.mechanism && obj.mechanism.indexOf('Futures') === 0 ? ' futures' :
          ''
        ) + '">' + escapeHtml(obj.mechanism) + '</span>'
      : '';
    addPanel(
      '<div class="meta">Score · idea ' + (obj.i + 1) + mechTag + '</div>' +
      '<div class="score-row">' +
        '<span class="score-pill">feasibility ' + obj.feasibility + '</span>' +
        '<span class="score-pill">unexpectedness ' + obj.unexpectedness + '</span>' +
        '<span class="score-pill">topic-fit ' + obj.topic_fit + '</span>' +
        '<span class="score-pill total">total ' + obj.total + '/300</span>' +
      '</div>' +
      (obj.notes ? '<div class="critic-notes">' + escapeHtml(obj.notes) + '</div>' : '')
    );
  } else if (event === 'winner') {
    const mechSuffix = obj.mechanism
      ? ' · ' + escapeHtml(obj.mechanism) + ' mechanism wins'
      : '';
    addPanel(
      '<div class="meta">Winner</div>' +
      '<div class="winner-banner">Idea ' + (obj.i + 1) +
      ' wins with ' + obj.total + '/300' + mechSuffix + '</div>' +
      (obj.notes ? '<div class="critic-notes">' + escapeHtml(obj.notes) + '</div>' : '')
    );
  } else if (event === 'refined') {
    finishStatus('refine', 'refined idea ' + (obj.i + 1));
    addPanel(
      '<div class="meta refined-meta">Refined · idea ' + (obj.i + 1) + '</div>' +
      '<div class="idea refined"><pre>' + escapeHtml(obj.text) + '</pre></div>'
    );
  } else if (event === 'evolved') {
    finishStatus('evolve', 'deck evolved: ' + obj.pairs.length + ' card(s) updated');
    const rows = obj.pairs.map(function (pair) {
      const before = pair.before, after = pair.after;
      const fields = Object.keys(after).filter(function (k) {
        return k !== 'name' && k !== 'domain' && before[k] !== after[k];
      });
      const fieldRows = fields.map(function (f) {
        return (
          '<div class="evolve-field"><b>' + escapeHtml(f.replace(/_/g, ' ')) + '</b>' +
          '<div class="evolve-before">- was: ' + escapeHtml(String(before[f] || '')) + '</div>' +
          '<div class="evolve-after">+ now: ' + escapeHtml(String(after[f] || '')) + '</div>' +
          '</div>'
        );
      }).join('');
      return (
        '<div class="evolve-card"><div class="evolve-name">◆ ' +
        escapeHtml(before.name) +
        ' <span class="evolve-domain">(' + escapeHtml(before.domain) + ')</span></div>' +
        fieldRows + '</div>'
      );
    }).join('');
    addPanel(
      '<div class="meta refined-meta">Deck evolved (winner’s cards sharpened ' +
      'and written back to cache)</div>' + rows
    );
  } else if (event === 'run') {
    // captured for completeness; nothing to render
  } else if (event === 'usage') {
    renderUsage(obj);
  } else if (event === 'done') {
    addPanel('<div class="status done"><span class="dot"></span>' +
             '<span class="msg">Done.</span></div>');
  } else if (event === 'error') {
    addPanel('<div class="error">' + escapeHtml(obj.message) + '</div>');
  }
}

function upsertStatus(phase, message) {
  let panel = statusByPhase[phase];
  if (!panel) {
    panel = document.createElement('div');
    panel.className = 'panel';
    panel.innerHTML =
      '<div class="status">' +
      '<span class="dot"></span>' +
      '<span class="msg"></span>' +
      '<span class="elapsed"></span>' +
      '</div>';
    out.appendChild(panel);
    statusByPhase[phase] = panel;
  }
  panel.querySelector('.status').classList.remove('done', 'error');
  panel.querySelector('.msg').textContent = message;
  panel.querySelector('.elapsed').textContent = '';
  panel.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function updateElapsed(phase, elapsed) {
  const panel = statusByPhase[phase];
  if (!panel) return;
  panel.querySelector('.elapsed').textContent =
    'elapsed ' + Number(elapsed).toFixed(1) + 's';
}

function finishStatus(phase, finalMsg) {
  const panel = statusByPhase[phase];
  if (!panel) return;
  const sd = panel.querySelector('.status');
  sd.classList.add('done');
  panel.querySelector('.msg').textContent = finalMsg;
  if (startByPhase[phase]) {
    const ms = performance.now() - startByPhase[phase];
    panel.querySelector('.elapsed').textContent =
      'took ' + (ms / 1000).toFixed(1) + 's';
  }
}

function renderCard(c) {
  const skip = new Set(['name', 'domain']);
  const body = Object.entries(c)
    .filter(([k]) => !skip.has(k))
    .map(([k, v]) => '<div class="body"><i>' + escapeHtml(k.replace(/_/g, ' ')) +
                     ':</i> ' + escapeHtml(String(v)) + '</div>')
    .join('');
  return '<div class="card"><span class="name">' + escapeHtml(c.name) +
         '</span> <span class="domain">(' + escapeHtml(c.domain) + ')</span>' +
         body + '</div>';
}

function addPanel(html) {
  const div = document.createElement('div');
  div.className = 'panel';
  div.innerHTML = html;
  out.appendChild(div);
  div.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function fmtTokens(n) {
  if (typeof n !== 'number' || isNaN(n)) return '0';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'k';
  return String(n);
}
function fmtMs(n) {
  if (!n) return '0s';
  if (n < 1000) return n + 'ms';
  if (n < 60_000) return (n / 1000).toFixed(1) + 's';
  return Math.floor(n / 60_000) + 'm ' + Math.round((n % 60_000) / 1000) + 's';
}
function fmtUsd(n) {
  if (!n) return '$0.00';
  if (n < 0.01) return '$' + n.toFixed(4);
  return '$' + n.toFixed(2);
}
function usageCard(label, val, sub) {
  return '<div class="usage-card"><div class="label">' + escapeHtml(label) +
    '</div><div class="val">' + escapeHtml(val) + '</div>' +
    (sub ? '<div class="sub">' + escapeHtml(sub) + '</div>' : '') +
    '</div>';
}
function usageBucket(label, b) {
  if (!b) return '';
  const sub = (b.calls || 0) + ' call' + (b.calls === 1 ? '' : 's') +
              ' · ' + fmtMs(b.duration_ms || 0) +
              ' · ' + fmtUsd(b.total_cost_usd || 0);
  return usageCard(label,
                   fmtTokens(b.input_tokens || 0) + ' in / ' +
                   fmtTokens(b.output_tokens || 0) + ' out',
                   sub);
}
function renderUsage(u) {
  const strip = document.getElementById('usage-strip');
  const body = document.getElementById('usage-body');
  if (!strip || !body) return;
  strip.hidden = false;

  let rateHtml = '';
  if (u.rate_limit) {
    const rl = u.rate_limit;
    const status = rl.status || 'unknown';
    const sev = status === 'allowed' ? 'ok' :
                status === 'warning' ? 'warn' : 'bad';
    let resets = '';
    if (rl.resets_at) {
      const dt = rl.resets_at - Date.now() / 1000;
      if (dt > 0) {
        const h = Math.floor(dt / 3600), m = Math.floor((dt % 3600) / 60);
        resets = 'resets in ' + (h ? h + 'h ' : '') + m + 'm';
      } else {
        resets = 'reset due';
      }
    }
    const util = (rl.utilization != null)
      ? Math.round(rl.utilization * 100) + '%'
      : '—';
    rateHtml = usageCard(
      'Subscription window (' + (rl.type || '?') + ')',
      'status: ' + status + ', utilization ' + util,
      resets
    ).replace('class="label"',
              'class="label usage-rate-' + sev + '"');
  }

  body.innerHTML =
    '<div class="usage-grid">' +
    usageBucket('This run', u.this_run) +
    usageBucket('Last 7 days', u.last_7d) +
    usageBucket('Last 30 days', u.last_30d) +
    usageBucket('All time', u.total) +
    usageCard('5h windows touched (last 7d)',
              String(u.five_h_windows_last_7d || 0),
              'local heuristic; not subscription truth') +
    (rateHtml || '') +
    '</div>' +
    (u.note ? '<div class="usage-note">' + escapeHtml(u.note) + '</div>' : '');
}

async function loadUsage() {
  try {
    const r = await fetch('/api/usage');
    if (r.ok) renderUsage(await r.json());
  } catch (_) { /* ignore */ }
}
document.getElementById('usage-refresh').addEventListener('click', loadUsage);
loadUsage();
</script>
</body>
</html>"""
