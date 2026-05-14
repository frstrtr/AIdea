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
import random
import time
from typing import Any, AsyncIterator, Awaitable

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from aidea import (
    CARD_DEPTH_BY_NAME,
    CARD_DEPTHS,
    ENTROPY_LEVELS,
    Card,
    build_prompt,
    cards_from_static_bank,
    critic_score,
    load_bank,
    load_or_generate_deck,
    parse_entropy,
    refine_idea,
    sample_cards,
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
        try:
            async for kind, payload in _watched(
                load_or_generate_deck(
                    topic=req.topic,
                    n=req.cards,
                    depth=depth,
                    model=req.model,
                    force_regen=req.regen_deck,
                    verbose=False,
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

    # --- Stage 1 & 2: sample + synthesize, per idea -----------------------
    rng = random.Random(req.seed)
    ideas: list[str] = []
    for i in range(req.n_ideas):
        cards = sample_cards(deck=deck, n=req.n_concepts, spread=spread, rng=rng)
        yield _sse("sample", {
            "i": i,
            "level": level.name,
            "spread": spread,
            "cards": [_card_to_dict(c) for c in cards],
        })

        synth_phase = f"synth-{i}"
        yield _sse("status", {
            "phase": synth_phase,
            "i": i,
            "message": f"Synthesizing idea {i + 1}/{req.n_ideas}...",
        })

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

        yield _sse("idea", {"i": i, "text": idea})
        ideas.append(idea)

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
            scored.append(score)
            yield _sse("score", score)

        winner = max(scored, key=lambda s: s["total"])
        yield _sse("winner", {
            "i": winner["i"],
            "total": winner["total"],
            "notes": winner.get("notes", ""),
        })

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
  depart from current practice). Feasibility is enforced at every level.
</p>

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
  const payload = {};
  for (const [k, v] of fd.entries()) {
    if (v === '') continue;
    if (['cards','n_concepts','n_ideas','seed'].includes(k)) payload[k] = Number(v);
    else if (k === 'regen_deck' || k === 'refine') payload[k] = true;
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
    addPanel(
      '<div class="meta">Drawn for idea ' + (obj.i + 1) +
      ' — entropy=' + escapeHtml(obj.level) +
      ' (spread=' + obj.spread.toFixed(2) + ')</div>' +
      cardHtml
    );
  } else if (event === 'idea') {
    finishStatus('synth-' + obj.i, 'idea ' + (obj.i + 1) + ' ready');
    addPanel(
      '<div class="meta">Idea ' + (obj.i + 1) + '</div>' +
      '<div class="idea"><pre>' + escapeHtml(obj.text) + '</pre></div>'
    );
  } else if (event === 'score') {
    finishStatus('critic-' + obj.i,
                 'idea ' + (obj.i + 1) + ' scored ' + obj.total + '/300');
    addPanel(
      '<div class="meta">Score · idea ' + (obj.i + 1) + '</div>' +
      '<div class="score-row">' +
        '<span class="score-pill">feasibility ' + obj.feasibility + '</span>' +
        '<span class="score-pill">unexpectedness ' + obj.unexpectedness + '</span>' +
        '<span class="score-pill">topic-fit ' + obj.topic_fit + '</span>' +
        '<span class="score-pill total">total ' + obj.total + '/300</span>' +
      '</div>' +
      (obj.notes ? '<div class="critic-notes">' + escapeHtml(obj.notes) + '</div>' : '')
    );
  } else if (event === 'winner') {
    addPanel(
      '<div class="meta">Winner</div>' +
      '<div class="winner-banner">Idea ' + (obj.i + 1) +
      ' wins with ' + obj.total + '/300</div>' +
      (obj.notes ? '<div class="critic-notes">' + escapeHtml(obj.notes) + '</div>' : '')
    );
  } else if (event === 'refined') {
    finishStatus('refine', 'refined idea ' + (obj.i + 1));
    addPanel(
      '<div class="meta refined-meta">Refined · idea ' + (obj.i + 1) + '</div>' +
      '<div class="idea refined"><pre>' + escapeHtml(obj.text) + '</pre></div>'
    );
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
</script>
</body>
</html>"""
