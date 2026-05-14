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
from typing import AsyncIterator

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
    load_bank,
    load_or_generate_deck,
    parse_entropy,
    sample_cards,
    synthesize,
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


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


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
    yield _sse("status", {
        "phase": "deck",
        "message": (
            f"Using static bank: {req.bank}" if req.bank
            else f"Generating donor deck (n={req.cards}, depth={depth.name})..."
        ),
    })
    try:
        if req.bank:
            deck = cards_from_static_bank(load_bank(req.bank))
        else:
            deck = await load_or_generate_deck(
                topic=req.topic,
                n=req.cards,
                depth=depth,
                model=req.model,
                force_regen=req.regen_deck,
                verbose=False,
            )
    except Exception as e:
        yield _sse("error", {"message": f"deck stage failed: {e}"})
        return

    yield _sse("deck", {
        "size": len(deck),
        "depth": depth.name,
        "cards": [_card_to_dict(c) for c in deck],
    })

    # --- Stage 1 & 2: sample + synthesize, per idea -----------------------
    rng = random.Random(req.seed)
    for i in range(req.n_ideas):
        cards = sample_cards(deck=deck, n=req.n_concepts, spread=spread, rng=rng)
        yield _sse("sample", {
            "i": i,
            "level": level.name,
            "spread": spread,
            "cards": [_card_to_dict(c) for c in cards],
        })

        yield _sse("status", {
            "phase": "synth",
            "i": i,
            "message": f"Synthesizing idea {i + 1}/{req.n_ideas}...",
        })

        prompt = build_prompt(req.topic, cards, level)
        try:
            idea = await synthesize(prompt=prompt, model=req.model, stream_to_stdout=False)
        except Exception as e:
            yield _sse("error", {"message": f"synthesis failed: {e}"})
            return

        yield _sse("idea", {"i": i, "text": idea})

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
  .status { color: #555; font-style: italic; font-size: 0.92rem; }
  .status::before { content: "● "; color: #999; }
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
  .error { color: #b00020; border-left-color: #b00020; }
  @media (prefers-color-scheme: dark) {
    body { color: #e5e5e5; }
    input, select, textarea { background: #1e1e1e; border-color: #444; color: #e5e5e5; }
    button { background: #e5e5e5; color: #1a1a1a; }
    .panel { background: #1a1a1a; border-color: #333; }
    .idea { background: #111; border-left-color: #e5e5e5; }
    .sub, .status, summary, .card .domain { color: #aaa; }
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

  <button id="submit" type="submit">Generate</button>
</form>

<div id="output"></div>

<script>
const form = document.getElementById('form');
const out = document.getElementById('output');
const btn = document.getElementById('submit');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  const payload = {};
  for (const [k, v] of fd.entries()) {
    if (v === '') continue;
    if (['cards','n_concepts','n_ideas','seed'].includes(k)) payload[k] = Number(v);
    else if (k === 'regen_deck') payload[k] = true;
    else payload[k] = v;
  }
  if (!('regen_deck' in payload)) payload.regen_deck = false;

  out.innerHTML = '';
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
    addPanel('<div class="status">' + escapeHtml(obj.message) + '</div>');
  } else if (event === 'deck') {
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
    addPanel(
      '<div class="meta">Idea ' + (obj.i + 1) + '</div>' +
      '<div class="idea"><pre>' + escapeHtml(obj.text) + '</pre></div>'
    );
  } else if (event === 'done') {
    addPanel('<div class="status">Done.</div>');
  } else if (event === 'error') {
    addPanel('<div class="error">' + escapeHtml(obj.message) + '</div>');
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
