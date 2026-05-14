# AIdea

Generate unexpected-but-feasible ideas attached to your own topic, with
adjustable entropy.

The pipeline models what the brain does when it has an idea: take past
experience (raw material), shuffle it with controlled noise (entropy),
and reorganize the result through inference. AIdea exposes those three
moves as separate knobs:

```
USER TOPIC ─────────┐
                    │
[ stage 0 ] ───→  donor deck   (topic-aware concepts, generated once, cached)
                    │
[ stage 1 ] ───→  shuffle      (entropy controls cross-domain spread)
                    │
[ stage 2 ] ───→  synthesis    (audacity controls departure from convention)
                    │
[ stage 3 ] ───→  critic + refine   (optional: score every idea, harden the winner)
                    │
                    ▼
            applied, feasible idea
```

## Why this exists

LLMs are oracles by default: ask once, get one answer that sits in the
median of training data. Asking the same question at higher temperature
produces noisier *prose*, not deeper *novelty* — the model still draws
from the same conceptual space. AIdea injects entropy outside the model,
at the raw-material layer, then uses inference for synthesis only. This
keeps the two forces orthogonal — exactly the way conceptual blending
seems to work in humans.

The output is **always tied to your stated topic** and feasibility is
enforced at every audacity level (including the wildest); the framework
refuses to produce science fiction.

## Install

Requires Python 3.10+ and an authenticated agent CLI on PATH.

```bash
git clone https://github.com/frstrtr/AIdea.git
cd AIdea
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The library shells out through the agent CLI so it inherits its
authentication — no separate API key needed.

## CLI

```bash
.venv/bin/python aidea.py --topic "your problem or project in a sentence"
```

Useful flags:

| Flag                       | What it controls                                                |
| -------------------------- | --------------------------------------------------------------- |
| `--topic` / `-t`           | Your problem, project, or question.                             |
| `--entropy` / `-e`         | Named (`sane / wild / insane / crazy / mad`) or float `0..1`.   |
| `--cards N`                | Deck size — total donor cards generated and cached. Default 30. |
| `--card-depth D`           | Per-card detail: `shallow / medium / deep / max`. Default medium. |
| `--n-concepts K`           | Cards drawn from the deck per idea. Default 3.                  |
| `--n-ideas N`              | Independent ideas to generate in one run. Default 1.            |
| `--refine`                 | After generating, score every idea and refine the winner.       |
| `--seed N`                 | Lock the sampler for reproducible runs.                         |
| `--bank file.json`         | Static donor bank `{domain: [concept, ...]}` instead of LLM-gen. |
| `--regen-deck`             | Force regenerate the cached deck.                               |
| `--show-deck`              | Print the resolved deck before generating ideas.                |
| `--list-levels`            | Print named entropy levels and exit.                            |
| `--list-depths`            | Print named card-depth presets and exit.                        |

### Example

```bash
.venv/bin/python aidea.py \
  -t "Reducing churn in a B2B SaaS for civil engineers" \
  --entropy insane --n-ideas 3 --refine
```

Generates three ideas at high cross-domain spread, scores each on
feasibility / unexpectedness / topic-fit (0–100 per axis, 300 total),
picks the winning total, and re-prompts the winner at low entropy to
harden the first step and address the named weakness.

## Web UI

```bash
.venv/bin/uvicorn server:app --port 8000 --reload --reload-exclude 'decks/*'
```

Open http://127.0.0.1:8000.

The page exposes the same knobs as the CLI plus:

- A form for the topic, entropy, deck size, card depth, draws per idea,
  number of ideas, seed, model, refine toggle.
- An optional "bring your own deck" textarea that accepts inline JSON of
  the form `{"domain_name": ["concept", "concept", ...], ...}`. When set,
  deck generation is skipped and your concepts are the raw material.
- Live progress: every stage shows a pulsing dot and elapsed-time counter
  that ticks every ~1.5 seconds, so you can see the long inference calls
  are still alive.
- Visual differentiation: each idea is rendered as a panel; if `refine`
  is on, per-idea score pills, a yellow winner banner, and a green-bordered
  "REFINED" panel appear below.

## Entropy levels

| Level   | Spread | What it asks of the synthesizer                                |
| ------- | ------ | -------------------------------------------------------------- |
| sane    | 0.10   | Stay within established practice. "We should just do that."    |
| wild    | 0.40   | Combine familiar approaches in uncommon ways.                  |
| insane  | 0.65   | Transplant a mechanism from an unrelated domain.               |
| crazy   | 0.85   | Challenge a load-bearing assumption the field treats as fixed. |
| mad     | 0.98   | Reinterpret the problem itself; ship v0.1 within six months.   |

Feasibility is a hard constraint at every level, including `mad`. Even
the wildest output must end with a concrete first step you could take
this week and a sketch of how a small team would ship a working
prototype.

## Card depth

Each donor card can carry more or less pre-seeded detail:

| Depth   | Fields per card                                              | Token budget |
| ------- | ------------------------------------------------------------ | ------------ |
| shallow | name, domain                                                 | ~15          |
| medium  | + mechanism (one sentence)                                   | ~60          |
| deep    | + why-it-works + transfer-hint                               | ~200         |
| max     | + invariants + prior cross-domain application                | ~500         |

Deeper cards give the synthesizer richer raw material at the cost of
prompt size. Cards are generated once per `(topic, cards, depth, model)`
combination and cached in `decks/` as JSON; subsequent runs hit the
cache.

## Deployment

To run on a fresh Linux VM (Proxmox VE or any cloud-init-capable host),
see [`deploy/README.md`](deploy/README.md). Two systemd units (web + bot),
one `.env`, one shared `usage.jsonl`. Path A uses cloud-init for an
unattended install; Path B is `bash install.sh` on any Linux VM you
already have.

## Telegram bot

```bash
export TELEGRAM_BOT_TOKEN=<token from @BotFather>
.venv/bin/python bot.py
```

Same pipeline, exposed over Telegram chat. Commands:

| Command | What it does |
|---|---|
| `/idea <topic>` | One idea at the current entropy |
| `/einstein <topic>` | Four mechanism-specific ideas (Adjacent Possible · Exaptation · Slow Hunch · Productive Error) |
| `/lsd <topic>` | Prior dissolution — re-perceive under a different frame |
| `/futures <topic>` | Temporal projection (+1y · +3y · +10y · +30y) |
| `/settings` | Show entropy / cards / card-depth / refine / evolve knobs |
| `/set k=v` | Tune one knob (entropy / cards / card_depth / n_concepts / n_ideas / refine / evolve_deck / seed / model) |
| `/usage` | LLM usage summary + observed subscription-window state |
| `/cancel` | Abort the in-flight task in this chat |

Progress is delivered by editing a single "working…" message every ~1.5s,
so long synthesis runs don't spam the chat. Ideas longer than Telegram's
4096-char cap are split on paragraph boundaries.

The bot inherits its model auth from whatever the agent CLI is logged in
against — same auth model as the web app.

## Usage tracking

Every LLM call writes one record to `usage.jsonl`: tokens (input / output /
cache hits), duration, USD-equivalent cost, primary model, and the most
recently observed rate-limit window from the agent SDK's `RateLimitEvent`.

Both surfaces expose it:

- **Web**: a sticky usage panel at the bottom of the page shows this-run,
  7-day, 30-day, and all-time totals. The "Subscription window" card
  reflects the actual rate-limit type and reset time from the SDK; the
  "5h windows touched (last 7d)" card is a local heuristic and labelled
  as such. `GET /api/usage` returns the same payload as JSON.
- **Telegram**: `/usage` prints the equivalent summary in chat.

## Project layout

```
aidea.py          # Library + CLI: deck generation, sampling, synthesis,
                  # critic, refinement, all assembled in run_pipeline().
server.py         # FastAPI app + inline single-page UI; SSE stream of
                  # status / progress / deck / sample / idea / score /
                  # winner / refined / evolved / usage / done / error.
bot.py            # Telegram bot binding the same pipeline.
usage.py          # JSONL usage log + summarizer (this-run / 7d / 30d
                  # + real rate-limit window from the SDK).
requirements.txt
decks/            # Per-topic donor-deck cache (gitignored).
usage.jsonl       # Per-call usage log (gitignored).
```

## License

No license has been chosen yet. Treat the code as all-rights-reserved
until a license is added.
