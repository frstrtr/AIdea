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

## Modes

Pick at most one per request. Each is a structurally different way to
generate ideas — not a different temperature setting on the same default.

| Mode | What it does | Feasibility | n_ideas |
|---|---|---|---|
| `default` | Standard pass at requested entropy | required | configurable |
| `einstein` | Four mechanism-specific passes (Adjacent Possible · Exaptation · Slow Hunch · Productive Error) | required | forced to 4 |
| `lsd` | Prior Dissolution (Friston / Seth / REBUS). Suspends the field's interpretive prior and re-perceives | required | configurable |
| `futures` | Four temporal horizons (+1y / +3y / +10y / +30y). Names what's obvious from each future and ships a v0.1 today | required | forced to 4 |
| `dream` | Prediction-error offline. Generative model runs free, no feasibility constraint. Output is a dream image + "what survives waking" | **off** | configurable |
| `lucid` | Dream + a directional prior you inject. Hallucination biases toward your prior, with one reality check at the end | **off** | configurable |

CLI flags: `--einstein`, `--lsd`, `--futures`, `--dream`, `--lucid "<prior sentence>"`.
The five are mutually exclusive. Web UI exposes them as checkboxes + a
prior textarea. Telegram exposes them as `/einstein`, `/lsd`, `/futures`,
`/dream`, `/lucid <prior> :: <topic>`.

## Scientific basis per mode

Each mode is a cognitive operation borrowed from a specific framework in
neuroscience / cognitive science / philosophy of invention. The prompt
templates implement these mechanisms verbatim — the model is not asked
to "be creative"; it is asked to execute a named process from the
literature.

### Overarching frame

The pipeline implements **stochastic synthesis** as described by
Stuart Kauffman and popularized by Steven Johnson:

- **Adjacent Possible** (Kauffman, *Investigations*, 2000) — at any moment
  the set of "next-door rooms" is bounded by prior work. Discovery walks
  through doors that have just been unlocked.
- **Conceptual blending** (Fauconnier & Turner, *The Way We Think*, 2002) —
  novel ideas come from forced cross-domain combination, not from raw
  invention. "Originality" is a high-entropy reorganization of low-entropy
  data.

Entropy lives at the *raw material* layer (which donor concepts collide).
Inference lives at the *synthesis* layer (the model finds the structural
overlap). The two are kept orthogonal, which mirrors how the brain seems
to separate *combinatorial play* (the Default Mode Network during rest)
from *evaluation* (executive function when awake).

### `default` — applied conceptual blending

| | |
|---|---|
| **What it does** | One idea per call at the requested entropy. Standard format: title, pitch, mechanism, first step, risks. |
| **Cognitive model** | Conceptual blending under feasibility constraint. Picks 1+ donor concepts whose structural mechanism transfers onto the user's topic. |
| **Sources** | Fauconnier & Turner (*The Way We Think*, 2002); Kauffman (*Investigations*, 2000). |
| **When to use** | First-look on any topic; baseline for comparing the other modes. |

### `einstein` — four generative mechanisms

Runs four passes in series, each implementing one of the four routes-to-
ideas catalogued by Steven Johnson in *Where Good Ideas Come From* (2010):

1. **Adjacent Possible** (Kauffman 2000 / Johnson 2010) — name a
   capability that became available in the last 1-3 years, walk through
   that newly-unlocked door.
2. **Exaptation** (Gould & Vrba, "Exaptation — a missing term in the
   science of form", *Paleobiology* 1982) — biological term for a trait
   that evolved for one purpose then got co-opted for another. Applied to
   ideation: transplant a mechanism from a far-distant field. (Gutenberg
   moving the wine-press mechanism to inked type is the canonical
   example.)
3. **Slow Hunch** (Johnson 2010) — articulate a latent tension or quiet
   contradiction the field has lived with for years without naming
   cleanly, then propose its resolution.
4. **Productive Error** (Fleming 1928 / mutation as evolutionary engine)
   — invert one load-bearing assumption the field treats as fixed.
   Misreadings that survive contact with reality become discoveries.

Combine with `refine` to score across mechanisms and harden the winning
route — gives empirical signal on *which route works for your topic*.

### `lsd` — REBUS, two-pass (anarchic generation + sober validation)

| | |
|---|---|
| **What it does** | Two LLM calls per pass: an anarchic generation with priors relaxed and error-detection offline, followed by a sober validation pass with priors restored. The sober output is what enters downstream refine. |
| **Cognitive model** | **REBUS** — RElaxed Beliefs Under pSychedelics. Predictive-processing neuroscience treats perception as a controlled hallucination held in check by high-precision priors; psychedelics (especially LSD, via 5-HT2A receptor agonism in deep cortical layers) flatten the hierarchy, weaken those priors, and increase brain entropy. |
| **Mechanism** | Anarchic prompt forces (a) naming the local minimum the field is stuck in, (b) listing 2-3 load-bearing priors to treat as noise, (c) a cross-module connection between maximally distant donor concepts, (d) an "uphill move" that gets worse on the current metric short-term. Validation prompt brings error detection back online and asks the model to separate the structural insight from the hallucination. |
| **Sources** | Carhart-Harris & Friston, "REBUS and the Anarchic Brain: Toward a Unified Model of the Brain Action of Psychedelics", *Pharmacological Reviews* 2019. Anil Seth, *Being You: A New Science of Consciousness*, 2021. Karl Friston, "The free-energy principle: a unified brain theory?", *Nature Reviews Neuroscience* 2010. |
| **When to use** | When you suspect you're stuck inside a "good-enough" local minimum and need to escape it — at the cost of 2× the LLM calls per pass. The pro-tip from the literature: "Best psychedelic breakthroughs are when the anarchic brain generates the connection and the sober inference engine validates it 24 hours later." This is implemented as the two-pass structure. |

### `futures` — temporal projection at four horizons

| | |
|---|---|
| **What it does** | Four passes at +1y / +3y / +10y / +30y. Each names three concrete shifts at that horizon, identifies what's obvious from there but invisible today, then translates to a v0.1 you can ship this year. |
| **Cognitive model** | The brain runs ~100ms behind the world and compensates by hallucinating the immediate future (perceptual forward-modeling). This mode runs the same forward-simulation operation at much longer horizons. Wright-brothers vantage point: see the airplane from 1910, build it in 1903. |
| **Sources** | Anil Seth, *Being You*, 2021 (the 100ms-delayed-past framing). Foresight methodology: Pierre Wack at Royal Dutch Shell, 1970s (scenario planning); Stewart Brand, *The Clock of the Long Now*, 1999. |
| **When to use** | When the field is about to inflect and you want to ship the v0.1 of something that's obvious from 2035 but invisible from 2026. |

### `dream` — unconstrained generative (feasibility OFF)

| | |
|---|---|
| **What it does** | One pass per `n_ideas`. The prediction-error signal is offline; the generative model runs without external correction. Output is a "dream image" that may violate physics / economics / regulation, followed by a "what survives waking" line naming the salvageable fragment. |
| **Cognitive model** | Sleep-state cognition under predictive processing. Friston: dreams are *complexity reduction* — the brain replaying the day's data with the prediction-error mechanism disabled so it can test connections that would have been rejected as too high-error in waking life. Dreams as **synaptic garbage collection** (Friston 2010) and as **invention labs** (the structure of benzene, the periodic table, the sewing machine — all examples of dream-state recombination yielding waking-state insight). |
| **Sources** | Friston, "The free-energy principle: a unified brain theory?", *Nature Reviews Neuroscience* 2010. Hobson & Friston, "Waking and dreaming consciousness: Neurobiological and functional considerations", *Progress in Neurobiology* 2012. Activation-synthesis hypothesis (J. Allan Hobson, 1970s). Stickgold on memory consolidation in REM sleep. |
| **When to use** | When you want to escape the gravity of current practice entirely. Output looks wild on first read — the value is in the "what survives waking" line, the way real dream-insight only resolves after the priors come back. |

### `lucid` — hybrid: dream + injected prior

| | |
|---|---|
| **What it does** | Same dream-state generation, but you inject a directional prior the hallucination must bias toward. One explicit reality check fires before waking. |
| **Cognitive model** | Lucid dreaming as a hybrid state: prefrontal metacognition is online (the dreamer knows they are dreaming) while the rest of the brain remains in the generative, prediction-error-suppressed mode. The dreamer can then *inject high-confidence priors* into the running model — wanting to fly biases "upward motion" toward the high-confidence state rather than fighting gravity. |
| **Sources** | Stephen LaBerge, *Exploring the World of Lucid Dreaming*, 1990 (Stanford lucid-dream research). Voss, Holzmann, Tuin & Hobson, "Lucid dreaming: a state of consciousness with features of both waking and non-lucid dreaming", *Sleep* 2009 (gamma-band activity findings). |
| **When to use** | When you want the wild combinatorial reach of dream-mode but with a steering input — "stay solo-founder-sized", "free version must remain genuinely free", "no recruiters". The prior anchors the hallucination, so more salvage survives waking than pure `dream`. |

### Entropy levels — Carhart-Harris's Entropic Brain hypothesis

The named entropy levels (`sane / wild / insane / crazy / mad`) parametrize
*how high-precision the field's priors should be treated as*, on the
spectrum Carhart-Harris describes in "The Entropic Brain" (*Frontiers in
Human Neuroscience* 2014). Low entropy = priors precise = constrained
inference = standard practice. High entropy = priors weak = expanded
search space = REBUS-like state. The `mad` level is not implemented by
asking for higher temperature — it is implemented by explicitly
instructing the model to *treat one prior as noise*, then ship anyway.
This is the structural translation of the Entropic Brain framework into
a prompt.

### Refine + Evolve-Deck — Bayesian updating across runs

The `refine` flag is **Bayesian model selection**: after generating N
ideas, score each on three independent axes (feasibility / unexpectedness
/ topic-fit), select the maximum posterior, then re-prompt at a tighter
prior. This is the "sober validation" step the LSD literature names as
the second half of psychedelic-assisted insight.

The `evolve_deck` flag implements **memory consolidation** in the
Loftus / reconstructive-memory sense (Elizabeth Loftus, decades of work
on memory as a Read/Write process). When a card contributes to a
winning idea, the card itself is rewritten to better carry the
structural insight that worked. The deck becomes a learning artifact
across sessions — closer to how the brain actually represents
experience (reconstructed each time, not retrieved verbatim).

## Theme generator

The donor-domain list is **LLM-generated per request**, biased by entropy
— not a hardcoded set. At low entropy the themes stay in-field
("VC due diligence", "product-led growth loops"). At high entropy they go
wildly distant ("Carthaginian tophet rites", "cordyceps host
manipulation", "Edo-period firefighter matoi signaling"). Override
explicitly with `--themes "harbor pilotage, mycology, monastic rules"` or
tune separately from the main entropy with `--theme-entropy 0.85`.

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
