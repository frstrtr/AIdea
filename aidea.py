"""AIdea — inference-driven entropy idea generator.

Models the brain's "stochastic synthesis" pattern, applied to a user's
actual request:

  user topic (target)         concept bank (donor material)
        \\                          /
         \\                        /
          \\  [ entropy knob ] <--/
           \\        |
            v       v
      [ inference engine (LLM) ]
            |
            v
  unexpected-but-feasible idea attached to the topic

The entropy knob does two things at once:
  (1) Spread: how cross-domain the donor concepts are.
  (2) Audacity: how far the synthesis is allowed to depart from
      established practice in the topic's field.

Five named levels span a deliberate gradient:
  sane  -> wild  -> insane  -> crazy  -> mad
Any float in [0,1] is also accepted for fine control.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)

DECK_CACHE_DIR = Path(__file__).parent / "decks"


# Seed concept bank. These are *donor concepts* — structural mechanisms
# from various domains that get cross-pollinated onto the user's topic.
DEFAULT_BANK: dict[str, list[str]] = {
    "biology": [
        "mycelium networks", "swarm intelligence", "neural plasticity",
        "symbiosis", "epigenetic inheritance", "morphogenesis",
        "immune memory", "apoptosis", "quorum sensing",
    ],
    "physics": [
        "quantum entanglement", "phase transitions", "entropy gradients",
        "self-organized criticality", "resonance", "wave interference",
        "topological defects", "broken symmetry", "thermodynamic equilibrium",
    ],
    "computing": [
        "garbage collection", "content-addressable memory", "lazy evaluation",
        "exponential backoff", "merkle trees", "gossip protocols",
        "copy-on-write", "bloom filters", "vector clocks",
    ],
    "economics": [
        "tragedy of the commons", "moral hazard", "creative destruction",
        "network effects", "signaling", "Pareto frontiers",
        "auction theory", "principal-agent problem", "price discovery",
    ],
    "art": [
        "negative space", "chiaroscuro", "wabi-sabi",
        "memento mori", "trompe-l'oeil", "found object",
        "serial composition", "ekphrasis", "palimpsest",
    ],
    "urbanism": [
        "desire paths", "third places", "induced demand",
        "permeable surfaces", "edge density", "transit-oriented development",
        "fifteen-minute city", "shared streets", "land value capture",
    ],
    "psychology": [
        "flow state", "default mode network", "embodied cognition",
        "predictive coding", "Dunning-Kruger", "intermittent reinforcement",
        "cognitive load", "anchoring", "narrative identity",
    ],
    "linguistics": [
        "code-switching", "grammaticalization", "metaphor extension",
        "phonosemantics", "loan translation", "deixis",
        "register shift", "polysemy", "implicature",
    ],
}


@dataclass(frozen=True)
class EntropyLevel:
    name: str
    spread: float          # cross-domain spread for concept sampling
    instruction: str       # audacity guidance for the synthesizer


# Ordered from tame to unhinged. Spread values are deliberately spaced.
ENTROPY_LEVELS: list[EntropyLevel] = [
    EntropyLevel(
        name="sane",
        spread=0.10,
        instruction=(
            "Stay within established practice in the topic's field. Refine, "
            "recombine, or apply known techniques cleanly. The user should "
            "react with 'yes, we should just do that.' Boring-but-correct is "
            "the target."
        ),
    ),
    EntropyLevel(
        name="wild",
        spread=0.40,
        instruction=(
            "Combine familiar approaches in uncommon ways. Surprise the user "
            "while leaving the building blocks recognizable. They should react "
            "with 'huh, I didn't think of putting those together.'"
        ),
    ),
    EntropyLevel(
        name="insane",
        spread=0.65,
        instruction=(
            "Transplant a structural mechanism from an unrelated domain onto "
            "the topic. The user should react with 'wait, can we actually do "
            "that here?' and the answer must be yes — with a path you can "
            "describe in 3 sentences."
        ),
    ),
    EntropyLevel(
        name="crazy",
        spread=0.85,
        instruction=(
            "Challenge a load-bearing assumption the field currently treats as "
            "fixed. The proposal should sound reckless to a domain expert on "
            "first read, but survive their second read. If it doesn't survive, "
            "you haven't done the work — pick a different assumption to break."
        ),
    ),
    EntropyLevel(
        name="mad",
        spread=0.98,
        instruction=(
            "Reinterpret the problem itself: argue the user is solving the "
            "wrong version of it, then propose the right version's solution. "
            "The first read should feel absurd. You MUST finish by showing "
            "exactly how a small team ships a v0.1 within six months — if you "
            "can't, the idea isn't mad, it's just fiction. Reject fiction."
        ),
    ),
]


def find_level_near(value: float) -> EntropyLevel:
    """Return the named level whose spread is closest to value."""
    return min(ENTROPY_LEVELS, key=lambda lvl: abs(lvl.spread - value))


def parse_entropy(raw: str) -> tuple[float, EntropyLevel]:
    """Accept a named level or a float; return (spread, level-for-instruction)."""
    by_name = {lvl.name: lvl for lvl in ENTROPY_LEVELS}
    if raw in by_name:
        lvl = by_name[raw]
        return lvl.spread, lvl
    try:
        v = float(raw)
    except ValueError:
        names = ", ".join(lvl.name for lvl in ENTROPY_LEVELS)
        raise SystemExit(
            f"--entropy must be one of [{names}] or a float in [0.0, 1.0]; "
            f"got {raw!r}"
        )
    if not 0.0 <= v <= 1.0:
        raise SystemExit("--entropy float must be in [0.0, 1.0]")
    return v, find_level_near(v)


@dataclass(frozen=True)
class CardDepth:
    name: str
    target_tokens: int       # rough per-card budget (for user awareness)
    fields: tuple[str, ...]  # which fields each generated card must carry
    description: str         # for the deck-gen prompt


# Ordered from minimal to maximal pre-seeded detail.
CARD_DEPTHS: list[CardDepth] = [
    CardDepth(
        name="shallow",
        target_tokens=15,
        fields=("name", "domain"),
        description=(
            "Just the bare concept name (2-5 words) and its source domain. "
            "No mechanism, no commentary."
        ),
    ),
    CardDepth(
        name="medium",
        target_tokens=60,
        fields=("name", "domain", "mechanism"),
        description=(
            "Concept name, source domain, and ONE sentence describing the "
            "structural mechanism — how it actually works, not what it is "
            "called."
        ),
    ),
    CardDepth(
        name="deep",
        target_tokens=200,
        fields=("name", "domain", "mechanism", "why", "transfer"),
        description=(
            "Concept name and domain. Then: a 2-3 sentence mechanism (how it "
            "works), one sentence on WHY it works (the structural invariant), "
            "and one sentence sketching how it might transfer onto a problem "
            "in a different domain."
        ),
    ),
    CardDepth(
        name="max",
        target_tokens=500,
        fields=(
            "name", "domain", "mechanism", "why", "invariants",
            "prior_application",
        ),
        description=(
            "Full structured card: name, domain, mechanism (3-4 sentences), "
            "why it works (1-2 sentences), the domain-independent invariants "
            "that allow transfer (1-2 sentences), and at least one prior "
            "cross-domain application or analogy showing the mechanism has "
            "moved before."
        ),
    ),
]

CARD_DEPTH_BY_NAME = {d.name: d for d in CARD_DEPTHS}


@dataclass
class Card:
    """One donor concept. Body fields are optional depending on depth."""
    name: str
    domain: str
    mechanism: str | None = None
    why: str | None = None
    transfer: str | None = None
    invariants: str | None = None
    prior_application: str | None = None

    def render(self) -> str:
        """Render the card for the synthesizer prompt."""
        lines = [f"- {self.name} (from {self.domain})"]
        for label, val in (
            ("Mechanism", self.mechanism),
            ("Why it works", self.why),
            ("Transfer hint", self.transfer),
            ("Invariants", self.invariants),
            ("Prior application", self.prior_application),
        ):
            if val:
                lines.append(f"    {label}: {val}")
        return "\n".join(lines)


DECK_GEN_SYSTEM = (
    "You generate donor concept decks for an idea-synthesis tool. Your output "
    "is JSON Lines (one JSON object per line, no commentary, no surrounding "
    "fences). Each object is one card."
)


def _deck_gen_prompt(topic: str, n: int, depth: CardDepth) -> str:
    field_list = ", ".join(f'"{f}"' for f in depth.fields)
    return f"""\
Generate {n} donor concepts that could cross-pollinate with this user topic:

  {topic.strip()}

These will be shuffled to inject controlled entropy into idea generation.
Optimize for BREADTH of source domains and STRUCTURAL diversity. Aim for
concepts whose mechanisms are domain-independent enough to be transferable,
not concepts that already live near the user's topic.

Hard requirements:
- Span at least 8 distinct source domains (biology, physics, computing,
  economics, art, urbanism, psychology, linguistics, history, music,
  warfare, religion, law, sports, etc.). Do not cluster.
- Avoid domains that would feel obvious for this topic — pick the
  unobvious-but-transferable instead.
- No duplicates. No near-duplicates.
- {depth.description}

Output format: exactly {n} JSON objects, one per line, no array brackets,
no preamble, no closing remarks. Each object MUST have these fields:
[{field_list}]. All string values. No nested objects, no arrays.

Begin:"""


def _normalize_topic_for_hash(topic: str) -> str:
    return re.sub(r"\s+", " ", topic.strip().lower())


def _deck_cache_path(topic: str, n: int, depth: CardDepth, model: str) -> Path:
    key = json.dumps(
        {
            "topic": _normalize_topic_for_hash(topic),
            "n": n,
            "depth": depth.name,
            "model": model,
        },
        sort_keys=True,
    )
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    DECK_CACHE_DIR.mkdir(exist_ok=True)
    return DECK_CACHE_DIR / f"{h}.json"


def _parse_jsonl_cards(text: str, depth: CardDepth) -> list[Card]:
    """Parse the model's JSONL output. Tolerant of stray fences/commentary."""
    cards: list[Card] = []
    for raw in text.splitlines():
        line = raw.strip().strip(",")
        if not line or line.startswith("```") or line in ("[", "]"):
            continue
        # Strip trailing array commas: }, -> }
        if line.endswith(",}"):
            line = line[:-2] + "}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "name" not in obj or "domain" not in obj:
            continue
        cards.append(
            Card(
                name=str(obj.get("name", "")).strip(),
                domain=str(obj.get("domain", "")).strip(),
                mechanism=_opt(obj, "mechanism"),
                why=_opt(obj, "why"),
                transfer=_opt(obj, "transfer"),
                invariants=_opt(obj, "invariants"),
                prior_application=_opt(obj, "prior_application"),
            )
        )
    return cards


def _opt(obj: dict, key: str) -> str | None:
    v = obj.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


async def generate_deck(
    topic: str,
    n: int,
    depth: CardDepth,
    model: str,
    verbose: bool,
) -> list[Card]:
    """One LLM call -> a topic-aware donor deck."""
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=DECK_GEN_SYSTEM,
        max_turns=1,
        allowed_tools=[],
    )
    prompt = _deck_gen_prompt(topic, n, depth)

    chunks: list[str] = []
    if verbose:
        print(f"[deck] generating {n} cards at depth={depth.name}...", flush=True)
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
    text = "".join(chunks)
    cards = _parse_jsonl_cards(text, depth)
    if verbose:
        print(f"[deck] parsed {len(cards)} cards", flush=True)
    return cards


async def load_or_generate_deck(
    topic: str,
    n: int,
    depth: CardDepth,
    model: str,
    force_regen: bool,
    verbose: bool,
) -> list[Card]:
    path = _deck_cache_path(topic, n, depth, model)
    if path.exists() and not force_regen:
        if verbose:
            print(f"[deck] using cached deck at {path}", flush=True)
        data = json.loads(path.read_text())
        return [Card(**c) for c in data]
    cards = await generate_deck(topic, n, depth, model, verbose)
    if not cards:
        raise RuntimeError(
            "Deck generation produced no parseable cards. "
            "Try a different model or --regen-deck."
        )
    path.write_text(
        json.dumps([c.__dict__ for c in cards], indent=2, ensure_ascii=False)
    )
    if verbose:
        print(f"[deck] cached {len(cards)} cards to {path}", flush=True)
    return cards


def cards_from_static_bank(bank: dict[str, list[str]]) -> list[Card]:
    """Convert the legacy {domain: [concept,...]} bank into shallow Cards."""
    return [
        Card(name=concept, domain=domain)
        for domain, concepts in bank.items()
        for concept in concepts
    ]


def sample_cards(
    deck: list[Card],
    n: int,
    spread: float,
    rng: random.Random,
) -> list[Card]:
    """Sample n cards with controlled cross-domain spread."""
    if not deck:
        raise ValueError("Deck is empty.")
    # Group by domain
    by_domain: dict[str, list[Card]] = {}
    for c in deck:
        by_domain.setdefault(c.domain, []).append(c)
    domains = list(by_domain.keys())

    start_domain = rng.choice(domains)
    chosen: list[Card] = [rng.choice(by_domain[start_domain])]
    used_ids = {id(chosen[0])}

    while len(chosen) < n:
        if rng.random() < spread and len(domains) > 1:
            d = rng.choice([x for x in domains if x != start_domain])
        else:
            d = start_domain
        pool = [c for c in by_domain.get(d, []) if id(c) not in used_ids]
        if not pool:
            # Domain exhausted — fall back to any unused card anywhere
            remaining = [c for c in deck if id(c) not in used_ids]
            if not remaining:
                break
            pick = rng.choice(remaining)
            chosen.append(pick)
            used_ids.add(id(pick))
            continue
        pick = rng.choice(pool)
        chosen.append(pick)
        used_ids.add(id(pick))

    return chosen


APPLIED_PROMPT_TEMPLATE = """\
The user is working on this problem / question / project:

  {topic}

You have these donor concepts as raw material for cross-pollination. They
were sampled stochastically — your job is to find which one(s) carry
structural mechanisms that actually apply to the user's problem, not to
shoehorn all of them in.

Donor concepts:
{seeds}

Audacity level: {level_name}
  {instruction}

Hard requirements (apply at EVERY audacity level — even "mad"):
  - The idea must address the user's stated problem, not a related one.
  - It must be executable with technology and resources available today.
  - It must be specific enough that the user can identify a first step
    to try this week.
  - At least one donor concept must contribute the core structural
    mechanism, not just decoration or a name-drop.

Respond in exactly this format, no preamble, no closing remarks:

Title: <3-7 memorable words>
One-line pitch: <single sentence connecting the idea to the user's problem>
How it addresses the request: <2-3 sentences — be concrete about which
  aspect of the user's problem this targets>
Mechanism: <2-4 sentences — name which donor concept(s) supply the
  structure and how the borrowing actually works>
Why it's unexpected: <1-2 sentences>
First step the user could take this week: <one concrete action>
Risks / what could break: <1-2 sentences naming the most likely failure
  mode of this specific idea>
"""


# ---------------------------------------------------------------------------
# Einstein Mode: ideas don't appear from thin air. They emerge through four
# distinct, well-attested mechanisms. Run one synthesis per mechanism so the
# user can see WHICH route produced their winning idea, not just a temperature
# variant of the same default route.
# ---------------------------------------------------------------------------


EINSTEIN_PREAMBLE = """\
The user is working on this problem / question / project:

  {topic}

The four mechanisms below produce ideas in fundamentally different ways.
This run uses the {mechanism} mechanism. Apply ONLY that mechanism — do
not slide into any of the others.

Donor concepts (raw material; their role varies by mechanism, see below):
{seeds}

Hard requirements (apply to every mechanism):
  - The idea must address the user's stated problem, not a related one.
  - It must be executable with current technology and resources.
  - It must be specific enough that the user can identify a first step
    to try this week.
"""


EINSTEIN_MECHANISMS: dict[str, dict[str, str]] = {
    "adjacent_possible": {
        "label": "Adjacent Possible",
        "blurb": (
            "Kauffman / Steven Johnson. A field's frontier only steps to "
            "doors recently unlocked by prior work. Find one such door."
        ),
        "instruction": """\
Process:
  1. Identify ONE capability that has become widely available in the
     user's field in the last 1-3 years (a new technique, library,
     dataset, regulation, piece of hardware, market shift).
  2. Identify ONE adjacent unmet need that this capability makes
     newly addressable — something that wasn't viable before.
  3. Propose the specific step through that door.

Donor cards play a supporting role here: use them to widen your search
for "what just got unlocked", but do not let them become the focus.

Respond in this exact format, no preamble:

Title: <3-7 memorable words>
Mechanism: Adjacent Possible
Recently unlocked: <the specific capability that became viable in the last 1-3 years>
Adjacent unmet need: <what becomes newly addressable now>
One-line pitch: <how the user's problem walks through this door>
How it addresses the request: <2-3 sentences>
First step the user could take this week: <one concrete action>
Risks / what could break: <1-2 sentences>
""",
    },
    "exaptation": {
        "label": "Exaptation",
        "blurb": (
            "Gutenberg moved the wine-press mechanism to inked type. Pick a "
            "mechanism built for something else and re-contextualize it."
        ),
        "instruction": """\
Process:
  1. Pick exactly ONE donor card from the list — preferably the one whose
     home domain is FURTHEST from the user's topic.
  2. Name the structural operating principle that makes it work in its
     home domain — not the surface description.
  3. Apply that same operating principle to the user's topic. The
     transplant is the idea. The remaining donor cards are decoration.

Respond in this exact format, no preamble:

Title: <3-7 memorable words>
Mechanism: Exaptation
Borrowed from: <donor card name and its home domain>
Operating principle being transplanted: <one precise sentence>
Structural mapping: <2-3 sentences showing how the principle attaches to the topic>
One-line pitch: <single sentence stating the resulting idea>
How it addresses the request: <2-3 sentences>
First step the user could take this week: <one concrete action>
Risks / what could break: <1-2 sentences>
""",
    },
    "slow_hunch": {
        "label": "Slow Hunch",
        "blurb": (
            "Long-incubated background tensions that haven't been articulated. "
            "State the unsaid resolution."
        ),
        "instruction": """\
Process:
  1. Identify ONE tension, contradiction, or half-answered question that
     practitioners in the user's field have lived with for years but
     never cleanly resolved. Watch for: two widely-held beliefs that
     don't square; a workaround everyone uses but no one defends; a
     pattern that "everyone notices" but no product addresses.
  2. State the simplest framing that resolves it.
  3. Propose the project that bets on the resolution being correct.

Donor cards play a background role: use them only if one obviously
illuminates the tension. Do not force them in.

Respond in this exact format, no preamble:

Title: <3-7 memorable words>
Mechanism: Slow Hunch
The latent tension: <what practitioners feel but don't say cleanly>
Proposed resolution: <the framing that resolves the tension>
One-line pitch: <single sentence>
How it addresses the request: <2-3 sentences>
First step the user could take this week: <one concrete action>
Risks / what could break: <1-2 sentences — especially: what if the tension is real but the resolution is wrong>
""",
    },
    "productive_error": {
        "label": "Productive Error",
        "blurb": (
            "Penicillin / biological mutation. A misread or inverted assumption "
            "that turns out to be more useful than the correct one."
        ),
        "instruction": """\
Process:
  1. Pick ONE assumption practitioners in the user's field take as
     obviously true. Watch for: "of course", "everyone knows",
     "naturally we...", "the whole point of X is Y".
  2. Deliberately invert it, misread it, or take the opposite as true.
  3. Argue why the misreading is actually the better framing for the
     user's problem, then propose the project that bets on it.

Donor cards play a background role.

Respond in this exact format, no preamble:

Title: <3-7 memorable words>
Mechanism: Productive Error
Assumption being inverted: <the load-bearing belief, as the field states it>
The misreading: <the deliberately wrong version that turns out useful>
Why the misreading wins: <2-3 sentences arguing why this framing is better>
One-line pitch: <single sentence>
How it addresses the request: <2-3 sentences>
First step the user could take this week: <one concrete action>
Risks / what could break: <1-2 sentences>
""",
    },
}


def build_einstein_prompt(
    topic: str,
    cards: list[Card],
    mechanism_key: str,
) -> str:
    """Build a synthesis prompt locked to one of the four Einstein mechanisms."""
    if mechanism_key not in EINSTEIN_MECHANISMS:
        raise ValueError(
            f"unknown Einstein mechanism {mechanism_key!r}; "
            f"known: {list(EINSTEIN_MECHANISMS)}"
        )
    mech = EINSTEIN_MECHANISMS[mechanism_key]
    seeds = "\n".join(card.render() for card in cards)
    preamble = EINSTEIN_PREAMBLE.format(
        topic=topic.strip(),
        mechanism=mech["label"],
        seeds=seeds,
    )
    return preamble + "\n" + mech["instruction"]


# ---------------------------------------------------------------------------
# LSD / Prior-Dissolution mode. Predictive-processing framing: perception is
# a controlled hallucination — the brain runs a model and constructs
# "reality" from priors plus minimal sensory data (Friston, Seth, REBUS).
# Fields work the same way: the way practitioners perceive their problem is
# a construct, not a direct view. This mode loosens the interpretive prior
# instead of inverting a single belief.
# ---------------------------------------------------------------------------


LSD_LABEL = "Prior Dissolution"

LSD_PROMPT_TEMPLATE = """\
The user is working on this problem / question / project:

  {topic}

This synthesis uses PRIOR DISSOLUTION. Predictive-processing neuroscience
(Friston, Seth, Carhart-Harris's REBUS framing of psychedelics) treats
perception as a controlled hallucination: the brain runs a generative
model and constructs "reality" from high-level priors plus minimal
sensory data. Practitioners in any field do the same thing — the way
they perceive their problem is a learned interpretive frame, not a
direct view of the situation. Psychedelics interest neuroscientists
because they relax those high-level priors and let the same raw input
be re-organized under a different category.

Your job is to suspend the field's interpretive frame and re-organize
the same underlying situation under a different category. This is NOT:

  - inverting a single assumption (that's Productive Error)
  - borrowing a mechanism from another field (that's Exaptation)
  - stepping forward through a recently-unlocked door (that's Adjacent
    Possible)

It is dissolving the silent category the field uses to perceive the
problem at all, and seeing what becomes obvious without that prior.

Donor concepts (raw material; some may resonate, most will not — that
is expected):
{seeds}

Process:
1. Name ONE load-bearing prior the field uses to perceive this problem.
   A prior is the silent, almost-invisible category-membership the field
   assigns the problem. Examples in other domains:
     - "Email is messages between inboxes" (prior: communication is
       atomic, addressed, durable).
     - "A car is for getting from A to B" (prior: cars are tools, not
       experiences).
     - "Education is content delivery" (prior: knowledge moves from
       teacher to student).
   Pick the prior NO ONE in the user's field would think to question.

2. Dissolve it. Suspend the category and look at the same raw situation
   without that frame installed.

3. Re-perceive. The same situation, re-categorized, often makes the
   problem dissolve, become trivial, or reveal a completely different
   project. Your idea is what becomes obvious AFTER the prior is gone.

Hard requirements:
  - The idea must address the user's stated problem, OR show why under
    the new frame the "problem" disappears (that is a valid output).
  - It must be executable with technology and resources available today.
  - It must be specific enough that the user can identify a first step.

Respond in EXACTLY this format, no preamble, no closing remarks:

Title: <3-7 memorable words>
Mechanism: Prior Dissolution
The prior being dissolved: <the silent interpretive frame the field uses>
What the same situation becomes without that prior: <2-3 sentences re-categorizing the situation>
One-line pitch: <how the idea follows naturally from the new frame>
How it addresses the request: <2-3 sentences>
Mechanism (technical): <2-3 sentences on how the re-organization actually works in practice>
First step the user could take this week: <one concrete action>
Risks / what could break: <1-2 sentences — especially: what if the original prior was load-bearing for a reason no one ever wrote down>
"""


def build_lsd_prompt(topic: str, cards: list[Card]) -> str:
    seeds = "\n".join(card.render() for card in cards)
    return LSD_PROMPT_TEMPLATE.format(topic=topic.strip(), seeds=seeds)


# ---------------------------------------------------------------------------
# Futures Mode: predictive-processing framing taken seriously. The brain
# compensates for its ~100ms perceptual delay by hallucinating the immediate
# future. This mode runs that forward simulation for the user's field at a
# specific horizon and asks what idea is OBVIOUS from there but invisible
# from today. Then translates it back to a v0.1 the user can ship now.
# ---------------------------------------------------------------------------


FUTURES_LABEL = "Futures Projection"


FUTURES_HORIZONS: list[dict[str, str]] = [
    {
        "key": "1y",
        "label": "Futures · +1 year",
        "horizon_name": "1 year from now",
        "framing": (
            "Trends already in motion; what is now in pilot becomes the "
            "default. Low-confidence step, low risk. The job is to spot "
            "what is about to be obvious that the field is still calling "
            "an early-adopter quirk."
        ),
    },
    {
        "key": "3y",
        "label": "Futures · +3 years",
        "horizon_name": "3 years from now",
        "framing": (
            "What is currently rare-but-validated becomes commodity. The "
            "first big incumbent loses ground or dies. Costs collapse on "
            "things people still treat as expensive today."
        ),
    },
    {
        "key": "10y",
        "label": "Futures · +10 years",
        "horizon_name": "10 years from now",
        "framing": (
            "Structural shifts visible only in today's early-adopter "
            "cohorts have become the default for everyone. Today's "
            "fringe is tomorrow's mainstream. Whole job categories "
            "appear or vanish."
        ),
    },
    {
        "key": "30y",
        "label": "Futures · +30 years",
        "horizon_name": "30 years from now",
        "framing": (
            "Regime change. The frame everyone uses to think about this "
            "field today is gone — and would sound naive to a "
            "practitioner from then. The Wright-brothers vantage point: "
            "what is obvious only from a future that hasn't been built "
            "yet."
        ),
    },
]

FUTURES_HORIZONS_BY_KEY = {h["key"]: h for h in FUTURES_HORIZONS}


FUTURES_PROMPT_TEMPLATE = """\
The user is working on this problem / question / project:

  {topic}

This synthesis uses TEMPORAL PROJECTION. The brain compensates for its
~100ms perceptual delay by hallucinating the immediate future — running
forward simulations off the current model. This mode runs that forward
simulation for the user's field at a specific horizon and asks what idea
is OBVIOUS from there but invisible from today.

Your horizon: {horizon_name}.
Framing for that horizon: {framing}

Donor concepts (raw material; use them to widen your forecast, not to
constrain it):
{seeds}

Process:
1. Simulate the user's field at {horizon_name}. Name three concrete
   shifts that are likely by then: a capability, a constraint, and a
   behavior change. Be specific — name technologies, regulations, user
   habits, cost curves — not adjectives. If you cannot name them, your
   forecast is too vague; tighten it.
2. From that vantage point, identify ONE idea that is OBVIOUS at the
   horizon but invisible (or laughable, or "impossible") from today.
   The Wright brothers in 1900 could see "people fly between cities";
   the world in 1900 said "impossible".
3. Translate it back to NOW. What is the v0.1 a small team could ship
   THIS YEAR that walks toward that obvious-from-the-future idea? The
   v0.1 must be feasible with today's tools even if the full version is
   years away.

Hard requirements:
  - The v0.1 must be executable today with current technology.
  - It must be specific enough that the user can identify a first step
    to try this week.
  - The "three shifts" must each be concrete and defensible — not
    "AI gets better".

Respond in EXACTLY this format, no preamble:

Title: <3-7 memorable words>
Mechanism: Futures Projection
Horizon: {horizon_name}
Three shifts by then: <three concrete near-certain shifts: capability, constraint, behavior>
What is obvious from there: <2-3 sentences on the idea visible from the future>
One-line pitch: <single sentence describing the v0.1 you can ship NOW that walks toward it>
How it addresses the request: <2-3 sentences>
Mechanism (technical): <2-3 sentences on how the v0.1 actually works today>
First step the user could take this week: <one concrete action>
Risks / what could break: <1-2 sentences — especially: WHICH of your three shifts is the weakest assumption, and what happens if it doesn't materialize on schedule>
"""


def build_futures_prompt(
    topic: str,
    cards: list[Card],
    horizon_key: str,
) -> str:
    if horizon_key not in FUTURES_HORIZONS_BY_KEY:
        raise ValueError(
            f"unknown futures horizon {horizon_key!r}; "
            f"known: {list(FUTURES_HORIZONS_BY_KEY)}"
        )
    h = FUTURES_HORIZONS_BY_KEY[horizon_key]
    seeds = "\n".join(card.render() for card in cards)
    return FUTURES_PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        seeds=seeds,
        horizon_name=h["horizon_name"],
        framing=h["framing"],
    )


def build_prompt(
    topic: str,
    cards: list[Card],
    level: EntropyLevel,
) -> str:
    seeds = "\n".join(card.render() for card in cards)
    return APPLIED_PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        seeds=seeds,
        level_name=level.name,
        instruction=level.instruction,
    )


SYNTHESIZER_SYSTEM = (
    "You are an applied-ideas synthesizer. You take a user problem and a set "
    "of cross-domain donor concepts and produce ONE specific, executable idea. "
    "Respond in the exact format requested by the user — no preamble, no "
    "closing remarks, no offers to help further."
)


async def _query_text(prompt: str, system: str, model: str) -> str:
    """One-shot inference call returning concatenated text."""
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system,
        max_turns=1,
        allowed_tools=[],
    )
    chunks: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
    return "".join(chunks)


async def synthesize(
    prompt: str,
    model: str,
    stream_to_stdout: bool,
) -> str:
    """Run the inference engine via the agent SDK (inherits CLI auth)."""
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=SYNTHESIZER_SYSTEM,
        max_turns=1,
        allowed_tools=[],
    )

    chunks: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
                    if stream_to_stdout:
                        print(block.text, end="", flush=True)

    if stream_to_stdout:
        print()
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Critic + Refinement: score ideas, then harden the winner at low entropy.
# ---------------------------------------------------------------------------


CRITIC_SYSTEM = (
    "You are a critic for an applied-ideas synthesizer. You score a single "
    "idea against a user problem on three axes. Output STRICT JSON only — "
    "one object, no surrounding code fences, no preamble, no commentary."
)


CRITIC_PROMPT_TEMPLATE = """\
The user is working on this problem:

  {topic}

Score this idea on three axes from 0 to 100. Be honest — your job is to
differentiate good ideas from filler, so use the full range. Reserve 90+
for genuinely strong, reserve 0-20 for clearly weak; cluster the middle.

  - feasibility: can a small team execute this with current technology in
    the user's stated field within months? 100 = obviously yes, 0 = science
    fiction.
  - unexpectedness: how non-obvious is the structural choice? 100 = "I did
    not see that coming"; 0 = "I would have thought of this in 30 seconds".
  - topic_fit: does this address the user's stated problem (vs a tangent)?
    100 = bullseye; 0 = answers a different question.

Idea to score:

{idea}

Respond as ONE JSON object only, with these keys exactly:
  {{"feasibility": <int 0-100>, "unexpectedness": <int 0-100>,
    "topic_fit": <int 0-100>, "notes": "<one sentence — what's strongest,
    what's weakest>"}}
"""


def _try_parse_json_object(raw: str) -> dict | None:
    """Tolerantly extract the first JSON object from a model response."""
    s = raw.strip()
    # Strip leading/trailing code fences if present.
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?|\n?```$", "", s).strip()
    # Greedy from the first { to the last } in the string.
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1 or b < a:
        return None
    try:
        obj = json.loads(s[a : b + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


async def critic_score(topic: str, idea: str, model: str) -> dict:
    """Return {feasibility, unexpectedness, topic_fit, notes} (ints clipped 0-100)."""
    prompt = CRITIC_PROMPT_TEMPLATE.format(topic=topic.strip(), idea=idea.strip())
    raw = await _query_text(prompt, CRITIC_SYSTEM, model)
    obj = _try_parse_json_object(raw) or {}

    def _clip(v: object) -> int:
        try:
            return max(0, min(100, int(float(v))))
        except (TypeError, ValueError):
            return 0

    return {
        "feasibility": _clip(obj.get("feasibility")),
        "unexpectedness": _clip(obj.get("unexpectedness")),
        "topic_fit": _clip(obj.get("topic_fit")),
        "notes": str(obj.get("notes", "")).strip()[:500],
    }


def total_score(score: dict) -> int:
    return (
        score.get("feasibility", 0)
        + score.get("unexpectedness", 0)
        + score.get("topic_fit", 0)
    )


REFINE_PROMPT_TEMPLATE = """\
The user is working on this problem:

  {topic}

Below is a candidate idea that scored highest in a previous pass. The
critic's note: "{notes}"

Your job now is to HARDEN it. Keep the core structural mechanism — do not
swap it for a different one. Instead:

  - Make the "First step the user could take this week" radically more
    concrete: name specific tools, file formats, command names, data
    sources, or vendors a single person could touch on day one.
  - Take the named risk seriously: state the specific mitigation in the
    Risks line, not just "we'd have to be careful".
  - Tighten the one-line pitch so a smart stranger gets it in one read.
  - Do not change the title unless the original is genuinely misleading.

Original idea:

{idea}

Respond in EXACTLY the same format the synthesizer uses (Title,
One-line pitch, How it addresses the request, Mechanism, Why it's
unexpected, First step the user could take this week, Risks). No preamble,
no closing remarks.
"""


async def refine_idea(
    topic: str, idea: str, notes: str, model: str,
) -> str:
    prompt = REFINE_PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        idea=idea.strip(),
        notes=notes.strip() or "no specific critique provided",
    )
    return await _query_text(prompt, SYNTHESIZER_SYSTEM, model)


# ---------------------------------------------------------------------------
# Deck evolution: when an idea wins the critic round, sharpen the cards
# that contributed to it so they better carry the structural insight that
# made the idea work. Memory as Read/Write: the deck is a learning artifact.
# ---------------------------------------------------------------------------


DECK_EVOLVE_SYSTEM = (
    "You sharpen donor concept cards in a deck used by an applied-ideas "
    "tool. Given a winning idea + the cards that contributed to it, you "
    "rewrite each card so a future reader could see — from the card "
    "alone — why this concept was transferable to that kind of problem. "
    "Output strict JSONL: one JSON object per line, no preamble, no "
    "code fence, no commentary."
)


DECK_EVOLVE_PROMPT_TEMPLATE = """\
The user's problem was:

  {topic}

The winning idea (after refinement):

  {idea}

The cards below were drawn from the deck to generate that idea. Sharpen
each card so the structural insight the winning idea exploited is now
visible in the card itself. Keep the card name and source domain
EXACTLY as given. Update only the body fields ({fields_list}). Do NOT
invent new fields. Do NOT swap one card for a different concept.

Cards to sharpen:
{cards_json}

For each card, output exactly one JSON object on its own line, with
fields: {fields_list_quoted}. Do NOT wrap them in an array. Do NOT add
commentary. Begin now:
"""


async def evolve_cards(
    topic: str,
    idea: str,
    cards: list[Card],
    depth: CardDepth,
    model: str,
) -> list[Card]:
    """Rewrite donor cards in light of a winning idea. Returns updated cards
    (matched by name to the input cards; un-matched stay as-is)."""
    if not cards:
        return []
    fields_list = ", ".join(f for f in depth.fields if f not in ("name", "domain"))
    fields_list_quoted = ", ".join(f'"{f}"' for f in depth.fields)
    cards_json = "\n".join(
        json.dumps(
            {k: v for k, v in c.__dict__.items() if v is not None},
            ensure_ascii=False,
        )
        for c in cards
    )
    prompt = DECK_EVOLVE_PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        idea=idea.strip(),
        cards_json=cards_json,
        fields_list=fields_list or "(none for shallow depth)",
        fields_list_quoted=fields_list_quoted,
    )
    raw = await _query_text(prompt, DECK_EVOLVE_SYSTEM, model)
    new_cards = _parse_jsonl_cards(raw, depth)

    # Match by case-insensitive name; drop any model-invented entries.
    new_by_name = {c.name.lower(): c for c in new_cards}
    out: list[Card] = []
    for original in cards:
        new = new_by_name.get(original.name.lower())
        # Force the original name and domain — model is not allowed to swap.
        if new is not None:
            new.name = original.name
            new.domain = original.domain
            out.append(new)
        else:
            out.append(original)
    return out


def merge_evolved_into_deck(
    deck: list[Card], evolved: list[Card],
) -> list[Card]:
    by_name = {c.name.lower(): c for c in evolved}
    return [by_name.get(c.name.lower(), c) for c in deck]


def save_deck_to_cache(
    topic: str,
    n: int,
    depth: CardDepth,
    model: str,
    deck: list[Card],
) -> Path:
    path = _deck_cache_path(topic, n, depth, model)
    path.write_text(
        json.dumps(
            [
                {k: v for k, v in c.__dict__.items() if v is not None}
                for c in deck
            ],
            indent=2,
            ensure_ascii=False,
        )
    )
    return path


def load_bank(path: str | None) -> dict[str, list[str]]:
    if path is None:
        return DEFAULT_BANK
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not all(
        isinstance(v, list) for v in data.values()
    ):
        raise ValueError(
            "Custom bank must be a JSON object mapping domain -> [concepts]."
        )
    return data


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate unexpected-but-feasible ideas attached to a user "
            "request, with adjustable entropy."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--topic", "-t",
        type=str,
        default=None,
        help=(
            "The user's request / problem / project. Required unless using "
            "--list-domains or --list-levels."
        ),
    )
    p.add_argument(
        "--entropy", "-e",
        type=str,
        default="wild",
        help=(
            "Named level (sane | wild | insane | crazy | mad) "
            "or a float in [0.0, 1.0]."
        ),
    )
    p.add_argument(
        "--n-concepts",
        type=int,
        default=3,
        help="Number of donor cards to SAMPLE per idea (the shuffle pick).",
    )
    p.add_argument(
        "--cards",
        type=int,
        default=30,
        help=(
            "Size of the donor deck to shuffle FROM (pre-seeded raw "
            "material). Generated topic-aware on first use, cached."
        ),
    )
    p.add_argument(
        "--card-depth",
        type=str,
        default="medium",
        choices=[d.name for d in CARD_DEPTHS],
        help=(
            "How detailed each pre-seeded card is. Deeper cards give the "
            "synthesizer richer raw material at the cost of prompt size."
        ),
    )
    p.add_argument(
        "--regen-deck",
        action="store_true",
        help="Force regeneration of the donor deck even if cached.",
    )
    p.add_argument(
        "--n-ideas",
        type=int,
        default=1,
        help="How many independent ideas to generate this run.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-7",
        help="Model ID for the inference engine.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (for reproducible concept sampling).",
    )
    p.add_argument(
        "--bank",
        type=str,
        default=None,
        help="Path to a custom JSON concept bank (domain -> [concepts]).",
    )
    p.add_argument(
        "--list-domains",
        action="store_true",
        help="Print the donor domains in the active bank and exit.",
    )
    p.add_argument(
        "--list-levels",
        action="store_true",
        help="Print the named entropy levels and exit.",
    )
    p.add_argument(
        "--list-depths",
        action="store_true",
        help="Print the named card-depth presets and exit.",
    )
    p.add_argument(
        "--show-deck",
        action="store_true",
        help="Print the resolved donor deck before generating ideas.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Don't stream the model's output; only print the final idea.",
    )
    p.add_argument(
        "--refine",
        action="store_true",
        help=(
            "After generating all ideas, score each and refine the winner at "
            "low entropy ('sane'). Adds 1 critic call per idea + 1 refinement "
            "call. Best used with --n-ideas 3 or more."
        ),
    )
    p.add_argument(
        "--einstein",
        action="store_true",
        help=(
            "Einstein mode: run four mechanism-specific synthesis passes "
            "instead of n_ideas identical ones — Adjacent Possible, "
            "Exaptation, Slow Hunch, Productive Error. Each produces one "
            "idea via a different generative move. Forces n_ideas=4. "
            "Combine with --refine to pick and harden the strongest mechanism."
        ),
    )
    p.add_argument(
        "--lsd",
        action="store_true",
        help=(
            "LSD mode (a.k.a. Prior Dissolution): predictive-processing "
            "framing — perception is a controlled hallucination, so loosen "
            "the field's interpretive prior and re-perceive the same "
            "situation under a different category. Each pass dissolves one "
            "load-bearing prior. Mutually exclusive with --einstein."
        ),
    )
    p.add_argument(
        "--futures",
        action="store_true",
        help=(
            "Futures mode: project the user's field forward at four time "
            "horizons (+1y / +3y / +10y / +30y), identify what is obvious "
            "from each future, and ship a v0.1 today that walks toward it. "
            "Forces n_ideas=4. Mutually exclusive with --einstein and --lsd."
        ),
    )
    p.add_argument(
        "--evolve-deck",
        action="store_true",
        help=(
            "When --refine produces a winner, sharpen the cards that "
            "contributed to it and write them back to the deck cache. "
            "Subsequent runs against the same (topic, cards, depth, model) "
            "see the evolved deck. Has no effect without --refine, and "
            "is rejected with --bank (static banks aren't writable)."
        ),
    )
    args = p.parse_args(argv)
    if args.n_concepts < 2:
        p.error("--n-concepts must be at least 2 (blending needs >= 2 seeds)")
    modes_on = sum(1 for f in (args.einstein, args.lsd, args.futures) if f)
    if modes_on > 1:
        p.error(
            "--einstein, --lsd, --futures are mutually exclusive; pick one."
        )
    if args.evolve_deck and args.bank:
        p.error("--evolve-deck requires a generated deck; not compatible with --bank")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.list_levels:
        for lvl in ENTROPY_LEVELS:
            print(f"{lvl.name:>6}  (spread={lvl.spread:.2f})  {lvl.instruction}")
        return 0

    if args.list_depths:
        for d in CARD_DEPTHS:
            print(
                f"{d.name:>8}  (~{d.target_tokens} tok/card)  "
                f"fields={list(d.fields)}"
            )
            print(f"          {d.description}")
        return 0

    if args.list_domains:
        if args.bank:
            bank = load_bank(args.bank)
            for d in sorted(bank):
                print(f"{d} ({len(bank[d])} concepts)")
            return 0
        print(
            "Donor domains are now generated per-topic. Use --show-deck "
            "with a --topic to inspect the live deck, or pass --bank to use "
            "a static JSON bank.",
            file=sys.stderr,
        )
        return 0

    if not args.topic:
        print(
            "error: --topic is required. Describe your request, e.g.\n"
            '  --topic "reducing churn in a B2B SaaS for civil engineers"',
            file=sys.stderr,
        )
        return 2

    return asyncio.run(run_pipeline(args))


async def run_pipeline(args: argparse.Namespace) -> int:
    spread, level = parse_entropy(args.entropy)
    depth = CARD_DEPTH_BY_NAME[args.card_depth]

    # Resolve the donor deck: static bank OR topic-aware generation
    if args.bank:
        deck = cards_from_static_bank(load_bank(args.bank))
        deck_origin = f"static bank: {args.bank}"
    else:
        deck = await load_or_generate_deck(
            topic=args.topic,
            n=args.cards,
            depth=depth,
            model=args.model,
            force_regen=args.regen_deck,
            verbose=not args.quiet,
        )
        deck_origin = (
            f"topic-aware deck (n={len(deck)}, depth={depth.name})"
        )

    if args.show_deck:
        print(f"\n=== Donor deck ({deck_origin}) ===")
        for c in deck:
            print(c.render())
        print()

    # Mode mutual exclusion (parse_args also enforces this; belt and braces).
    if sum(1 for f in (args.einstein, args.lsd, args.futures) if f) > 1:
        print(
            "error: --einstein, --lsd, --futures are mutually exclusive.",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.seed)
    ideas: list[str] = []
    mechanisms_used: list[str | None] = []
    cards_per_idea: list[list[Card]] = []

    if args.futures:
        total = len(FUTURES_HORIZONS)
        for i, h in enumerate(FUTURES_HORIZONS):
            cards = sample_cards(
                deck=deck, n=args.n_concepts, spread=spread, rng=rng,
            )
            header = (
                f"\n=== Futures pass {i + 1}/{total}: {h['label']} | "
                f"entropy={level.name} (spread={spread:.2f}) | "
                f"deck={len(deck)}@{depth.name} | "
                f"model={args.model} ===\n"
            )
            print(header)
            print(f"Topic: {args.topic.strip()}")
            print(f"Horizon: {h['horizon_name']}")
            print(f"Framing: {h['framing']}")
            print("Sampled cards:")
            for c in cards:
                print(c.render())
            print()

            prompt = build_futures_prompt(args.topic, cards, h["key"])
            idea = await synthesize(
                prompt=prompt,
                model=args.model,
                stream_to_stdout=not args.quiet,
            )
            ideas.append(idea)
            mechanisms_used.append(h["label"])
            cards_per_idea.append(cards)

            if args.quiet:
                print(idea)
    elif args.lsd:
        # n_ideas passes, each dissolving one (model's-choice) load-bearing prior.
        for i in range(args.n_ideas):
            cards = sample_cards(
                deck=deck, n=args.n_concepts, spread=spread, rng=rng,
            )
            header = (
                f"\n=== LSD pass {i + 1}/{args.n_ideas}: Prior Dissolution | "
                f"entropy={level.name} (spread={spread:.2f}) | "
                f"deck={len(deck)}@{depth.name} | "
                f"model={args.model} ===\n"
            )
            print(header)
            print(f"Topic: {args.topic.strip()}")
            print(
                "Mechanism: Prior Dissolution — loosen the field's "
                "interpretive prior and re-perceive."
            )
            print("Sampled cards:")
            for c in cards:
                print(c.render())
            print()

            prompt = build_lsd_prompt(args.topic, cards)
            idea = await synthesize(
                prompt=prompt,
                model=args.model,
                stream_to_stdout=not args.quiet,
            )
            ideas.append(idea)
            mechanisms_used.append(LSD_LABEL)
            cards_per_idea.append(cards)

            if args.quiet:
                print(idea)
    elif args.einstein:
        # One pass per mechanism, all sharing the entropy-controlled card draw.
        mech_keys = list(EINSTEIN_MECHANISMS.keys())
        total = len(mech_keys)
        for i, key in enumerate(mech_keys):
            cards = sample_cards(
                deck=deck, n=args.n_concepts, spread=spread, rng=rng,
            )
            mech = EINSTEIN_MECHANISMS[key]
            header = (
                f"\n=== Einstein pass {i + 1}/{total}: {mech['label']} | "
                f"entropy={level.name} (spread={spread:.2f}) | "
                f"deck={len(deck)}@{depth.name} | "
                f"model={args.model} ===\n"
            )
            print(header)
            print(f"Topic: {args.topic.strip()}")
            print(f"Mechanism: {mech['label']} — {mech['blurb']}")
            print("Sampled cards:")
            for c in cards:
                print(c.render())
            print()

            prompt = build_einstein_prompt(args.topic, cards, key)
            idea = await synthesize(
                prompt=prompt,
                model=args.model,
                stream_to_stdout=not args.quiet,
            )
            ideas.append(idea)
            mechanisms_used.append(mech["label"])
            cards_per_idea.append(cards)

            if args.quiet:
                print(idea)
    else:
        for i in range(args.n_ideas):
            cards = sample_cards(
                deck=deck,
                n=args.n_concepts,
                spread=spread,
                rng=rng,
            )

            header = (
                f"\n=== Idea {i + 1}/{args.n_ideas} | "
                f"entropy={level.name} (spread={spread:.2f}) | "
                f"deck={len(deck)}@{depth.name} | "
                f"model={args.model} ===\n"
            )
            print(header)
            print(f"Topic: {args.topic.strip()}")
            print("Sampled cards:")
            for c in cards:
                print(c.render())
            print()

            prompt = build_prompt(args.topic, cards, level)
            idea = await synthesize(
                prompt=prompt,
                model=args.model,
                stream_to_stdout=not args.quiet,
            )
            ideas.append(idea)
            mechanisms_used.append(None)
            cards_per_idea.append(cards)

            if args.quiet:
                print(idea)

    # --- Critic + refine ---------------------------------------------------
    if args.refine and ideas:
        print(f"\n=== Critic pass ({len(ideas)} ideas) ===")
        scored: list[dict] = []
        for i, idea in enumerate(ideas):
            score = await critic_score(args.topic, idea, args.model)
            score["i"] = i
            scored.append(score)
            tag = (
                f" [{mechanisms_used[i]}]"
                if i < len(mechanisms_used) and mechanisms_used[i]
                else ""
            )
            print(
                f"  idea {i + 1}{tag}: "
                f"feasibility={score['feasibility']:>3}  "
                f"unexpectedness={score['unexpectedness']:>3}  "
                f"topic_fit={score['topic_fit']:>3}  "
                f"total={total_score(score):>3}  — {score['notes']}"
            )

        winner = max(scored, key=total_score)
        winner_mech = (
            f" [{mechanisms_used[winner['i']]}]"
            if winner["i"] < len(mechanisms_used) and mechanisms_used[winner["i"]]
            else ""
        )
        print(
            f"\n=== Winner: idea {winner['i'] + 1}{winner_mech} "
            f"(total {total_score(winner)}) — refining at low entropy ===\n"
        )
        refined = await refine_idea(
            topic=args.topic,
            idea=ideas[winner["i"]],
            notes=winner.get("notes", ""),
            model=args.model,
        )
        print(refined)

        # --- Deck evolution (opt-in via --evolve-deck) --------------------
        if args.evolve_deck and not args.bank:
            print(
                f"\n=== Deck evolution: sharpening "
                f"{len(cards_per_idea[winner['i']])} card(s) that "
                f"contributed to the winner ===\n"
            )
            try:
                evolved = await evolve_cards(
                    topic=args.topic,
                    idea=refined,
                    cards=cards_per_idea[winner["i"]],
                    depth=depth,
                    model=args.model,
                )
            except Exception as e:
                print(f"  evolution failed: {e}", file=sys.stderr)
                evolved = []
            if evolved:
                deck = merge_evolved_into_deck(deck, evolved)
                path = save_deck_to_cache(
                    args.topic, args.cards, depth, args.model, deck,
                )
                print(f"  wrote {len(evolved)} updated card(s) to {path}\n")
                for original, new in zip(cards_per_idea[winner["i"]], evolved):
                    print(f"  ◆ {original.name} ({original.domain})")
                    for field in depth.fields:
                        if field in ("name", "domain"):
                            continue
                        old_val = getattr(original, field, None) or ""
                        new_val = getattr(new, field, None) or ""
                        if old_val != new_val:
                            print(f"    {field}:")
                            print(f"      was: {old_val[:140]}")
                            print(f"      now: {new_val[:140]}")
                    print()
            else:
                print("  (no evolved cards parsed; deck unchanged)\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
