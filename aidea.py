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
    args = p.parse_args(argv)
    if args.n_concepts < 2:
        p.error("--n-concepts must be at least 2 (blending needs >= 2 seeds)")
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

    rng = random.Random(args.seed)

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

        if args.quiet:
            print(idea)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
