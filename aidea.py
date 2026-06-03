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
import os
import asyncio
import contextvars
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ClaudeAgentOptions,
    ClaudeSDKError,
    ProcessError,
    RateLimitEvent,
    ResultMessage,
    TextBlock,
    query,
)


class EmptyResponseError(RuntimeError):
    """The SDK delivered a ResultMessage with no usable assistant text.
    Treated as transient — retry layer will back off and try again."""


# Errors we retry on. CLINotFoundError (claude CLI missing) is non-transient
# and surfaces immediately; CLIJSONDecodeError, ProcessError, and connection
# errors are typically transient (process crash, network blip, brief rate
# pressure). EmptyResponseError covers the case where the SDK silently
# returned no text (e.g. a refusal or a server hiccup).
_RETRYABLE: tuple[type[BaseException], ...] = (
    CLIConnectionError,
    CLIJSONDecodeError,
    ProcessError,
    asyncio.TimeoutError,
    EmptyResponseError,
    ConnectionError,
    OSError,
)

# The agent SDK raises a bare ``Exception(message)`` from several internal
# failure paths (subprocess crash, malformed ResultMessage, "Claude Code
# returned an error result: <subtype>", "Command failed with exit code N").
# These slip past both the _RETRYABLE tuple AND the ClaudeSDKError except
# branch in _run_query, so the user gets the raw SDK string on the very
# first attempt with no retries. Pattern-match the message so we can route
# them through the retry layer like any other transient failure.
_SDK_TRANSIENT_MESSAGE_PATTERNS = (
    "Claude Code returned an error result",
    "Command failed with exit code",
    "Unknown error",
)


def _is_transient_sdk_exception(exc: BaseException) -> bool:
    """True iff a bare ``Exception`` from the SDK matches a known transient
    failure mode — see _SDK_TRANSIENT_MESSAGE_PATTERNS. Conservative: only
    plain Exception subclasses are matched, not arbitrary RuntimeError /
    ValueError / etc., so genuine programmer mistakes still propagate."""
    if type(exc) is not Exception:
        return False
    msg = str(exc)
    return any(p in msg for p in _SDK_TRANSIENT_MESSAGE_PATTERNS)

from transcripts import log_event as transcript_log
from usage import (
    build_call_record,
    current_run_id,
    record_call,
)

DECK_CACHE_DIR = Path(__file__).parent / "decks"


# ---------------------------------------------------------------------------
# Language rule — appended to every system prompt.
#
# Users have reported English replies when their topic was in another
# language. The English prompt templates were inadvertently anchoring the
# output language. Make the rule explicit and unconditional: detect the
# topic's primary language, respond entirely in it — section labels and
# all. JSON keys stay English (we parse them); JSON values follow the
# topic language.
# ---------------------------------------------------------------------------


LANGUAGE_RULE = (
    "\n\nLANGUAGE: Detect the primary language of the user's topic and "
    "respond ENTIRELY in that language — including section labels like "
    "'Title:', 'One-line pitch:', 'Mechanism:', etc. (translate them). If "
    "the topic is mixed-language, use whichever language clearly dominates; "
    "fall back to English only if you genuinely cannot identify a dominant "
    "language. Technical terms native to English (API / library / product "
    "names, established jargon) stay in English. The ONE exception: when "
    "the prompt asks you to emit JSON, the JSON KEY NAMES stay English "
    "verbatim (they are parsed by code); the VALUES inside the JSON follow "
    "the topic language."
)


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
    # 0-100 structural-alignment score to the current topic. Populated by
    # score_alignment() when AIDEA_STRUCTURE_BIAS is on; left None otherwise.
    # When present, sample_cards uses it to weight within-domain selection
    # so high-aligned cards rise, mimicking the brain's structure-mapping
    # bias (Gentner) on top of the entropy radius.
    alignment_score: int | None = None
    # Anti-fatigue Layer 1: 1.0 by default, set to <1.0 in sample_cards when
    # the card's domain has appeared in the current user's last N runs.
    # Multiplied into the sampling weight to soft-downweight repeats.
    novelty_penalty: float = 1.0
    # Anti-fatigue Layer 2: one of pragmatic / poetic / contrarian /
    # nostalgic / forensic / playful (see MOOD_TAGS). Populated by the
    # alignment+mood batch scorer when AIDEA_STRUCTURE_BIAS is on. Used
    # in sample_cards to rotate AWAY from moods the user has seen
    # recently, fighting per-user theme fatigue.
    mood_tag: str | None = None

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


# ---------------------------------------------------------------------------
# Entropy-biased theme generator: an LLM-picked list of donor domains tuned
# to the requested entropy, replacing the previously hardcoded landscape.
# ---------------------------------------------------------------------------


THEMES_SYSTEM = (
    "You pick donor domains for an applied-ideas tool. Output ONE domain "
    "name per line, no preamble, no commentary, no numbering, no surrounding "
    "code fence. Domain names should be specific noun-phrases that suggest "
    "where structural mechanisms live (e.g. 'harbor pilotage', 'monastic "
    "rules', 'mycorrhizal symbiosis'), not generic categories ('transport', "
    "'religion', 'biology')."
    + LANGUAGE_RULE
)


def _theme_guidance(theme_entropy: float) -> str:
    """Map entropy to instructions for how distant the donor domains should be."""
    if theme_entropy < 0.3:
        return (
            "STAY CLOSE. Pick 4-5 specific sub-disciplines INSIDE the user's "
            "home field and 2-3 of its nearest neighbors. Surprise should "
            "come from depth (specificity), not distance."
        )
    if theme_entropy < 0.6:
        return (
            "MIX. Pick at least 4 recognizable fields plus at least 2 "
            "lateral domains the user would not spontaneously reach for. "
            "Balance familiarity with novelty."
        )
    if theme_entropy < 0.9:
        return (
            "GO DISTANT. At least 6 domains should sit far from the user's "
            "home field. Include 2 obscure specialist domains the user has "
            "probably never named (e.g. 'lacquerware restoration', "
            "'reservoir sedimentation', 'medieval bestiaries')."
        )
    return (
        "GO MAXIMALLY DISTANT. Span hard science, manual trades, religious "
        "practice, ancient history, niche sports, obscure crafts, animal "
        "behavior, military doctrine, art movements, dead industries. If a "
        "domain name does not make the user blink, you picked the wrong one. "
        "Reach for the unfamiliar."
    )


THEMES_PROMPT_TEMPLATE = """\
Pick {n_themes} donor domains for cross-pollination with this user
problem / question / project:

  {topic}

Entropy of theme selection: {entropy_pct:.0f}% — see guidance below.

Guidance:
{guidance}

Output exactly {n_themes} domain names, one per line, no numbering, no
bullets, no commentary. The names must be specific enough to suggest
mechanisms — not bare category labels.

Begin:"""


async def generate_themes(
    topic: str,
    n_themes: int,
    theme_entropy: float,
    model: str,
) -> list[str]:
    """Ask the LLM to propose N donor domains, biased by entropy. Returns a
    de-duplicated, length-capped list of domain names."""
    prompt = THEMES_PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        n_themes=n_themes,
        entropy_pct=theme_entropy * 100,
        guidance=_theme_guidance(theme_entropy),
    )
    text = await _query_text(prompt, THEMES_SYSTEM, model, kind="themes")
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        # Strip common bullet / numbering artifacts the model sometimes adds.
        line = re.sub(r"^[\-\*••]\s+", "", line)
        line = re.sub(r"^\d+[\.\)]\s+", "", line)
        line = line.strip("`*_ \"'")
        if not line or len(line) > 80:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out[:n_themes]


DECK_GEN_SYSTEM = (
    "You generate donor concept decks for an idea-synthesis tool. Your output "
    "is JSON Lines (one JSON object per line, no commentary, no surrounding "
    "fences). Each object is one card."
    + LANGUAGE_RULE
)


def _deck_gen_prompt(
    topic: str,
    n: int,
    depth: CardDepth,
    themes: list[str] | None = None,
    warm_start: str = "",
) -> str:
    field_list = ", ".join(f'"{f}"' for f in depth.fields)
    if themes:
        domains_clause = (
            "Pull concepts from the following donor domains (use all or a "
            "subset — if a domain does not yield a transferable mechanism, "
            "skip it rather than force):\n"
            + "\n".join(f"  - {t}" for t in themes)
        )
        spread_clause = (
            "- Spread the deck ACROSS the listed donor domains. Aim for at "
            "least half of them to be represented."
        )
    else:
        # Fallback if themes weren't generated for whatever reason — keep
        # the legacy guidance so the pipeline still works degraded.
        domains_clause = (
            "Pull from a wide range of source domains (biology, physics, "
            "computing, economics, art, urbanism, psychology, linguistics, "
            "history, music, warfare, religion, law, sports, etc.)."
        )
        spread_clause = "- Span at least 8 distinct source domains."
    warm_block = ""
    if warm_start.strip():
        warm_block = f"""\

Warm-start examples — donor concepts that have produced ideas before in
THIS USER'S problem space (retrieved from their own past sessions; never
from another user's). Use them as a SIGNAL for which kinds of mechanism
transfer well here. Generate FRESH cards with the same generalizing
spirit. You may include a card with the same name as a warm-start card
ONLY if you sharpen its mechanism / why / transfer fields with new
insight; otherwise pick a different angle. Do not include all of them —
treat the list as informative, not mandatory.

{warm_start}
"""
    return f"""\
Generate {n} donor concepts that could cross-pollinate with this user topic:

  {topic.strip()}

{domains_clause}
{warm_block}
Optimize for STRUCTURAL diversity. Aim for concepts whose mechanisms are
domain-independent enough to be transferable, not concepts that already
live near the user's topic.

Hard requirements:
{spread_clause}
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
    theme_entropy: float = 0.5,
    themes: list[str] | None = None,
) -> list[Card]:
    """Generate a topic-aware donor deck.

    Two LLM steps unless ``themes`` is provided:
      1) ``generate_themes`` picks N donor domains biased by
         ``theme_entropy``  — 0 = inside the user's home field,
         1 = wildly distant specialist domains.
      2) ``_deck_gen_prompt`` produces the deck spread across those domains.

    Usage records: ``kind='themes'`` then ``kind='deck'``.
    """
    if themes is None:
        n_themes = max(6, min(20, n // 3))
        if verbose:
            print(
                f"[deck] picking {n_themes} themes (entropy={theme_entropy:.2f})...",
                flush=True,
            )
        themes = await generate_themes(topic, n_themes, theme_entropy, model)
        if verbose and themes:
            preview = ", ".join(themes[:8]) + (
                "..." if len(themes) > 8 else ""
            )
            print(f"[deck] themes: {preview}", flush=True)
        try:
            from transcripts import log_event as _tlog
            _tlog(
                "themes",
                themes=themes,
                theme_entropy=theme_entropy,
                topic=topic,
            )
        except Exception:
            pass

    # ---- RAG: retrieve past donor cards proven useful in this source's
    # problem space, scoped per-source for privacy. The synthesizer never
    # sees retrieval — only the deck-gen prompt does — so the entropy
    # mechanism stays clean.
    #
    # AIDEA_RAG_DISABLE=1 short-circuits retrieval entirely (pure-entropy
    # mode). The corpus still accumulates from this run via ingest_deck,
    # so toggling the flag back on later resumes the flywheel without
    # data loss. Used to A/B test whether the quality-boosted warm-start
    # is actually helping or just amplifying past winners into unrelated
    # topics.
    warm_block = ""
    if os.environ.get("AIDEA_RAG_DISABLE", "").strip() not in ("", "0", "false", "False"):
        if verbose:
            print("[deck] AIDEA_RAG_DISABLE set — pure-entropy mode (no warm start)", flush=True)
    else:
        try:
            from rag import retrieve_similar, render_warm_start
            from transcripts import current_source
            src = current_source() or "unknown"
            retrieved = retrieve_similar(source=src, topic=topic, k=8)
            warm_block = render_warm_start(retrieved)
            if verbose and retrieved:
                names = ", ".join(
                    (r.get("card") or {}).get("name", "?") for r in retrieved[:5]
                )
                print(
                    f"[deck] warm-start from corpus: {len(retrieved)} card(s) — "
                    f"{names}{'...' if len(retrieved) > 5 else ''}",
                    flush=True,
                )
        except Exception:
            warm_block = ""

    prompt = _deck_gen_prompt(
        topic, n, depth, themes=themes or None, warm_start=warm_block,
    )

    # Anti-fatigue Layer 3 wildcard injection: when load_or_generate_deck
    # has detected this user's novelty curve is flattening, append an
    # explicit "wander further" instruction so the deck-gen model knows
    # to pull from less-obvious donor neighbourhoods than its default for
    # this topic. ContextVar is set per pipeline run.
    if _saturated_run.get():
        prompt += (
            "\n\nWILDCARD WANDER: this user has been seeing similar donor "
            "themes recently. For this deck only, deliberately pull from "
            "donor domains far from the obvious neighbourhood of the topic — "
            "domains the model would NOT normally pick on the first pass for "
            "a query like this. Bias toward unexpected fields, sub-cultures, "
            "historical practices, niche industries. Still feasibility-bound "
            "in the synthesizer downstream, so the cards can be wild on "
            "domain origin but should each still have a real mechanism worth "
            "describing.\n"
        )

    if verbose:
        print(f"[deck] generating {n} cards at depth={depth.name}...", flush=True)
    text = await _run_query(prompt, DECK_GEN_SYSTEM, model, kind="deck")
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
    theme_entropy: float = 0.5,
    themes: list[str] | None = None,
) -> list[Card]:
    # Anti-fatigue Layer 3: per-user saturation alarm. If this user's
    # last-week donor-domains overlap their prior-week donors above the
    # threshold, force a fresh deck-gen + wildcard wander instruction.
    # Bypasses cache so the same topic for the same user actually gets
    # a different deck on consecutive runs once saturation is detected.
    saturated = False
    try:
        from retention import donor_repeat_rate, SATURATION_THRESHOLD
        from transcripts import current_source
        src = current_source() or ""
        if src:
            rate = donor_repeat_rate(src, days=7)
            if rate >= SATURATION_THRESHOLD:
                saturated = True
                force_regen = True  # invalidate cache for this run
                _saturated_run.set(True)
                if verbose:
                    print(
                        f"[novelty] saturation alarm: 7d repeat rate {rate:.2f} "
                        f"≥ {SATURATION_THRESHOLD:.2f} — forcing fresh deck + wider wander",
                        flush=True,
                    )
    except Exception:
        pass

    path = _deck_cache_path(topic, n, depth, model)
    if path.exists() and not force_regen:
        if verbose:
            print(f"[deck] using cached deck at {path}", flush=True)
        data = json.loads(path.read_text())
        cards = [Card(**c) for c in data]
    else:
        cards = await generate_deck(
            topic, n, depth, model, verbose,
            theme_entropy=theme_entropy, themes=themes,
        )
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

    # Optional brain-inspired structural-alignment scoring (Gentner
    # structure-mapping) + Layer-2 mood tagging. When AIDEA_STRUCTURE_BIAS
    # is on, one batch LLM call scores every card 0-100 on whether its
    # core mechanism actually maps onto the user's problem AND tags it
    # with a mood (pragmatic/poetic/contrarian/nostalgic/forensic/playful).
    # sample_cards uses both: alignment weights within-domain selection
    # toward structural fits; mood biases AWAY from moods the user has
    # seen recently, fighting per-user theme fatigue.
    if os.environ.get("AIDEA_STRUCTURE_BIAS", "").strip() not in ("", "0", "false", "False"):
        try:
            scores = await score_alignment(topic, cards, model)
            for c in cards:
                entry = scores.get(c.name) or {}
                c.alignment_score = entry.get("alignment")
                c.mood_tag = entry.get("mood")
            if verbose:
                vals = [e.get("alignment", 50) for e in scores.values() if isinstance(e, dict)]
                moods = [e.get("mood") for e in scores.values() if isinstance(e, dict) and e.get("mood")]
                if vals:
                    print(
                        f"[deck] alignment scored: mean {sum(vals)/len(vals):.0f} · "
                        f"range {min(vals)}–{max(vals)} · "
                        f"moods tagged: {len(moods)}/{len(scores)}",
                        flush=True,
                    )
        except Exception as e:
            if verbose:
                print(f"[deck] alignment scoring failed ({e}); falling back to uniform sampling", flush=True)

    # Anti-fatigue Layer 1: per-user donor cooldown. Annotate each card
    # with a novelty_penalty < 1.0 if its domain appeared in the user's
    # last 3 runs. sample_cards multiplies the sampling weight by this
    # to soft-downweight repeat domains. Cheap — one transcripts.jsonl
    # scan, no LLM call.
    try:
        from retention import recent_card_domains
        from transcripts import current_source
        src = current_source() or ""
        if src:
            seen = recent_card_domains(src, n_runs=3)
            n_cool = 0
            for c in cards:
                if c.domain in seen:
                    c.novelty_penalty = 0.4
                    n_cool += 1
            if verbose and n_cool:
                print(
                    f"[novelty] cooldown applied to {n_cool}/{len(cards)} cards "
                    f"(domain seen in user's last 3 runs)",
                    flush=True,
                )
    except Exception:
        pass

    # Ingest into the per-source corpus on every pipeline run, including
    # cache hits — the deck cache is keyed by topic/depth/model and is
    # source-agnostic, so a new source reusing a cached deck still needs
    # the (source, run_id) link recorded for tenant-isolated retrieval.
    try:
        from rag import ingest_deck
        from transcripts import current_source
        from usage import current_run_id
        n_ingested = ingest_deck(
            source=current_source() or "unknown",
            run_id=current_run_id() or "unknown",
            topic=topic,
            cards=cards,
            mode="default",
        )
        if verbose and n_ingested:
            print(
                f"[deck] ingested {n_ingested} card(s) into RAG corpus",
                flush=True,
            )
    except Exception:
        pass
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
    """Sample n cards with controlled cross-domain spread.

    Within each chosen domain, if the deck's cards carry
    ``alignment_score`` (populated by ``score_alignment`` when
    AIDEA_STRUCTURE_BIAS is on), selection is weighted by that score
    blended against entropy: at low ``spread`` alignment dominates
    (tight transplants); at high ``spread`` the weights flatten toward
    uniform so AIdea's signature surprise isn't smothered. Domain-
    hopping at rate ``spread`` is unchanged either way.

    Anti-fatigue stack (applied to the within-domain weight, on top of
    alignment scoring):
      Layer 1: ``novelty_penalty`` (set by load_or_generate_deck when the
        card's domain was in the user's last 3 runs) multiplies the
        weight by 0.4, soft-suppressing repeat domains.
      Layer 2: cards whose ``mood_tag`` does NOT appear in the user's
        recent moods get a 1.5× boost so the engine rotates away from
        whatever flavour the user has been getting.
      Layer 3: if _saturated_run is set (saturation alarm fired) spread
        is floored at 0.65 (insane minimum) so the cross-domain hop rate
        rises — combined with the wildcard prompt injection in
        generate_deck, this is the explicit BeReal-Day-200 defense.
    """
    if not deck:
        raise ValueError("Deck is empty.")

    # Layer 3 spread floor — see docstring.
    if _saturated_run.get():
        spread = max(spread, 0.65)

    # Layer 2 mood rotation — load the user's recent moods once and
    # boost cards whose mood is NOT in the recent set. If we have no
    # source / no history / no mood tags yet, this is a no-op.
    recent_moods: set[str] = set()
    try:
        from retention import recent_mood_tags
        from transcripts import current_source
        src = current_source() or ""
        if src:
            counter = recent_mood_tags(src, n_runs=3)
            # Mood is "recently seen" if it appears at least twice in the
            # last 3 runs — single occurrences don't count, since most
            # decks have a mix of moods anyway and 1-hit doesn't signal
            # a pattern worth rotating away from.
            recent_moods = {m for m, k in counter.items() if k >= 2}
    except Exception:
        recent_moods = set()

    def _weight(c: Card) -> float:
        # Baseline (alignment-aware if scored, else uniform 1.0).
        if c.alignment_score is None:
            base = 1.0
        else:
            base = c.alignment_score * (1.0 - spread) + 50.0 * spread

        # Layer 1 cooldown.
        base *= getattr(c, "novelty_penalty", 1.0) or 1.0

        # Layer 2 mood rotation — 1.5× for unseen-recently moods.
        if c.mood_tag and recent_moods and c.mood_tag not in recent_moods:
            base *= 1.5

        return max(0.01, base)

    def _pick(pool: list[Card]) -> Card:
        weights = [_weight(c) for c in pool]
        # Guard against degenerate all-zero (shouldn't happen — _weight
        # floors at 0.01) by falling back to plain choice.
        if not any(w > 0 for w in weights):
            return rng.choice(pool)
        return rng.choices(pool, weights=weights, k=1)[0]

    # Group by domain
    by_domain: dict[str, list[Card]] = {}
    for c in deck:
        by_domain.setdefault(c.domain, []).append(c)
    domains = list(by_domain.keys())

    start_domain = rng.choice(domains)
    chosen: list[Card] = [_pick(by_domain[start_domain])]
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
            pick = _pick(remaining)
            chosen.append(pick)
            used_ids.add(id(pick))
            continue
        pick = _pick(pool)
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
  - No gimmicks. Reject superficial mash-ups (two existing products glued
    together), "X-but-for-Y" clones, and clever-sounding combinations that
    don't change HOW the problem is solved. The borrowed mechanism must do
    real work, not just supply a novel-sounding label.

Respond in exactly this format, no preamble, no closing remarks:

Title: <3-7 memorable words>
One-line pitch: <single sentence connecting the idea to the user's problem>
How it addresses the request: <2-3 sentences — be concrete about which
  aspect of the user's problem this targets>
Mechanism: <2-4 sentences — name which donor concept(s) supply the
  structure and how the borrowing actually works>
Where it has worked: <1-2 sentences citing a concrete prior case — name
  the field, organization, or product where this mechanism (the donor's,
  not this exact transplant) has been observed working. Pull from the
  donor's "Prior application" line above when available; otherwise cite
  a real published case from the donor's source field. This is the
  "proven" half of the unique+proven pairing — be specific, not generic>
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


LSD_LABEL = "LSD"
LSD_VALIDATION_LABEL = "LSD (sober validation)"

LSD_PROMPT_TEMPLATE = """\
The user is working on this problem / question / project:

  {topic}

This synthesis uses LSD MODE — the REBUS framework (RElaxed Beliefs
Under pSychedelics, Carhart-Harris & Friston). Predictive-processing
neuroscience treats perception as a controlled hallucination; under
LSD, the hierarchy of priors FLATTENS. High-precision top-level
beliefs lose their grip on incoming sensory data. The brain stops
filtering "noise" as noise. Two mechanical effects drive idea
generation:

  (A) INCREASED GLOBAL CONNECTIVITY. Modules that normally don't
      exchange signals start handshaking across boundaries —
      "electrician neurons meet mushroom neurons." This is
      combinatorial play at the wiring level.

  (B) ESCAPE FROM LOCAL MINIMA. Daily expertise and field consensus
      are deep grooves the brain optimizes inside. LSD raises brain
      entropy ("shaking the snow globe") — supplying the uphill
      energy to jump out of one basin of attraction and land in a
      new conceptual valley.

THIS PASS IS THE ANARCHIC PHASE. Error-detection is OFF. Generate
freely; a separate sober-validation pass will run afterward to check
what survives. Do not pre-censor.

Donor concepts (sensory flooding — let MANY collide at once; do NOT
filter for which ones "fit"):
{seeds}

Process:
1. Name ONE local minimum the user's field is sitting in — the
   "good-enough" optimum everyone has been improving inside, and the
   metric the field uses to score moves there.
2. Force a connection between AT LEAST TWO donor concepts from
   maximally distant home domains. The handshake should feel weird
   at first. State the connection plainly.
3. Loosen at least one normally load-bearing prior — treat it as
   noise (just as LSD does when the hierarchy flattens). Name which
   prior you are treating as optional.
4. Propose an "uphill move" — an idea that gets WORSE on the field's
   current metric in the short term but, if it works, unlocks a
   strictly better minimum. The bigger the short-term loss, the
   sharper the test.

Hard requirements (these alone stay on in the anarchic phase):
  - The idea must still ADDRESS the user's stated problem (not a
    tangent — flattened priors does not mean wandering off-topic).
  - It may sound implausible at first read. That is expected.
  - The sober-validation pass that follows will check whether the
    rest of it survives daylight.

Respond in EXACTLY this format, no preamble, no closing remarks:

Title: <3-7 memorable words>
Mechanism: LSD (anarchic generation)
Local minimum being escaped: <the "good-enough" optimum the field's stuck in, plus the metric used to score moves inside it>
Cross-module connection: <the two distant donor concepts, what handshake they make, why neither alone would have suggested this>
Loosened prior: <the load-bearing assumption you are treating as noise>
The uphill move: <2-4 sentences describing the idea — including which direction the field's current metric moves on the short-term scoreboard>
Why this is hard to see normally: <1-2 sentences — what filter usually suppresses it>
"""


LSD_VALIDATION_PROMPT_TEMPLATE = """\
You are the sober inference engine running a 24-hours-later validation
pass on an idea generated under LSD MODE (anarchic, error-detection
offline). High-precision priors are back online. Your job is the
OPPOSITE of generation: filter what survives, name what was
hallucination, and produce the buildable v0.1.

User topic:
  {topic}

Anarchic idea (generated with relaxed beliefs):

{anarchic_idea}

Process:
1. Identify the structural insight the anarchic mind got right —
   something a sober planner would have missed because it sits below
   the threshold of "obviously feasible." Be honest: name it even if
   the delivery is wild.
2. Identify what was hallucination — claims that don't hold up to
   physics, economics, regulation, user behavior, or basic
   arithmetic. Name them specifically, not vaguely.
3. Propose the corrected v0.1 — the version that RETAINS the
   structural insight from (1) but is shippable today by a small
   team. If the insight cannot be retained without the hallucination,
   say so plainly.

Respond in EXACTLY this format, no preamble:

Mechanism: LSD (sober validation)
What survives the morning: <1-2 sentences naming the structural insight worth keeping>
What was hallucination: <1-2 sentences naming the parts that don't hold and WHY>
The corrected v0.1: <2-4 sentences — the buildable version>
First step the user could take this week: <one concrete action> OR the literal phrase "(none — the insight does not survive sober checking)"
Risks / what could still break: <1-2 sentences>
"""


def build_lsd_prompt(topic: str, cards: list[Card]) -> str:
    seeds = "\n".join(card.render() for card in cards)
    return LSD_PROMPT_TEMPLATE.format(topic=topic.strip(), seeds=seeds)


async def lsd_validate(
    topic: str,
    anarchic_idea: str,
    model: str,
) -> str:
    """Run the sober validation pass on an LSD-mode anarchic idea.

    Two-stage faithfulness to the REBUS framing: the anarchic phase
    above generates with error-detection offline; this phase brings
    high-precision priors back online and filters what survives. Logs
    as kind='lsd_validate' for transcript/usage tracking.
    """
    prompt = LSD_VALIDATION_PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        anarchic_idea=anarchic_idea.strip(),
    )
    return await _query_text(
        prompt, SYNTHESIZER_SYSTEM, model, kind="lsd_validate",
    )


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


# ---------------------------------------------------------------------------
# Dreaming Mode: prediction-error signal offline. Generative model runs
# free, no feasibility constraint, no "ship by Friday". Output is a "dream
# image" — vivid, possibly impossible, internally coherent — plus an
# explicit "what survives waking" interpretation pass.
# ---------------------------------------------------------------------------


DREAM_LABEL = "Dreaming"


DREAM_PROMPT_TEMPLATE = """\
The user is working on this problem / question / project:

  {topic}

This synthesis uses DREAMING MODE. Predictive-processing framing: in
sleep the prediction-error signal is offline, so the brain's generative
model runs without external correction. The job is to do the same — let
the generative model run free, ignore "what's buildable today", and
emit a dream image. Internal coherence is the only hard rule;
physical / economic / political plausibility is NOT.

Donor concepts (let them combine in unexpected ways; pick the ones that
grip you emotionally, not the ones a rational planner would pick):
{seeds}

Process:
1. Pick whichever donor concepts have the strongest pull. Don't decide
   rationally — pick by feel.
2. Build a vivid scene, system, or experience that fuses them. Present
   tense, sensory detail welcome. Internal dream-logic, not external
   physics.
3. The dream must still be ABOUT the user's stated topic — it cannot
   wander off-topic — but it may render the topic in literally
   impossible form (negative time, inverse causality, anthropomorphic
   abstractions, etc.).
4. Finish with a "What survives waking" line that names ONE fragment
   the user could actually salvage when they open their eyes.

Hard requirements (only these — every other waking constraint is OFF):
  - The dream must address the user's stated topic.
  - The "What survives waking" line must identify at least one fragment
    that could be acted on tomorrow, even if the rest of the dream is
    unrealisable.

Respond in EXACTLY this format, no preamble:

Title: <3-7 evocative words>
Mechanism: Dreaming
The dream image: <2-5 sentences. Present tense. Sensory. Implausible
  is fine; incoherent is not.>
Which donors fused: <name the 1-3 donor concepts whose collision produced this>
Why these collided: <1-2 sentences>
What survives waking: <1-2 sentences naming the salvageable fragment>
First step the user could take this week: <one concrete action, OR the literal phrase "(none — this is dream material only)">
"""


def build_dream_prompt(topic: str, cards: list[Card]) -> str:
    seeds = "\n".join(card.render() for card in cards)
    return DREAM_PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        seeds=seeds,
    )


# ---------------------------------------------------------------------------
# Lucid Dreaming Mode: hybrid. The prediction-error signal is still
# relaxed (the generative model runs free), but metacognition is online
# and the USER injects a directional prior the dream must bias toward.
# Includes a single "reality check" that swaps out the most internally
# inconsistent element. Output retains more salvage than pure dream.
# ---------------------------------------------------------------------------


LUCID_LABEL = "Lucid Dreaming"


LUCID_PROMPT_TEMPLATE = """\
The user is working on this problem / question / project:

  {topic}

The user has injected this directional prior — a high-confidence belief
the dream must bias toward (do not argue with it; honor it):

  {prior}

This synthesis uses LUCID DREAMING. Like dreams, the prediction-error
signal is relaxed and you may generate without feasibility constraints.
UNLIKE dreams, metacognition is online: the user has told you which
direction the hallucination should bias toward, and once the dream is
built you must perform exactly ONE reality check — identify the most
internally inconsistent element and either justify it or swap it for
something the dream can actually sustain.

Donor concepts (raw material; the injected prior should color which
ones grip you most):
{seeds}

Process:
1. Treat the injected prior as a fixed truth in the dream world.
2. Hallucinate a vivid idea that addresses the user's stated topic
   AND honors the prior.
3. Reality check: pick the single most absurd element of your dream.
   If it has internal coherence (dream-logic), keep it. If it
   contradicts the prior or contradicts itself, swap it for something
   the dream can sustain.
4. Identify what survives waking — more than dreaming-mode does,
   because the prior keeps the hallucination anchored.

Respond in EXACTLY this format, no preamble:

Title: <3-7 evocative words>
Mechanism: Lucid Dreaming
Injected prior: {prior}
The dream image (biased toward the prior): <2-5 sentences. Present tense.>
Reality check: <which element you tested, what you decided>
What survives waking: <2-3 sentences naming the executable fragment>
First step the user could take this week: <one concrete action>
Risks / what could still break on waking: <1-2 sentences>
"""


def build_lucid_prompt(topic: str, cards: list[Card], prior: str) -> str:
    seeds = "\n".join(card.render() for card in cards)
    return LUCID_PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        seeds=seeds,
        prior=prior.strip() or "(no prior provided — fall back to pure dream)",
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
    + LANGUAGE_RULE
)


# ---------------------------------------------------------------------------
# Audience level — selectable register for the final idea. Injected into the
# synthesizer / refiner system prompt (no extra LLM calls). Set once per run
# via set_level(); reads through a ContextVar so it propagates across the
# awaits within one pipeline task without threading a param through every
# synthesize() call. Default 'normal'.
# ---------------------------------------------------------------------------

LEVEL_NAMES = ("dummies", "normal", "expert")

LEVEL_DIRECTIVES: dict[str, str] = {
    "dummies": (
        "\n\nAUDIENCE LEVEL — DUMMIES: Write the ENTIRE response in the "
        "simplest everyday language. Short sentences. No jargon, no technical "
        "or field-specific terms. Explain it as if to a smart 12-year-old or a "
        "complete newcomer; if a specialised idea is unavoidable, immediately "
        "restate it in plain words."
    ),
    "normal": (
        "\n\nAUDIENCE LEVEL — NORMAL: Use clear, general language a "
        "non-specialist understands. Avoid undefined jargon and field-specific "
        "terms; if one is unavoidable, gloss it briefly in passing."
    ),
    "expert": (
        "\n\nAUDIENCE LEVEL — EXPERT: Maximum depth and precision. Use the "
        "field's exact terminology, name specific methods, tools, metrics and "
        "prior art, and include the technical detail a domain expert expects. "
        "Assume full domain fluency — do not over-explain the basics."
    ),
}

_level_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aidea_level", default="normal",
)


def set_level(level: str | None) -> str:
    """Set the audience level for the current pipeline run. Unknown/empty
    falls back to 'normal'. Returns the resolved level."""
    lvl = (level or "normal").strip().lower()
    if lvl not in LEVEL_DIRECTIVES:
        lvl = "normal"
    _level_var.set(lvl)
    return lvl


def _level_directive() -> str:
    return LEVEL_DIRECTIVES.get(_level_var.get(), "")


async def _run_query_once(
    prompt: str,
    system: str,
    model: str,
    *,
    kind: str,
    stream_to_stdout: bool = False,
) -> tuple[str, object | None]:
    """One uninstrumented attempt: drive an agent-SDK query, capture
    ResultMessage + RateLimitEvent, write a usage record, return
    (text, last_rate_limit_info). Raises ``EmptyResponseError`` if no
    assistant text came back so the retry wrapper can catch it."""
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system,
        max_turns=1,
        allowed_tools=[],
    )

    chunks: list[str] = []
    rate_limit_info = None
    result_message = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
                    if stream_to_stdout:
                        print(block.text, end="", flush=True)
        elif isinstance(message, ResultMessage):
            result_message = message
        elif isinstance(message, RateLimitEvent):
            rate_limit_info = message.rate_limit_info

    if stream_to_stdout:
        print()

    # Record usage. Tolerate the case where the SDK didn't emit a
    # ResultMessage (extremely rare; never block the pipeline on logging).
    if result_message is not None:
        try:
            record_call(build_call_record(
                run_id=current_run_id() or "unknown",
                kind=kind,
                result_message=result_message,
                rate_limit_info=rate_limit_info,
            ))
        except Exception:
            pass

    text = "".join(chunks)
    if not text.strip():
        # Treat as transient — usually means a rate-limit refusal or a
        # claude-cli hiccup. Retry layer will back off.
        err = getattr(result_message, "is_error", False)
        stop = getattr(result_message, "stop_reason", None)
        raise EmptyResponseError(
            f"empty assistant response (is_error={err}, stop_reason={stop})"
        )

    # Log the full round-trip for future analysis.
    transcript_log(
        "llm_call",
        call_kind=kind,
        model=model,
        system=system,
        prompt=prompt,
        response=text,
        usage=(getattr(result_message, "usage", None) or {}),
        total_cost_usd=float(getattr(result_message, "total_cost_usd", 0.0) or 0.0),
        duration_ms=int(getattr(result_message, "duration_ms", 0) or 0),
    )
    return text, rate_limit_info


def _backoff_seconds(attempt: int, rate_limit_info: object | None) -> float:
    """Pick a delay for the next retry attempt.

    Honors a RateLimitInfo.resets_at if present and in the future, capped
    at 5 minutes. Otherwise: 1s, 2s, 4s, 8s, 16s, ... up to 30s.
    """
    if rate_limit_info is not None:
        resets_at = getattr(rate_limit_info, "resets_at", None)
        status = getattr(rate_limit_info, "status", None)
        if (
            isinstance(resets_at, (int, float))
            and status not in (None, "allowed")
        ):
            wait = float(resets_at) - time.time()
            if wait > 0:
                return min(wait, 300.0)
    # Exponential backoff with a 30s cap.
    return float(min(2 ** (attempt - 1), 30))


async def _run_query(
    prompt: str,
    system: str,
    model: str,
    *,
    kind: str,
    stream_to_stdout: bool = False,
    max_attempts: int = 4,
) -> str:
    """Retry-wrapped agent-SDK call. All five LLM kinds (deck / synth /
    critic / refine / evolve) flow through here, so retry policy is
    enforced uniformly.

    Transient failures (connection drops, JSON-decode errors, process
    crashes, empty responses, rate-limit rejections) are retried with
    exponential backoff. Rate-limit rejections wait until the
    ``resets_at`` timestamp reported by the SDK when present. Hard
    errors (the CLI binary missing, programmer mistakes) propagate
    immediately."""
    last_rate_limit: object | None = None
    last_err: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            text, rl = await _run_query_once(
                prompt, system, model,
                kind=kind, stream_to_stdout=stream_to_stdout,
            )
            return text
        except CLINotFoundError:
            # claude CLI is missing on PATH — non-transient. Don't retry.
            raise
        except _RETRYABLE as e:
            last_err = e
            last_rate_limit = locals().get("rl", last_rate_limit)
            if attempt >= max_attempts:
                break
            wait = _backoff_seconds(attempt, last_rate_limit)
            transcript_log(
                "llm_error",
                call_kind=kind, attempt=attempt, max_attempts=max_attempts,
                error_type=type(e).__name__, error=str(e),
                sleep_s=wait, retryable=True,
            )
            print(
                f"[retry] {kind} attempt {attempt}/{max_attempts} failed "
                f"({type(e).__name__}: {e}); sleeping {wait:.1f}s",
                file=sys.stderr, flush=True,
            )
            await asyncio.sleep(wait)
        except ClaudeSDKError as e:
            # Any other SDK error — single retry then give up.
            last_err = e
            if attempt >= max_attempts:
                break
            wait = _backoff_seconds(attempt, last_rate_limit)
            transcript_log(
                "llm_error",
                call_kind=kind, attempt=attempt, max_attempts=max_attempts,
                error_type=type(e).__name__, error=str(e),
                sleep_s=wait, retryable=False,
            )
            print(
                f"[retry] {kind} attempt {attempt}/{max_attempts} failed "
                f"({type(e).__name__}: {e}); sleeping {wait:.1f}s",
                file=sys.stderr, flush=True,
            )
            await asyncio.sleep(wait)
        except Exception as e:
            # Bare Exception from the SDK — "Claude Code returned an error
            # result: <subtype>" / "Command failed with exit code N". Only
            # retry if the message matches a known transient pattern, so
            # we don't accidentally retry genuine bugs.
            if not _is_transient_sdk_exception(e):
                raise
            last_err = e
            if attempt >= max_attempts:
                break
            wait = _backoff_seconds(attempt, last_rate_limit)
            transcript_log(
                "llm_error",
                call_kind=kind, attempt=attempt, max_attempts=max_attempts,
                error_type=type(e).__name__, error=str(e),
                sleep_s=wait, retryable=True,
            )
            print(
                f"[retry] {kind} attempt {attempt}/{max_attempts} failed "
                f"(transient SDK: {e}); sleeping {wait:.1f}s",
                file=sys.stderr, flush=True,
            )
            await asyncio.sleep(wait)
    # Out of attempts.
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"_run_query exhausted attempts for kind={kind}")


async def _query_text(prompt: str, system: str, model: str, *, kind: str = "misc") -> str:
    return await _run_query(prompt, system, model, kind=kind)


async def synthesize(
    prompt: str,
    model: str,
    stream_to_stdout: bool,
) -> str:
    """Run the inference engine via the agent SDK (inherits CLI auth)."""
    return await _run_query(
        prompt, SYNTHESIZER_SYSTEM + _level_directive(), model,
        kind="synth", stream_to_stdout=stream_to_stdout,
    )


# ---------------------------------------------------------------------------
# Critic + Refinement: score ideas, then harden the winner at low entropy.
# ---------------------------------------------------------------------------


CRITIC_SYSTEM = (
    "You are a critic for an applied-ideas synthesizer. You score a single "
    "idea against a user problem on three axes. Output STRICT JSON only — "
    "one object, no surrounding code fences, no preamble, no commentary."
    + LANGUAGE_RULE
)


CRITIC_PROMPT_TEMPLATE = """\
The user is working on this problem:

  {topic}

Score this idea on four axes from 0 to 100. Be honest — your job is to
differentiate good ideas from filler, so use the full range. Reserve 90+
for genuinely strong, reserve 0-20 for clearly weak; cluster the middle.

  - feasibility: can a small team execute this with current technology in
    the user's stated field within months? 100 = obviously yes, 0 = science
    fiction.
  - unexpectedness: how non-obvious is the structural choice? 100 = "I did
    not see that coming"; 0 = "I would have thought of this in 30 seconds".
  - uniqueness: is this approach genuinely new TO THE USER'S STATED FIELD,
    or is it already a well-known/published/textbook solution there? Use
    your training-data knowledge of prior art in that specific field as the
    baseline. 100 = no evidence this exact transplant has been done in this
    field; 0 = this is the standard textbook approach you'd find on page 1
    of any introductory resource for that field. IMPORTANT: a transplant of
    a mechanism from another field counts as NEW in the user's field even
    if the donor mechanism is well-established in its origin field — what
    counts is novelty in the USER's domain, not the donor's. If the idea
    is a generic restatement of a common pattern in the user's field,
    score low even if the wording is fresh.
  - topic_fit: does this address the user's stated problem (vs a tangent)?
    100 = bullseye; 0 = answers a different question.

Idea to score:

{idea}

Respond as ONE JSON object only, with these keys exactly:
  {{"feasibility": <int 0-100>, "unexpectedness": <int 0-100>,
    "uniqueness": <int 0-100>, "topic_fit": <int 0-100>,
    "notes": "<one sentence — what's strongest, what's weakest>"}}
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


CRITIC_AXES = ("feasibility", "unexpectedness", "uniqueness", "topic_fit")


def _clip100(v: object) -> int:
    try:
        return max(0, min(100, int(float(v))))
    except (TypeError, ValueError):
        return 0


async def _critic_single(topic: str, idea: str, model: str) -> dict:
    """One-critic scoring — the original behaviour."""
    prompt = CRITIC_PROMPT_TEMPLATE.format(topic=topic.strip(), idea=idea.strip())
    raw = await _query_text(prompt, CRITIC_SYSTEM, model, kind="critic")
    obj = _try_parse_json_object(raw) or {}
    return {
        "feasibility": _clip100(obj.get("feasibility")),
        "unexpectedness": _clip100(obj.get("unexpectedness")),
        "uniqueness": _clip100(obj.get("uniqueness")),
        "topic_fit": _clip100(obj.get("topic_fit")),
        "notes": str(obj.get("notes", "")).strip()[:500],
    }


# ---------------------------------------------------------------------------
# Diverse critic panel (Hong–Page diversity-prediction theorem).
#
# A crowd of deliberately DIFFERENT, individually-imperfect critics predicts
# better than one strong critic, because their errors are uncorrelated and
# cancel under averaging. Each lens below scores the same four axes but from a
# distinct prior, so they err in different directions. The aggregate is the
# plain per-axis mean (the form the theorem is stated over). Enable with
# AIDEA_CRITIC_PANEL=1; critic_score() transparently dispatches here so every
# call site (bot / web / CLI) gets it with no other change.
# ---------------------------------------------------------------------------

CRITIC_PANEL_SYSTEM = (
    "You are {persona}, acting as one member of a panel of critics for an "
    "applied-ideas synthesizer. Score a single idea against a user problem on "
    "four axes (0-100) FROM YOUR PERSPECTIVE — lean into your bias; the panel "
    "averages everyone out, so your job is to represent your view honestly, "
    "not to be balanced. Output STRICT JSON only — one object, no code fences, "
    "no preamble." + LANGUAGE_RULE
)

CRITIC_LENSES = (
    {
        "key": "feasibility_hawk",
        "persona": "a skeptical operator who has watched a hundred clever "
                   "ideas die in execution",
        "directive": "Lens: FEASIBILITY HAWK. Weight execution risk, resource "
                     "reality, and time-to-first-value above all. Distrust "
                     "novelty for its own sake; reward what a small team could "
                     "actually ship.",
    },
    {
        "key": "novelty_seeker",
        "persona": "a restless idea-hunter allergic to the obvious — and "
                   "equally allergic to gimmicks dressed up as innovation",
        "directive": "Lens: NOVELTY SEEKER. Reward genuine STRUCTURAL surprise "
                     "— a mechanism transplanted from another domain that "
                     "changes HOW the problem is solved. Do NOT reward "
                     "superficial mash-ups (two existing products glued "
                     "together), 'X-but-for-Y' clones, or clever-sounding "
                     "gimmicks that add no real leverage — score those LOW on "
                     "uniqueness. Originality must do work, not just sound new.",
    },
    {
        "key": "domain_skeptic",
        "persona": "a seasoned expert in the user's stated field who knows the "
                   "prior art cold",
        "directive": "Lens: DOMAIN SKEPTIC. Catch ideas that only LOOK new to "
                     "outsiders. Penalize anything textbook in THIS field; "
                     "reward true gaps an insider would respect.",
    },
    {
        "key": "plain_user",
        "persona": "the non-expert who would actually have to adopt this",
        "directive": "Lens: PLAIN USER. Reward ideas you immediately understand "
                     "and want. Penalize jargon, vagueness, and 'so what'. "
                     "Clarity and pull matter as much as cleverness.",
    },
)


async def critic_panel(topic: str, idea: str, model: str) -> dict:
    """Score with the diverse panel and return the per-axis mean, in the same
    shape as _critic_single plus a ``panel`` breakdown and ``n_critics``.
    Falls back to a single critic if every lens fails to parse."""
    topic_s, idea_s = topic.strip(), idea.strip()
    base = CRITIC_PROMPT_TEMPLATE.format(topic=topic_s, idea=idea_s)

    async def run_lens(lens: dict) -> tuple[str, dict, str]:
        system = CRITIC_PANEL_SYSTEM.format(persona=lens["persona"])
        prompt = lens["directive"] + "\n\n" + base
        raw = await _query_text(prompt, system, model, kind="critic")
        obj = _try_parse_json_object(raw) or {}
        scores = {ax: _clip100(obj.get(ax)) for ax in CRITIC_AXES}
        return lens["key"], scores, str(obj.get("notes", "")).strip()[:160]

    results = await asyncio.gather(
        *(run_lens(lens) for lens in CRITIC_LENSES), return_exceptions=True,
    )
    valid = [r for r in results if not isinstance(r, Exception)]
    if not valid:
        return await _critic_single(topic, idea, model)

    panel = {key: scores for key, scores, _ in valid}
    agg = {
        ax: round(sum(s[ax] for s in panel.values()) / len(panel))
        for ax in CRITIC_AXES
    }
    notes = " | ".join(f"{key}: {note}" for key, _, note in valid)[:500]
    return {**agg, "notes": notes, "panel": panel, "n_critics": len(panel)}


def _critic_panel_enabled() -> bool:
    return os.environ.get("AIDEA_CRITIC_PANEL", "").strip() not in (
        "", "0", "false", "False",
    )


async def critic_score(
    topic: str, idea: str, model: str, *, force_panel: bool = False,
) -> dict:
    """Score one idea on the four axes. Uses the diverse critic panel when
    EITHER the caller passes force_panel=True (the refine path — winner
    selection is where the panel's accuracy actually changes the kept output)
    OR the global AIDEA_CRITIC_PANEL flag is set (panel on every generation —
    reserved as the future paid-tier feature). Otherwise the single critic.
    Return shape is identical, so total_score() and all call sites are
    unchanged."""
    if force_panel or _critic_panel_enabled():
        return await critic_panel(topic, idea, model)
    return await _critic_single(topic, idea, model)


# Maximum value of total_score. Used as the divisor for the RAG retrieval
# boost (rag.py) so the boost saturates at the right place when all axes
# are maxed. Bumped from 300 → 400 with the addition of the uniqueness
# axis. Old outcome records max at 300; their boost just won't fully
# saturate, which is acceptable for a gentle transition.
CRITIC_TOTAL_MAX = 400


def total_score(score: dict) -> int:
    return (
        score.get("feasibility", 0)
        + score.get("unexpectedness", 0)
        + score.get("uniqueness", 0)
        + score.get("topic_fit", 0)
    )


# ---------------------------------------------------------------------------
# Structural-alignment scorer (Gentner structure-mapping, brain-inspired).
# One cheap batch call rates every card in the deck on whether its core
# mechanism maps structurally onto the user's problem — separate from
# topic-fit (which is "does the IDEA address the topic") and feasibility.
# Used by sample_cards to weight within-domain selection when
# AIDEA_STRUCTURE_BIAS is enabled.
# ---------------------------------------------------------------------------


# Mood-tag vocabulary for anti-fatigue Layer 2. Closed set so rotation
# is meaningful (a free-form mood per card would produce an unbounded
# tag space and never rotate). Six tags cover the emotional axes that
# differentiate donor concepts the most — pragmatic/poetic, contrarian/
# nostalgic, forensic/playful — based on the donor-domain spread we've
# seen in real traffic (cooking, infra, art, philosophy, etc.).
MOOD_TAGS: tuple[str, ...] = (
    "pragmatic", "poetic", "contrarian",
    "nostalgic", "forensic", "playful",
)


ALIGNMENT_SYSTEM = (
    "You score donor concepts on TWO axes — structural alignment to a "
    "user's stated problem, and a closed-set mood tag. Output STRICT JSON "
    "only — one object per card, no commentary, no fences. The JSON keys "
    "(card names) stay verbatim in whatever language they were written; "
    "do not translate them."
)


ALIGNMENT_PROMPT_TEMPLATE = """\
User problem / project / question:

  {topic}

For each donor concept below, return TWO things:

(1) STRUCTURAL ALIGNMENT to the user's problem, 0-100 integer:
  - 100 = the donor's core relational schema (constraint shape, dynamic,
    transformation pattern) directly transplants onto this problem — a
    competent person seeing both would immediately see the analogy.
  - 50  = it could plausibly transplant with creative work; partial overlap.
  - 0   = no mappable structural correspondence; decoration only.

  This is NOT feasibility, NOT topic-fit, NOT novelty. Pure
  structure-mapping in Dedre Gentner's sense: do the relations between
  the donor's parts match the relations the user's problem needs?

  Use the full range. Most donors should land in 20-70; reserve 80+ for
  genuine structural matches, 0-20 for clear non-fits.

(2) MOOD TAG, exactly one of:
    pragmatic   — concrete, ops-flavoured, "ship it tomorrow"
    poetic      — metaphorical, image-led, evocative
    contrarian  — challenges a consensus assumption
    nostalgic   — uses old / pre-digital / craft mechanisms
    forensic    — investigative, root-cause, dissection-style
    playful     — game-like, mischievous, low-stakes
  Pick the dominant flavour of the donor; ignore the user's problem when
  picking the mood (mood is a property of the donor, not the fit).

Donor concepts:

{card_list}

Respond as ONE JSON object only, with one entry per card. Keys are the
EXACT card names verbatim (no edits, no translation). Values are objects
of shape {{"a": <int 0-100>, "m": "<mood>"}}. Example:
  {{"Card Name A": {{"a": 72, "m": "pragmatic"}},
    "Card Name B": {{"a": 35, "m": "poetic"}}}}
"""


async def score_alignment(
    topic: str,
    cards: list[Card],
    model: str,
) -> dict[str, dict]:
    """Return {card.name: {"alignment": 0-100, "mood": str|None}} for every
    card in the deck.

    Single batch LLM call — caller applies the scores by setting
    ``card.alignment_score`` and ``card.mood_tag`` before sampling.
    Combined into one call so the mood-tag scoring (Layer 2 of the
    anti-fatigue stack) costs zero extra tokens vs. alignment alone.
    """
    if not cards:
        return {}
    card_list = "\n".join(f"- {c.name} (from {c.domain})" for c in cards)
    prompt = ALIGNMENT_PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        card_list=card_list,
    )
    raw = await _query_text(prompt, ALIGNMENT_SYSTEM, model, kind="alignment")
    obj = _try_parse_json_object(raw) or {}

    out: dict[str, dict] = {}
    for c in cards:
        entry = obj.get(c.name)
        alignment: int = 50
        mood: str | None = None
        # The new schema is {"a": int, "m": str}; the legacy schema was
        # just an int. Accept both so an old prompt response or a malformed
        # response degrades gracefully.
        if isinstance(entry, dict):
            try:
                alignment = max(0, min(100, int(float(entry.get("a", 50)))))
            except (TypeError, ValueError):
                alignment = 50
            m = entry.get("m")
            if isinstance(m, str) and m.strip().lower() in MOOD_TAGS:
                mood = m.strip().lower()
        elif isinstance(entry, (int, float, str)):
            try:
                alignment = max(0, min(100, int(float(entry))))
            except (TypeError, ValueError):
                alignment = 50
        out[c.name] = {"alignment": alignment, "mood": mood}
    return out


# Anti-fatigue Layer 3 saturation flag. set by load_or_generate_deck when
# donor_repeat_rate crosses the threshold; read by generate_deck (for the
# wildcard prompt injection) and sample_cards (for the spread floor).
# ContextVar so it propagates across awaits within the current pipeline
# run without polluting any function signatures.
_saturated_run: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "aidea_saturated_run", default=False,
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
    return await _query_text(
        prompt, SYNTHESIZER_SYSTEM + _level_directive(), model, kind="refine",
    )


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
    + LANGUAGE_RULE
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
    raw = await _query_text(prompt, DECK_EVOLVE_SYSTEM, model, kind="evolve")
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
        default=3,
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
        "--level",
        choices=list(LEVEL_NAMES),
        default="normal",
        help="Audience register for the final idea: dummies (plainest), "
             "normal (default), or expert (max detail + field terms).",
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
        "--dream",
        action="store_true",
        help=(
            "Dreaming mode: prediction-error signal offline. Generates a "
            "dream image with no feasibility constraint, then names what "
            "survives waking. Mutually exclusive with --einstein / --lsd / "
            "--futures / --lucid."
        ),
    )
    p.add_argument(
        "--lucid",
        type=str,
        default=None,
        metavar="PRIOR",
        help=(
            "Lucid dreaming mode: relaxed prediction error + a directional "
            "prior the dream biases toward. Pass the prior as a sentence, "
            "e.g. --lucid 'the team must stay solo-founder-sized'. "
            "Mutually exclusive with the other mode flags."
        ),
    )
    p.add_argument(
        "--themes",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Override the auto-generated donor-domain themes. Comma-separated "
            "list, e.g. --themes 'harbor pilotage, mycology, monastic rules'."
        ),
    )
    p.add_argument(
        "--theme-entropy",
        type=float,
        default=None,
        metavar="X",
        help=(
            "Theme-selection entropy in [0,1]. 0 = stay in the user's home "
            "field; 1 = wildly distant specialist domains. Defaults to the "
            "value of --entropy."
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
    mode_flags = (
        args.einstein, args.lsd, args.futures, args.dream, bool(args.lucid),
    )
    if sum(1 for f in mode_flags if f) > 1:
        p.error(
            "--einstein / --lsd / --futures / --dream / --lucid are mutually "
            "exclusive; pick at most one."
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
    from usage import start_run
    from transcripts import set_source
    # AIDEA_SOURCE lets operators tag a CLI replay with the source of the
    # original request — e.g. when re-running a Telegram pipeline that was
    # killed mid-flight, set it to "telegram-<chat_id>" so the ingested
    # cards stay tenant-attributed correctly.
    source = os.environ.get("AIDEA_SOURCE", "").strip() or "cli"
    run_id = start_run(source)
    set_source(source)
    set_level(getattr(args, "level", "normal"))  # final-idea register
    if not args.quiet:
        print(f"[usage] run_id={run_id}", file=sys.stderr)
    spread, level = parse_entropy(args.entropy)
    depth = CARD_DEPTH_BY_NAME[args.card_depth]

    mode = (
        "einstein" if args.einstein else
        "lsd" if args.lsd else
        "futures" if args.futures else
        "default"
    )
    transcript_log(
        "request_started",
        topic=args.topic,
        mode=mode,
        entropy=level.name,
        spread=spread,
        cards=args.cards,
        card_depth=depth.name,
        n_concepts=args.n_concepts,
        n_ideas=args.n_ideas,
        seed=args.seed,
        model=args.model,
        refine=bool(args.refine),
        evolve_deck=bool(args.evolve_deck),
        bank=args.bank,
        regen_deck=bool(args.regen_deck),
    )

    # Resolve the donor deck: static bank OR topic-aware generation
    if args.bank:
        deck = cards_from_static_bank(load_bank(args.bank))
        deck_origin = f"static bank: {args.bank}"
    else:
        explicit_themes: list[str] | None = None
        if args.themes:
            explicit_themes = [
                t.strip() for t in args.themes.split(",") if t.strip()
            ] or None
        theme_entropy = (
            args.theme_entropy
            if args.theme_entropy is not None
            else spread
        )
        deck = await load_or_generate_deck(
            topic=args.topic,
            n=args.cards,
            depth=depth,
            model=args.model,
            force_regen=args.regen_deck,
            verbose=not args.quiet,
            theme_entropy=theme_entropy,
            themes=explicit_themes,
        )
        deck_origin = (
            f"topic-aware deck (n={len(deck)}, depth={depth.name}, "
            f"theme_entropy={theme_entropy:.2f})"
        )

    transcript_log(
        "deck",
        origin=deck_origin,
        size=len(deck),
        depth=depth.name,
        cards=[{
            k: v for k, v in c.__dict__.items() if v is not None
        } for c in deck],
    )

    if args.show_deck:
        print(f"\n=== Donor deck ({deck_origin}) ===")
        for c in deck:
            print(c.render())
        print()

    # Mode mutual exclusion (parse_args also enforces this; belt and braces).
    _mode_flags = (
        args.einstein, args.lsd, args.futures, args.dream, bool(args.lucid),
    )
    if sum(1 for f in _mode_flags if f) > 1:
        print(
            "error: --einstein / --lsd / --futures / --dream / --lucid are mutually exclusive.",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.seed)
    ideas: list[str] = []
    mechanisms_used: list[str | None] = []
    cards_per_idea: list[list[Card]] = []

    if args.dream:
        for i in range(args.n_ideas):
            cards = sample_cards(
                deck=deck, n=args.n_concepts, spread=spread, rng=rng,
            )
            header = (
                f"\n=== Dream pass {i + 1}/{args.n_ideas}: Dreaming | "
                f"entropy={level.name} (spread={spread:.2f}) | "
                f"deck={len(deck)}@{depth.name} | model={args.model} ===\n"
            )
            print(header)
            print(f"Topic: {args.topic.strip()}")
            print(
                "Mechanism: Dreaming — prediction-error offline; let the "
                "generative model run free."
            )
            print("Sampled cards:")
            for c in cards:
                print(c.render())
            print()
            prompt = build_dream_prompt(args.topic, cards)
            idea = await synthesize(
                prompt=prompt, model=args.model,
                stream_to_stdout=not args.quiet,
            )
            ideas.append(idea)
            mechanisms_used.append(DREAM_LABEL)
            cards_per_idea.append(cards)
            transcript_log(
                "idea", i=i, mechanism=DREAM_LABEL, text=idea,
                cards=[{k: v for k, v in c.__dict__.items() if v is not None} for c in cards],
            )
            if args.quiet:
                print(idea)
    elif args.lucid:
        for i in range(args.n_ideas):
            cards = sample_cards(
                deck=deck, n=args.n_concepts, spread=spread, rng=rng,
            )
            header = (
                f"\n=== Lucid pass {i + 1}/{args.n_ideas}: Lucid Dreaming | "
                f"entropy={level.name} (spread={spread:.2f}) | "
                f"deck={len(deck)}@{depth.name} | model={args.model} ===\n"
            )
            print(header)
            print(f"Topic: {args.topic.strip()}")
            print(f"Injected prior: {args.lucid}")
            print(
                "Mechanism: Lucid Dreaming — relaxed feasibility, "
                "user-injected prior, one reality check."
            )
            print("Sampled cards:")
            for c in cards:
                print(c.render())
            print()
            prompt = build_lucid_prompt(args.topic, cards, args.lucid)
            idea = await synthesize(
                prompt=prompt, model=args.model,
                stream_to_stdout=not args.quiet,
            )
            ideas.append(idea)
            mechanisms_used.append(LUCID_LABEL)
            cards_per_idea.append(cards)
            transcript_log(
                "idea", i=i, mechanism=LUCID_LABEL, text=idea,
                lucid_prior=args.lucid,
                cards=[{k: v for k, v in c.__dict__.items() if v is not None} for c in cards],
            )
            if args.quiet:
                print(idea)
    elif args.futures:
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
            transcript_log(
                "idea", i=i,
                mechanism=mechanisms_used[-1],
                text=idea,
                cards=[{k: v for k, v in c.__dict__.items() if v is not None} for c in cards],
            )

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
            transcript_log(
                "idea", i=i, mechanism=LSD_LABEL, text=idea,
                cards=[{k: v for k, v in c.__dict__.items() if v is not None} for c in cards],
            )
            if args.quiet:
                print(idea)

            # Sober-validation pass — error detection back online.
            print(
                "\n=== Sober validation (LSD pass "
                f"{i + 1}/{args.n_ideas}) ===\n"
            )
            sober = await lsd_validate(args.topic, idea, args.model)
            print(sober)
            transcript_log(
                "lsd_validation", i=i, anarchic=idea, sober=sober,
            )

            # The sober version is what we keep for downstream critic /
            # refine — it's the one that survived priors coming back online.
            ideas.append(sober)
            mechanisms_used.append(LSD_VALIDATION_LABEL)
            cards_per_idea.append(cards)
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
            transcript_log(
                "idea", i=i,
                mechanism=mechanisms_used[-1],
                text=idea,
                cards=[{k: v for k, v in c.__dict__.items() if v is not None} for c in cards],
            )

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
            transcript_log(
                "idea", i=i,
                mechanism=mechanisms_used[-1],
                text=idea,
                cards=[{k: v for k, v in c.__dict__.items() if v is not None} for c in cards],
            )

            if args.quiet:
                print(idea)

    # --- Critic + refine ---------------------------------------------------
    if args.refine and ideas:
        print(f"\n=== Critic pass ({len(ideas)} ideas) ===")
        scored: list[dict] = []
        for i, idea in enumerate(ideas):
            # CLI only scores inside the refine branch → always panel.
            score = await critic_score(
                args.topic, idea, args.model, force_panel=True,
            )
            score["i"] = i
            scored.append(score)
            transcript_log(
                "score", i=i,
                mechanism=mechanisms_used[i] if i < len(mechanisms_used) else None,
                feasibility=score["feasibility"],
                unexpectedness=score["unexpectedness"],
                topic_fit=score["topic_fit"],
                total=total_score(score),
                notes=score.get("notes", ""),
            )
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
        transcript_log(
            "winner", i=winner["i"], total=total_score(winner),
            mechanism=mechanisms_used[winner["i"]] if winner["i"] < len(mechanisms_used) else None,
            notes=winner.get("notes", ""),
        )
        # RAG: link the winning cards back to their critic_total so future
        # retrievals can boost them.
        try:
            from rag import record_winner as _rag_record_winner
            from transcripts import current_source
            from usage import current_run_id
            winning_card_names = [
                c.name for c in cards_per_idea[winner["i"]]
            ]
            _rag_record_winner(
                run_id=current_run_id() or "unknown",
                winning_card_names=winning_card_names,
                critic_total=total_score(winner),
                source=current_source() or "unknown",
            )
        except Exception:
            pass
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
        transcript_log("refined", i=winner["i"], text=refined)
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
                transcript_log(
                    "evolved",
                    pairs=[
                        {
                            "before": {k: v for k, v in o.__dict__.items() if v is not None},
                            "after":  {k: v for k, v in n.__dict__.items() if v is not None},
                        }
                        for o, n in zip(cards_per_idea[winner["i"]], evolved)
                    ],
                    cache_path=str(path),
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

    transcript_log("request_completed", n_ideas=len(ideas), refined=bool(args.refine))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
