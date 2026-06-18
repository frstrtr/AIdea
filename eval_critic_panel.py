"""A/B harness: does the diverse critic panel pick better winners than the
single critic?

For each topic it generates K ideas, scores every idea BOTH ways (single critic
vs panel), and asks an independent judge — framed differently from either
scorer — to pick the best idea. The judge is the ground-truth proxy. We then
measure how often each scorer's top-1 matches the judge (agreement@1), and on
the topics where the two scorers disagree, which winner the judge preferred.

This is the honest way to evaluate a scorer: you can't grade it with itself, so
a third, independent evaluator breaks the tie. The panel "wins" if it agrees
with the judge more often and/or wins the head-to-head on disagreements.

Run on a host with the agent SDK authenticated (e.g. the LXC):
  python eval_critic_panel.py --topics 2 --k 3
  python eval_critic_panel.py --dry           # print the plan + call estimate
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics

import aidea
from aidea import (
    CARD_DEPTH_BY_NAME,
    build_prompt,
    critic_panel,
    load_or_generate_deck,
    parse_entropy,
    sample_cards,
    synthesize,
    total_score,
    _critic_single,
    _query_text,
    _try_parse_json_object,
    CRITIC_AXES,
)

DEFAULT_TOPICS = [
    "help a small specialty coffee roaster increase repeat orders",
    "reduce no-shows for a solo physiotherapy clinic",
    "make a city bike-share scheme break even in a mid-size town",
    "grow an indie iOS weather app past 1k paying users",
]

# Independent judge — deliberately framed unlike either scorer (a funder
# choosing ONE idea), so its agreement is a meaningful external signal.
JUDGE_SYSTEM = (
    "You are the head of research choosing the single idea to fund. You back "
    "ideas that are UNEXPECTED YET FEASIBLE for the stated problem — neither "
    "obvious nor science-fiction. Output STRICT JSON only, no preamble."
    + aidea.LANGUAGE_RULE
)
JUDGE_TEMPLATE = """The problem:

  {topic}

Here are {k} candidate ideas. Choose the ONE best on 'unexpected yet feasible'.

{ideas}

Respond as ONE JSON object only: {{"best": <0-based index>, "reason": "<one sentence>"}}"""


async def judge_pick(topic: str, ideas: list[str], model: str) -> tuple[int, str]:
    block = "\n\n".join(f"[{i}]\n{idea}" for i, idea in enumerate(ideas))
    raw = await _query_text(
        JUDGE_TEMPLATE.format(topic=topic.strip(), k=len(ideas), ideas=block),
        JUDGE_SYSTEM, model, kind="judge",
    )
    obj = _try_parse_json_object(raw) or {}
    try:
        best = int(obj.get("best"))
    except (TypeError, ValueError):
        best = 0
    return max(0, min(len(ideas) - 1, best)), str(obj.get("reason", "")).strip()[:160]


async def gen_ideas(topic: str, k: int, level_name: str, model: str) -> list[str]:
    spread, level = parse_entropy(level_name)
    depth = CARD_DEPTH_BY_NAME["medium"]
    deck = await load_or_generate_deck(
        topic=topic, n=24, depth=depth, model=model,
        force_regen=False, verbose=False,
    )
    rng = random.Random(0)  # fixed: both scorers see the same K ideas
    ideas = []
    for _ in range(k):
        cards = sample_cards(deck=deck, n=3, spread=spread, rng=rng)
        ideas.append(await synthesize(
            prompt=build_prompt(topic, cards, level),
            model=model, stream_to_stdout=False,
        ))
    return ideas


def _spread(panel_score: dict) -> float:
    """Mean across-lens std per axis — how much the lenses actually disagree."""
    panel = panel_score.get("panel") or {}
    if len(panel) < 2:
        return 0.0
    spreads = []
    for ax in CRITIC_AXES:
        vals = [s[ax] for s in panel.values()]
        spreads.append(statistics.pstdev(vals))
    return round(statistics.mean(spreads), 1)


async def eval_topic(topic: str, k: int, level_name: str, model: str) -> dict:
    ideas = await gen_ideas(topic, k, level_name, model)
    singles, panels = [], []
    for idea in ideas:                      # sequential to bound concurrency
        singles.append(await _critic_single(topic, idea, model))
    for idea in ideas:
        panels.append(await critic_panel(topic, idea, model))
    win_s = max(range(k), key=lambda i: total_score(singles[i]))
    win_p = max(range(k), key=lambda i: total_score(panels[i]))
    j, jreason = await judge_pick(topic, ideas, model)
    axis_gap = statistics.mean(
        abs(singles[i][ax] - panels[i][ax])
        for i in range(k) for ax in CRITIC_AXES
    )
    return {
        "topic": topic, "k": k,
        "win_single": win_s, "win_panel": win_p, "judge": j, "jreason": jreason,
        "single_hit": win_s == j, "panel_hit": win_p == j,
        "disagree": win_s != win_p,
        "axis_gap": round(axis_gap, 1),
        "panel_spread": round(statistics.mean(_spread(p) for p in panels), 1),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=2, help="how many default topics")
    ap.add_argument("--k", type=int, default=3, help="ideas generated per topic")
    ap.add_argument("--entropy", default="wild",
                    help="sane | wild | insane | crazy | mad, or a float 0..1")
    ap.add_argument("--model", default=os.environ.get("AIDEA_MODEL", "claude-opus-4-8"))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    topics = DEFAULT_TOPICS[: args.topics]
    n_lenses = len(aidea.CRITIC_LENSES)
    calls = len(topics) * (1 + args.k + args.k + args.k * n_lenses + 1)
    print(f"{len(topics)} topics × {args.k} ideas · panel={n_lenses} lenses · "
          f"model={args.model}\n~{calls} LLM calls "
          f"(deck + synth + single + panel + judge)\n")
    if args.dry:
        for t in topics:
            print("  •", t)
        return

    rows = []
    for t in topics:
        print(f"▶ {t}")
        r = await eval_topic(t, args.k, args.entropy, args.model)
        rows.append(r)
        print(f"   single→#{r['win_single']}  panel→#{r['win_panel']}  "
              f"judge→#{r['judge']}  "
              f"[{'single✓' if r['single_hit'] else 'single✗'} "
              f"{'panel✓' if r['panel_hit'] else 'panel✗'}]  "
              f"axis-gap {r['axis_gap']}  panel-spread {r['panel_spread']}")
        print(f"   judge: {r['jreason']}")

    n = len(rows)
    s_hits = sum(r["single_hit"] for r in rows)
    p_hits = sum(r["panel_hit"] for r in rows)
    disagreements = [r for r in rows if r["disagree"]]
    # head-to-head on disagreements: judge sided with panel vs single
    h2h_panel = sum(1 for r in disagreements if r["panel_hit"] and not r["single_hit"])
    h2h_single = sum(1 for r in disagreements if r["single_hit"] and not r["panel_hit"])
    print("\n" + "=" * 56)
    print(f"agreement@1 with judge:  single {s_hits}/{n}   panel {p_hits}/{n}")
    print(f"scorers disagreed on winner: {len(disagreements)}/{n}")
    print(f"  head-to-head (judge's side): panel {h2h_panel} · single {h2h_single}")
    print(f"avg per-axis gap single↔panel: "
          f"{round(statistics.mean(r['axis_gap'] for r in rows), 1)}")
    print(f"avg panel internal spread:     "
          f"{round(statistics.mean(r['panel_spread'] for r in rows), 1)} "
          f"(0 = lenses agree, higher = genuine diversity)")


if __name__ == "__main__":
    asyncio.run(main())
