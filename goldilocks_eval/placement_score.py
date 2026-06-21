"""Championship scoring for opening settlement placement — THE TUNABLE MODULE.

This is the ground-truth / reward used to train and grade placement. It is pure
board mechanics (pips, resources, numbers) — NONE of it is ever shown to the
model (that would be coaching; see words-vs-rl.md). Michael (the Catan expert)
tunes WEIGHTS here after seeing results; nothing else needs to change.

A spot (intersection/node) is scored from the tiles it borders, on three
championship criteria, each normalized to roughly [0, 1] so the weights are
comparable:

  pip                 total dice-probability of the bordering tiles (pips),
                      normalized by MAX_PIPS. Higher = more production.
  resource_diversity  distinct resource types among those tiles / 3.
  number_diversity    distinct dice-numbers among those tiles / (#tiles).
                      Penalizes stacking the same number on one spot.

score = w_pip*pip_norm + w_res*resource_diversity + w_num*number_diversity

v1 scores each placement INDEPENDENTLY (no cross-settlement complementarity).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from goldilocks_eval.prompt import PIPS, _node_production

# ───────────────────────── EXPERT KNOB: tune these ─────────────────────────
WEIGHTS: Dict[str, float] = {
    "pip": 3.0,                 # production quantity (dice odds) — DOMINATES the score
    "resource_diversity": 1.0,  # breadth of resource types (tiebreaker)
    "number_diversity": 1.0,    # avoid stacking the same number on one spot
}
# A single spot borders up to 3 tiles; the best case is 3 tiles at 5 pips each.
MAX_PIPS = 15.0
# Reward = 1.0 if the chosen spot is among the TOP_K by score, else 0.0. Dead
# simple, maximally discriminative, and identical to the eval metric (top-3).
TOP_K = 3
# ─────────────────────────────────────────────────────────────────────────


def score_components(game, node: int) -> dict:
    """Per-criterion breakdown for one node on `game`'s board."""
    tiles: List[Tuple[str, int]] = _node_production(game).get(node, [])
    numbers = [n for _, n in tiles]
    resources = [r for r, _ in tiles]
    n_tiles = len(tiles)
    pip_total = sum(PIPS.get(n, 0) for n in numbers)
    return {
        "tiles": tiles,                                  # [(resource, number), ...]
        "pip_total": pip_total,
        "pip_norm": pip_total / MAX_PIPS,
        "resource_diversity": (len(set(resources)) / 3.0) if tiles else 0.0,
        "number_diversity": (len(set(numbers)) / n_tiles) if n_tiles else 0.0,
    }


def score_node(game, node: int, weights: Optional[Dict[str, float]] = None
               ) -> Tuple[float, dict]:
    """Return (combined_score, components) for one node."""
    w = weights or WEIGHTS
    c = score_components(game, node)
    total = (w["pip"] * c["pip_norm"]
             + w["resource_diversity"] * c["resource_diversity"]
             + w["number_diversity"] * c["number_diversity"])
    return total, c


def score_legal_spots(game, legal_nodes: List[int],
                      weights: Optional[Dict[str, float]] = None
                      ) -> Tuple[Dict[int, Tuple[float, dict]], int]:
    """Score every legal spot; return ({node: (score, components)}, best_node)."""
    scored = {n: score_node(game, n, weights) for n in legal_nodes}
    best = max(scored, key=lambda n: scored[n][0])
    return scored, best


def _reward(chosen, totals: dict, mode: str = "topk") -> float:
    """Reward for `chosen` given {key: score}. 0.0 if `chosen` absent (illegal).

    Default "topk": 1.0 iff chosen is among the TOP_K highest-scoring spots, else
    0.0 (ties at the cutoff all count). Other modes kept for comparison."""
    if chosen not in totals or not totals:
        return 0.0
    c = totals[chosen]
    order = sorted(totals.values(), reverse=True)
    best, worst = order[0], order[-1]
    if mode == "topk":
        threshold = order[min(TOP_K - 1, len(order) - 1)]
        return 1.0 if c >= threshold else 0.0
    if mode == "ratio":
        return c / best if best > 0 else 1.0
    if mode == "rank":
        return 1.0 - order.index(c) / (len(order) - 1) if len(order) > 1 else 1.0
    return (c - worst) / (best - worst) if best > worst else 1.0  # normalized (legacy)


def regret(chosen, scored_or_totals) -> float:
    """Normalized regret in [0,1]: (best - chosen) / (best - worst). 0 = optimal
    pick, 1 = worst. 1.0 if chosen is missing/illegal. The discriminative eval
    metric (mean regret + regret==0 rate, i.e. top-1)."""
    totals = ({k: (v[0] if isinstance(v, tuple) else v)
               for k, v in scored_or_totals.items()})
    if chosen not in totals or not totals:
        return 1.0
    best, worst = max(totals.values()), min(totals.values())
    return (best - totals[chosen]) / (best - worst) if best > worst else 0.0


def placement_reward(chosen: int, scored: Dict[int, Tuple[float, dict]],
                     mode: str = "topk") -> float:
    """Reward for picking node `chosen` from the scored legal set (eval path).

    mode:
      "normalized" (default, recommended for GRPO) — (c-worst)/(best-worst):
        1.0 = optimal spot, 0.0 = worst; uses the full range each decision.
      "ratio" — chosen_score / best_score (gentler).
      "rank" — 1 - rank/(n-1), ordinal.
    """
    return _reward(chosen, {n: v[0] for n, v in scored.items()}, mode)


def reward_from_scores(chosen, scores: Dict, mode: str = "topk") -> float:
    """Reward from a flat {node_id_str: score} map (training-reward path — the
    reward function reconstructs this from the dataset's ground_truth). Keys and
    `chosen` are normalized to 'node_<int>'."""
    from goldilocks_eval.prompting import node_id_str
    totals = {node_id_str(k): float(v) for k, v in scores.items()}
    return _reward(node_id_str(chosen) if chosen is not None else None, totals, mode)
