"""Championship scoring for BUILD decisions — THE TUNABLE MODULE.

Ground-truth / reward for the under-building weakness (the baseline built only 70
settlements + 4 cities across 10 games — it passes/hoards when it should expand
or upgrade). Pure board mechanics (production + victory points + expansion) —
NONE of it is shown to the model (that would be coaching; see words-vs-rl.md).
Michael tunes WEIGHTS here; nothing else changes.

THE PRINCIPLE: reward making affordable, PRODUCTIVE builds; penalize passing /
hoarding when a strong build is affordable. Each legal build is scored on
computable value (production it adds + VP it gives + expansion it unlocks):

  settlement(node)  new production (the node's pips) + resource/number diversity
                    + 1 VP.  Node value via placement_score.score_components.
  city(node)        a city DOUBLES the node, so the marginal production added
                    equals the node's CURRENT production; + 1 VP over the
                    settlement. Cheap, efficient VP + production.
  road(edge)        the best NEW settlement spot it makes buildable (clone the
                    game, diff board.buildable_node_ids), scored as a settlement
                    node, discounted (you still must build the settlement later).
  dev card          a modest fixed expected value (not the focus).
  pass / non-build  0 build value — the hoard baseline.

    reward(build)            =  value(build)
    reward(pass / non-build) =  - w_hoard * max(0, best_affordable_build - HOARD_OK)

Components are computed once at generation (the clone work for roads) and stored
per option; `build_reward` is a pure combine over stored components + tunable
weights, so you RE-TUNE WITHOUT REGENERATING. Same function is the GRPO reward
and the eval scorer.
"""
from __future__ import annotations

from typing import Dict, Optional

from catanatron.models.enums import ActionType

from goldilocks_eval.placement_score import score_components

# ───────────────────────── EXPERT KNOB: tune these ─────────────────────────
WEIGHTS: Dict[str, float] = {
    "prod": 1.0,    # production a build adds (pip-normalized, [0,1])
    "div": 0.3,     # resource/number diversity of a new settlement
    "vp": 0.7,      # victory-point gain (settlement +1, city +1 over settlement)
    "road": 0.5,    # weight on expansion value a road unlocks
    "dev": 0.3,     # weight on a development-card buy
    "hoard": 0.8,   # penalty weight for passing on an affordable strong build
}
DEV_VALUE = 1.0        # expected utility of a dev card, in node-value units
ROAD_DISCOUNT = 0.6    # a road only SETS UP a future settlement, not built yet
HOARD_OK = 0.3         # passing is only penalized when the best build exceeds this
REWARD_CLAMP = (-1.5, 2.0)
# ─────────────────────────────────────────────────────────────────────────

_KIND = {
    ActionType.BUILD_SETTLEMENT: "settlement",
    ActionType.BUILD_CITY: "city",
    ActionType.BUILD_ROAD: "road",
    ActionType.BUY_DEVELOPMENT_CARD: "dev",
}
BUILD_TYPES = set(_KIND)


def _clamp(x: float) -> float:
    lo, hi = REWARD_CLAMP
    return max(lo, min(hi, x))


def _node_components(game, node: int) -> dict:
    """pip_norm + combined diversity for a settlement on `node`."""
    c = score_components(game, node)
    return {
        "pip_norm": round(c["pip_norm"], 4),
        "diversity": round((c["resource_diversity"] + c["number_diversity"]) / 2, 4),
    }


def score_build_option(game, color, action) -> dict:
    """Compute the (weight-free) COMPONENTS of one legal build action. Returns a
    JSON-able dict stored on the scenario; combine with `build_reward`."""
    kind = _KIND[action.action_type]
    if kind == "settlement":
        nc = _node_components(game, action.value)
        return {"kind": "settlement", "node": action.value, "vp": 1.0, **nc}
    if kind == "city":
        nc = _node_components(game, action.value)   # current production (city doubles it)
        return {"kind": "city", "node": action.value, "vp": 1.0,
                "pip_norm": nc["pip_norm"]}
    if kind == "road":
        # Clone, build the road, and value it by the best NEW settlement spot it
        # opens (board.buildable_node_ids after - before). Catanatron is the oracle.
        before = set(game.state.board.buildable_node_ids(color))
        clone = game.copy()
        clone.execute(action)
        newly = set(clone.state.board.buildable_node_ids(color)) - before
        best_node, best = None, {"pip_norm": 0.0, "diversity": 0.0}
        for n in newly:
            nc = _node_components(game, n)
            if nc["pip_norm"] >= best["pip_norm"]:
                best_node, best = n, nc
        return {"kind": "road", "edge": list(action.value),
                "opens_node": best_node,
                "opens_pip_norm": best["pip_norm"],
                "opens_diversity": best["diversity"]}
    return {"kind": "dev"}


# ─────────────────────────── pure reward (lookup) ──────────────────────────
def _settlement_value(pip_norm: float, diversity: float, w: Dict[str, float]) -> float:
    return w["prod"] * pip_norm + w["div"] * diversity


def build_reward(c: dict, weights: Optional[Dict[str, float]] = None) -> float:
    """Value of making the build described by stored components `c`. Pure."""
    w = weights or WEIGHTS
    k = c["kind"]
    if k == "settlement":
        r = _settlement_value(c["pip_norm"], c["diversity"], w) + w["vp"] * c["vp"]
    elif k == "city":
        r = w["prod"] * c["pip_norm"] + w["vp"] * c["vp"]
    elif k == "road":
        opened = (_settlement_value(c["opens_pip_norm"], c["opens_diversity"], w)
                  if c.get("opens_node") is not None else 0.0)
        r = w["road"] * ROAD_DISCOUNT * opened
    elif k == "dev":
        r = w["dev"] * DEV_VALUE
    else:
        r = 0.0
    return _clamp(r)


def best_build_value(build_options: Dict[str, dict],
                     weights: Optional[Dict[str, float]] = None) -> float:
    """The strongest affordable build in this state (0 if none)."""
    if not build_options:
        return 0.0
    return max(build_reward(c, weights) for c in build_options.values())


def hoard_penalty(best_value: float, weights: Optional[Dict[str, float]] = None) -> float:
    """Penalty for NOT building (pass / trade) when a strong build was affordable;
    scales with the best build forgone, 0 if nothing was worth building."""
    w = weights or WEIGHTS
    return _clamp(-w["hoard"] * max(0.0, best_value - HOARD_OK))


def classify(components: dict) -> str:
    """One-word label for reporting (the build kind)."""
    return components.get("kind", "?")
