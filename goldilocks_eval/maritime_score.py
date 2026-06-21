"""Championship scoring for MARITIME TRADE decisions — THE TUNABLE MODULE.

This is the ground-truth / reward used to train and grade the over-trading
weakness. It is pure board mechanics (what a trade does to your hand and to what
you can build) — NONE of it is ever shown to the model (that would be coaching;
see words-vs-rl.md). Michael (the Catan expert) tunes WEIGHTS / BUILD_VALUE here
after seeing results; nothing else needs to change.

THE PRINCIPLE: a maritime trade is *productive* only when it converts surplus
into progress toward a build. So each legal trade is scored on four mechanical,
computable components (all derived from the live game state):

  enables      builds that are ILLEGAL now but become LEGAL after this trade —
               the exact "enables a concrete build this turn" test. Computed by
               cloning the game, applying the trade, and reading the new
               playable_actions (Catanatron itself is the oracle, so piece
               supply, the city-upgrade rule, and legal-location are all honored).
  progresses   the trade strictly reduces the resource-distance to the cheapest
               REACHABLE pursuable build without yet enabling it (a meaningful
               advance, partial credit).
  gives_scarce the resource you GIVE is needed by a reachable build (you cut
               below its requirement) and/or is production-scarce for you (your
               settlements/cities make it at <= SCARCE_PIPS total pips).
  churns       neither enables nor progresses — pointless churn.

    reward(trade) =  w.enable   * build_value[best enabled]      # unlocks a build
                  +  w.progress * 1{progresses}                  # advances toward one
                  -  w.churn    * 1{churns}                      # pointless churn
                  -  w.scarcity * gives_scarce                   # dumps a needed resource
    reward(no-trade / build / end_turn) = 0.0                    # neutral baseline

Components are computed once at generation time (the expensive clone work) and
stored per option; `maritime_reward` is a pure combine over the stored
components + the tunable weights, so you can RE-TUNE WEIGHTS WITHOUT
REGENERATING DATA. The same function is the GRPO reward and the eval scorer.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from catanatron.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
)
from catanatron.models.enums import RESOURCES, ActionType
from catanatron.state_functions import player_key, player_num_resource_cards

from goldilocks_eval.prompt import PIPS, _node_production

# ───────────────────────── EXPERT KNOB: tune these ─────────────────────────
WEIGHTS: Dict[str, float] = {
    "enable": 1.0,     # the productive case: trade unlocks a build this turn
    "progress": 0.35,  # partial credit: advances toward a reachable build
    "churn": 0.6,      # penalty for a trade that does nothing useful
    "scarcity": 0.5,   # penalty for trading away a needed / production-scarce resource
}
# Value of the build a trade unlocks (city >> road). Multiplied by WEIGHTS["enable"].
BUILD_VALUE: Dict[str, float] = {
    "BUILD_CITY": 1.0,
    "BUILD_SETTLEMENT": 0.85,
    "BUY_DEVELOPMENT_CARD": 0.5,
    "BUILD_ROAD": 0.4,
}
SCARCE_PIPS = 3            # you produce a resource "scarcely" at <= this many total pips
REACHABLE_MISSING = 4      # a build is "reachable" if still missing <= this many cards
PROD_SCARCE_WEIGHT = 0.5   # gives_scarce credit for production-scarcity (vs 1.0 for needed)
REWARD_CLAMP = (-1.1, 1.2)
# ─────────────────────────────────────────────────────────────────────────

# Build targets and their cost freqdecks (order = RESOURCES = WOOD,BRICK,SHEEP,WHEAT,ORE).
TARGET_COSTS: Dict[str, List[int]] = {
    "BUILD_ROAD": ROAD_COST_FREQDECK,
    "BUILD_SETTLEMENT": SETTLEMENT_COST_FREQDECK,
    "BUILD_CITY": CITY_COST_FREQDECK,
    "BUY_DEVELOPMENT_CARD": DEVELOPMENT_CARD_COST_FREQDECK,
}
_BUILD_TYPES = {
    ActionType.BUILD_ROAD,
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_CITY,
    ActionType.BUY_DEVELOPMENT_CARD,
}


# ───────────────────────────── live-game helpers ──────────────────────────
def _hand_freqdeck(state, color) -> List[int]:
    return [player_num_resource_cards(state, color, r) for r in RESOURCES]


def _missing(hand: List[int], cost: List[int]) -> int:
    """Total resource cards still needed to afford `cost` from `hand`."""
    return sum(max(0, cost[i] - hand[i]) for i in range(len(cost)))


def player_production_pips(game, color) -> Dict[str, int]:
    """resource -> total dice-pips this player produces for it (settlement=1x,
    city=2x). The structural-scarcity signal: a resource you barely make."""
    prod = _node_production(game)
    pips = {r: 0 for r in RESOURCES}
    for node, (c, btype) in game.state.board.buildings.items():
        if c != color:
            continue
        mult = 2 if str(btype).upper() == "CITY" else 1
        for res, num in prod.get(node, []):
            if res in pips:
                pips[res] += PIPS.get(num, 0) * mult
    return pips


def pursuable_targets(game, color) -> List[str]:
    """Build targets the player still has the PIECES to pursue (the oracle handles
    legal-location/affordability for `enables`; this gates the softer signals)."""
    state = game.state
    k = player_key(state, color)
    ps = state.player_state
    owns_settlement = any(
        c == color and str(bt).upper() == "SETTLEMENT"
        for c, bt in game.state.board.buildings.values()
    )
    out = ["BUY_DEVELOPMENT_CARD"]
    if ps[f"{k}_ROADS_AVAILABLE"] > 0:
        out.append("BUILD_ROAD")
    if ps[f"{k}_SETTLEMENTS_AVAILABLE"] > 0:
        out.append("BUILD_SETTLEMENT")
    if ps[f"{k}_CITIES_AVAILABLE"] > 0 and owns_settlement:
        out.append("BUILD_CITY")
    return out


def score_trade_option(game, color, action, *, weights=None) -> dict:
    """Compute the (tunable-weight-free) COMPONENTS of one MARITIME_TRADE action.

    Uses game.copy() to read which builds become legal after the trade — the
    exact "enables a build this turn" oracle. Returns a JSON-able dict stored on
    the scenario; combine it with `maritime_reward` (weights applied there)."""
    value = action.value  # (give, give, give, give, receive); give-slots padded with None
    give = value[0]
    receive = value[-1]
    rate = sum(1 for v in value[:-1] if v is not None)

    state = game.state
    hand = _hand_freqdeck(state, color)
    gi, ri = RESOURCES.index(give), RESOURCES.index(receive)
    post = list(hand)
    post[gi] -= rate
    post[ri] += 1

    # enables: builds illegal now that are legal after the trade (same turn).
    builds_now = {a.action_type for a in game.playable_actions
                  if a.action_type in _BUILD_TYPES}
    clone = game.copy()
    clone.execute(action)
    builds_after = {a.action_type for a in clone.playable_actions
                    if a.action_type in _BUILD_TYPES} if clone.state.current_color() == color else set()
    enables = sorted(bt.value for bt in (builds_after - builds_now))

    targets = pursuable_targets(game, color)

    # progresses: not enabling, but strictly closer to a reachable pursuable build.
    progresses = False
    if not enables:
        for t in targets:
            cost = TARGET_COSTS[t]
            before, after = _missing(hand, cost), _missing(post, cost)
            if after < before and after <= REACHABLE_MISSING:
                progresses = True
                break

    # gives_scarce: the resource handed away is needed by your cheapest REACHABLE
    # build (cutting below its requirement) and/or is production-scarce for you.
    # Dev card is excluded from the "needed" set — it costs 1 of every resource,
    # so it would make every resource look needed and wash the signal out.
    build_targets = [(_missing(hand, TARGET_COSTS[t]), t) for t in targets
                     if t != "BUY_DEVELOPMENT_CARD"
                     and _missing(hand, TARGET_COSTS[t]) <= REACHABLE_MISSING]
    needed = False
    if build_targets:
        cost = TARGET_COSTS[min(build_targets)[1]]   # cheapest reachable build's cost
        needed = cost[gi] > 0 and post[gi] < cost[gi]
    prod_scarce = player_production_pips(game, color)[give] <= SCARCE_PIPS
    gives_scarce = 1.0 if needed else (PROD_SCARCE_WEIGHT if prod_scarce else 0.0)

    churns = (not enables) and (not progresses)
    return {
        "give": give, "receive": receive, "rate": rate,
        "enables": enables,                 # [] or e.g. ["BUILD_CITY"]
        "progresses": progresses,
        "gives_scarce": gives_scarce,
        "churns": churns,
        "hand_before": dict(zip(RESOURCES, hand)),
        "hand_after": dict(zip(RESOURCES, post)),
    }


# ─────────────────────────── pure reward (lookup) ──────────────────────────
def maritime_reward(components: dict,
                    weights: Optional[Dict[str, float]] = None,
                    build_value: Optional[Dict[str, float]] = None) -> float:
    """Combine stored COMPONENTS into a scalar reward. Pure: no game needed.

    This is both the GRPO reward (for a chosen maritime trade) and the eval
    scorer. Weights/build-values are applied HERE so they can be tuned without
    regenerating the per-option components."""
    w = weights or WEIGHTS
    bv = build_value or BUILD_VALUE
    enables = components.get("enables") or []
    if enables:
        r = w["enable"] * max(bv.get(b, 0.0) for b in enables)
    elif components.get("progresses"):
        r = w["progress"]
    else:
        r = -w["churn"]
    r -= w["scarcity"] * float(components.get("gives_scarce", 0.0))
    lo, hi = REWARD_CLAMP
    return max(lo, min(hi, r))


# Reward for any NON-trade choice (build / end_turn / dev / roll …): the neutral
# baseline an over-trader must learn beats every churning trade.
NO_TRADE_REWARD = 0.0


def classify(components: dict) -> str:
    """One-word label for reporting."""
    if components.get("enables"):
        return "enabling"
    if components.get("progresses"):
        return "progressing"
    return "churn"
