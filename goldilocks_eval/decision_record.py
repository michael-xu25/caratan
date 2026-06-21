"""Shared per-decision record builder — the atomic unit of an enriched transcript.

ONE source of truth so both runners (`goldilocks_eval` and `harness`) emit the
identical structured shape. A decision point is logged as a single record with
the counterfactual context weakness-discovery needs: the full legal set the agent
chose among, the decision-time game state, the chosen option, and (secondary) the
reasoning. Multi-action turns are a *sequence* of these single-decision records —
the env advances state between each, so each is built from the live game at that
moment.

The legal set is `playable_actions` (Catanatron's enumeration, the same list
rendered to the model); state is snapshotted from `game.state` BEFORE the action
is applied, so "while behind on VP with this hand" is captured as it was when the
agent decided.
"""
from __future__ import annotations

from typing import Any, Dict, List

from catanatron.models.enums import RESOURCES
from catanatron.state_functions import (
    get_actual_victory_points,
    get_longest_road_length,
    player_num_dev_cards,
    player_num_resource_cards,
)

from goldilocks_eval.prompt import render_action


def game_phase(state) -> str:
    """Coarse phase from the current VP leader: early ≤3, mid 4–7, late ≥8."""
    max_vp = max(get_actual_victory_points(state, c) for c in state.colors)
    if max_vp <= 3:
        return "early"
    if max_vp <= 7:
        return "mid"
    return "late"


def build_decision_record(game, color, legal_actions, chosen, *,
                          reasoning: str = "", fell_back: bool = False,
                          latency_ms: int = 0) -> Dict[str, Any]:
    """Build the enriched record for one decision. `legal_actions` is the full
    legal set (`playable_actions`); `chosen` is the selected Action."""
    state = game.state
    robber = getattr(state.board, "robber_coordinate", None)
    return {
        "turn": state.num_turns,
        "phase": game_phase(state),
        "color": color.value,
        "action_type": chosen.action_type.value,
        "chosen": render_action(chosen),
        "legal_actions": [render_action(a) for a in legal_actions],
        "num_legal": len(legal_actions),
        "state": {
            "vp": {c.value: get_actual_victory_points(state, c) for c in state.colors},
            "hand": {r: player_num_resource_cards(state, color, r) for r in RESOURCES},
            "dev_cards": player_num_dev_cards(state, color),
            "longest_road": get_longest_road_length(state, color),
            "robber": list(robber) if robber is not None else None,
        },
        "reasoning": reasoning or "",
        "fell_back": fell_back,
        "latency_ms": latency_ms,
    }
