"""Derive the grader's structured context from the live game state:
  - decision_type  (placement | trade | build_spend)  from the action, and
  - state_tags     (the frozen STATE_TAGS vocab)       from the position.

Both come from Catanatron's own State during the deterministic replay, so the
tags are accurate (buildable nodes, robber adjacency, awards) rather than guessed
from the transcript summary. decision_type is None for moves we don't grade
(rolls, end-turn, discards).
"""
from __future__ import annotations

from catanatron.models.player import Color
from catanatron.state_functions import (
    get_actual_victory_points, player_num_resource_cards, player_key)
from catanatron.models.enums import SETTLEMENT, CITY

# action_type -> grader decision_type (None = not graded)
_DECISION_TYPE = {
    "BUILD_SETTLEMENT": "placement",
    "BUILD_CITY": "placement",
    "MARITIME_TRADE": "trade",
    "BUILD_ROAD": "build_spend",
    "BUY_DEVELOPMENT_CARD": "build_spend",
    "PLAY_KNIGHT": "build_spend",
    "PLAY_MONOPOLY": "build_spend",
    "PLAY_YEAR_OF_PLENTY": "build_spend",
    "PLAY_ROAD_BUILDING": "build_spend",
    "MOVE_ROBBER": "build_spend",
}


def decision_type_of(action_type: str) -> str | None:
    return _DECISION_TYPE.get(action_type)


def _phase_tag(game, self_vp: int, opp_vp: int) -> str:
    if game.state.is_initial_build_phase:
        return "opening"
    top = max(self_vp, opp_vp)
    if top <= 2:
        return "early"
    if top <= 5:
        return "mid"
    if top <= 7:
        return "late"
    return "endgame"


def _standing_tag(self_vp: int, opp_vp: int) -> str:
    d = self_vp - opp_vp
    if d >= 2:
        return "leading"
    if d <= -4:
        return "far_behind"
    if d <= -2:
        return "behind"
    return "even"


def derive_state_tags(game, color: Color) -> list[str]:
    """All frozen STATE_TAGS that hold for `color` at the current state."""
    state = game.state
    colors = list(state.colors)
    opp = next((c for c in colors if c != color), color)
    self_vp = get_actual_victory_points(state, color)
    opp_vp = get_actual_victory_points(state, opp)

    tags = [_phase_tag(game, self_vp, opp_vp), _standing_tag(self_vp, opp_vp)]

    # resources
    n_cards = player_num_resource_cards(state, color)
    if n_cards >= 8:
        tags += ["near_discard", "resource_rich"]
    elif n_cards >= 6:
        tags.append("resource_rich")
    elif n_cards <= 2:
        tags.append("resource_starved")

    # board: expansion room from buildable nodes
    try:
        n_build = len(state.board.buildable_node_ids(color))
    except Exception:
        n_build = None
    if n_build is not None:
        if n_build <= 1:
            tags.append("boxed_in")
        elif n_build >= 5:
            tags.append("open_board")
        else:
            tags.append("contested")

    # robber threat: robber sits on a tile adjacent to one of our buildings
    owned = (state.buildings_by_color[color].get(SETTLEMENT, [])
             + state.buildings_by_color[color].get(CITY, []))
    owned_tiles = set()
    for n in owned:
        owned_tiles.update(state.board.map.adjacent_tiles[n])
    robber_tile = state.board.map.land_tiles.get(state.board.robber_coordinate)
    if robber_tile is not None and robber_tile in owned_tiles:
        tags.append("robber_threat")

    # awards + win proximity
    key, okey = player_key(state, color), player_key(state, opp)
    if state.player_state.get(f"{key}_HAS_ROAD"):
        tags.append("has_longest_road")
    if state.player_state.get(f"{key}_HAS_ARMY"):
        tags.append("has_largest_army")
    if self_vp >= 9:
        tags.append("self_one_from_win")
    if opp_vp >= 9:
        tags.append("opp_one_from_win")

    # de-dup, preserve order
    seen, out = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t); out.append(t)
    return out
