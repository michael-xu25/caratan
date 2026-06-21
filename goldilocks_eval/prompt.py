"""Render Catanatron game state into LLM-readable text, and parse the reply.

Kept deliberately small and dependency-light: a textual board+hand summary plus
a numbered list of the legal actions. The model returns an index, so we never
have to parse free-form Catan notation back into an Action.
"""
from __future__ import annotations

import re
from typing import List

from catanatron import Action, Color, Game
from catanatron.models.enums import RESOURCES
from catanatron.state_functions import (
    get_actual_victory_points,
    get_longest_road_length,
    player_num_dev_cards,
    player_num_resource_cards,
)


def _player_line(game: Game, color: Color, label: str) -> str:
    hand = {r: player_num_resource_cards(game.state, color, r) for r in RESOURCES}
    hand_str = " ".join(f"{r[:2]}={hand[r]}" for r in RESOURCES)
    return (
        f"{label} ({color.value}): "
        f"VP={get_actual_victory_points(game.state, color)} "
        f"longest_road={get_longest_road_length(game.state, color)} "
        f"dev_cards={player_num_dev_cards(game.state, color)} "
        f"| hand: {hand_str}"
    )


def render_action(action: Action) -> str:
    """Compact one-line rendering of an Action."""
    value = action.value
    if value is None:
        return action.action_type.value
    return f"{action.action_type.value} {value}"


def render_state(game: Game, me: Color, opponent: Color) -> str:
    """Human/LLM-readable snapshot of the game from `me`'s perspective."""
    lines = [
        f"Turn {game.state.num_turns}. You are {me.value}.",
        _player_line(game, me, "YOU"),
        _player_line(game, opponent, "OPPONENT"),
        f"Robber at: {game.state.board.robber_coordinate}",
    ]
    return "\n".join(lines)


def render_actions(playable_actions: List[Action]) -> str:
    return "\n".join(f"  [{i}] {render_action(a)}" for i, a in enumerate(playable_actions))


SYSTEM_PROMPT = (
    "You are an expert Settlers of Catan player in a 1v1 game (first to 10 "
    "victory points wins). You will be given the current game state and a "
    "numbered list of legal actions. Choose the single best action.\n\n"
    "Reply with ONLY a JSON object on one line:\n"
    '{"action": <index>, "reasoning": "<one short sentence>"}\n'
    "The index must be one of the listed action indices. Do not add prose "
    "outside the JSON."
)


def build_user_prompt(game: Game, me: Color, opponent: Color,
                      playable_actions: List[Action]) -> str:
    return (
        f"{render_state(game, me, opponent)}\n\n"
        f"Legal actions:\n{render_actions(playable_actions)}\n\n"
        f"Respond with the JSON object."
    )


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_choice(text: str, num_actions: int) -> tuple[int | None, str]:
    """Extract (index, reasoning) from the model reply.

    Returns (None, reasoning) if no valid in-range index could be parsed; the
    caller is responsible for falling back to a default action.
    """
    import json

    reasoning = ""
    match = _JSON_RE.search(text or "")
    if match:
        try:
            obj = json.loads(match.group(0))
            reasoning = str(obj.get("reasoning", ""))[:500]
            idx = int(obj["action"])
            if 0 <= idx < num_actions:
                return idx, reasoning
        except (ValueError, KeyError, TypeError):
            pass
    # Last resort: first standalone integer in the text.
    int_match = re.search(r"\d+", text or "")
    if int_match:
        idx = int(int_match.group(0))
        if 0 <= idx < num_actions:
            return idx, reasoning or "(parsed bare integer)"
    return None, reasoning or "(unparseable reply)"
