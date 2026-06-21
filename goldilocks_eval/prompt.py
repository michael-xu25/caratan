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


# Dice pip counts (ways to roll each number with 2d6) — higher = more likely.
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

# Action types whose `value` is a board node id (so we can annotate production).
from catanatron.models.enums import ActionType  # noqa: E402

_NODE_ACTIONS = {ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY}


def render_action(action: Action) -> str:
    """Compact one-line rendering of an Action (also used in transcripts — keep
    it stable; prompt-facing annotation lives in `render_actions`)."""
    value = action.value
    if value is None:
        return action.action_type.value
    return f"{action.action_type.value} {value}"


def _node_production(game: Game) -> dict:
    """node id -> list of (resource, number) for adjacent tiles (live board)."""
    prod: dict = {}
    for tile in game.state.board.map.land_tiles.values():
        res, num = getattr(tile, "resource", None), getattr(tile, "number", None)
        if res is None or num is None:
            continue  # desert
        rname = getattr(res, "value", res)
        for node_id in tile.nodes.values():
            prod.setdefault(node_id, []).append((rname, num))
    return prod


def _node_ports(game: Game) -> dict:
    """node id -> port label ('3:1' or 'WOOD 2:1')."""
    out: dict = {}
    for resource, nodes in game.state.board.map.port_nodes.items():
        label = "3:1" if resource is None else f"{getattr(resource,'value',resource)} 2:1"
        for n in nodes:
            out[n] = label
    return out


def _node_brief(node_id, prod: dict, ports: dict) -> str:
    parts, total = [], 0
    for res, num in prod.get(node_id, ()):
        total += PIPS.get(num, 0)
        parts.append(f"{res}:{num}({PIPS.get(num,0)}p)")
    s = ", ".join(parts) if parts else "no production"
    if node_id in ports:
        s += f", port {ports[node_id]}"
    return f"{s} [{total} pips]"


def render_board_summary(game: Game) -> str:
    """Global board picture: each resource's tiles (number+pips) and the ports."""
    from collections import defaultdict
    by_res = defaultdict(list)
    for tile in game.state.board.map.land_tiles.values():
        res, num = getattr(tile, "resource", None), getattr(tile, "number", None)
        if res is None or num is None:
            continue
        by_res[getattr(res, "value", res)].append(num)
    lines = ["Board tiles (resource: number(pips); robber blocks a tile):"]
    for res in sorted(by_res):
        nums = sorted(by_res[res], key=lambda n: (-PIPS.get(n, 0), n))
        lines.append(f"  {res}: " + " ".join(f"{n}({PIPS.get(n,0)}p)" for n in nums))
    ports = defaultdict(list)
    for n, label in _node_ports(game).items():
        ports[label].append(n)
    if ports:
        lines.append("Ports: " + "; ".join(
            f"{label} @ nodes {sorted(ns)}" for label, ns in sorted(ports.items())))
    return "\n".join(lines)


def render_state(game: Game, me: Color, opponent: Color) -> str:
    """Human/LLM-readable snapshot of the game from `me`'s perspective."""
    lines = [
        f"Turn {game.state.num_turns}. You are {me.value}. First to 10 VP wins.",
        _player_line(game, me, "YOU"),
        _player_line(game, opponent, "OPPONENT"),
        f"Robber is on tile: {game.state.board.robber_coordinate}",
        "",
        render_board_summary(game),
    ]
    return "\n".join(lines)


def render_actions(playable_actions: List[Action], game: Game = None) -> str:
    """Numbered legal-action list. When `game` is given, settlement/city options
    are annotated with the adjacent production + port so the node id is legible."""
    prod = _node_production(game) if game is not None else {}
    ports = _node_ports(game) if game is not None else {}
    out = []
    for i, a in enumerate(playable_actions):
        line = f"  [{i}] {render_action(a)}"
        if game is not None and a.action_type in _NODE_ACTIONS:
            line += f"   -> {_node_brief(a.value, prod, ports)}"
        out.append(line)
    return "\n".join(out)


# Format-agnostic 1v1 Catan rules primer. Shared by the live-play prompt AND the
# placement/grading prompt (goldilocks_eval.prompting) so the model is graded
# with the same game knowledge it plays with.
CATAN_RULES = (
    "You are an expert Settlers of Catan player in a 1-vs-1 game (exactly one "
    "opponent). Rules that matter:\n\n"
    "GOAL: first to 10 victory points (VP). VP come from: settlement = 1, "
    "city = 2, Longest Road (>=5 segments, the most) = 2, Largest Army "
    "(>=3 knights played, the most) = 2, and Victory-Point dev cards = 1 each.\n\n"
    "BUILD COSTS:\n"
    "- Road: 1 wood + 1 brick\n"
    "- Settlement: 1 wood + 1 brick + 1 sheep + 1 wheat (must be on a free node "
    ">=2 edges from any settlement and connected to your road)\n"
    "- City: upgrade a settlement for 2 wheat + 3 ore (doubles its production)\n"
    "- Development card: 1 sheep + 1 wheat + 1 ore\n\n"
    "PRODUCTION: each turn the active player rolls 2d6; every tile with that "
    "number pays its resource to adjacent settlements (1) / cities (2). Pips = "
    "how likely a number is (6/8=5, 5/9=4, 4/10=3, 3/11=2, 2/12=1); 7 = no "
    "production, move robber + steal. Settle high-pip, resource-diverse nodes.\n\n"
    "DEV CARDS: Knight = move robber + steal + counts toward Largest Army; "
    "Road Building = 2 free roads; Year of Plenty = take any 2 resources; "
    "Monopoly = name a resource, take ALL the opponent's of it; VP = +1 VP.\n\n"
    "ROBBER: blocks a tile (its owners get nothing from it) and steals 1 card "
    "from an adjacent player. TRADE: 4:1 with the bank, or 3:1/2:1 at ports you "
    "have settled.\n\n"
    "1v1 STRATEGY: tempo and denial decide games. Race Longest Road and Largest "
    "Army (each is a 2-VP swing), block the opponent's best tile with the robber, "
    "and don't over-trade away scarce resources. No multiplayer politics."
)

# Live-play action glossary (index-based selection over the full legal set).
_LIVE_ACTION_GLOSSARY = (
    "READING THE ACTIONS — you get the EXACT list of legal moves; pick one by "
    "index:\n"
    "- BUILD_SETTLEMENT N / BUILD_CITY N: N is a node id; its adjacent "
    "production (and port) is shown after '->'.\n"
    "- BUILD_ROAD (a, b): an edge between nodes a and b.\n"
    "- MOVE_ROBBER ((x,y,z), victim): tile to block + whom to steal from.\n"
    "- MARITIME_TRADE (...): give the listed resources to the bank for the last.\n"
    "- PLAY_KNIGHT/PLAY_*: play that dev card. ROLL / END_TURN / "
    "BUY_DEVELOPMENT_CARD as named."
)

RULES_1V1 = CATAN_RULES + "\n\n" + _LIVE_ACTION_GLOSSARY

_OUTPUT_WITH_REASONING = (
    "\n\nReply with ONLY a JSON object on one line:\n"
    '{"action": <index>, "reasoning": "<one short sentence>"}\n'
    "The index must be one of the listed action indices. No prose outside the JSON."
)

SYSTEM_PROMPT = RULES_1V1 + _OUTPUT_WITH_REASONING


def build_user_prompt(game: Game, me: Color, opponent: Color,
                      playable_actions: List[Action]) -> str:
    return (
        f"{render_state(game, me, opponent)}\n\n"
        f"Legal actions (choose one by index):\n"
        f"{render_actions(playable_actions, game)}\n\n"
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
