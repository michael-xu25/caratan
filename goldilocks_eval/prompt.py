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


# Pip count = the dots printed under a number on the board = the number of ways
# two dice can make it. Pure board fact; shown neutrally (no ranking implied).
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
    parts = [f"{res} on {num} ({PIPS.get(num, 0)} pips)"
             for res, num in prod.get(node_id, ())]
    s = ", ".join(parts) if parts else "no production"
    if node_id in ports:
        s += f", {ports[node_id]} port"
    return s


def render_board_summary(game: Game) -> str:
    """Global board picture: each resource's tiles (the numbers that produce it)
    and the ports. Facts only — no scoring of tiles or nodes."""
    from collections import defaultdict
    by_res = defaultdict(list)
    for tile in game.state.board.map.land_tiles.values():
        res, num = getattr(tile, "resource", None), getattr(tile, "number", None)
        if res is None or num is None:
            continue
        by_res[getattr(res, "value", res)].append(num)
    lines = ["Tiles (each produces its resource when the two dice add up to its "
             "number; pips = dots on that number; the robber, if on a tile, stops "
             "it producing):"]
    for res in sorted(by_res):
        nums = sorted(by_res[res])
        lines.append(f"  {res}: produces on " +
                     ", ".join(f"{n} ({PIPS.get(n, 0)} pips)" for n in nums))
    ports = defaultdict(list)
    for n, label in _node_ports(game).items():
        ports[label].append(n)
    if ports:
        lines.append("Ports (trade rate available from a settlement/city on these "
                     "nodes): " + "; ".join(
            f"{label} at nodes {sorted(ns)}" for label, ns in sorted(ports.items())))
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
    "You are playing a 1-vs-1 game of Settlers of Catan (you and exactly one "
    "opponent). Here is how the game works.\n\n"

    "WINNING: the first player to reach 10 victory points (VP) wins. You get VP "
    "from each of these: a settlement you own is worth 1 VP; a city is worth "
    "2 VP; holding Longest Road is worth 2 VP (you hold it if you have the single "
    "longest connected run of your own roads and it is at least 5 roads long, and "
    "longer than the opponent's); holding Largest Army is worth 2 VP (you hold it "
    "if you have played the most knight cards and have played at least 3, more "
    "than the opponent); and each victory-point development card you hold is "
    "worth 1 VP (these stay hidden until you win).\n\n"

    "RESOURCES: there are five resources - wood, brick, sheep, wheat, and ore. "
    "You spend them to build things.\n\n"

    "WHAT THINGS COST:\n"
    "- Road: 1 wood + 1 brick.\n"
    "- Settlement: 1 wood + 1 brick + 1 sheep + 1 wheat. It must go on an empty "
    "intersection that is connected to one of your own roads and is at least two "
    "edges away from any existing settlement or city (yours or the opponent's).\n"
    "- City: 2 wheat + 3 ore to upgrade one of your existing settlements into a "
    "city. A city collects 2 resources from each of its tiles instead of 1.\n"
    "- Development card: 1 sheep + 1 wheat + 1 ore (you draw a random one).\n\n"

    "HOW TILES PRODUCE RESOURCES: every land tile shows one resource and a number "
    "from 2 to 12 (one tile is the desert and produces nothing). On a player's "
    "turn they roll two dice and add them up; every tile whose number equals that "
    "total produces, giving its resource to each player who has a settlement "
    "(1 resource) or city (2 resources) on a corner of that tile. Because two "
    "dice are added, the totals are not equally likely: there are 5 ways to roll "
    "a 6 or an 8, 4 ways to roll a 5 or 9, 3 ways for a 4 or 10, 2 ways for a 3 "
    "or 11, and 1 way for a 2 or 12. This count is the number's pip count - the "
    "dots printed under it on the board. The board and action lists write a tile "
    "as e.g. 'SHEEP on 9 (4 pips)', meaning a sheep tile that produces when the "
    "two dice add up to 9.\n\n"

    "ROLLING A 7 AND THE ROBBER: if the two dice add up to 7, no tile produces. "
    "Instead the player who rolled moves the robber onto any tile (while the "
    "robber sits on a tile, that tile produces nothing for anyone) and steals "
    "1 random resource card from an opponent who has a settlement or city on that "
    "tile. Also, whenever a 7 is rolled, any player holding more than 7 resource "
    "cards must discard half of them.\n\n"

    "DEVELOPMENT CARDS (what each one does when played): Knight - move the robber "
    "and steal, the same as rolling a 7; played knights count toward Largest "
    "Army. Road Building - place 2 roads for free. Year of Plenty - take any "
    "2 resources from the bank. Monopoly - name one resource and take every card "
    "of that resource from the opponent. Victory Point - worth 1 VP. You may play "
    "at most one development card per turn, and not on the same turn you bought "
    "it (victory-point cards are the exception).\n\n"

    "TRADING: on your turn you may trade resources with the bank at 4 of one "
    "resource for 1 of any other (4:1). If you own a settlement or city on a "
    "port, you may instead trade at that port's ratio: a generic port is 3:1 "
    "(any 3 matching resources for 1), and a resource-specific port is 2:1 "
    "(2 of that resource for 1 of any other).\n\n"

    "A TURN, STEP BY STEP: first you roll the two dice (this produces resources, "
    "or triggers the robber on a 7). After rolling you may take as many actions "
    "as you can afford, in any order you like: build roads, settlements, and "
    "cities, buy a development card, play one development card, and trade. Each of "
    "these is a separate action; you keep taking actions until you choose to end "
    "your turn."
)

# Live-play action glossary (index-based selection over the full legal set).
_LIVE_ACTION_GLOSSARY = (
    "HOW TO ANSWER: you are given the exact list of legal moves available right "
    "now; choose one of them by its index number. What each move means:\n"
    "- BUILD_SETTLEMENT N / BUILD_CITY N: N is a board intersection (node) id. "
    "After '->' you are shown which resources that node would collect (which tile "
    "and on what dice roll) and whether it sits on a port.\n"
    "- BUILD_ROAD (a, b): build a road on the edge between nodes a and b.\n"
    "- MOVE_ROBBER ((x, y, z), victim): move the robber onto the tile at that "
    "coordinate and steal from the named opponent (or no one).\n"
    "- MARITIME_TRADE (...): give the listed resources to the bank in exchange "
    "for the last one listed.\n"
    "- PLAY_KNIGHT and other PLAY_* moves play that development card. "
    "ROLL, END_TURN, and BUY_DEVELOPMENT_CARD do what their names say."
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
