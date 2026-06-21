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
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)


def _player_line(game: Game, color: Color, label: str, public_only: bool = False) -> str:
    """One-line player summary. Public info (resources, VP, longest road,
    dev-card COUNT) is shown for everyone. The only hidden info is dev-card
    IDENTITY: for the OPPONENT (`public_only`) we show the *visible* VP, which
    excludes their hidden victory-point cards; for YOU we show actual VP."""
    state = game.state
    lr = get_longest_road_length(state, color)
    ndev = player_num_dev_cards(state, color)            # count is public; types are not
    hand = {r: player_num_resource_cards(state, color, r) for r in RESOURCES}  # resources are PUBLIC
    hand_str = " ".join(f"{r[:2]}={hand[r]}" for r in RESOURCES)
    ps, k = state.player_state, player_key(state, color)
    if public_only:
        # visible VP excludes the opponent's hidden victory-point dev cards
        vp_str = f"{ps[f'{k}_VICTORY_POINTS']} (visible)"
    else:
        vp_str = str(get_actual_victory_points(state, color))
    return (
        f"{label} ({color.value}): "
        f"VP={vp_str} longest_road={lr} dev_cards={ndev} "
        f"| pieces left to build: {ps[f'{k}_SETTLEMENTS_AVAILABLE']} settlements, "
        f"{ps[f'{k}_CITIES_AVAILABLE']} cities, {ps[f'{k}_ROADS_AVAILABLE']} roads "
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
        _player_line(game, opponent, "OPPONENT", public_only=True),
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
    "opponent). This explains the whole game from scratch — assume no prior "
    "knowledge.\n\n"

    "THE BOARD: the board is made of hexagonal (six-sided) tiles fitted together. "
    "Each land tile is one resource type (wood, brick, sheep, wheat, or ore) and "
    "carries a number token from 2 to 12 — except one desert tile, which has no "
    "number and never produces anything. Three things you place sit at different "
    "places on this grid:\n"
    "- A point where the corners of tiles meet is an INTERSECTION (also called a "
    "node). Each intersection is shared by up to 3 tiles at once. You build "
    "SETTLEMENTS and CITIES on intersections.\n"
    "- The line connecting two neighbouring intersections is an EDGE. You build "
    "ROADS on edges.\n"
    "So a settlement/city occupies a corner shared by up to 3 tiles, and a road "
    "occupies the border line between two corners.\n\n"

    "RESOURCES: the five resources are wood, brick, sheep, wheat, and ore. You "
    "spend them to build.\n\n"

    "HOW YOU GET RESOURCES (this is the heart of the game): on a player's turn "
    "they roll two dice and add them into a total from 2 to 12. EVERY land tile "
    "whose number token equals that total produces its resource — this happens "
    "for BOTH players on every roll, not only the player who rolled. You collect "
    "from a producing tile only if one of YOUR settlements or cities sits on a "
    "corner of that tile: a settlement collects 1 of the resource, a city "
    "collects 2. Since one intersection touches up to 3 tiles, a single "
    "settlement can earn from up to 3 tiles over the game — whenever any of those "
    "tiles' numbers is rolled. Worked example: your settlement sits on the corner "
    "where a 'wheat on 6' tile, an 'ore on 9' tile, and a 'wood on 9' tile meet. "
    "Then a roll of 6 gives you 1 wheat; a roll of 9 gives you 1 wood AND 1 ore "
    "(both 9-tiles produce); any other roll gives that settlement nothing. Make "
    "it a city and those same rolls give 2 wheat, or 2 wood and 2 ore.\n\n"

    "DICE ODDS: because two dice are added, totals are not equally likely. The "
    "number of ways to roll each total is its PIP COUNT — the dots printed under "
    "the number on the board: 5 ways for a 6 or 8, 4 for a 5 or 9, 3 for a 4 or "
    "10, 2 for a 3 or 11, 1 for a 2 or 12. The board and action lists write a "
    "tile as e.g. 'SHEEP on 9 (4 pips)' = a sheep tile that produces when the two "
    "dice add up to 9, and 9 can be rolled 4 ways.\n\n"

    "WINNING: the first player to reach 10 victory points (VP) wins. You get VP "
    "from each of these: a settlement you own is worth 1 VP; a city is worth "
    "2 VP; holding Longest Road is worth 2 VP (you hold it if you have the single "
    "longest connected run of your own roads and it is at least 5 roads long, and "
    "longer than the opponent's — and note an opponent who builds a settlement on "
    "a node partway along your road splits that run into shorter separate pieces); "
    "holding Largest Army is worth 2 VP (you hold it if you have played the most "
    "knight cards and have played at least 3, more than the opponent); and each "
    "victory-point development card you hold is worth 1 VP — it counts toward your "
    "10 as soon as you hold it, and stays hidden from the opponent until the game "
    "ends.\n\n"

    "WHAT THINGS COST AND THE BUILD RULES:\n"
    "- Road: 1 wood + 1 brick. A road must connect to one of your existing roads, "
    "settlements, or cities (your roads form a connected network).\n"
    "- Settlement: 1 wood + 1 brick + 1 sheep + 1 wheat. It must go on an empty "
    "intersection that is connected to one of your own roads, and that is at "
    "least two edges away from any existing settlement or city (the 'distance "
    "rule' — no two settlements/cities can be neighbours).\n"
    "- City: 2 wheat + 3 ore to upgrade one of your OWN existing settlements into "
    "a city on the same spot. A city collects 2 resources per tile instead of 1.\n"
    "- Development card: 1 sheep + 1 wheat + 1 ore (you draw a random card from "
    "the deck, which holds 14 Knights, 5 Victory Point cards, and 2 each of Road "
    "Building, Year of Plenty, and Monopoly).\n\n"

    "PIECE SUPPLY: each player has a limited stock of pieces — 5 settlements, "
    "4 cities, and 15 roads in total — and cannot build beyond that. (Upgrading a "
    "settlement to a city returns the settlement to your stock.)\n\n"

    "THE OPENING: the game begins with each player placing a couple of starting "
    "settlements and roads for free (the distance rule still applies; the "
    "road-connection rule is waived for these opening placements). You will "
    "simply be shown the legal placements to choose from. After the opening, "
    "normal dice-rolling turns begin.\n\n"

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
    "(any 3 of the same resource for 1), and a resource-specific port is 2:1 "
    "(2 of that resource for 1 of any other). There is no player-to-player "
    "trading in this game — you exchange resources only with the bank or at your "
    "ports.\n\n"

    "A TURN, STEP BY STEP: a turn has two parts.\n"
    "(1) BEFORE rolling, you may play one development card. This matters because "
    "a Knight played before the roll moves the robber BEFORE that roll's "
    "production is paid out, so it changes which tiles produce this turn.\n"
    "(2) You ROLL the two dice — this pays out production to everyone, or triggers "
    "the robber on a 7.\n"
    "(3) After rolling you may take as many actions as you can afford, in any "
    "order: build roads, settlements, and cities, buy a development card, play a "
    "development card, and trade with the bank. Each is a separate action; you "
    "keep acting until you END_TURN.\n"
    "You may play at most ONE development card in the entire turn — whether before "
    "or after the roll — and never one you bought on this same turn.\n\n"

    "YOU DO NOT ENFORCE THE RULES: the game checks every rule for you and always "
    "offers you exactly the moves that are legal right now (placement distance, "
    "road connection, costs, whose turn it is, discards, and so on are all handled "
    "for you). Your only job each step is to pick one move from the list you are "
    "given."
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
    "- MARITIME_TRADE (give1, give2, give3, give4, receive): hand the bank the "
    "resources listed in the give slots (empty slots show as 'None') and take the "
    "'receive' resource. Example: (WHEAT, WHEAT, None, None, WOOD) means give "
    "2 wheat, get 1 wood (a 2:1 wheat-port trade).\n"
    "- DISCARD_RESOURCE R: discard one card of resource R (offered only when a 7 "
    "was rolled and you must discard down).\n"
    "- PLAY_KNIGHT_CARD: play a knight (then move the robber and steal). "
    "PLAY_MONOPOLY R: name resource R and take all of it from the opponent. "
    "PLAY_YEAR_OF_PLENTY (R1, R2): take those two resources from the bank. "
    "PLAY_ROAD_BUILDING: place two roads for free.\n"
    "- ROLL, END_TURN, and BUY_DEVELOPMENT_CARD do what their names say."
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
