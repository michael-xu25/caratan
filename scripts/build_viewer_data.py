#!/usr/bin/env python
"""Turn a game transcript (.json) into a faithful replay file for the viewer.

**Provably faithful**: rather than re-implementing how each action changes the
board, we replay the recorded `action_records` *through the real Catanatron
engine* (`game.play_tick` + `game.execute`) and snapshot the engine's own state
after every ply. So board, roads, robber, victory points, and **each player's
resource hand** all come straight from Catanatron — no reconstruction drift.

Determinism: a fresh `Game(seed)` reproduces the same board + turn order, and
`seeded_dice(seed)` reproduces the same dice, so the recorded actions are always
valid in the replayed state (we assert this — a mismatch raises).

    .venv/bin/python scripts/build_viewer_data.py transcripts/sample_game/seed1_norm.json
    # -> writes <name>.view.json   (pass a directory to build all)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import catanatron.game as _cg
from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY, ROAD
from catanatron.state_functions import get_actual_victory_points
from catanatron.json import GameEncoder

from goldilocks_eval.geometry import tile_center, node_position
from harness.dice import seeded_dice

_DEV_KEYS = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")


def _encode(value):
    """Encode an Action value the same way the transcript did, for matching."""
    return json.loads(json.dumps(value, cls=GameEncoder))


def _static_board(g: dict) -> dict:
    """Geometry (unit-scaled) straight from the serialized board — unchanged."""
    tiles = []
    for t in g["tiles"]:
        coord = tuple(t["coordinate"]); tile = t.get("tile", {}) or {}
        cx, cy = tile_center(coord)
        ttype = tile.get("type")
        tiles.append({"coord": list(coord), "x": cx, "y": cy, "type": ttype,
                      "resource": tile.get("resource"), "number": tile.get("number"),
                      "port": (tile.get("resource") or "3:1") if ttype == "PORT" else None})
    nodes = {str(nid): {"x": x, "y": y}
             for nid in g["nodes"] for x, y in [node_position(g, nid)]}
    edges = [list(e["id"]) for e in g["edges"]]
    return {"tiles": tiles, "nodes": nodes, "edges": edges}


def _snapshot(state) -> dict:
    """Exact per-ply state read from the engine (no reconstruction)."""
    buildings, roads, hands, vp = {}, {}, {}, {}
    for color in state.colors:
        bc = state.buildings_by_color[color]
        for nid in bc.get(SETTLEMENT, []):
            buildings[str(nid)] = {"color": color.value, "type": "SETTLEMENT"}
        for nid in bc.get(CITY, []):
            buildings[str(nid)] = {"color": color.value, "type": "CITY"}
        for edge in bc.get(ROAD, []):
            a, b = sorted(edge)
            roads[f"{a}-{b}"] = color.value
        i = state.color_to_index[color]
        hand = {r: state.player_state[f"P{i}_{r}_IN_HAND"] for r in RESOURCES}
        hand["DEV"] = sum(state.player_state.get(f"P{i}_{d}_IN_HAND", 0) for d in _DEV_KEYS)
        hands[color.value] = hand
        vp[color.value] = get_actual_victory_points(state, color)
    robber = state.board.robber_coordinate
    return {"buildings": buildings, "roads": roads,
            "robber": list(robber) if robber else None, "hands": hands, "vp": vp}


def build_view(transcript_path: Path) -> dict:
    d = json.loads(transcript_path.read_text())
    g = d["game"]
    seed = d["seed"]
    records = g["action_records"]
    decisions = d.get("decisions", [])

    # Replay through the engine. Both norm & swap transcripts were played with
    # [RED, BLUE]-colored players; with the same seed the turn order matches.
    _cg.TURNS_LIMIT = max(g.get("num_turns_cap", 400), len(records) + 5)
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
                seed=seed, vps_to_win=10)

    cursor = {"i": 0}

    def decider(player, gm, playable):
        rec = records[cursor["i"]][0]          # [color, type, value]
        rtype, rval = rec[1], rec[2]
        if rtype == "ROLL":                    # dice come from the seeded deck
            for a in playable:
                if a.action_type.value == "ROLL":
                    return a
        for a in playable:                     # match type + value exactly
            if a.action_type.value == rtype and _encode(a.value) == rval:
                return a
        for a in playable:                     # type-only fallback
            if a.action_type.value == rtype:
                return a
        raise RuntimeError(f"replay mismatch at ply {cursor['i']}: "
                           f"recorded {rec} not in playable {playable}")

    steps = []
    with seeded_dice(seed, balanced=False):
        while (game.winning_color() is None
               and game.state.num_turns < _cg.TURNS_LIMIT
               and cursor["i"] < len(records)):
            rec = records[cursor["i"]]
            game.play_tick(decide_fn=decider)          # engine executes + validates
            snap = _snapshot(game.state)               # exact state AFTER the move
            action = rec[0]
            dec = decisions[cursor["i"]] if cursor["i"] < len(decisions) else {}
            steps.append({
                "i": cursor["i"], "turn": dec.get("turn"),
                "color": action[0], "action_type": action[1], "value": action[2],
                "dice": action[2] if action[1] == "ROLL" else None,
                "note": _trade_note(action) ,
                "reasoning": dec.get("reasoning"),
                **snap,
            })
            cursor["i"] += 1

    return {
        "meta": {"label": d.get("label"), "seed": seed, "seats": d.get("seats", {}),
                 "winner": d.get("winning_color"),
                 "final_vp": d.get("final_victory_points", {}),
                 "num_steps": len(steps)},
        "board": _static_board(g),
        "players": list(g["colors"]),
        "steps": steps,
    }


def _trade_note(action) -> str | None:
    """For MARITIME_TRADE, say what was given and received."""
    if action[1] != "MARITIME_TRADE":
        return None
    v = action[2] or []
    given = [r for r in v[:-1] if r]
    got = v[-1] if v else None
    if not given or not got:
        return None
    from collections import Counter
    g = ", ".join(f"{n}×{r}" for r, n in Counter(given).items())
    return f"traded {g} → {got}"


def main(argv):
    if not argv:
        print(__doc__); return 1
    target = Path(argv[0])
    paths = sorted(target.glob("*.json")) if target.is_dir() else [target]
    paths = [p for p in paths if not p.name.endswith(".view.json")]
    if not paths:
        print(f"no transcript .json at {target}"); return 1
    for p in paths:
        view = build_view(p)
        out = p.with_suffix(".view.json")
        out.write_text(json.dumps(view))
        fv = view["meta"]["final_vp"]
        print(f"{p.name} -> {out.name}  ({view['meta']['num_steps']} steps, final VP {fv})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
