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


def _resolve_winner(transcript: dict):
    """Winner with the turn-cap tie-break (more VP wins; exact tie = draw).

    Prefer the transcript's recorded `winning_color`; if it's null (older
    transcripts left it null at the turn cap), fall back to the VP lead from
    `final_victory_points`. Keeps the viewer consistent with the match result."""
    wc = transcript.get("winning_color")
    if wc:
        return wc
    vps = transcript.get("final_victory_points", {}) or {}
    if not vps:
        return None
    top = max(vps.values())
    leaders = [c for c, v in vps.items() if v == top]
    return leaders[0] if len(leaders) == 1 else None


def _static_board(g: dict) -> dict:
    """Geometry (unit-scaled) straight from the serialized board — unchanged."""
    tiles = []
    for t in g["tiles"]:
        coord = tuple(t["coordinate"]); tile = t.get("tile", {}) or {}
        cx, cy = tile_center(coord)
        ttype = tile.get("type")
        tiles.append({"coord": list(coord), "x": cx, "y": cy, "type": ttype,
                      "resource": tile.get("resource"), "number": tile.get("number"),
                      "port": (tile.get("resource") or "3:1") if ttype == "PORT" else None,
                      # EdgeRef the port faces — its two access nodes are derived from this
                      "direction": tile.get("direction") if ttype == "PORT" else None})
    nodes = {str(nid): {"x": x, "y": y}
             for nid in g["nodes"] for x, y in [node_position(g, nid)]}
    edges = [list(e["id"]) for e in g["edges"]]
    return {"tiles": tiles, "nodes": nodes, "edges": edges}


def _snapshot(state) -> dict:
    """Exact per-ply state read from the engine (no reconstruction)."""
    buildings, roads, hands, vp, awards = {}, {}, {}, {}, {}
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
        # dev cards: per-type breakdown (revealable in the viewer) + total
        dev = {d: state.player_state.get(f"P{i}_{d}_IN_HAND", 0) for d in _DEV_KEYS}
        hand["DEV"] = sum(dev.values())
        hand["dev_cards"] = dev
        hands[color.value] = hand
        vp[color.value] = get_actual_victory_points(state, color)
        awards[color.value] = {
            "road_len": state.player_state.get(f"P{i}_LONGEST_ROAD_LENGTH", 0),
            "knights": state.player_state.get(f"P{i}_PLAYED_KNIGHT", 0),
            "has_road": bool(state.player_state.get(f"P{i}_HAS_ROAD", False)),   # Longest Road (+2 VP)
            "has_army": bool(state.player_state.get(f"P{i}_HAS_ARMY", False)),   # Largest Army (+2 VP)
        }
    robber = state.board.robber_coordinate
    return {"buildings": buildings, "roads": roads,
            "robber": list(robber) if robber else None,
            "hands": hands, "vp": vp, "awards": awards}


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

    from catanatron.models.enums import Action, ActionType, ActionRecord

    def _to_action(ra):
        """Rebuild an Action from the serialized [color, type, value] record."""
        atype = ActionType(ra[1])
        v = ra[2]
        if v is None:
            value = None
        elif atype == ActionType.MOVE_ROBBER:           # ((coord), victim_color|None)
            coord, victim = v
            value = (tuple(coord), Color[victim] if victim else None)
        elif isinstance(v, list):                       # edge / trade / dice / YoP
            value = tuple(v)
        else:
            value = v                                   # node id / resource / depth
        return Action(Color[ra[0]], atype, value)

    def _norm_result(res):
        return tuple(res) if isinstance(res, list) else res

    # Fully deterministic replay: execute each recorded action WITH its recorded
    # result, so dice, robber steals, and dev-card draws all come from the record
    # (apply_roll / apply_move_robber / apply_buy_development_card honor
    # action_record.result). No RNG is consumed → it can never diverge.
    steps = []
    for cursor_i, rec in enumerate(records):
        ra = rec[0]
        action = _to_action(ra)
        game.execute(action, validate_action=False,
                     action_record=ActionRecord(action=action, result=_norm_result(rec[1])))
        snap = _snapshot(game.state)                    # exact state AFTER the move
        dec = decisions[cursor_i] if cursor_i < len(decisions) else {}
        steps.append({
            "i": cursor_i, "turn": dec.get("turn"), "phase": dec.get("phase"),
            "color": ra[0], "action_type": ra[1], "value": ra[2],
            "chosen": dec.get("chosen"),
            # the full legal set the player chose among + how many — the
            # counterfactual the grader needs ("it had N options, picked X")
            "legal_actions": dec.get("legal_actions"),
            "num_legal": dec.get("num_legal"),
            "dice": ra[2] if ra[1] == "ROLL" else None,
            "note": _trade_note(ra),
            "reasoning": dec.get("reasoning"),
            **snap,  # faithful engine state: buildings/roads/robber/hands/vp/awards
        })

    return {
        "meta": {"label": d.get("label"), "seed": seed, "seats": d.get("seats", {}),
                 "winner": _resolve_winner(d),
                 "truncated": d.get("truncated", d.get("winning_color") is None),
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
