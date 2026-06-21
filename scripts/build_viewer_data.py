#!/usr/bin/env python
"""Turn a game transcript (.json) into a compact replay file for the viewer.

Reconstructs per-step board state by replaying the recorded `action_records`
(NOT by re-simulating the seed) — so it works for LLM games too, which aren't
reproducible from a seed. Node/hex pixel positions come from
`goldilocks_eval.geometry` (the same math Catanatron's own renderer uses), so the
viewer just plots the coordinates we emit.

    .venv/bin/python scripts/build_viewer_data.py transcripts/sample/batch/seed1_norm.json
    # -> writes transcripts/sample/batch/seed1_norm.view.json

Pass a directory to build every *.json transcript under it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goldilocks_eval.geometry import tile_center, node_position


def _resolve(value):
    """action_records store clean values; decisions store stringified ones."""
    return value


def build_view(transcript_path: Path) -> dict:
    d = json.loads(transcript_path.read_text())
    g = d["game"]
    decisions = d.get("decisions", [])

    # --- static board geometry (unit-scaled; viewer applies size + center) ---
    tiles = []
    desert_coord = None
    for t in g["tiles"]:
        coord = tuple(t["coordinate"])
        tile = t.get("tile", {}) or {}
        ttype = tile.get("type")
        cx, cy = tile_center(coord)
        if ttype == "DESERT" and desert_coord is None:
            desert_coord = list(coord)          # robber starts on the desert
        port = None
        if ttype == "PORT":
            port = tile.get("resource") or "3:1"   # 2:1 resource port, else generic
        tiles.append({
            "coord": list(coord), "x": cx, "y": cy,
            "type": ttype,
            "resource": tile.get("resource"),
            "number": tile.get("number"),
            "port": port,
        })

    nodes = {}
    for nid in g["nodes"]:
        x, y = node_position(g, nid)
        nodes[str(nid)] = {"x": x, "y": y}

    edges = [list(e["id"]) for e in g["edges"]]

    colors = list(g["colors"])
    seats = d.get("seats", {})

    # --- replay action_records into cumulative per-step snapshots ------------
    buildings: dict = {}                 # node_id(str) -> {color, type}
    roads: dict = {}                     # "n1-n2" -> color
    robber = list(desert_coord) if desert_coord else None
    steps = []

    def vp_snapshot():
        vp = {c: 0 for c in colors}
        for b in buildings.values():
            vp[b["color"]] += 2 if b["type"] == "CITY" else 1
        return vp

    records = g["action_records"]
    for i, rec in enumerate(records):
        action = rec[0]
        color, atype, value = action[0], action[1], action[2]
        dice = None
        note = None

        if atype == "BUILD_SETTLEMENT":
            buildings[str(value)] = {"color": color, "type": "SETTLEMENT"}
        elif atype == "BUILD_CITY":
            buildings[str(value)] = {"color": color, "type": "CITY"}
        elif atype == "BUILD_ROAD":
            n1, n2 = value
            roads[f"{min(n1, n2)}-{max(n1, n2)}"] = color
        elif atype == "ROLL":
            dice = list(value) if value else None
        elif atype == "MOVE_ROBBER":
            coord, victim = value[0], value[1]
            robber = list(coord)
            stolen = rec[1]
            if victim:
                note = f"robber → steal from {victim}" + (f" ({stolen})" if stolen else "")

        reasoning = decisions[i].get("reasoning") if i < len(decisions) else None
        turn = decisions[i].get("turn") if i < len(decisions) else None

        steps.append({
            "i": i,
            "turn": turn,
            "color": color,
            "action_type": atype,
            "value": value,
            "dice": dice,
            "note": note,
            "reasoning": reasoning,
            # cumulative board after this action
            "buildings": {k: dict(v) for k, v in buildings.items()},
            "roads": dict(roads),
            "robber": list(robber) if robber else None,
            "vp": vp_snapshot(),
        })

    return {
        "meta": {
            "label": d.get("label"),
            "seed": d.get("seed"),
            "seats": seats,
            "winner": d.get("winning_color"),
            "final_vp": d.get("final_victory_points", {}),
            "num_steps": len(steps),
        },
        "board": {"tiles": tiles, "nodes": nodes, "edges": edges},
        "players": colors,
        "steps": steps,
    }


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    target = Path(argv[0])
    paths = (sorted(target.glob("*.json")) if target.is_dir()
             else [target])
    paths = [p for p in paths if not p.name.endswith(".view.json")]
    if not paths:
        print(f"no transcript .json found at {target}")
        return 1
    for p in paths:
        view = build_view(p)
        out = p.with_suffix(".view.json")
        out.write_text(json.dumps(view))
        print(f"{p.name} -> {out.name}  ({view['meta']['num_steps']} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
