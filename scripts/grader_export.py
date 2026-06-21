#!/usr/bin/env python
"""Compact, grader-ready export of a game transcript.

The raw `<label>.json` is great for machines but heavy (it embeds the full
GameEncoder board dump). For the GRADER (an LLM reading a whole game to find
failure modes) we want something easy to read and self-contained:

  - the board summarized ONCE (resources/numbers/ports) — the static context
  - one entry per *real decision* (skips forced moves) carrying exactly the
    counterfactual the grader needs: turn, phase, whose move, the decision-time
    state (VP, hand, dev count, longest road, robber), the FULL legal set it
    chose among, what it chose, and its stated reasoning.

This reads straight from the enriched `decisions[]` that the runner captures
LIVE at each decision (no replay, no RNG) — so it's faithful by construction.

    .venv/bin/python scripts/grader_export.py transcripts/<run>/<label>.json
    # -> writes <label>.grader.json  (pass a dir to do all)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _board_summary(game: dict) -> dict:
    """Static board context: land tiles (resource@number) + ports."""
    tiles, ports = [], []
    for t in game.get("tiles", []):
        tile = t.get("tile", {}) or {}
        ty = tile.get("type")
        if ty == "RESOURCE_TILE":
            tiles.append({"coord": t["coordinate"], "resource": tile.get("resource"),
                          "number": tile.get("number")})
        elif ty == "DESERT":
            tiles.append({"coord": t["coordinate"], "resource": "DESERT", "number": None})
        elif ty == "PORT":
            ports.append({"coord": t["coordinate"], "kind": tile.get("resource") or "3:1",
                          "direction": tile.get("direction")})
    return {"tiles": tiles, "ports": ports}


def export(transcript_path: Path) -> dict:
    d = json.loads(transcript_path.read_text())
    decisions = d.get("decisions", [])
    # real choices = the analyzable decisions (more than one legal option)
    enriched = [e for e in decisions if e.get("num_legal", 0) and e["num_legal"] > 1
                and "legal_actions" in e]
    out_decisions = [{
        "ply": e.get("ply"), "turn": e.get("turn"), "phase": e.get("phase"),
        "player": e.get("color"),
        "state": e.get("state"),               # vp, hand, dev_cards, longest_road, robber
        "chose": e.get("chosen"),
        "options": e.get("legal_actions"),      # the full legal menu it picked from
        "num_options": e.get("num_legal"),
        "reasoning": e.get("reasoning") or "",
        "fell_back": e.get("fell_back", False),
    } for e in enriched]
    return {
        "meta": {
            "label": d.get("label"), "seed": d.get("seed"),
            "seats": d.get("seats", {}), "winner": d.get("winning_color"),
            "final_vp": d.get("final_victory_points", {}),
            "total_plies": len(decisions), "real_decisions": len(out_decisions),
            "enriched": bool(out_decisions),
        },
        "board": _board_summary(d.get("game", {})),
        "decisions": out_decisions,
    }


def main(argv):
    if not argv:
        print(__doc__); return 1
    target = Path(argv[0])
    paths = (sorted(p for p in target.glob("*.json")
                    if not p.name.endswith((".view.json", ".grader.json")))
             if target.is_dir() else [target])
    if not paths:
        print(f"no transcript .json at {target}"); return 1
    for p in paths:
        data = export(p)
        out = p.with_suffix(".grader.json")
        out.write_text(json.dumps(data, indent=1))
        m = data["meta"]
        note = "" if m["enriched"] else "  ⚠️ stale transcript (no enriched decisions — re-run)"
        print(f"{p.name} -> {out.name}  ({m['real_decisions']} real decisions){note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
