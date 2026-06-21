"""Canonical prompt / answer / reward contract for placement scenarios.

ONE source of truth, imported by all three consumers so their numbers can't
drift (the seam the scenario-generation-spec flags hardest):

    - generation      (producer side: builds scenarios, may render for sanity)
    - calibration     (calibration_harness.py: base_solve_rate sampling)
    - eval            (goldilocks_eval/scenario.py: before/after scoring)

Contract (matches scenario-generation-spec.md):
    - node ids are canonical strings "node_<int>" (Catanatron actions use raw
      ints; normalize at the boundary with `node_id_str` / `node_id_int`).
    - the model must reply:
          <reasoning>...</reasoning>
          <answer>node_27</answer>
    - reward is tiered: 1.0 gold / 0.5 acceptable / 0.0 else (incl. unparseable).

This is the placement (settlement-choice) contract. Live full-game play uses a
different surface (`goldilocks_eval/prompt.py`, index-based over arbitrary
actions) — don't conflate them.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, List, Mapping, Optional

# Pip count (number of dot-probabilities) for each dice number.
# Pip count = dots printed under a number = ways two dice can make it. Board fact.
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

# Reuse the SAME 1v1 rules primer the model plays with (goldilocks_eval.prompt),
# so placement grading and live play share game knowledge — then the placement-
# specific task + answer format. NOTE: keep this mechanics-only — no hint about
# what makes a spot good (that judgment is the reward/rubric, never the prompt).
from goldilocks_eval.prompt import CATAN_RULES  # noqa: E402

SYSTEM = CATAN_RULES + "\n\n" + (
    "TASK: you are choosing where to place an opening settlement. Follow the "
    "response format exactly: a <reasoning> block, then an <answer> block "
    "containing exactly one node id."
)


# ---------------------------------------------------------------- node ids ----
def node_id_str(n: Any) -> str:
    s = str(n)
    return s if s.startswith("node_") else f"node_{s}"


def node_id_int(s: Any) -> int:
    s = str(s)
    return int(s[len("node_"):]) if s.startswith("node_") else int(s)


@lru_cache(maxsize=1)
def _node_to_coords() -> dict:
    """node_id(int) -> tuple of adjacent tile coordinates. Seed-independent for
    the base map (seed changes resource/number assignment, not topology)."""
    from catanatron import Color, Game, RandomPlayer

    g = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=0)
    mapping: dict = {}
    for coord, tile in g.state.board.map.land_tiles.items():
        for node_id in tile.nodes.values():
            mapping.setdefault(node_id, set()).add(tuple(coord))
    return {k: tuple(sorted(v)) for k, v in mapping.items()}


# ----------------------------------------------------------------- render ----
def _tiles_by_coord(serialized_state: Mapping) -> dict:
    out = {}
    for entry in serialized_state.get("tiles", []) or []:
        out[tuple(entry["coordinate"])] = entry.get("tile", {})
    return out


def _node_summary(node_int: int, tiles_by_coord: dict) -> str:
    parts = []
    for coord in _node_to_coords().get(node_int, ()):  # adjacent tiles
        t = tiles_by_coord.get(coord, {})
        res, num = t.get("resource"), t.get("number")
        if res and num:  # skip desert / non-resource
            parts.append(f"{res} on {num} ({PIPS.get(num, 0)} pips)")
    body = ", ".join(parts) if parts else "no production"
    return f"{node_id_str(node_int)}: {body}"


def render_board(serialized_state: Mapping,
                 legal_actions: Optional[List[Any]] = None) -> str:
    tiles_by_coord = _tiles_by_coord(serialized_state)
    lines = ["Tiles (resource, the dice total it produces on, and that number's "
             "pips = dots on the board):"]
    for coord, t in sorted(tiles_by_coord.items()):
        res, num = t.get("resource"), t.get("number")
        if res and num:
            lines.append(f"  {coord} {res} on {num} ({PIPS.get(num, 0)} pips)")
        elif t.get("type") == "RESOURCE_TILE":
            lines.append(f"  {coord} DESERT")

    # Existing buildings.
    built = []
    for nid, node in (serialized_state.get("nodes") or {}).items():
        if node.get("building"):
            built.append(f"{node_id_str(nid)} {node['building']} {node.get('color')}")
    lines.append("Existing buildings: " + ("; ".join(built) if built else "none"))
    if serialized_state.get("robber_coordinate") is not None:
        lines.append(f"Robber: {tuple(serialized_state['robber_coordinate'])}")

    if legal_actions:
        lines.append("Legal nodes (the resources each would collect):")
        for a in legal_actions:
            lines.append("  " + _node_summary(node_id_int(a), tiles_by_coord))
    return "\n".join(lines)


# ----------------------------------------------------------------- prompt ----
def build_prompt(scenario: Mapping) -> str:
    """Self-contained prompt. Works as a single prompt (calibration) or as the
    user message paired with `SYSTEM` (eval). Accepts a dict or any mapping
    with serialized_state / legal_actions."""
    legal = [node_id_str(a) for a in scenario["legal_actions"]]
    board = render_board(scenario.get("serialized_state") or {}, legal)
    return (
        "Choose where to place an opening settlement in this 1v1 game.\n\n"
        f"{board}\n\n"
        f"Legal settlement nodes: {', '.join(legal)}\n\n"
        "Reason, then answer.\n"
        "Respond EXACTLY as:\n"
        "<reasoning>your reasoning</reasoning>\n"
        "<answer>node_ID</answer>"
    )


# ----------------------------------------------------------------- parse ------
_ANS = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def parse_answer(text: str) -> Optional[str]:
    """Extract the chosen node id, normalized to 'node_<int>'. None if no
    parseable <answer> tag."""
    m = _ANS.search(text or "")
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        return node_id_str(node_id_int(raw))  # normalize "27" / "node_27" alike
    except (ValueError, TypeError):
        return raw or None


# ----------------------------------------------------------------- reward -----
def score(answer: Optional[str], gold: Any,
          acceptable: Optional[List[Any]] = None) -> float:
    """Tiered reward. 1.0 gold / 0.5 acceptable / 0.0 else (incl. unparseable)."""
    if answer is None:
        return 0.0
    gold_n = node_id_str(gold)
    acc_n = {node_id_str(a) for a in (acceptable or [])}
    ans_n = node_id_str(answer)
    if ans_n == gold_n:
        return 1.0
    if ans_n in acc_n:
        return 0.5
    return 0.0
