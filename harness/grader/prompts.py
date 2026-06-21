"""Whole-game grading prompt + parsing.

ONE grader call per game (per board): the grader reads a compact timeline of the
whole game for context, then scores a set of that game's decisions — returning a
list of per-decision verdicts in a single response. This is the unit the proposal
calls for (§2 whole-game read, §9 one subagent per transcript): grading a Catan
move needs the game's context, and it keeps calls to ~one per game per grader
instead of one per move.

Both graders get this verbatim (temperature 0) so verdicts are comparable. We do
NOT ask the grader for state_tags — those are derived objectively from the engine
(context.py) so the two graders share identical aggregation buckets.
"""
from __future__ import annotations

import json

from harness.grader.taxonomy import criteria_block, CRITERIA_BY_TYPE, DECISION_TYPES

SYSTEM = (
    "You are a Settlers of Catan decision grader. You are given ONE full 1v1 game "
    "(a compact move timeline for context) and a list of specific decisions from it "
    "to score. For each listed decision you score the player's choice on each "
    "criterion for that decision's type.\n\n"
    "Criterion definitions by decision type:\n"
    f"placement:\n{criteria_block('placement')}\n"
    f"trade:\n{criteria_block('trade')}\n"
    f"build_spend:\n{criteria_block('build_spend')}\n\n"
    "Rules:\n"
    "1. Use the whole-game timeline as context (tempo, who's ahead, how it ended), "
    "but score each decision as of when it was made.\n"
    "2. For each listed decision, score EVERY criterion for its decision_type, using "
    "ONLY those criterion IDs. Scale: 2 = good, 1 = suboptimal but defensible, "
    "0 = clear mistake. Set failed=true ONLY on score 0. Irrelevant criterion -> "
    "score 2, failed=false, reason \"n/a\".\n"
    "3. Judge against the legal options actually available, not an ideal not on the "
    "menu. The oracle's best legal move + regret are given as evidence.\n"
    "4. One-sentence reasons. Output ONLY this JSON object, nothing else:\n"
    '{"verdicts": [{"decision_id": <int>, "criteria": '
    '[{"name": "<criterion id>", "score": 0|1|2, "failed": true|false, '
    '"reason": "<= 1 sentence"}], "summary": "<= 1 sentence"}]}\n'
    "Return one verdict object per listed decision_id, scoring exactly that "
    "decision's criteria. Never invent criterion names."
)


def _fmt_action(a) -> str:
    if isinstance(a, str):
        return a
    if isinstance(a, (list, tuple)) and len(a) >= 3:
        return f"{a[1]} {a[2]}" if a[2] is not None else f"{a[1]}"
    return str(a)


# action types worth showing in the context timeline (skip rolls / end-turn noise)
_TIMELINE_ACTIONS = {
    "BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "MARITIME_TRADE",
    "BUY_DEVELOPMENT_CARD", "MOVE_ROBBER", "PLAY_KNIGHT", "PLAY_MONOPOLY",
    "PLAY_YEAR_OF_PLENTY", "PLAY_ROAD_BUILDING",
}


def _timeline(transcript: dict, max_lines: int = 160) -> str:
    """Compact whole-game story: the meaningful moves with turn + running VP."""
    lines = []
    for d in transcript.get("decisions", []):
        at = d.get("action_type")
        if at not in _TIMELINE_ACTIONS:
            continue
        vp = (d.get("state", {}) or {}).get("vp", {})
        vps = " ".join(f"{c[:1]}{v}" for c, v in vp.items())
        val = d.get("value")
        lines.append(f"t{d.get('turn')} {str(d.get('color'))[:1]} {at}"
                     f"{(' ' + str(val)) if val is not None else ''}  [{vps}]")
    if len(lines) > max_lines:                       # keep head + tail if very long
        head, tail = lines[: max_lines // 2], lines[-max_lines // 2:]
        lines = head + [f"… ({len(lines) - max_lines} moves omitted) …"] + tail
    meta = transcript.get("final_victory_points", {})
    winner = transcript.get("winning_color")
    return ("\n".join(lines)
            + f"\n[final VP {meta} | winner {winner}]")


def build_game_prompt(transcript: dict, selected, decisions_by_ply: dict) -> str:
    """Whole-game context + the decisions to grade (each with local context)."""
    blocks = []
    for r in selected:
        d = decisions_by_ply.get(r.ply, {}) or {}
        st = d.get("state", {}) or {}
        legal = d.get("legal_actions") or []
        legal_str = ", ".join(map(str, legal[:30])) + (
            f", …(+{len(legal) - 30})" if len(legal) > 30 else "")
        blocks.append(
            f"decision_id {r.ply} | type={r.decision_type} | turn {r.turn} | "
            f"player {r.color}\n"
            f"  score criteria: {', '.join(CRITERIA_BY_TYPE[r.decision_type])}\n"
            f"  state tags: {', '.join(r.state_tags)} | VP {st.get('vp', {})} | "
            f"hand {st.get('hand', {})}\n"
            f"  chose: {_fmt_action(r.action)}  | legal ({r.num_legal}): {legal_str}\n"
            f"  reasoning: \"{d.get('reasoning') or '(none)'}\"\n"
            f"  oracle: regret {r.regret_vp:.2f} VP-eq, best legal = {_fmt_action(r.oracle_best)}"
        )
    return (
        f"GAME TIMELINE (context — {transcript.get('label')}):\n{_timeline(transcript)}\n\n"
        f"DECISIONS TO GRADE ({len(selected)}):\n" + "\n\n".join(blocks) + "\n\n"
        f"Return the JSON object with one verdict per decision_id above."
    )


def parse_game_verdicts(text: str) -> dict | None:
    """Pull {verdicts:[...]} and index by decision_id (int ply)."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):] if "{" in s else s
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None
    out = {}
    for v in (obj.get("verdicts", []) if isinstance(obj, dict) else []):
        if isinstance(v, dict) and "decision_id" in v:
            try:
                out[int(v["decision_id"])] = v
            except (TypeError, ValueError):
                continue
    return out or None
