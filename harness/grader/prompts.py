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
    "menu. The oracle's best legal move + regret are a 1-ply value-function HEURISTIC: "
    "good at catching blunders, but myopic — it ignores multi-turn payoff and the "
    "opponent's reply, and it undervalues dev-card buys, knight plays, and roads "
    "toward longest road. Use it as a hint, not ground truth; trust your own judgment "
    "when it conflicts (especially on dev-card / longest-road / setup moves).\n"
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
        _assert_aligned(r, d or None)
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


# ============================================================================
# Hybrid path: ONE call per DECISION (full attention, less compression) that
# still carries compact whole-game CONTEXT, so all decisions can fan out in a
# global parallel pool (boards × decisions) for minimum wall-clock.
# ============================================================================

SYSTEM_DECISION = (
    "You are a Settlers of Catan decision grader. You score ONE decision from a 1v1 "
    "game. You get a compact summary of the whole game (for context) and the recent "
    "moves around the decision, then score the player's choice on each criterion for "
    "this decision's type.\n\n"
    "Criterion definitions:\n"
    f"placement:\n{criteria_block('placement')}\n"
    f"trade:\n{criteria_block('trade')}\n"
    f"build_spend:\n{criteria_block('build_spend')}\n\n"
    "Rules:\n"
    "1. Use the game context (tempo, who won, the race) but score the move as of when "
    "it was made.\n"
    "2. Score EVERY criterion for this decision_type, using ONLY those IDs. Scale: "
    "2 = good, 1 = suboptimal but defensible, 0 = a real mistake on this criterion. "
    "Score 0 (failed=true) when the move is CLEARLY worse on this criterion than an "
    "available legal alternative — you do NOT need it to be the single worst move or a "
    "unanimous blunder; a clearly better legal option existing is enough. Reserve 1 for "
    "genuinely defensible-but-imperfect. Don't fail a move that's actually fine. Give a "
    "real one-sentence reason for each (no 'n/a' unless truly irrelevant); when you score "
    "0, name the specific better legal move.\n"
    "3. Judge against the legal options actually available; the oracle's best legal move "
    "+ regret are a 1-ply value-function HEURISTIC (catches blunders but is myopic — "
    "ignores multi-turn payoff/opponent reply and undervalues dev-card, knight, and "
    "longest-road plays). Use it as a hint, not ground truth; trust your own judgment "
    "when it conflicts.\n"
    "4. Output ONLY this JSON, nothing else:\n"
    '{"criteria": [{"name": "<id>", "score": 0|1|2, "failed": true|false, '
    '"reason": "<= 1 sentence"}], "summary": "<= 1 sentence"}'
)


def game_context(transcript: dict) -> str:
    """Compact whole-game summary: outcome + a few VP snapshots over time."""
    decs = transcript.get("decisions", [])
    snaps = []
    if decs:
        idxs = [int(k * (len(decs) - 1) / 4) for k in range(5)]   # 5 points across the game
        for i in idxs:
            d = decs[i]
            vp = (d.get("state", {}) or {}).get("vp", {})
            snaps.append(f"t{d.get('turn')}:" + "/".join(f"{c[:1]}{v}" for c, v in vp.items()))
    fvp = transcript.get("final_victory_points", {})
    return (f"Game {transcript.get('label')} | final VP {fvp} | winner "
            f"{transcript.get('winning_color')}\nVP trajectory: " + "  ".join(snaps))


def local_window(transcript: dict, ply: int, before: int = 6, after: int = 3) -> str:
    """The moves immediately around this decision (recent-context window)."""
    decs = transcript.get("decisions", [])
    # ply IS the list position (decisions[] is 1:1 with action_records by position).
    pos = ply if 0 <= ply < len(decs) else None
    if pos is None:
        return ""
    lines = []
    for d in decs[max(0, pos - before): pos + after + 1]:
        mark = " <-- THIS" if d.get("ply", d.get("i")) == ply else ""
        val = d.get("value")
        lines.append(f"  t{d.get('turn')} {str(d.get('color'))[:1]} {d.get('action_type')}"
                     f"{(' ' + str(val)) if val is not None else ''}{mark}")
    return "\n".join(lines)


def _assert_aligned(regret, decision):
    """Loud crash on context misalignment — the bug class that already cost a re-grade.
    The grader prompt must describe the SAME action the oracle scored."""
    assert decision is None or decision.get("action_type") == regret.action_type, (
        f"grader context misalignment at ply {regret.ply}: decision "
        f"{decision.get('action_type')!r} != regret {regret.action_type!r}")


def build_decision_prompt(transcript: dict, regret, decision: dict | None) -> str:
    _assert_aligned(regret, decision)
    d = decision or {}
    st = d.get("state", {}) or {}
    legal = d.get("legal_actions") or []
    legal_str = ", ".join(map(str, legal[:30])) + (
        f", …(+{len(legal) - 30})" if len(legal) > 30 else "")
    return (
        f"GAME CONTEXT:\n{game_context(transcript)}\n\n"
        f"RECENT MOVES:\n{local_window(transcript, regret.ply)}\n\n"
        f"DECISION TO SCORE  (type={regret.decision_type}; "
        f"score: {', '.join(CRITERIA_BY_TYPE[regret.decision_type])})\n"
        f"- turn {regret.turn}, player {regret.color} | state tags: {', '.join(regret.state_tags)}\n"
        f"- VP {st.get('vp', {})} | hand {st.get('hand', {})}\n"
        f"- chose: {_fmt_action(regret.action)} | legal ({regret.num_legal}): {legal_str}\n"
        f"- reasoning: \"{d.get('reasoning') or '(none)'}\"\n"
        f"- oracle: regret {regret.regret_vp:.2f} VP-eq, best legal = {_fmt_action(regret.oracle_best)}\n\n"
        f"Return the JSON verdict scoring every criterion for this decision."
    )


def parse_one_verdict(text: str) -> dict | None:
    """Parse a single-decision {criteria:[...], summary} object."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`"); s = s[s.find("{"):] if "{" in s else s
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        obj = json.loads(s[a:b + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and "criteria" in obj else None
