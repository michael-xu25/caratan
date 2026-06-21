"""Grading prompt + verdict parsing — the grader-spec.md system prompt, scoped to
score the criteria for ONE decision's type.

Both graders get this verbatim (temperature 0) so their per-criterion verdicts are
comparable. We do NOT ask the grader to assign state_tags — those are derived
objectively from the engine (harness/grader/context.py) so the two graders share
identical tags and the aggregator's (criterion, tag) buckets never split on a tag
disagreement. The grader scores criteria; we attach the derived tags + oracle
context ourselves.
"""
from __future__ import annotations

import json

from harness.grader.taxonomy import criteria_block, CRITERIA_BY_TYPE

SYSTEM = (
    "You are a Settlers of Catan decision grader. You receive ONE decision: the game "
    "state, the legal options available, the choice the player made (and its stated "
    "reasoning), and an objective engine oracle's view of how much value the choice gave "
    "up. You score the choice on each listed criterion.\n\n"
    "Rules:\n"
    "1. Score EVERY criterion listed for this decision, using ONLY those criterion IDs.\n"
    "   Scale: 2 = good / no issue, 1 = suboptimal but defensible, 0 = clear mistake.\n"
    "   Set failed=true ONLY on score 0. If a criterion is not relevant to this specific "
    "decision, score 2, failed=false, reason \"n/a\".\n"
    "2. Judge against the legal options actually available, not an ideal that wasn't on "
    "the menu. The oracle's best legal move and the regret are given as evidence.\n"
    "3. Keep every reason to one sentence.\n"
    "4. Output ONLY a JSON object, no prose, no markdown:\n"
    '{"criteria": [{"name": "<criterion id>", "score": 0|1|2, "failed": true|false, '
    '"reason": "<= 1 sentence"}], "summary": "<= 1 sentence"}\n'
    "Never invent criterion names. Score exactly the criteria you are given, all of them."
)


def _fmt_action(a) -> str:
    if isinstance(a, str):
        return a
    if isinstance(a, (list, tuple)) and len(a) >= 3:
        return f"{a[1]} {a[2]}" if a[2] is not None else f"{a[1]}"
    return str(a)


def build_user_prompt(regret, decision: dict | None) -> str:
    decision = decision or {}
    state = decision.get("state", {}) or {}
    legal = decision.get("legal_actions") or []
    reasoning = decision.get("reasoning") or "(no reasoning captured)"
    legal_str = ", ".join(map(str, legal[:40])) + (
        f", … (+{len(legal) - 40} more)" if len(legal) > 40 else "")

    return (
        f"DECISION TYPE: {regret.decision_type}\n"
        f"Score these criteria (all of them):\n{criteria_block(regret.decision_type)}\n\n"
        f"STATE (objective tags: {', '.join(regret.state_tags)})\n"
        f"- turn {regret.turn}, acting player {regret.color} | VP {state.get('vp', {})}\n"
        f"- hand {state.get('hand', {})} | longest_road={state.get('longest_road')} "
        f"dev_cards={state.get('dev_cards')} robber={state.get('robber')}\n\n"
        f"THE DECISION\n"
        f"- chose: {_fmt_action(regret.action)}\n"
        f"- legal options ({regret.num_legal}): {legal_str}\n"
        f"- stated reasoning: \"{reasoning}\"\n\n"
        f"ORACLE (objective engine)\n"
        f"- regret of this move: {regret.regret_vp:.3f} VP-equivalents (higher = more value given up)\n"
        f"- oracle's best legal move: {_fmt_action(regret.oracle_best)}\n\n"
        f"Return the JSON verdict scoring every listed criterion."
    )


def parse_criteria(text: str, dtype: str) -> dict | None:
    """Pull {criteria:[...], summary} out of a grader reply and keep only valid
    criterion IDs for this decision_type."""
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
    if not isinstance(obj, dict) or "criteria" not in obj:
        return None
    valid = set(CRITERIA_BY_TYPE[dtype])
    crit = [c for c in obj.get("criteria", [])
            if isinstance(c, dict) and c.get("name") in valid]
    return {"criteria": crit, "summary": str(obj.get("summary", "")).strip()} if crit else None
