"""Grading prompt + verdict schema, shared verbatim by BOTH graders.

The graders never decide *whether* a move was a mistake (the regret oracle already
did, objectively). They receive the oracle's verdict as context and only:
  - classify the failure into the closed-set taxonomy,
  - judge whether the model's stated reasoning justified the action,
  - rate their own confidence,
  - explain in one or two Catan sentences.

Identical prompt + temperature 0 for both graders so their answers are comparable
and merges are measurable.
"""
from __future__ import annotations

import json

from harness.grader.taxonomy import taxonomy_block, LABELS

SYSTEM = (
    "You are an expert Settlers of Catan analyst grading ONE decision from a "
    "1v1 game. An objective engine oracle has already determined this move gave "
    "up value versus the best legal move — you do NOT decide whether it was a "
    "mistake. Your job is to CATEGORIZE why, using only the fixed taxonomy.\n\n"
    "Failure-mode taxonomy (choose exactly one label):\n"
    f"{taxonomy_block()}\n\n"
    "Reply with ONLY a JSON object, no prose outside it:\n"
    '{"category": "<one taxonomy label>", '
    '"reasoning_consistency": <0.0-1.0>, '
    '"confidence": <0.0-1.0>, '
    '"explanation": "<one or two sentences in Catan terms>"}\n'
    "- category MUST be one of the listed labels exactly.\n"
    "- reasoning_consistency: did the player's stated reasoning justify the action? "
    "1.0 = fully justified, 0.0 = contradicts the action / absent. Use 'none' for "
    "category if the oracle's regret looks like a value-function artifact, not a real error."
)


def _fmt_action(a) -> str:
    """[color, type, value] or 'TYPE value' string -> readable."""
    if isinstance(a, str):
        return a
    if isinstance(a, (list, tuple)) and len(a) >= 3:
        return f"{a[1]} {a[2]}" if a[2] is not None else f"{a[1]}"
    return str(a)


def build_user_prompt(regret, decision: dict | None) -> str:
    """Assemble the decision context for the graders from the oracle result and
    the matching transcript decision (reasoning, hand, VP, legal options)."""
    decision = decision or {}
    state = decision.get("state", {}) or {}
    vp = state.get("vp", {})
    hand = state.get("hand", {})
    legal = decision.get("legal_actions") or []
    reasoning = decision.get("reasoning") or "(no reasoning captured)"

    legal_str = ", ".join(map(str, legal[:40]))
    if len(legal) > 40:
        legal_str += f", … (+{len(legal) - 40} more)"

    return (
        f"GAME CONTEXT\n"
        f"- turn {regret.turn}, ply {regret.ply}, acting player: {regret.color}\n"
        f"- victory points: {vp}\n"
        f"- {regret.color} hand: {hand} | longest_road={state.get('longest_road')} "
        f"dev_cards={state.get('dev_cards')} robber={state.get('robber')}\n\n"
        f"THE DECISION\n"
        f"- chose: {_fmt_action(regret.action)}\n"
        f"- legal options ({regret.num_legal}): {legal_str}\n"
        f"- model's stated reasoning: \"{reasoning}\"\n\n"
        f"ORACLE (objective engine)\n"
        f"- this move's regret: {regret.regret_vp:.3f} VP-equivalents "
        f"(positive = value given up)\n"
        f"- oracle's best legal move: {_fmt_action(regret.oracle_best)}\n\n"
        f"Classify the failure into one taxonomy label and return the JSON verdict."
    )


def parse_verdict(text: str) -> dict | None:
    """Robustly pull the JSON verdict object out of a grader reply."""
    if not text:
        return None
    s = text.strip()
    # strip code fences
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s[3:] else s.strip("`")
        s = s[s.find("{"):] if "{" in s else s
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
