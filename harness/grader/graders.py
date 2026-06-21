"""Run a single LLM grader over ONE whole game (one call), returning a per-decision,
per-criterion verdict for every decision it was asked to score.
"""
from __future__ import annotations

from dataclasses import dataclass

from goldilocks_eval.agents.factory import make_backend
from harness.grader.prompts import (
    SYSTEM, build_game_prompt, parse_game_verdicts,
    SYSTEM_DECISION, build_decision_prompt, parse_one_verdict)
from harness.grader.taxonomy import CRITERIA_BY_TYPE

# Whole-game verdicts list is large; raise the cap so it isn't truncated.
GRADER_MAX_TOKENS = 4096


@dataclass
class Verdict:
    grader: str
    criteria: dict           # name -> {"score": int, "failed": bool, "reason": str}
    summary: str = ""
    ok: bool = True


def _full_criteria(dtype: str, parsed_criteria: list) -> dict:
    """Index parsed criteria by name; fill any omitted as a pass (the aggregator
    denominator needs every criterion present)."""
    by_name = {c.get("name"): c for c in (parsed_criteria or []) if isinstance(c, dict)}
    out = {}
    for name in CRITERIA_BY_TYPE[dtype]:
        c = by_name.get(name)
        if c is None:
            out[name] = {"score": 2, "failed": False, "reason": "(not returned)"}
        else:
            out[name] = {"score": int(c.get("score", 2)),
                         "failed": bool(c.get("failed", False)),
                         "reason": str(c.get("reason", ""))[:200]}
    return out


def grade_game(backend, transcript: dict, selected, decisions_by_ply: dict) -> dict:
    """One call. Returns {ply: Verdict} for every selected decision.

    On a failed/garbled call, returns all-pass verdicts (failed=false) so the run
    continues and the denominator stays intact (those decisions just contribute no
    failures from this grader)."""
    by_ply = {r.ply: r for r in selected}
    try:
        raw = backend.complete(SYSTEM, build_game_prompt(transcript, selected, decisions_by_ply))
        parsed = parse_game_verdicts(raw)
    except Exception:
        parsed = None

    out = {}
    for ply, r in by_ply.items():
        v = (parsed or {}).get(ply)
        crit = _full_criteria(r.decision_type, v.get("criteria") if v else None)
        summary = (v.get("summary", "") if v else "")
        out[ply] = Verdict(backend.name, crit, str(summary)[:200], ok=v is not None)
    return out


def grade_decision(backend, transcript: dict, regret, decision: dict | None) -> Verdict:
    """Hybrid path: score ONE decision in its own call (full attention) with compact
    game context. Designed to fan out across all decisions in a global parallel pool."""
    try:
        raw = backend.complete(SYSTEM_DECISION, build_decision_prompt(transcript, regret, decision))
        obj = parse_one_verdict(raw)
    except Exception:
        obj = None
    crit = _full_criteria(regret.decision_type, obj.get("criteria") if obj else None)
    return Verdict(backend.name, crit, str(obj.get("summary", "") if obj else "")[:200], ok=obj is not None)


def make_grader(spec: str):
    """Build a grader backend (claude:… / openai:…); temperature 0, larger max_tokens
    for the whole-game verdict list."""
    b = make_backend(spec)
    try:
        b.max_tokens = GRADER_MAX_TOKENS
    except Exception:
        pass
    return b
