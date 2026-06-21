"""Run a single LLM grader (claude:… / openai:…) over one decision and return a
per-criterion verdict. Optional self-consistency: majority `failed` per criterion.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from goldilocks_eval.agents.factory import make_backend
from harness.grader.prompts import SYSTEM, build_user_prompt, parse_criteria
from harness.grader.taxonomy import CRITERIA_BY_TYPE


@dataclass
class Verdict:
    grader: str
    criteria: dict           # name -> {"score": int, "failed": bool, "reason": str}
    summary: str = ""
    ok: bool = True


def _full_criteria(dtype: str, parsed: list) -> dict:
    """Index parsed criteria by name; fill any the grader omitted as a pass (score 2),
    so every criterion is present (the aggregator denominator needs all of them)."""
    by_name = {c["name"]: c for c in parsed}
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


def grade_once(backend, regret, decision: dict | None) -> Verdict:
    dtype = regret.decision_type
    try:
        raw = backend.complete(SYSTEM, build_user_prompt(regret, decision))
    except Exception as e:
        return Verdict(backend.name, _full_criteria(dtype, []), f"(grader error: {e})", ok=False)
    obj = parse_criteria(raw, dtype)
    if obj is None:
        return Verdict(backend.name, _full_criteria(dtype, []), "(unparseable)", ok=False)
    return Verdict(backend.name, _full_criteria(dtype, obj["criteria"]), obj["summary"])


def grade(backend, regret, decision: dict | None, samples: int = 1) -> Verdict:
    if samples <= 1:
        return grade_once(backend, regret, decision)
    vs = [grade_once(backend, regret, decision) for _ in range(samples)]
    good = [v for v in vs if v.ok] or vs
    # majority `failed` per criterion across samples
    merged = {}
    for name in CRITERIA_BY_TYPE[regret.decision_type]:
        fails = sum(1 for v in good if v.criteria[name]["failed"])
        failed = fails > len(good) / 2
        rep = next((v.criteria[name] for v in good if v.criteria[name]["failed"] == failed),
                   good[0].criteria[name])
        merged[name] = {"score": rep["score"], "failed": failed, "reason": rep["reason"]}
    return Verdict(good[0].grader + f"×{samples}", merged, good[0].summary)


def make_grader(spec: str):
    """Build a grader backend (claude:… / openai:…); both default to temperature 0."""
    return make_backend(spec)
