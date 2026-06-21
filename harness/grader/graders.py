"""Run a single LLM grader and parse its verdict into a normalized record.

A grader is just an `LLMBackend` (claude:… / openai:…) plus the shared prompt.
Optional self-consistency: sample N times and take the majority category (use it
on the highest-regret decisions where robustness matters).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from goldilocks_eval.agents.factory import make_backend
from harness.grader.prompts import SYSTEM, build_user_prompt, parse_verdict
from harness.grader.taxonomy import normalize_label


@dataclass
class Verdict:
    grader: str                       # backend name, e.g. "claude:claude-opus-4-8"
    category: str                     # normalized taxonomy label
    reasoning_consistency: Optional[float]
    confidence: Optional[float]
    explanation: str
    ok: bool = True                   # False if the call/parse failed
    samples: list[str] = field(default_factory=list)  # categories if self-consistency


def _clip01(x) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return None


def grade_once(backend, regret, decision: dict | None) -> Verdict:
    user = build_user_prompt(regret, decision)
    try:
        raw = backend.complete(SYSTEM, user)
    except Exception as e:
        return Verdict(backend.name, "none", None, None, f"(grader error: {e})", ok=False)
    obj = parse_verdict(raw)
    if obj is None:
        return Verdict(backend.name, "none", None, None, "(unparseable verdict)", ok=False)
    return Verdict(
        grader=backend.name,
        category=normalize_label(obj.get("category")),
        reasoning_consistency=_clip01(obj.get("reasoning_consistency")),
        confidence=_clip01(obj.get("confidence")),
        explanation=str(obj.get("explanation", "")).strip(),
    )


def grade(backend, regret, decision: dict | None, samples: int = 1) -> Verdict:
    """Grade a decision; with samples>1 take the majority category (self-consistency)."""
    if samples <= 1:
        return grade_once(backend, regret, decision)
    verdicts = [grade_once(backend, regret, decision) for _ in range(samples)]
    good = [v for v in verdicts if v.ok] or verdicts
    cats = [v.category for v in good]
    winner_cat = Counter(cats).most_common(1)[0][0]
    rep = next(v for v in good if v.category == winner_cat)  # representative verdict
    rep.samples = cats
    return rep


def make_grader(spec: str):
    """Build a grader backend from a spec (claude:… / openai:…). Temperature is 0
    in those backends by default — keep it deterministic for grading."""
    return make_backend(spec)
