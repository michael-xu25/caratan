"""Merge two graders' verdicts into one finding, and measure their agreement.

The recommended reconciliation (grading-rubric-proposal.md §8):
  1. categories match            -> CONSENSUS (high confidence)
  2. disagree                    -> oracle-anchored tie-break (free: no LLM call)
  3. still unresolved + judge    -> a third judge backend picks (optional)
Numeric fields (reasoning_consistency, confidence) are averaged; the spread is
kept as an uncertainty signal. Both raw verdicts are always retained.

Agreement is reported with Cohen's kappa (chance-corrected), not raw % — and
per-category so the taxonomy itself can be audited (low-kappa labels are fuzzy).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from harness.grader.graders import Verdict, grade_once
from harness.grader.prompts import _fmt_action

# Action-type -> taxonomy labels the oracle's best move makes most plausible.
# Used to break ties objectively before spending a judge call.
_ORACLE_HINT = {
    "BUILD_SETTLEMENT": {"placement-low-pip", "placement-no-variety", "boxed-in"},
    "BUILD_CITY": {"placement-low-pip", "inefficient-hand"},
    "BUILD_ROAD": {"boxed-in", "longest-road-ignored"},
    "MOVE_ROBBER": {"robber-not-denying-leader", "robber-victim-suboptimal"},
    "MARITIME_TRADE": {"bad-trade", "inefficient-hand"},
    "BUY_DEVELOPMENT_CARD": {"dev-timing", "inefficient-hand"},
    "PLAY_KNIGHT": {"dev-timing", "robber-not-denying-leader"},
}


@dataclass
class Finding:
    ply: int
    turn: Optional[int]
    color: str
    action: list
    action_type: str
    regret: float
    regret_vp: float
    oracle_best: list
    category: str                       # the merged label
    agreement: str                      # consensus | oracle_broke_tie | judge | unresolved
    reasoning_consistency: Optional[float]
    confidence: Optional[float]
    score_spread: Optional[float]       # |claude.conf - openai.conf|, uncertainty signal
    explanation: str
    graders: dict = field(default_factory=dict)  # raw per-grader verdicts
    game_id: str = ""


def _avg(*xs):
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def reconcile(regret, va: Verdict, vb: Verdict, judge_backend=None,
              decision: dict | None = None) -> Finding:
    rc = _avg(va.reasoning_consistency, vb.reasoning_consistency)
    conf = _avg(va.confidence, vb.confidence)
    spread = (abs(va.confidence - vb.confidence)
              if va.confidence is not None and vb.confidence is not None else None)
    base = dict(
        ply=regret.ply, turn=regret.turn, color=regret.color, action=regret.action,
        action_type=regret.action_type, regret=regret.regret, regret_vp=regret.regret_vp,
        oracle_best=regret.oracle_best, reasoning_consistency=rc, confidence=conf,
        score_spread=spread, graders={va.grader: asdict(va), vb.grader: asdict(vb)},
    )

    if va.category == vb.category:
        return Finding(category=va.category, agreement="consensus",
                       explanation=va.explanation or vb.explanation, **base)

    # disagree -> oracle-anchored tie-break (free)
    plausible = _ORACLE_HINT.get(regret.oracle_best[1], set()) | _ORACLE_HINT.get(regret.action_type, set())
    a_ok, b_ok = va.category in plausible, vb.category in plausible
    if a_ok != b_ok:
        win = va if a_ok else vb
        return Finding(category=win.category, agreement="oracle_broke_tie",
                       explanation=win.explanation, **base)

    # still tied -> optional judge backend picks between the two
    if judge_backend is not None:
        pick = _judge(judge_backend, regret, va, vb, decision)
        if pick is not None:
            win = va if pick == va.category else vb
            return Finding(category=win.category, agreement="judge",
                           explanation=win.explanation, **base)

    # unresolved: keep the more-confident grader's label, flag it
    win = va if (va.confidence or 0) >= (vb.confidence or 0) else vb
    return Finding(category=win.category, agreement="unresolved",
                   explanation=win.explanation, **base)


def _judge(backend, regret, va: Verdict, vb: Verdict, decision) -> Optional[str]:
    sys = ("You are the deciding judge between two Catan analysts who disagree on "
           "the failure category for one move. Pick the single better-fitting label. "
           'Reply ONLY as JSON: {"category": "<one of the two>"}.')
    user = (f"Move: {_fmt_action(regret.action)} | oracle best: {_fmt_action(regret.oracle_best)} "
            f"| regret {regret.regret_vp:.3f} VP.\n"
            f"Analyst A says '{va.category}': {va.explanation}\n"
            f"Analyst B says '{vb.category}': {vb.explanation}\n"
            f"Which label fits better, '{va.category}' or '{vb.category}'?")
    try:
        from harness.grader.prompts import parse_verdict
        obj = parse_verdict(backend.complete(sys, user)) or {}
        from harness.grader.taxonomy import normalize_label
        cat = normalize_label(obj.get("category"))
        return cat if cat in (va.category, vb.category) else None
    except Exception:
        return None


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa for two raters over paired categorical labels."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    labels = sorted({c for p in pairs for c in p})
    po = sum(1 for a, b in pairs if a == b) / n
    # expected agreement from marginals
    from collections import Counter
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca.get(l, 0) / n) * (cb.get(l, 0) / n) for l in labels)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def agreement_report(findings: list[Finding]) -> dict:
    """Headline agreement stats: raw %, Cohen's kappa, and the reconciliation mix."""
    pairs = []
    for f in findings:
        cats = [v["category"] for v in f.graders.values() if isinstance(v, dict)]
        if len(cats) == 2:
            pairs.append((cats[0], cats[1]))
    from collections import Counter
    mix = Counter(f.agreement for f in findings)
    raw = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else float("nan")
    return {
        "graded": len(findings),
        "raw_agreement": round(raw, 3) if pairs else None,
        "cohen_kappa": round(cohen_kappa(pairs), 3) if pairs else None,
        "consensus": mix.get("consensus", 0),
        "oracle_broke_tie": mix.get("oracle_broke_tie", 0),
        "judge": mix.get("judge", 0),
        "unresolved": mix.get("unresolved", 0),
    }
