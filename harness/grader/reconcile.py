"""Merge two graders' per-criterion verdicts into one aggregator-ready grader
object, and measure their agreement.

Merge policy (per criterion): a criterion is marked failed in the merged object
ONLY when BOTH graders fail it (consensus-fail). This gives a high-precision
measurement metric — the before/after fail-rate counts only mistakes both models
agree on. The disagreements are kept (per-criterion `agreement` + both raw
verdicts) so you can also compute the union (recall) and audit fuzzy criteria.

Agreement is reported with Cohen's kappa over the per-criterion `failed` booleans
(chance-corrected), overall and per criterion.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict

from harness.grader.graders import Verdict
from harness.grader.taxonomy import CRITERIA_BY_TYPE


def reconcile(regret, va: Verdict, vb: Verdict, decision_id: str) -> dict:
    """Produce one grader object (the shape aggregator.flatten_grader_object reads)
    with merged criteria, plus provenance: regret, both raw verdicts, agreement."""
    dtype = regret.decision_type
    merged, per_crit_agree = [], {}
    for name in CRITERIA_BY_TYPE[dtype]:
        ca, cb = va.criteria[name], vb.criteria[name]
        agree = ca["failed"] == cb["failed"]
        per_crit_agree[name] = agree
        failed = ca["failed"] and cb["failed"]          # consensus-fail
        reason = (ca["reason"] if ca["failed"] else cb["reason"]) if failed else (
            ca["reason"] or cb["reason"])
        merged.append({"name": name, "score": 0 if failed else max(ca["score"], cb["score"]),
                       "failed": failed, "reason": reason})

    weakness_labels = [{"criterion": c["name"], "state_tags": regret.state_tags}
                       for c in merged if c["failed"]]

    return {
        "decision_id": decision_id,
        "decision_type": dtype,
        "state_tags": regret.state_tags,
        "criteria": merged,
        "summary": va.summary or vb.summary,
        "weakness_labels": weakness_labels,
        # ---- provenance (ignored by the aggregator) ----
        "ply": regret.ply,
        "turn": regret.turn,
        "color": regret.color,
        "action": regret.action,
        "regret_vp": round(regret.regret_vp, 4),
        "oracle_best": regret.oracle_best,
        "agreement": per_crit_agree,
        "graders": {va.grader: asdict(va), vb.grader: asdict(vb)},
    }


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Cohen's kappa for two raters over paired booleans."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca.get(v, 0) / n) * (cb.get(v, 0) / n) for v in (True, False))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def agreement_report(objects: list[dict]) -> dict:
    """Cohen's kappa over per-criterion `failed` booleans, overall + per criterion,
    plus the union/intersection fail counts (recall vs precision view)."""
    overall, by_crit = [], defaultdict(list)
    union = inter = 0
    for o in objects:
        for c in o["criteria"]:
            g = list(o["graders"].values())
            if len(g) != 2:
                continue
            fa = g[0]["criteria"][c["name"]]["failed"]
            fb = g[1]["criteria"][c["name"]]["failed"]
            overall.append((fa, fb)); by_crit[c["name"]].append((fa, fb))
            union += int(fa or fb); inter += int(fa and fb)
    return {
        "graded_decisions": len(objects),
        "cohen_kappa_overall": round(cohen_kappa(overall), 3) if overall else None,
        "kappa_by_criterion": {k: round(cohen_kappa(v), 3) for k, v in sorted(by_crit.items())},
        "union_fails": union,        # recall view (either grader)
        "consensus_fails": inter,    # precision view (both graders) — what the metric counts
    }
