"""Merge two graders' per-criterion verdicts into one aggregator-ready grader
object, and measure their agreement.

Merge policy (per criterion): a criterion is marked failed in the merged object
when EITHER grader fails it (UNION-fail). We deliberately err on the side of being
over-critical — if even one grader thinks a move is a mistake, we flag it — rather
than requiring both to agree (which under-counts real weaknesses). High recall is
the right bias here: a flagged-but-defensible decision costs a second look; a
missed weakness never gets targeted or measured.

Disparities are NOT swept away: every one-sided flag is marked `disputed=True` on
the criterion, its `reason` records BOTH graders' takes (who failed, who passed,
and why), and per-criterion `agreement` + both raw verdicts are kept on the object.
So you can always recover the consensus (high-precision) view and audit fuzzy
criteria, while the headline metric counts the union.

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
    n_disputed = 0
    for name in CRITERIA_BY_TYPE[dtype]:
        ca, cb = va.criteria[name], vb.criteria[name]
        fa, fb = bool(ca["failed"]), bool(cb["failed"])
        agree = fa == fb
        per_crit_agree[name] = agree
        failed = fa or fb                               # UNION-fail: flag if EITHER grader fails
        disputed = fa != fb                             # exactly one grader flagged it
        if disputed:
            n_disputed += 1
            flagger = va.grader if fa else vb.grader
            passer = vb.grader if fa else va.grader
            fr = (ca if fa else cb)["reason"]
            pr = (cb if fa else ca)["reason"]
            reason = (f"[DISPUTED] {flagger} FAILED: {fr or '(no reason)'} || "
                      f"{passer} passed: {pr or '(no reason)'}")
        else:
            reason = (ca["reason"] if fa else (ca["reason"] or cb["reason"]))
        # over-critical scoring: any fail -> 0; ties keep the better (agreed) score.
        merged.append({"name": name,
                       "score": 0 if failed else max(ca["score"], cb["score"]),
                       "failed": failed, "disputed": disputed, "reason": reason})

    weakness_labels = [{"criterion": c["name"], "state_tags": regret.state_tags,
                        "disputed": c["disputed"]}
                       for c in merged if c["failed"]]

    return {
        "decision_id": decision_id,
        "decision_type": dtype,
        "state_tags": regret.state_tags,
        "criteria": merged,
        "summary": va.summary or vb.summary,
        "weakness_labels": weakness_labels,
        "n_disputed": n_disputed,        # how many criteria were one-sided flags
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
        "union_fails": union,        # either grader — THIS is what the metric now counts
        "consensus_fails": inter,    # both graders — high-precision cross-check (kept for audit)
        "disputed_fails": union - inter,  # one-sided flags (over-critical bias picks these up)
    }
