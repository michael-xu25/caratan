"""End-to-end: transcripts -> gated decisions -> dual grade -> reconcile ->
ranked failure modes. The deliverable that feeds env generation.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from harness.grader.oracle import compute_regrets, gate_threshold, apply_gate
from harness.grader.graders import grade
from harness.grader.reconcile import reconcile, agreement_report, Finding


def _decisions_by_ply(transcript: dict) -> dict:
    return {d.get("i", d.get("ply", k)): d for k, d in enumerate(transcript.get("decisions", []))}


def select_decisions(regrets, gate_pct: float, low_regret_rate: float):
    """Gated (high-regret) decisions + a deterministic low-regret sample (to catch
    lucky-right reasoning failures and audit the oracle)."""
    thr = gate_threshold(regrets, gate_pct)
    apply_gate(regrets, thr)
    gated = [r for r in regrets if r.gated]
    chosen = list(gated)
    if low_regret_rate > 0:
        low = [r for r in regrets if not r.gated and r.regret > 0]
        stride = max(1, int(round(1 / low_regret_rate)))
        chosen += low[::stride]
    return chosen, thr


def grade_transcript(transcript: dict, grader_a, grader_b, judge=None,
                     gate_pct: float = 75.0, low_regret_rate: float = 0.1,
                     samples: int = 1, max_workers: int = 8) -> list[Finding]:
    regrets = compute_regrets(transcript)
    decisions = _decisions_by_ply(transcript)
    todo, _ = select_decisions(regrets, gate_pct, low_regret_rate)
    game_id = transcript.get("label", "game")

    def _one(r):
        dec = decisions.get(r.ply)
        va = grade(grader_a, r, dec, samples=samples)
        vb = grade(grader_b, r, dec, samples=samples)
        f = reconcile(r, va, vb, judge_backend=judge, decision=dec)
        f.game_id = game_id
        return f

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, todo))


def aggregate(findings: list[Finding]) -> list[dict]:
    """Bucket findings by category, rank by total VP-equivalent regret (= ROI to fix)."""
    by_cat: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_cat[f.category].append(f)

    modes = []
    for cat, fs in by_cat.items():
        fs_sorted = sorted(fs, key=lambda x: -x.regret_vp)
        rep = fs_sorted[0]
        consensus = sum(1 for f in fs if f.agreement == "consensus")
        modes.append({
            "category": cat,
            "frequency": len(fs),
            "total_regret_vp": round(sum(f.regret_vp for f in fs), 4),
            "mean_regret_vp": round(sum(f.regret_vp for f in fs) / len(fs), 4),
            "consensus_rate": round(consensus / len(fs), 3),
            "example_plies": [f"{f.game_id or '?'}:{f.ply}" for f in fs_sorted[:5]],
            "representative_explanation": rep.explanation,
        })
    modes.sort(key=lambda m: -m["total_regret_vp"])
    for i, m in enumerate(modes):
        m["roi_rank"] = i + 1
    return modes


def report(findings: list[Finding]) -> dict:
    """Full grading report: ranked modes + agreement headline + per-category kappa-ish."""
    return {
        "agreement": agreement_report(findings),
        "failure_modes": aggregate(findings),
        "num_findings": len(findings),
    }
