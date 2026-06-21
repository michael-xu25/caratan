"""End-to-end: transcripts -> per-decision dual grading -> aggregator-ready grader
objects -> ranked (criterion, tag) fail-rate weakness table.

By default every decision that maps to a taxonomy decision_type is graded (the
fail-rate denominator must be unbiased — see aggregator.py). The regret oracle is
kept as an auxiliary signal on each object, NOT a gate. Optional budget controls
(`gate_pct`, `max_per_game`) trade coverage for cost — `gate_pct` biases the
denominator toward hard decisions, so prefer `max_per_game` (a uniform sample).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from harness.grader.oracle import compute_regrets, gate_threshold, apply_gate
from harness.grader.graders import grade
from harness.grader.reconcile import reconcile, agreement_report
from harness.grader.aggregator import flatten_grader_object, aggregate


def _decisions_by_ply(transcript: dict) -> dict:
    return {d.get("i", d.get("ply", k)): d for k, d in enumerate(transcript.get("decisions", []))}


def _select(regrets, gate_pct: float, max_per_game: int):
    sel = regrets
    if gate_pct and gate_pct > 0:
        thr = gate_threshold(regrets, gate_pct)
        apply_gate(regrets, thr)
        sel = [r for r in regrets if r.gated]
    if max_per_game and len(sel) > max_per_game:        # uniform stride sample (unbiased)
        stride = len(sel) / max_per_game
        sel = [sel[int(i * stride)] for i in range(max_per_game)]
    return sel


def grade_transcript(transcript: dict, grader_a, grader_b, gate_pct: float = 0.0,
                     max_per_game: int = 0, samples: int = 1, max_workers: int = 8) -> list[dict]:
    regrets = compute_regrets(transcript)
    decisions = _decisions_by_ply(transcript)
    game_id = transcript.get("label", "game")
    todo = _select(regrets, gate_pct, max_per_game)

    def _one(r):
        dec = decisions.get(r.ply)
        va = grade(grader_a, r, dec, samples=samples)
        vb = grade(grader_b, r, dec, samples=samples)
        return reconcile(r, va, vb, decision_id=f"{game_id}:{r.ply}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, todo))


def report(objects: list[dict], min_samples: int = 15) -> dict:
    """Ranked weakness table (Wilson-ranked fail-rate) + dual-grader agreement."""
    verdicts = []
    for o in objects:
        verdicts.extend(flatten_grader_object(o))
    rows = aggregate(verdicts, min_samples=min_samples)
    return {
        "agreement": agreement_report(objects),
        "weakness_table": rows,
        "num_decisions": len(objects),
        "num_verdicts": len(verdicts),
    }
