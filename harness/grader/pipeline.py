"""End-to-end: each game (board) -> ONE whole-game call per grader -> per-decision
reconciled verdicts -> ranked (criterion, tag) fail-rate weakness table.

Cost is ~2 LLM calls per game (one per grader), independent of how many decisions
are scored within the game. We score a uniform sample of N decisions per game
(unbiased denominator); the oracle's regret is given to the grader as evidence.
"""
from __future__ import annotations

from harness.grader.oracle import compute_regrets
from harness.grader.graders import grade_game
from harness.grader.reconcile import reconcile, agreement_report
from harness.grader.aggregator import flatten_grader_object, aggregate


def _decisions_by_ply(transcript: dict) -> dict:
    return {d.get("i", d.get("ply", k)): d for k, d in enumerate(transcript.get("decisions", []))}


def _sample(regrets, per_game: int):
    """Uniform sample of up to per_game decisions (stride) — unbiased denominator."""
    if per_game <= 0 or len(regrets) <= per_game:
        return list(regrets)
    stride = len(regrets) / per_game
    return [regrets[int(i * stride)] for i in range(per_game)]


def grade_transcript(transcript: dict, grader_a, grader_b, per_game: int = 15) -> list[dict]:
    regrets = compute_regrets(transcript)
    decisions = _decisions_by_ply(transcript)
    selected = _sample(regrets, per_game)
    if not selected:
        return []
    game_id = transcript.get("label", "game")

    # one whole-game call per grader
    va = grade_game(grader_a, transcript, selected, decisions)
    vb = grade_game(grader_b, transcript, selected, decisions)

    objects = []
    for r in selected:
        objects.append(reconcile(r, va[r.ply], vb[r.ply], decision_id=f"{game_id}:{r.ply}"))
    return objects


def report(objects: list[dict], min_samples: int = 15) -> dict:
    verdicts = []
    for o in objects:
        verdicts.extend(flatten_grader_object(o))
    return {
        "agreement": agreement_report(objects),
        "weakness_table": aggregate(verdicts, min_samples=min_samples),
        "num_decisions": len(objects),
        "num_verdicts": len(verdicts),
    }
