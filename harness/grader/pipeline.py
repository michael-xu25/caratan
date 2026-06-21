"""End-to-end: each game (board) -> ONE whole-game call per grader -> per-decision
reconciled verdicts -> ranked (criterion, tag) fail-rate weakness table.

Cost is ~2 LLM calls per game (one per grader), independent of how many decisions
are scored within the game. We score a uniform sample of N decisions per game
(unbiased denominator); the oracle's regret is given to the grader as evidence.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from harness.grader.oracle import compute_regrets
from harness.grader.graders import grade_game, grade_decision
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


def grade_run(transcripts: list[dict], grader_a, grader_b, per_game: int = 15,
              concurrency: int = 16, progress=None) -> list[dict]:
    """HYBRID path: score each sampled decision in its OWN call (full attention +
    compact game context), fanning ALL (board × decision × grader) calls into one
    global rate-limited pool. Wall-clock ~= total_calls / concurrency, so detail
    doesn't cost time — only API rate limits bound it.
    """
    # build the task list across all transcripts
    tasks = []  # (key, transcript, regret, decision)
    for t in transcripts:
        try:
            regrets = compute_regrets(t)
        except Exception:
            continue
        decs = _decisions_by_ply(t)
        gid = t.get("label", "game")
        for r in _sample(regrets, per_game):
            tasks.append((f"{gid}:{r.ply}", t, r, decs.get(r.ply)))

    # one pool, two jobs per decision (one per grader) — full call-level parallelism
    results: dict = {}
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {}
        for key, t, r, d in tasks:
            futs[ex.submit(grade_decision, grader_a, t, r, d)] = (key, "a", r)
            futs[ex.submit(grade_decision, grader_b, t, r, d)] = (key, "b", r)
        for f in as_completed(futs):
            key, slot, r = futs[f]
            cell = results.setdefault(key, {"r": r})
            cell[slot] = f.result()
            done += 1
            if progress and done % 50 == 0:
                progress(done, len(futs))

    objects = []
    for key, cell in results.items():
        if "a" in cell and "b" in cell:
            objects.append(reconcile(cell["r"], cell["a"], cell["b"], decision_id=key))
    return objects


def _merged_object(o: dict, merge: str) -> dict:
    """Re-derive an object's criteria as consensus (both fail) or union (either fail)
    from the two raw grader verdicts, for the aggregator."""
    from harness.grader.taxonomy import CRITERIA_BY_TYPE
    graders = [g for g in o["graders"].values() if isinstance(g, dict) and "criteria" in g]
    crit = []
    for name in CRITERIA_BY_TYPE[o["decision_type"]]:
        fs = [bool(g["criteria"][name]["failed"]) for g in graders if name in g["criteria"]]
        failed = (any(fs) if merge == "union" else (all(fs) if fs else False))
        crit.append({"name": name, "failed": failed})
    return {"decision_type": o["decision_type"], "state_tags": o["state_tags"],
            "criteria": crit, "decision_id": o.get("decision_id")}


def report(objects: list[dict], min_samples: int = 15, merge: str = "consensus") -> dict:
    verdicts = []
    for o in objects:
        verdicts.extend(flatten_grader_object(_merged_object(o, merge)))
    return {
        "merge": merge,
        "agreement": agreement_report(objects),
        "weakness_table": aggregate(verdicts, min_samples=min_samples),
        "num_decisions": len(objects),
        "num_verdicts": len(verdicts),
    }
