# harness/grader — dual-grader weakness-discovery pipeline

Turns game transcripts into a **ranked `(decision_type, criterion, tag)` fail-rate
table** — the generation target for Michael's env gen *and* the before/after
measurement baseline. Implements the shared contract in `grader-spec.md` +
`MICHAEL-HANDOFF.md`, with a dual-grader + regret-oracle layer on top.

```
transcripts ─▶ derive (decision_type, state_tags) + regret
            ─▶ Claude + OpenAI score each criterion ─▶ reconcile (consensus-fail)
            ─▶ aggregator: Wilson-ranked fail-rate table ─▶ env gen / before-after demo
```

## The contract (shared with Michael — do not drift)

- `taxonomy.py` — **single source of truth** for the vocab: `decision_type`
  (placement / trade / build_spend) × `criterion` (per type) × `state_tags`
  (frozen set). The grader, the aggregator, and Michael's env generator all import
  it. A weakness we *measure* and one Michael *generates for* must be the same
  string. Append IDs; never rename mid-run.
- `grader-spec.md` — the grader rubric/prompt + output schema.
- `aggregator.py` — verdicts → `(criterion, tag)` buckets → **fail-rate** (not
  count), Wilson lower-bound ranked (small-n flukes sink), `MIN_SAMPLES` floor;
  `compare()` does the held-out before/after scoreboard. Run it standalone for a
  synthetic demo: `python harness/grader/aggregator.py`.
- `MICHAEL-HANDOFF.md` — env JSON schema, ground-truth rules, train/held-out
  disjointness, Goldilocks difficulty.

## Our layer on top of the spec

- `context.py` — derives `decision_type` from the action and `state_tags` from the
  **live engine state** (buildable nodes, robber adjacency, awards, VP standing).
  Objective tags mean both graders share identical tags, so aggregator buckets
  never split on a tag disagreement.
- `oracle.py` — Catanatron value-fn **regret** per decision (deterministic replay).
  Kept as an *auxiliary* objective signal on every graded decision (and given to
  the graders as evidence) — NOT a gate, because the fail-rate denominator must be
  unbiased.
- `graders.py` — runs one LLM grader; optional self-consistency (majority `failed`).
- `reconcile.py` — merges the two graders per criterion: **a criterion is failed
  only when BOTH agree** (consensus-fail → high-precision metric). Keeps both raw
  verdicts + per-criterion agreement; reports **Cohen's κ** overall and per
  criterion, plus union (recall) vs consensus (precision) counts.
- `pipeline.py` + `scripts/grade_transcripts.py` — orchestrate and emit the table.

## Grading granularity (hybrid, parallel)

The CLI default is the **hybrid** path: each sampled decision is graded in its **own
call** (full attention, no compression) that carries a **compact game-context blurb**
(outcome + VP trajectory + the local move window), and ALL `(board × decision ×
grader)` calls fan out in **one global rate-limited pool** (`pipeline.grade_run`).
Because per-decision calls are small and fast, wall-clock ≈ `total_calls /
--concurrency`, so detail doesn't cost time — only API rate limits bound it.

The older **whole-game** path (one call/game scoring N decisions; `grade_game` /
`build_game_prompt`) is still available in code — cheaper in raw calls but slower
per call and lower per-decision attention. The hybrid is preferred.

## Usage

```bash
# free preview: decisions, call count, decision_type + tags
python scripts/grade_transcripts.py transcripts/selfplay --per-game 15 --dry-run

# hybrid dual grade (set both keys first); tune --concurrency to your rate limits:
python scripts/grade_transcripts.py transcripts/selfplay \
    --per-game 15 --concurrency 16 --merge union
```

- `--per-game N` — decisions uniformly sampled per game (unbiased denominator).
- `--concurrency` — global parallel calls; raise it (rate-limit permitting) to cut
  wall-clock. Calls = decisions × 2 graders.
- `--merge consensus|union` — weakness table counts a criterion failed when BOTH
  graders agree (consensus, precision) or EITHER does (union, recall/discovery).

Outputs `<run>/grading/findings.jsonl` (one object per decision, both raw verdicts
kept) + `report.json` (weakness table + agreement). Re-rank without re-grading via
`pipeline.report(objects, merge=...)`.

## Cost / time note

These 400-cap games have ~230 gradeable decisions each. Bound work with `--per-game`
(coverage) and speed with `--concurrency` (wall-clock). The denominator only ever
counts graded decisions of each decision_type — never rolls/forced moves.

## Decisions made integrating the two designs

1. The handoff files are canonical — adopted their vocab, schema, and fail-rate
   aggregator wholesale (loop-closing with Michael).
2. Kept the dual grader + Cohen's κ (the spec was single-grader) — consensus-fail
   for a precision metric, agreement retained as confidence/audit signal.
3. Kept regret as evidence + auxiliary cross-check, not a gate.
4. `state_tags` derived objectively from the engine, not asked of the grader, so
   the two graders' buckets always align.
5. No human gold yet (`data/examples` labels empty) → the oracle is the objective
   anchor; add a calibration pass against gold when labels land.
```
