# Grading — One-Page Overview

A concise summary of the grading setup and its current status. For the full
rationale + FAQ see `GRADING-REVIEW.md`; for the contract see `grader-spec.md`.

## What it produces

A ranked **`(decision_type, criterion, tag)` fail-rate table** from self-play
transcripts — the same table that targets env generation and measures before/after.

## Pipeline (per game = one sub-grader, run in parallel)

1. **Replay** the transcript through the Catanatron engine (deterministic).
2. **Derive** `decision_type` (placement / trade / build_spend) + `state_tags`
   objectively from the engine state.
3. **Regret oracle** scores every legal move (value-fn) → objective "how much value
   was given up," as evidence (not a gate).
4. **Dual grade** — Claude + OpenAI each score every criterion for the decision,
   with whole-game context.
5. **Reconcile** per criterion: **default union = either grader fails (over-critical
   / recall)**; consensus = both agree (precision) via `--merge consensus`. One-sided
   flags marked `disputed` (both takes kept); **Cohen's κ** per criterion.
6. **Aggregate** by `(decision_type, criterion, tag)`, ranked by **fail-rate**
   (not count), **Wilson-discounted** so small-n flukes don't top the list.

## Why it's trustworthy (the load-bearing choices)

- **Frozen `taxonomy.py`** shared by grader + aggregator + env gen → no vocab drift.
- **fail-rate, not count** → measures skill deficit, not situation frequency.
- **group by tag** → "robber-timing-when-behind", targetable by env gen.
- **dual grader + κ** → agreement is a measured confidence signal, not assumed.
- **regret oracle** → objective ground signal; no human gold yet (calibration TBD).
- **no self-grading** → players are Qwen/Gemma, graders are Claude/OpenAI.

## Output: the detailed table (catch nuance, not just a number)

The headline output is a **detailed** weakness table — one row per
`(decision_type, criterion, tag)` keeping the nuances flat aggregation drops:
- **both consensus & union** fail rates (precision view + recall view),
- **per-grader fail rate** (so a one-sided grader is visible *in the row*, not
  hidden — this is how we caught GPT-4o flagging trades 62% vs Claude's 5%),
- **inter-grader agreement** + `disputed` count (one-sided flags kept, both takes
  recorded), and **Cohen's κ** per criterion (low κ = a fuzzy criterion to refine),
- **mean oracle regret** (objective corroboration) and **example plies** to drill in.

`report.json` carries `detailed_table` + `findings.jsonl` (both raw verdicts per
decision); re-rank either view without re-grading via `pipeline.report(merge=…)`.

## Grader calibration (keeping the two graders aligned)

Two levers keep the graders from diverging (and keep over-flagging down):
- **Strict scale:** score **0/fail only for a clear, explainable error with a
  clearly better legal move**; "two reasonable players could differ" → 1, not 0.
- **Specific FAIL conditions** (esp. placement): e.g. `expansion_room` fails only
  when *genuinely boxed in* (an open-board opening is not a fail); `blocking_value`
  fails only when an *obvious available take* was passed up — not "a good node
  remains open." Vague criteria were the main source of one-sided flags.
- **Oracle is a hint, not truth:** it's a myopic 1-ply value-fn heuristic (misses
  multi-turn/positional payoff), so graders trust their own judgment when it
  conflicts — but a low-regret move defaults toward "defensible" absent a clear error.

## How to run

```bash
python scripts/grade_transcripts.py transcripts/<run> --dry-run                 # free preview
python scripts/grade_transcripts.py transcripts/<run> --per-game 15 \
    --concurrency 16 --merge union                                              # hybrid dual grade
```
- **Hybrid (default):** each decision graded in its own call (full attention) with
  a compact game-context blurb; all calls fan out in one rate-limited pool, so
  wall-clock ≈ calls / `--concurrency`. `--per-game` = coverage, `--merge` = view.
- Outputs `findings.jsonl` (per decision, both raw verdicts) + `report.json`
  (weakness table + agreement). Re-rank without re-grading via `pipeline.report`.

## Status & caveats (current)

- **Fixed (important):** a decision-context off-by-one — the grader prompt had been
  paired with the *adjacent* decision's reasoning/hand/legal set (transcript `ply`
  is 1-based; the oracle indexes 0-based by `action_records`). Now keyed by list
  index. **Any grading run from before this fix is invalid and must be re-graded.**
- **Conservative magnitudes:** consensus under-flags; union is for discovery. Read
  both, and watch κ — a low-κ criterion is fuzzy and should be refined.
- **VP-dominant oracle:** regret is dominated by VP-changing mistakes (`regret_vp`
  for legibility); finer per-type normalization is a tunable.
- **No human gold yet:** oracle + grader agreement are the trust anchors; add a
  calibration pass against gold labels when they exist.
