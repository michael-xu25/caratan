# Grading Setup — Review & Q&A

A self-contained briefing on the dual-grader weakness-discovery pipeline: what it
does, the design decisions and *why*, and an FAQ for defending it in review.

---

## TL;DR

We turn self-play game transcripts into a **ranked `(decision_type, criterion,
tag)` fail-rate table**. That one table is *both* the target for environment
generation *and* the before/after measurement baseline, so the training loop can't
"improve" something it didn't target or target something it doesn't measure. Two
independent LLM graders (Claude + OpenAI) score each decision against a frozen
rubric; an engine-computed **regret oracle** provides objective evidence; results
aggregate into Wilson-ranked weaknesses.

---

## Where it fits in the loop

```
self-play transcripts ─▶ GRADER ─▶ ranked weakness table ─▶ env generation ─▶ RL ─▶ re-grade (before/after)
```

The grader is the bridge between "we have games" and "we know what to train." Its
credibility rests on being an **independent, frozen measuring standard** (it is not
the player, not the bot's value weights, and its vocabulary can't drift).

## How it works (pipeline)

1. **Replay** each transcript deterministically through the real Catanatron engine
   (recorded actions + results), so it works even on LLM games that aren't
   seed-reproducible.
2. **Derive context** objectively from the engine at each decision: `decision_type`
   (placement / trade / build_spend) and `state_tags` (phase, standing, resources,
   board, flags).
3. **Regret oracle**: score every legal action with Catanatron's value function;
   `regret = value(best legal) − value(chosen)`, reported in VP-equivalents. This
   is objective evidence given to the graders — it answers "how much value was
   given up" without an LLM.
4. **Dual grade**: Claude and OpenAI each score the decision on *every* criterion
   for its type (2 = good, 1 = defensible, 0 = mistake; `failed` = score 0), with
   the whole-game context and the oracle's evidence.
5. **Reconcile**: per criterion — consensus (both agree it failed) for a
   high-precision metric; disagreements are kept, both raw verdicts retained,
   **Cohen's κ** reported per criterion.
6. **Aggregate**: group verdicts by `(decision_type, criterion, tag)`; rank by
   **fail-rate** (not count), discounted by the **Wilson lower bound** so small-n
   flukes don't reach the top. `compare()` does the held-out before/after.

## Key design decisions (and why each is defensible)

| Decision | Why |
|---|---|
| **Frozen taxonomy as code** (`taxonomy.py`) | The grader, the aggregator, and Michael's env generator all import the *same* IDs. A weakness we measure and one he generates for must be the same string, or the loop silently splits and never closes. Off-vocab fails loudly. |
| **The rubric is separate from the bot's value weights** | Grading the model by "did it play like one specific bot" is circular and multiplayer-flavored. The value fn is *called* as an oracle, but the rubric is its own owned standard. |
| **Two signals, combined by role** | Quantitative regret **grounds & gates** (objective "how bad"); the LLMs **explain & categorize** (the "why" the value fn can't see — reasoning, trade EV, tempo). |
| **Dual grader, agreement = confidence** | Two independent models cross-check each other; consensus is high-precision, disagreement is a signal to audit (and a taxonomy-fuzziness probe via per-criterion κ). |
| **fail-RATE, not fail-count** | Ranking by count just rediscovers which situations are *common*, not which the model is *bad at*. Rate normalizes for frequency → it measures skill deficit. (This is why passes are scored too — they're the denominator.) |
| **Group by `(criterion, TAG)`** | "Bad at robber timing" is too coarse to generate from. Localizing to the *state* where it fails ("robber timing when behind") is what makes generated scenarios targeted. |
| **Wilson lower bound for ranking** | 3/3 fails is 100% on pure noise. Wilson discounts small samples automatically, so the top of the list is high-fail *and* well-supported. |
| **One table for generation + measurement** | Train-target and eval-target are identical by construction — you can't game the metric. |
| **Objective `state_tags`, not grader-assigned** | Derived from the engine so both graders share identical aggregation buckets (no split on a tag disagreement). |
| **Disjoint discovery vs held-out pools** | Before/after is measured on fresh positions → the number is learning, not memorization. |

## Outputs

- `findings.jsonl` — one object per graded decision: merged criteria, `state_tags`,
  regret, oracle's best move, **both raw grader verdicts**, per-criterion agreement.
- `report.json` — the Wilson-ranked weakness table + agreement (κ overall and per
  criterion, plus union/consensus counts). Re-rankable without re-grading.

## Cost & parallelism

- **Hybrid (default):** each decision graded in its own call (full attention) with
  a compact game-context blurb; all `(board × decision × grader)` calls fan out in
  one rate-limited pool. **Wall-clock ≈ total_calls / concurrency**, so detail is
  bounded by API rate limits, not call count. `--per-game` sets coverage,
  `--concurrency` sets speed.
- **Whole-game (alt):** one call/game scoring N decisions — fewer raw calls but
  slower per call and less per-decision attention. Kept in code; hybrid preferred.

## FAQ (anticipated review questions)

**Q: Why two graders instead of one?**
Cross-checking. One LLM judge can be confidently wrong; two independent ones give a
confidence signal (agreement) and let us separate "real mistake both caught" from
"one grader's opinion." We report Cohen's κ so the reliability is measured, not
assumed.

**Q: How do you merge disagreements?**
Per criterion, the **default is union** (either grader fails it → flagged). We
deliberately bias over-critical: a missed weakness never gets targeted, whereas a
flagged-but-defensible one only costs a second look. One-sided flags are marked
`disputed` and record BOTH graders' takes (who failed, who passed, why), and we
keep both raw verdicts + a per-criterion agreement flag — so the conservative
consensus (both-agree) view is always recoverable. We never silently pick a winner.

**Q: The fail-rates look low (a few %). Is the model actually fine?**
The headline now uses the *union* (either-grader) view, which surfaces the real
signal — e.g. trade `enables_key_build` runs ~11–14% (boxed-in / robber-threat /
endgame). The *consensus* (both-agree) view is the conservative high-precision
cross-check (`--merge consensus`). Both come from the same data.

**Q: The two graders disagree a lot (low κ, many `disputed`). Is that a problem?**
It's expected signal, and we act on it rather than hide it. Low per-criterion κ
means that *criterion* is fuzzy — vaguely-worded criteria let the two models apply
different bars (we saw GPT-4o flag placement criteria 3–4× more than Claude). The
fix is calibration, not averaging: (1) **specific FAIL conditions** — e.g.
`expansion_room` fails only when genuinely boxed in, `blocking_value` only when an
obvious take was passed up; (2) **a strict scale** — score 0 only for a clear,
explainable error with a clearly better legal move, else 1–2. Re-grading after a
calibration pass should raise κ and shrink the disputed share. Disputed flags are
always kept (both takes recorded) so nothing is silently dropped.

**Q: Isn't the denominator diluted by rolls / forced moves?**
No — verified. Only decisions that map to a real `decision_type`
(placement/trade/build_spend) are graded; rolls, end-turn, and discards never enter.
The fail-rate denominator for a bucket equals the count of *graded decisions of that
type+tag*, nothing else.

**Q: How do you know it's measuring skill, not just frequent situations?**
We rank by fail-*rate* (failures ÷ times-scored), not count, and we score passing
decisions too so the denominator is real. Wilson then keeps only rates that are both
high and well-sampled.

**Q: Is the grader biased toward the player it's grading?**
No self-preference: players are Qwen/Gemma, graders are Claude/OpenAI — no family
overlap. The rubric is also independent of the bot's value weights.

**Q: What's the objective ground truth?**
The regret oracle (engine value function) gives an objective "how much value was
given up" per decision — used as evidence to the graders and as a cross-check
(high regret + graders say "fine" = audit the value fn or the grader). No human gold
labels exist yet; when they do, we add a calibration pass against them.

**Q: What stops the two halves of the loop from drifting?**
The frozen `taxonomy.py`, imported by both sides. A mismatched ID is an import/
validation error, not a silent miscount.

**Q: How do you prove training helped?**
Re-grade a **held-out** pool (positions disjoint from training) before vs after, and
report the per-`(criterion, tag)` fail-rate delta (e.g. "robber-timing-when-behind:
70% → 22%"). Same table, fresh positions → it's learning, not memorization.

**Q: How much does a run cost / how long?**
~`per_game × #games × 2` calls; wall-clock is `calls / concurrency`. The 20-board
(40-game) run at `--per-game 15` is ~1,200 calls, parallelized to the rate limit.

## Fixed issues

- **Decision-context off-by-one (fixed):** the grader prompt was paired with the
  *adjacent* decision's reasoning/hand/legal set — the transcript's `ply` field is
  1-based while the oracle indexes decisions 0-based by `action_records` position.
  Symptom: e.g. robber reasoning shown on a trade. Now keyed by list index (1:1
  with `action_records`). **Grading runs from before this fix used misaligned
  context and must be re-graded.**

## Known limitations (state them before someone else does)

- **No human gold yet** — the oracle + dual-grader agreement are the trust anchors;
  a human-labeled calibration set is the next rigor step.
- **VP-dominant value fn** — the oracle weights VP at 3e14, so raw regret is
  dominated by VP-changing mistakes; positional regret is comparatively small.
  Reported as `regret_vp` for legibility; finer per-type normalization is a tunable.
- **κ is moderate (~0.45)** — the two graders agree well on `tempo`/`vp_efficiency`/
  `port_access` but weakly on fuzzy criteria (`expansion_room`, `timing_strength`),
  which flags those taxonomy entries for refinement.
- **Grader leniency** — strong graders + consensus can under-flag; the union view
  and tightened prompts mitigate, but the magnitudes are conservative by design.
