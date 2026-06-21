# Grading Rubric & Grader Design — PROPOSAL (draft for review)

> Status: **design only, not implemented.** This is a proposal for how the
> grader turns game transcripts into ranked, targetable failure modes. Edit
> freely; open decisions are marked **[DECIDE]**. Nothing gets built until this
> is signed off.

## 1. Purpose & where it fits

Input: ~100 self-play transcripts (the player model, e.g. Gemma, playing itself,
fair mirrored 1v1). Output: a **ranked list of targetable failure modes**, each
anchored to specific decisions, that feeds environment generation → RL → re-eval.

The grader is the bridge between "we have games" and "we know what to train."
Its credibility rests on being an **independent measuring standard** — see §2.

```
self-play transcripts ─▶ GRADER (this doc) ─▶ ranked failure modes ─▶ env generation ─▶ RL ─▶ re-eval
```

## 2. Key design decisions (the load-bearing ones)

1. **The rubric is separate from Catanatron's player weights.** Catanatron's
   `value.py` weights are tuned so a *bot's argmax* sorts moves correctly — an
   optimization artifact, not a measurement standard, and multiplayer-flavored.
   The rubric is its own owned artifact. The value function may be *called* by
   the rubric as an oracle, but it is not the rubric. (Avoids circularity: don't
   grade the model purely by "did it play like one specific bot.")
2. **Whole-game read, decision-anchored output.** Grade with full-game context
   (Catan moves are contextual) but emit findings tied to specific plies, so they
   map 1:1 onto the per-decision scenario schema for env generation.
3. **Two signals, combined by role** (§5): quantitative *grounds & gates*,
   qualitative *explains & categorizes*.
4. **Dual grader (Claude + OpenAI)** for cross-checking; agreement = confidence.
5. **Data-driven & engine-grounded** — works from recorded transcripts (incl.
   LLM games, which aren't seed-reproducible) and the Catanatron engine.

## 3. The quantitative rubric (objective, cheap, runs on everything)

Per real decision (skip rolls / forced single-option moves):

```
regret = value(best legal action) − value(chosen action)
```

- `value()` = Catanatron's value function (`base_fn`), or AlphaBetaPlayer depth-N
  for a stronger oracle. Computed by replaying the recorded action into a `State`,
  then scoring each legal action's resulting state.
- Gives: **which** decisions were mistakes, **how bad** (magnitude), and **what
  was better** (the oracle's top action).
- Normalize regret to 0–1 per decision-type for legibility (raw value-fn units
  are not interpretable).

**Quant dimensions (derived from `value.py` features, reweighted for 1v1):**

| Dimension | From value.py | 1v1 adjustment |
|---|---|---|
| VP / win focus | `public_vps` | unchanged (paramount) |
| Effective production (pip-weighted) + variety | `production`, `value_production` | unchanged |
| **Opponent denial** | `enemy_production` | **weight up** (only one opponent to block) |
| Expansion potential / not boxed in | `reachable_production_1`, `buildable_nodes` | unchanged |
| Longest road / largest army | `longest_road`, `army_size` | keep "matters more when boxed in" nuance |
| Hand efficiency (distance-to-build) | `hand_synergy` | unchanged |
| Over-holding (discard risk >7) | `discard_penalty` | unchanged |

**[DECIDE]** oracle strength: 1-ply value vs AlphaBeta depth-3 (stronger, slower).
**[DECIDE]** regret threshold that gates a decision into qualitative review.

## 4. The qualitative rubric (LLM, runs on gated decisions)

For each gated decision the LLM grader receives the board, the decision, the
model's **stated reasoning**, AND the oracle context (regret + best action), then
produces:

- **category** — the failure-mode label (§6 taxonomy)
- **explanation** — why it was a mistake, in Catan terms
- **reasoning-consistency** — did the stated reasoning match/justify the action?
  (catches "confidently wrong" and "lucky-right")

**Qual-only dimensions the value fn can't see:**
- Reasoning quality / consistency
- Trade & port EV (1v1 maritime trades)
- Tempo / race awareness (ahead vs behind → risk posture)
- Dev-card timing (when to buy / play knight / road-building)

## 5. Integration — how quant + qual combine

**Pipeline:** `regret oracle → gate → dual LLM grade (with oracle context) → agreement 2×2 → aggregate by regret`

The agreement matrix is the point of combining — disagreements are signal:

| | LLM says **bad** | LLM says **fine** |
|---|---|---|
| **High regret** | ✅ high-confidence failure → env gen | ⚠️ value-fn blind spot or lenient grader → calibrate |
| **Low regret** | 💡 "lucky-right": OK move, bad reasoning → reasoning failure | skip |

- Store `regret`, `llm_quality`, `reasoning_consistency`, `agreement_cell` as
  **separate fields** — do not collapse to one number.
- Sample some **low-regret** decisions for qual grading too, to catch lucky-right
  and to audit the oracle.

**Compute notes:** regret only on decision nodes (post-roll chance nodes make it
ill-defined); evaluating each legal action is cheap (Catanatron is ms-fast).

## 6. Failure-mode taxonomy (starter — edit me)

Categories double as the `weakness` field in the scenario schema:

- `placement-low-pip` — settlement/city on weak production vs a better option
- `placement-no-variety` — ignored resource diversity / port synergy
- `boxed-in` — failed to keep expansion open
- `robber-not-denying-leader` — robber/knight not used to block the leader's best tile
- `robber-victim-suboptimal` — stole from the wrong opponent
- `overheld-cards` — sat on >7, took avoidable discards
- `inefficient-hand` — held resources without converting to a build
- `bad-trade` — maritime/port trade with poor EV
- `longest-road-ignored` — neglected longest road when expansion was blocked
- `dev-timing` — bought/played dev cards at the wrong time
- `reasoning-inconsistent` — stated reasoning didn't match the action (qual-only)
- `tempo-misread` — wrong risk posture for ahead/behind

**[DECIDE]** final taxonomy + which map to v1 scenario `env`s.

## 7. Output schemas

**Per-decision finding** (one per graded decision):
```json
{
  "game_id": "seed42_norm", "ply": 47, "turn": 22, "color": "RED",
  "action": ["RED","MOVE_ROBBER",["...",null]],
  "regret": 0.83, "oracle_best": ["RED","MOVE_ROBBER",["...","BLUE"]],
  "category": "robber-not-denying-leader",
  "llm_quality": 0.2, "reasoning_consistency": 0.4,
  "agreement_cell": "high_regret__llm_bad",
  "graders": {"claude": {...}, "openai": {...}},
  "explanation": "Behind on VP; should have blocked BLUE's 8-ore..."
}
```

**Aggregated failure mode** (the deliverable to env generation):
```json
{
  "category": "robber-not-denying-leader",
  "frequency": 37, "total_regret": 21.4, "roi_rank": 1,
  "example_plies": ["seed42_norm:47", "seed7_swap:88"],
  "representative_explanation": "...",
  "grader_agreement": 0.86
}
```
Ranked by `total_regret` (= highest ROI to fix), which is the objective
prioritization, not an LLM guess.

## 8. Dual-grader reconciliation

- Both graders score each gated decision independently.
- **Category agreement** → high confidence. **Disagreement** → keep both, flag.
- **[DECIDE]** reconciliation rule: union (don't miss anything) vs intersection
  (high precision) vs weighted vote. Suggest: union for discovery, mark agreement
  level so env-gen can prioritize agreed-upon modes.
- **[DECIDE]** grader models (e.g. `claude:claude-opus-4-8` + `openai:gpt-4o`),
  and whether graders read the human `.log`, the JSON, or both.

## 9. Fan-out / scale

- One subagent per transcript (≈100), each: regret-flag → grade flagged decisions.
- Then a reduce step: dedupe + bucket by category + rank by total regret.
- **[DECIDE]** sampling: grade all gated decisions, or cap per game?

## 10. What already exists vs. to build

**Exists:** transcripts with `decisions[]` (ply/turn/action/VP/reasoning) + replay
viewer; Catanatron value fn + AlphaBeta (the oracle); Claude + OpenAI backends;
the frozen per-decision scenario schema (the env-gen target).

**To build (after sign-off):** regret computation (replay → value each legal
action); the gate; the grading-prompt + rubric; the dual-grader fan-out;
the aggregation/ranking; the failure-mode → scenario handoff.

## 11. Open decisions (collected)

- [ ] Oracle strength (1-ply vs AlphaBeta depth-N)
- [ ] Regret gate threshold + low-regret sampling rate
- [ ] Final failure-mode taxonomy
- [ ] Grader models + reconciliation rule (union/intersection/vote)
- [ ] Graders read log / JSON / both
- [ ] Per-game grading cap (cost vs coverage)
- [ ] How aggregated modes map onto scenario `env`s for generation
