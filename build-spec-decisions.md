# Goldilocks × Catan — Master Decision Log / Build Spec

*Everything we've locked, for driving the build. My half = scenario generation + training. Cara's half = eval infra (separate doc). Shared contract = the scenario schema below — both sides build to it.*

---

## The loop (what we're proving)
Play games → identify model weaknesses (via a big API model, e.g. Claude) → auto-generate verifiable envs targeting them → GRPO-train a small open LLM → measure improvement on held-out spots. Catan = petri dish; champion (me, #1) = the legibility layer that makes improvement visible. **The loop is the product.**

- **No self-play as a training method** (too slow overnight). Self-play is eval-only (trained vs base), with caveats.
- **Multiple envs; placement is the first/floor env** — dice-free, verifiable, champion's edge. Other envs = critical mid-game decision spots (extraction method TBD).

---

## Format & environment
- **Catanatron** (bcollazo), `pip install catanatron` + gym. State via `game.state`, actions via `playable_actions`, serialize with `GameEncoder`.
- **1v1 throughout** — low variance, clean head-to-head, matches balanced-dice competitive ruleset. (Placement in 1v1 = 4 opening settlements; "best remaining spot" still exists at picks 2–4. If we later want richer placement data, we can generate 4p boards for placement only — optional, not now.)
- Board control is native: `Game(players, seed=…, number_placement='official_spiral'|'random', catan_map=…)`. `official_spiral` is the default (canonical token order); `seed` makes any board reproducible.

---

## Dice — DECISION: seeded purely-random (independent rolls)
- **Seeded purely-random rolls** (independent, fixed RNG seed). NOT a balanced deck.
  - Why not a balanced deck: draw-without-replacement is *countable* — once the 7s are spent the agent (or a human) knows none are coming and plays differently. That's an artifact real Catan doesn't have. Colonist avoids it by *reweighting probabilities* (independent rolls, nudged against droughts), which keeps it uncountable — but we agreed not to build that mechanism.
  - Why seeded-random is enough: **mirroring already cancels dice luck regardless of distribution** (same seeded sequence, both seats, luck subtracts out). So the balanced deck's only benefit (variance reduction) is redundant once we mirror. Seeded-random is realistic, artifact-free, and reproducible — which is all mirroring needs.
- **Dice do NOT affect per-decision training** (scenarios are frozen post-roll states; no roll in the rollout). They matter only for (a) realism of generated states and (b) full-game head-to-head eval. Placement env: no dice at all.
- Implementation: just seed the RNG (`Game(players, seed=…)`) — no deck logic. Mirroring handles variance in the full-game eval.

---

## Model — DECISION
- **Trained model:** small open-weights, GRPO. First choice **Gemma 4 E4B-it** (~8GB, Apache 2.0, downloadable, TRL/Unsloth/vLLM support). **Fallback: Qwen3-4B** if E4B isn't supported on the training platform's catalog.
- **Discoverer/judge:** big API model (Claude/Gemini), untrained — no size limit, since we don't train it.
- Hard rule: you can only GRPO a model whose weights you control → the trained model must be open-weights. API models can't be the trained artifact.

---

## Training platform — DECISION: Fireworks
- **Fireworks RFT / Training API:** write the loop + reward in Python locally; rollouts + GRPO forward/backward run on Fireworks' remote GPUs. **No local GPU needed.** Custom Python reward via reward-kit. Free credits.
- **Follow the HUD `fireworks-rl-training` cookbook.** Wrapping our Catan env in HUD's task spec doubles as the generic task interface (for domain #2) AND the sanctioned training path.
- Confirm E4B is in the catalog first; else Qwen3-4B.
- **Demo:** pull the trained checkpoint, run the model **locally** (laptop, offline) if export works; else use the hosted endpoint. Either way, **cache eval results for the stage** — no live inference dependency during the talk.

---

## Scenario data model (shared contract — build to this exactly)

Unit = a critical decision scenario. **Multiple scenarios per game.** One row per scenario:

```json
{
  "scenario_id": "string",
  "game_id": "string",          // grouping key — all scenarios from one board share this
  "board_seed": 12345,
  "env": "placement",           // first env; others added later
  "serialized_state": { },      // Catanatron GameEncoder JSON (the frozen post-roll state)
  "legal_actions": [ ],         // playable_actions at this decision point
  "gold_action": "node_27",     // champion label
  "acceptable_actions": [ ],    // near-optimal alternatives
  "base_solve_rate": 0.25,      // filled during calibration (training pool only)
  "split": "train"              // "train" | "heldout"
}
```

Persist as versioned JSONL in the repo: `data/placement_train.jsonl`, `data/placement_heldout.jsonl`. Regenerable from seeds.

---

## Generation + split — DECISION (this is the part to get exactly right)
1. **Two disjoint board pools up front:** pool A → training scenarios, pool B (~50 boards) → held-out eval scenarios. **No board ever appears on both sides** (split grouped by `game_id`). This prevents leakage through shared board geometry — the flaw in a naive scenario-level split.
2. **Calibrate the training pool only:** for each candidate scenario, sample the **base** model ~8–10× and record `base_solve_rate`. **Keep only ~1–3 of 8 (20–50% / Goldilocks).** Reason: if all rollouts in a GRPO group score identically (always-solve or always-fail), group advantage = 0 → no gradient. We train only where the base is inconsistent.
3. **Do NOT calibrate/filter the held-out eval pool by base failure.** Keep it a representative sample of the decision type, or you bias the before/after upward.

---

## Reward — placement env
Tiered, champion labels = ground truth:
- `1.0` if chosen == gold_action
- `0.5` if chosen ∈ acceptable_actions
- `0.0` otherwise

(Python reward function for reward-kit. Placement reward is pure match — no dice, no playout.)

---

## Eval (Cara owns infra; here's the contract)
- **Primary metric: per-weakness before/after accuracy on held-out scenarios** (e.g. "placement: 30% → 78%"). Scored as the model's decision on a fixed held-out set vs ground truth.
- **Secondary: mirrored full-game head-to-head**, trained vs base, 1v1, seeded purely-random dice, **~100 mirrored pairs** (each board+dice played both ways, seats swapped → cancels luck — which is what makes seeded-random sufficient without a balanced deck).
- **Concurrency:** max it out — 100+ concurrent calls is fine, bounded by serving throughput not hardware. **Sample size:** per-weakness ~50–100 is plenty for a big effect; head-to-head ~100 mirrored pairs.
- **Watch catastrophic forgetting:** head-to-head can regress while the targeted decision improves — per-weakness is the headline, head-to-head is the cherry.
- Agent interface must be **model-agnostic** (swap model by config flag) for ablations.

---

## Build order
1. Catan placement env: generate boards (pool A + pool B), extract opening-placement scenarios, serialize states.
2. Champion labeling pass (me) → gold + acceptable per scenario.
3. Calibration loop on pool A → filter to Goldilocks band.
4. Write JSONL (train/heldout, split by game_id).
5. Wrap env in HUD task spec; wire the Fireworks cookbook reward (tiered match).
6. Smoke-test GRPO on E4B-it (or Qwen3-4B): one scenario, group of 8, confirm it trains.
7. Scale: full placement run → measure before/after on held-out. **Then** turn Claude loose to discover the next weakness.
