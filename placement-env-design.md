# Placement RL environment — design (for approval before GRPO)

Trains the model to place the four opening settlements well. Mechanics-only
prompt (identical for baseline & trained); all judgment lives in the reward. See
`words-vs-rl.md`.

## 1. Scoring = the reward ground truth (TUNABLE — your knob)
`goldilocks_eval/placement_score.py`. A spot is scored from the tiles it borders,
three championship criteria, each normalized to ~[0,1] so weights are comparable:

| criterion | meaning | weight (default) |
|---|---|---|
| `pip` | total dice-probability (pips) of bordering tiles ÷ 15 | 1.0 |
| `resource_diversity` | distinct resource types ÷ 3 | 1.0 |
| `number_diversity` | distinct dice-numbers ÷ #tiles (anti-stacking) | 1.0 |

`score = w_pip·pip_norm + w_res·resource_diversity + w_num·number_diversity`

**This is the only thing you need to tune.** Adjust `WEIGHTS` (and `MAX_PIPS`),
re-run `show`, regenerate. Sanity check on seed 1000 (default weights):

```
opening 1: GOLD node_15  score 2.867  [WOOD6, WHEAT5, SHEEP9]  13 pips, 3 resources, 3 numbers
double-brick spot node_22  score 2.400  (penalized: resdiv 0.67)
desert-only spot           score 0.000  (correctly worst — the blunder we saw)
```
Inspect any board: `python -m goldilocks_eval.placement_env show --seed <N>`

Note with equal weights, among fully-diverse spots the ranking is driven by pips
(most strong spots already have resdiv=numdiv=1.0). If you want diversity to
matter more vs. raw production, raise `resource_diversity`/`number_diversity` or
lower `pip`.

## 2. Reward (proposed: `normalized`)
`placement_reward(chosen, scored, mode)` → [0,1]. Three modes; **I recommend
`normalized`** for GRPO:
- **`normalized`** (default): `(chosen−worst)/(best−worst)`. 1.0 = optimal spot,
  0.0 = worst. Uses the full range each decision, so even small quality gaps give
  a gradient — and GRPO is group-relative, so per-decision rescaling is fine.
- `ratio`: `chosen/best` (gentler, compresses the low end).
- `rank`: `1 − rank/(n−1)` (purely ordinal).

## 3. Four placements, snake order A,B,B,A (`placement_env.py`)
Drives a real Catanatron opening. At each of the 4 settlement decisions: take the
legal remaining spots Catanatron offers, score each, record a scenario with
`gold_action` = best + all `spot_scores`, then place the greedy-best and continue
(forced initial roads auto-played). So **later placements are graded against the
correctly-reduced option set**, exactly as asked.
- **v1: each placement scored independently** (no cross-settlement complementarity).
  Prior placements are the greedy-best, so each decision = "given an optimal-so-far
  board, pick the best remaining." `snake_player` (A/B/B/A) is recorded as metadata
  so per-player complementarity can be added later without a schema change.
- Model sees `prompting.build_prompt` (board + legal spots + production/pips/ports).
  **Verified: scores/gold/weights never appear in the prompt.**

## 4. Data + split (leak-free)
- **Train**: `example_pool` (300 boards). Proposed first run **N = 50** boards →
  200 scenarios (4 per board). `generate --split example_pool --n 50 --out data/placement_opening_train.jsonl`
- **Held-out eval**: `grader_games` (100 boards, disjoint from example_pool by the
  meta-seeded shuffle — no leakage).

## 5. Eval (shows all four improving)
`eval --split grader_games --n 30 --model <spec>` lets the model place all four
openings itself and reports **mean reward per placement (1–4)** + overall. Run it
on the base model and the trained model on the same held-out boards → a 4-row
before/after table proving each opening improved.

## 6. GRPO wiring (sketch — NOT launched; needs your go + the Fireworks cookbook)
- Each training example: `prompt = build_prompt(scenario)`, plus the scenario's
  `spot_scores` carried as reward context (keyed by node id).
- Reward fn (reward-kit): `placement_reward(parse_answer(completion), spot_scores, mode)`.
- GRPO samples G completions/prompt; group-relative advantage from the rewards;
  trains across all four placement positions (they're just scenarios in the set).
- Temperature > 0 so the group has variance to learn from.
- Base model = `fireworks:$FIREWORKS_MODEL` (Qwen2.5-7B); train, then eval the
  trained deployment with the same `eval` command.

## Approve to proceed
1. **Weights** — keep `pip/res/num = 1/1/1`, or tune? (run `show` on a few boards)
2. **Reward mode** — `normalized` ok?
3. **N** — start at 50 train boards / 30 eval boards?
4. **Go for GRPO?** (I'll wire the reward-kit reward + training config to the
   Fireworks cookbook and hand it to you to launch — I won't kick it off.)
