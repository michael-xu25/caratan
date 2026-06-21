# Placement GRPO — launch guide (wired; you run it)

Trains Qwen2.5-7B to place the four opening settlements well. Approved config:
**reward = normalized**, **weights = pip/res/num 1/1/1**, **50 train boards
(example_pool) / 30 eval boards (grader_games, held out)**. GRPO is **not** kicked
off here — this doc + the wired pieces are for you to launch on Fireworks.

## 1. Generate data (already run; regenerate after tuning weights)
```bash
# rich scenarios (canonical; also used by eval) — 50 boards x 4 = 200 scenarios
python -m goldilocks_eval.placement_env generate \
    --split example_pool --n 50 --out data/placement_opening_train.jsonl
# reward-kit / TRL rows: prompt (mechanics only) + ground_truth (node->score)
python -m goldilocks_eval.placement_env traindata \
    --in data/placement_opening_train.jsonl --out data/placement_opening_train.trl.jsonl
```
Tuned the weights in `goldilocks_eval/placement_score.py`? Re-run both commands.

Each TRL row:
```json
{"id": "1000_o1",
 "prompt": [{"role":"system","content":"<rules...>"},
            {"role":"user","content":"<board + legal spots, mechanics only>"}],
 "ground_truth": {"spot_scores": {"node_15": 2.867, ...}, "gold": "node_15"}}
```

## 2. Reward function
`training/placement_reward.py` → `placement_reward_fn` (reward-kit
`@reward_function`). It parses the model's `<answer>node_N</answer>`, looks up the
row's `ground_truth.spot_scores`, and returns `EvaluateResult(score=...)` where
`score` is the normalized reward (1.0 = optimal spot, 0.0 = worst). Reward-kit
entry point: `training.placement_reward:placement_reward_fn`.

Sanity-check the reward on the dataset before training (no GPUs):
```bash
pip install reward-kit
reward-kit run --reward training.placement_reward:placement_reward_fn \
    --dataset data/placement_opening_train.trl.jsonl     # preview scores
```
(Exact `reward-kit` subcommand/flags per its README — `fw-ai-external/reward-kit`,
`examples/math_example`. The function signature + dataset shape above are correct;
adapt the CLI to the installed version.)

## 3. GRPO run (Fireworks RFT — follow the HUD `fireworks-rl-training` cookbook)
- **Base model:** the deployed Qwen2.5-7B (`fireworks:$FIREWORKS_MODEL` =
  `accounts/brickedup25/deployments/qhzroqz3`), or the public
  `accounts/fireworks/models/qwen2p5-7b-instruct` for training.
- **Reward:** `training.placement_reward:placement_reward_fn`.
- **Dataset:** `data/placement_opening_train.trl.jsonl` (200 rows; the four
  placement positions are mixed in).
- **Suggested GRPO hyperparameters (first run):** group size G = 8–16,
  temperature 0.8–1.0 (the group needs variance), lr ≈ 1e-6, 2–4 epochs over the
  200 rows. Start small; scale boards (`--n`) once the loop trains.

Per the build spec this runs on Fireworks RFT (rollouts + GRPO on their GPUs, the
reward runs as the Python function above). Wire it via the cookbook; **do not**
commit any API key.

## 4. Eval — before/after (the demo)
Run the SAME held-out boards on the base and the trained model; each places all
four openings itself and is scored vs the optimum at each decision:
```bash
set -a; source .env; set +a
# BEFORE (baseline)
python -m goldilocks_eval.placement_env eval --split grader_games --n 30 \
    --model fireworks:$FIREWORKS_MODEL
# AFTER (point at the trained deployment)
python -m goldilocks_eval.placement_env eval --split grader_games --n 30 \
    --model fireworks:accounts/brickedup25/deployments/<trained-id>
```
Reports mean reward per placement (1–4) + overall — a 4-row before/after table
showing every opening improving.

## Extending later (kept structured for it)
- Second-settlement complementarity: `snake_player` is already recorded per
  scenario; add a complementarity term in `placement_score.py` keyed on the
  player's prior settlement(s). Schema unchanged.
- Mid-game envs: same pattern (scenario + scored legal set + reward), new scorer.
