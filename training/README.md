# Placement GRPO — launch guide (wired; you run it)

Trains Qwen2.5-7B to place the four opening settlements well. Approved config:
**reward = normalized**, **weights = pip/res/num 1/1/1**, **50 train boards
(example_pool) / 30 eval boards (grader_games, held out)**.

**Output format: answer-only (no chain-of-thought).** The model replies with ONLY
`<answer>node_N</answer>` — no `<reasoning>` block. CoT was for baseline
weakness-discovery in live play; for GRPO rollouts it only slows generation and
adds reward variance, and we reward the decision not the prose. Training rollouts
and eval BOTH render their prompt from `goldilocks_eval/prompting.py` (one module),
so the format is provably identical and the baseline-vs-trained comparison can't be
confounded by a format mismatch.

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

## 3. GRPO run (Fireworks RFT — GRPO on their GPUs, our reward fn grades rollouts)
- **Base model:** `accounts/fireworks/models/qwen3-4b-instruct-2507`.
  ⚠️ NOT Qwen2.5 — **no Qwen2.5 model is RL-trainable on Fireworks** (all are
  `rlLoraTunable=False`; the RFT API 400s). Only Qwen3 is. qwen3-4b-instruct-2507
  is instruct-tuned, non-thinking (clean fit for answer-only), free RFT (<16B).
- **Evaluator/reward:** discovered from `training/rft/test_placement_rft.py`
  (`@evaluation_test`). MUST live in an isolated dir (`training/rft/`) with its own
  `requirements.txt: eval-protocol` and NOTHING that imports catanatron — Fireworks
  pytest-collects the whole upload, so a catanatron import anywhere fails the
  evaluator build with an opaque `INTERNAL` error.
- **Dataset:** `data/placement_opening_train.trl.jsonl` (200 rows; copied into
  `training/rft/data/` so the isolated bundle is self-contained).
- **GRPO hyperparameters (first run):** group size (`response-candidates-count`) 8,
  temperature 0.9, lr 1e-6, 3 epochs over 200 rows, max-output-tokens 16
  (answer-only needs few tokens). Scale boards (`--n`) once the loop trains.

Launch (`eval-protocol` — uploads evaluator + dataset, builds, then creates job):
```bash
set -a; source ../../.env; set +a        # FIREWORKS_API_KEY (run from training/rft/)
cd training/rft
eval-protocol create rft \
    --training-config-base-model accounts/fireworks/models/qwen3-4b-instruct-2507 \
    --training-config-output-model placement-opening-grpo \
    --training-config-epochs 3 --training-config-learning-rate 1e-6 \
    --inference-parameters-temperature 0.9 \
    --inference-parameters-response-candidates-count 8 \
    --inference-parameters-max-output-tokens 16 \
    --dataset data/placement_opening_train.trl.jsonl \
    --force --skip-validation --yes
```
`--skip-validation` skips the local litellm rollout test (needs Fireworks litellm
creds; redundant — the evaluator logic is unit-tested in `test_placement_rft.py`
and cross-checked vs the canonical reward over 10k pairs). `--force` overwrites the
prior evaluator build.

**W&B is optional** and we're not using it — monitor via the Fireworks RFT
dashboard (job link printed on create). For W&B, add `--wandb-config-enabled
--wandb-config-project <p> --wandb-config-entity <e>` and set `WANDB_API_KEY`.
**Never commit any API key.**

When training finishes, deploy the resulting LoRA as an on-demand deployment and
note its deployment id for the AFTER eval below.

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
