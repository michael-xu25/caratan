# Baseline placement eval — BEFORE training

- Model: `accounts/fireworks/models/qwen3-4b-instruct-2507` (the RFT base model)
- Boards: 30 held-out (`grader_games`), reward mode: normalized
- Output format: answer-only (same as GRPO rollouts — no format confound)

| opening | snake | mean reward (1.0 = optimal) |
|---|---|---|
| 1 | A | 0.799 |
| 2 | B | 0.739 |
| 3 | B | 0.770 |
| 4 | A | 0.770 |
| **overall** | | **0.770** |

Re-run the identical command on the trained deployment for the AFTER row:
`python -m goldilocks_eval.placement_env eval --split grader_games --n 30 --model fireworks:<trained-id>`

### Why Qwen3, not Qwen2.5
The project model is Qwen2.5-7B-Instruct, but **no Qwen2.5 model is RL-trainable on
Fireworks** (all are `rlLoraTunable=False`/`rlFullParameterTunable=False`; the RFT
API rejects them with a 400). Only the Qwen3 family is RL-trainable. We picked
`qwen3-4b-instruct-2507`: instruction-tuned, **non-thinking by design** (clean fit
for our answer-only format), RL LoRA+full tunable, free RFT (<16B).
For reference, the original Qwen2.5-7B deployment (`blpxetwj`) scored overall 0.742
(0.717/0.791/0.698/0.761) on the same boards — not the training base, kept only as
a data point.
