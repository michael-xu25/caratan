# Baseline placement eval — BEFORE training

- Model: `accounts/brickedup25/deployments/blpxetwj` (Qwen2.5-7B, untrained)
- Boards: 30 held-out (`grader_games`), reward mode: normalized
- Output format: answer-only (same as GRPO rollouts — no format confound)

| opening | snake | mean reward (1.0 = optimal) |
|---|---|---|
| 1 | A | 0.717 |
| 2 | B | 0.791 |
| 3 | B | 0.698 |
| 4 | A | 0.761 |
| **overall** | | **0.742** |

Re-run the identical command on the trained deployment for the AFTER row:
`python -m goldilocks_eval.placement_env eval --split grader_games --n 30 --model fireworks:<trained-id>`
