# Overnight autonomous GRPO chain — MORNING REPORT ✅

All three envs trained on one fork `catan-grpo-q8b` (Qwen3-8B, Tinker via HUD),
sequential warm-start chain. Chain complete 05:42. **All three show a clear climb.**

## Results — before → after (reward on the 16 training boards per env)
| env | metric | before | after | what it learned |
|---|---|---|---|---|
| **placement** | top-3 hit rate | 0.68 | **1.000** | picks an optimal opening spot every time (perfect by step 29) |
| **maritime** | mean reward | −0.46 | **+0.10** | stopped over-trading — crossed net-negative (churn) to net-positive |
| **build** | mean reward | 0.64 | **1.46** | builds instead of hoarding (128/128 rollouts by step 3) |

Trained model: `catan-grpo-q8b` (id 7c330c53-bf45-45f5-94fa-265e697a45e8),
active checkpoint **step-104**. https://hud.ai/models/7c330c53-bf45-45f5-94fa-265e697a45e8?tab=checkpoints

## What worked
- **HUD/Tinker** sidestepped the Fireworks RFT platform bug entirely (managed
  trainer, our self-contained graders as the reward).
- **lr 4e-5** (1e-5 was too timid — loss collapsed, no movement).
- **Lean mechanics-only prompts** (dropped the 1841-tok from-scratch primer): 2× faster
  AND the model got better (placement 0.30→0.65 baseline just from the cleaner prompt).
- **Top-3 binary / pip-first** placement reward; maritime & build use their structured
  rewards (enable/churn; build/hoard) — all 100% within-scenario variance (smoke-gated).
- Step time ~45–55s (rollout-bound on the gateway). Tinker held up — no env hit the
  4-skip abort.

## Timing
Started ~02:30, chain done 05:42 (~3h, within the 3–4h window). 104 optim steps total.

## Honest caveats / follow-ups
1. **Before/after is on the TRAINING boards** (step 0 vs final) — proof each env
   learned. A held-out generalization eval needs a small HUD-gateway eval backend
   (our `*_env.py eval` targets Fireworks); ~30 min to add — recommended next.
2. Warm-start chain: each env's "before" is after prior envs' training (curriculum).
   A from-base eval per env would isolate each — same HUD-gateway backend.
3. Log cosmetic: per-step "top3 <x>" is really mean-reward for maritime/build.
4. Data is 16 fixed boards/env (fast, clear climb). Scale `--limit` up for breadth.

## How to use the trained model
```bash
set -a; source .env; set +a
# sample it via the HUD gateway (OpenAI-compatible), model "catan-grpo-q8b"
# or inspect runs: hud jobs ; hud jobs <id> ; hud trace <id> ; https://hud.ai/jobs
```
Re-run any env: `cd hud_training && ../.venv-hud/bin/python train.py --env <placement|maritime|build> --steps 25 --limit 16 --group 8 --max-concurrent 48 --learning-rate 4e-5`
