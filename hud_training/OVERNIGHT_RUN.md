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

## HELD-OUT before/after (grader_games split, disjoint from training)
Fair comparison: both models forced no-think (`/no_think`) so we measure PICK
quality, not format (base Qwen3-8B otherwise reasons forever and never answers —
its raw "as-deployed" score is ~0, an even bigger but less fair gap). Deterministic.
`hud_training/eval_holdout.py`.

| env (n) | BEFORE (base) → AFTER (trained) |
|---|---|
| placement (80) | top-1 29%→**49%**, top-3 41%→**78%**, regret 0.125→**0.044** |
| build (90) | build-rate 78%→**97%**, mean reward 0.93→**1.38** |
| maritime (90) | trade-rate 20%→**3%**, mean reward −0.03→**+0.004** |

- **placement, build: clear generalization** to unseen boards.
- **maritime: weak** — it learned to trade far less (the over-trading target), but the
  mean-reward gain is marginal and 3% trade-rate may be over-corrected (skips good
  trades too). Candidate for reward retuning (raise enable/progress vs churn).

## Honest caveats / follow-ups
1. **Before/after is on the TRAINING boards** (step 0 vs final) — proof each env
   learned. A held-out generalization eval needs a small HUD-gateway eval backend
   (our `*_env.py eval` targets Fireworks); ~30 min to add — recommended next.
2. Warm-start chain: each env's "before" is after prior envs' training (curriculum).
   A from-base eval per env would isolate each — same HUD-gateway backend.
3. Log cosmetic: per-step "top3 <x>" is really mean-reward for maritime/build.
4. Data is 16 fixed boards/env (fast, clear climb). Scale `--limit` up for breadth.

## DEPLOYED for Cara (gateway-queryable by name)
Decision: **ship 2 envs (placement + build)** — the generalizing wins. Maritime is
kept in the chain but not pursued further: discovery showed it *over-corrected*
(takes a productive trade only 2% vs base 10%) and the env has thin trainable signal
(not-trading is near-optimal on sampled states). The order was placement→maritime→
**build**, so the post-build model includes all three.

| model name | = checkpoint | use |
|---|---|---|
| `Qwen/Qwen3-8B` | base (untrained) | BEFORE baseline |
| `catan-placement-only` | step-54 (post-placement) | placement-only |
| `catan-grpo-q8b` | step-104 (post-build, full chain) | the shipped model |
| ~~`catan-postplacement`~~ | (accidental dup of post-build — **ignore**) | n/a |

Verified differ as expected: placement identical (no forgetting); build
post-placement 1.12/87% → post-build **1.38/97%**.

**Cara — run evals** (held-out grader_games scenarios, both no-think, fair):
```bash
cd hud_training && set -a; source ../.env; set +a
../.venv-hud/bin/python eval_holdout.py \
    --models Qwen/Qwen3-8B catan-placement-only catan-grpo-q8b
# add --envs placement build  to pick envs; data in hud_training/data/*_eval.trl.jsonl
```
Or query any model directly: HUD gateway `https://inference.beta.hud.ai/v1` (OpenAI-
compatible), model = the name above, `HUD_API_KEY` from `.env`. Append `/no_think`
to the user message so the base answers directly.

## HUD run stats (the whole session on the fork)
- **104 optim steps**, **12,979 rollouts** (datums), across **12 jobs** (smokes +
  the real runs), **123 min** of active training (10:39–12:42).
- Tokens (est): ~19.5M in / ~454k out → **~$4.6 inference** for rollout sampling
  (Tinker training/gradient compute billed separately).
- Step time ~45–55s, rollout-bound on the HUD→Tinker gateway (~85% rollout / 15%
  train); concurrency 48.
- **Caveat on the raw checkpoint reward curve** (`hud models checkpoints`): the per-
  step reward is NOT comparable across the session because the reward function
  changed during iteration (normalized → sharpness → top-3-binary; pip 1.0→1.5→3.0).
  The clean per-phase climbs are the FINAL runs only (the Results tables above).
  Visible artifacts in the curve: steps 9–14 (lr=1e-5) reward fell 0.79→0.27 — the
  timid-lr failure; step 55 = −0.46 = maritime's over-trading baseline.

## How to use the trained model
```bash
set -a; source .env; set +a
# sample it via the HUD gateway (OpenAI-compatible), model "catan-grpo-q8b"
# or inspect runs: hud jobs ; hud jobs <id> ; hud trace <id> ; https://hud.ai/jobs
```
Re-run any env: `cd hud_training && ../.venv-hud/bin/python train.py --env <placement|maritime|build> --steps 25 --limit 16 --group 8 --max-concurrent 48 --learning-rate 4e-5`
