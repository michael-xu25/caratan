# Overnight autonomous GRPO chain — run log

Train `catan-grpo-q8b` (Qwen3-8B fork) sequentially on placement → maritime →
build (warm-start chain on one fork). Get a clear reward climb on each, like
placement. Running while Michael sleeps; no sign-offs ~3–4h.

## Guardrails (all in place)
- One fork, strictly sequential (chain waits for placement to exit before maritime,
  etc. — never concurrent). lr 4e-5, group 8, ≤25–30 steps/env, concurrency 48.
- Tinker 503 → retry w/ backoff; if a step can't get a full group, it's skipped;
  **4 skips in a row aborts that env** (no infinite spin).
- Offline smoke-gate passed for all 3 (100% within-scenario reward variance).
- No secrets committed, no deploys.

## How to watch
- `hud jobs` / `hud jobs <id>` / `hud trace <id>`; web https://hud.ai/jobs
- training logs: `hud_training` background tasks; reward printed per step.

## Results — before → after (top-3 / mean reward on the training boards)
| env | metric | before | after | notes |
|---|---|---|---|---|
| placement | top-3 hit | 0.68 | **0.91+** | climbed cleanly; lr 4e-5 was the fix |
| maritime | mean reward | — | — | chain queued (churn −1.1 / no-trade 0 / enable +1) |
| build | mean reward | — | — | chain queued (pass −0.96 / build-city +1.5) |

## Pipeline (each env)
generate (catanatron) → traindata (lean prompt + baked per-option scores) →
self-contained HUD grader (index/node → reward) → smoke-gate → GRPO on the fork.

## TODO at end
- Record maritime/build before→after when the chain finishes.
- Held-out before/after eval needs a HUD-gateway backend (eval currently targets
  Fireworks); training-board climb is the primary evidence for now.

## Status (newest first)
- [running] chain orchestrator `brzv9anu3`: waits for placement, then maritime, then build.
- [running] placement `bk0p3bkog` @ lr 4e-5: 0.68 → 0.906 (step 22/30), still climbing.
