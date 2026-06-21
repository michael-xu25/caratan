# Overnight autonomous GRPO chain — run log

Goal: train `catan-grpo-q8b` (Qwen3-8B fork) sequentially on placement → maritime →
build (warm-start chain on one fork), get a clear top-3 climb on each like placement.
Started while Michael is asleep; no sign-offs for ~3–4h.

## Guardrails
- One fork, sequential (never concurrent). Smoke-gate each env (abort if no reward
  variance / Tinker down). ≤30 steps & ≤45 min/env. lr 4e-5. 503-retry. No secrets
  committed, no deploys.

## Status (newest first)
- [running] **placement** @ lr 4e-5, 16 boards, group 8 — climbing 0.68 → 0.805 (step 7).
  Building maritime + build-env graders offline while it trains.

## Results (before → after top-3 on training boards)
| env | before | after | steps | notes |
|---|---|---|---|---|
| placement | 0.68 | … | … | in progress |
| maritime | — | — | — | building |
| build | — | — | — | building |
