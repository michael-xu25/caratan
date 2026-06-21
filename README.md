# Goldilocks × Catan — Eval Harness

Hud RL-environments hackathon. We're proving one loop:
play games → identify model weaknesses (via Claude) → auto-generate
verifiable envs targeting them → GRPO-train an LLM → measure improvement.

This repo = the **eval/measurement half**. Trained artifact is an LLM
(e.g. Qwen 7B / Gemini / Claude — backend is swappable). "Untrained
version" = the base model. No self-play training; self-play is eval-only.

## Setup
- Env: `catanatron/` (vendored). `pip install -e catanatron` + gym.
- Format: **1v1 throughout** (low variance, clean head-to-head).

## What we're building
- Async match runner: two agents head-to-head, model-agnostic agent
  interface (swap Gemini/Claude/small model by config flag).
- Readable transcripts: JSON via Catanatron's GameEncoder + a
  human-readable log (board, each decision + the model's reasoning, outcome).
- Fairness: seeded board+dice, **mirrored games** (same seed played
  both ways with seats swapped to cancel luck), balanced dice deck,
  held-out eval seeds.
- Parallelism ceiling = concurrent LLM calls (Catanatron is ms-fast;
  the bottleneck is model throughput), so build the runner async.

## Metrics
- Primary: per-weakness before/after accuracy on held-out instances.
- Secondary: mirrored full-game head-to-head. (Watch catastrophic
  forgetting — head-to-head can regress while the target decision improves.)

## Baselines in Catanatron
RandomPlayer, WeightedRandomPlayer, ValueFunctionPlayer, AlphaBetaPlayer.
