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

## Running (MVP)

Catanatron needs **Python ≥ 3.11**. Set up and install:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e catanatron        # the vendored env
pip install -r requirements.txt  # anthropic SDK (only needed for the Claude agent)
```

**Secondary metric — mirrored head-to-head** (baselines need no API key):

```bash
# value-function bot vs weighted-random, 10 seeds, mirrored (seats swapped)
python -m goldilocks_eval --a value --b weighted -n 10

# Claude vs a baseline (export ANTHROPIC_API_KEY first)
python -m goldilocks_eval --a claude --b value --seeds 5 --concurrency 4
```

**Primary metric — per-weakness before/after on held-out scenarios** (consumes
the shared scenario JSONL contract; LLM backends only, since a frozen scenario
is graded from its serialized state, not a live game):

```bash
python -m goldilocks_eval.scenario_cli \
    --scenarios data/placement_heldout.jsonl \
    --before claude:claude-haiku-4-5 --after claude
# -> per-env accuracy + headline delta, e.g. placement 30% -> 78%
```

Agent specs (the swappable backend flag): `random`, `weighted`, `value`,
`alphabeta[:depth]`, `claude[:model-id]` (default `claude-opus-4-8`),
`gemini` (stub — implement `LLMBackend.complete` to wire it in).

Each match prints head-to-head win-rate and writes per-game transcripts to
`transcripts/` — a full JSON state dump (Catanatron's `GameEncoder`) plus a
human-readable log with every action and the model's per-decision reasoning.

### Layout
- `goldilocks_eval/agents/` — `LLMBackend` interface, `LLMPlayer`, baseline
  factory, Claude backend.
- `goldilocks_eval/runner.py` — async seeded + mirrored head-to-head runner
  (concurrency = max in-flight LLM calls).
- `goldilocks_eval/scenario.py` + `scenario_cli.py` — per-weakness scenario
  scorer (tiered 1.0/0.5/0.0 vs champion labels) and before/after report.
- `goldilocks_eval/prompt.py` — state→text rendering and reply parsing.
- `goldilocks_eval/transcript.py` — JSON + human-readable transcript writers.

### Notes / contract
- **Dice = seeded purely-random** (`Game(seed=…)`), per the build-spec decision —
  *not* a balanced deck. Mirroring already cancels dice luck, and a draw-without-
  replacement deck would be countable (an artifact real Catan lacks).
- The scenario scorer reads the JSONL schema in `build-spec-decisions.md`
  (`serialized_state`, `legal_actions`, `gold_action`, `acceptable_actions`,
  `split`). Generating + champion-labeling those scenarios is the other half of
  the loop; this repo consumes them.
