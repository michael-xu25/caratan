# Goldilocks × Catan — Eval Harness

Hud RL-environments hackathon. We're proving one loop:
play games → identify model weaknesses (via Claude + GPT-5.5) → auto-generate
verifiable envs targeting them → GRPO-train a small open LLM → measure improvement.

This repo = the **eval/measurement half**. Trained artifact is a small
open-weights model (Gemma E4B, fallback Qwen3-4B). The analyst/judge models
(Claude, GPT-5.5) are API models we don't train. The agent interface is
model-agnostic (swap by config flag). No self-play training; self-play is eval-only.

## Setup
- Env: `catanatron/` (vendored). `pip install -e catanatron` + gym.
- Format: **1v1 throughout** (low variance, clean head-to-head).

## What we're building
- Async match runner: two agents head-to-head, model-agnostic agent
  interface (swap Gemini/Claude/small model by config flag).
- Interpretable transcripts: JSON via Catanatron's GameEncoder + a human-readable
  log — each decision records the full legal option set it chose among, VP
  context, the model's reasoning, and the outcome (so weaknesses are analyzable).
- Fairness: **seeded purely-random** board+dice (not a balanced deck — mirroring
  cancels luck), **mirrored games** (same seed both ways, seats swapped),
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

### Scenario pipeline (placement env)

```bash
# 1. Emit unlabeled placement scenarios (STOPGAP sample emitter — the real
#    pooled generator is the producer half, see scenario-generation-spec.md)
python -m goldilocks_eval.sample_scenarios --start 0 --n 20 \
    --split heldout --out data/placement_unlabeled.jsonl

# 2. Champion-label them (board render + node-id overlay; resumable)
python -m goldilocks_eval.labeling_cli data/placement_unlabeled.jsonl \
    --out data/placement_labeled.jsonl

# 3. Calibrate to the Goldilocks band (train pool); record base rate (heldout)
python calibration_harness.py data/placement_labeled.jsonl \
    --out data/placement_train.jsonl --filter --low 0.2 --high 0.5
```

The **prompt/answer/reward contract is one source of truth** —
`goldilocks_eval/prompting.py` (`build_prompt`, `parse_answer`, `score`,
`<answer>node_27</answer>` format). Generation, calibration, and the eval all
import it, so before/after numbers can't silently compare apples to oranges.

### Frozen scenario contract + fixtures
The scenario record is frozen in `goldilocks_eval/schema.py` (`Scenario`,
`new_unlabeled`, `apply_label` for the UI write-back, `validate`,
`json_schema`). Real example fixtures (unlabeled, both pick-1 and
existing-settlement cases) live in `data/examples/placement_examples.jsonl`,
with the JSON Schema and a build guide in `data/examples/README.md`.

Node ids resolve to board positions from `serialized_state` alone (each node
carries `tile_coordinate` + `direction`) — `goldilocks_eval/geometry.py`
(`node_position`) is the resolver, with the formula/JS-mirror notes in the
fixtures README. No serialization change is needed for the labeling UI.

Agent specs (the swappable backend flag): `random`, `weighted`, `value`,
`alphabeta[:depth]`, `claude[:model-id]` (default `claude-opus-4-8`),
`gemini` (stub — implement `LLMBackend.complete` to wire it in).

Each match prints head-to-head win-rate and writes per-game transcripts to
`transcripts/` — a full JSON state dump (Catanatron's `GameEncoder`) plus a
human-readable log: every action, and per LLM decision the legal option set it
chose among, both players' VP, and the model's reasoning.

### Layout
- `goldilocks_eval/agents/` — `LLMBackend` interface, `LLMPlayer`, baseline
  factory, Claude backend.
- `goldilocks_eval/runner.py` — async seeded + mirrored head-to-head runner
  (concurrency = max in-flight LLM calls).
- `goldilocks_eval/scenario.py` + `scenario_cli.py` — per-weakness scenario
  scorer (tiered 1.0/0.5/0.0 vs champion labels) and before/after report.
- `goldilocks_eval/prompting.py` — **canonical** placement prompt/parse/reward
  contract (shared by generation, calibration, and eval).
- `goldilocks_eval/labeling_cli.py` — champion-labeling CLI (board render +
  node-id overlay, resumable).
- `goldilocks_eval/sample_scenarios.py` — stopgap unlabeled-scenario emitter.
- `goldilocks_eval/prompt.py` — live full-game rendering/parsing (index-based,
  arbitrary actions — distinct from the placement contract above).
- `goldilocks_eval/transcript.py` — JSON + human-readable transcript writers.
- `goldilocks_eval/schema.py` — **frozen** scenario record (canonical, both
  directions); `goldilocks_eval/geometry.py` — node-id → board position resolver.
- `calibration_harness.py` — Goldilocks difficulty filter (drops zero-variance
  scenarios); wire its backend to Fireworks/your `LLMBackend`.

### Project docs
- `build-spec-decisions.md` — locked decisions (the master log).
- `scenario-generation-spec.md` — producer-half generator contract.
- `weakness-discovery-guidance.md` — eval → two independent analysts
  (Claude Opus 4.8 + GPT-5.5) → oracle/champion-verified weakness workflow.
- `data/examples/README.md` — frozen schema, fixtures, and the node→position
  recipe the labeling UI builds against.

### Notes / contract
- **Dice = seeded purely-random** (`Game(seed=…)`), per the build-spec decision —
  *not* a balanced deck. Mirroring already cancels dice luck, and a draw-without-
  replacement deck would be countable (an artifact real Catan lacks).
- The scenario record is frozen in `goldilocks_eval/schema.py` (canonical), with
  the JSON Schema + UI build guide in `data/examples/`. Generating +
  champion-labeling scenarios is the producer half; this repo consumes them.
