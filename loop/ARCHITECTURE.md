# Caratan — Autonomous Self-Improvement Loop

The bot plays itself, finds its own weaknesses, writes new verifiable RL envs to
target them, GRPO-trains on them, and only keeps a checkpoint if it measurably
improves on held-out scenarios. Runs unattended for days on Modal. Progress
streams to caratan.vercel.app.

## Decisions (locked 2026-06-23)

- **Backend:** self-host train + serve on **Modal** (GPU credits). Checkpoints
  are ours — no HUD/Tinker export dependency. Start fresh from `Qwen/Qwen3-8B`.
- **Autonomy:** full — Claude writes NEW env+grader code each round from mined
  weaknesses (not just retraining fixed envs).
- **Monitoring:** auto-updating **caratan.vercel.app** (each round pushes data).
- **Reasoning:** answer-only rollouts (`/no_think`), identical format train↔eval.
- Fallback compute available: $1k GCP, Fireworks, HUD, Anthropic API (env-gen brain).

## Components

| # | Component | Where | Status |
|---|-----------|-------|--------|
| ① | vLLM serving (best LoRA) | Modal GPU | new |
| ② | Self-play (catanatron) | Modal/CPU | reuse `harness/` |
| ③ | Weakness mining | Modal/CPU | extend `goldilocks_eval` regret |
| ④ | Env-gen brain (Claude) | Modal/CPU | new, the autonomous part |
| ⑤ | GRPO round (TRL) | Modal GPU | new (was HUD) |
| ⑥ | Eval + promotion gate | Modal | reuse `eval_holdout` logic |
| ⑦ | Push → Vercel | Modal→git | new |

## Reward plumbing

The existing env graders are already perfect TRL reward functions:
`hud_training/catan_*_env.py::_score(text, ground_truth) -> (reward, reason)`.
The `data/*.trl.jsonl` files are the prompt datasets. The Modal GRPO trainer
imports `_score` directly as the reward fn — no rewrite.

## Guardrails (must survive days unattended)

1. **Grader sanity gate** — every Claude-generated grader must: be deterministic,
   score a known-good answer strictly above a known-bad one, reject unparseable
   input with 0, and have no constant/degenerate output. Fails → env discarded,
   never trains.
2. **Promotion gate** — a new checkpoint replaces the best only if held-out mean
   reward improves (else rolled back). No silent overnight regressions.
3. **Round budget** — hard cap on tokens/GPU-minutes per round; abort+log on OOM
   or trainer divergence; resume from last good checkpoint.
4. **State in a Modal Volume** — checkpoints, active env registry, round logs,
   scenario sets. Loop is resume-safe across restarts.

## Round loop

```
load best checkpoint  →  serve (vLLM)  →  N self-play games
   →  mine weaknesses (regret clusters)
   →  Claude writes/refreshes an env  →  sanity gate
   →  GRPO round on active env set  →  held-out eval
   →  promote if improved, else roll back
   →  push curves+winrate+transcripts to Vercel  →  repeat
```

## Layout

```
loop/
  ARCHITECTURE.md     # this file
  modal_app.py        # Modal app: serve / train / selfplay / eval functions
  orchestrator.py     # the round loop driver (runs on Modal)
  weakness.py         # transcript -> failure-mode clusters
  envgen.py           # Claude env-gen brain + grader sanity gate
  state/              # local mirror of Volume state (gitignored except registry)
```

## Open prerequisites

- [ ] `modal token new` (interactive browser auth) — user
- [ ] confirm Modal GPU tier available (A100/H100 80GB for 8B GRPO)
- [ ] `caratan.vercel.app` domain (needs vercel CLI re-auth)
