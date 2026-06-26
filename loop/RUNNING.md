# Caratan loop — operations

The autonomous self-improvement loop is LIVE on Modal (workspace `michael01px2025`,
app `caratan`). It trains a self-hosted Qwen3-8B in GRPO rounds, invents new reward
graders for its own weaknesses, and only keeps checkpoints that beat the last on
held-out boards. Runs ~23h, then a daily cron relaunches it. State persists on the
`caratan-state` Volume, so it improves across days.

## URLs
- Live status (JSON, public): https://michael01px2025--caratan-status.modal.run
- Serve (vLLM, needs key):     https://michael01px2025--caratan-serve.modal.run
- Modal dashboard:             https://modal.com/apps/michael01px2025/main/deployed/caratan

## Watch it
```bash
curl -s https://michael01px2025--caratan-status.modal.run | python3 -m json.tool
.venv-modal/bin/python -m modal app logs caratan          # live logs
```
The site (caratan.vercel.app) polls the status URL client-side and renders it
(LiveLoop component) — no rebuilds needed.

## Control
```bash
# redeploy after code changes
PYTHONPATH=loop .venv-modal/bin/python -m modal deploy loop/modal_app.py
# start a run manually (the lock prevents overlap with the cron)
.venv-modal/bin/python -c "import modal; modal.Function.from_name('caratan','loop_run').spawn(n_rounds=200, steps=40, max_hours=23)"
# stop the daily cron: comment out @app.function(schedule=...) on daily_loop and redeploy
```

## How it's safe to leave unattended
- **Promotion gate**: a new checkpoint is kept only if it beats the current best on
  the CANONICAL held-out grader (never the generated one) — proven to reject
  regressions in testing.
- **Sanity gate**: every Claude-generated grader must be deterministic, rank
  good>bad, score garbage <=0, be bounded, and touch no os/io/net — else discarded.
- **Single-runner lock**: a heartbeat lockfile stops cron + manual runs overlapping.

## Pieces
- `modal_app.py` — serve / train / evaluate / promote / loop_run / daily_loop / status
- `graders.py`   — canonical text->reward graders (single source of truth)
- `weakness.py`  — transcripts -> failure-mode report
- `envgen.py`    — Claude env-gen brain + sanity gate
- `orchestrator.py` — laptop-side single-round driver (for manual rounds)
- `publish.py`   — registry -> web/public/data/loop.json (optional static publish)

## Cost knobs
40 steps/round ≈ 10 min on one A100-80GB (train) + 2 short L40S/A100 evals.
Lower `steps` or `eval_limit` to stretch credits; raise for bigger per-round jumps.
