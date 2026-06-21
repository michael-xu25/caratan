# Goldilocks × Catan — Eval Harness

Hud RL-environments hackathon. We prove one loop end-to-end:

> **play games → find model weaknesses (Claude + GPT graders) → auto-generate
> verifiable envs targeting them → GRPO-train a small open LLM → measure the
> improvement fairly.**

This repo is the **eval / measurement half** of that loop. The trained artifact is
a small open-weights model; the analyst/judge models (Claude, GPT) are API models
we don't train. The agent interface is **model-agnostic** — every policy (bot or
LLM) is selected by a single string spec, so swapping models is a config change,
not a code change.

---

## The model lineup

All three are served over the **HUD inference gateway** (Tinker-backed,
OpenAI-compatible) and addressed with the `hud:` backend:

| spec | what it is |
|---|---|
| `hud:Qwen/Qwen3-8B` | **untrained base** — the baseline we measure against |
| `hud:catan-placement-only` | trained on the **placement / settlement** stage only (mid checkpoint) |
| `hud:catan-grpo-q8b` | **full GRPO chain** (placement → build → trade) — the **shipped** model |

The headline question: *what did training actually buy?* We answer it two
independent ways — **head-to-head games** and a **per-decision held-out eval** —
and show them side by side, because they can disagree (a model can dominate the
per-decision exam yet be near a coin-flip in full games).

---

## Quickstart

Catanatron needs **Python ≥ 3.11**.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e catanatron          # vendored game engine
pip install -r requirements.txt    # SDKs for the LLM backends

# keys (only the backends you use): HUD_API_KEY, ANTHROPIC_API_KEY,
# OPENAI_API_KEY, FIREWORKS_API_KEY — keep them in .env (gitignored), never commit.
set -a; source .env; set +a
```

Serve the UI (replay viewer + results dashboard):

```bash
python -m http.server 8000
# open http://localhost:8000/viewer/index.html      (replay viewer)
#      http://localhost:8000/viewer/matchups.html    (results dashboard)
```

---

## The harness (`harness/`)

The measurement half. It plays fair 1v1 games between any two policies, writes
interpretable transcripts, and runs the per-weakness scorer. Catanatron is the
vendored engine; the harness wraps it. See `harness/HARNESS.md` for the deep dive.

- **`agents.py`** — the model-swap seam. `make_agent("<backend>[:<model>]", color)`
  returns a ready Catanatron `Player`. Bots: `random`, `weighted`, `value`,
  `alphabeta[:depth]`. LLM backends: `claude`, `openai`, `fireworks`, `hud`.
  Every decision records the full legal option set it chose among + VP context +
  (optionally) the model's reasoning, so weaknesses are analyzable.
- **`backends.py`** — `make_backend(spec)` → an `LLMBackend` with one method,
  `complete(system, user) -> str`. Re-exports Michael's `goldilocks_eval` factory
  so the two packages share one backend set.
- **`runner.py`** — async, process-isolated match runner: `run_batch` /
  `run_mirror_pair` / `run_match`. Mirrored seat-swap fairness, transcript writing,
  VP tie-break at the turn cap. Concurrency = max in-flight LLM calls (the engine is
  ms-fast; the gateway is the bottleneck, which plateaus ~6 turns/sec).
- **`grader/`** — dual-grader pipeline (Claude + GPT) over transcripts: regret
  oracle, union-merge, detailed tables, strategic failure-mode review.

**Fairness:** seeded purely-random board + dice (mirroring cancels luck — *not* a
balanced deck), **mirrored games** (same seed both seat orders), held-out
`grader_games` seeds disjoint from training. Game rules: **400-turn cap → VP
tie-break**, **10 VP to win**.

---

## Head-to-head matchups

Trained checkpoint vs the untrained base over the 100 held-out `grader_games`
boards, **mirrored** so win-rate is seat-fair.

```bash
# staged queue — focus the shipped full-chain model, reuse the diagnostic run,
# cap turns at 300, mirrored pairs, push results as each stage completes:
python scripts/run_matchup_queue.py --wait-for-diag --grpo-only \
    --max-turns 300 --concurrency 32 --push
```

`scripts/run_matchup_queue.py` flags: `--grpo-only` (focus the shipped model;
placement is Michael's), `--wait-for-diag` (start after the diagnostic finishes),
`--reasoning-on N` (first N seeds capture rationale; default 0 = all off, fastest),
`--max-turns`, `--concurrency`, `--push`, `--dry-run`. It reuses the diagnostic's
games (never re-runs them) and, after each stage, rebuilds the viewer data and
recomputes the stats. `scripts/run_hud_matchups.py` is the simpler both-matchups
runner with a 20-on/80-off reasoning split.

**Speed note:** the gateway caps total throughput (~6 turns/sec) regardless of
concurrency past ~24–32, so wall-clock = total turns ÷ throughput. The real levers
are a lower turn cap and fewer/shorter games — not more concurrency.

---

## Stats

Both read the saved transcripts and write JSON the dashboard consumes.

- **`scripts/winrate_stats.py <run_dir>`** — seat-fair win-rate (mirrored →
  draws excluded for the headline), avg VP, cap-stalls → `<run_dir>/winrate.json`
  + `WINRATE.md`, merged into `viewer/data/matchups.json`.
- **`scripts/gamestats.py [dirs]`** — per-model **game-quality** stats, averaged
  over every game the model played → `viewer/data/gamestats.json`:
  - `skipped_turn_rate` — turns (after the roll) where nothing happened (no
    build/buy/trade/dev); not inherently good or bad.
  - `resource_gain_per_game` — resources flowing into the hand (production + trade
    inflow); reflects placement + keeping the economy moving.
  - `settlements / cities / roads / dev_buys / trades` per game.
  - `pair_sweep_rate` — of seeds played both ways, how often the model won **both**
    orientations: **adaptability** (winning regardless of seat).

---

## The viewer & results dashboard (`viewer/`)

A static, dependency-free UI. See **`viewer/README.md`** for the full tour.

- **`index.html`** — replay any game step-by-step (board, dice, hands, VP, awards),
  with a run picker, a **mirror toggle** (jump a seed's norm ⇄ swap game),
  **next/prev-transcript** buttons, winner shown by **model**, and a ⚖️ grading
  overlay when a `.grading.json` sidecar exists.
- **`matchups.html`** — the results dashboard: **head-to-head win-rate**,
  **game-quality stats** (untrained vs settlement-only vs fully-trained), and the
  **per-decision held-out eval**, with full-chain vs settlement-only descriptors.
  **Auto-refreshes every 15s**, each panel drawing from its own stats file.

**Live pipeline:** while a matchup runs, a refresher recomputes the stats from the
growing transcript set and the dashboard polls the JSON — so numbers tick up on
their own. Build steps: `scripts/build_viewer_data.py <dir>` (transcript →
`.view.json`), `scripts/build_viewer_index.py` (run-picker manifest `runs.json`).

---

## Per-decision held-out eval (the other axis)

Each model answers the **same frozen held-out scenarios** independently (no
opponent, no game), scored by a deterministic reward fn vs ground truth — pure
pick-quality. Disjoint from training. This is the **primary** metric (per-weakness
before/after); the head-to-head games are the secondary, end-to-end check.

```bash
python -m goldilocks_eval.scenario_cli \
    --scenarios data/placement_heldout.jsonl \
    --before claude:claude-haiku-4-5 --after claude
# -> per-env accuracy + headline delta, e.g. placement 39% -> 78%
```

The prompt/answer/reward contract is **one source of truth**
(`goldilocks_eval/prompting.py`) so before/after numbers can't compare apples to
oranges. The frozen scenario record lives in `goldilocks_eval/schema.py`; node ids
resolve to board positions from serialized state via `goldilocks_eval/geometry.py`.

---

## Repo map

```
harness/            measurement half — agents, backends, runner, grader, prompts
  agents.py           make_agent: the model-swap seam (bots + claude/openai/fireworks/hud)
  backends.py         make_backend: LLMBackend(complete) — shared with goldilocks_eval
  runner.py           async mirrored match runner (process-isolated, VP tie-break)
  grader/             dual Claude+GPT transcript grader (regret oracle, failure modes)
goldilocks_eval/    Michael's package — env generation, placement/maritime envs, scorer
  agents/             LLMBackend + factory + claude/openai/fireworks/hud backends
scripts/
  run_matchup_queue.py  staged matchup queue (grpo-only, reuse diagnostic, push)
  run_hud_matchups.py   both-matchups runner (reasoning split)
  winrate_stats.py      seat-fair win-rate -> matchups.json / WINRATE.md
  gamestats.py          per-model game-quality stats -> gamestats.json
  build_viewer_data.py  transcript -> .view.json replay file
  build_viewer_index.py viewer run-picker manifest (runs.json)
viewer/             static UI — replay viewer + results dashboard (see viewer/README.md)
transcripts/        saved games (per run dir); diag-* and hud-*-vs-base matchups
dataset/            seeded boards + the grader_games held-out split index
catanatron/         vendored game engine
```

---

## Notes / contract

- **Dice = seeded purely-random** (`Game(seed=…)`), not a balanced deck —
  mirroring cancels luck and a draw-without-replacement deck would be countable.
- **Never commit API keys.** Backends read them from the environment / `.env`.
- **Shared code with `goldilocks_eval/`** (Michael's package) goes through a PR to
  `main`; the harness re-exports it rather than forking.
- Watch **catastrophic forgetting**: head-to-head can regress while the target
  decision improves — that's exactly why both axes are reported together.
```
