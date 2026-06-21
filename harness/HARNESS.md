# Eval Harness — what it is, how it works, how to use it

The **measurement half** of the Goldilocks × Catan loop. It plays fair 1v1
Catan games between any two policies (bots + Claude wired; other LLMs are one
method away), writes readable transcripts, and runs the per-weakness scorer.
Catanatron is the vendored game engine; this package wraps it.

**Relationship to `goldilocks_eval/` (Michael's package).** Michael's package is
the canonical home of the LLM **prompt rendering**, the **Claude backend**, and
the **scenario scorer / shared contract**. `harness/` *reuses his code directly*
— `harness/prompt.py`, `harness/backends.py`, `harness/scenario.py`, and
`harness/scenario_cli.py` are thin re-exports of `goldilocks_eval`, so there's a
single source of truth and no divergence. On top of that, `harness/` adds its
own value: the **process-isolated mirrored runner** with a board-fingerprint
fairness proof, the **rich human transcripts + replay viewer**, and the
**reasoning-mode flag**.

---

## Quick start

```bash
# from the repo root, with the venv active (or use .venv/bin/python)

# 1) one fairness pair: same board, seats swapped (start here)
python -m harness.cli --a value --b weighted --pair 1

# 2) a mirrored batch over N seeds, run concurrently
python -m harness.cli --a value --b random --n 10 --concurrency 8

# 3) explicit seeds, no mirroring
python -m harness.cli --a value --b weighted --seeds 1,2,3 --no-mirror
```

Outputs land in `--run-dir` (default `transcripts/`, git-ignored): a `.json` +
`.log` per game, plus a `summary.txt` (batch) or `pair_seed*.txt` (pair).

```bash
# 4) Claude vs a bot, with model reasoning captured for the transcripts
#    (export ANTHROPIC_API_KEY first; reasoning is OFF by default for cheap runs)
python -m harness.cli --a claude --b value --pair 1 --reasoning

# 5) PRIMARY metric: per-weakness scenario eval on a held-out set (Michael's scorer)
python -m harness.scenario_cli --scenarios data/placement_heldout.jsonl \
    --before claude:claude-haiku-4-5 --after claude
```

**Agent specs** are `"<backend>"` or `"<backend>:<arg>"`. Bots: `random`/`R`,
`weighted`/`W`, `value`/`VP`, `alphabeta`/`AB` (`alphabeta:3` sets depth). LLMs:
`claude[:model]` (wired, via Michael's Claude backend), `gemini[:model]` (stub —
one `complete()` method to wire).

**Reasoning mode** (`--reasoning`, default off): model reasoning is captured only
when on — use it for the small set of games you'll open in the viewer. Training/
production runs stay off (cheaper, faster). The grader's verdict in the scenario
scorer is always recorded regardless.

---

## The agent interface (`harness/agents.py`) — the load-bearing piece

Everything hangs off one method, Catanatron's own `Player` contract:

```python
def decide(self, game, playable_actions):  # -> one of playable_actions
```

The runner, logger, and scorer only ever call `decide`. That's what makes the
model swap *a config flag, not a rewrite*, and it's the same call the
per-weakness scorer makes against a held-out position.

On top of that, `Agent` (the base class) adds three things the build doc asks
for:

- **Backend config** — one `LLMAgent` holds an `LLMBackend` (Michael's
  `complete(system, user) -> str`); the backend is the only thing that branches.
  Swap Claude ↔ Gemini ↔ trained model by changing the spec.
- **Reasoning capture (mode-gated)** — when `capture_reasoning` is on, the agent
  calls `self._note(text)` during `decide` and the transcript logger picks it up
  via `pop_reasoning()`. Off (default) → the model returns only the action index
  (cheaper, for training/runs). Bots emit a short policy note.
- **Legality guarantee** — `decide` *must* return a legal action or the game
  crashes. `LLMAgent` wraps the model call in a try/except: any API error,
  out-of-range index, or unparseable reply falls back to a legal action
  (`fallback_action`, default = first legal move). The LLM equivalent of action
  masking — verified by running a never-responding backend through a full game.

### Backends

The Claude backend (`goldilocks_eval.agents.claude_backend`, re-exported via
`harness.backends`) is wired — `make_agent("claude[:model]", color)` just works
with `ANTHROPIC_API_KEY` set. To add Gemini/a local model, implement one method,
`complete(system, user) -> str`, and register it in Michael's factory; the
agent, runner, scorer, and transcripts are all backend-agnostic.

---

## Transcripts (`harness/transcripts.py`)

`TranscriptAccumulator` plugs into `Game.play(accumulators=[...])` and writes,
per game:

- **`<label>.json`** — full machine state via Catanatron's `GameEncoder`, plus
  a `decisions` array carrying each decision's captured reasoning. This is the
  machine-readable record.
- **`<label>.log`** — a human log built for *skimming in batches*: a one-line
  verdict banner (winner, VP, turns, seed), a compact board (resources →
  numbers in probability order), then decisions grouped by turn with reasoning
  indented under each. Routine rolls/end-turns are collapsed. Rendered with
  `rich` and exported as plain text (no color codes) so it reads anywhere.

`render_summary_table` produces the batch index (read this first); 
`render_pair_report` produces the fairness view for one mirrored pair.

> Gotcha handled: dynamic text (action values, model reasoning) is printed with
> `markup=False` so `rich` doesn't eat `[...]` as style tags.

---

## The runner & fairness (`harness/runner.py`)

The match loop is "ask current player to `decide`, apply, repeat until someone
wins" — Catanatron's `Game.play`. The harness adds:

**Async + concurrency.** Catanatron is ms-fast; a "game" is mostly waiting on
LLM calls. Games fan out across a process pool; `--concurrency` is the real
knob and maps to the concurrent-LLM-call ceiling (set it to your model's rate
limit, not CPU cores).

**Processes, not threads.** Catanatron draws from Python's **global** `random`
module (confirmed in source — no local-RNG option). Threads would share that
RNG and corrupt concurrent games' boards/dice (verified: mirrored boards
diverged under thread concurrency). Each game runs in its own process instead,
with an independent RNG. The worker boundary passes **specs in / a primitives
dataclass out**, so no unpicklable LLM-client objects cross it.

**Mirroring — the fairness primitive.** Catanatron has *no* built-in mirroring;
we add it on top of its seed + seating. `run_mirror_pair` plays one seed twice:
game 1 with A seated first, game 2 with seats swapped on the **same board**.
- `MatchResult.board_fingerprint` lets us *prove* the board was identical, not
  just assert it. `render_pair_report` checks it and prints `IDENTICAL`.
- The pair verdict separates **skill from seat luck**: if the same agent wins
  both seats it's a real edge; a 1–1 split means the seat decided it (no skill
  signal — need more pairs).

Why this matters, concretely: in a `value` vs `weighted` run, **the
first-placing seat won every game** — the seat advantage dominated two
similar bots. Without mirroring, a win rate is just measuring who got which
seat. This is the whole reason the metric needs mirrored pairs.

**Reproducibility (and how to swap it out).** Concurrent runs are reproducible
today: results are byte-identical across separate runs (only wall-clock timing
varies). That needs two things — per-process isolation *and* a pinned
`PYTHONHASHSEED` in workers (engine action ordering uses sets/dicts). Both live
in **one removable shim**, `harness/determinism.py`, behind a single
`make_pool()` call. When the team's own randomness / balanced-dice system lands
and owns reproducibility, flip the knobs there (or delete the file and inline a
plain pool) — nothing else in the harness depends on it.

---

## The ground-truth contract (`harness/contract.py`) — the seam with Michael

The primary metric is per-weakness accuracy: given a fixed position, did the
model pick the correct action? My scorer needs `position → correct answer`.
*Who* produces the label (a solver, a human move, an LLM judge) is Michael's
side — this module pins the **format** so the generator and scorer agree now,
not at 3am.

One instance = one decision point (JSONL, one object per line):

```json
{
  "id": "robber-when-behind-0007",
  "weakness": "robber-when-behind",
  "state": { "...": "Catanatron GameEncoder dump" },
  "playable_actions": [["RED", "MOVE_ROBBER", "..."]],
  "answer": {
    "type": "exact",                         // exact | set | ranking
    "action": ["RED", "MOVE_ROBBER", "..."],
    "source": "solver",                      // solver | human | judge
    "explanation": "why this is correct"
  },
  "meta": { "seed": 42, "split": "eval" }
}
```

Action triples are `[color, action_type, value]` — exactly what `GameEncoder`
emits — so a state Michael serializes round-trips into my scorer unchanged.
`validate_instance` / `load_instances` enforce the shape. The scorer itself
(drive `agent.decide` over each position, compare to `answer`, bucket by
`weakness`) is the next build step; the format is locked so it drops in.

---

## Layout

```
harness/
  agents.py       # Agent interface, bot backends, LLMAgent + legality fallback
  transcripts.py  # JSON + human log, batch summary, fairness-pair report
  runner.py       # async/process-pool match runner, mirroring, MatchResult
  contract.py     # per-weakness ground-truth schema (seam with Michael)
  determinism.py  # removable shim: process pool + pinned hash seed
  cli.py          # python -m harness.cli  (pair / batch)
  HARNESS.md      # this file
```

## Status

- ✅ Agent interface (backend flag, reasoning capture, legality guarantee)
- ✅ Transcripts (machine JSON + skimmable human log + batch/pair reports)
- ✅ Async process-pool runner + mirrored pairs + board-identity proof
- ✅ Ground-truth contract schema defined and validated
- ⬜ Wire a real LLM backend (`_complete`)
- ⬜ Per-weakness scorer over a held-out instance set
- ⬜ Team randomness / balanced-dice system (owns reproducible dice)
```
