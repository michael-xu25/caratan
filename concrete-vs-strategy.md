# Concrete moves vs. strategy — the grading split

Two layers we grade on, kept deliberately separate. The point: **concrete moves
are objectively measurable *inside Catanatron*; strategy is a pattern across
moves that only an LLM (or human) can name.** Everything we lean on for the
concrete layer is verified to be within Catanatron's actual API (§3) — we never
assume the engine knows anything about "plans" or "intent."

## 1. The two layers

| | **Concrete move** (tactical) | **Strategy** (intent / plan) |
|---|---|---|
| What it is | the single action chosen at one decision point | the multi-turn plan the moves add up to |
| Examples | "moved robber to tile 14", "settled node 27", "traded 4 wood → 1 ore" | "racing longest road", "hoarding for a city", "boxing out BLUE's expansion" |
| Time scale | one ply | many plies |
| Where it lives | **in Catanatron** — it's a legal `Action` the engine applies | **not in Catanatron** — inferred from the move sequence (+ stated reasoning) |
| How we grade it | **quantitative**: value-function *regret* (objective, cheap, no LLM) | **qualitative**: LLM grader over the transcript (seeded by the rubric, free to flag novel) |
| Output | per-decision reward/regret, bucketed by action type | named failure modes, clustered post-eval |

## 2. The split principle

- **Concrete moves are the atoms Catanatron hands us.** Every turn the engine
  enumerates the legal actions; the model picks one; the engine applies it. The
  quality of that pick is measurable *without any LLM* — see regret below.
- **Strategy is the pattern over the atoms.** Catanatron has no notion of "plan"
  or "why"; strategy only exists as the shape of a *sequence* of concrete moves.
  Judging it requires reading the game (an LLM grader), not querying the engine.
- So the grading pipeline is: **regret flags *which concrete moves* were bad
  (engine-grounded) → the LLM explains/clusters them into *strategic* failure
  modes (engine-external).** Quant grounds, qualitative interprets.

## 3. Catanatron limits check — what's IN vs OUT of engine scope

**IN scope (concrete layer — verified against the vendored Catanatron API):**

| We rely on | Catanatron provides | Verified |
|---|---|---|
| The legal action set | `generate_playable_actions(state)` | ✅ used every ply |
| Applying a move | `Game.copy()` + `game.execute(action)` | ✅ `game.py:214`, used by `ValueFunctionPlayer` |
| Scoring a state | `players/value.py` `base_fn(game, color)` | ✅ the value function |
| **Regret of a move** | `max_a value(copy.execute(a)) − value(copy.execute(chosen))` | ✅ exactly what `ValueFunctionPlayer.decide` already loops over |
| Victory points / winner | `get_actual_victory_points`, `winning_color()` | ✅ used by the runner |
| State for the prompt/grader | `GameEncoder` JSON, action types enum | ✅ transcripts |

So the entire concrete-move layer (legality, regret, VP, outcome) is computed
**only** from APIs Catanatron actually exposes. The regret oracle reuses the
exact `copy → execute → value_fn` path the built-in value bot uses.

**OUT of scope (strategy layer — must come from the grader, NOT the engine):**
intent, multi-turn plans, "why" behind a move, reading the opponent, reasoning
quality. Catanatron cannot answer any of these; pretending it can would be a
category error. These are the LLM grader's job, over the move sequence.

## 4. Implication for the eval

- **Primary metric (per-weakness) is concrete-move-level** — score a decision vs
  ground truth / regret. Fully engine-grounded, reproducible, cheap. This is the
  headline ("robber-when-behind: 30% → 78%").
- **Strategic findings are the qualitative overlay** — the LLM grader names the
  recurring patterns behind the high-regret moves; the *named* weaknesses are an
  output (clustered post-eval), not a Catanatron query.
- Keep them as separate fields: a move has a `regret` (concrete, from the engine)
  and may belong to a `strategy`/failure-mode category (qualitative, from the
  grader). Don't collapse them.

See `grading-rubric-proposal.md` for the full grader design (the regret oracle +
the agreement 2×2 + emergent taxonomy).
