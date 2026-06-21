# Scenario Generation Spec — Placement Env (v1)

*The producer half of the loop. Emits the JSONL that the eval harness + calibration harness consume. Build to this exactly — the schema is the contract with Cara's `goldilocks_eval`.*

---

## Where this sits
`generate (this) → champion-label → calibrate (harness) → train/heldout JSONL → GRPO + eval`

Placement is the **first env**: dice-free (opening settlements precede any roll), pure label-match grading, no state-deserializer needed. Mid-game envs come later and *will* need `state_from_json` (the GameEncoder is one-way) — out of scope for v1.

---

## What a placement scenario is
A frozen 1v1 opening state at a settlement-placement decision point, plus the legal nodes, asking: **"best settlement among the legal spots here."** The champion's answer is ground truth.

1v1 opening = 4 settlement placements in snake order (P1, P2, P2, P1). Snapshot at **all 4** → up to 4 scenarios/game. Picks 2–4 have prior settlements on the board → these are the "best remaining spot" decisions (the rich ones). Pick 1 = canonical best-opening.

---

## Generation algorithm
1. **Two disjoint board pools by seed range** — e.g. seeds `0..N` → pool A (train), seeds `10000..10000+M` → pool B (heldout). Disjoint seeds guarantee disjoint boards → the `game_id`-level split is leak-free by construction.
2. For each seed: `Game([ValueFunctionPlayer, ValueFunctionPlayer], seed=seed, number_placement='official_spiral')`. Use a real bot (Value) to fill the opening so prior placements look like a real game (don't fill randomly — unrealistic boards poison the "best remaining" signal). The bot's pick only advances the board; it is **not** the label.
3. At each of the 4 settlement-placement decision points, **before** the bot acts: snapshot
   - `serialized_state` ← `GameEncoder` JSON of `game.state`
   - `legal_actions` ← the legal settlement node ids from `playable_actions` (filter `is_initial_build_phase` + `BUILD_SETTLEMENT`)
   - `pick_index` (1–4), `game_id` (= board seed), `board_seed`, `scenario_id`
   - `env = "placement"`, `split` per pool
   Then let the bot place and continue.
4. Emit **unlabeled** scenarios (no `gold_action` yet).

---

## Champion labeling (your irreplaceable input — front-load this)
- A small labeling CLI/UI renders each scenario's board from `serialized_state` (resources, dice numbers/pips, ports, existing settlements) with **node ids overlaid** so you can name a node.
- You provide `gold_action` (best node) + `acceptable_actions` (near-optimal alternatives) per scenario.
- Merge labels back into the JSONL. Only labeled scenarios proceed.
- Cara's board-render-with-node-ids is the dependency here — that view is what makes labeling fast.

---

## Calibration (handled by `calibration_harness.py`)
- Pool A only: run base E4B ~8× per scenario, fill `base_solve_rate`, **keep Goldilocks band** (drop always-solved / always-failed — zero GRPO gradient).
- Pool B (heldout): run the same sampling to *record* `base_solve_rate` (the eval's "before" number) but **do not filter** — filtering the eval set by base failure biases the before/after upward.

---

## Output
- `data/placement_train.jsonl` (pool A, Goldilocks-filtered)
- `data/placement_heldout.jsonl` (pool B, unfiltered)
Regenerable from seeds. Versioned in repo.

---

## Schema — FROZEN in code (do not re-spec here)
The scenario record is now the single source of truth in **`goldilocks_eval/schema.py`**
(`Scenario`, `new_unlabeled`, `apply_label`, `validate`, `json_schema`). Build the
generator against that, not against a copy of the JSON — a copy will drift.

- Generator: emit via `schema.new_unlabeled(...)` (labels left empty).
- Field list, both-direction contract, and a JSON Schema export: see
  `data/examples/README.md` + `data/examples/scenario.schema.json`.
- Example records to build against: `data/examples/placement_examples.jsonl`.

---

## Shared prompt/answer contract — EXISTS (`goldilocks_eval/prompting.py`)
Generation, calibration, and the eval all import the **same** functions, so
before/after numbers can't compare apples to oranges. Don't re-implement them.

- `prompting.build_prompt(scenario)` — board render (resources, pips, ports,
  existing settlements) + `legal_actions` + "reason, then answer."
- Output format `‹reasoning›…‹/reasoning›\n‹answer›node_27‹/answer›`.
- `prompting.parse_answer(text)` → canonical `node_<int>`.
- `prompting.score(answer, gold, acceptable)` — tiered `1.0`/`0.5`/`0.0`.

Node id → board position (for any UI/render) is `goldilocks_eval/geometry.py`
(`node_position`) — derivable from `serialized_state` alone, no extra
serialization needed.
