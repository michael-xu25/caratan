# Task: Freeze the scenario schema + generate example fixtures

Goal: unblock clean parallel work. Michael builds the scenario generator; Cara builds the labeling UI that renders scenarios and writes labels back. Their work only stays independent if the scenario schema is **frozen** and there are **example fixtures** in the repo to build against. Do both now.

## 1. Freeze the schema as code, not prose
Right now the scenario schema lives in `build-spec-decisions.md` / `scenario-generation-spec.md` as documentation. Promote it to a single canonical definition in the codebase that both the generator and the UI import — so it can't drift between them.

- Create one source of truth (e.g. a dataclass / Pydantic model / JSON Schema — your call, match the codebase) for the scenario record.
- Fields (exact — this is the locked contract):
  - `scenario_id` (str)
  - `game_id` (str) — = board seed; the grouping key for the leak-free train/heldout split
  - `board_seed` (int)
  - `pick_index` (int, 1–4) — placement-specific
  - `env` (str, "placement" for v1)
  - `serialized_state` (object) — Catanatron `GameEncoder` JSON of the frozen state
  - `legal_actions` (list[str]) — legal settlement node ids
  - `gold_action` (str | null) — champion label; null until labeled
  - `acceptable_actions` (list[str]) — near-optimal alternatives
  - `base_solve_rate` (float | null) — filled by calibration
  - `split` (str, "train" | "heldout")
- Both directions of the contract must be specified, not just the read path:
  - **Generator → UI:** generator emits everything except `gold_action`/`acceptable_actions` (null/empty).
  - **UI → generator:** the labeling UI writes `gold_action` + `acceptable_actions` back into the same record. "What the UI writes" is part of the frozen contract — pin it explicitly.

## 2. Generate 3–5 example fixtures and commit them
These are Cara's build target so she is NOT blocked on the real generator.

- Produce 3–5 real placement scenarios by running the actual Catanatron path (seeded `Game`, official_spiral, Value bot filling the opening, snapshot at settlement-placement decision points). Use real `GameEncoder` output — not hand-faked JSON — so the UI renders against the true serialized shape.
- Cover variety: at least one pick_index 1 (empty-ish board) and a couple of picks 2–4 (existing settlements on board = the "best remaining spot" case).
- Leave `gold_action`/`acceptable_actions` empty so they double as labeling-UI test cases.
- Commit to `data/examples/placement_examples.jsonl` (or matching repo convention) + reference them in the README.

## 3. Resolve the one genuine integration risk: node-id → board position
This is the only place the UI and the serialized state must truly agree, beyond passing JSON.

- Cara's UI overlays `legal_actions` node ids onto board vertices so Michael can pick one. She can only do that if node ids in `legal_actions` are resolvable to on-screen coordinates from `serialized_state` (Catanatron has a coordinate system — confirm the mapping is derivable from what we serialize).
- Verify with the fixtures: can you take a fixture's `serialized_state` + `legal_actions` and unambiguously place each legal node on a rendered board? If the serialized state doesn't carry enough to resolve node positions, **add what's needed to the serialization now** — before either side builds on it.

## 4. Confirm prompt/parse contract is shared (related, don't skip)
The generator, the calibration harness, and the eval (`scenario.py`) must use the **same** prompt builder + answer parser, or before/after numbers compare apples to oranges. If a canonical `build_prompt` / `parse_answer` doesn't already exist in a shared module, create one and point all three at it. The answer format is `<reasoning>...</reasoning>\n<answer>NODE_ID</answer>`; reward is tiered 1.0 gold / 0.5 acceptable / 0.0 else.

## Deliverables
1. Canonical schema definition imported by both generator and UI.
2. `data/examples/placement_examples.jsonl` with 3–5 real fixtures.
3. A confirmed answer to: "can the UI resolve every `legal_actions` node to a board position from `serialized_state`?" — and a serialization fix if the answer is no.
4. Shared `build_prompt`/`parse_answer` (or confirmation one already exists).

Once 1–2 land, Cara builds the UI against the fixtures and Michael builds the generator in parallel — genuinely independent.
