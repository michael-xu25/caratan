# Placement scenario fixtures + the frozen contract

Build targets so the **generator** (Michael) and the **labeling UI** (Cara) can
work independently and meet at a frozen format.

## Files
- `placement_examples.jsonl` — 5 real placement scenarios (real Catanatron
  `GameEncoder` output, not hand-faked). Coverage: `0_p1` and `1_p1` are pick 1
  (empty board); `0_p2`/`0_p3`/`0_p4` have 1/2/3 existing settlements (the
  "best remaining spot" case). `gold_action`/`acceptable_actions` are
  intentionally empty — they double as labeling-UI test cases.
- `scenario.schema.json` — JSON Schema of the record (for a non-Python UI).

## The frozen schema
Canonical definition: `goldilocks_eval/schema.py` (`Scenario`, `new_unlabeled`,
`apply_label`, `validate`, `json_schema`). Both directions are pinned:

- **Generator → UI:** emits every field; `gold_action=null`,
  `acceptable_actions=[]`, `base_solve_rate=null`.
- **UI → generator:** writes `gold_action` + `acceptable_actions` back into the
  *same* record, nothing else changed. Do it via `schema.apply_label(record,
  gold, acceptable)` — it validates that every labeled node is in
  `legal_actions` and normalizes ids to `node_<int>`.

Node ids are canonical strings `node_<int>` everywhere in the contract
(Catanatron's raw settlement action value is a bare int; normalize at the edge).

## Node id → board position (the integration question — answered: YES)
The UI can place every `legal_actions` node on a rendered board using **only**
`serialized_state` — no extra serialization needed. Each entry in
`serialized_state["nodes"][<int id>]` carries `tile_coordinate` (cube) +
`direction` (NodeRef corner), and:

```
position(node) = cube_to_pixel(tile_coordinate) + node_delta(direction)

cube_to_pixel((x,y,z), size):           # q=x, r=z   (flat-top axial)
    px = size * (√3·q + √3/2·r)
    py = size * (3/2·r)

node_delta(direction, size):            # w=√3·size, h=2·size
    NORTH=(0,-h/2)  NORTHEAST=(w/2,-h/4)  SOUTHEAST=(w/2,h/4)
    SOUTH=(0, h/2)  SOUTHWEST=(-w/2,h/4)  NORTHWEST=(-w/2,-h/4)
```

This is copied verbatim from Catanatron's own renderer
(`gym/envs/pygame_renderer.py` → `cube_to_pixel` + `get_node_delta`), whose
docstring notes it "matches the frontend getNodeDelta function" — so a JS UI
mirrors it 1:1. A ready Python resolver is `goldilocks_eval/geometry.py`
(`node_position(serialized_state, node_id, size=1.0)`), un-centered and
unit-scaled; apply your own canvas center + size.

Verified on all 5 fixtures: every legal node (54/50/46/42/54) resolves to a
**unique** position, no collisions.

## Shared prompt/answer contract
`goldilocks_eval/prompting.py` is the single source of truth (`build_prompt`,
`parse_answer`, `score`) used by generation, calibration, and eval. Answer
format `<reasoning>…</reasoning>\n<answer>node_27</answer>`; reward 1.0 gold /
0.5 acceptable / 0.0 else.
