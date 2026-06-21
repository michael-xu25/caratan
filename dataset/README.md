# dataset/

Generated Catan boards. Each board is one integer **seed**; a board state file
records that seed plus the serialized initial game state, so any board is fully
reproducible and directly loadable.

## Layout

```
dataset/
  initial/              # boards generated from scratch (this is the base pool)
    index.json          # manifest: generation params + per-board seed/split/fingerprint
    init_0000.json      # one board state file per board
    init_0001.json
    ...
  <env_name>/           # later: env-generation steps write sibling subfolders here,
                        # using the same board-state-file schema
```

## Board state file schema

```jsonc
{
  "schema_version": 1,
  "board_id": "init_0000",
  "seed": 1000,                 // source of truth: regenerates the whole board
  "split": "example_pool",      // "grader_games" | "example_pool" (see below)
  "number_placement": "official_spiral",
  "map_template": "BASE",
  "vps_to_win": 10,
  "board_fingerprint": "fb639cbc9f8e",  // matches harness.runner._board_fingerprint
  "robber_coordinate": [0, 2, -2],
  "tiles": [ {"id":0,"coordinate":[0,0,0],"type":"RESOURCE_TILE","resource":"SHEEP","number":11}, ... ],
  "ports": [ {"id":0,"resource":"ORE","direction":"WEST","nodes":[25,26,...]}, ... ],
  "initial_state": { ...full catanatron.json.GameEncoder dump of the fresh game... }
}
```

- `seed` is canonical: `Game(players, seed=seed, number_placement=...)` regenerates
  the identical board. `tiles`/`ports`/`robber_coordinate`/`initial_state` are
  saved for convenience and verification.
- `board_fingerprint` is computed the same way as the match runner, so a dataset
  board and a runner game on the same seed **provably** share a layout.

## How boards are used

- **1 seed per board, played ×2.** The seed fixes the board; at run time the
  match runner plays it as a **mirrored seat-swap pair** (RED/BLUE swapped across
  the two orientations) — the project's fairness primitive.
- **`grader_games`** (default 100 boards): seeds for the full self-play games the
  grader runs on. Play each board out to a finished game, then grade it (the
  dual-grader pipeline).
- **`example_pool`** (the rest): boards to mine specific mid-game states from
  (e.g. opening placements). Load a board, advance it to a target state, and ask
  a model to act — used for per-weakness example/env construction, not full games.

Splits are assigned by a deterministic shuffle keyed on `meta_seed` (recorded in
`index.json`), so the assignment is reproducible and independent of file order.

## Regenerate

```bash
python scripts/generate_initial_boards.py            # 400 boards, 100 holdout, official_spiral
python scripts/generate_initial_boards.py --count 500 --holdout-eval 100
python scripts/generate_initial_boards.py --number-placement random   # also shuffle numbers
```
