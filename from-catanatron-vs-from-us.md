# What comes from Catanatron vs. what we add vs. what the model decides

The clean mental model for the whole system has **three layers**:

1. **Catanatron (the environment)** — the single source of truth for the *game*:
   rules, board, randomness, legal moves, state transitions, scoring. It computes
   and enforces; it never "decides" strategy.
2. **Us (the harness)** — everything *around* the game: fairness controls, the
   text rendering the model sees (incl. what's hidden), transcripts + viewer, the
   grader, and the RL reward. We add scaffolding; we don't change Catan's rules.
3. **The model (the policy)** — given the legal options Catanatron offers and the
   state we render, it picks **one action**. That single choice is the *only*
   thing the model contributes; everything else is computed around it.

> Why this matters: anything in column 1 is **ground truth we can trust and
> measure against** (the regret oracle, VP, awards all come from here). Anything
> in column 2 is **our design** (and our responsibility to keep fair/faithful).
> Column 3 is **what we're actually evaluating and training.**

## Catan, feature by feature

| Feature / rule / info | From **Catanatron** | From **us** | **Model** decides |
|---|---|---|---|
| **Board layout** (tiles, resources, numbers, ports) | generated from the seed | which seeds; held-out vs train split | — |
| **Dice** | rolled by `apply_action.roll_dice` | seeded i.i.d. (decoupled from global RNG) so mirrored pairs match; balanced deck opt-in; viewer replays the *recorded* dice | — |
| **Legal actions** | `generate_playable_actions(state)` — the exact legal set every ply | rendered into a numbered menu in the prompt | picks one index |
| **Applying a move / state transition** | `game.execute(action)` (validates + mutates state) | — | — |
| **Victory points** | `get_actual_victory_points` (+ visible `VICTORY_POINTS`) | shown in transcript/UI | — |
| **Longest Road** | `longest_acyclic_path` (DFS) + `maintain_longest_road` (+2 VP, ≥5, enemy nodes break it) | shown as an award badge | influenced indirectly by where it builds roads |
| **Largest Army** | `maintain_largest_army` (≥3 knights, +2 VP) | award badge | by choosing to play knights |
| **Robber** (move + steal) | `apply_action` resolves placement + steal | — | picks target tile/victim from legal options |
| **Dev cards** (buy / knight / monopoly / YoP / road-building / VP) | drawn from a seeded deck; effects applied by engine | dev *count* public, *types* private (see below) | when to buy / which to play |
| **Trades** (maritime 4:1 / 3:1 / 2:1) | legal trades enumerated + applied | trade shown as "gave X → got Y" in UI | which trade to make |
| **Win condition** (reach VP target) | `winning_color()` (`>= vps_to_win`) | we set `vps_to_win=10` + a 400-turn cap + VP tie-break at the cap | plays toward it (or not) |
| **Hidden information** | tracks `VICTORY_POINTS` (visible) vs `ACTUAL_VICTORY_POINTS` (incl. hidden VP cards); holds dev cards | **we do the masking** in the prompt (Catanatron serializes everything; it has no built-in "redact opponent's private info") | — |

### Public vs. private (our policy on top of Catanatron's data)
- **Public** (shown to both models): resource hands, visible VP, buildings/roads/
  robber, Longest Road + holder, Largest Army + holder, **dev-card count**, board.
- **Private** (only the owner): **dev-card identities** (which unplayed dev cards,
  incl. hidden VP cards) → an opponent's *visible* VP excludes hidden VP cards.
- Catanatron gives us the public/private *seam* for VP (`VICTORY_POINTS` vs
  `ACTUAL_VICTORY_POINTS`) but does **not** auto-hide anything — the prompt
  enforces the masking (`_player_line(..., public_only=True)` for the opponent).

## What only **we** add (no Catan analog)
- **Fairness**: deterministic seeds, mirrored seat-swap pairs, board+dice
  fingerprints (proof), seeded i.i.d. dice, the 400-turn cap + VP tie-break.
- **Rendering**: state → text the model reads, the legal-action menu, public/
  private masking, the rules primer.
- **Transcripts + replay viewer** (faithful: replays recorded actions+dice
  through the engine and snapshots its state).
- **Grading**: the value-function **regret** signal (engine-grounded) + the LLM
  grader's qualitative read; the emergent failure-mode taxonomy.
- **RL reward** and the before/after eval.

## What only the **model** does
- Reads the rendered state + the legal-action menu, and **chooses one action.**
  That's it. Legality, consequences, scoring, and awards are all Catanatron's.
  (If the model returns something unusable, our legality fallback picks a legal
  move — so a bad model degrades to "plays a legal move," never crashes.)

## Appendix: how Catanatron computes **Longest Road**
`longest_acyclic_path(board, component, color)` (`models/board.py`): a DFS from
every node in the player's road network that walks **only friendly roads**,
**cannot pass through an enemy-occupied node** (an opponent's settlement/city
breaks the road), and never reuses an edge (acyclic) — returning the longest such
path. `maintain_longest_road` then assigns `HAS_ROAD` + **2 VP** to the player
with the strictly-longest road of length **≥ 5**, transferring it (and the points)
when someone overtakes. The viewer reads `LONGEST_ROAD_LENGTH` and `HAS_ROAD`
straight from this — no recomputation.
