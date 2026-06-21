# Model Failure Modes — `iem5r8oy` (Qwen-based, thinking off)

Run: 20 `grader_games` boards × mirrored = **40 self-play games** (reasoning capture
on, model thinking off). Graded two ways: **per-decision** (dual Claude+OpenAI) and
**game-level** (Claude reads each finished game). Sources: `report.json`,
`findings.jsonl`, `game_review.json` (this folder).

## Headline

This model's weaknesses are **strategic and cumulative, not per-move blunders.**
Per-decision grading found almost no clear mistakes (both graders + the regret
oracle judge the moves mostly *defensible*). The real problem only shows up at the
**game level**: the model **over-trades and never builds a winning engine**, so it
stalls and loses despite individually-reasonable moves.

## Strategic failure modes (game-level — the strong signal)

| failure mode | games | % | what it looks like |
|---|---|---|---|
| **over_trading** | 33/40 | **82%** | churns maritime trades (≈**68 per game**) without converting them into builds or VP |
| **no_city_upgrades** | 28/40 | **70%** | stays on settlements; never upgrades to cities, forgoing VP + double production |
| **expansion_stall** | 27/40 | **68%** | builds roads endlessly but founds few new settlements — stalls at ~4–5 VP |
| **weak_production_engine** | 19/40 | **48%** | thin production base; relies on maritime trades instead of tiles |
| **no_path_to_10** | 8/40 | 20% | no coherent route to 10 VP; only scrapes VP from robber steals |
| **ignored_dev_cards** | 2/40 | 5% | never buys dev cards / cedes Largest Army |

**Root-cause narrative:** the model treats maritime trading as its main action
(~68 trades/game) but rarely **converts** resources into settlements/cities/dev
cards. It builds roads but doesn't found settlements (`expansion_stall`) or upgrade
to cities (`no_city_upgrades`), so its production stays thin (`weak_production_engine`)
and it never assembles a path to 10 VP. Result: games stall to the turn cap or are
lost late.

## Per-decision view (the weak/one-sided signal)

After grader calibration, the only per-decision failure that clears the floor is
**`build_spend / timing`** (robber + dev-card timing) at ~10–16% — but it is
**entirely one-sided** (Claude flags it, GPT-4o ~0%, **consensus ≈ 0**). The two
graders genuinely disagree on per-decision Catan "mistakes," and consensus is
near-empty (2 of 62 union flags). This is *why* we added the game-level pass — the
per-decision lens under-captures this model's cumulative weakness.

## Behavioral corroboration

- **≈68 maritime trades per game** — directly supports `over_trading`.
- **A win-rate 42.1%** (self-play; ~50% is the fair baseline) and **cap-stalls** —
  consistent with `expansion_stall` / `no_path_to_10`.
- Regret oracle ≈ 0 on most graded decisions — moves are locally fine; the loss is
  cumulative.

## What to target (env-gen / training)

Highest ROI, in order of prevalence:
1. **Trade discipline** — trade only to *enable a build this turn*, not as a default action.
2. **City upgrades** — convert ore/wheat into cities for VP + double production.
3. **Keep expanding** — found new settlements, don't just lay roads.
4. **Engine before trades** — build production tiles rather than leaning on the bank.

These map onto the shared taxonomy for env generation (trade `enables_key_build` /
`net_resource_value`; build_spend `vp_efficiency` / `tempo`; placement
`expansion_room`).

## Method notes / caveats

- **Game-level reviewer:** single model (Claude), one call per game, fixed
  strategic vocab (`harness/grader/game_review.py`).
- **Per-decision:** dual Claude+OpenAI, union-default merge, strict scale
  calibration; consensus is near-empty here (graders diverge), so union + the
  game-level pass are the usable signals, not consensus/κ.
- **Self-play:** both seats are the same model, so "the losing player" = this model;
  draws are turn-cap stalls.
- No human gold labels yet — the regret oracle (myopic 1-ply) + dual-grader
  agreement are the trust anchors.
