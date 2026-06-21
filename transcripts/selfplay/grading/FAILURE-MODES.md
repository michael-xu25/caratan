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

## Per-decision view (secondary signal — graded on the hardened grader)

Top per-decision weaknesses (union, Wilson-ranked): **`build_spend / timing`** (Claude's
lens, ~12–15%) and **placement `pip_coverage` / `resource_diversity`** (GPT-4o's lens,
~16–17%). Each is largely *one-sided* — the two graders favor different criteria — but
after loosening the scale they now **both contribute** (union 113, **consensus 11**, κ
0.156, up from union 62 / consensus 2 / κ 0.058). Still, per-decision blunders are rare
(regret ≈ 0 on most), which is exactly why the **game-level table above is the headline**
and the per-decision view is supporting detail.

### Grading robustness (this run is trustworthy)
Hardened per review feedback — the denominator is the load-bearing thing and it's clean:
- **0 oracle drops** (all decision types fully graded; trades 0%, build_spend 0% after a
  type-match fallback), **0 games broke early** — so no per-type denominator bias.
- **Parse failures excluded from the denominator** and reported: Claude 0.2%, GPT-4o 0.0%.
- **action_type assertion** fires on any context misalignment — validated 0 fires across
  600 prompts (the off-by-one bug class can't silently recur).

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

## See it in the viewer

Open the run in `viewer/` and click **⚖️ Grading** — it shows this game's failure
modes and, as you step through, each graded decision's verdict (failed criteria,
disputed/agreed, both graders' reasons, regret). Plies marked `⚖`/`⚖✗` were graded.

## Method notes / caveats

- **Game-level reviewer:** single model (Claude), one call per game, fixed
  strategic vocab (`harness/grader/game_review.py`). The strategic table is ranked by
  **presence-count** (# games a mode appears in), NOT Wilson-discounted — so read the
  82/70/68% as "share of games containing this mode," and its ordering less rigorously
  than the Wilson-ranked per-decision table.
- **Per-decision:** dual Claude+OpenAI, union-default merge, calibrated 0/1/2 scale;
  the two graders favor different criteria, so union (recall) + the game-level pass are
  the usable signals — κ/consensus are informative but low here by base-rate (raw
  agreement is ~97%).
- **Self-play:** both seats are the same model, so "the losing player" = this model;
  draws are turn-cap stalls.
- No human gold labels yet — the regret oracle (myopic 1-ply) + dual-grader
  agreement are the trust anchors.
