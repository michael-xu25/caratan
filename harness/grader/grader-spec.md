# Catan Decision Grader — Spec

Feed the **System Prompt** block to the grader. It scores one decision at a time and
emits one JSON object per decision. Outputs aggregate into discovered weaknesses.

**Cardinal rule:** the grader may ONLY use criterion IDs and state tags from the fixed
lists below. No invented names, no free-text labels. Aggregation groups on these exact
strings — one typo or synonym silently splits a weakness into two and breaks the loop.

---

## The weakness label = (decision_type, criterion, state_tags)

That triple is the unit everything turns on:
- The **rubric criteria** and the **weakness taxonomy** are the same list (by design).
- A failed criterion in a given state IS the weakness label.
- Aggregation: "criterion C failed in X% of decisions tagged Z" → discovered weakness.
- Env generation: "make scenarios that stress C in state Z" → loop closes.

So criterion IDs and state tags must be frozen *before* grading starts. Freeze them
with Michael, since his generated envs must be tagged from the same vocab your scorer reads.

---

## Criterion vocabulary (fixed IDs)

### decision_type = `placement`
| id | what it measures | a FAIL looks like |
|---|---|---|
| `resource_diversity` | breadth of resource types the player can produce | left unable to produce a key resource with no realistic path to it (mild doubling / 3-4 types is fine) |
| `pip_coverage` | probability-weighted production (sum of dot values) | takes a clearly low-pip spot when a materially higher-pip legal spot was open (slightly under max is fine) |
| `port_access` | useful port adjacency given resource plan | takes a port that doesn't fit the plan, or ignores a strong port that was clearly the single best spot (n/a→2 for most placements) |
| `expansion_room` | keeps the player's OWN future expansion open | genuinely boxed in — surrounded, no reachable open spots later (an opening on an open board is rarely a fail) |
| `blocking_value` | denying the opponent a strong spot | passed up an obvious chance to take the opponent's clearly-best node when it was also good for the player (a good node merely remaining open is not a fail) |

### decision_type = `trade`
| id | what it measures | a FAIL looks like |
|---|---|---|
| `net_resource_value` | resource utility/scarcity given vs received (NOT raw card count) | trades a scarcer/more-useful resource for a less-useful one, or a worse ratio than an available port (a plain 4:1/3:1 bank trade is not itself a fail) |
| `enables_key_build` | advances toward a needed build (do NOT require completing a build this turn) | trades away a resource needed for a planned/imminent build, or trades with no constructive purpose |
| `opponent_benefit` | does it help them more than you | hands opponent their missing piece |
| `timing_strength` | trading from strength vs desperation | panic-trades a scarce resource away |

### decision_type = `build_spend`
| id | what it measures | a FAIL looks like |
|---|---|---|
| `tempo` | speed toward next VP / engine step | sinks resources in a low-urgency play |
| `vp_efficiency` | VP per resource spent | over-invests for marginal VP |
| `board_control` | road/settlement positioning, longest-road race | cedes a key route or longest road |
| `timing` | robber + dev-card timing | wastes knight/robber, mistimes a dev buy |

> To extend: add a row with a new stable ID. Never rename an existing ID mid-run —
> renames orphan all prior aggregates.

---

## State-tag vocabulary (fixed; multi-tag per decision)

This is the `Z` in "criterion fails in state Z." Tag every decision with all that apply.
Keep this list frozen and shared with env generation.

**phase:** `opening` · `early` · `mid` · `late` · `endgame`
**standing:** `leading` · `even` · `behind` · `far_behind`
**resources:** `resource_rich` · `resource_starved` · `near_discard` (8+ cards)
**board:** `open_board` · `contested` · `boxed_in`
**flags:** `robber_threat` · `has_longest_road` · `has_largest_army` ·
`self_one_from_win` · `opp_one_from_win`

---

## Score scale (per criterion)

| score | meaning | `failed` |
|---|---|---|
| 2 | good / no issue | false |
| 1 | suboptimal but defensible | false |
| 0 | clear mistake on this criterion | **true** |

Score every criterion listed for the decision's type — even the ones that pass.
A criterion that isn't relevant to this specific decision → score `2`, `failed: false`,
reason `"n/a"`. (Don't drop it; missing rows distort the denominator in aggregation.)

`failed` is the explicit aggregation key. Score gives nuance; the boolean is what you count.

---

## Output schema (one object per decision, strict JSON)

> **⚠️ IMPLEMENTATION NOTE (current, read this).** The shipped grader does NOT ask
> the LLM for `state_tags` or `weakness_labels`. **`state_tags` are derived
> objectively from the engine** (`harness/grader/context.py`) so BOTH graders share
> identical aggregation buckets (a grader can't split a bucket by tagging
> differently). The LLM is asked for **only** `{criteria, summary}`. `decision_type`
> is also engine-derived. `weakness_labels` is **vestigial** — the aggregator
> explodes `criteria × state_tags` itself and never reads it. **For env generation,
> tag your envs from the same frozen `taxonomy.py` vocab — do NOT mirror
> grader-assigned tags (there are none).** The full record below is the *stored*
> object (engine fields + LLM criteria merged); the LLM only produces the criteria.

```json
{
  "decision_id": "string (engine: game_id:ply)",
  "decision_type": "placement | trade | build_spend   // engine-derived",
  "state_tags": ["from the fixed vocab only            // engine-derived"],
  "criteria": [
    { "name": "fixed_criterion_id", "score": 0, "failed": true, "reason": "<= 1 sentence" }
  ],
  "summary": "<= 1 sentence                              // LLM",
  "weakness_labels": [ /* vestigial — aggregator ignores it */ ]
}
```

---

## System Prompt (original spec — the LIVE prompt is `prompts.py:SYSTEM_DECISION`)

> The block below is the original rubric prompt and still references grader-assigned
> tags/`weakness_labels` (rules 3 & 5). The **shipped** prompt is
> `harness/grader/prompts.py::SYSTEM_DECISION`: it asks only for `{criteria, summary}`
> (tags are engine-derived), uses the calibrated 0/1/2 scale, and frames the regret
> oracle as a myopic 1-ply hint. Treat `prompts.py` as the source of truth.

```
You are a Settlers of Catan decision grader. You receive ONE decision: the full game
state, the legal options that were available, and the choice the player made (with its
stated reasoning if present). You output a single JSON object scoring that choice.

Rules:
1. Identify the decision_type: placement, trade, or build_spend.
2. Score EVERY criterion for that type, using ONLY the fixed criterion IDs you were given.
   Scale: 2 = good, 1 = suboptimal-but-defensible, 0 = clear mistake. Mark failed=true
   only on score 0. For a criterion not relevant to this decision, score 2, failed=false,
   reason "n/a".
3. Tag the decision with state_tags, using ONLY the fixed state vocabulary. Apply all
   that hold.
4. Judge the choice against the legal options actually available — not an ideal that
   wasn't on the menu. Reason about what the better legal option would have been.
5. For each failed criterion, add a weakness_labels entry pairing it with the state_tags.
6. Keep every reason to one sentence. Output ONLY the JSON object, no prose, no markdown.

Never invent criterion names or state tags outside the provided lists. If a mistake
doesn't fit any criterion, attach it to the closest one and note it in the reason —
do not create a new label.
```

---

## Pipeline (how this drives the loop)

1. **Log** at each decision: `(decision_id, state, legal_options, choice, reasoning)`.
2. **Grade** each logged decision with the spec above → per-criterion JSON.
3. **Aggregate** all `weakness_labels`: group by `(decision_type, criterion, state_tags)`,
   compute fail-rate = failures / times-that-criterion-was-scored-in-that-state.
   High fail-rate in a state = a discovered weakness.
4. **Generate** envs that stress the top weaknesses (criterion × state) — Michael's half,
   reading the same vocab.
5. **Re-grade** on held-out instances post-training; show the fail-rate on that
   (criterion, state) drop. That delta is the demo.

The before/after number is per-(criterion, state), which is exactly your primary metric
("robber-when-behind: 30% → 78%" = criterion `timing` + tag `behind`, fail-rate inverted).
