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
| `resource_diversity` | breadth of resource types covered | locked out of / doubled-up on a resource |
| `pip_coverage` | probability-weighted production (sum of dot values) | low total pips; sitting on 2s/12s |
| `port_access` | useful port adjacency given resource plan | takes a port that doesn't match production, or ignores a strong one |
| `expansion_room` | open buildable nodes/roads reachable later | boxed in, no second-ring spots |
| `blocking_value` | denying a strong spot to an opponent | leaves a premium node open to next player |

### decision_type = `trade`
| id | what it measures | a FAIL looks like |
|---|---|---|
| `net_resource_value` | value given vs received | trades down on raw value |
| `enables_key_build` | unlocks a settlement/city/dev this turn | gives away a card needed for own build |
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

```json
{
  "decision_id": "string, copied verbatim from the transcript",
  "decision_type": "placement | trade | build_spend",
  "state_tags": ["from the fixed vocab only"],
  "criteria": [
    { "name": "fixed_criterion_id", "score": 0, "failed": true, "reason": "<= 1 sentence" }
  ],
  "summary": "<= 1 sentence: the single most important thing about this decision",
  "weakness_labels": [
    { "criterion": "fixed_criterion_id", "state_tags": ["..."] }
  ]
}
```

`weakness_labels` = one entry per failed criterion, pairing it with this decision's
state_tags. It's derivable from `criteria` + `state_tags`, but emit it explicitly so
the aggregator just concatenates these lists across all games — no post-processing.

---

## System Prompt (paste this into the grader)

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
