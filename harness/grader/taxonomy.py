"""
taxonomy.py — the single source of truth for the grading / weakness vocabulary.

Everyone imports from here:
  - the grader        (criterion + tag lists go into its prompt)
  - the aggregator    (validation + the keys it groups on)
  - Michael's env gen (tagging generated scenarios)

Freezing the vocab as code is what stops the two halves of the loop from drifting
apart: a weakness Michael *generates for* and a weakness Cara *measures* must be the
SAME string, or aggregation silently splits one weakness into two and the loop never
closes. A mismatched tag should fail loudly (import/validation error), not silently.

RULE: add IDs by appending. NEVER rename an existing ID mid-run — a rename orphans
every aggregate computed before it.
"""

# ---------------------------------------------------------------- criteria
# Criterion IDs per decision type. The failed criterion IS the weakness label,
# so these names are the weakness taxonomy, not just a rubric.

PLACEMENT_CRITERIA = [
    "resource_diversity",   # breadth of resource types covered
    "pip_coverage",         # probability-weighted production (sum of dot values)
    "port_access",          # useful port adjacency given the resource plan
    "expansion_room",       # open buildable nodes/roads reachable later
    "blocking_value",       # denying a strong spot to an opponent
]

TRADE_CRITERIA = [
    "net_resource_value",   # value given vs received
    "enables_key_build",    # unlocks a settlement/city/dev this turn
    "opponent_benefit",     # does it help them more than you (fail = yes)
    "timing_strength",      # trading from strength vs desperation
]

BUILD_SPEND_CRITERIA = [
    "tempo",                # speed toward next VP / engine step
    "vp_efficiency",        # VP per resource spent
    "board_control",        # road/settlement positioning, longest-road race
    "timing",               # robber + dev-card timing
]

CRITERIA_BY_TYPE = {
    "placement":   PLACEMENT_CRITERIA,
    "trade":       TRADE_CRITERIA,
    "build_spend": BUILD_SPEND_CRITERIA,
}

DECISION_TYPES = list(CRITERIA_BY_TYPE.keys())

# ---------------------------------------------------------------- state tags
# The 'Z' in "criterion C fails in state Z". Multi-tag per decision: apply all
# that hold. Grouped by family below for readability; validation uses the flat set.

STATE_TAGS = {
    # phase
    "opening", "early", "mid", "late", "endgame",
    # standing
    "leading", "even", "behind", "far_behind",
    # resources
    "resource_rich", "resource_starved", "near_discard",   # near_discard = 8+ cards
    # board
    "open_board", "contested", "boxed_in",
    # flags
    "robber_threat", "has_longest_road", "has_largest_army",
    "self_one_from_win", "opp_one_from_win",
}

# ---------------------------------------------------------------- validators

def validate_decision_type(dtype):
    if dtype not in CRITERIA_BY_TYPE:
        raise ValueError(
            f"unknown decision_type {dtype!r}; expected one of {DECISION_TYPES}")

def validate_criterion(dtype, name):
    valid = CRITERIA_BY_TYPE[dtype]
    if name not in valid:
        raise ValueError(
            f"unknown criterion {name!r} for decision_type {dtype!r}; "
            f"expected one of {valid}")

def validate_tags(tags):
    bad = [t for t in tags if t not in STATE_TAGS]
    if bad:
        raise ValueError(
            f"unknown state tag(s) {bad}; not in the frozen STATE_TAGS vocab. "
            f"Add to taxonomy.STATE_TAGS first if intentional.")


# ---------------------------------------------------------------- prompt helpers
# Render the frozen vocab into prompt fragments so the grader sees exactly the IDs
# the aggregator will group on (no drift between prompt and code).

_CRITERION_DESC = {
    "resource_diversity": "breadth of resource types the player can produce (fail: leaves the player unable to produce a key resource with no realistic path to it; mild doubling or covering 3-4 types is NOT a fail)",
    "pip_coverage": "probability-weighted production / sum of dots (fail: takes a clearly low-pip spot when a MATERIALLY higher-pip legal spot was open; being slightly under the max is NOT a fail)",
    "port_access": "useful port adjacency given the resource plan (fail: takes a port that doesn't fit the plan, or ignores a strong port that was clearly the single best available spot; ports are optional, so this is n/a -> 2 for most placements)",
    "expansion_room": "keeps the player's OWN future expansion open (fail: genuinely boxed in — surrounded, no reachable open building spots later; an opening on an open board is almost never a fail here)",
    "blocking_value": "denying the opponent a strong spot (fail: passed up an obvious, available chance to take the opponent's clearly-best node when it was also good for the player; 'a good node remains open' is NOT a fail — you can't take them all)",
    "net_resource_value": "resource utility/scarcity given vs received, NOT raw card count (fail: trades a scarcer/more-useful resource for a less-useful one, or accepts a worse ratio than an available port; a plain 4:1/3:1 bank trade is NOT a fail by itself)",
    "enables_key_build": "advances toward a needed build (fail: trades away a resource it needed for a planned/imminent build, or trades with no constructive purpose; do NOT fail a trade merely because it doesn't COMPLETE a build this turn)",
    "opponent_benefit": "helps them more than you (fail: hands opponent their missing piece)",
    "timing_strength": "trading from strength vs desperation (fail: panic-trades a scarce resource)",
    "tempo": "speed toward next VP / engine step (fail: low-urgency sink)",
    "vp_efficiency": "VP per resource spent (fail: over-invests for marginal VP)",
    "board_control": "road/settlement positioning, longest-road race (fail: cedes a key route)",
    "timing": "robber + dev-card timing (fail: wastes knight/robber, mistimes a dev buy)",
}

_TAG_FAMILIES = [
    ("phase", ["opening", "early", "mid", "late", "endgame"]),
    ("standing", ["leading", "even", "behind", "far_behind"]),
    ("resources", ["resource_rich", "resource_starved", "near_discard"]),
    ("board", ["open_board", "contested", "boxed_in"]),
    ("flags", ["robber_threat", "has_longest_road", "has_largest_army",
               "self_one_from_win", "opp_one_from_win"]),
]


def criteria_block(dtype: str) -> str:
    return "\n".join(f"- {c}: {_CRITERION_DESC.get(c, '')}" for c in CRITERIA_BY_TYPE[dtype])


def tags_block() -> str:
    return "\n".join(f"{fam}: {', '.join(tags)}" for fam, tags in _TAG_FAMILIES)
