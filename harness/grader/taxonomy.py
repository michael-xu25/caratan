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
    "resource_diversity": "breadth of resource types covered (fail: locked out of / doubled-up on a resource)",
    "pip_coverage": "probability-weighted production / sum of dots (fail: low pips, sitting on 2s/12s)",
    "port_access": "useful port adjacency given the resource plan (fail: mismatched or ignored strong port)",
    "expansion_room": "open buildable nodes/roads reachable later (fail: boxed in, no second-ring spots)",
    "blocking_value": "denying a strong spot to an opponent (fail: leaves a premium node open)",
    "net_resource_value": "value given vs received (fail: trades down on raw value)",
    "enables_key_build": "unlocks a settlement/city/dev this turn (fail: gives away a needed card)",
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
