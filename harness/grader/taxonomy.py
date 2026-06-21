"""Closed-set failure-mode taxonomy for the dual grader.

Both graders MUST classify into this fixed enum (plus `other` / `none`) — a
closed set is what makes the two graders' labels comparable and merges
measurable (free text never string-matches). Categories double as the
`weakness` field handed to env generation. Edit here = single source of truth.

Mirrors grading-rubric-proposal.md §6.
"""
from __future__ import annotations

# label -> one-line definition (shown to the graders so both anchor on the same
# meaning of each category).
TAXONOMY: dict[str, str] = {
    "placement-low-pip": "settlement/city on weak production when a higher-pip option was legal",
    "placement-no-variety": "ignored resource diversity / port synergy at placement",
    "boxed-in": "failed to keep expansion open; got walled off from buildable nodes",
    "robber-not-denying-leader": "robber/knight not used to block the leader's best tile",
    "robber-victim-suboptimal": "moved robber/stole from the wrong opponent or tile",
    "overheld-cards": "sat on >7 cards and took avoidable discards",
    "inefficient-hand": "held resources without converting them into a build",
    "bad-trade": "maritime/port trade with poor expected value",
    "longest-road-ignored": "neglected longest road when expansion was blocked",
    "dev-timing": "bought/played a dev card at the wrong time",
    "reasoning-inconsistent": "stated reasoning did not match/justify the action (qual-only)",
    "tempo-misread": "wrong risk posture for being ahead/behind",
    # escapes — keep the set closed but let a grader decline a label honestly.
    "other": "a real mistake that fits none of the above categories",
    "none": "not actually a mistake (the oracle's regret looks like a value-fn artifact)",
}

LABELS: tuple[str, ...] = tuple(TAXONOMY.keys())

# Categories that are about reasoning rather than the move itself — only
# meaningful when the model's stated reasoning was captured.
QUAL_ONLY: frozenset[str] = frozenset({"reasoning-inconsistent"})


def is_valid_label(label: str) -> bool:
    return label in TAXONOMY


def normalize_label(label: str | None) -> str:
    """Coerce a grader's raw category string to a known label; fall back to
    `other` for an unrecognized-but-nonempty answer, `none` for empty."""
    if not label:
        return "none"
    key = label.strip().lower().replace(" ", "-").replace("_", "-")
    return key if key in TAXONOMY else "other"


def taxonomy_block() -> str:
    """Render the taxonomy as a prompt fragment (label: definition lines)."""
    return "\n".join(f"- {label}: {desc}" for label, desc in TAXONOMY.items())
