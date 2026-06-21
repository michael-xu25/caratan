"""Dice fairness — re-exported from the canonical `goldilocks_eval.dice`.

The implementation (the determinism fix Cara found + the i.i.d.-vs-balanced
choice) now lives in `goldilocks_eval/dice.py` so there's one source of truth,
the same way `harness.scenario` / `harness.backends` re-export Michael's code.

Per the build-spec decision: default dice are **seeded i.i.d.** (decoupled from
the global RNG, so a mirrored pair sees identical rolls); the balanced colonist
deck is opt-in via `seeded_dice(seed, balanced=True)` / `balanced_dice(seed)`.
"""
from __future__ import annotations

from goldilocks_eval.dice import (  # noqa: F401
    BalancedDice,
    SeededDice,
    balanced_dice,
    seeded_dice,
)
