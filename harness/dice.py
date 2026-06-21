"""Balanced, decision-independent dice — the dice half of fairness.

Why this exists
---------------
Catanatron rolls dice from Python's **global** `random` stream as the game
advances (`apply_action.roll_dice` -> `random.randint(1, 6)` twice). The number
of draws from that stream depends on what the players *do* (dev-card buys,
robber/knight, steals), so in a mirrored pair the two seatings desync the moment
they make a different action and from then on see **different dice** — exactly
the dice luck that mirroring is supposed to cancel. (Observed directly: same
board, 66 turns one seating vs 382 the other.)

The fix
-------
Deal dice from a DEDICATED `random.Random(seed)` deck, independent of the global
stream. Both games of a mirrored pair use the same `seed` -> the same deck, and
since a 1v1 turn always rolls exactly once, **the Nth roll is identical in both
games regardless of any dev-card/robber RNG**. Dice become a pure function of
(seed, turn index). Everything non-dice still draws from the global RNG, so the
board (generated at init) and dev-deck shuffle stay identical too.

"Balanced" (colonist-style)
---------------------------
Instead of two independent d6s, deal from a shuffled deck of the 36 (d1, d2)
face combinations, reshuffling once exhausted. Over each 36-roll cycle every
outcome appears exactly once, so the 2..12 sum distribution matches true
probability with far less variance than i.i.d. `randint` — fewer "no 6s for 20
rolls" swings polluting a short eval game.

Removable
---------
`balanced_dice(seed)` monkeypatches `catanatron.apply_action.roll_dice` in the
CURRENT process only (each game runs in its own worker process, so there is no
cross-game contamination). To revert to vanilla global-RNG dice, stop calling it
in `runner.py` — or delete this file. Nothing else depends on it.
"""

from __future__ import annotations

import hashlib
import random as _random
from contextlib import contextmanager

import catanatron.apply_action as _apply_action

# The 36 equally-likely ordered face combinations of two dice.
_ALL_FACES = [(a, b) for a in range(1, 7) for b in range(1, 7)]


class BalancedDice:
    """A per-seed dice deck dealt from its own RNG (not the global stream)."""

    def __init__(self, seed: int):
        # A dedicated RNG so dice never touch — and are never perturbed by —
        # the global stream Catanatron uses for board/dev-cards/robber.
        self._rng = _random.Random(seed)
        self._deck: list[tuple[int, int]] = []
        self.dealt: list[tuple[int, int]] = []  # full history, for the fingerprint

    def _refill(self) -> None:
        self._deck = list(_ALL_FACES)
        self._rng.shuffle(self._deck)

    def roll(self) -> tuple[int, int]:
        """Drop-in replacement for `apply_action.roll_dice`."""
        if not self._deck:
            self._refill()
        faces = self._deck.pop()
        self.dealt.append(faces)
        return faces

    def fingerprint(self) -> str:
        """Stable hash of the dice actually dealt this game.

        Two games that saw the same dice sequence produce the same fingerprint —
        so a mirrored pair can *prove* (not just assert) the dice were identical,
        the same way `board_fingerprint` proves the board was.
        """
        s = ",".join(f"{a}{b}" for a, b in self.dealt)
        return hashlib.sha1(s.encode()).hexdigest()[:12]


@contextmanager
def balanced_dice(seed: int):
    """Install a per-seed balanced dice deck for the duration of one game.

    Patches `apply_action.roll_dice` (referenced there by bare name, so a module
    attribute swap takes effect at the call site) and restores it on exit.
    Yields the deck so the caller can read `deck.fingerprint()` afterwards.
    """
    deck = BalancedDice(seed)
    original = _apply_action.roll_dice
    _apply_action.roll_dice = deck.roll
    try:
        yield deck
    finally:
        _apply_action.roll_dice = original
