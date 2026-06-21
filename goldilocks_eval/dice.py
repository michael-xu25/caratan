"""Seeded, decision-independent dice (canonical; harness/dice.py re-exports this).

Catanatron rolls from Python's **global** random stream, so the two halves of a
mirrored pair desync their dice the moment play diverges — reintroducing exactly
the luck mirroring is supposed to cancel (Cara observed same board, 66 vs 382
turns). We deal dice from a dedicated per-seed RNG instead, so roll #N is
identical across both halves regardless of dev-card/robber draws.

Default = `SeededDice`: independent (i.i.d.) rolls — the build-spec decision
(seeded purely-random). A balanced deck is *countable* (an artifact real Catan
lacks) and its variance reduction is redundant under mirroring, so the
colonist-style `BalancedDice` is opt-in for A/B only.

Thread- AND process-safe: one global dispatcher reads a thread-local deck, so
concurrent games never collide on the patched `apply_action.roll_dice` symbol.
"""
from __future__ import annotations

import hashlib
import random as _random
import threading
from contextlib import contextmanager

import catanatron.apply_action as _apply_action

# The 36 equally-likely ordered face combinations of two dice.
_ALL_FACES = [(a, b) for a in range(1, 7) for b in range(1, 7)]


class SeededDice:
    """i.i.d. rolls from a dedicated per-seed RNG (decoupled from the global
    stream). The build-spec default."""

    def __init__(self, seed: int):
        self._rng = _random.Random(seed)
        self.dealt: list[tuple[int, int]] = []

    def roll(self) -> tuple[int, int]:
        faces = (self._rng.randint(1, 6), self._rng.randint(1, 6))
        self.dealt.append(faces)
        return faces

    def fingerprint(self) -> str:
        """Stable hash of the dice dealt — lets a pair *prove* dice identity."""
        s = ",".join(f"{a}{b}" for a, b in self.dealt)
        return hashlib.sha1(s.encode()).hexdigest()[:12]


class BalancedDice(SeededDice):
    """Colonist-style shuffled-36 deck (opt-in A/B). Each 36-roll cycle yields
    the exact 2–12 distribution; lower variance but countable."""

    def __init__(self, seed: int):
        super().__init__(seed)
        self._deck: list[tuple[int, int]] = []

    def roll(self) -> tuple[int, int]:
        if not self._deck:
            self._deck = list(_ALL_FACES)
            self._rng.shuffle(self._deck)
        faces = self._deck.pop()
        self.dealt.append(faces)
        return faces


# --- thread/process-safe install: a global dispatcher over a thread-local deck.
_local = threading.local()
_ORIGINAL_ROLL = _apply_action.roll_dice
_installed = False
_install_lock = threading.Lock()


def _dispatch() -> tuple[int, int]:
    deck = getattr(_local, "deck", None)
    return deck.roll() if deck is not None else _ORIGINAL_ROLL()


def _ensure_installed() -> None:
    global _installed
    if not _installed:
        with _install_lock:
            if not _installed:
                # Referenced by bare name inside apply_action, so a module-attr
                # swap takes effect at the call site. Falls back to vanilla when
                # no thread-local deck is set, so unpatched games are unaffected.
                _apply_action.roll_dice = _dispatch
                _installed = True


@contextmanager
def seeded_dice(seed: int, balanced: bool = False):
    """Install a per-seed dice deck for one game. Yields it so the caller can
    read `.fingerprint()` / `.dealt` afterwards. `balanced=True` opts into the
    colonist deck; default is i.i.d. seeded (the spec)."""
    _ensure_installed()
    deck = BalancedDice(seed) if balanced else SeededDice(seed)
    prev = getattr(_local, "deck", None)
    _local.deck = deck
    try:
        yield deck
    finally:
        _local.deck = prev


@contextmanager
def balanced_dice(seed: int):
    """Back-compat alias (the colonist deck). Prefer `seeded_dice(seed, balanced=...)`."""
    with seeded_dice(seed, balanced=True) as deck:
        yield deck
