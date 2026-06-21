"""Async 1v1 match runner with seeded + mirrored games.

Catanatron is millisecond-fast and synchronous; the only slow part is an LLM
backend's network call inside `decide()`. So each game runs to completion in a
worker thread, and we run many games concurrently under a semaphore whose size
is the real ceiling: the number of concurrent LLM calls you want in flight.

Fairness:
- Each `seed` fixes the board layout and dice rolls.
- `mirror=True` plays every seed twice with seats swapped (A as RED then A as
  BLUE), so first-move / luck asymmetries cancel in the aggregate.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from catanatron import Color, Game
from catanatron.state_functions import get_actual_victory_points

from goldilocks_eval.agents.factory import label_for, make_player
from goldilocks_eval.dice import seeded_dice
from goldilocks_eval.transcript import write_transcripts

# 1v1 throughout (README): two fixed seats.
SEATS = (Color.RED, Color.BLUE)


@dataclass
class GameOutcome:
    game_id: str
    seed: int
    orientation: str            # "AB" (A=RED) or "BA" (B=RED)
    winner_color: Optional[str]
    winner_side: str            # "A", "B", or "DRAW"
    turns: int
    victory_points: Dict[str, int]


@dataclass
class MatchResult:
    label_a: str
    label_b: str
    outcomes: List[GameOutcome] = field(default_factory=list)

    @property
    def a_wins(self) -> int:
        return sum(o.winner_side == "A" for o in self.outcomes)

    @property
    def b_wins(self) -> int:
        return sum(o.winner_side == "B" for o in self.outcomes)

    @property
    def draws(self) -> int:
        return sum(o.winner_side == "DRAW" for o in self.outcomes)

    @property
    def decided(self) -> int:
        return self.a_wins + self.b_wins

    @property
    def a_winrate(self) -> float:
        return self.a_wins / self.decided if self.decided else 0.0

    def summary(self) -> str:
        n = len(self.outcomes)
        lines = [
            f"Match: {self.label_a} (A) vs {self.label_b} (B)",
            f"Games: {n}   draws/timeouts: {self.draws}",
            f"  A '{self.label_a}': {self.a_wins} wins",
            f"  B '{self.label_b}': {self.b_wins} wins",
            f"  A win-rate (of decided): {self.a_winrate:.1%}",
        ]
        return "\n".join(lines)


def _play_one(label_a: str, label_b: str, spec_a: str, spec_b: str,
              seed: int, orientation: str, out_dir: Optional[str]) -> GameOutcome:
    """Synchronous single game (runs inside a worker thread)."""
    if orientation == "AB":
        seat_specs = {Color.RED: spec_a, Color.BLUE: spec_b}
        seat_labels = {Color.RED: label_a, Color.BLUE: label_b}
        side_of = {Color.RED: "A", Color.BLUE: "B"}
    else:
        seat_specs = {Color.RED: spec_b, Color.BLUE: spec_a}
        seat_labels = {Color.RED: label_b, Color.BLUE: label_a}
        side_of = {Color.RED: "B", Color.BLUE: "A"}

    players = [make_player(seat_specs[c], c) for c in SEATS]
    game = Game(players, seed=seed)
    # Seeded dice decoupled from the global RNG so a mirrored pair sees identical
    # rolls (default i.i.d., per the build-spec). NOTE: board generation still
    # draws from the global RNG, so for *concurrent* head-to-head use the
    # process-isolated runner in `harness/` (canonical for multi-game runs).
    with seeded_dice(seed):
        winner = game.play()  # Color or None (turn-limit timeout)

    vps = {c.value: get_actual_victory_points(game.state, c) for c in SEATS}
    winner_color = winner.value if winner is not None else None
    winner_side = side_of[winner] if winner is not None else "DRAW"

    game_id = f"{seed}_{orientation}"
    if out_dir is not None:
        meta = {
            "game_id": game_id,
            "seed": seed,
            "orientation": orientation,
            "seats": {c.value: seat_labels[c] for c in SEATS},
            "winner": winner_color,
            "winner_agent": seat_labels[winner] if winner is not None else "DRAW",
            "winner_side": winner_side,
            "turns": game.state.num_turns,
            "victory_points": vps,
        }
        write_transcripts(out_dir, game, meta, players)

    return GameOutcome(
        game_id=game_id,
        seed=seed,
        orientation=orientation,
        winner_color=winner_color,
        winner_side=winner_side,
        turns=game.state.num_turns,
        victory_points=vps,
    )


async def run_match(spec_a: str, spec_b: str, seeds: List[int],
                    concurrency: int = 8, out_dir: Optional[str] = None,
                    mirror: bool = True, progress: bool = True) -> MatchResult:
    label_a, label_b = label_for(spec_a), label_for(spec_b)
    result = MatchResult(label_a=label_a, label_b=label_b)

    jobs = []
    for seed in seeds:
        jobs.append((seed, "AB"))
        if mirror:
            jobs.append((seed, "BA"))

    sem = asyncio.Semaphore(concurrency)
    done = 0
    total = len(jobs)

    async def worker(seed: int, orientation: str) -> GameOutcome:
        nonlocal done
        async with sem:
            outcome = await asyncio.to_thread(
                _play_one, label_a, label_b, spec_a, spec_b,
                seed, orientation, out_dir,
            )
        done += 1
        if progress:
            print(f"  [{done}/{total}] seed={seed} {orientation} "
                  f"-> {outcome.winner_side} ({outcome.turns} turns)", flush=True)
        return outcome

    result.outcomes = await asyncio.gather(
        *(worker(seed, orient) for seed, orient in jobs)
    )
    # Stable order for reproducible reports.
    result.outcomes.sort(key=lambda o: (o.seed, o.orientation))
    return result
