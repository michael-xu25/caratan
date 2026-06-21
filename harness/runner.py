"""Async match runner.

Catanatron itself is ms-fast and synchronous; the only slow part of a "game"
is waiting on LLM calls. So we fan games out and bound how many run at once.
With one decision in flight per game, the concurrency limit IS the
concurrent-LLM-call limit -- exactly the throughput / rate-limit ceiling the
build doc calls out. Tune `concurrency` to your model's limits.

Processes, not threads. Catanatron seeds and draws from Python's **global**
`random` module (verified: no local-RNG path, hardwired to the module). Running
games as threads makes them share that one RNG; concurrent games interleave its
calls and corrupt each other's board + dice (verified: mirrored boards diverged
under thread concurrency). We therefore run each game in its own PROCESS via a
ProcessPoolExecutor -- each gets an independent global RNG, and the worker
boundary takes specs (strings) in / a primitives dataclass out, so no
unpicklable agent/LLM-client objects ever cross it.

NOTE on reproducibility: per-process isolation removes the concurrency
corruption, but byte-identical replay across separate runs also depends on
Python's hash seed (Catanatron's action ordering uses sets/dicts). Full
seed->dice control is being handled by the team's own randomness/balanced-dice
system; this runner just needs to isolate games and assign seats fairly.

Fairness:
  * Every match is seeded -> reproducible board + dice.
  * `run_batch(..., mirror=True)` plays every seed twice with seats swapped.
    Verified property for 1v1: reusing a seed with the player list swapped
    yields an IDENTICAL board and cleanly swapped seating (RED<->BLUE). Dice
    are identical too when both policies consume RNG identically; for policies
    that draw from Python's RNG differently (or trigger different robber/dev
    draws) the dice streams can diverge -- a decision-independent "balanced
    dice deck" is the proper fix and is noted as future work.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.state_functions import get_actual_victory_points

from harness.agents import make_agent
from harness.transcripts import TranscriptAccumulator
from harness.determinism import make_pool  # single, removable determinism shim

# 1v1 throughout: seat 0 = RED, seat 1 = BLUE.
SEATS = (Color.RED, Color.BLUE)


@dataclass
class MatchResult:
    seed: int
    label: str
    agent_a: str
    agent_b: str
    seat_of_a: Optional[Color]      # which color agent A actually played
    winner: Optional[str]           # "A" | "B" | None (true tie at turn cap)
    winner_color: Optional[str]
    vp_a: int
    vp_b: int
    num_turns: int
    duration: float
    truncated: bool = False        # hit the turn cap (winner decided by VP, not by reaching vps_to_win)
    board_fingerprint: str = ""    # identical across a mirrored pair == fair board
    dice_fingerprint: str = ""     # hash of THIS game's full dice sequence (replay id)
    dice_rolls: tuple = ()         # the (d1,d2) sequence dealt; prefix-equal across a fair pair
    json_path: Optional[str] = None
    log_path: Optional[str] = None


@dataclass
class BatchResult:
    agent_a: str
    agent_b: str
    matches: list = field(default_factory=list)
    num_pairs: int = 0

    @property
    def wins_a(self) -> int:
        return sum(m.winner == "A" for m in self.matches)

    @property
    def wins_b(self) -> int:
        return sum(m.winner == "B" for m in self.matches)

    @property
    def draws(self) -> int:
        return sum(m.winner is None for m in self.matches)

    @property
    def win_rate_a(self) -> float:
        decided = self.wins_a + self.wins_b
        return self.wins_a / decided if decided else 0.0


def _board_fingerprint(game) -> str:
    """Stable hash of the board layout (tiles -> resource+number).

    Two games with an identical board produce the same fingerprint. We use it
    to PROVE a mirrored pair was played on the same board, not just assert it.
    """
    import hashlib

    parts = []
    for coord, tile in sorted(game.state.board.map.tiles.items(), key=lambda kv: str(kv[0])):
        resource = getattr(getattr(tile, "resource", None), "value", None)
        number = getattr(tile, "number", None)
        parts.append(f"{coord}:{resource}:{number}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def _play_match_sync(agent_a_spec, agent_b_spec, seed, swap_seats, run_dir,
                     capture_reasoning=False, balanced_dice=True,
                     max_turns=400, vps_to_win=10):
    """Run one full game synchronously and return a MatchResult.

    Seat assignment: by default agent A is passed first (RED-ward), B second.
    `swap_seats` swaps the list order so A and B trade opening seats while the
    seed keeps the board fixed. `capture_reasoning` turns on model reasoning for
    LLM agents (testing / viewable transcripts); off keeps runs cheap.
    `balanced_dice` deals dice from a per-seed deck decoupled from the global RNG
    (the dice half of fairness) so a mirrored pair sees IDENTICAL dice; turn it
    off to fall back to Catanatron's vanilla global-RNG dice.
    `vps_to_win` is the victory-point target (10, standard).
    `max_turns` caps the game; if neither side reaches `vps_to_win` by then the
    game is truncated and the winner is whoever has more VP (true tie -> draw).
    """
    from contextlib import nullcontext
    import catanatron.game as _catan_game
    from harness.dice import balanced_dice as _balanced_dice
    # Per-worker turn cap (module global, read at play() time; worker is isolated).
    _catan_game.TURNS_LIMIT = max_turns
    a = make_agent(agent_a_spec, SEATS[0], capture_reasoning=capture_reasoning)
    b = make_agent(agent_b_spec, SEATS[1], capture_reasoning=capture_reasoning)
    # Give each agent its seat color, then order the list to set who seats first.
    a.color, b.color = (SEATS[1], SEATS[0]) if swap_seats else (SEATS[0], SEATS[1])
    players = [b, a] if swap_seats else [a, b]

    seat_tag = "swap" if swap_seats else "norm"
    label = f"seed{seed}_{seat_tag}"
    agents_by_color = {a.color: a, b.color: b}
    transcript = TranscriptAccumulator(run_dir, label, agents_by_color)

    game = Game(players, seed=seed, vps_to_win=vps_to_win)
    fingerprint = _board_fingerprint(game)
    # Deal dice from a per-seed deck (decoupled from the global RNG) so the
    # mirrored pair sees identical dice; the deck records what it dealt so we can
    # fingerprint it the same way we fingerprint the board.
    dice_ctx = _balanced_dice(seed) if balanced_dice else nullcontext()
    with dice_ctx as deck:
        game.play(accumulators=[transcript])
    dice_fingerprint = deck.fingerprint() if deck is not None else ""
    dice_rolls = tuple(deck.dealt) if deck is not None else ()
    duration = transcript.duration

    color_a = a.color
    color_b = b.color
    vp_a = get_actual_victory_points(game.state, color_a)
    vp_b = get_actual_victory_points(game.state, color_b)
    winning_color = game.winning_color()
    truncated = False
    if winning_color is None:
        # Hit the turn cap without anyone reaching vps_to_win -> decide by VP.
        truncated = True
        if vp_a > vp_b:
            winning_color, winner = color_a, "A"
        elif vp_b > vp_a:
            winning_color, winner = color_b, "B"
        else:
            winner = None  # genuine tie on VP
    else:
        winner = "A" if winning_color == color_a else "B"

    run_dir = Path(run_dir)
    return MatchResult(
        seed=seed,
        label=label,
        agent_a=agent_a_spec,
        agent_b=agent_b_spec,
        seat_of_a=color_a,
        winner=winner,
        winner_color=winning_color.value if winning_color else None,
        vp_a=vp_a,
        vp_b=vp_b,
        num_turns=game.state.num_turns,
        duration=duration,
        truncated=truncated,
        board_fingerprint=fingerprint,
        dice_fingerprint=dice_fingerprint,
        dice_rolls=dice_rolls,
        json_path=str(run_dir / f"{label}.json"),
        log_path=str(run_dir / f"{label}.log"),
    )


async def run_match(agent_a_spec, agent_b_spec, seed, swap_seats=False,
                    run_dir="transcripts/adhoc", executor=None,
                    capture_reasoning=False, balanced_dice=True,
                    max_turns=400, vps_to_win=10):
    """Run a single match in a worker process (isolated global RNG).

    Pass a shared `executor` to run within a batch's process pool; otherwise a
    dedicated single-worker process is spun up just for this match.
    """
    loop = asyncio.get_running_loop()
    args = (_play_match_sync, agent_a_spec, agent_b_spec, seed, swap_seats,
            run_dir, capture_reasoning, balanced_dice, max_turns, vps_to_win)
    if executor is not None:
        return await loop.run_in_executor(executor, *args)
    with make_pool(max_workers=1) as ex:
        return await loop.run_in_executor(ex, *args)


async def run_mirror_pair(agent_a_spec, agent_b_spec, seed,
                          run_dir="transcripts/pair", capture_reasoning=False,
                          balanced_dice=True, max_turns=400, vps_to_win=10):
    """The fairness primitive: one board, two games with seats swapped.

    Game 1: A seats first (RED), B second (BLUE).
    Game 2: same seed -> same board, seats swapped (B first, A second).

    Returns (normal_match, swapped_match). Compare their `board_fingerprint`
    to confirm the board really was identical; compare winners to separate
    skill from seat luck.
    """
    with make_pool(max_workers=2) as ex:
        normal, swapped = await asyncio.gather(
            run_match(agent_a_spec, agent_b_spec, seed, False, run_dir,
                      executor=ex, capture_reasoning=capture_reasoning,
                      balanced_dice=balanced_dice, max_turns=max_turns,
                      vps_to_win=vps_to_win),
            run_match(agent_a_spec, agent_b_spec, seed, True, run_dir,
                      executor=ex, capture_reasoning=capture_reasoning,
                      balanced_dice=balanced_dice, max_turns=max_turns,
                      vps_to_win=vps_to_win),
        )
    return normal, swapped


async def run_batch(agent_a_spec, agent_b_spec, seeds: Sequence[int],
                    mirror: bool = True, concurrency: int = 8,
                    run_dir: str = "transcripts/batch",
                    capture_reasoning: bool = False,
                    balanced_dice: bool = True,
                    max_turns: int = 400, vps_to_win: int = 10) -> BatchResult:
    """Run many matches concurrently, bounded by `concurrency`.

    With `mirror=True` each seed is played twice (seats swapped). `concurrency`
    is the real parallelism knob: set it to your model's safe concurrent-call
    limit.
    """
    plan = []  # (seed, swap)
    for seed in seeds:
        plan.append((seed, False))
        if mirror:
            plan.append((seed, True))

    # The process pool's worker count IS the concurrency ceiling.
    with make_pool(max_workers=concurrency) as ex:
        jobs = [
            run_match(agent_a_spec, agent_b_spec, seed, swap, run_dir,
                      executor=ex, capture_reasoning=capture_reasoning,
                      balanced_dice=balanced_dice, max_turns=max_turns,
                      vps_to_win=vps_to_win)
            for seed, swap in plan
        ]
        matches = await asyncio.gather(*jobs)
    matches.sort(key=lambda m: (m.seed, m.seat_of_a.value if m.seat_of_a else ""))
    return BatchResult(
        agent_a=agent_a_spec,
        agent_b=agent_b_spec,
        matches=list(matches),
        num_pairs=len(seeds) if mirror else 0,
    )
