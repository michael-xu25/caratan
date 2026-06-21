"""Transcript logging: machine JSON + a human log built for skimming.

Two artifacts per game, written under a run directory:

  <run>/<label>.json   full machine state via Catanatron's GameEncoder,
                       plus a `decisions` array carrying captured reasoning.
  <run>/<label>.log    a human-readable, batch-skimmable rendering.

Design goal for the human log: when you've run dozens of games you want to
flip through them fast. So each file leads with a one-line verdict banner,
then a compact board, then decisions grouped by turn with running VP. Built
with `rich` and exported as plain (un-colored) text so it reads cleanly in any
editor, pager, or `cat`.

`render_summary_table` produces the batch index -- the first thing to read.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.box import SIMPLE_HEAVY, ROUNDED

from catanatron.game import GameAccumulator
from catanatron.json import GameEncoder
from catanatron.models.enums import ActionType
from catanatron.state_functions import get_actual_victory_points

from harness.agents import Agent


def _plain_console(width: int = 100) -> Console:
    """A rich Console that records output as plain text (no ANSI escapes)."""
    return Console(record=True, width=width, file=io.StringIO(),
                   no_color=True, highlight=False, soft_wrap=False)


# Actions that are routine bookkeeping; collapsed in the human log to keep
# the signal (settlements, robber, dev cards, trades) readable.
_QUIET_ACTIONS = {ActionType.ROLL, ActionType.END_TURN}


class TranscriptAccumulator(GameAccumulator):
    """Catanatron accumulator that records a single game to JSON + a human log.

    Pass one per game to `Game.play(accumulators=[...])`. `agents` maps each
    seated Color to its Agent so we can label seats and pull reasoning.
    """

    def __init__(self, run_dir, label: str, agents: dict, width: int = 100):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.label = label
        self.agents = agents  # {Color: Agent}
        self.width = width
        self.events: list[dict] = []
        self.duration: float = 0.0
        self._start = 0.0
        self._board_rows: list[tuple] = []

    # -- lifecycle ----------------------------------------------------------
    def before(self, game):
        self._start = time.perf_counter()
        self._board_rows = _board_rows(game)

    def step(self, game, action):
        state = game.state
        decider = state.current_player()
        reasoning = None
        if isinstance(decider, Agent):
            reasoning = decider.pop_reasoning()
        self.events.append({
            "ply": len(self.events) + 1,
            "turn": state.num_turns,
            "color": action.color.value,
            "seat_agent": self._agent_name(action.color),
            "action_type": action.action_type.value,
            "value": _stringify(action.value),
            "reasoning": reasoning,
        })

    def after(self, game):
        self.duration = time.perf_counter() - self._start
        self._write_json(game, self.duration)
        self._write_human(game, self.duration)

    # -- helpers ------------------------------------------------------------
    def _agent_name(self, color) -> str:
        agent = self.agents.get(color)
        return agent.name if isinstance(agent, Agent) else str(color.value)

    def _write_json(self, game, duration: float):
        payload = {
            "label": self.label,
            "seed": getattr(game, "seed", None),
            "duration_seconds": round(duration, 4),
            "seats": {c.value: self._agent_name(c) for c in game.state.colors},
            "winning_color": game.winning_color().value if game.winning_color() else None,
            "final_victory_points": {
                c.value: get_actual_victory_points(game.state, c)
                for c in game.state.colors
            },
            "game": game,            # serialized by GameEncoder
            "decisions": self.events,
        }
        path = self.run_dir / f"{self.label}.json"
        path.write_text(json.dumps(payload, cls=GameEncoder, indent=2))
        return path

    def _write_human(self, game, duration: float):
        c = _plain_console(self.width)
        winner = game.winning_color()
        vps = {col: get_actual_victory_points(game.state, col) for col in game.state.colors}

        # --- verdict banner -------------------------------------------------
        seats = " vs ".join(
            f"{col.value}={self._agent_name(col)}" for col in game.state.colors
        )
        if winner:
            verdict = f"WINNER {winner.value} ({self._agent_name(winner)}) — {vps[winner]} VP"
        else:
            verdict = "NO WINNER (turn limit reached)"
        c.rule(f"[bold]GAME {self.label}")
        c.print(f"seats : {seats}")
        c.print(f"result: {verdict}")
        c.print(f"meta  : {game.state.num_turns} turns · {duration:.3f}s · "
                f"seed={getattr(game, 'seed', '?')} · "
                + " / ".join(f"{col.value} {vps[col]}VP" for col in game.state.colors))

        # --- board ----------------------------------------------------------
        c.print()
        board = Table(title="Board", box=ROUNDED, title_justify="left",
                      show_edge=True, pad_edge=False)
        board.add_column("resource", style="bold")
        board.add_column("numbers (probability order)")
        for resource, numbers in self._board_rows:
            board.add_row(resource, numbers)
        c.print(board)

        # --- decisions, grouped by turn ------------------------------------
        c.print()
        c.print("[bold]Decisions[/bold]  (routine rolls/end-turns collapsed)")
        last_turn = None
        for e in self.events:
            if e["action_type"] in {a.value for a in _QUIET_ACTIONS}:
                continue
            if e["turn"] != last_turn:
                last_turn = e["turn"]
                c.print(f"\n[dim]── turn {e['turn']} ──[/dim]")
            line = (f"  #{e['ply']:<4} {e['color']:<5} "
                    f"{e['action_type']:<22} {e['value']}")
            # markup=False: action values and reasoning are dynamic and may
            # contain '[...]' which rich would otherwise eat as style markup.
            c.print(line, markup=False)
            if e["reasoning"]:
                reasoning = " ".join(str(e["reasoning"]).split())  # flatten newlines
                c.print(f"        ↳ {reasoning}", markup=False)

        path = self.run_dir / f"{self.label}.log"
        path.write_text(c.export_text())
        return path


# --------------------------------------------------------------------------
# Batch index
# --------------------------------------------------------------------------
def render_summary_table(batch_result, width: int = 100) -> str:
    """Render a batch as a single skim-first table + headline win rate."""
    c = _plain_console(width)
    c.rule("[bold]BATCH SUMMARY")
    c.print(f"A = {batch_result.agent_a}    B = {batch_result.agent_b}")
    c.print(f"games: {len(batch_result.matches)}   "
            f"mirrored_pairs: {batch_result.num_pairs}")
    c.print(f"[bold]A win rate: {batch_result.win_rate_a:.1%}[/bold]  "
            f"(A {batch_result.wins_a} / B {batch_result.wins_b} / "
            f"draws {batch_result.draws})")
    c.print()

    t = Table(box=SIMPLE_HEAVY, show_edge=False)
    for col in ("seed", "A seat", "winner", "VP A", "VP B", "turns", "secs"):
        t.add_column(col)
    for m in batch_result.matches:
        t.add_row(
            str(m.seed), m.seat_of_a.value if m.seat_of_a else "?",
            m.winner or "draw", str(m.vp_a), str(m.vp_b),
            str(m.num_turns), f"{m.duration:.3f}",
        )
    c.print(t)
    return c.export_text()


def render_pair_report(normal, swapped, width: int = 100) -> str:
    """Fairness view of one mirrored pair: same board, swapped seats, verdict."""
    c = _plain_console(width)
    c.rule("[bold]FAIRNESS PAIR (mirrored)")
    c.print(f"A = {normal.agent_a}    B = {normal.agent_b}    seed = {normal.seed}")

    same_board = normal.board_fingerprint == swapped.board_fingerprint
    flag = "IDENTICAL" if same_board else "DIFFERENT (!)"
    c.print(f"board: {flag}  (fp {normal.board_fingerprint} vs {swapped.board_fingerprint})")
    if normal.dice_rolls or swapped.dice_rolls:
        # The two games have different lengths (different policies per seat play
        # different numbers of turns), so the FULL dice sequences differ. The
        # fairness property is that roll #N is identical in both — i.e. the
        # shared prefix matches. That's what cancels dice luck across the pair.
        n = min(len(normal.dice_rolls), len(swapped.dice_rolls))
        prefix_match = normal.dice_rolls[:n] == swapped.dice_rolls[:n]
        dflag = "IDENTICAL" if prefix_match else "DIFFERENT (!)"
        c.print(f"dice:  first {n} rolls {dflag}  (shared deck → roll N same in both seatings)")
    c.print()

    t = Table(box=SIMPLE_HEAVY, show_edge=False)
    for col in ("game", "A seat", "B seat", "winner", "VP A", "VP B", "turns"):
        t.add_column(col)
    for tag, m in (("normal", normal), ("swapped", swapped)):
        b_seat = "BLUE" if m.seat_of_a and m.seat_of_a.value == "RED" else "RED"
        t.add_row(tag, m.seat_of_a.value if m.seat_of_a else "?", b_seat,
                  f"{m.winner} ({m.winner_color})" if m.winner else "draw",
                  str(m.vp_a), str(m.vp_b), str(m.num_turns))
    c.print(t)
    c.print()

    if normal.truncated or swapped.truncated:
        c.print("[dim]note: turn cap reached — winner decided by victory-point lead[/dim]")

    # Verdict: did the same agent win regardless of seat?
    winners = {normal.winner, swapped.winner}
    if normal.winner is None or swapped.winner is None:
        verdict = "INCONCLUSIVE — at least one game was a true VP tie at the cap."
    elif winners == {"A"}:
        verdict = "A won BOTH seats → real edge for A (seat-independent)."
    elif winners == {"B"}:
        verdict = "B won BOTH seats → real edge for B (seat-independent)."
    else:
        verdict = ("SPLIT 1-1 → seat decided both games, not skill. "
                   "This pair contributes no skill signal; need more pairs.")
    c.print(f"[bold]verdict:[/bold] {verdict}")
    return c.export_text()


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------
def _stringify(value) -> str:
    if value is None:
        return ""
    return str(value)


def _board_rows(game):
    """Group land tiles by resource -> their dice numbers, for a compact board.

    Numbers shown high-probability first (6,8 ... 2,12) so the strong tiles
    jump out when skimming.
    """
    from collections import defaultdict

    prob = {6: 5, 8: 5, 5: 4, 9: 4, 4: 3, 10: 3, 3: 2, 11: 2, 2: 1, 12: 1}
    by_resource: dict = defaultdict(list)
    for tile in game.state.board.map.tiles.values():
        resource = getattr(tile, "resource", None)
        number = getattr(tile, "number", None)
        if number is None:
            continue
        name = getattr(resource, "value", resource) or "DESERT"
        by_resource[name].append(number)

    rows = []
    for resource, numbers in by_resource.items():
        numbers.sort(key=lambda n: (-prob.get(n, 0), n))
        rows.append((str(resource), "  ".join(str(n) for n in numbers)))
    rows.sort()
    return rows
