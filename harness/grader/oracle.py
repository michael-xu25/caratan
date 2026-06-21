"""Regret oracle — the objective, engine-grounded signal that GATES grading.

For each real decision we replay the recorded game to that ply, then score every
legal action with Catanatron's value function (the same `base_fn` the
ValueFunctionPlayer uses). Regret is how much value the chosen action gave up vs
the oracle's best legal action:

    regret = value(best legal action) - value(chosen action)     (>= 0)

This answers the most important question objectively, so the two LLM graders
never have to vote on "was this a mistake?" — they only categorize *why*. Decisions
with regret above the gate threshold go to the LLM graders.

Replay is deterministic (we execute the recorded ActionRecords with their stored
results), mirroring scripts/build_viewer_data.py — so it works on LLM games that
aren't seed-reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import catanatron.game as _cg
from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.enums import Action, ActionType, ActionRecord
from catanatron.players.value import base_fn, DEFAULT_WEIGHTS

from harness.grader.context import decision_type_of, derive_state_tags


def _value_fn():
    return base_fn(DEFAULT_WEIGHTS)


def _to_action(ra: list) -> Action:
    """Rebuild an Action from a serialized [color, type, value] record."""
    atype = ActionType(ra[1])
    v = ra[2]
    if v is None:
        value = None
    elif atype == ActionType.MOVE_ROBBER:                 # ((coord), victim|None)
        coord, victim = v
        value = (tuple(coord), Color[victim] if victim else None)
    elif isinstance(v, list):                             # edge / trade / dice / YoP
        value = tuple(v)
    else:
        value = v                                         # node id / resource / depth
    return Action(Color[ra[0]], atype, value)


def _norm(res):
    return tuple(res) if isinstance(res, list) else res


@dataclass
class DecisionRegret:
    ply: int
    turn: Optional[int]
    color: str
    action: list                      # the chosen [color, type, value]
    action_type: str
    decision_type: str                # placement | trade | build_spend (taxonomy)
    state_tags: list                  # frozen STATE_TAGS that hold at this decision
    num_legal: int
    value_chosen: float
    value_best: float
    regret: float                     # value_best - value_chosen, >= 0 (raw value-fn units)
    regret_vp: float                  # regret in VP-equivalents (regret / public_vps weight)
    oracle_best: list                 # [color, type, value] of the best legal action
    gated: bool = False               # regret >= threshold (set by gate())


def compute_regrets(transcript: dict) -> list[DecisionRegret]:
    """Replay a transcript and compute per-decision regret for every real choice."""
    g = transcript["game"]
    records = g["action_records"]
    seed = transcript.get("seed", 0) or 0
    value_fn = _value_fn()

    # Replay through the engine with recorded results (no RNG drift). Colors are
    # always [RED, BLUE] in our 1v1 transcripts; seed reproduces the board.
    _cg.TURNS_LIMIT = max(g.get("num_turns_cap", 400), len(records) + 5)
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
                seed=seed, vps_to_win=10)

    out: list[DecisionRegret] = []
    for i, rec in enumerate(records):
        ra = rec[0]
        chosen = _to_action(ra)
        legal = list(game.playable_actions)

        dtype = decision_type_of(ra[1])
        gradeable = (dtype is not None and len(legal) >= 2)
        if gradeable:
            color = chosen.color
            tags = derive_state_tags(game, color)
            scored: list[tuple[float, Action]] = []
            for a in legal:
                gc = game.copy()
                try:
                    gc.execute(a)
                except Exception:
                    continue  # skip an action the engine won't simulate cleanly
                scored.append((float(value_fn(gc, color)), a))
            if scored:
                value_best, best_action = max(scored, key=lambda t: t[0])
                # value of the chosen action: match it in the scored legal set
                value_chosen = next(
                    (v for v, a in scored if a.action_type == chosen.action_type
                     and a.value == chosen.value), None)
                if value_chosen is not None:
                    raw = max(0.0, value_best - value_chosen)
                    out.append(DecisionRegret(
                        ply=i, turn=None, color=ra[0], action=ra,
                        action_type=ra[1], decision_type=dtype, state_tags=tags,
                        num_legal=len(legal),
                        value_chosen=value_chosen, value_best=value_best,
                        regret=raw,
                        regret_vp=raw / DEFAULT_WEIGHTS["public_vps"],
                        oracle_best=[best_action.color.value,
                                     best_action.action_type.value,
                                     _encode_value(best_action.value)],
                    ))

        # advance the real game with the recorded result (deterministic). If a
        # recorded action can't be replayed (rare engine/result edge cases, e.g.
        # a maritime trade whose replayed hand diverged), stop here and return the
        # decisions gathered so far rather than crashing the whole run.
        try:
            game.execute(chosen, validate_action=False,
                         action_record=ActionRecord(action=chosen, result=_norm(rec[1])))
        except Exception:
            break

    # attach turn numbers from the transcript's decisions[] if present
    decisions = transcript.get("decisions", [])
    turn_by_ply = {d.get("i", k): d.get("turn") for k, d in enumerate(decisions)}
    for dr in out:
        dr.turn = turn_by_ply.get(dr.ply)
    return out


def _encode_value(value):
    """JSON-friendly rendering of an Action value (tuples -> lists)."""
    if isinstance(value, tuple):
        return [_encode_value(v) for v in value]
    if isinstance(value, Color):
        return value.value
    return value


def gate_threshold(regrets: list[DecisionRegret], percentile: float = 75.0,
                   floor: float = 1e-6) -> float:
    """Regret cutoff for sending a decision to the LLM graders.

    Default: the `percentile`-th percentile of positive regrets in this batch
    (the recommended ~75th). Decisions at/above it are gated in.
    """
    pos = sorted(r.regret for r in regrets if r.regret > floor)
    if not pos:
        return float("inf")
    k = max(0, min(len(pos) - 1, int(round((percentile / 100.0) * (len(pos) - 1)))))
    return pos[k]


def apply_gate(regrets: list[DecisionRegret], threshold: float) -> list[DecisionRegret]:
    for r in regrets:
        r.gated = r.regret >= threshold
    return regrets
