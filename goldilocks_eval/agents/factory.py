"""Build a Catanatron `Player` from a string spec (the config flag).

Baselines (no API key needed):
    random      -> RandomPlayer
    weighted    -> WeightedRandomPlayer
    value       -> ValueFunctionPlayer
    alphabeta   -> AlphaBetaPlayer (default depth 2)  |  alphabeta:3 for depth 3

LLM agents (model-agnostic; backend chosen by prefix):
    claude               -> Claude, default model (claude-opus-4-8)
    claude:<model-id>    -> Claude, specific model (e.g. claude:claude-haiku-4-5)
    gemini[:<model>]     -> stub (not yet wired) — shows the swap point
"""
from __future__ import annotations

from catanatron import Color, Player, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from goldilocks_eval.agents.base import LLMPlayer

BASELINES = {"random", "weighted", "value", "alphabeta"}


def make_player(spec: str, color: Color) -> Player:
    spec = spec.strip()
    head, _, arg = spec.partition(":")
    head = head.lower()

    if head == "random":
        return RandomPlayer(color)
    if head == "weighted":
        return WeightedRandomPlayer(color)
    if head == "value":
        return ValueFunctionPlayer(color)
    if head == "alphabeta":
        depth = int(arg) if arg else 2
        return AlphaBetaPlayer(color, depth=depth)

    if head == "claude":
        from goldilocks_eval.agents.claude_backend import ClaudeBackend, DEFAULT_MODEL
        return LLMPlayer(color, ClaudeBackend(model=arg or DEFAULT_MODEL))

    if head == "gemini":
        raise NotImplementedError(
            "Gemini backend is a stub. Implement LLMBackend.complete() in "
            "goldilocks_eval/agents/gemini_backend.py and wire it here — the "
            "LLMPlayer / runner / transcript code is backend-agnostic."
        )

    raise ValueError(
        f"Unknown agent spec: {spec!r}. "
        f"Use one of {sorted(BASELINES)} or claude[:model] / gemini[:model]."
    )


def label_for(spec: str) -> str:
    """Stable human-readable label used in metrics and transcripts."""
    return spec.strip()


def make_backend(spec: str):
    """Build a bare `LLMBackend` from a spec (for scenario eval, which has no
    live Game). Only LLM specs are valid here — baselines need a game object."""
    head, _, arg = spec.strip().partition(":")
    head = head.lower()
    if head == "claude":
        from goldilocks_eval.agents.claude_backend import ClaudeBackend, DEFAULT_MODEL
        return ClaudeBackend(model=arg or DEFAULT_MODEL)
    if head == "gemini":
        raise NotImplementedError("Gemini backend is a stub — see make_player().")
    raise ValueError(
        f"Scenario eval needs an LLM backend (claude[:model]); got {spec!r}. "
        f"Baselines like {sorted(BASELINES)} can only run in the head-to-head runner."
    )
