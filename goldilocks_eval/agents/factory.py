"""Build a Catanatron `Player` from a string spec (the config flag).

Baselines (no API key needed):
    random      -> RandomPlayer
    weighted    -> WeightedRandomPlayer
    value       -> ValueFunctionPlayer
    alphabeta   -> AlphaBetaPlayer (default depth 2)  |  alphabeta:3 for depth 3

LLM agents (model-agnostic; backend chosen by prefix):
    claude               -> Claude, default model (claude-opus-4-8)
    claude:<model-id>    -> Claude, specific model (e.g. claude:claude-haiku-4-5)
    fireworks:<model-id> -> Fireworks-served model (the models we train)
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

    if head == "fireworks":
        from goldilocks_eval.agents.fireworks_backend import FireworksBackend, DEFAULT_MODEL
        return LLMPlayer(color, FireworksBackend(model=arg or DEFAULT_MODEL))

    if head == "openai":
        from goldilocks_eval.agents.openai_backend import OpenAIBackend, DEFAULT_MODEL
        return LLMPlayer(color, OpenAIBackend(model=arg or DEFAULT_MODEL))

    if head == "hud":
        from goldilocks_eval.agents.hud_backend import HudBackend
        return LLMPlayer(color, HudBackend(model=arg or None))

    if head == "modal":
        from goldilocks_eval.agents.modal_backend import ModalBackend, DEFAULT_MODEL
        return LLMPlayer(color, ModalBackend(model=arg or DEFAULT_MODEL))

    raise ValueError(
        f"Unknown agent spec: {spec!r}. Use one of {sorted(BASELINES)} or "
        f"claude / fireworks / openai / modal [:model]."
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
    if head == "fireworks":
        from goldilocks_eval.agents.fireworks_backend import FireworksBackend, DEFAULT_MODEL
        return FireworksBackend(model=arg or DEFAULT_MODEL)
    if head == "openai":
        from goldilocks_eval.agents.openai_backend import OpenAIBackend, DEFAULT_MODEL
        return OpenAIBackend(model=arg or DEFAULT_MODEL)
    if head == "hud":
        from goldilocks_eval.agents.hud_backend import HudBackend
        return HudBackend(model=arg or None)
    if head == "modal":
        from goldilocks_eval.agents.modal_backend import ModalBackend, DEFAULT_MODEL
        return ModalBackend(model=arg or DEFAULT_MODEL)
    raise ValueError(
        f"Scenario eval needs an LLM backend (claude / fireworks / openai / hud / modal [:model]); got {spec!r}. "
        f"Baselines like {sorted(BASELINES)} can only run in the head-to-head runner."
    )
