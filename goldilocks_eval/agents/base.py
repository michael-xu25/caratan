"""Model-agnostic agent interface + an LLM-backed Catanatron player.

The integration point with Catanatron is `Player.decide(game, playable_actions)`.
Baseline bots already subclass `Player`; here we add:

- `LLMBackend`: a tiny abstract interface over "given system+user text, return a
  completion string". Swap Claude / Gemini / a small local model behind it.
- `LLMPlayer`: a `Player` that renders the state, asks a backend to pick an
  action by index, and records each decision (with the model's reasoning) so the
  match runner can write a readable transcript.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from catanatron import Action, Color, Game, Player
from catanatron.state_functions import get_actual_victory_points

from goldilocks_eval import prompt as P


class LLMBackend(ABC):
    """Swappable model backend. Implementations must be thread-safe for use
    inside the runner's worker threads."""

    name: str = "llm"

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the model's text completion for the given prompts."""
        raise NotImplementedError


@dataclass
class Decision:
    turn: int
    my_vp: int             # VP context — makes "while behind/ahead" legible
    opp_vp: int
    chosen: str            # rendered chosen action
    options: List[str]     # the full legal set it chose among (not just a count)
    reasoning: str
    latency_ms: int
    fell_back: bool        # True if we couldn't use the model's choice

    @property
    def num_options(self) -> int:
        return len(self.options)


class LLMPlayer(Player):
    """A Catanatron player that delegates each decision to an `LLMBackend`."""

    def __init__(self, color: Color, backend: LLMBackend, is_bot: bool = True):
        super().__init__(color, is_bot)
        self.backend = backend
        self.decisions: List[Decision] = []

    def reset_state(self):
        self.decisions = []

    def decide(self, game: Game, playable_actions) -> Action:
        actions: List[Action] = list(playable_actions)
        # No real choice -> don't spend a model call.
        if len(actions) <= 1:
            if actions:
                return actions[0]
            return playable_actions[0]

        opponent = next(c for c in game.state.colors if c != self.color)
        user = P.build_user_prompt(game, self.color, opponent, actions)

        start = time.time()
        idx: Optional[int] = None
        reasoning = ""
        fell_back = False
        try:
            text = self.backend.complete(P.SYSTEM_PROMPT, user)
            idx, reasoning = P.parse_choice(text, len(actions))
        except Exception as exc:  # network/parse/etc. — never crash a game
            reasoning = f"(backend error: {exc})"
        if idx is None:
            idx = 0  # deterministic, safe fallback
            fell_back = True
        latency_ms = int((time.time() - start) * 1000)

        chosen = actions[idx]
        self.decisions.append(Decision(
            turn=game.state.num_turns,
            my_vp=get_actual_victory_points(game.state, self.color),
            opp_vp=get_actual_victory_points(game.state, opponent),
            chosen=P.render_action(chosen),
            options=[P.render_action(a) for a in actions],
            reasoning=reasoning,
            latency_ms=latency_ms,
            fell_back=fell_back,
        ))
        return chosen

    def __repr__(self):
        return f"LLMPlayer({self.color.value}, backend={self.backend.name})"
