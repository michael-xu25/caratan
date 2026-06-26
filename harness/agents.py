"""Model-agnostic agent interface.

This is the load-bearing seam of the whole harness:

  * It is the **model-swap flag** -- swap Claude <-> Gemini <-> a bot by
    changing one string spec, no code changes.
  * It is the **reasoning capture point** -- every decision can stash a
    free-text rationale that the transcript logger picks up.
  * It is the **integration seam with Michael's data** -- the scorer drives
    an agent's `decide()` over held-out positions; same interface, same call.

The contract is exactly Catanatron's `Player`:

    decide(game, playable_actions) -> action   (one of playable_actions)

Everything else (async, logging, scoring) hangs off that one method.

Spec grammar
------------
A spec is `"<backend>"` or `"<backend>:<model>"`, e.g.:

    "value"                 # ValueFunctionPlayer bot
    "R"                     # RandomPlayer (CLI-style alias)
    "claude:claude-opus-4-8"
    "gemini:gemini-2.5-pro"

`make_agent(spec, color)` returns a ready `Player`.
"""

from __future__ import annotations

from typing import Optional

from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer


# --------------------------------------------------------------------------
# Reasoning capture
# --------------------------------------------------------------------------
class Agent(Player):
    """Base class for harness agents.

    Adds two things on top of Catanatron's `Player`:

      * `backend` / `model` metadata so logs and configs can identify the
        policy without inspecting the Python class.
      * a one-slot reasoning buffer. An agent calls `self._note(text)` inside
        `decide()`; the transcript accumulator calls `pop_reasoning()` right
        after the decision to attach it to the log line.
    """

    backend: str = "agent"

    def __init__(self, color: Color, model: Optional[str] = None, is_bot: bool = True):
        super().__init__(color, is_bot=is_bot)
        self.model = model
        self._reasoning: Optional[str] = None
        self._decision: Optional[dict] = None

    def _note(self, text: Optional[str]) -> None:
        self._reasoning = text

    def pop_reasoning(self) -> Optional[str]:
        """Return and clear the rationale for the most recent decision."""
        r, self._reasoning = self._reasoning, None
        return r

    def _capture(self, record: Optional[dict]) -> None:
        """Stash the enriched decision record for the transcript to pop."""
        self._decision = record

    def pop_decision(self) -> Optional[dict]:
        """Return and clear the enriched record for the most recent decision.

        None for forced/single-option moves (nothing to analyze)."""
        d, self._decision = self._decision, None
        return d

    @property
    def name(self) -> str:
        return self.model and f"{self.backend}:{self.model}" or self.backend

    def __repr__(self) -> str:
        return f"{self.name}:{self.color.value}"


# --------------------------------------------------------------------------
# Bot-backed agents (the stubs we develop against before LLMs exist)
# --------------------------------------------------------------------------
class BotAgent(Agent):
    """Wraps a built-in Catanatron bot as an Agent.

    Bots have no natural language reasoning, so `pop_reasoning()` yields a
    compact, machine description of the policy instead of None -- enough to
    keep the human log self-explanatory.
    """

    def __init__(self, color: Color, backend: str, bot_cls, **bot_kwargs):
        super().__init__(color, model=None, is_bot=True)
        self.backend = backend
        self._bot = bot_cls(color, **bot_kwargs)

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        action = self._bot.decide(game, actions)
        # Single-option decisions are forced; only record/annotate real choices.
        if len(actions) > 1:
            from goldilocks_eval.decision_record import build_decision_record
            note = f"[{self.backend}] chose among {len(actions)} options"
            self._note(note)
            self._capture(build_decision_record(
                game, self.color, actions, action, reasoning=note))
        return action

    def reset_state(self):
        self._bot.reset_state()


_BOT_BACKENDS = {
    "random": RandomPlayer,
    "weighted": WeightedRandomPlayer,
    "value": ValueFunctionPlayer,
    "alphabeta": AlphaBetaPlayer,
}

# CLI-style short codes -> canonical backend names.
_ALIASES = {
    "R": "random",
    "W": "weighted",
    "VP": "value",
    "AB": "alphabeta",
}


# --------------------------------------------------------------------------
# LLM-backed agent (uses a swappable backend from harness.backends)
# --------------------------------------------------------------------------
class LLMAgent(Agent):
    """Model-agnostic LLM player.

    Holds an `LLMBackend` (the only thing that branches per model). Each turn it
    renders the state to text, asks the backend for an action index, and returns
    the chosen legal action -- with two guarantees:

      * **Legality** -- any API error / out-of-range / unparseable reply falls
        back to a legal action (`fallback_action`), so a game never crashes.
      * **Reasoning mode** -- if `capture_reasoning` is on (testing / viewable
        transcripts) the prompt asks the model to state *why* and we record it;
        if off (training / production) the model returns only the index, which
        is cheaper and faster, and nothing is logged.
    """

    def __init__(self, color: Color, backend, model: Optional[str] = None,
                 capture_reasoning: bool = False):
        super().__init__(color, model=model, is_bot=True)
        self.backend_obj = backend          # an LLMBackend
        self.backend = backend.name.split(":")[0] if getattr(backend, "name", None) else "llm"
        self.capture_reasoning = capture_reasoning

    def decide(self, game, playable_actions):
        import time

        from harness import prompt as P
        from goldilocks_eval.decision_record import build_decision_record

        actions = list(playable_actions)
        if len(actions) <= 1:
            return actions[0]  # forced move, don't spend a model call

        opponent = next(c for c in game.state.colors if c != self.color)
        user = P.build_user_prompt(game, self.color, opponent, actions)
        system = P.system_prompt(self.capture_reasoning)

        start = time.time()
        reasoning = ""
        fell_back = False
        try:
            text = self.backend_obj.complete(system, user)
            index, reasoning = P.parse_choice(text, len(actions))
            if index is None:
                raise ValueError("model returned no valid action index")
            chosen = actions[index]
        except Exception as exc:  # noqa: BLE001 - any failure must stay in-game
            chosen = self.fallback_action(game, actions)
            fell_back = True
            reasoning = (f"[fallback: {type(exc).__name__}: {exc}] "
                         f"played {chosen.action_type.value}")
        latency_ms = int((time.time() - start) * 1000)

        # Enriched record ALWAYS (legal set + state is the analyzable signal,
        # independent of reasoning mode); reasoning only when captured.
        self._capture(build_decision_record(
            game, self.color, actions, chosen,
            reasoning=(reasoning if self.capture_reasoning else ""),
            fell_back=fell_back, latency_ms=latency_ms))
        if self.capture_reasoning:
            self._note(reasoning)
        return chosen

    def fallback_action(self, game, playable_actions):
        """Legal move to play when the model fails. Override for a smarter
        heuristic; default is the first legal action (deterministic)."""
        return playable_actions[0]


# What kind of construction each backend needs.
_LLM_BACKENDS = {"claude", "fireworks", "openai", "hud", "modal"}
AGENT_BACKENDS = {
    **{name: "bot" for name in _BOT_BACKENDS},
    **{name: "llm" for name in _LLM_BACKENDS},
}


def make_agent(spec: str, color: Color, capture_reasoning: bool = False) -> Agent:
    """Build an agent from a `<backend>` or `<backend>:<arg>` spec.

    This is THE config flag. Pass a different spec, get a different policy --
    bots and LLMs are constructed through the same call. `arg` is a model id for
    LLMs (e.g. `claude:claude-haiku-4-5`) or a depth for `alphabeta:3`.
    `capture_reasoning` gates model reasoning (on for testing, off for runs).
    """
    head, _, arg = spec.partition(":")
    head = _ALIASES.get(head.upper(), head).lower()  # resolve CLI codes
    arg = arg or None

    kind = AGENT_BACKENDS.get(head)
    if kind == "bot":
        kwargs = {}
        if head == "alphabeta" and arg:
            kwargs["depth"] = int(arg)
        return BotAgent(color, head, _BOT_BACKENDS[head], **kwargs)
    if kind == "llm":
        from harness.backends import make_backend
        backend = make_backend(spec)
        return LLMAgent(color, backend, model=arg, capture_reasoning=capture_reasoning)
    raise ValueError(
        f"Unknown agent backend '{head}' (from spec '{spec}'). "
        f"Known: {sorted(AGENT_BACKENDS)} plus aliases {sorted(_ALIASES)}."
    )
