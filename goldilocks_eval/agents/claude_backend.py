"""Claude backend for the LLM agent (Anthropic SDK).

Reference implementation of `LLMBackend`. Other backends (Gemini, a small local
model) implement the same `complete(system, user) -> str` contract.

Model defaults to claude-opus-4-8. A Catan game is hundreds of decisions, so we
keep each call cheap: no extended thinking, small max_tokens. Flip `thinking=True`
(or point at a cheaper model via the spec, e.g. `claude:claude-haiku-4-5`) to
trade throughput for move quality.
"""
from __future__ import annotations

from goldilocks_eval.agents.base import LLMBackend

DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 512,
                 thinking: bool = False):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The Claude backend needs the anthropic SDK: pip install anthropic"
            ) from exc
        # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile) from env.
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.name = f"claude:{model}"

    def complete(self, system: str, user: str) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        resp = self._client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if b.type == "text")
