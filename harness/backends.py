"""Swappable LLM backends — re-exported from Michael's goldilocks_eval.

A backend is just `complete(system, user) -> str`. This module is a thin
re-export of Michael's canonical implementation so `harness/` uses his code
directly (Claude backend, factory) rather than duplicating it.
"""
from __future__ import annotations

# Single source of truth: Michael's backend interface + Claude implementation.
from goldilocks_eval.agents.base import LLMBackend  # noqa: F401
from goldilocks_eval.agents.claude_backend import (  # noqa: F401
    ClaudeBackend,
    DEFAULT_MODEL as DEFAULT_CLAUDE_MODEL,
)
from goldilocks_eval.agents.factory import make_backend  # noqa: F401

# Backends his factory understands (claude/fireworks/openai wired; gemini is his stub).
LLM_BACKENDS = {"claude", "fireworks", "openai"}
