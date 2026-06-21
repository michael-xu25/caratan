"""Goldilocks x Catan eval harness.

The measurement half of the loop: play fair 1v1 games, log readable
transcripts, and score model decisions against the shared scenario contract.

Public API:
    make_agent(spec, color)   -> a Catanatron Player (bot or LLM-backed)
    make_backend(spec)        -> a bare LLMBackend (for the scenario scorer)
    run_match / run_batch     -> async, process-isolated, mirrored matches
    run_mirror_pair           -> the fairness primitive (same board, swapped seats)
    TranscriptAccumulator     -> JSON + human-readable per-game logs
    evaluate / before_after   -> per-weakness scenario scoring (primary metric)
"""

from harness.agents import make_agent, Agent, BotAgent, LLMAgent, AGENT_BACKENDS
from harness.backends import LLMBackend, ClaudeBackend, make_backend
from harness.runner import (
    run_match, run_batch, run_mirror_pair, MatchResult, BatchResult,
)
from harness.transcripts import TranscriptAccumulator
from harness.scenario import (
    Scenario, load_scenarios, score, evaluate, before_after,
    EvalReport, ScenarioResult,
)

__all__ = [
    "make_agent", "Agent", "BotAgent", "LLMAgent", "AGENT_BACKENDS",
    "LLMBackend", "ClaudeBackend", "make_backend",
    "run_match", "run_batch", "run_mirror_pair", "MatchResult", "BatchResult",
    "TranscriptAccumulator",
    "Scenario", "load_scenarios", "score", "evaluate", "before_after",
    "EvalReport", "ScenarioResult",
]
