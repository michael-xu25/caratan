"""Per-weakness scenario eval — re-exported from Michael's goldilocks_eval.

This is the PRIMARY metric and the shared contract. Michael's
`goldilocks_eval.scenario` is the canonical implementation (scenario schema,
tiered scoring, evaluate/before_after); `harness/` re-exports it so there's one
source of truth. See build-spec-decisions.md for the locked JSONL schema.
"""
from __future__ import annotations

from goldilocks_eval.scenario import (  # noqa: F401
    Scenario,
    load_scenarios,
    score,
    ScenarioResult,
    EvalReport,
    evaluate,
    before_after,
)
