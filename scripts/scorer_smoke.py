#!/usr/bin/env python
"""Smoke-test the PRIMARY metric (per-weakness scenario scorer) end to end.

No API key needed: we drive Michael's `goldilocks_eval` scorer with tiny stub
backends so we can assert the tiered reward + before/after delta deterministically.

Two parts:
  1. Load + validate the FROZEN fixtures through the harness seam — proves the
     shared scenario contract round-trips into the scorer unchanged.
  2. Label one real fixture (gold + acceptable) and score three stub models —
     oracle (picks gold), polite (picks an acceptable alt), wrong (illegal-of-
     intent) — asserting 1.0 / 0.5 / 0.0, then a before_after headline delta.

Run:  .venv/bin/python scripts/scorer_smoke.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

# Everything routes through the harness re-exports = the integration seam.
from harness.backends import LLMBackend
from harness.scenario import (
    Scenario, load_scenarios, score, evaluate, before_after,
)
from goldilocks_eval import schema, prompting

FIXTURES = "data/examples/placement_examples.jsonl"


class StubBackend(LLMBackend):
    """Returns a fixed <answer> so scoring is deterministic and key-free."""

    def __init__(self, name: str, answer_node: str):
        self.name = name
        self._answer = answer_node

    def complete(self, system: str, user: str) -> str:
        return f"<reasoning>stub {self.name}</reasoning><answer>{self._answer}</answer>"


def part1_load_and_validate() -> list[Scenario]:
    scenarios = load_scenarios(FIXTURES)
    errors = []
    for s in scenarios:
        errors += [f"{s.scenario_id}: {e}" for e in schema.validate(s.to_dict())]
    if errors:
        print("FAIL — fixtures do not validate:")
        for e in errors:
            print("  ", e)
        sys.exit(1)
    labeled = sum(s.is_labeled for s in scenarios)
    print(f"[1] loaded {len(scenarios)} fixtures, all valid "
          f"({labeled} labeled, {len(scenarios) - labeled} unlabeled).")
    return scenarios


def part2_score_pipeline(scenarios: list[Scenario]) -> None:
    raw = [json.loads(l) for l in Path(FIXTURES).read_text().splitlines() if l.strip()]
    rec = raw[0]
    legal = [prompting.node_id_str(a) for a in rec["legal_actions"]]
    gold, acceptable_alt, wrong = legal[0], legal[1], legal[-1]

    # Label a real fixture via Michael's write-back helper (validates legality).
    labeled = schema.Scenario.from_dict(
        schema.apply_label(rec, gold_action=gold, acceptable_actions=[acceptable_alt]))

    # Unit-level tiered reward (the scoring contract).
    assert score(gold, gold, [acceptable_alt]) == 1.0
    assert score(acceptable_alt, gold, [acceptable_alt]) == 0.5
    assert score(wrong, gold, [acceptable_alt]) == 0.0
    assert score(None, gold, [acceptable_alt]) == 0.0
    print(f"[2] tiered reward OK  (gold={gold} 1.0 / acc={acceptable_alt} 0.5 / "
          f"wrong={wrong} 0.0 / unparseable 0.0)")

    oracle = StubBackend("oracle", gold)
    polite = StubBackend("polite", acceptable_alt)
    bad = StubBackend("wrong", wrong)

    one = [labeled]
    r_oracle = asyncio.run(evaluate(oracle, one, "oracle"))
    r_polite = asyncio.run(evaluate(polite, one, "polite"))
    r_bad = asyncio.run(evaluate(bad, one, "wrong"))
    assert (r_oracle.accuracy, r_polite.accuracy, r_bad.accuracy) == (1.0, 0.5, 0.0)
    print(f"[3] evaluate() accuracy  oracle={r_oracle.accuracy} "
          f"polite={r_polite.accuracy} wrong={r_bad.accuracy}; "
          f"by_env(oracle)={r_oracle.by_env()}")

    ba = asyncio.run(before_after(bad, oracle, one))  # before=wrong, after=oracle
    delta = ba["after"].accuracy - ba["before"].accuracy
    print(f"[4] before_after headline: {ba['before'].accuracy:.0%} -> "
          f"{ba['after'].accuracy:.0%}  (Δ +{delta:.0%}) — the metric the demo reports.")
    print("\nPASS — scenario scorer integrates cleanly with the frozen schema.")


if __name__ == "__main__":
    scenarios = part1_load_and_validate()
    part2_score_pipeline(scenarios)
