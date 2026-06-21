"""Per-weakness scenario eval — the PRIMARY metric from the build spec.

Consumes the shared scenario contract (frozen decision points) as JSONL:

    {
      "scenario_id": "...", "game_id": "...", "board_seed": 12345,
      "env": "placement",
      "serialized_state": { ... },     # Catanatron GameEncoder JSON
      "legal_actions": [ ... ],        # playable_actions at this point
      "gold_action": "node_27",        # champion label (ground truth)
      "acceptable_actions": [ ... ],   # near-optimal alternatives
      "split": "train" | "heldout"
    }

Scoring is tiered (champion labels = ground truth):
    1.0 if chosen == gold_action
    0.5 if chosen in acceptable_actions
    0.0 otherwise

`evaluate()` scores one model spec over a set of scenarios; `before_after()`
runs base vs trained and reports the headline delta per env (e.g. 30% -> 78%).

Note: a frozen scenario is graded from its serialized state + legal actions —
no live `Game` is simulated. So only text-reading agents (LLM backends) can be
scored here; Catanatron baselines (value/alphabeta) need a live game object and
belong in the head-to-head runner instead.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from goldilocks_eval.agents.base import LLMBackend
from goldilocks_eval import prompting  # canonical shared contract
from goldilocks_eval.prompting import score  # re-export
from goldilocks_eval.schema import Scenario  # frozen canonical record


def load_scenarios(path: str, split: Optional[str] = None) -> List[Scenario]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = Scenario.from_dict(json.loads(line))
            if split is None or s.split == split:
                out.append(s)
    return out


# Scoring/prompt/parse all come from the canonical contract (prompting.py) so
# eval, calibration, and generation can never drift. `score` is re-exported above.


def _scenario_dict(s: Scenario) -> dict:
    return {"serialized_state": s.serialized_state, "legal_actions": s.legal_actions,
            "env": s.env}


@dataclass
class ScenarioResult:
    scenario_id: str
    env: str
    chosen: Any        # parsed node id ("node_27") or None marker
    gold: Any
    reward: float
    reasoning: str
    fell_back: bool    # True when the answer was unparseable


def _decide(backend: LLMBackend, s: Scenario) -> ScenarioResult:
    answer: Optional[str] = None
    reasoning = ""
    try:
        text = backend.complete(prompting.SYSTEM, prompting.build_prompt(_scenario_dict(s)))
        answer = prompting.parse_answer(text)
        reasoning = (text or "")[:500]
    except Exception as exc:
        reasoning = f"(backend error: {exc})"
    fell_back = answer is None
    return ScenarioResult(
        scenario_id=s.scenario_id, env=s.env,
        chosen=answer if answer is not None else "(unparseable)",
        gold=prompting.node_id_str(s.gold_action),
        reward=score(answer, s.gold_action, s.acceptable_actions),
        reasoning=reasoning, fell_back=fell_back,
    )


@dataclass
class EvalReport:
    label: str
    results: List[ScenarioResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return (sum(r.reward for r in self.results) / len(self.results)
                if self.results else 0.0)

    def by_env(self) -> Dict[str, float]:
        envs: Dict[str, List[float]] = {}
        for r in self.results:
            envs.setdefault(r.env, []).append(r.reward)
        return {e: sum(v) / len(v) for e, v in envs.items()}


async def evaluate(backend: LLMBackend, scenarios: List[Scenario], label: str,
                   concurrency: int = 8) -> EvalReport:
    sem = asyncio.Semaphore(concurrency)

    async def one(s: Scenario) -> ScenarioResult:
        async with sem:
            return await asyncio.to_thread(_decide, backend, s)

    results = await asyncio.gather(*(one(s) for s in scenarios))
    return EvalReport(label=label, results=list(results))


async def before_after(base: LLMBackend, trained: LLMBackend,
                       scenarios: List[Scenario], concurrency: int = 8
                       ) -> Dict[str, EvalReport]:
    """Run base ('before') and trained ('after') over the same held-out set."""
    before = await evaluate(base, scenarios, "before", concurrency)
    after = await evaluate(trained, scenarios, "after", concurrency)
    return {"before": before, "after": after}
