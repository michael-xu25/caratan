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
from goldilocks_eval import prompt as P


@dataclass
class Scenario:
    scenario_id: str
    env: str
    serialized_state: Any
    legal_actions: List[Any]
    gold_action: Any
    acceptable_actions: List[Any] = field(default_factory=list)
    split: str = "heldout"
    game_id: str = ""
    board_seed: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scenario":
        return cls(
            scenario_id=d["scenario_id"],
            env=d.get("env", "unknown"),
            serialized_state=d.get("serialized_state"),
            legal_actions=list(d["legal_actions"]),
            gold_action=d["gold_action"],
            acceptable_actions=list(d.get("acceptable_actions", [])),
            split=d.get("split", "heldout"),
            game_id=d.get("game_id", ""),
            board_seed=d.get("board_seed"),
        )


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


def score(chosen: Any, gold: Any, acceptable: List[Any]) -> float:
    if chosen == gold:
        return 1.0
    if chosen in acceptable:
        return 0.5
    return 0.0


# --- prompting (decoder-free: present state text + numbered legal actions) ---

SCENARIO_SYSTEM = (
    "You are an expert Settlers of Catan player. Given a frozen decision point "
    "(the game state and a numbered list of legal actions), pick the single "
    "best action.\n\n"
    'Reply with ONLY one-line JSON: {"action": <index>, "reasoning": "<short>"}'
)


def build_scenario_prompt(s: Scenario) -> str:
    state_str = json.dumps(s.serialized_state, indent=2)
    if len(state_str) > 6000:  # keep prompts bounded for big GameEncoder dumps
        state_str = state_str[:6000] + "\n... (state truncated)"
    actions = "\n".join(f"  [{i}] {a}" for i, a in enumerate(s.legal_actions))
    return (
        f"Environment: {s.env}\n\n"
        f"State:\n{state_str}\n\n"
        f"Legal actions:\n{actions}\n\n"
        f"Respond with the JSON object."
    )


@dataclass
class ScenarioResult:
    scenario_id: str
    env: str
    chosen: Any
    gold: Any
    reward: float
    reasoning: str
    fell_back: bool


def _decide(backend: LLMBackend, s: Scenario) -> ScenarioResult:
    idx: Optional[int] = None
    reasoning = ""
    fell_back = False
    try:
        text = backend.complete(SCENARIO_SYSTEM, build_scenario_prompt(s))
        idx, reasoning = P.parse_choice(text, len(s.legal_actions))
    except Exception as exc:
        reasoning = f"(backend error: {exc})"
    if idx is None:
        idx = 0
        fell_back = True
    chosen = s.legal_actions[idx]
    return ScenarioResult(
        scenario_id=s.scenario_id, env=s.env, chosen=chosen, gold=s.gold_action,
        reward=score(chosen, s.gold_action, s.acceptable_actions),
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
