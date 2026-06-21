"""
calibration_harness.py — Goldilocks difficulty calibration.

Reads a LABELED scenario JSONL, samples the base model N times per scenario at
temperature (semaphore-bounded concurrency), scores each rollout with the tiered
reward, and:
  - fills `base_solve_rate` on every scenario
  - (train pool) keeps only scenarios in the Goldilocks band with real variance
  - (heldout pool) records base_solve_rate but keeps everything (--no-filter)

This is the pre-training difficulty filter. GRPO learns from within-group
disagreement, so a scenario the base always solves or always fails contributes
zero gradient — we drop those from the training pool.

WIRE-IN POINTS (two seams to your repo — marked WIRE below):
  1. Backend:  replace `_DemoBackend` with your goldilocks_eval LLMBackend.
  2. Prompt/parse/score: import the canonical ones from goldilocks_eval so
     calibration and eval use IDENTICAL prompting (see the spec's "shared
     contract" section). Local fallbacks are provided so this runs standalone.

Usage:
  # train pool (filter to Goldilocks)
  python calibration_harness.py data/placement_train.jsonl \
      --out data/placement_train.calibrated.jsonl \
      --model accounts/fireworks/models/gemma-4-e4b \
      --samples 8 --concurrency 100 --low 0.2 --high 0.5 --temperature 0.8

  # heldout pool (record base rate, keep all)
  python calibration_harness.py data/placement_heldout.jsonl \
      --out data/placement_heldout.calibrated.jsonl --no-filter
"""

import argparse
import asyncio
import json
import re
import statistics
from dataclasses import dataclass

# ── WIRE 2: canonical prompt/parse/score (one source of truth) ──────────────
# Import the SAME builder/parser/scorer the eval uses, so base_solve_rate here
# and the eval's before/after measure the identical thing. Falls back to inline
# copies only if goldilocks_eval isn't importable (keeps this file standalone).
try:
    from goldilocks_eval.prompting import build_prompt, parse_answer, score  # noqa: F401
except Exception:  # pragma: no cover - standalone fallback
    def build_prompt(scenario: dict) -> str:
        legal = ", ".join(str(a) for a in scenario["legal_actions"])
        return (
            "Choose the best opening settlement.\n\n"
            f"Board state (JSON):\n{json.dumps(scenario['serialized_state'])}\n\n"
            f"Legal settlement nodes: {legal}\n\n"
            "Reason, then answer exactly as:\n"
            "<reasoning>...</reasoning>\n<answer>node_ID</answer>"
        )

    _ANS = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.DOTALL | re.IGNORECASE)

    def parse_answer(text: str) -> str | None:
        m = _ANS.search(text or "")
        return m.group(1).strip() if m else None

    def score(answer: str | None, gold: str, acceptable: list[str]) -> float:
        if answer is None:
            return 0.0
        if answer == gold:
            return 1.0
        if answer in (acceptable or []):
            return 0.5
        return 0.0
# ────────────────────────────────────────────────────────────────────────────


# ── WIRE 1: replace with your goldilocks_eval LLMBackend ────────────────────
class _DemoBackend:
    """Stand-in so this file runs without the repo. Returns a random legal node."""
    def __init__(self, model: str):
        self.model = model

    async def generate(self, prompt: str, temperature: float) -> str:
        import random
        await asyncio.sleep(0.01)
        nodes = re.search(r"Legal settlement nodes: (.+)", prompt)
        pick = random.choice(nodes.group(1).split(", ")) if nodes else "node_0"
        return f"<reasoning>demo</reasoning>\n<answer>{pick}</answer>"
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class CalResult:
    solve_rate: float      # fraction of rollouts scoring 1.0 (gold)
    reward_std: float      # variance proxy across rollouts
    rewards: list[float]


async def calibrate_one(backend, scenario, samples, temperature, sem) -> CalResult:
    prompt = build_prompt(scenario)
    gold = scenario["gold_action"]
    acceptable = scenario.get("acceptable_actions", [])

    async def one_rollout():
        async with sem:
            text = await backend.generate(prompt, temperature)
        return score(parse_answer(text), gold, acceptable)

    rewards = await asyncio.gather(*[one_rollout() for _ in range(samples)])
    solve_rate = sum(1 for r in rewards if r == 1.0) / len(rewards)
    reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return CalResult(solve_rate, reward_std, list(rewards))


def in_goldilocks(res: CalResult, low: float, high: float) -> bool:
    # Real within-group variance is the mechanically correct GRPO filter;
    # the band is a heuristic on top of it.
    return res.reward_std > 0.0 and low <= res.solve_rate <= high


async def main_async(args):
    scenarios = [json.loads(l) for l in open(args.input) if l.strip()]
    if any(s.get("gold_action") is None for s in scenarios):
        raise SystemExit("Some scenarios are unlabeled (gold_action=null). "
                         "Label them before calibrating.")

    backend = _DemoBackend(args.model)  # WIRE 1
    sem = asyncio.Semaphore(args.concurrency)

    results = await asyncio.gather(*[
        calibrate_one(backend, s, args.samples, args.temperature, sem)
        for s in scenarios
    ])

    kept, dropped = [], 0
    for s, res in zip(scenarios, results):
        s["base_solve_rate"] = round(res.solve_rate, 4)
        if args.filter and not in_goldilocks(res, args.low, args.high):
            dropped += 1
            continue
        kept.append(s)

    with open(args.out, "w") as f:
        for s in kept:
            f.write(json.dumps(s) + "\n")

    band = f"[{args.low}, {args.high}]" if args.filter else "OFF (--no-filter)"
    print(f"scenarios in:        {len(scenarios)}")
    print(f"goldilocks band:     {band}")
    print(f"kept:                {len(kept)}")
    print(f"dropped (off-band):  {dropped}")
    if scenarios:
        rates = [round(r.solve_rate, 2) for r in results]
        print(f"base solve_rate dist: min {min(rates)} / "
              f"mean {round(sum(rates)/len(rates),2)} / max {max(rates)}")
    print(f"wrote: {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="labeled scenario JSONL")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="accounts/fireworks/models/gemma-4-e4b")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--low", type=float, default=0.2)
    p.add_argument("--high", type=float, default=0.5)
    f = p.add_mutually_exclusive_group()
    f.add_argument("--filter", dest="filter", action="store_true", default=True,
                   help="filter to Goldilocks band (default; use for train pool)")
    f.add_argument("--no-filter", dest="filter", action="store_false",
                   help="record base_solve_rate only, keep all (heldout pool)")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
