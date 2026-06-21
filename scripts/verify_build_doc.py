#!/usr/bin/env python
"""Acceptance test: verify the harness against eval-infra-build-doc.md.

Runs the real code path for each build-doc requirement and prints PASS/FAIL with
evidence. Bot-only by default (no API key); the model-agnostic interface is
checked structurally so it needs no network.

    .venv/bin/python scripts/verify_build_doc.py

NOTE: the runner fans games out over a spawn-based process pool, which re-imports
this module in each worker — so all executable code MUST live under the
`if __name__ == "__main__"` guard below (same reason harness.cli has one).
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, ok, evidence=""):
    _results.append(bool(ok))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  — {evidence}" if evidence else ""))


def main():
    from catanatron.game import Game
    from catanatron.models.player import Color
    from harness.agents import make_agent, AGENT_BACKENDS
    from harness.runner import run_match, run_batch, run_mirror_pair, SEATS
    from harness import determinism
    from harness.dice import BalancedDice
    from harness.scenario import evaluate, before_after, score
    from harness.backends import LLMBackend
    from goldilocks_eval import schema, prompting

    # ---- Start here ------------------------------------------------------
    g = Game([make_agent("value", Color.RED), make_agent("random", Color.BLUE)], seed=1)
    g.play()
    check("Catanatron installed + bot game runs", g.winning_color() is not None,
          f"winner={g.winning_color()}")
    check("Baselines present (Random/Weighted/Value/AlphaBeta)",
          all(make_agent(b, Color.RED) for b in ["random", "weighted", "value", "alphabeta"]),
          "+ aliases R/W/VP/AB")

    # ---- Match runner ----------------------------------------------------
    check("1v1 head-to-head (two seats only)", SEATS == (Color.RED, Color.BLUE),
          f"SEATS={[c.value for c in SEATS]}")
    check("Model-agnostic interface (swap by spec flag)",
          AGENT_BACKENDS.get("claude") == "llm" and AGENT_BACKENDS.get("fireworks") == "llm"
          and AGENT_BACKENDS.get("value") == "bot",
          "claude/fireworks=llm, value=bot; one make_agent()+decide() contract")

    m = asyncio.run(run_match("value", "weighted", 1, run_dir="transcripts/_verify"))
    doc = json.loads(Path(m.json_path).read_text())
    log_txt = Path(m.log_path).read_text()
    check("Transcripts: machine JSON via GameEncoder + decisions[]",
          "game" in doc and len(doc.get("decisions", [])) > 0,
          f"{len(doc['decisions'])} decisions, keys={sorted(doc)[:4]}")
    check("Transcripts: human log (board + outcome + reasoning)",
          "resource" in log_txt.lower() and "WINNER" in log_txt and "↳" in log_txt
          and any(d.get("reasoning") for d in doc["decisions"]),
          f"{Path(m.log_path).name}: board + winner banner + reasoning lines")

    batch = asyncio.run(run_batch("value", "random", [1, 2, 3], concurrency=4,
                                  run_dir="transcripts/_verify_batch"))
    check("Async runner + concurrency ceiling knob",
          "concurrency" in inspect.signature(run_batch).parameters
          and hasattr(determinism, "make_pool") and len(batch.matches) == 6,
          f"--concurrency param; spawn process pool; {len(batch.matches)} games / 3 seeds")

    # ---- Fairness --------------------------------------------------------
    a1 = asyncio.run(run_match("value", "weighted", 9, run_dir="transcripts/_rep_a"))
    a2 = asyncio.run(run_match("value", "weighted", 9, run_dir="transcripts/_rep_b"))
    check("Seeded board + dice, reproducible/replayable",
          a1.board_fingerprint == a2.board_fingerprint
          and a1.dice_fingerprint == a2.dice_fingerprint and a1.winner == a2.winner,
          f"board={a1.board_fingerprint} dice={a1.dice_fingerprint} stable across runs")

    normal, swapped = asyncio.run(run_mirror_pair("value", "weighted", 1,
                                                  run_dir="transcripts/_pair"))
    n = min(len(normal.dice_rolls), len(swapped.dice_rolls))
    check("Mirrored games (same board, swapped seats)",
          normal.board_fingerprint == swapped.board_fingerprint
          and normal.seat_of_a != swapped.seat_of_a,
          f"board IDENTICAL, A: {normal.seat_of_a.value}->{swapped.seat_of_a.value}")
    check("Balanced dice deck cancels dice luck (prefix identical)",
          normal.dice_rolls[:n] == swapped.dice_rolls[:n],
          f"first {n} rolls identical across the pair")

    _deck = BalancedDice(0)  # one deck; a full 36-roll cycle covers every combo
    sums = Counter(x + y for x, y in (_deck.roll() for _ in range(36)))
    expected = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    check("Balanced dice are colonist-style (exact distribution)", dict(sums) == expected,
          "36-cycle matches true 2..12 frequencies")
    check("Held-out eval seeds supported (schema split field)", "heldout" in schema.SPLITS,
          f"splits={schema.SPLITS}; overlap guard deferred until training seeds pinned")
    check("Volume knob (N mirrored pairs)",
          "seeds" in inspect.signature(run_batch).parameters,
          "--n / --seeds")

    # ---- Metrics ---------------------------------------------------------
    rec = [json.loads(l) for l in
           Path("data/examples/placement_examples.jsonl").read_text().splitlines() if l.strip()][0]
    legal = [prompting.node_id_str(a) for a in rec["legal_actions"]]
    labeled = schema.Scenario.from_dict(
        schema.apply_label(rec, gold_action=legal[0], acceptable_actions=[legal[1]]))

    class Stub(LLMBackend):
        def __init__(self, ans): self.name, self._a = "stub", ans
        def complete(self, s, u): return f"<answer>{self._a}</answer>"

    oracle = asyncio.run(evaluate(Stub(legal[0]), [labeled], "o"))
    wrong = asyncio.run(evaluate(Stub(legal[-1]), [labeled], "w"))
    tiered = (score(legal[0], legal[0], [legal[1]]),
              score(legal[1], legal[0], [legal[1]]),
              score(legal[-1], legal[0], [legal[1]]))
    check("PRIMARY metric: per-weakness accuracy vs ground truth",
          tiered == (1.0, 0.5, 0.0) and oracle.accuracy == 1.0 and wrong.accuracy == 0.0,
          f"tiered reward {tiered}; oracle acc={oracle.accuracy}")
    ba = asyncio.run(before_after(Stub(legal[-1]), Stub(legal[0]), [labeled]))
    check("PRIMARY metric: before/after delta (the headline)",
          ba["after"].accuracy - ba["before"].accuracy == 1.0,
          f"{ba['before'].accuracy:.0%} -> {ba['after'].accuracy:.0%}")
    check("SECONDARY metric: mirrored head-to-head win-rate",
          hasattr(batch, "win_rate_a"), f"A win rate={batch.win_rate_a:.0%}")

    n_pass = sum(_results)
    print(f"\n{'='*60}\n{n_pass}/{len(_results)} checks passed")
    return 0 if n_pass == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
