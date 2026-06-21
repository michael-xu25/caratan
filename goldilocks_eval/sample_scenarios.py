"""STOPGAP sample scenario emitter — for testing the consumers + labeling today.

This is NOT the real generator. The real pooled generator (disjoint pool A/B by
seed range, all-4-pick snapshots, calibration integration) is the producer half,
specified in scenario-generation-spec.md. This emits a few real *unlabeled*
placement scenarios so the eval / labeling / calibration plumbing can be
exercised before that lands — and so champion labeling can start now.

Usage:
    python -m goldilocks_eval.sample_scenarios --start 0 --n 3 \
        --split heldout --out data/placement_unlabeled.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from catanatron import Color, Game
from catanatron.json import GameEncoder
from catanatron.models.enums import ActionType
from catanatron.players.value import ValueFunctionPlayer

from goldilocks_eval import schema


def scenarios_for_seed(seed: int, split: str) -> List[dict]:
    """Snapshot the (up to 4) opening settlement decisions of one seeded game.

    A real bot (Value) fills the opening so prior placements look like real play;
    its pick only advances the board and is NOT the label (gold stays null)."""
    game = Game([ValueFunctionPlayer(Color.RED), ValueFunctionPlayer(Color.BLUE)], seed=seed)
    out: List[dict] = []
    pick = 0
    guard = 0
    while pick < 4 and guard < 50:
        guard += 1
        pa = game.playable_actions
        if pa and all(a.action_type == ActionType.BUILD_SETTLEMENT for a in pa):
            pick += 1
            out.append(schema.new_unlabeled(
                scenario_id=f"{seed}_p{pick}",
                board_seed=seed,
                env="placement",
                serialized_state=json.loads(json.dumps(game, cls=GameEncoder)),
                legal_actions=[a.value for a in pa],
                split=split,
                pick_index=pick,
            ))
        game.play_tick()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="goldilocks_eval.sample_scenarios",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=0, help="first board seed")
    p.add_argument("--n", type=int, default=3, help="number of boards")
    p.add_argument("--split", default="heldout", choices=["train", "heldout"])
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    records: List[dict] = []
    for seed in range(args.start, args.start + args.n):
        records.extend(scenarios_for_seed(seed, args.split))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} unlabeled scenarios from {args.n} boards -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
