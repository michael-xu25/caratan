"""Opening-placement RL environment + training-data generation + eval.

Covers all four opening settlements in standard snake order A, B, B, A. For each
of the four decisions we drive a real Catanatron opening, enumerate the legal
remaining spots Catanatron offers, score each with the tunable championship
function (`placement_score`), and record a scenario whose `gold_action` is the
best-scoring remaining spot. Reward = how close the chosen spot is to the best
available AT THAT DECISION (so later placements are graded against the correctly
reduced option set).

v1: each placement scored independently (no cross-settlement complementarity).
Prior placements in the generated trajectory are the greedy-best spots, so every
decision is "given an optimal-so-far board, pick the best remaining."

The model is shown mechanics only (`goldilocks_eval.prompting.build_prompt`):
board, legal spots, per-spot production/pips/ports. It never sees the scores or
weights — those are reward-only.

Structured so the SECOND-settlement-specific logic (complementarity) can be added
later without changing the scenario schema.

CLI:
    # show the scoring for one board (sanity-check the weights — no model needed)
    python -m goldilocks_eval.placement_env show --seed 1000
    # generate training data from N example_pool boards
    python -m goldilocks_eval.placement_env generate --split example_pool --n 50 \
        --out data/placement_opening_train.jsonl
    # eval a model's four openings vs the optimum (held-out boards)
    python -m goldilocks_eval.placement_env eval --split grader_games --n 30 \
        --model fireworks:$FIREWORKS_MODEL
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catanatron import Color, Game, RandomPlayer
from catanatron.json import GameEncoder
from catanatron.models.enums import ActionType

from goldilocks_eval import prompting
from goldilocks_eval.placement_score import (
    WEIGHTS, placement_reward, regret, score_legal_spots,
)

SNAKE_ORDER = ["A", "B", "B", "A"]  # who is placing each opening settlement


def _seeds_for_split(split: str, n: int, index_path="dataset/initial/index.json") -> list[int]:
    idx = json.loads(Path(index_path).read_text())
    seeds = [b["seed"] for b in idx["boards"] if b["split"] == split]
    return seeds[:n] if n else seeds


def _serialize(game) -> dict:
    return json.loads(json.dumps(game, cls=GameEncoder))


def _settlement_actions(game):
    return [a for a in game.playable_actions
            if a.action_type == ActionType.BUILD_SETTLEMENT]


def generate_opening_scenarios(seed: int, weights=None, split="train") -> list[dict]:
    """Drive one board's four opening settlements (snake order), scoring each.

    At each settlement decision: score every legal spot, record the scenario with
    gold = best spot + all spot scores, then place the gold spot (greedy-optimal
    trajectory) and continue. Forced initial roads are auto-played."""
    weights = weights or WEIGHTS
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=seed)
    scenarios, pick, guard = [], 0, 0
    while pick < 4 and guard < 50:
        guard += 1
        settle = _settlement_actions(game)
        if not settle:
            game.execute(game.playable_actions[0])  # forced initial road, etc.
            continue
        legal = [a.value for a in settle]
        scored, best = score_legal_spots(game, legal, weights)
        scenarios.append({
            "scenario_id": f"{seed}_o{pick + 1}",
            "game_id": str(seed),
            "board_seed": seed,
            "env": "placement_opening",
            "placement_index": pick + 1,          # 1..4
            "snake_player": SNAKE_ORDER[pick],    # A/B/B/A (metadata; v1 ignores)
            "serialized_state": _serialize(game),
            "legal_actions": [prompting.node_id_str(n) for n in legal],
            "gold_action": prompting.node_id_str(best),
            "spot_scores": {
                prompting.node_id_str(n): {
                    "score": round(s, 4),
                    "pip_total": c["pip_total"],
                    "resource_diversity": round(c["resource_diversity"], 3),
                    "number_diversity": round(c["number_diversity"], 3),
                    "tiles": c["tiles"],
                } for n, (s, c) in scored.items()
            },
            "weights": dict(weights),
            "split": split,
        })
        game.execute(next(a for a in settle if a.value == best))  # place greedy-best
        pick += 1
    return scenarios


# ───────────────────────────── generation CLI ─────────────────────────────
def cmd_generate(args):
    weights = dict(WEIGHTS)
    seeds = _seeds_for_split(args.split, args.n)
    out = []
    for seed in seeds:
        out.extend(generate_opening_scenarios(seed, weights, split=args.split))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} scenarios ({len(seeds)} boards x 4 placements) -> {args.out}")
    print(f"weights: {weights}")


def cmd_traindata(args):
    """Convert generated scenarios -> reward-kit/TRL rows: a `prompt` (system+user
    messages, mechanics only) and `ground_truth` (flat node->score map + gold) the
    reward function reads. No scores ever appear in the prompt."""
    rows = []
    for line in open(args.in_):
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        scn = {"serialized_state": s["serialized_state"],
               "legal_actions": s["legal_actions"], "env": s["env"]}
        rows.append({
            "id": s["scenario_id"],
            "placement_index": s["placement_index"],
            "prompt": [
                {"role": "system", "content": prompting.SYSTEM},
                {"role": "user", "content": prompting.build_prompt(scn)},
            ],
            "ground_truth": {
                "spot_scores": {n: sc["score"] for n, sc in s["spot_scores"].items()},
                "gold": s["gold_action"],
            },
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} reward-kit training rows -> {args.out}")


def cmd_show(args):
    """Print the scoring for one board's four openings — to sanity-check weights."""
    scns = generate_opening_scenarios(args.seed, dict(WEIGHTS))
    print(f"board seed {args.seed}   weights={WEIGHTS}\n")
    for s in scns:
        ranked = sorted(s["spot_scores"].items(), key=lambda kv: -kv[1]["score"])
        print(f"--- opening {s['placement_index']} ({s['snake_player']}), "
              f"{len(ranked)} legal spots, GOLD={s['gold_action']} ---")
        for node, sc in ranked[:args.top]:
            tiles = ", ".join(f"{r}{n}" for r, n in sc["tiles"]) or "desert/none"
            star = "  <== gold" if node == s["gold_action"] else ""
            print(f"  {node:9} score={sc['score']:.3f}  pips={sc['pip_total']:<2} "
                  f"resdiv={sc['resource_diversity']:.2f} numdiv={sc['number_diversity']:.2f}"
                  f"  [{tiles}]{star}")
        worst = ranked[-1]
        print(f"  (worst legal: {worst[0]} score={worst[1]['score']:.3f})\n")


# ───────────────────────────────── eval ───────────────────────────────────
def _eval_one(backend, seed: int, weights: dict, mode: str) -> list[dict]:
    """Let the MODEL place all four openings (its own trajectory); score each
    pick against the best available at that decision."""
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=seed)
    out, pick, guard = [], 0, 0
    while pick < 4 and guard < 50:
        guard += 1
        settle = _settlement_actions(game)
        if not settle:
            game.execute(game.playable_actions[0])
            continue
        legal = [a.value for a in settle]
        scored, best = score_legal_spots(game, legal, weights)
        scn = {"serialized_state": _serialize(game),
               "legal_actions": [prompting.node_id_str(n) for n in legal],
               "env": "placement_opening"}
        chosen, status = None, "ok"
        try:
            text = backend.complete(prompting.SYSTEM, prompting.build_prompt(scn))
            ans = prompting.parse_answer(text)
            if ans is None:
                status = "unparseable"
            elif prompting.node_id_int(ans) not in scored:
                status = "illegal"
            else:
                chosen = prompting.node_id_int(ans)
        except Exception:
            status = "error"
        # A failed/invalid answer scores 0.0 (model did NOT pick a valid spot) and
        # is counted — NEVER silently treated as best/first (that fakes the score).
        # The trajectory advances on the optimal spot so later placements still run.
        reward = round(placement_reward(chosen, scored, mode), 4) if chosen is not None else 0.0
        order = sorted(scored, key=lambda n: -scored[n][0])
        rank = order.index(chosen) if chosen is not None else len(order)
        out.append({
            "placement": pick + 1,
            "chosen": prompting.node_id_str(chosen) if chosen is not None else None,
            "gold": prompting.node_id_str(best),
            "status": status,
            "reward": reward,
            "regret": round(regret(chosen, scored), 4),
            "rank": rank,
            "best_score": round(scored[best][0], 4),
        })
        game.execute(next(a for a in settle if a.value == (chosen if chosen is not None else best)))
        pick += 1
    return out


async def _eval_model(backend, seeds, weights, mode, concurrency):
    sem = asyncio.Semaphore(concurrency)

    async def one(seed):
        async with sem:
            return await asyncio.to_thread(_eval_one, backend, seed, weights, mode)
    return await asyncio.gather(*(one(s) for s in seeds))


def cmd_eval(args):
    from goldilocks_eval.agents.factory import make_backend
    weights = dict(WEIGHTS)
    seeds = _seeds_for_split(args.split, args.n)
    backend = make_backend(args.model)
    results = asyncio.run(_eval_model(backend, seeds, weights, args.reward, args.concurrency))
    # Discriminative metrics: top-1 (regret==0), top-3, mean regret — these
    # separate models (mean reward is too generous). Plus answer-validity so a
    # 404-storm / unparseable outputs can't masquerade as a real score.
    by_pos = {1: [], 2: [], 3: [], 4: []}   # (reward, regret, top1, top3) per decision
    status_counts = {}
    for game in results:
        for d in game:
            by_pos[d["placement"]].append(d)
            status_counts[d["status"]] = status_counts.get(d["status"], 0) + 1
    total = sum(status_counts.values()) or 1
    ok = status_counts.get("ok", 0)
    print(f"model: {args.model}   boards: {len(seeds)}   reward sharpness applied")
    print(f"valid-answer rate: {ok}/{total} = {ok/total:.1%}"
          + (f"   FAILURES: {dict((k, v) for k, v in status_counts.items() if k != 'ok')}"
             if ok < total else ""))
    if ok == 0:
        print("  ⚠️  0 valid answers — the model was never reached (check the deployment "
              "id). The numbers below are MEANINGLESS.")
    print(f"{'opening':<12}{'top1':>8}{'top3':>8}{'mean_regret':>13}{'mean_reward':>13}")
    for p in (1, 2, 3, 4):
        ds = by_pos[p]
        n = len(ds) or 1
        t1 = sum(d["regret"] <= 1e-9 for d in ds) / n
        t3 = sum(d["rank"] < 3 for d in ds) / n
        mreg = sum(d["regret"] for d in ds) / n
        mrew = sum(d["reward"] for d in ds) / n
        print(f"{p} ({SNAKE_ORDER[p-1]}){'':<6}{t1:>8.1%}{t3:>8.1%}{mreg:>13.3f}{mrew:>13.3f}")
    allds = [d for ds in by_pos.values() for d in ds]
    n = len(allds) or 1
    print(f"{'OVERALL':<12}{sum(d['regret']<=1e-9 for d in allds)/n:>8.1%}"
          f"{sum(d['rank']<3 for d in allds)/n:>8.1%}"
          f"{sum(d['regret'] for d in allds)/n:>13.3f}{sum(d['reward'] for d in allds)/n:>13.3f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--split", default="example_pool")
    g.add_argument("--n", type=int, default=50); g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_generate)
    t = sub.add_parser("traindata"); t.add_argument("--in", dest="in_", required=True)
    t.add_argument("--out", required=True); t.set_defaults(func=cmd_traindata)
    s = sub.add_parser("show"); s.add_argument("--seed", type=int, default=1000)
    s.add_argument("--top", type=int, default=6); s.set_defaults(func=cmd_show)
    e = sub.add_parser("eval"); e.add_argument("--split", default="grader_games")
    e.add_argument("--n", type=int, default=30); e.add_argument("--model", required=True)
    e.add_argument("--reward", default="normalized", choices=["normalized", "ratio", "rank"])
    e.add_argument("--concurrency", type=int, default=16); e.set_defaults(func=cmd_eval)
    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
