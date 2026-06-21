"""Under-building RL environment + training-data generation + eval.

Targets the measured UNDER-BUILDING weakness: the baseline built only 70
settlements + 4 cities across 10 games — it passes/hoards when it can and should
expand or upgrade. This env trains the model to make affordable, productive
builds.

Sibling of `maritime_env.py` (same harness, schema, prompt, reasoning-off):
  - Drive real Catanatron games from the dataset/initial seeds (same leak-free
    splits).
  - At each state where >= 1 build is affordable AND the turn can be ended
    (a real "build or pass" decision, post-roll, NOT a forced opening placement),
    record a scenario whose prompt is the UNCHANGED live-play surface — board,
    hands, full numbered legal menu, chosen by index. No build advice in the
    prompt; all judgment is in the reward (`build_score`).
  - Score every legal build (`build_score.score_build_option`) and store the
    per-option components on the scenario.

Because settlement/city decisions are a minority of build states, generation
OVERSAMPLES states that offer a settlement or city — the measured weakness.

Reasoning is OFF for training rollouts AND eval (prompt.SYSTEM_PROMPT_ACTION_ONLY).

CLI:
    python -m goldilocks_eval.build_env show --seed 1000
    python -m goldilocks_eval.build_env generate --split example_pool \
        --n 50 --per-game 6 --out data/build_train.jsonl
    python -m goldilocks_eval.build_env generate --split grader_games \
        --n 30 --per-game 6 --out data/build_eval.jsonl
    python -m goldilocks_eval.build_env eval --data data/build_eval.jsonl \
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
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer

from goldilocks_eval import prompt as P
from goldilocks_eval.build_score import (
    BUILD_TYPES, WEIGHTS, best_build_value, build_reward, classify,
    hoard_penalty, score_build_option,
)
from goldilocks_eval.decision_record import build_decision_record

_PRIORITY_KINDS = {"settlement", "city"}   # the measured weakness — oversample these


def _seeds_for_split(split: str, n: int, index_path="dataset/initial/index.json") -> list[int]:
    idx = json.loads(Path(index_path).read_text())
    seeds = [b["seed"] for b in idx["boards"] if b["split"] == split]
    return seeds[:n] if n else seeds


def _serialize(game) -> dict:
    return json.loads(json.dumps(game, cls=GameEncoder))


def _driver(spec: str):
    head, _, arg = spec.partition(":")
    if head == "value":
        return ValueFunctionPlayer
    if head == "alphabeta":
        depth = int(arg) if arg else 2
        return lambda color: AlphaBetaPlayer(color, depth=depth)
    if head == "random":
        return RandomPlayer
    raise ValueError(f"unknown driver {spec!r} (use value / alphabeta[:d] / random)")


def _subsample(items: list, k: int) -> list:
    if not k or len(items) <= k:
        return items
    stride = len(items) / k
    return [items[int(i * stride)] for i in range(k)]


def _subsample_priority(items: list, k: int) -> list:
    """Thin to <= k, but keep every settlement/city scenario first (the weakness),
    then fill the remaining slots evenly from road/dev scenarios."""
    if not k or len(items) <= k:
        return items
    prio = [s for s in items if s["_priority"]]
    rest = [s for s in items if not s["_priority"]]
    if len(prio) >= k:
        return _subsample(prio, k)
    return prio + _subsample(rest, k - len(prio))


def _is_build_or_pass(actions) -> bool:
    atypes = {a.action_type for a in actions}
    return bool(atypes & BUILD_TYPES) and ActionType.END_TURN in atypes


def _scenario_at(game, color, actions, seed, idx_in_game, split, weights) -> dict:
    """Build one scenario from the live state at a build-or-pass decision point."""
    opponent = next(c for c in game.state.colors if c != color)
    user_prompt = P.build_user_prompt(game, color, opponent, actions)

    rec = build_decision_record(game, color, actions, actions[0])
    for kdrop in ("chosen", "action_type", "reasoning", "fell_back", "latency_ms"):
        rec.pop(kdrop, None)

    build_options = {
        P.render_action(a): score_build_option(game, color, a)
        for a in actions if a.action_type in BUILD_TYPES
    }
    priority = any(c["kind"] in _PRIORITY_KINDS for c in build_options.values())
    rec.update({
        "scenario_id": f"{seed}_b{idx_in_game}",
        "game_id": str(seed),
        "board_seed": seed,
        "env": "build",
        "split": split,
        "serialized_state": _serialize(game),
        "user_prompt": user_prompt,                 # mechanics-only user message
        "output_mode": "action_only",               # -> P.SYSTEM_PROMPT_ACTION_ONLY
        "build_options": build_options,             # {rendered build -> components}
        "weights": dict(weights),
        "_priority": priority,                      # generation-only; drop on write
    })
    return rec


def generate_for_seed(seed, driver_spec="value", per_game=6, split="train",
                      weights=None) -> list[dict]:
    """Drive one game; record a (priority-subsampled) scenario at each build-or-pass state."""
    weights = weights or WEIGHTS
    Driver = _driver(driver_spec)
    players = [Driver(Color.RED), Driver(Color.BLUE)]
    game = Game(players, seed=seed)
    scenarios, guard, idx = [], 0, 0
    while game.winning_color() is None and guard < 4000:
        guard += 1
        actions = list(game.playable_actions)
        if _is_build_or_pass(actions):
            idx += 1
            color = game.state.current_color()
            scenarios.append(_scenario_at(game, color, actions, seed, idx, split, weights))
        cur = game.state.current_color()
        player = next(p for p in players if p.color == cur)
        game.execute(player.decide(game, actions))
    return _subsample_priority(scenarios, per_game)


def generate_scenarios(seeds, driver_spec, per_game, split, weights) -> list[dict]:
    out = []
    for seed in seeds:
        out.extend(generate_for_seed(seed, driver_spec, per_game, split, weights))
    for s in out:
        s.pop("_priority", None)   # generation-only flag, not part of the schema
    return out


# ───────────────────────── reward over a model choice ─────────────────────
def reward_for_choice(scenario: dict, idx, weights=None):
    """Map a chosen action index to (reward, built, kind). Unparseable/illegal
    index falls back to action 0 (matching the live LLMPlayer fallback)."""
    legal = scenario["legal_actions"]
    if idx is None or not (0 <= idx < len(legal)):
        idx = 0
    chosen = legal[idx]
    opts = scenario["build_options"]
    if chosen in opts:
        comp = opts[chosen]
        return build_reward(comp, weights), True, classify(comp)
    # passing / trading / any non-build: the hoard penalty, scaled by what was forgone.
    best = best_build_value(opts, weights)
    return hoard_penalty(best, weights), False, "pass"


# ─────────────────────── HUD/Tinker training-row export ───────────────────
# Lean, mechanics-only system prompt (NOT the from-scratch RULES_1V1 primer). NO
# build advice (judgment is in the reward — that's the whole point). The numbered
# legal menu + state live in the user prompt.
BUILD_LEAN_SYSTEM = (
    "You are playing 1-vs-1 Settlers of Catan. The current game state and a "
    "numbered list of every legal action (build a settlement/city/road, buy a "
    "development card, end the turn) are given below. Choose the single best "
    'action by its index. Respond with ONLY {"action": N} where N is the chosen '
    "index, and nothing else."
)


def cmd_traindata(args):
    """Generated scenarios -> HUD/Tinker rows: lean-prompt + ground_truth
    (legal_actions, per-build components, and the best affordable build value for
    the hoard penalty). The index grader combines these. No advice leaks."""
    rows = []
    for line in open(args.in_):
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        opts = s["build_options"]
        rows.append({
            "id": s["scenario_id"],
            "env": s["env"],
            "prompt": [
                {"role": "system", "content": BUILD_LEAN_SYSTEM},
                {"role": "user", "content": s["user_prompt"]},
            ],
            "ground_truth": {
                "legal_actions": s["legal_actions"],       # rendered, index order
                "build_options": opts,                     # rendered build -> components
                "best_value": best_build_value(opts, s["weights"]),  # for hoard penalty
                "weights": s["weights"],
            },
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} build HUD rows -> {args.out}")


# ───────────────────────────── generation CLI ─────────────────────────────
def cmd_generate(args):
    weights = dict(WEIGHTS)
    seeds = _seeds_for_split(args.split, args.n)
    out = generate_scenarios(seeds, args.driver, args.per_game, args.split, weights)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    n_builds = sum(len(r["build_options"]) for r in out)
    n_sc = sum(any(c["kind"] in _PRIORITY_KINDS for c in r["build_options"].values()) for r in out)
    print(f"wrote {len(out)} scenarios from {len(seeds)} boards "
          f"({n_builds} scored build options; {n_sc} offer a settlement/city) -> {args.out}")
    print(f"driver: {args.driver}   per-game cap: {args.per_game}   weights: {weights}")


def cmd_show(args):
    """Print each legal build + its value for one board (sanity-check weights)."""
    weights = dict(WEIGHTS)
    scns = generate_for_seed(args.seed, args.driver, args.top, "show", weights)
    print(f"board seed {args.seed}   driver={args.driver}   weights={weights}\n")
    for s in scns:
        hand = " ".join(f"{r[:2]}{n}" for r, n in s["state"]["hand"].items())
        print(f"--- {s['scenario_id']}  turn {s['turn']} [{s['phase']}] "
              f"{s['color']}  hand: {hand} ---")
        ranked = sorted(s["build_options"].items(),
                        key=lambda kv: -build_reward(kv[1], weights))
        for act, comp in ranked:
            r = build_reward(comp, weights)
            where = (f"node {comp['node']}" if comp["kind"] in ("settlement", "city")
                     else f"edge {comp['edge']} opens {comp['opens_node']}" if comp["kind"] == "road"
                     else "")
            print(f"  value={r:+.2f}  {comp['kind']:10} {where}")
        best = best_build_value(s["build_options"], weights)
        print(f"  (pass/hoard = {hoard_penalty(best, weights):+.2f}; "
              f"best affordable build = {best:+.2f})\n")


# ───────────────────────────────── eval ───────────────────────────────────
def _eval_one(backend, scenario, weights, system_prompt):
    """Ask the model for one decision on a fixed held-out state; score it."""
    try:
        text = backend.complete(system_prompt, scenario["user_prompt"])
        idx, _ = P.parse_choice(text, scenario["num_legal"])
    except Exception:
        idx = None
    reward, built, kind = reward_for_choice(scenario, idx, weights)
    return {"reward": reward, "built": built, "kind": kind,
            "best": best_build_value(scenario["build_options"], weights)}


async def _eval_model(backend, scenarios, weights, concurrency, system_prompt):
    sem = asyncio.Semaphore(concurrency)

    async def one(scn):
        async with sem:
            return await asyncio.to_thread(_eval_one, backend, scn, weights, system_prompt)
    return await asyncio.gather(*(one(s) for s in scenarios))


def _summarize(results, weights) -> dict:
    from goldilocks_eval.build_score import HOARD_OK
    n = len(results)
    built = [r for r in results if r["built"]]
    settle_city = [r for r in built if r["kind"] in _PRIORITY_KINDS]
    # states where a strong build was affordable but the model didn't build it
    strong = [r for r in results if r["best"] > HOARD_OK]
    hoarded = [r for r in strong if not r["built"]]
    return {
        "n": n,
        "build_rate": len(built) / n if n else 0.0,
        "settle_city_rate": len(settle_city) / n if n else 0.0,
        "build_quality": sum(r["reward"] for r in built) / len(built) if built else 0.0,
        "hoard_rate": len(hoarded) / len(strong) if strong else 0.0,
        "mean_reward": sum(r["reward"] for r in results) / n if n else 0.0,
        "n_built": len(built),
    }


def cmd_eval(args):
    from goldilocks_eval.agents.factory import make_backend
    weights = dict(WEIGHTS)
    if args.data:
        scenarios = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    else:
        seeds = _seeds_for_split(args.split, args.n)
        scenarios = generate_scenarios(seeds, args.driver, args.per_game, args.split, weights)

    system_prompt = P.SYSTEM_PROMPT if args.reasoning else P.SYSTEM_PROMPT_ACTION_ONLY
    mode = "reasoning" if args.reasoning else "action-only"
    print(f"{len(scenarios)} held-out build states   output: {mode}   "
          f"reward weights: {weights}\n")

    rows = []
    for spec in args.model:
        backend = make_backend(spec)
        results = asyncio.run(_eval_model(backend, scenarios, weights,
                                          args.concurrency, system_prompt))
        rows.append((spec, _summarize(results, weights)))

    hdr = (f"{'model':<46} {'build%':>8} {'settl/city%':>12} {'quality':>8} "
           f"{'hoard%':>8} {'mean_r':>8}")
    print(hdr)
    print("-" * len(hdr))
    for spec, m in rows:
        print(f"{spec[:46]:<46} {m['build_rate']*100:>7.1f}% {m['settle_city_rate']*100:>11.1f}% "
              f"{m['build_quality']:>8.3f} {m['hoard_rate']*100:>7.1f}% {m['mean_reward']:>8.3f}")
    print("\nhigher build% / settl-city% / quality + lower hoard% = builds more AND better.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--split", default="example_pool")
    g.add_argument("--n", type=int, default=50)
    g.add_argument("--per-game", type=int, default=6)
    g.add_argument("--driver", default="value")
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("show")
    s.add_argument("--seed", type=int, default=1000)
    s.add_argument("--driver", default="value")
    s.add_argument("--top", type=int, default=8)
    s.set_defaults(func=cmd_show)

    t = sub.add_parser("traindata")
    t.add_argument("--in", dest="in_", required=True)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_traindata)

    e = sub.add_parser("eval")
    e.add_argument("--data", default=None, help="generated scenarios jsonl (held-out)")
    e.add_argument("--split", default="grader_games")
    e.add_argument("--n", type=int, default=30)
    e.add_argument("--per-game", type=int, default=6)
    e.add_argument("--driver", default="value")
    e.add_argument("--model", action="append", required=True,
                   help="repeat to compare baseline vs trained on the same states")
    e.add_argument("--concurrency", type=int, default=16)
    e.add_argument("--reasoning", action="store_true",
                   help="use the reasoning-ON format (default OFF, matching training rollouts)")
    e.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
