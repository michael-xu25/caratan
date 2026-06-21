"""Maritime-trade RL environment + training-data generation + eval.

Targets the measured OVER-TRADING weakness: the baseline made 504 maritime
trades across 10 games, churning resources pointlessly. This env trains the model
to trade only when a trade is *productive* (unlocks / advances a build).

Mirror of `placement_env.py`:
  - Drive real Catanatron games from the dataset/initial seeds (same splits, so
    train and held-out eval don't leak).
  - At every state where >= 1 MARITIME_TRADE is legal, record a scenario whose
    prompt is the UNCHANGED live-play surface (goldilocks_eval.prompt) — board,
    hands, full numbered legal menu (trades AND builds AND END_TURN), chosen by
    index. No trading advice in the prompt; all judgment is in the reward.
  - Score every legal trade option with the tunable championship function
    (`maritime_score`) and store the per-option components on the scenario.

The model decides WHETHER to trade and WHAT to trade by picking an index from the
same menu the over-trading was counted on. "Don't trade" is a real, reward-able
option (reward 0.0), and a churning trade scores below it.

CLI:
    # sanity-check the reward on one board (no model) — each legal trade + reward
    python -m goldilocks_eval.maritime_env show --seed 1000

    # generate training data (modest, configurable)
    python -m goldilocks_eval.maritime_env generate --split example_pool \
        --n 50 --per-game 6 --out data/maritime_trade_train.jsonl

    # eval baseline vs trained on held-out states (same scenarios for both)
    python -m goldilocks_eval.maritime_env generate --split grader_games \
        --n 30 --per-game 6 --out data/maritime_trade_eval.jsonl
    python -m goldilocks_eval.maritime_env eval --data data/maritime_trade_eval.jsonl \
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
from goldilocks_eval.decision_record import build_decision_record
from goldilocks_eval.maritime_score import (
    NO_TRADE_REWARD, WEIGHTS, classify, maritime_reward, score_trade_option,
)


def _seeds_for_split(split: str, n: int, index_path="dataset/initial/index.json") -> list[int]:
    idx = json.loads(Path(index_path).read_text())
    seeds = [b["seed"] for b in idx["boards"] if b["split"] == split]
    return seeds[:n] if n else seeds


def _serialize(game) -> dict:
    return json.loads(json.dumps(game, cls=GameEncoder))


def _driver(spec: str):
    """Bot that advances the game between snapshots (its moves are irrelevant —
    only the STATES it produces are recorded). Default 'value'."""
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
    """Evenly thin `items` down to <= k, keeping the spread across the game."""
    if not k or len(items) <= k:
        return items
    stride = len(items) / k
    return [items[int(i * stride)] for i in range(k)]


def _scenario_at(game, color, actions, seed, idx_in_game, split, weights) -> dict:
    """Build one scenario from the live state at a maritime-legal decision point."""
    opponent = next(c for c in game.state.colors if c != color)
    user_prompt = P.build_user_prompt(game, color, opponent, actions)

    # Reuse the shared enriched record for turn/phase/color/state/legal_actions,
    # then drop the per-decision fields (no action is chosen at generation time).
    rec = build_decision_record(game, color, actions, actions[0])
    for kdrop in ("chosen", "action_type", "reasoning", "fell_back", "latency_ms"):
        rec.pop(kdrop, None)

    trade_options = {
        P.render_action(a): score_trade_option(game, color, a, weights=weights)
        for a in actions if a.action_type == ActionType.MARITIME_TRADE
    }
    rec.update({
        "scenario_id": f"{seed}_m{idx_in_game}",
        "game_id": str(seed),
        "board_seed": seed,
        "env": "maritime_trade",
        "split": split,
        "serialized_state": _serialize(game),
        "user_prompt": user_prompt,                 # mechanics-only user message
        # Reasoning-OFF format for training rollouts AND eval (identical surface).
        # The GRPO harness pairs `user_prompt` with this system prompt.
        "output_mode": "action_only",               # -> P.SYSTEM_PROMPT_ACTION_ONLY
        "trade_options": trade_options,             # {rendered trade -> components}
        "no_trade_reward": NO_TRADE_REWARD,
        "weights": dict(weights),
    })
    return rec


def generate_for_seed(seed, driver_spec="value", per_game=6, split="train",
                      weights=None) -> list[dict]:
    """Drive one game; record a (subsampled) scenario at each maritime-legal state."""
    weights = weights or WEIGHTS
    Driver = _driver(driver_spec)
    players = [Driver(Color.RED), Driver(Color.BLUE)]
    game = Game(players, seed=seed)
    scenarios, guard, idx = [], 0, 0
    while game.winning_color() is None and guard < 4000:
        guard += 1
        actions = list(game.playable_actions)
        if any(a.action_type == ActionType.MARITIME_TRADE for a in actions):
            idx += 1
            color = game.state.current_color()
            scenarios.append(_scenario_at(game, color, actions, seed, idx, split, weights))
        cur = game.state.current_color()
        player = next(p for p in players if p.color == cur)
        game.execute(player.decide(game, actions))
    return _subsample(scenarios, per_game)


def generate_scenarios(seeds, driver_spec, per_game, split, weights) -> list[dict]:
    out = []
    for seed in seeds:
        out.extend(generate_for_seed(seed, driver_spec, per_game, split, weights))
    return out


# ───────────────────────── reward over a model choice ─────────────────────
def reward_for_choice(scenario: dict, idx, weights=None):
    """Map a chosen action index to (reward, traded, label). Unparseable/illegal
    index falls back to action 0 (matching the live LLMPlayer fallback)."""
    legal = scenario["legal_actions"]
    if idx is None or not (0 <= idx < len(legal)):
        idx = 0
    chosen = legal[idx]
    opts = scenario["trade_options"]
    if chosen in opts:
        comp = opts[chosen]
        return maritime_reward(comp, weights), True, classify(comp)
    return scenario.get("no_trade_reward", NO_TRADE_REWARD), False, "no_trade"


# ───────────────────────────── generation CLI ─────────────────────────────
def cmd_generate(args):
    weights = dict(WEIGHTS)
    seeds = _seeds_for_split(args.split, args.n)
    out = generate_scenarios(seeds, args.driver, args.per_game, args.split, weights)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    n_trades = sum(len(r["trade_options"]) for r in out)
    print(f"wrote {len(out)} scenarios from {len(seeds)} boards "
          f"({n_trades} scored trade options) -> {args.out}")
    print(f"driver: {args.driver}   per-game cap: {args.per_game}   weights: {weights}")


def cmd_show(args):
    """Print each legal trade + its reward for one board (sanity-check weights)."""
    weights = dict(WEIGHTS)
    scns = generate_for_seed(args.seed, args.driver, args.top, "show", weights)
    print(f"board seed {args.seed}   driver={args.driver}   weights={weights}\n")
    for s in scns:
        hand = " ".join(f"{r[:2]}{n}" for r, n in s["state"]["hand"].items())
        print(f"--- {s['scenario_id']}  turn {s['turn']} [{s['phase']}] "
              f"{s['color']}  hand: {hand} ---")
        ranked = sorted(s["trade_options"].items(),
                        key=lambda kv: -maritime_reward(kv[1], weights))
        for trade, comp in ranked:
            r = maritime_reward(comp, weights)
            tag = classify(comp)
            extra = f" enables={comp['enables']}" if comp["enables"] else ""
            sc = f" scarce={comp['gives_scarce']}" if comp["gives_scarce"] else ""
            give, recv = comp["give"], comp["receive"]
            print(f"  reward={r:+.2f}  {tag:11} give {comp['rate']}x{give:5} -> {recv:5}"
                  f"{extra}{sc}")
        print(f"  (not trading = {s['no_trade_reward']:+.2f})\n")


# ─────────────────────── HUD/Tinker training-row export ───────────────────
# Lean, mechanics-only system prompt (NOT the full RULES_1V1 primer — it bloats
# every rollout and the model already knows Catan). NO trade advice (judgment is
# in the reward). The numbered legal menu + state live in the user prompt.
MARITIME_LEAN_SYSTEM = (
    "You are playing 1-vs-1 Settlers of Catan. The current game state and a "
    "numbered list of every legal action (maritime trades, builds, buying a "
    "development card, ending the turn) are given below. Choose the single best "
    'action by its index. Respond with ONLY {"action": N} where N is the chosen '
    "index, and nothing else."
)


def cmd_traindata(args):
    """Generated scenarios -> HUD/Tinker rows: a `prompt` (lean system + the
    mechanics-only user message) and `ground_truth` (legal_actions, the per-trade
    components, and the no-trade reward) the index grader reads. No advice leaks."""
    rows = []
    for line in open(args.in_):
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        rows.append({
            "id": s["scenario_id"],
            "env": s["env"],
            "prompt": [
                {"role": "system", "content": MARITIME_LEAN_SYSTEM},
                {"role": "user", "content": s["user_prompt"]},
            ],
            "ground_truth": {
                "legal_actions": s["legal_actions"],       # rendered, index order
                "trade_options": s["trade_options"],       # rendered trade -> components
                "no_trade_reward": s.get("no_trade_reward", NO_TRADE_REWARD),
                "weights": s["weights"],
            },
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} maritime HUD rows -> {args.out}")


# ───────────────────────────────── eval ───────────────────────────────────
def _eval_one(backend, scenario, weights, system_prompt):
    """Ask the model for one decision on a fixed held-out state; score it. The
    same `system_prompt` (reasoning-off by default) is used for every model, so
    baseline and trained are compared on an identical output format."""
    try:
        text = backend.complete(system_prompt, scenario["user_prompt"])
        idx, _ = P.parse_choice(text, scenario["num_legal"])
    except Exception:
        idx = None
    reward, traded, label = reward_for_choice(scenario, idx, weights)
    return {"reward": reward, "traded": traded, "label": label}


async def _eval_model(backend, scenarios, weights, concurrency, system_prompt):
    sem = asyncio.Semaphore(concurrency)

    async def one(scn):
        async with sem:
            return await asyncio.to_thread(_eval_one, backend, scn, weights, system_prompt)
    return await asyncio.gather(*(one(s) for s in scenarios))


def _summarize(results) -> dict:
    n = len(results)
    trades = [r for r in results if r["traded"]]
    enabling = [r for r in trades if r["label"] in ("enabling", "progressing")]
    rewards = [r["reward"] for r in results]
    trade_rewards = [r["reward"] for r in trades]
    return {
        "n": n,
        "trade_rate": len(trades) / n if n else 0.0,
        "mean_reward": sum(rewards) / n if n else 0.0,
        "trade_productivity": sum(trade_rewards) / len(trades) if trades else 0.0,
        "enabling_trade_rate": len(enabling) / len(trades) if trades else 0.0,
        "n_trades": len(trades),
    }


def cmd_eval(args):
    from goldilocks_eval.agents.factory import make_backend
    weights = dict(WEIGHTS)
    if args.data:
        scenarios = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    else:
        seeds = _seeds_for_split(args.split, args.n)
        scenarios = generate_scenarios(seeds, args.driver, args.per_game, args.split, weights)
    # Reasoning-off by default — identical to the training rollout format.
    system_prompt = P.SYSTEM_PROMPT if args.reasoning else P.SYSTEM_PROMPT_ACTION_ONLY
    mode = "reasoning" if args.reasoning else "action-only"
    print(f"{len(scenarios)} held-out maritime states   output: {mode}   "
          f"reward weights: {weights}\n")

    rows = []
    for spec in args.model:
        backend = make_backend(spec)
        results = asyncio.run(_eval_model(backend, scenarios, weights,
                                          args.concurrency, system_prompt))
        rows.append((spec, _summarize(results)))

    hdr = f"{'model':<48} {'trade_rate':>10} {'productivity':>12} {'enabling%':>10} {'mean_r':>8}"
    print(hdr)
    print("-" * len(hdr))
    for spec, m in rows:
        print(f"{spec[:48]:<48} {m['trade_rate']*100:>9.1f}% {m['trade_productivity']:>12.3f} "
              f"{m['enabling_trade_rate']*100:>9.1f}% {m['mean_reward']:>8.3f}")
    print("\nlower trade_rate + higher productivity/enabling% = trades less AND better.")


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
    s.add_argument("--top", type=int, default=6)
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
