#!/usr/bin/env python
"""Per-model game-quality stats from head-to-head matchup transcripts.

Aggregates, per MODEL (trained grpo / placement-only / untrained base), averaged
over every game that model played across the matchup run dirs:

  * wasted_turn_rate   -- fraction of the model's post-roll turns with NO productive
                          action (build/buy/trade/play-dev). Resource management:
                          lower = fewer turns that "do nothing".
  * resource_gain      -- total resources flowing INTO the model's hand over the game
                          (sum of positive hand-size deltas). Proxy for production,
                          driven by placement + resource management.
  * settlements / cities / roads / dev_buys / trades per game (build/economy volume).
  * pair_sweep_rate    -- of the seeds played both ways (mirror), how often this model
                          won BOTH orientations. Adaptability: winning regardless of
                          seat/board side.

Writes viewer/data/gamestats.json (consumed by the matchups dashboard). With no
args it auto-discovers the matchup dirs; or pass dirs explicitly.

    python scripts/gamestats.py
    python scripts/gamestats.py transcripts/hud-grpo-vs-base transcripts/diag-grpo-vs-base
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_HINT = "Qwen3-8B"
RES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
PRODUCTIVE = {"BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "BUY_DEVELOPMENT_CARD",
              "MARITIME_TRADE", "PLAY_KNIGHT_CARD", "PLAY_MONOPOLY",
              "PLAY_YEAR_OF_PLENTY", "PLAY_ROAD_BUILDING"}

ROLE = {"catan-grpo-q8b": "trained (full chain)",
        "catan-placement-only": "trained (placement only)",
        "Qwen/Qwen3-8B": "untrained base"}


def _mshort(name: str) -> str:
    return str(name).split(":")[-1]


def _transcripts(d: Path):
    return [Path(p) for p in sorted(glob.glob(str(d / "*.json")))
            if not (p.endswith(".view.json") or p.endswith(".grading.json")
                    or p.endswith("winrate.json"))]


def _per_color_game(g: dict):
    """Counts from action_records, per color: builds, trades, turns, wasted turns."""
    ar = g["action_records"]
    builds = defaultdict(lambda: defaultdict(int))   # color -> action_type -> n
    turns = defaultdict(int)
    wasted = defaultdict(int)
    prod_since_roll = defaultdict(bool)
    for rec in ar:
        color, atype = rec[0][0], rec[0][1]
        if atype == "ROLL":
            prod_since_roll[color] = False
        elif atype in PRODUCTIVE:
            builds[color][atype] += 1
            prod_since_roll[color] = True
        elif atype == "END_TURN":
            turns[color] += 1
            if not prod_since_roll[color]:
                wasted[color] += 1
            prod_since_roll[color] = False
    return builds, turns, wasted


def _resource_gain(view_path: Path):
    """Sum of positive hand-size deltas per color across the replay (resource inflow)."""
    if not view_path.exists():
        return {}
    steps = json.loads(view_path.read_text()).get("steps", [])
    gain = defaultdict(int)
    last = {}
    for st in steps:
        for c, h in (st.get("hands") or {}).items():
            tot = sum(int(h.get(r, 0)) for r in RES)
            if c in last and tot > last[c]:
                gain[c] += tot - last[c]
            last[c] = tot
    return gain


def aggregate(dirs):
    acc = defaultdict(lambda: defaultdict(float))   # model -> metric -> sum
    games = defaultdict(int)
    # pair-sweep bookkeeping: (matchup, seed) -> {variant: winner_model}
    pairs = defaultdict(dict)
    pair_models = defaultdict(set)                    # (matchup, seed) -> models in it

    for d in dirs:
        d = Path(d)
        for p in _transcripts(d):
            try:
                t = json.loads(p.read_text())
            except Exception:
                continue
            if "seats" not in t or "game" not in t:
                continue
            seats = t["seats"]                          # color -> agent spec
            model_of = {c: _mshort(n) for c, n in seats.items()}
            builds, turns, wasted = _per_color_game(t["game"])
            gain = _resource_gain(p.with_suffix("").with_suffix(".view.json"))
            for c, model in model_of.items():
                games[model] += 1
                b = builds[c]
                acc[model]["settlements"] += b.get("BUILD_SETTLEMENT", 0)
                acc[model]["cities"] += b.get("BUILD_CITY", 0)
                acc[model]["roads"] += b.get("BUILD_ROAD", 0)
                acc[model]["dev_buys"] += b.get("BUY_DEVELOPMENT_CARD", 0)
                acc[model]["trades"] += b.get("MARITIME_TRADE", 0)
                acc[model]["turns"] += turns[c]
                acc[model]["wasted"] += wasted[c]
                acc[model]["resource_gain"] += gain.get(c, 0)
            # pair bookkeeping
            seed = t.get("seed")
            variant = "swap" if "swap" in p.name else "norm"
            key = (d.name, seed)
            wc = t.get("winning_color")
            pairs[key][variant] = model_of.get(wc) if wc else None
            pair_models[key].update(model_of.values())

    # pair sweeps per model
    swept = defaultdict(int)
    pair_total = defaultdict(int)
    for key, res in pairs.items():
        if set(res) != {"norm", "swap"}:
            continue                                    # incomplete pair
        n, s = res["norm"], res["swap"]
        if n is None or s is None:
            continue                                    # a draw -> not a clean pair
        for m in pair_models[key]:
            pair_total[m] += 1
            if n == m and s == m:
                swept[m] += 1

    out = {}
    for model, gn in games.items():
        a = acc[model]
        turns_n = a["turns"] or 1
        out[model] = {
            "role": ROLE.get(model, model),
            "trained": model != "Qwen/Qwen3-8B",
            "games": gn,
            "wasted_turn_rate": round(a["wasted"] / turns_n, 3),
            "resource_gain_per_game": round(a["resource_gain"] / gn, 1),
            "settlements_per_game": round(a["settlements"] / gn, 2),
            "cities_per_game": round(a["cities"] / gn, 2),
            "roads_per_game": round(a["roads"] / gn, 2),
            "dev_buys_per_game": round(a["dev_buys"] / gn, 2),
            "trades_per_game": round(a["trades"] / gn, 2),
            "pairs": pair_total.get(model, 0),
            "pair_sweep_rate": round(swept[model] / pair_total[model], 3) if pair_total.get(model) else None,
        }
    return out


METRIC_INFO = {
    "what": "Per-decision quality measured from the head-to-head games themselves, averaged per model.",
    "metrics": {
        "wasted_turn_rate": "Fraction of the model's turns (after its roll) with NO build/buy/trade/dev action — 'turns that do nothing'. Lower = better resource management.",
        "resource_gain_per_game": "Total resources flowing into the model's hand over a game (production + trade inflow). Higher reflects better placement + keeping the economy moving.",
        "settlements_per_game": "Settlements built per game (includes the 2 initial).",
        "cities_per_game": "Cities built per game (settlement upgrades).",
        "pair_sweep_rate": "Of seeds played BOTH ways (mirror), how often the model won both orientations — adaptability / winning regardless of seat.",
    },
}


def main(argv) -> int:
    if argv:
        dirs = [Path(a) for a in argv]
    else:
        # canonical matchup dirs only (hud-*); the grpo diagnostic is REUSED inside
        # hud-grpo-vs-base, so globbing diag-* too would double-count those games.
        dirs = [Path(p) for p in glob.glob(str(REPO / "transcripts/hud-*vs-base"))]
        dirs = [d for d in dirs if d.is_dir()]
    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        print("no matchup dirs found"); return 1
    models = aggregate(dirs)
    payload = {**METRIC_INFO, "dirs": [d.name for d in dirs], "models": models}
    out = REPO / "viewer/data/gamestats.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"{out.relative_to(REPO)}: {len(models)} models from {[d.name for d in dirs]}")
    for m, s in sorted(models.items(), key=lambda kv: kv[1]["trained"]):
        print(f"  {m:22} games={s['games']:>3} wasted={s['wasted_turn_rate']:.0%} "
              f"resgain={s['resource_gain_per_game']:>5} setl={s['settlements_per_game']} "
              f"city={s['cities_per_game']} sweep={s['pair_sweep_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
