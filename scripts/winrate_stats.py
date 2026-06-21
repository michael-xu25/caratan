#!/usr/bin/env python
"""Win-rate stats for a head-to-head matchup run dir (trained vs base).

Reads the saved transcripts, computes seat-fair win-rate (the run is mirrored, so
each seed is played both ways) plus VP and cap-stall stats, and writes:
  - <run_dir>/winrate.json   (machine, for the UI)
  - <run_dir>/WINRATE.md     (human)
and merges the row into viewer/data/matchups.json (the dashboard manifest).

    python scripts/winrate_stats.py transcripts/hud-grpo-vs-base
    python scripts/winrate_stats.py transcripts/hud-*-vs-base   # several at once
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_HINT = "Qwen3-8B"   # the untrained baseline's name contains this


def _transcripts(run_dir: Path):
    return [p for p in sorted(run_dir.glob("*.json"))
            if not (p.name.endswith(".view.json") or p.name.endswith(".grading.json")
                    or p.name == "winrate.json")]


def compute(run_dir: Path) -> dict:
    games = []
    for p in _transcripts(run_dir):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "seats" not in d:
            continue
        games.append(d)

    # identify the two agents (trained vs base)
    names = set()
    for g in games:
        names.update(g["seats"].values())
    base = next((n for n in names if BASE_HINT in n), None)
    trained = next((n for n in names if n != base), None)

    n = tw = bw = draws = trunc = 0
    vp_t = vp_b = 0
    for g in games:
        seats = g["seats"]            # {color: agent_name}
        win_color = g.get("winning_color")
        fvp = g.get("final_victory_points", {}) or {}
        n += 1
        trunc += int(bool(g.get("truncated")))
        for color, name in seats.items():
            v = fvp.get(color, 0)
            if name == trained:
                vp_t += v
            elif name == base:
                vp_b += v
        if not win_color:
            draws += 1
        elif seats.get(win_color) == trained:
            tw += 1
        else:
            bw += 1
    decided = tw + bw
    try:
        rel = str(run_dir.relative_to(REPO))
    except ValueError:
        rel = str(run_dir)
    return {
        "run_dir": rel,
        "matchup": run_dir.name,
        "trained": trained, "base": base,
        "games": n, "trained_wins": tw, "base_wins": bw, "draws": draws,
        "trained_winrate": round(tw / decided, 3) if decided else None,
        "trained_winrate_incl_draws": round(tw / n, 3) if n else None,
        "avg_vp_trained": round(vp_t / n, 2) if n else None,
        "avg_vp_base": round(vp_b / n, 2) if n else None,
        "cap_stalls": trunc,
    }


def write_outputs(stats: dict, run_dir: Path):
    (run_dir / "winrate.json").write_text(json.dumps(stats, indent=2))
    wr = stats["trained_winrate"]
    md = (f"# Win-rate — {stats['matchup']}\n\n"
          f"**{stats['trained']}** (trained) vs **{stats['base']}** (untrained base), "
          f"mirrored seat-swap.\n\n"
          f"| games | trained wins | base wins | draws | trained win-rate | avg VP (trained/base) | cap-stalls |\n"
          f"|---|---|---|---|---|---|---|\n"
          f"| {stats['games']} | {stats['trained_wins']} | {stats['base_wins']} | {stats['draws']} | "
          f"{'%.0f%%' % (wr*100) if wr is not None else 'n/a'} | "
          f"{stats['avg_vp_trained']}/{stats['avg_vp_base']} | {stats['cap_stalls']} |\n\n"
          f"Win-rate is over decided games (draws excluded). Mirrored → seat-fair.\n")
    (run_dir / "WINRATE.md").write_text(md)


def merge_manifest(all_stats: list[dict]):
    """Merge into viewer/data/matchups.json (keyed by matchup; preserves others)."""
    mpath = REPO / "viewer/data/matchups.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if mpath.exists():
        try:
            existing = {m["matchup"]: m for m in json.loads(mpath.read_text()).get("matchups", [])}
        except Exception:
            existing = {}
    for s in all_stats:
        existing[s["matchup"]] = s
    mpath.write_text(json.dumps({"matchups": list(existing.values())}, indent=2))
    return mpath


def main(argv) -> int:
    if not argv:
        print(__doc__); return 1
    dirs = []
    for a in argv:
        dirs += [Path(p) for p in glob.glob(a)] if "*" in a else [Path(a)]
    all_stats = []
    for d in dirs:
        if not d.is_dir():
            print(f"  skip (not a dir): {d}"); continue
        s = compute(d)
        write_outputs(s, d)
        all_stats.append(s)
        wr = s["trained_winrate"]
        print(f"{s['matchup']:30} games={s['games']:>3} "
              f"trained {s['trained_wins']}-{s['base_wins']} base, draws {s['draws']} "
              f"| win-rate {('%.0f%%' % (wr*100)) if wr is not None else 'n/a'}")
    if all_stats:
        mp = merge_manifest(all_stats)
        print(f"\nwrote per-run winrate.json/WINRATE.md + merged {mp.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
