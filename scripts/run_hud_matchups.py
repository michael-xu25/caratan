#!/usr/bin/env python
"""Head-to-head matchups over the HUD gateway: trained checkpoint vs untrained base.

Each matchup plays the 100 grader_games seeds mirrored (seat-swap) so win-rate is
seat-fair. The first N seeds (default 20 — the same ones we ran the base on) are
played with reasoning ON; the rest with reasoning OFF to save compute. Transcripts
are saved per matchup (browsable in the viewer); win-rate via scripts/winrate_stats.py.

    set -a; source .env; set +a            # HUD_API_KEY
    python scripts/run_hud_matchups.py --dry-run          # verify config, no games
    python scripts/run_hud_matchups.py --concurrency 32 --max-turns 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.runner import run_batch
from harness.transcripts import render_summary_table

BASE = "hud:Qwen/Qwen3-8B"   # untrained baseline
MATCHUPS = [
    # (run_dir name, trained model spec, label)
    ("hud-postplacement-vs-base", "hud:catan-placement-only", "post-placement (trained, placement only)"),
    ("hud-grpo-vs-base",          "hud:catan-grpo-q8b",       "post-build (trained, full chain — shipped)"),
]


def grader_games_seeds():
    idx = json.loads((Path(__file__).resolve().parent.parent / "dataset/initial/index.json").read_text())
    return sorted(b["seed"] for b in idx["boards"] if b.get("split") == "grader_games")


async def run_matchup(trained_spec, run_dir, on_seeds, off_seeds, concurrency, max_turns):
    """Two batches into one run dir: reasoning-ON seeds then reasoning-OFF seeds."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for seeds, reasoning in ((on_seeds, True), (off_seeds, False)):
        if not seeds:
            continue
        print(f"  [{'reasoning ON ' if reasoning else 'reasoning OFF'}] {len(seeds)} seeds "
              f"x mirror = {len(seeds)*2} games ...", flush=True)
        res = await run_batch(trained_spec, BASE, seeds, mirror=True, concurrency=concurrency,
                              run_dir=run_dir, capture_reasoning=reasoning, max_turns=max_turns,
                              vps_to_win=10)
        results.append(res)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reasoning-on", type=int, default=20, help="first N seeds played with reasoning on")
    ap.add_argument("--concurrency", type=int, default=32, help="parallel games (gateway plateaus ~conc 24-32)")
    ap.add_argument("--max-turns", type=int, default=200, help="turn cap (games stall; lower = faster, VP tie-break)")
    ap.add_argument("--limit-seeds", type=int, default=0, help="cap total seeds (0 = all 100) — for a smaller run")
    ap.add_argument("--dry-run", action="store_true", help="resolve specs + print the plan, play NO games")
    args = ap.parse_args()

    seeds = grader_games_seeds()
    if args.limit_seeds:
        seeds = seeds[:args.limit_seeds]
    on_seeds, off_seeds = seeds[:args.reasoning_on], seeds[args.reasoning_on:]

    print(f"grader_games seeds: {len(seeds)}  | reasoning ON: {len(on_seeds)} ({on_seeds[0] if on_seeds else '-'}"
          f"..{on_seeds[-1] if on_seeds else '-'})  | reasoning OFF: {len(off_seeds)}")
    print(f"base (untrained): {BASE}")
    games_per_matchup = len(seeds) * 2
    print(f"matchups: {len(MATCHUPS)}  | games each: {games_per_matchup}  | total: {len(MATCHUPS)*games_per_matchup}")
    for run_dir, spec, label in MATCHUPS:
        print(f"  - {spec:26} vs base   [{label}]  -> transcripts/{run_dir}/")

    if not os.environ.get("HUD_API_KEY"):
        print("ERROR: HUD_API_KEY not set (set -a; source .env; set +a)", file=sys.stderr)
        return 2

    # Resolve every spec via the harness path — constructs the backend (checks the
    # key + model id) WITHOUT playing a game. This is the bug check.
    from harness.agents import make_agent
    from catanatron.models.player import Color
    for run_dir, spec, label in MATCHUPS:
        for s in (spec, BASE):
            a = make_agent(s, Color.RED)
            print(f"  resolved {s} -> {a.__class__.__name__}", flush=True)

    if args.dry_run:
        print("\nDRY RUN ok — specs resolve, config valid, no games played.")
        return 0

    os.environ.setdefault("FIREWORKS_TEMPERATURE", "0")  # deterministic head-to-head
    for run_dir, spec, label in MATCHUPS:
        full = f"transcripts/{run_dir}"
        print(f"\n===== MATCHUP: {spec} vs {BASE}  -> {full} =====", flush=True)
        asyncio.run(run_matchup(spec, full, on_seeds, off_seeds, args.concurrency, args.max_turns))
        print(f"  done: {full}/ (run scripts/winrate_stats.py {full} for the table)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
