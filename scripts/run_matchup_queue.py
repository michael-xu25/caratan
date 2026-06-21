#!/usr/bin/env python
"""Staged matchup queue — reuse the grpo diagnostic, never redo a game.

Order (per the plan):
  1. WAIT for the running grpo diagnostic (12 seeds = 24 games, already on disk).
  2. placement-only vs base on those SAME 12 diag seeds  -> placement catches up.
  3. BOTH matchups on the remaining 88 seeds.
The 12 grpo diagnostic games are REUSED (copied into the grpo matchup dir), never
re-run. Each stage builds viewer replay data + manifest, recomputes win-rate, and
(optionally) commits & pushes -- results land in the UI as we go.

    set -a; source .env; set +a
    python scripts/run_matchup_queue.py --wait-for-diag --max-turns 300 --push
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness.runner import run_batch

BASE = "hud:Qwen/Qwen3-8B"
PLACE_SPEC, PLACE_RUN = "hud:catan-placement-only", "hud-postplacement-vs-base"
GRPO_SPEC,  GRPO_RUN  = "hud:catan-grpo-q8b",       "hud-grpo-vs-base"
# 12 seeds covered by the running grpo diagnostic (cap 400, reasoning off)
DIAG_SEEDS = [1001, 1004, 1020, 1021, 1023, 1026, 1028, 1029, 1032, 1035, 1036, 1038]
DIAG_DIR = REPO / "transcripts/_hud_diag"


def grader_games_seeds():
    idx = json.loads((REPO / "dataset/initial/index.json").read_text())
    return sorted(b["seed"] for b in idx["boards"] if b.get("split") == "grader_games")


def _is_transcript(p: str) -> bool:
    return not (p.endswith(".view.json") or p.endswith(".grading.json"))


def _count(d: Path) -> int:
    return len([p for p in glob.glob(str(d / "*.json")) if _is_transcript(p)])


def wait_for_diag(target=24, poll=15):
    print(f"[queue] waiting for diagnostic to reach {target} games ...", flush=True)
    while _count(DIAG_DIR) < target:
        alive = subprocess.run(["pgrep", "-f", "catan-grpo|run_batch"], capture_output=True).returncode == 0
        if not alive and _count(DIAG_DIR) < target:
            print(f"[queue] diag process gone at {_count(DIAG_DIR)}/{target}; proceeding.", flush=True)
            break
        time.sleep(poll)
    print(f"[queue] diagnostic complete ({_count(DIAG_DIR)} games).", flush=True)


def reuse_grpo_diag(dst: Path):
    """Copy the 12 grpo diagnostic transcripts into the grpo matchup dir (reuse, no rerun)."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in glob.glob(str(DIAG_DIR / "*.json")):
        if not _is_transcript(p):
            continue
        tgt = dst / Path(p).name
        if not tgt.exists():
            shutil.copy(p, tgt); n += 1
    print(f"[queue] reused {n} grpo diagnostic transcript(s) into {dst.name}/", flush=True)


def post_stage(run_dir: Path, label: str, push: bool):
    subprocess.run([sys.executable, "scripts/build_viewer_data.py", str(run_dir)], cwd=REPO)
    subprocess.run([sys.executable, "scripts/build_viewer_index.py"], cwd=REPO)
    subprocess.run([sys.executable, "scripts/winrate_stats.py", str(run_dir)], cwd=REPO)
    if push:
        subprocess.run(["git", "add", str(run_dir), "viewer/runs.json", "viewer/data/matchups.json"], cwd=REPO)
        subprocess.run(["git", "commit", "-q", "-m", f"matchups: {label}"], cwd=REPO)
        subprocess.run(["git", "push", "origin", "cara"], cwd=REPO)
        print(f"[queue] pushed: {label}", flush=True)


async def run_stage(spec, run_dir: Path, seeds, on_seeds, concurrency, max_turns, label):
    run_dir.mkdir(parents=True, exist_ok=True)
    on = [s for s in seeds if s in on_seeds]
    off = [s for s in seeds if s not in on_seeds]
    for batch, reasoning in ((on, True), (off, False)):
        if not batch:
            continue
        print(f"  [{label}] {'reason ON ' if reasoning else 'reason OFF'} "
              f"{len(batch)} seeds x mirror = {len(batch)*2} games (cap {max_turns}, conc {concurrency})",
              flush=True)
        await run_batch(spec, BASE, batch, mirror=True, concurrency=concurrency,
                        run_dir=str(run_dir), capture_reasoning=reasoning,
                        max_turns=max_turns, vps_to_win=10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--reasoning-on", type=int, default=0,
                    help="first N grader_games seeds (global) played with reasoning ON (0 = all off)")
    ap.add_argument("--wait-for-diag", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("HUD_API_KEY"):
        print("ERROR: HUD_API_KEY not set (set -a; source .env; set +a)", file=sys.stderr)
        return 2
    os.environ.setdefault("FIREWORKS_TEMPERATURE", "0")

    seeds = grader_games_seeds()
    diag = [s for s in seeds if s in DIAG_SEEDS]            # 12 (grpo already done)
    rest = [s for s in seeds if s not in DIAG_SEEDS]        # 88
    on_seeds = set(seeds[:args.reasoning_on])               # global first-N get reasoning

    # ordered stages (label, spec, run_dir, seeds)
    stages = [
        ("placement catch-up (12 diag seeds)", PLACE_SPEC, PLACE_RUN, diag),
        ("placement remaining 88",             PLACE_SPEC, PLACE_RUN, rest),
        ("grpo remaining 88",                  GRPO_SPEC,  GRPO_RUN,  rest),
    ]
    new_games = sum(len(s) * 2 for _, _, _, s in stages)
    print(f"STAGED QUEUE | cap {args.max_turns} | conc {args.concurrency} | reasoning-on {args.reasoning_on}")
    print(f"  reuse: 24 grpo diagnostic games (seeds {diag[0]}..{diag[-1]}) -> copied into {GRPO_RUN}/")
    for label, spec, rd, s in stages:
        print(f"  stage: {label:34} {spec:26} {len(s)} seeds x mirror = {len(s)*2} games -> {rd}/")
    print(f"  NEW games to run: {new_games}   (grpo total 100 seeds, placement total 100 seeds)")
    if args.dry_run:
        print("DRY RUN — no games played."); return 0

    if args.wait_for_diag:
        wait_for_diag()

    # reuse grpo's diagnostic into its matchup dir up front so its win-rate covers 100 seeds
    reuse_grpo_diag(REPO / "transcripts" / GRPO_RUN)
    post_stage(REPO / "transcripts" / GRPO_RUN, "grpo (reused 12 diag seeds)", args.push)

    for label, spec, rd, s in stages:
        full = REPO / "transcripts" / rd
        print(f"\n===== STAGE: {label} =====", flush=True)
        t0 = time.time()
        asyncio.run(run_stage(spec, full, s, on_seeds, args.concurrency, args.max_turns, label))
        print(f"  stage done in {(time.time()-t0)/60:.1f} min", flush=True)
        post_stage(full, label, args.push)
    print("\n[queue] ALL STAGES DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
