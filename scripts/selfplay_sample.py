"""Base-model self-play over the board sample (the weakness-discovery feeder).

Runs the base 4B (Gemma E4B on Fireworks) against an IDENTICAL copy of itself,
mirrored — each board played twice with seats swapped (P1 then P2). Produces:
  - per-game transcripts (the games the Claude+GPT-5.5 analysts will read), and
  - a head-to-head summary. Identical models → ~50% after mirroring, so the
    win-rate here is a FAIRNESS sanity check, not a skill measure.

PARALLELISM
-----------
Each board = one mirrored PAIR = 2 games. Games run concurrently in a process
pool (the canonical, RNG-isolated `harness` runner). `--concurrency` = max games
in flight. A 1v1 game has exactly one decision in flight at a time, so:

    concurrent Fireworks requests  ≈  --concurrency   (one per running game)

Both seats hit the SAME deployment (identical model), so you do NOT need a second
Fireworks instance — only the deployment's replica/batch capacity bounds
throughput. Your deployment is min=max=1 replica: it will batch many requests on
the one H200, but ~100 simultaneous full games will queue. To actually get 100
in parallel, raise max replicas on the deployment (autoscaling will fan out);
otherwise keep --concurrency to what one replica serves comfortably (try 16–32,
watch latency, scale up).

BOARDS (paths connected later)
------------------------------
Cara's 300–500 boards are seeds. Point `--seeds-file` at her manifest once it
lands; until then use `--seeds` or `--n/--start`. `load_board_seeds` is tolerant:
newline ints, a JSON list, {"seeds":[...]}, or scenario/board dicts with
`board_seed`. ⚠️ ADAPT it to her exact format when the path is connected.

USAGE
-----
    set -a; source .env; set +a          # FIREWORKS_API_KEY + FIREWORKS_MODEL
    # smoke (3 boards) — confirm the deployment is Ready and calls land:
    python scripts/selfplay_sample.py --n 3 --concurrency 3
    # full run over Cara's boards:
    python scripts/selfplay_sample.py --seeds-file data/boards.jsonl --concurrency 32
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Repo root on path so `harness` / `goldilocks_eval` import when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.runner import run_batch  # noqa: E402
from harness.transcripts import render_summary_table  # noqa: E402


def load_board_seeds(path: str) -> list[int]:
    """Tolerant loader for the board-seed list. ADAPT to Cara's manifest format
    when the path is connected (the seam this function exists to absorb)."""
    text = Path(path).read_text().strip()
    if not text:
        return []
    # Try JSON (list of ints, list of board/scenario dicts, or wrapper object).
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj = obj.get("seeds") or obj.get("boards") or obj.get("board_seeds") or []
        seeds = []
        for item in obj:
            if isinstance(item, (int, str)):
                seeds.append(int(item))
            elif isinstance(item, dict):
                seeds.append(int(item["board_seed"]))
        return _dedup(seeds)
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass
    # JSONL: one JSON object per line with a board_seed (e.g. Cara's scenarios).
    seeds = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            seeds.append(int(json.loads(line)["board_seed"]))
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            seeds.append(int(line))  # plain int per line
    return _dedup(seeds)


def _dedup(seeds: list[int]) -> list[int]:
    return sorted(dict.fromkeys(seeds))


def resolve_seeds(args) -> list[int]:
    if args.seeds_file:
        return load_board_seeds(args.seeds_file)
    if args.seeds:
        return _dedup(int(s) for s in args.seeds.split(",") if s.strip())
    return list(range(args.start, args.start + args.n))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("board source (pick one; default --n range)")
    src.add_argument("--seeds-file", help="path to Cara's board manifest")
    src.add_argument("--seeds", help="comma-separated board seeds")
    src.add_argument("--n", type=int, default=5, help="number of boards (range stand-in)")
    src.add_argument("--start", type=int, default=0, help="first seed for the range")
    p.add_argument("--model", default=None,
                   help="Fireworks model spec; default fireworks:$FIREWORKS_MODEL")
    p.add_argument("--concurrency", type=int, default=32,
                   help="max games in flight ≈ concurrent Fireworks calls (default 32)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="sampling temperature for the models (default 0.7). >0 so two "
                        "identical models don't play identically into turn-limit draws; "
                        "0 for greedy/reproducible play.")
    p.add_argument("--no-reasoning", action="store_true",
                   help="don't capture model reasoning (cheaper; but the weakness "
                        "analysts need reasoning — leave on for real runs)")
    p.add_argument("--run-dir", default="transcripts/selfplay")
    args = p.parse_args()

    # Resolve the model: explicit spec, or fireworks:$FIREWORKS_MODEL.
    model = args.model
    if not model:
        fw = os.environ.get("FIREWORKS_MODEL")
        if not fw:
            print("ERROR: set FIREWORKS_MODEL (set -a; source .env; set +a) or pass "
                  "--model fireworks:accounts/<acct>/deployments/<id>", file=sys.stderr)
            return 2
        model = f"fireworks:{fw}"
    if model.startswith("fireworks") and not os.environ.get("FIREWORKS_API_KEY"):
        print("ERROR: FIREWORKS_API_KEY not set (set -a; source .env; set +a)", file=sys.stderr)
        return 2

    # Cross the process-pool boundary via env: spawned game workers build their
    # FireworksBackend from the spec and read FIREWORKS_TEMPERATURE.
    os.environ["FIREWORKS_TEMPERATURE"] = str(args.temperature)

    seeds = resolve_seeds(args)
    if not seeds:
        print("ERROR: no board seeds resolved.", file=sys.stderr)
        return 2

    n_games = len(seeds) * 2  # mirrored
    print(f"Self-play (identical models): {model}")
    print(f"boards: {len(seeds)}  ->  games: {n_games} (mirrored)  "
          f"concurrency: {args.concurrency}")
    print(f"temperature: {args.temperature}  "
          f"reasoning capture: {'OFF' if args.no_reasoning else 'ON'}  "
          f"run-dir: {args.run_dir}")
    print("note: identical models → ~50% A win-rate after mirroring is the "
          "FAIRNESS signal, not skill. Transcripts are the real output.\n")

    result = asyncio.run(run_batch(
        model, model, seeds,
        mirror=True,
        concurrency=args.concurrency,
        run_dir=args.run_dir,
        capture_reasoning=not args.no_reasoning,
    ))

    summary = render_summary_table(result)
    print(summary)
    summary_path = Path(args.run_dir) / "summary.txt"
    summary_path.write_text(summary)
    print(f"\nTranscripts + summary -> {args.run_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
