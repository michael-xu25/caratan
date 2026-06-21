"""CLI entrypoint: run a 1v1 match and print metrics.

Examples:
    python -m goldilocks_eval --a weighted --b value --seeds 10
    python -m goldilocks_eval --a claude --b value --seeds 5 --concurrency 4
    python -m goldilocks_eval --a alphabeta:3 --b weighted --seeds 20 --no-mirror
"""
from __future__ import annotations

import argparse
import asyncio
import os

from goldilocks_eval.runner import run_match


def _parse_seeds(seeds_arg: str, n: int, base: int) -> list[int]:
    if seeds_arg:
        return [int(s) for s in seeds_arg.split(",") if s.strip()]
    return list(range(base, base + n))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="goldilocks_eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="Agent A spec (e.g. weighted, value, claude)")
    p.add_argument("--b", required=True, help="Agent B spec")
    p.add_argument("--seeds", default="", help="Comma-separated seeds (overrides -n)")
    p.add_argument("-n", "--num-seeds", type=int, default=5, help="Number of seeds (default 5)")
    p.add_argument("--base-seed", type=int, default=0, help="First seed when using -n")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max games in flight = max concurrent LLM calls (default 8)")
    p.add_argument("--no-mirror", action="store_true",
                   help="Disable mirrored (seat-swapped) games")
    p.add_argument("--out-dir", default="transcripts/run",
                   help="Transcript output dir (default transcripts/run); empty to disable")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    seeds = _parse_seeds(args.seeds, args.num_seeds, args.base_seed)
    out_dir = args.out_dir or None

    print(f"Running {args.a} (A) vs {args.b} (B) over {len(seeds)} seeds"
          f"{' x2 mirrored' if not args.no_mirror else ''}...")
    result = asyncio.run(run_match(
        spec_a=args.a, spec_b=args.b, seeds=seeds,
        concurrency=args.concurrency, out_dir=out_dir,
        mirror=not args.no_mirror,
    ))
    print("\n" + result.summary())
    if out_dir:
        print(f"\nTranscripts written to {os.path.abspath(out_dir)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
