"""Tiny CLI to run a mirrored batch and write transcripts + a summary.

Examples
--------
    python -m harness.cli --a value --b random --seeds 1,2,3
    python -m harness.cli --a value --b weighted --n 10 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from harness.runner import run_batch, run_mirror_pair
from harness.transcripts import render_summary_table, render_pair_report


def _parse_seeds(args) -> list[int]:
    if args.seeds:
        return [int(s) for s in args.seeds.split(",") if s.strip()]
    return list(range(args.seed_start, args.seed_start + args.n))


def main():
    p = argparse.ArgumentParser(description="Run a mirrored 1v1 eval batch.")
    p.add_argument("--a", required=True, help="agent A spec, e.g. 'value' or 'claude:claude-opus-4-8'")
    p.add_argument("--b", required=True, help="agent B spec")
    p.add_argument("--pair", type=int, metavar="SEED",
                   help="fairness mode: play ONE mirrored pair on this seed and "
                        "print the fairness report (same board, swapped seats)")
    p.add_argument("--seeds", help="comma-separated seeds (overrides --n/--seed-start)")
    p.add_argument("--n", type=int, default=5, help="number of seeds when --seeds omitted")
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--no-mirror", action="store_true", help="disable seat-swapped mirror games")
    p.add_argument("--no-balanced-dice", action="store_true",
                   help="fall back to Catanatron's global-RNG dice (dice will diverge "
                        "across a mirrored pair); on by default for fair dice")
    p.add_argument("--concurrency", type=int, default=8,
                   help="max concurrent games == max concurrent LLM calls")
    p.add_argument("--reasoning", action="store_true",
                   help="capture model reasoning (for viewable testing transcripts); "
                        "off by default to keep runs cheap")
    p.add_argument("--run-dir", default="transcripts/batch")
    args = p.parse_args()

    if args.pair is not None:
        normal, swapped = asyncio.run(
            run_mirror_pair(args.a, args.b, args.pair, run_dir=args.run_dir,
                            capture_reasoning=args.reasoning,
                            balanced_dice=not args.no_balanced_dice))
        report = render_pair_report(normal, swapped)
        print(report)
        report_path = Path(args.run_dir) / f"pair_seed{args.pair}.txt"
        report_path.write_text(report)
        print(f"\nReport + per-game transcripts written to {args.run_dir}/")
        return

    seeds = _parse_seeds(args)
    result = asyncio.run(run_batch(
        args.a, args.b, seeds,
        mirror=not args.no_mirror,
        concurrency=args.concurrency,
        run_dir=args.run_dir,
        capture_reasoning=args.reasoning,
        balanced_dice=not args.no_balanced_dice,
    ))

    summary = render_summary_table(result)
    print(summary)
    summary_path = Path(args.run_dir) / "summary.txt"
    summary_path.write_text(summary)
    print(f"\nTranscripts + summary written to {args.run_dir}/")


if __name__ == "__main__":
    main()
