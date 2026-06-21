"""CLI for the per-weakness scenario eval (primary metric).

Examples:
    # Score one model on a held-out set
    python -m goldilocks_eval.scenario_cli --scenarios data/placement_heldout.jsonl --model claude

    # Before/after: base vs trained on the same held-out set (the headline)
    python -m goldilocks_eval.scenario_cli --scenarios data/placement_heldout.jsonl \
        --before claude:claude-haiku-4-5 --after claude
"""
from __future__ import annotations

import argparse
import asyncio

from goldilocks_eval.agents.factory import make_backend
from goldilocks_eval.scenario import before_after, evaluate, load_scenarios


def _print_report(report) -> None:
    print(f"\n[{report.label}] scenarios: {len(report.results)}  "
          f"accuracy: {report.accuracy:.1%}")
    for env, acc in sorted(report.by_env().items()):
        n = sum(1 for r in report.results if r.env == env)
        print(f"  {env}: {acc:.1%}  (n={n})")
    fb = sum(r.fell_back for r in report.results)
    if fb:
        print(f"  (fell back to default on {fb} scenarios)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="goldilocks_eval.scenario_cli",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", required=True, help="Path to scenarios JSONL")
    p.add_argument("--split", default="heldout",
                   help="Only score this split (default heldout; '' = all)")
    p.add_argument("--model", help="Single model spec to score (e.g. claude)")
    p.add_argument("--before", help="Base model spec for before/after")
    p.add_argument("--after", help="Trained model spec for before/after")
    p.add_argument("--concurrency", type=int, default=8)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    split = args.split or None
    scenarios = load_scenarios(args.scenarios, split=split)
    print(f"Loaded {len(scenarios)} scenarios"
          f"{f' (split={split})' if split else ''} from {args.scenarios}")

    if args.before and args.after:
        reports = asyncio.run(before_after(
            make_backend(args.before), make_backend(args.after),
            scenarios, concurrency=args.concurrency))
        _print_report(reports["before"])
        _print_report(reports["after"])
        delta = reports["after"].accuracy - reports["before"].accuracy
        print(f"\nHeadline delta: {reports['before'].accuracy:.1%} -> "
              f"{reports['after'].accuracy:.1%}  ({delta:+.1%})")
    elif args.model:
        report = asyncio.run(evaluate(
            make_backend(args.model), scenarios, args.model,
            concurrency=args.concurrency))
        _print_report(report)
    else:
        print("Provide --model, or both --before and --after.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
