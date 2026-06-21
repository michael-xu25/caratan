"""Per-weakness scenario eval CLI — delegates to Michael's goldilocks_eval.

    python -m harness.scenario_cli --scenarios data/placement_heldout.jsonl --model claude
    python -m harness.scenario_cli --scenarios data/placement_heldout.jsonl \
        --before claude:claude-haiku-4-5 --after claude
"""
from goldilocks_eval.scenario_cli import build_parser, main  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
