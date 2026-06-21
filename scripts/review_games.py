#!/usr/bin/env python
"""Game-level failure review: one LLM pass per game -> ranked STRATEGIC failure modes.

Complements the per-decision grader (scripts/grade_transcripts.py): per-decision
finds blunders; this finds cumulative/strategic reasons the model fails to win
(stalls, never converts a lead). One call per game, single grader by default.

    export ANTHROPIC_API_KEY="$(scripts/anthropic_api_key.sh)"
    python scripts/review_games.py transcripts/selfplay --concurrency 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.grader.game_review import review_run, aggregate_reviews, STRATEGIC_MODES


def _transcripts(target: Path):
    if target.is_dir():
        return sorted(p for p in target.glob("*.json") if not p.name.endswith(".view.json"))
    return [target]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="run dir or transcript .json")
    ap.add_argument("--model", default="claude:claude-opus-4-8", help="reviewer backend")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    target = Path(args.target)
    paths = _transcripts(target)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"no transcripts at {target}", file=sys.stderr); return 1

    if args.model.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr); return 2
    if args.model.startswith("openai") and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); return 2

    from goldilocks_eval.agents.factory import make_backend
    backend = make_backend(args.model)
    print(f"reviewer: {backend.name}  ({len(paths)} games, one call each)\n")

    transcripts = [json.loads(p.read_text()) for p in paths]
    reviews = review_run(transcripts, backend, concurrency=args.concurrency)
    agg = aggregate_reviews(reviews)

    out = Path(args.out) if args.out else (target if target.is_dir() else target.parent) / "grading"
    out.mkdir(parents=True, exist_ok=True)
    (out / "game_review.json").write_text(json.dumps({"reviews": reviews, **agg}, indent=2))

    print(f"=== STRATEGIC FAILURE MODES ({agg['n_games']} games) ===")
    print(f"{'mode':<24}{'games':>6}{'rate':>7}  examples")
    for r in agg["failure_modes"]:
        print(f"{r['mode']:<24}{r['games']:>6}{r['rate']*100:>6.0f}%  {', '.join(r['examples'][:3])}")
    print(f"\nWrote {out}/game_review.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
