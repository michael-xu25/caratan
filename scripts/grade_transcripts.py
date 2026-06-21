#!/usr/bin/env python
"""Dual-grader CLI: one whole-game call per grader per transcript -> ranked
(decision_type, criterion, tag) weakness table.

    # free preview: games, decisions sampled, and the call count:
    python scripts/grade_transcripts.py transcripts/selfplay --dry-run

    # full dual grade (Claude + OpenAI), ~2 calls per game:
    export ANTHROPIC_API_KEY="$(scripts/anthropic_api_key.sh)"
    export OPENAI_API_KEY="$(scripts/openai_api_key.sh)"
    python scripts/grade_transcripts.py transcripts/selfplay --per-game 15

Each grader reads the whole game once and scores a uniform sample of N=`--per-game`
decisions from it. Total LLM calls = (#transcripts) × 2 graders. Writes
<out>/findings.jsonl + <out>/report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.grader.oracle import compute_regrets
from harness.grader.pipeline import grade_transcript, report, _sample
from harness.grader.aggregator import print_weakness_table


def _transcripts(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.glob("*.json") if not p.name.endswith(".view.json"))
    return [target]


def _dry_run(paths, per_game):
    dtypes, tags, total = Counter(), Counter(), 0
    for p in paths:
        sel = _sample(compute_regrets(json.loads(p.read_text())), per_game)
        total += len(sel)
        for r in sel:
            dtypes[r.decision_type] += 1
            tags.update(r.state_tags)
    print(f"transcripts (games): {len(paths)}")
    print(f"decisions graded: {total}  (~{per_game}/game)")
    print(f"LLM CALLS (hybrid, per-decision): {total} decisions × 2 graders = {total * 2}")
    print(f"by decision_type: {dict(dtypes)}")
    print(f"top state_tags: {dict(tags.most_common(10))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="transcript .json or a run directory")
    ap.add_argument("--grader-a", default="claude:claude-opus-4-8")
    ap.add_argument("--grader-b", default="openai:gpt-4o")
    ap.add_argument("--per-game", type=int, default=15, help="decisions sampled & scored per game")
    ap.add_argument("--concurrency", type=int, default=16, help="global parallel grader calls (tune to API rate limits)")
    ap.add_argument("--merge", choices=["consensus", "union"], default="union",
                    help="weakness table: consensus (both graders, precision) or union (either, recall/discovery)")
    ap.add_argument("--min-samples", type=int, default=15, help="aggregator floor per (criterion,tag)")
    ap.add_argument("--limit", type=int, default=0, help="cap transcripts (0 = all)")
    ap.add_argument("--out", default=None, help="output dir (default <target>/grading)")
    ap.add_argument("--dry-run", action="store_true", help="selection preview only, no API")
    args = ap.parse_args()

    target = Path(args.target)
    paths = _transcripts(target)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"no transcripts at {target}", file=sys.stderr); return 1

    if args.dry_run:
        return _dry_run(paths, args.per_game)

    if (args.grader_a.startswith("claude") or args.grader_b.startswith("claude")) and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr); return 2
    if (args.grader_a.startswith("openai") or args.grader_b.startswith("openai")) and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); return 2

    from harness.grader.graders import make_grader
    from harness.grader.pipeline import grade_run
    ga, gb = make_grader(args.grader_a), make_grader(args.grader_b)
    transcripts = [json.loads(p.read_text()) for p in paths]
    print(f"graders: {ga.name} + {gb.name}  (hybrid: per-decision, concurrency {args.concurrency})\n")

    objects = grade_run(transcripts, ga, gb, per_game=args.per_game,
                        concurrency=args.concurrency,
                        progress=lambda d, n: print(f"  {d}/{n} grader calls done"))

    out_dir = Path(args.out) if args.out else (target if target.is_dir() else target.parent) / "grading"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "findings.jsonl").write_text("\n".join(json.dumps(o) for o in objects))
    rep = report(objects, min_samples=args.min_samples, merge=args.merge)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))

    print(f"\n=== agreement === {rep['agreement']}")
    print("=== top weaknesses (Wilson-ranked fail-rate) ===")
    print_weakness_table(rep["weakness_table"], top=12)
    print(f"\nWrote {out_dir}/findings.jsonl + report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
