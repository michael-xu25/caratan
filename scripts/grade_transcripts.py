#!/usr/bin/env python
"""Dual-grader CLI: grade game transcripts into a ranked (criterion, tag) weakness table.

    # preview what would be graded (no API calls, free) — counts by decision_type + tags:
    python scripts/grade_transcripts.py transcripts/selfplay --dry-run

    # full dual grade (Claude + OpenAI):
    export ANTHROPIC_API_KEY="$(scripts/anthropic_api_key.sh)"
    export OPENAI_API_KEY="$(scripts/openai_api_key.sh)"
    python scripts/grade_transcripts.py transcripts/selfplay --max-per-game 40

Writes <out>/findings.jsonl (one aggregator-ready grader object per decision, both
raw verdicts kept) and <out>/report.json (Wilson-ranked weakness table + dual-grader
agreement). Every decision that maps to a taxonomy decision_type is graded by both
graders; the reconciler marks a criterion failed only on consensus.
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
from harness.grader.pipeline import grade_transcript, report, _select
from harness.grader.aggregator import print_weakness_table


def _transcripts(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.glob("*.json") if not p.name.endswith(".view.json"))
    return [target]


def _dry_run(paths, gate_pct, max_per_game):
    dtypes, tags, total_sel = Counter(), Counter(), 0
    for p in paths:
        regrets = compute_regrets(json.loads(p.read_text()))
        sel = _select(regrets, gate_pct, max_per_game)
        total_sel += len(sel)
        for r in sel:
            dtypes[r.decision_type] += 1
            tags.update(r.state_tags)
        print(f"{p.name}: {len(regrets)} gradeable decisions -> {len(sel)} selected")
    print(f"\nTOTAL selected: {total_sel}  (= {total_sel * 2} grader calls)")
    print(f"by decision_type: {dict(dtypes)}")
    print(f"top state_tags: {dict(tags.most_common(10))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="transcript .json or a run directory")
    ap.add_argument("--grader-a", default="claude:claude-opus-4-8")
    ap.add_argument("--grader-b", default="openai:gpt-4o")
    ap.add_argument("--max-per-game", type=int, default=0, help="uniform-sample cap per game (0 = all; budget control)")
    ap.add_argument("--gate-pct", type=float, default=0.0, help="optional regret-percentile gate (biases the denominator — prefer --max-per-game)")
    ap.add_argument("--samples", type=int, default=1, help="self-consistency samples per grader")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--min-samples", type=int, default=15, help="aggregator floor: (criterion,tag) below this isn't ranked")
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
        return _dry_run(paths, args.gate_pct, args.max_per_game)

    if (args.grader_a.startswith("claude") or args.grader_b.startswith("claude")) and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr); return 2
    if (args.grader_a.startswith("openai") or args.grader_b.startswith("openai")) and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); return 2

    from harness.grader.graders import make_grader
    ga, gb = make_grader(args.grader_a), make_grader(args.grader_b)
    print(f"graders: {ga.name} + {gb.name}  ({len(paths)} transcripts)\n")

    objects = []
    for p in paths:
        objs = grade_transcript(json.loads(p.read_text()), ga, gb,
                                gate_pct=args.gate_pct, max_per_game=args.max_per_game,
                                samples=args.samples, max_workers=args.max_workers)
        objects.extend(objs)
        print(f"  {p.name}: {len(objs)} decisions graded")

    out_dir = Path(args.out) if args.out else (target if target.is_dir() else target.parent) / "grading"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "findings.jsonl").write_text("\n".join(json.dumps(o) for o in objects))
    rep = report(objects, min_samples=args.min_samples)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))

    print(f"\n=== agreement === {rep['agreement']}")
    print("=== top weaknesses (Wilson-ranked fail-rate) ===")
    print_weakness_table(rep["weakness_table"], top=12)
    print(f"\nWrote {out_dir}/findings.jsonl + report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
