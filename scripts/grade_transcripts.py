#!/usr/bin/env python
"""Dual-grader CLI: grade game transcripts into ranked failure modes.

    # preview gating only (no API calls, free):
    python scripts/grade_transcripts.py transcripts/selfplay --dry-run

    # full dual grade (Claude + OpenAI) over a run:
    export ANTHROPIC_API_KEY="$(scripts/anthropic_api_key.sh)"
    export OPENAI_API_KEY="$(scripts/openai_api_key.sh)"
    python scripts/grade_transcripts.py transcripts/selfplay

Writes <out>/findings.jsonl (one per graded decision) and <out>/report.json
(ranked failure modes + grader agreement). The regret oracle gates which
decisions get graded; Claude+OpenAI categorize them; the reconciler merges.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.grader.oracle import compute_regrets, gate_threshold, apply_gate
from harness.grader.pipeline import grade_transcript, report, select_decisions


def _transcripts(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.glob("*.json") if not p.name.endswith(".view.json"))
    return [target]


def _dry_run(paths: list[Path], gate_pct: float, low_rate: float) -> int:
    total_dec = total_gated = total_low = 0
    for p in paths:
        t = json.loads(p.read_text())
        regrets = compute_regrets(t)
        todo, thr = select_decisions(regrets, gate_pct, low_rate)
        gated = sum(1 for r in regrets if r.gated)
        print(f"{p.name}: {len(regrets)} decisions | gate(thr={thr:.3g}) "
              f"{gated} gated + {len(todo) - gated} low-regret sample = {len(todo)} to grade")
        total_dec += len(regrets); total_gated += gated; total_low += len(todo) - gated
    print(f"\nTOTAL: {total_dec} decisions -> {total_gated} gated + {total_low} sampled "
          f"= {total_gated + total_low} grading calls × 2 graders")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="transcript .json or a run directory")
    ap.add_argument("--grader-a", default="claude:claude-opus-4-8")
    ap.add_argument("--grader-b", default="openai:gpt-4o")
    ap.add_argument("--judge", default=None, help="tie-break backend spec (e.g. claude:claude-opus-4-8); off by default")
    ap.add_argument("--gate-pct", type=float, default=75.0, help="regret percentile to gate (default 75)")
    ap.add_argument("--low-regret-rate", type=float, default=0.1, help="fraction of low-regret decisions to sample")
    ap.add_argument("--samples", type=int, default=1, help="self-consistency samples per grader")
    ap.add_argument("--max-workers", type=int, default=8, help="concurrent grading calls")
    ap.add_argument("--limit", type=int, default=0, help="cap transcripts (0 = all) — for cheap smokes")
    ap.add_argument("--out", default=None, help="output dir (default <target>/grading)")
    ap.add_argument("--dry-run", action="store_true", help="oracle/gating only, no API calls")
    args = ap.parse_args()

    target = Path(args.target)
    paths = _transcripts(target)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"no transcripts at {target}", file=sys.stderr); return 1

    if args.dry_run:
        return _dry_run(paths, args.gate_pct, args.low_regret_rate)

    # real grading needs both keys
    if args.grader_a.startswith("claude") or args.grader_b.startswith("claude"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr); return 2
    if args.grader_a.startswith("openai") or args.grader_b.startswith("openai"):
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); return 2

    from harness.grader.graders import make_grader
    grader_a = make_grader(args.grader_a)
    grader_b = make_grader(args.grader_b)
    judge = make_grader(args.judge) if args.judge else None
    print(f"graders: {grader_a.name}  +  {grader_b.name}"
          + (f"  | judge: {judge.name}" if judge else "") + f"  ({len(paths)} transcripts)\n")

    all_findings = []
    for p in paths:
        t = json.loads(p.read_text())
        fs = grade_transcript(t, grader_a, grader_b, judge=judge,
                              gate_pct=args.gate_pct, low_regret_rate=args.low_regret_rate,
                              samples=args.samples, max_workers=args.max_workers)
        all_findings.extend(fs)
        print(f"  {p.name}: {len(fs)} findings")

    out_dir = Path(args.out) if args.out else (target if target.is_dir() else target.parent) / "grading"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "findings.jsonl").write_text(
        "\n".join(json.dumps(asdict(f)) for f in all_findings))
    rep = report(all_findings)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))

    print(f"\n=== agreement === {rep['agreement']}")
    print("=== top failure modes (ranked by total VP-regret) ===")
    for m in rep["failure_modes"][:8]:
        print(f"  #{m['roi_rank']} {m['category']}: freq={m['frequency']} "
              f"total_regret_vp={m['total_regret_vp']} consensus={m['consensus_rate']}")
    print(f"\nWrote {out_dir}/findings.jsonl + report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
