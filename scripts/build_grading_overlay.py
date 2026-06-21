#!/usr/bin/env python
"""Build per-game grading sidecars for the replay viewer.

Reads a run's grading outputs (grading/findings.jsonl + grading/game_review.json)
and writes one `<game>.grading.json` next to each `<game>.view.json`, so the
viewer can overlay grading without re-fetching the whole findings file:

  <run>/<game>.grading.json = {
    failure_modes: [...game-level strategic modes for THIS game...],
    review_summary: "...",
    steps: { "<ply>": { decision_type, regret_vp, failed:[{criterion,disputed,reason,
                        claude,openai}], summary } }
  }

Ply keys match the viewer step `i` (= action_records index = grader decision ply).

    python scripts/build_grading_overlay.py transcripts/selfplay
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def _per_grader_reason(o: dict, crit: str, which: str) -> str:
    for name, g in o.get("graders", {}).items():
        if which in name and isinstance(g, dict):
            c = g.get("criteria", {}).get(crit)
            if c:
                return str(c.get("reason", ""))[:200]
    return ""


def build(run_dir: Path) -> int:
    gdir = run_dir / "grading"
    findings_path = gdir / "findings.jsonl"
    if not findings_path.exists():
        print(f"no grading at {gdir} — run scripts/grade_transcripts.py first", file=sys.stderr)
        return 1

    # game-level reviews, by game_id
    reviews = {}
    gr = gdir / "game_review.json"
    if gr.exists():
        for r in json.loads(gr.read_text()).get("reviews", []):
            reviews[r["game_id"]] = r

    # per-decision findings, grouped by game_id
    by_game: dict[str, dict] = defaultdict(lambda: {"steps": {}})
    for line in findings_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        gid, _, ply = o["decision_id"].rpartition(":")
        failed = []
        for c in o.get("criteria", []):
            if not c.get("failed"):
                continue
            failed.append({
                "criterion": c["name"], "disputed": bool(c.get("disputed")),
                "claude": _per_grader_reason(o, c["name"], "claude"),
                "openai": _per_grader_reason(o, c["name"], "openai"),
            })
        by_game[gid]["steps"][ply] = {
            "decision_type": o.get("decision_type"),
            "regret_vp": o.get("regret_vp"),
            "failed": failed,
            "summary": o.get("summary", ""),
        }

    written = 0
    for gid, data in by_game.items():
        rv = reviews.get(gid, {})
        out = {
            "game_id": gid,
            "failure_modes": rv.get("failures", []),
            "review_summary": rv.get("summary", ""),
            "winner": rv.get("winner"),
            "steps": data["steps"],
        }
        (run_dir / f"{gid}.grading.json").write_text(json.dumps(out))
        written += 1
    print(f"wrote {written} <game>.grading.json sidecars in {run_dir}/ "
          f"({len(reviews)} with game-level reviews)")
    return 0


def main(argv) -> int:
    if not argv:
        print(__doc__); return 1
    return build(Path(argv[0]))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
