#!/usr/bin/env python
"""Add per-decision regret-oracle insight to every grading sidecar.

The dual LLM grader only scores a sampled subset per game (API cost). The regret
oracle, however, is FREE — it's Catanatron's value function (harness/grader/
oracle.py), no LLM. So we compute it for EVERY gradeable decision and merge it
into <game>.grading.json, giving the replay viewer objective grading insight
(regret in VP-equivalents + the oracle's best legal move) on *every* decision,
not just the LLM-sampled ones. Existing LLM verdicts are preserved; the plies
that carried them are flagged `llm_graded` so the viewer can show the richer
Claude+OpenAI criteria there and oracle-only insight everywhere else.

    python scripts/add_oracle_insights.py [run_dir=transcripts/selfplay]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.grader.oracle import compute_regrets


def _fmt(ob):
    """[color, type, value] -> 'TYPE value' (the oracle's best legal move)."""
    if not ob:
        return None
    typ, val = ob[1], ob[2]
    return f"{typ} {val}".strip() if val is not None else typ


def main(run_dir: str = "transcripts/selfplay") -> int:
    trans = [p for p in sorted(glob.glob(os.path.join(run_dir, "*.json")))
             if not p.endswith(".view.json") and not p.endswith(".grading.json")]
    if not trans:
        print(f"no transcripts in {run_dir}", file=sys.stderr)
        return 1
    for tp in trans:
        base = tp[:-len(".json")]
        gp = base + ".grading.json"
        try:
            tr = json.load(open(tp))
        except Exception as e:
            print(f"skip {os.path.basename(base)}: bad transcript ({e})")
            continue
        grad = json.load(open(gp)) if os.path.exists(gp) else {
            "game_id": os.path.basename(base), "steps": {}}
        steps = grad.setdefault("steps", {})
        orig = set(steps.keys())                      # the LLM-graded plies
        try:
            regs = compute_regrets(tr)
        except Exception as e:
            print(f"skip {os.path.basename(base)}: oracle failed ({e})")
            continue
        added = 0
        for r in regs:
            k = str(r.ply)
            if k not in steps:
                steps[k] = {}
                added += 1
            s = steps[k]
            s["decision_type"] = r.decision_type
            s["regret_vp"] = round(r.regret_vp, 3)
            s["oracle_best"] = _fmt(r.oracle_best)
            s["num_legal"] = r.num_legal
        for k, s in steps.items():
            s["llm_graded"] = k in orig               # did Claude+OpenAI score this one?
        json.dump(grad, open(gp, "w"))
        print(f"{os.path.basename(base):<22} oracle decisions={len(regs)} "
              f"(+{added} new) · {len(orig)} LLM-graded · {len(steps)} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "transcripts/selfplay"))
