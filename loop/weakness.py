"""Weakness mining — turn self-play transcripts into a failure-mode report.

No game reconstruction needed: works off the per-decision records the harness
already writes (action_type, chosen, legal_actions, phase, fell_back, ...) plus
game outcomes. Produces (a) aggregate stats per model and (b) a sample of
concrete decisions from LOST games — exactly the two things the env-gen brain
reads to decide what new env to build next round.

    .venv-modal/bin/python loop/weakness.py transcripts/selfplay-xxxx --model modal:catan
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load(transcript_dir):
    for p in sorted(Path(transcript_dir).glob("*.json")):
        if p.name.endswith(".view.json"):
            continue
        try:
            yield json.loads(p.read_text())
        except Exception:
            continue


def mine(transcript_dir, model=None, max_samples=40):
    games = list(_load(transcript_dir))
    if not games:
        return {"error": f"no transcripts in {transcript_dir}"}

    by_type = Counter()
    fell_back = Counter()          # invalid-answer fallbacks per action_type
    by_phase = Counter()
    losing_samples = []
    n_games = n_truncated = 0
    vp_for = []

    for g in games:
        n_games += 1
        n_truncated += int(g.get("truncated", False))
        seats = g.get("seats", {})
        win = g.get("winning_color")
        # which color is our model? (seat_agent matches `model`)
        our_colors = {c for c, a in seats.items()
                      if model is None or model in str(a)}
        vps = g.get("final_victory_points", {})
        for c in our_colors:
            if c in vps:
                vp_for.append(vps[c])
        lost = win is not None and win not in our_colors

        for d in g.get("decisions", []):
            if model is not None and model not in str(d.get("seat_agent", "")):
                continue
            at = d.get("action_type", "?")
            by_type[at] += 1
            by_phase[d.get("phase", "?")] += 1
            if d.get("fell_back"):
                fell_back[at] += 1
            # collect concrete decisions from games our model lost
            if lost and len(losing_samples) < max_samples and d.get("num_legal", 0) > 1:
                losing_samples.append({
                    "phase": d.get("phase"), "action_type": at,
                    "chosen": d.get("chosen"),
                    "num_legal": d.get("num_legal"),
                    "fell_back": bool(d.get("fell_back")),
                })

    total = sum(by_type.values()) or 1
    fb_total = sum(fell_back.values())
    report = {
        "transcript_dir": str(transcript_dir),
        "model": model or "all",
        "n_games": n_games,
        "truncated_rate": round(n_truncated / n_games, 3),
        "avg_vp": round(sum(vp_for) / len(vp_for), 2) if vp_for else None,
        "n_decisions": total,
        "fallback_rate": round(fb_total / total, 4),
        "fallback_by_type": dict(fell_back.most_common(8)),
        "action_mix": {k: round(v / total, 3) for k, v in by_type.most_common(12)},
        "phase_mix": {k: round(v / total, 3) for k, v in by_phase.most_common()},
        "losing_decision_samples": losing_samples,
    }
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("transcript_dir")
    p.add_argument("--model", default=None, help="seat_agent substring to filter on")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    rep = mine(a.transcript_dir, model=a.model)
    s = json.dumps(rep, indent=2)
    if a.out:
        Path(a.out).write_text(s)
        print("wrote", a.out)
    else:
        print(s)
