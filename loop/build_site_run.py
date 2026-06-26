"""Refresh the website to feature a head-to-head run.

After a matchup run + build_viewer_data, this:
  - picks a balanced set of games (trained wins / base wins / draws),
  - copies their .view.json into web/public/transcripts/<run>/,
  - writes web/public/viewer/runs.json to feature ONLY the new run.

    .venv/bin/python loop/build_site_run.py transcripts/trained-vs-base --run trained-vs-base --n 12
"""
import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC_LABEL = {"modal:trained": "Trained", "modal:base": "Base (Qwen3-8B)"}


def _entry(raw_path, view_web_path, view_local):
    raw = json.loads(Path(raw_path).read_text())
    seats_spec = raw.get("seats", {})
    seats = {c: SPEC_LABEL.get(s, s) for c, s in seats_spec.items()}
    win_c = raw.get("winning_color")
    vps = raw.get("final_victory_points", {})
    winner_model = seats.get(win_c) if win_c else None
    nsteps = len(json.loads(Path(view_local).read_text()).get("steps", []))
    return {
        "view": view_web_path, "label": Path(raw_path).stem, "seed": raw.get("seed"),
        "winner": win_c, "winner_model": winner_model,
        "final_vp": vps, "num_steps": nsteps, "seats": seats,
        "truncated": bool(raw.get("truncated")),
    }


def main(run_dir, run_name, n):
    run_dir = Path(run_dir)
    dest = REPO / "web" / "public" / "transcripts" / run_name
    dest.mkdir(parents=True, exist_ok=True)

    # all games with a built view
    games = []
    for raw in sorted(run_dir.glob("*.json")):
        if raw.name == "summary.txt" or raw.suffix != ".json":
            continue
        view_local = raw.with_suffix(".view.json")
        if not view_local.exists():
            continue
        games.append((raw, view_local))

    # classify by outcome for a balanced feature set
    recs = []
    for raw, view_local in games:
        r = json.loads(raw.read_text())
        seats = {c: SPEC_LABEL.get(s, s) for c, s in r.get("seats", {}).items()}
        wm = seats.get(r.get("winning_color")) if r.get("winning_color") else None
        recs.append((raw, view_local, wm))
    trained = [x for x in recs if x[2] == "Trained"]
    base = [x for x in recs if x[2] == "Base (Qwen3-8B)"]
    draws = [x for x in recs if x[2] is None]

    # feature mostly trained wins, a couple base wins, a draw if any
    pick = trained[: max(1, n - 4)] + base[:3] + draws[:1]
    pick = pick[:n] if len(pick) >= n else pick + [g for g in recs if g not in pick][: n - len(pick)]

    entries = []
    for raw, view_local, _ in pick:
        shutil.copy(view_local, dest / view_local.name)
        web_path = f"/transcripts/{run_name}/{view_local.name}"
        entries.append(_entry(raw, web_path, view_local))

    runs = {"runs": [{
        "name": "Trained model vs base model",
        "kind": "trained-vs-base",
        "path": f"transcripts/{run_name}",
        "games": entries,
    }]}
    out = REPO / "web" / "public" / "viewer" / "runs.json"
    out.write_text(json.dumps(runs, indent=2))
    print(f"featured {len(entries)} games -> {out}")
    print(f"  trained wins available: {len(trained)} | base wins: {len(base)} | draws: {len(draws)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--run", default="trained-vs-base")
    p.add_argument("--n", type=int, default=12)
    a = p.parse_args()
    main(a.run_dir, a.run, a.n)
