"""Publish loop progress -> web/public/data/loop.json for caratan.vercel.app.

Non-destructive: writes a SEPARATE loop.json (the live self-improvement feed), so
the curated demo results.json is never clobbered. The site's live section reads
this. Triggering the actual Vercel rebuild (git push or deploy hook) is the
orchestrator's job; this only produces the JSON.

    .venv-modal/bin/python loop/publish.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = Path(__file__).resolve().parent / "state" / "registry.json"
OUT = ROOT / "web" / "public" / "data" / "loop.json"


def build(registry):
    rounds = registry.get("rounds", [])
    # per-env held-out reward curve = the `after` of each round for that env
    curves = {}
    for r in rounds:
        curves.setdefault(r["env"], []).append({
            "round": len([x for x in rounds[:rounds.index(r) + 1] if x["env"] == r["env"]]),
            "before": r["before"], "after": r["after"],
            "promoted": r["promoted"],
        })
    promotions = sum(1 for r in rounds if r["promoted"])
    by_env = {}
    for env, pts in curves.items():
        first = pts[0]["before"]
        best = max(p["after"] for p in pts)
        by_env[env] = {"start": round(first, 4), "best": round(best, 4),
                       "rounds": len(pts)}
    return {
        "updated_rounds": len(rounds),
        "promotions": promotions,
        "best_by_env": registry.get("best", {}),
        "summary_by_env": by_env,
        "curves": curves,
        "recent": rounds[-12:],
    }


def main():
    if not REGISTRY.exists():
        print("no registry yet:", REGISTRY)
        return
    reg = json.loads(REGISTRY.read_text())
    data = build(reg)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))
    print(f"wrote {OUT} ({data['updated_rounds']} rounds, {data['promotions']} promotions)")


if __name__ == "__main__":
    main()
