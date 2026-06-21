#!/usr/bin/env python
"""Scan built replay files and write the viewer's run picker manifest.

The viewer (`viewer/index.html`) loads one game at a time. To let the user pick
among different sample runs, it reads a manifest that lists every built
`*.view.json`, grouped by the directory it lives in (one directory == one "run",
the same way `transcripts/<run>/seedN_{norm,swap}.view.json` is already
organized -- mirroring the `data/` convention of a directory of artifacts plus a
small index describing them).

    .venv/bin/python scripts/build_viewer_index.py            # scan transcripts/ -> viewer/runs.json
    .venv/bin/python scripts/build_viewer_index.py transcripts/sample

Directories whose name starts with "_" are treated as internal/scratch and
skipped. Paths in the manifest are server-root-absolute ("/transcripts/...") so
they match the viewer's existing `?data=/...` convention and resolve regardless
of where the viewer page is served from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _game_entry(view_path: Path) -> dict:
    """Read a .view.json's meta into a compact picker entry."""
    meta = json.loads(view_path.read_text()).get("meta", {})
    rel = view_path.relative_to(REPO).as_posix()
    return {
        "view": "/" + rel,                       # server-root-absolute, matches ?data=
        "label": meta.get("label") or view_path.stem.replace(".view", ""),
        "seed": meta.get("seed"),
        "winner": meta.get("winner"),
        "final_vp": meta.get("final_vp", {}),
        "num_steps": meta.get("num_steps", 0),
        "seats": meta.get("seats", {}),
    }


def build_index(root: Path) -> dict:
    runs: dict[str, list] = {}
    for view_path in sorted(root.rglob("*.view.json")):
        # Skip internal/scratch dirs (any path component starting with "_").
        rel_parts = view_path.relative_to(REPO).parts
        if any(p.startswith("_") for p in rel_parts[:-1]):
            continue
        run_dir = view_path.parent.relative_to(REPO).as_posix()
        runs.setdefault(run_dir, []).append(_game_entry(view_path))

    run_list = []
    for run_dir in sorted(runs):
        # norm before swap, then by seed/label for a stable, readable order.
        games = sorted(runs[run_dir],
                       key=lambda g: (g["seed"] if g["seed"] is not None else 0,
                                      "swap" in g["label"], g["label"]))
        run_list.append({
            "name": run_dir.split("/", 1)[1] if "/" in run_dir else run_dir,
            "path": run_dir,
            "games": games,
        })
    return {"runs": run_list, "count": sum(len(r["games"]) for r in run_list)}


def main(argv) -> int:
    root = Path(argv[0]) if argv else REPO / "transcripts"
    if not root.is_absolute():
        root = REPO / root
    if not root.exists():
        print(f"no such directory: {root}", file=sys.stderr)
        return 1
    index = build_index(root)
    out = REPO / "viewer" / "runs.json"
    out.write_text(json.dumps(index, indent=2))
    print(f"{out.relative_to(REPO)}: {len(index['runs'])} runs, "
          f"{index['count']} games (scanned {root.relative_to(REPO)}/)")
    for r in index["runs"]:
        print(f"  - {r['path']}  ({len(r['games'])} games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
