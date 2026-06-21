"""Champion labeling CLI — your irreplaceable input, front-loaded.

Reads an UNLABELED scenario JSONL (gold_action = null, as emitted by the
generator), renders each board with node ids + per-node production so you can
judge spots fast, and records your `gold_action` + `acceptable_actions`.

Resumable: re-run against the same --out and it skips already-labeled scenarios.
Writes after every label, so a crash/quit never loses work.

Usage:
    python -m goldilocks_eval.labeling_cli \
        data/placement_unlabeled.jsonl --out data/placement_labeled.jsonl

At each prompt enter a node id (`node_27` or `27`), `s` to skip, `q` to save+quit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from goldilocks_eval import prompting, schema


def _load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, records: List[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)  # atomic — never leave a half-written file


def _legal_set(scn: dict) -> Dict[str, str]:
    """Map each legal node's canonical id to itself (for validation)."""
    return {prompting.node_id_str(a): prompting.node_id_str(a)
            for a in scn["legal_actions"]}


def _prompt_node(msg: str, legal: Dict[str, str]) -> Optional[str]:
    """Return a canonical node id, '' for skip-field/none, or None for quit."""
    while True:
        raw = input(msg).strip()
        if raw.lower() == "q":
            return None
        if raw == "" or raw.lower() == "s":
            return ""
        try:
            cand = prompting.node_id_str(prompting.node_id_int(raw))
        except (ValueError, TypeError):
            print("  ! not a node id; try e.g. node_27 or 27")
            continue
        if cand not in legal:
            print(f"  ! {cand} is not a legal node here ({len(legal)} legal)")
            continue
        return cand


def label_one(scn: dict) -> Optional[dict]:
    """Interactively label one scenario. Returns the updated record, the
    original unchanged (skip), or None to quit."""
    legal = _legal_set(scn)
    print("\n" + "=" * 70)
    head = f"{scn.get('scenario_id', '?')}  env={scn.get('env', '?')}"
    if scn.get("pick_index") is not None:
        head += f"  pick {scn['pick_index']}/4"
    print(head)
    print(prompting.render_board(scn.get("serialized_state") or {},
                                 scn["legal_actions"]))

    gold = _prompt_node("\nGOLD node (q=save+quit, s=skip): ", legal)
    if gold is None:
        return None
    if gold == "":
        print("  (skipped)")
        return scn  # unchanged; stays unlabeled

    acceptable: List[str] = []
    while True:
        a = _prompt_node(f"  acceptable alt (blank=done) [{len(acceptable)} so far]: ", legal)
        if a is None:
            return None
        if a == "":
            break
        if a != gold and a not in acceptable:
            acceptable.append(a)

    # Pinned write-back path (schema-frozen UI -> generator direction).
    return schema.apply_label(scn, gold, acceptable)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="goldilocks_eval.labeling_cli",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="unlabeled scenario JSONL")
    p.add_argument("--out", required=True, help="labeled output JSONL (resumable)")
    args = p.parse_args(argv)

    if not sys.stdin.isatty():
        # Allow piped input for testing, but warn.
        print("(note: stdin is not a TTY — reading answers from the pipe)", file=sys.stderr)

    scenarios = _load_jsonl(args.input)
    # Resume: anything already labeled in --out keeps its labels.
    done: Dict[str, dict] = {
        r["scenario_id"]: r for r in _load_jsonl(args.out)
        if r.get("gold_action") is not None
    }
    records: List[dict] = []
    labeled_now = 0

    for scn in scenarios:
        sid = scn.get("scenario_id")
        if sid in done:
            records.append(done[sid])
            continue
        if scn.get("gold_action") is not None:  # already labeled in input
            records.append(scn)
            continue
        result = label_one(scn)
        if result is None:  # quit
            # carry the rest through unlabeled so the file stays complete
            records.append(scn)
            idx = scenarios.index(scn)
            records.extend(scenarios[idx + 1:])
            _write_jsonl(args.out, records)
            print(f"\nSaved. Labeled {labeled_now} this session -> {args.out}")
            return 0
        records.append(result)
        if result.get("gold_action") is not None:
            labeled_now += 1
            _write_jsonl(args.out, records + scenarios[scenarios.index(scn) + 1:])

    _write_jsonl(args.out, records)
    total = sum(1 for r in records if r.get("gold_action") is not None)
    print(f"\nDone. {labeled_now} labeled this session; "
          f"{total}/{len(records)} total labeled -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
