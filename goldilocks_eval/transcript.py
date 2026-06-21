"""Write per-game transcripts: machine-readable JSON + a human-readable log."""
from __future__ import annotations

import dataclasses
import json
import os
from typing import List, Optional

from catanatron import Color, Game
from catanatron.json import GameEncoder

from goldilocks_eval.agents.base import LLMPlayer
from goldilocks_eval.prompt import render_action


def _decisions_payload(players) -> dict:
    out = {}
    for p in players:
        if isinstance(p, LLMPlayer):
            out[p.color.value] = [dataclasses.asdict(d) for d in p.decisions]
    return out


def write_json(path: str, game: Game, meta: dict, players) -> None:
    payload = {
        "meta": meta,
        "decisions": _decisions_payload(players),
        # Full Catanatron state via the library's own GameEncoder.
        "game": json.loads(json.dumps(game, cls=GameEncoder)),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_human(path: str, game: Game, meta: dict, players) -> None:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append(f"GAME {meta['game_id']}  (seed={meta['seed']})")
    seats = "  vs  ".join(f"{c}={a}" for c, a in meta["seats"].items())
    lines.append(f"Seats: {seats}")
    lines.append(f"Winner: {meta['winner']}  ({meta['winner_agent']})")
    lines.append(f"Turns: {meta['turns']}")
    lines.append("=" * 70)

    # Final standing.
    lines.append("\nFinal victory points:")
    for color_str, vp in meta["victory_points"].items():
        lines.append(f"  {color_str}: {vp}")

    # LLM reasoning trace (the interesting part — each model decision).
    for p in players:
        if isinstance(p, LLMPlayer) and p.decisions:
            lines.append(f"\n--- {p.color.value} ({meta['seats'][p.color.value]}) "
                         f"decisions: {len(p.decisions)} ---")
            for i, d in enumerate(p.decisions):
                flag = " [FELL BACK]" if d.fell_back else ""
                lines.append(
                    f"  #{i} turn{d.turn} ({d.num_options} options, "
                    f"{d.latency_ms}ms){flag}: chose `{d.chosen}`"
                )
                if d.reasoning:
                    lines.append(f"        reason: {d.reasoning}")

    # Full action log (every decision, both players).
    lines.append("\n--- Action log ---")
    for i, rec in enumerate(game.state.action_records):
        lines.append(f"  {i:4d} {rec.action.color.value:7} {render_action(rec.action)}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_transcripts(out_dir: str, game: Game, meta: dict, players) -> None:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"game_{meta['game_id']}")
    write_json(base + ".json", game, meta, players)
    write_human(base + ".txt", game, meta, players)
