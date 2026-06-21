#!/usr/bin/env python
"""Play Settlers of Catan against the model, in the browser.

A small stdlib HTTP server that (a) serves the viewer static files and (b) runs
interactive games: YOU (human) pick from the legal moves; the MODEL (Qwen on
Fireworks) plays the other seat. It reuses the harness agent and the replay
viewer's board geometry + state snapshot, so the play board renders exactly like
the replay viewer.

    export FIREWORKS_API_KEY="$(scripts/fireworks_api_key.sh)"
    .venv/bin/python scripts/play_server.py --model "fireworks:accounts/.../deployments/<id>"
    # then open  http://localhost:8000/viewer/play.html

Endpoints (JSON):
  POST /api/new   {seed?, human?: "RED"|"BLUE", model?}  -> {view, model_moves}
  POST /api/move  {game_id, index}                       -> {view, model_moves}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.json import GameEncoder
from harness.agents import make_agent
from goldilocks_eval.prompt import render_action
from scripts.build_viewer_data import _static_board, _snapshot

GAMES: dict = {}
LOCK = threading.Lock()
MODEL_SPEC = "value"          # default opponent; --model overrides (e.g. the Qwen spec)


def _board(game) -> dict:
    g = json.loads(json.dumps(game, cls=GameEncoder))
    return _static_board(g)


def _advance(s: dict) -> list:
    """Play model turns + all forced moves until it's the human's real decision
    (or the game ends). Returns the model's moves taken (with reasoning)."""
    game, human, model = s["game"], s["human"], s["model"]
    moves, steps = [], 0
    while game.winning_color() is None and steps < 3000:
        cur = game.state.current_color()
        pa = game.playable_actions
        if cur == human and len(pa) > 1:
            break                                   # human's turn to choose
        action = pa[0] if cur == human else model.decide(game, pa)
        game.play_tick(decide_fn=lambda p, g, a, act=action: act)
        if cur != human and len(pa) > 1:            # a real model decision
            moves.append({"action": render_action(action),
                          "reasoning": model.pop_reasoning() or ""})
        steps += 1
    return moves


def _view(s: dict) -> dict:
    game, human = s["game"], s["human"]
    over = game.winning_color()
    your_turn = over is None and game.state.current_color() == human and len(game.playable_actions) > 1
    legal = ([{"i": i, "label": render_action(a)} for i, a in enumerate(game.playable_actions)]
             if your_turn else [])
    return {"game_id": s["id"], "board": s["board"], "players": [c.value for c in game.state.colors],
            "human": human.value, "turn": game.state.num_turns, "your_turn": your_turn,
            "legal": legal, "winner": over.value if over else None, **_snapshot(game.state)}


def _new_game(data: dict) -> dict:
    human = Color.RED if str(data.get("human", "RED")).upper() != "BLUE" else Color.BLUE
    model_color = Color.BLUE if human == Color.RED else Color.RED
    seed = int(data.get("seed") or random.randrange(1, 10**6))
    spec = data.get("model") or MODEL_SPEC
    agent = make_agent(spec, model_color, capture_reasoning=True)
    players = [agent if c == model_color else RandomPlayer(c) for c in (Color.RED, Color.BLUE)]
    game = Game(players, seed=seed)
    s = {"id": uuid.uuid4().hex[:8], "game": game, "human": human, "model": agent,
         "model_spec": spec, "seed": seed, "board": _board(game)}
    GAMES[s["id"]] = s
    moves = _advance(s)                              # model may move first
    return {"view": _view(s), "model_moves": moves, "seed": seed, "model": spec}


def _move(data: dict) -> dict:
    s = GAMES.get(data.get("game_id"))
    if s is None:
        return {"error": "unknown game_id"}
    game = s["game"]
    if game.winning_color() is not None or game.state.current_color() != s["human"]:
        return {"error": "not your turn", "view": _view(s)}
    idx = int(data["index"])
    pa = game.playable_actions
    if not (0 <= idx < len(pa)):
        return {"error": "illegal index", "view": _view(s)}
    action = pa[idx]
    game.play_tick(decide_fn=lambda p, g, a, act=action: act)
    moves = _advance(s)
    return {"view": _view(s), "model_moves": moves}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        try:
            with LOCK:
                if self.path == "/api/new":
                    return self._json(_new_game(data))
                if self.path == "/api/move":
                    return self._json(_move(data))
        except Exception as exc:  # never 500 the UI silently
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 200)
        return self._json({"error": "not found"}, 404)

    def do_GET(self):
        # static file server rooted at the repo (so /viewer/* and /transcripts/* work)
        rel = self.path.split("?", 1)[0].lstrip("/") or "viewer/play.html"
        path = (REPO / rel).resolve()
        if path.is_dir():                       # serve directory index (like http.server)
            path = path / "index.html"
        if not str(path).startswith(str(REPO)) or not path.is_file():
            return self.send_error(404)
        ctype = {".html": "text/html", ".json": "application/json", ".js": "text/javascript",
                 ".css": "text/css"}.get(path.suffix, "application/octet-stream")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global MODEL_SPEC
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="value",
                   help="opponent spec (e.g. fireworks:accounts/.../deployments/<id>, "
                        "claude:..., or a bot like 'value'). Default: value bot (no API key).")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    MODEL_SPEC = args.model
    print(f"Play server on http://localhost:{args.port}/viewer/play.html")
    print(f"opponent model: {MODEL_SPEC}")
    ThreadingHTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
