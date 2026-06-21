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
import os
import random
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from catanatron.game import Game
from catanatron.models.enums import ActionType
from catanatron.models.player import Color, RandomPlayer
from catanatron.json import GameEncoder
from harness.agents import make_agent
from goldilocks_eval.prompt import render_action
from scripts.build_viewer_data import _static_board, _snapshot

GAMES: dict = {}
LOCK = threading.Lock()
MODEL_SPEC = "value"          # default opponent; --model overrides (e.g. the Qwen spec)

# Interactive demo games are persisted HERE — deliberately SEPARATE from the
# recorded transcripts/ (which stay read-only). This is temp demo storage: the
# full serialized state (the context the model needs to move) + the move history
# (incl. the initial placements) per game, rewritten on every new/fork/move.
DEMO_DIR = REPO / "demo_games"


def _persist(s: dict) -> None:
    """Best-effort: dump this demo game's state + history to demo_games/<id>.json.
    Never raises — persistence must not break the live demo."""
    try:
        DEMO_DIR.mkdir(exist_ok=True)
        game = s["game"]
        won = game.winning_color()
        data = {
            "id": s["id"], "seed": s["seed"], "human": s["human"].value,
            "model": s["model_spec"], "turn": game.state.num_turns,
            "winner": won.value if won else None,
            "forked_from": s.get("forked_from"), "fork_ply": s.get("fork_ply"),
            "history": s.get("history", []),                 # human + model moves (incl. placements)
            "state": json.loads(json.dumps(game, cls=GameEncoder)),  # full context the model sees
        }
        (DEMO_DIR / f"{s['id']}.json").write_text(json.dumps(data))
    except Exception:
        pass


def _board(game) -> dict:
    g = json.loads(json.dumps(game, cls=GameEncoder))
    return _static_board(g)


def _advance(s: dict) -> list:
    """Play model turns + all forced moves until it's the human's real decision
    (or the game ends). Returns the model's moves taken (with reasoning)."""
    game, human, model = s["game"], s["human"], s["model"]
    hist = s.setdefault("history", [])
    moves, steps = [], 0
    while game.winning_color() is None and steps < 3000:
        cur = game.state.current_color()
        pa = game.playable_actions
        if cur == human and len(pa) > 1:
            break                                   # human's turn to choose
        action = pa[0] if cur == human else model.decide(game, pa)
        game.play_tick(decide_fn=lambda p, g, a, act=action: act)
        if cur != human and len(pa) > 1:            # a real model decision
            entry = {"player": cur.value, "action": render_action(action),
                     "reasoning": model.pop_reasoning() or ""}
            moves.append(entry); hist.append(entry)
        steps += 1
    return moves


def _jsonable(v):
    """Serialize an Action.value (node id, edge, robber coord+victim, trade tuple,
    resource) to JSON so the UI can map it onto board elements."""
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    if isinstance(v, Color):
        return v.value
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    return str(v)


def _rolls(game) -> list:
    """Every dice roll so far (newest last): who rolled, the two dice, the total.
    Read straight from the recorded ROLL actions (their value is the (d1, d2) pair)."""
    out = []
    for rec in game.state.action_records:
        act = rec.action if hasattr(rec, "action") else rec[0]
        if act.action_type == ActionType.ROLL and act.value:
            d = list(act.value)
            out.append({"color": act.color.value, "dice": d, "total": sum(d)})
    return out


def _view(s: dict) -> dict:
    game, human = s["game"], s["human"]
    over = game.winning_color()
    your_turn = over is None and game.state.current_color() == human and len(game.playable_actions) > 1
    legal = ([{"i": i, "label": render_action(a),
               "type": a.action_type.value, "value": _jsonable(a.value)}
              for i, a in enumerate(game.playable_actions)]
             if your_turn else [])
    model_color = next((c.value for c in game.state.colors if c != human), None)
    return {"game_id": s["id"], "board": s["board"], "players": [c.value for c in game.state.colors],
            "human": human.value, "model_color": model_color,
            "turn": game.state.num_turns, "your_turn": your_turn,
            "legal": legal, "rolls": _rolls(game),
            "winner": over.value if over else None, **_snapshot(game.state)}


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
    _persist(s)                                      # store demo state (incl. placements)
    return {"view": _view(s), "model_moves": moves, "seed": seed, "model": spec}


def _to_action(ra):
    """Rebuild an Action from a serialized [color, type, value] record."""
    from catanatron.models.enums import Action, ActionType
    atype = ActionType(ra[1]); v = ra[2]
    if v is None:
        value = None
    elif atype == ActionType.MOVE_ROBBER:
        coord, victim = v; value = (tuple(coord), Color[victim] if victim else None)
    elif isinstance(v, list):
        value = tuple(v)
    else:
        value = v
    return Action(Color[ra[0]], atype, value)


def _fork(data: dict) -> dict:
    """Start an interactive game from a recorded transcript at a given ply: replay
    the recorded actions [0..ply) deterministically (recorded results -> no RNG),
    then the HUMAN takes over whoever is on turn and the MODEL plays the other seat."""
    import catanatron.game as _cg
    from catanatron.models.enums import ActionRecord
    rel = str(data.get("transcript", "")).lstrip("/").replace(".view.json", ".json")
    path = (REPO / rel).resolve()
    if not str(path).startswith(str(REPO)) or not path.is_file():
        return {"error": f"transcript not found: {rel}"}
    d = json.loads(path.read_text()); g = d["game"]; records = g["action_records"]
    ply = max(0, min(int(data.get("ply", 0)), len(records)))
    spec = data.get("model") or MODEL_SPEC
    _cg.TURNS_LIMIT = max(g.get("num_turns_cap", 400), len(records) + 5)
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
                seed=d["seed"], vps_to_win=10)
    for rec in records[:ply]:                       # replay up to the fork point
        action = _to_action(rec[0])
        result = tuple(rec[1]) if isinstance(rec[1], list) else rec[1]
        game.execute(action, validate_action=False,
                     action_record=ActionRecord(action=action, result=result))
    human = game.state.current_color()              # you take over whoever's on turn
    model_color = Color.BLUE if human == Color.RED else Color.RED
    agent = make_agent(spec, model_color, capture_reasoning=True)
    s = {"id": uuid.uuid4().hex[:8], "game": game, "human": human, "model": agent,
         "model_spec": spec, "seed": d["seed"], "board": _board(game),
         "forked_from": rel, "fork_ply": ply}
    GAMES[s["id"]] = s
    moves = _advance(s)                             # skip forced moves to your decision
    _persist(s)
    return {"view": _view(s), "model_moves": moves, "seed": d["seed"], "model": spec,
            "forked_from": rel, "fork_ply": ply}


def _state(data: dict) -> dict:
    s = GAMES.get(data.get("game_id"))
    if s is None:
        return {"error": "unknown game_id"}
    return {"view": _view(s), "model_moves": [], "seed": s["seed"], "model": s["model_spec"]}


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
    s.setdefault("history", []).append({"player": s["human"].value,
                                        "action": render_action(action)})
    game.play_tick(decide_fn=lambda p, g, a, act=action: act)
    moves = _advance(s)
    _persist(s)
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
                if self.path == "/api/fork":
                    return self._json(_fork(data))
                if self.path == "/api/state":
                    return self._json(_state(data))
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


def _default_opponent() -> str:
    """Opponent default: the live Fireworks deployment from the env (FIREWORKS_MODEL,
    set by `.env`). Deployment ids ROTATE — reading them from .env keeps this in sync
    instead of hardcoding a stale id. Falls back to the value bot (no API key needed)."""
    fw = os.environ.get("FIREWORKS_MODEL")
    if fw and os.environ.get("FIREWORKS_API_KEY"):
        return f"fireworks:{fw}"
    return "value"


def main():
    global MODEL_SPEC
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None,
                   help="opponent spec (e.g. fireworks:accounts/.../deployments/<id>, "
                        "claude:..., or a bot like 'value'). Default: fireworks:$FIREWORKS_MODEL "
                        "from .env if set, else the value bot.")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    MODEL_SPEC = args.model or _default_opponent()
    print(f"Play server on http://localhost:{args.port}/viewer/play.html")
    print(f"opponent model: {MODEL_SPEC}")
    ThreadingHTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
