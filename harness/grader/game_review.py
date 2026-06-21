"""Game-level failure review — why did the player fail to WIN this game?

Per-decision grading finds few clear blunders for a mostly-defensible model; its
real weaknesses are cumulative/strategic (stalls to the turn cap, never converts a
lead). This pass reads each finished game whole and names the STRATEGIC failure
modes that cost it, then ranks them across a run. One LLM call per game (single
grader by default — fast, small output), complementing the per-decision hybrid.

Modes are a small fixed vocab so the output aggregates into a clean table and can
tag env-gen, with `other` as the escape.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from harness.grader.prompts import _timeline

# Fixed strategic-failure vocab (cumulative, whole-game — NOT single-move).
STRATEGIC_MODES = {
    "weak_production_engine": "never built enough production (pips/cities) to fuel a win",
    "no_city_upgrades": "stayed on settlements; didn't upgrade to cities for VP + double production",
    "ignored_dev_cards": "underused dev cards / never contested Largest Army",
    "ignored_longest_road": "never contested Longest Road when it was attainable",
    "over_trading": "churned maritime trades without converting to builds/VP",
    "passive_robber": "didn't use robber/knights to deny the opponent's engine",
    "expansion_stall": "stopped expanding / got boxed in, capping growth",
    "no_path_to_10": "no coherent route to 10 VP; failed to convert position/lead into a win",
    "other": "a strategic failure not covered above (explain in why)",
}


def _modes_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in STRATEGIC_MODES.items())


SYSTEM_REVIEW = (
    "You are reviewing a FINISHED 1v1 Settlers of Catan self-play game (both seats are "
    "the same model). Identify the STRATEGIC, cumulative reasons the player failed to "
    "win — patterns across the whole game (engine-building, VP conversion, board "
    "control), NOT single-move blunders. Use ONLY the failure-mode IDs you are given.\n\n"
    "Return ONLY this JSON:\n"
    '{"failures": [{"mode": "<id>", "why": "<= 1 sentence"}], "summary": "<= 1 sentence"}\n'
    "List the top 1-3 modes that actually apply (fewer is fine). Never invent IDs."
)


def build_review_prompt(transcript: dict) -> str:
    fvp = transcript.get("final_victory_points", {})
    winner = transcript.get("winning_color")
    focus = ("why did NEITHER player reach 10 VP (it stalled to the turn cap)"
             if not winner else
             "why did the LOSING player fail to reach 10 VP before the winner")
    return (
        f"OUTCOME: final VP {fvp}, winner {winner or 'draw (turn cap)'}.\n"
        f"GAME TIMELINE (meaningful moves with running VP):\n{_timeline(transcript)}\n\n"
        f"Strategic failure modes (pick from these IDs):\n{_modes_block()}\n\n"
        f"Identify {focus}. Top 1-3 STRATEGIC/cumulative modes only. Return the JSON."
    )


def _parse(text: str) -> dict | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`"); s = s[s.find("{"):] if "{" in s else s
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        obj = json.loads(s[a:b + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    fails = [f for f in obj.get("failures", [])
             if isinstance(f, dict) and f.get("mode") in STRATEGIC_MODES]
    return {"failures": fails, "summary": str(obj.get("summary", ""))[:200]}


def review_game(backend, transcript: dict) -> dict:
    gid = transcript.get("label", "game")
    try:
        obj = _parse(backend.complete(SYSTEM_REVIEW, build_review_prompt(transcript)))
    except Exception:
        obj = None
    return {"game_id": gid, "winner": transcript.get("winning_color"),
            "failures": (obj or {}).get("failures", []),
            "summary": (obj or {}).get("summary", ""), "ok": obj is not None}


def review_run(transcripts: list[dict], backend, concurrency: int = 16) -> list[dict]:
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(lambda t: review_game(backend, t), transcripts))


def aggregate_reviews(reviews: list[dict]) -> dict:
    """Rank strategic failure modes by how many games they appear in."""
    n_games = len(reviews)
    by_mode = defaultdict(lambda: {"games": 0, "examples": [], "why": []})
    for r in reviews:
        for f in {x["mode"]: x for x in r["failures"]}.values():  # dedup per game
            m = by_mode[f["mode"]]
            m["games"] += 1
            if len(m["examples"]) < 4:
                m["examples"].append(r["game_id"])
            if f.get("why") and len(m["why"]) < 3:
                m["why"].append(f["why"])
    rows = [{"mode": k, "games": v["games"],
             "rate": round(v["games"] / n_games, 3) if n_games else 0,
             "examples": v["examples"], "sample_why": v["why"]}
            for k, v in by_mode.items()]
    rows.sort(key=lambda r: -r["games"])
    return {"n_games": n_games, "failure_modes": rows}
