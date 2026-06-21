"""HUD environment for the Catan maritime-trade task (GRPO via HUD/Tinker).

Targets the over-trading weakness: trade only when it's productive. The model
picks an action by index from a numbered menu (trades + builds + end-turn);
reward = the tunable maritime score of that choice (enabling trade > no-trade >
churn). Self-contained (stdlib only): the per-option components are baked into
each task's ground_truth, so no catanatron at train time. Pure reward mirrors
goldilocks_eval.maritime_score.maritime_reward exactly.
"""
import json
import re

from hud.environment import Environment
from hud.graders import EvaluationResult

env = Environment(name="catan-maritime")

# Mirror of goldilocks_eval.maritime_score (pure combine; tunable weights ride on
# each row's ground_truth so they stay in sync with generation).
BUILD_VALUE = {"BUILD_CITY": 1.0, "BUILD_SETTLEMENT": 0.85,
               "BUY_DEVELOPMENT_CARD": 0.5, "BUILD_ROAD": 0.4}
REWARD_CLAMP = (-1.1, 1.2)
INVALID_REWARD = -1.0   # unparseable / out-of-range index: penalize, don't reward garbage

_ACTION = re.compile(r'"action"\s*:\s*(\d+)')
_ANS = re.compile(r"<answer>\s*(\d+)\s*</answer>", re.IGNORECASE)
_INT = re.compile(r"\d+")


def _parse_index(text, n):
    """Extract the chosen action index from {"action": N} / <answer>N</answer> /
    last bare integer. None if nothing in [0, n)."""
    for rx in (_ACTION, _ANS):
        m = rx.search(text or "")
        if m:
            i = int(m.group(1))
            return i if 0 <= i < n else None
    ints = _INT.findall(text or "")
    for tok in reversed(ints):              # last in-range integer
        i = int(tok)
        if 0 <= i < n:
            return i
    return None


def _maritime_reward(c, w):
    enables = c.get("enables") or []
    if enables:
        r = w["enable"] * max(BUILD_VALUE.get(b, 0.0) for b in enables)
    elif c.get("progresses"):
        r = w["progress"]
    else:
        r = -w["churn"]
    r -= w["scarcity"] * float(c.get("gives_scarce", 0.0))
    lo, hi = REWARD_CLAMP
    return max(lo, min(hi, r))


def _score(text, gt):
    legal = gt["legal_actions"]
    idx = _parse_index(text, len(legal))
    if idx is None:
        return INVALID_REWARD, "invalid/illegal index"
    chosen = legal[idx]
    topt = gt["trade_options"]
    if chosen in topt:
        r = _maritime_reward(topt[chosen], gt["weights"])
        return r, f"[{idx}] trade r={r:+.2f} {chosen[:40]}"
    return float(gt.get("no_trade_reward", 0.0)), f"[{idx}] no-trade {chosen[:40]}"


@env.template()
async def maritime(prompt: str, legal_actions: list, trade_options: dict,
                   no_trade_reward: float, weights: dict):
    answer = yield prompt
    text = answer if isinstance(answer, str) else str(answer)
    gt = {"legal_actions": legal_actions, "trade_options": trade_options,
          "no_trade_reward": no_trade_reward, "weights": weights}
    reward, reason = _score(text, gt)
    yield EvaluationResult(reward=reward, content=text[:160], info={"reason": reason})


if __name__ == "__main__":  # no-API logic check
    import sys
    rows = [json.loads(l) for l in open("data/maritime_trade_train.trl.jsonl")]
    gt = rows[0]["ground_truth"]
    n = len(gt["legal_actions"])
    for label, txt in [("valid idx0", '{"action": 0}'), ("answer-tag", "<answer>1</answer>"),
                       ("out-of-range", '{"action": 999}'), ("garbage", "i pick none")]:
        print(f"  {label:13} -> {_score(txt, gt)}")
    print("ALL GOOD")
