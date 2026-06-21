"""HUD environment for the Catan build-decision task (GRPO via HUD/Tinker).

Targets the under-building / hoarding weakness: make affordable productive builds
instead of passing. The model picks an action by index; reward = build value of
the chosen build, or a hoard penalty for passing when a strong build was
affordable. Self-contained (stdlib only); pure reward mirrors
goldilocks_eval.build_score (build_reward + hoard_penalty) exactly.
"""
import json
import re

from hud.environment import Environment
from hud.graders import EvaluationResult

env = Environment(name="catan-build")

ROAD_DISCOUNT = 0.6
DEV_VALUE = 1.0
HOARD_OK = 0.3
REWARD_CLAMP = (-1.5, 2.0)
INVALID_REWARD = -1.5   # unparseable / out-of-range index: penalize

_ACTION = re.compile(r'"action"\s*:\s*(\d+)')
_ANS = re.compile(r"<answer>\s*(\d+)\s*</answer>", re.IGNORECASE)
_INT = re.compile(r"\d+")


def _parse_index(text, n):
    for rx in (_ACTION, _ANS):
        m = rx.search(text or "")
        if m:
            i = int(m.group(1))
            return i if 0 <= i < n else None
    for tok in reversed(_INT.findall(text or "")):
        i = int(tok)
        if 0 <= i < n:
            return i
    return None


def _clamp(x):
    lo, hi = REWARD_CLAMP
    return max(lo, min(hi, x))


def _build_reward(c, w):
    k = c["kind"]
    if k == "settlement":
        r = w["prod"] * c["pip_norm"] + w["div"] * c["diversity"] + w["vp"] * c["vp"]
    elif k == "city":
        r = w["prod"] * c["pip_norm"] + w["vp"] * c["vp"]
    elif k == "road":
        opened = (w["prod"] * c["opens_pip_norm"] + w["div"] * c["opens_diversity"]
                  if c.get("opens_node") is not None else 0.0)
        r = w["road"] * ROAD_DISCOUNT * opened
    elif k == "dev":
        r = w["dev"] * DEV_VALUE
    else:
        r = 0.0
    return _clamp(r)


def _hoard_penalty(best_value, w):
    return _clamp(-w["hoard"] * max(0.0, best_value - HOARD_OK))


def _score(text, gt):
    legal = gt["legal_actions"]
    idx = _parse_index(text, len(legal))
    if idx is None:
        return INVALID_REWARD, "invalid/illegal index"
    chosen = legal[idx]
    opts = gt["build_options"]
    if chosen in opts:
        r = _build_reward(opts[chosen], gt["weights"])
        return r, f"[{idx}] build r={r:+.2f} {chosen[:40]}"
    r = _hoard_penalty(gt["best_value"], gt["weights"])
    return r, f"[{idx}] pass/hoard r={r:+.2f} (forgone best {gt['best_value']:.2f})"


@env.template()
async def build(prompt: str, legal_actions: list, build_options: dict,
                best_value: float, weights: dict):
    answer = yield prompt
    text = answer if isinstance(answer, str) else str(answer)
    gt = {"legal_actions": legal_actions, "build_options": build_options,
          "best_value": best_value, "weights": weights}
    reward, reason = _score(text, gt)
    yield EvaluationResult(reward=reward, content=text[:160], info={"reason": reason})


if __name__ == "__main__":
    rows = [json.loads(l) for l in open("data/build_train.trl.jsonl")]
    gt = rows[0]["ground_truth"]
    for label, txt in [("idx0", '{"action": 0}'), ("answer-tag", "<answer>1</answer>"),
                       ("out-of-range", '{"action": 999}'), ("garbage", "none")]:
        print(f"  {label:13} -> {_score(txt, gt)}")
    print("ALL GOOD")
