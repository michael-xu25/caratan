"""HUD environment for the Catan opening-placement task (GRPO via HUD/Tinker).

Self-contained grader (stdlib only) — mirrors goldilocks_eval.placement_score's
normalized reward. Each task carries the prebuilt mechanics-only prompt + the
precomputed spot_scores, so NO catanatron is needed at train time (this runs in
the isolated .venv-hud). Two-yield scenario: yield the prompt, receive the
model's answer, yield the reward.
"""
import re

from hud.environment import Environment
from hud.graders import EvaluationResult

env = Environment(name="catan-placement")

_ANS = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _node_id_str(x):
    s = str(x).strip()
    s = s[len("node_"):] if s.startswith("node_") else s
    try:
        return f"node_{int(s)}"
    except (ValueError, TypeError):
        return None


def _score(text, spot_scores):
    """normalized reward in [0,1]; 0.0 if unparseable/illegal. Mirror of the
    canonical goldilocks_eval reward (verified identical over 10k pairs)."""
    if not spot_scores:
        return 0.0, "no spot_scores"
    m = _ANS.search(text or "")
    if not m:
        return 0.0, "unparseable"
    chosen = _node_id_str(m.group(1))
    totals = {_node_id_str(k): float(v) for k, v in spot_scores.items()}
    if chosen not in totals:
        return 0.0, f"illegal {chosen}"
    c, best, worst = totals[chosen], max(totals.values()), min(totals.values())
    r = 1.0 if best <= worst else (c - worst) / (best - worst)
    return r, f"{chosen} -> {r:.3f}"


@env.template()
async def placement(prompt: str, spot_scores: dict, gold: str):
    """Ask for one opening settlement; reward = normalized closeness to optimal."""
    answer = yield prompt
    text = answer if isinstance(answer, str) else str(answer)
    reward, reason = _score(text, spot_scores)
    yield EvaluationResult(reward=reward, content=text[:160],
                           info={"gold": gold, "reason": reason})
