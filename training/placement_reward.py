"""reward-kit reward function for the opening-placement env (GRPO / Fireworks RFT).

The model is shown a mechanics-only placement prompt and replies with a chosen
node (`<answer>node_N</answer>`). This scores that choice against the championship
scoring read from the dataset row's `ground_truth` — 1.0 = the optimal legal spot,
0.0 = the worst — with no strategy in the prompt (judgment lives here, in the
reward; see words-vs-rl.md / placement-env-design.md).

Dataset rows (from `python -m goldilocks_eval.placement_env traindata`):
    {"id": ..., "prompt": [system, user],
     "ground_truth": {"spot_scores": {"node_N": score, ...}, "gold": "node_K"}}

Entry point for reward-kit: `training.placement_reward:placement_reward_fn`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goldilocks_eval.placement_score import reward_from_scores
from goldilocks_eval.prompting import parse_answer

# Matches the approved reward (see placement-env-design.md). Change here to A/B.
REWARD_MODE = "normalized"

try:  # real reward-kit when training; shim so this module imports for unit tests
    from reward_kit import reward_function
    from reward_kit.models import EvaluateResult, MetricResult
except Exception:  # pragma: no cover
    def reward_function(fn):
        return fn

    class MetricResult:  # minimal stand-ins matching reward-kit's shape
        def __init__(self, score, success, reason):
            self.score, self.success, self.reason = score, success, reason

    class EvaluateResult:
        def __init__(self, score, reason, metrics=None):
            self.score, self.reason, self.metrics = score, reason, metrics or {}


def _last_content(messages) -> str:
    m = messages[-1]
    return m["content"] if isinstance(m, dict) else getattr(m, "content", str(m))


@reward_function
def placement_reward_fn(messages, ground_truth=None, **kwargs):
    """messages = [system, user, assistant]; assistant is the model's reply."""
    scores = (ground_truth or {}).get("spot_scores") or {}
    if not scores:
        return EvaluateResult(score=0.0, reason="missing ground_truth.spot_scores", metrics={})
    chosen = parse_answer(_last_content(messages))
    if chosen is None:
        return EvaluateResult(
            score=0.0, reason="no parseable <answer> node in the reply",
            metrics={"placement": MetricResult(score=0.0, success=False, reason="unparseable")})
    r = reward_from_scores(chosen, scores, REWARD_MODE)
    gold = (ground_truth or {}).get("gold")
    return EvaluateResult(
        score=r,
        reason=f"chose {chosen} (gold {gold}); {REWARD_MODE} reward {r:.3f}",
        metrics={"placement_normalized": MetricResult(
            score=r, success=(r >= 0.999), reason=f"{r:.3f} of optimal")})
