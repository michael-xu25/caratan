"""eval-protocol evaluator for the opening-placement GRPO / RFT job on Fireworks.

`eval-protocol create rft` discovers the `@evaluation_test` below, transforms the
dataset through `placement_dataset_adapter`, uploads both, and launches GRPO.
During training Fireworks generates rollouts on its GPUs and calls this evaluator
to score each completion; the score is the GRPO reward.

SELF-CONTAINED ON PURPOSE: this file imports only stdlib + eval_protocol, NO
catanatron / goldilocks_eval. At rollout-scoring time everything needed is already
in the dataset row's `ground_truth.spot_scores` (precomputed at generation time),
so the Fireworks evaluator build has zero Catan dependencies. The parse + reward
below are the exact mirror of `goldilocks_eval.prompting.parse_answer` and
`goldilocks_eval.placement_score` (normalized mode) — kept in lockstep by the
shared dataset and the no-API self-check at the bottom.

Reward = normalized closeness to the optimal legal spot (1.0 = best, 0.0 = worst).
No strategy is in the prompt (mechanics-only, answer-only) — judgment lives here.

Run the free, no-API logic check:  python training/test_placement_rft.py
Launch the job:                    see training/README.md  (eval-protocol create rft)
"""
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from eval_protocol import evaluation_test
from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata, Message

REWARD_MODE = "normalized"
DATASET = "data/placement_opening_train.trl.jsonl"
# The RFT base model; litellm route is only used for LOCAL testing (not RFT rollouts).
LOCAL_MODEL = "fireworks_ai/accounts/fireworks/models/qwen3-4b-instruct-2507"

_ANS = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _node_id_str(x: Any) -> Optional[str]:
    """Normalize 'node_27' / '27' -> 'node_27'. None if not an int-ish node."""
    s = str(x).strip()
    s = s[len("node_"):] if s.startswith("node_") else s
    try:
        return f"node_{int(s)}"
    except (ValueError, TypeError):
        return None


def _parse_answer(text: str) -> Optional[str]:
    m = _ANS.search(text or "")
    return _node_id_str(m.group(1)) if m else None


def _score(last_content: str, ground_truth: Dict[str, Any]) -> Tuple[float, str]:
    """Pure scoring: (reward, reason) from assistant text + row ground_truth.
    normalized = (chosen-worst)/(best-worst); 0.0 if absent/unparseable."""
    scores = (ground_truth or {}).get("spot_scores") or {}
    if not scores:
        return 0.0, "missing ground_truth.spot_scores"
    chosen = _parse_answer(last_content or "")
    if chosen is None:
        return 0.0, "no parseable <answer> node"
    totals = {_node_id_str(k): float(v) for k, v in scores.items()}
    if chosen not in totals:
        return 0.0, f"chose {chosen} (illegal/absent)"
    c, best, worst = totals[chosen], max(totals.values()), min(totals.values())
    r = 1.0 if best <= worst else (c - worst) / (best - worst)
    return r, f"chose {chosen} (gold {ground_truth.get('gold')}); {REWARD_MODE} {r:.3f}"


def placement_dataset_adapter(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """Map our reward-kit rows {id, prompt, ground_truth} -> EvaluationRow.
    `prompt` (system+user, mechanics + answer-only) becomes the rollout messages;
    `ground_truth` (node->score map + gold) is carried for the reward."""
    out: List[EvaluationRow] = []
    for r in rows:
        out.append(EvaluationRow(
            messages=[Message(role=m["role"], content=m["content"]) for m in r["prompt"]],
            ground_truth=r["ground_truth"],
            input_metadata=InputMetadata(row_id=str(r.get("id", ""))),
        ))
    return out


@evaluation_test(
    input_dataset=[DATASET],
    dataset_adapter=placement_dataset_adapter,
    completion_params=[{"model": LOCAL_MODEL, "temperature": 0.9, "max_tokens": 16}],
    mode="pointwise",
    passed_threshold=0.01,
)
def test_placement_opening(row: EvaluationRow) -> EvaluationRow:
    """Score one rollout: the model's last message vs the row's optimal spot."""
    last = row.messages[-1].content if row.messages else ""
    score, reason = _score(last or "", row.ground_truth or {})
    row.evaluation_result = EvaluateResult(score=score, is_score_valid=True, reason=reason)
    return row


if __name__ == "__main__":  # free logic check, no model / no API
    import json
    gt = json.loads(open(DATASET).readline())["ground_truth"]
    gold = gt["gold"]
    worst = min(gt["spot_scores"], key=gt["spot_scores"].get)
    cases = [("gold", f"<answer>{gold}</answer>", 1.0),
             ("worst", f"<answer>{worst}</answer>", 0.0),
             ("illegal", "<answer>node_999</answer>", 0.0),
             ("unparseable", "i pick the desert", 0.0)]
    ok = True
    for label, content, want in cases:
        got, reason = _score(content, gt)
        flag = "OK" if abs(got - want) < 1e-9 else "FAIL"
        ok &= flag == "OK"
        print(f"  {label:11} -> {got:.3f} (want {want})  {flag}  [{reason}]")
    print("ALL GOOD" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
