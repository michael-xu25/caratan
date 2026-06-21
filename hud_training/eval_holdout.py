"""Held-out before/after eval over the HUD gateway (Tinker models).

Queries a model on held-out scenarios (the grader_games split, disjoint from
training) and scores each with the SAME self-contained grader used in training,
so the before/after is honest. Compares the base Qwen3-8B against the trained
fork. Deterministic (temperature 0).

    set -a; source ../.env; set +a
    ../.venv-hud/bin/python eval_holdout.py        # all 3 envs, base vs trained
"""
import concurrent.futures as cf
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from catan_placement_env import _score as placement_score, _node_id_str, _ANS as PLC_ANS
from catan_maritime_env import _score as maritime_score
from catan_build_env import _score as build_score

GATEWAY = "https://inference.beta.hud.ai/v1/chat/completions"
KEY = os.environ["HUD_API_KEY"]
BASE = "Qwen/Qwen3-8B"          # before (un-trained)
TRAINED = "catan-grpo-q8b"      # after (the warm-start chain fork)

ENVS = {
    "placement": ("data/placement_eval.trl.jsonl", placement_score),
    "maritime": ("data/maritime_eval.trl.jsonl", maritime_score),
    "build": ("data/build_eval.trl.jsonl", build_score),
}


def query(model, messages, max_tokens=32):
    # Append /no_think so BOTH models answer directly (fair pick-quality compare).
    # The trained model already answers directly (trained answer-only), so this is
    # a no-op for it; the base Qwen3-8B otherwise reasons forever and never commits
    # (chat_template_kwargs enable_thinking=False is ignored by the gateway).
    messages = [dict(m) for m in messages]
    messages[-1]["content"] = messages[-1]["content"] + " /no_think"
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(GATEWAY, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
        "User-Agent": "caratan-eval/1.0"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=90))
        return d["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def _placement_extra(text, gt):
    """top-1 / top-3 / regret from spot_scores."""
    ss = {_node_id_str(k): float(v) for k, v in gt["spot_scores"].items()}
    m = PLC_ANS.search(text or "")
    chosen = _node_id_str(m.group(1)) if m else None
    order = sorted(ss.values(), reverse=True)
    best, worst = order[0], order[-1]
    if chosen not in ss:
        return {"top1": 0, "top3": 0, "regret": 1.0}
    c = ss[chosen]
    thr3 = order[min(2, len(order) - 1)]
    return {"top1": int(c >= best), "top3": int(c >= thr3),
            "regret": (best - c) / (best - worst) if best > worst else 0.0}


def eval_model(env, model, rows):
    _, scorer = ENVS[env]
    def one(row):
        text = query(model, row["prompt"])
        gt = row["ground_truth"]
        if env == "placement":
            reward, _ = scorer(text, gt["spot_scores"])
        else:
            reward, _ = scorer(text, gt)
        rec = {"reward": reward, "text": text}
        if env == "placement":
            rec.update(_placement_extra(text, gt))
        else:
            r2 = scorer(text, gt)[1]
            rec["invalid"] = int("invalid" in r2)
            rec["acted"] = int((" trade " in r2) if env == "maritime" else (" build " in r2))
        return rec
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        return list(ex.map(one, rows))


def summarize(env, res):
    n = len(res)
    mean = sum(r["reward"] for r in res) / n
    if env == "placement":
        return (f"mean_reward {mean:+.3f} | top1 {sum(r['top1'] for r in res)/n:.0%} "
                f"| top3 {sum(r['top3'] for r in res)/n:.0%} "
                f"| mean_regret {sum(r['regret'] for r in res)/n:.3f}")
    act = "trade_rate" if env == "maritime" else "build_rate"
    return (f"mean_reward {mean:+.3f} | {act} {sum(r['acted'] for r in res)/n:.0%} "
            f"| invalid {sum(r['invalid'] for r in res)/n:.0%}")


def main():
    for env, (path, _) in ENVS.items():
        rows = [json.loads(l) for l in open(HERE / path) if l.strip()]
        print(f"\n===== {env}  ({len(rows)} held-out scenarios) =====")
        for label, model in [("BEFORE (base Qwen3-8B)", BASE), ("AFTER  (trained)", TRAINED)]:
            res = eval_model(env, model, rows)
            print(f"  {label:24} {summarize(env, res)}")


if __name__ == "__main__":
    main()
