"""Round driver for the self-improvement loop.

One round = pick an env -> measure current best on held-out -> GRPO a candidate
-> measure candidate -> PROMOTE only if it beat the best by a margin. Calls the
deployed Modal functions (train / evaluate / promote). Resume-safe: all state
lives on the Modal Volume; the local registry mirror records each round.

    # one round on placement (drives Modal GPUs from the laptop)
    .venv-modal/bin/python loop/orchestrator.py --env placement --steps 40

Phase 3 will wrap `run_round` in a long-running Modal loop with the weakness
miner + env-gen brain choosing the env each round; this is the inner step.
"""
import argparse
import json
import time
from pathlib import Path

import modal

APP = "caratan"
STATE_BEST = "/state/best"
PROMOTE_MARGIN = 0.01           # candidate must beat best by this to promote
REGISTRY = Path(__file__).resolve().parent / "state" / "registry.json"


def _fn(name):
    return modal.Function.from_name(APP, name)


def _load_registry():
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"rounds": [], "best": {}}


def _save_registry(reg):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2))


def run_round(env, steps=40, group=8, lr=4e-5, eval_limit=0, base_env=None):
    """One round on `env`. If `base_env` is set, `env` is an autonomous env whose
    generated grader shapes training, but the gate measures base_env's CANONICAL
    held-out reward (unhackable)."""
    train, evaluate, promote = _fn("train"), _fn("evaluate"), _fn("promote")
    reg = _load_registry()
    gate_env = base_env or env          # what the gate measures (always canonical)
    has_best = "best" in reg and reg["best"]   # one shared adapter chain

    # 1. baseline = current best adapter (or base model on the very first round)
    before = evaluate.remote(gate_env, adapter=STATE_BEST if has_best else None,
                             limit=eval_limit)
    print(f"[{env}] before({gate_env}): {before}")

    # 2. GRPO a candidate (warm-starts from best if present)
    tr = train.remote(env, steps=steps, group=group, lr=lr, base_env=base_env)
    print(f"[{env}] trained: reward {tr['reward_first']} -> {tr['reward_last']}")

    # 3. measure the candidate on the CANONICAL gate env
    after = evaluate.remote(gate_env, adapter=f"/state/candidate/{env}", limit=eval_limit)
    print(f"[{env}] after({gate_env}):  {after}")

    # 4. promotion gate (on the canonical gate env)
    gain = after["mean_reward"] - before["mean_reward"]
    promoted = gain >= PROMOTE_MARGIN
    if promoted:
        promote.remote(env)
        reg.setdefault("best", {})[gate_env] = after["mean_reward"]
    print(f"[{env}] gain {gain:+.4f} -> {'PROMOTED' if promoted else 'rejected'}")

    reg["rounds"].append({
        "env": env, "gate_env": gate_env, "steps": steps, "lr": lr,
        "before": before["mean_reward"], "after": after["mean_reward"],
        "gain": round(gain, 4), "promoted": promoted,
        "train_first": tr["reward_first"], "train_last": tr["reward_last"],
        "valid_before": before["valid_rate"], "valid_after": after["valid_rate"],
    })
    _save_registry(reg)
    return reg["rounds"][-1]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=["placement", "maritime", "build"])
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--eval-limit", type=int, default=0, help="0 = full held-out set")
    a = p.parse_args()
    t0 = time.time()
    res = run_round(a.env, steps=a.steps, group=a.group, lr=a.lr, eval_limit=a.eval_limit)
    print(f"\nround done in {(time.time()-t0)/60:.1f} min: {res}")
