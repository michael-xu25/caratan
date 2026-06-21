"""On-policy GRPO training for a Catan env via HUD/Tinker (cookbook pattern).

Rolls out a taskset under one long-lived job, trains on the batch with a
server-side importance-sampling loss (GRPO over each group), promotes the new
weights behind the SAME model string, repeats. Runs in .venv-hud (py3.11).

    set -a; source ../.env; set +a
    ../.venv-hud/bin/python train.py --env placement --steps 1 --limit 4 --group 4  # smoke
    ../.venv-hud/bin/python train.py --env placement --steps 20 --group 8           # real

`--env` selects the taskset file + HUD template. The forked trainable model
`catan-grpo` is trained in place, so running envs back-to-back warm-starts each
from the previous env's weights (the env1->env2->env3 chain).
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from hud import TrainingClient
from hud.agents import create_agent
from hud.agents.types import AgentStep
from hud.eval import Job, LocalRuntime, Taskset

MODEL = "catan-grpo-q8b"  # forked Qwen3-8B (Tinker), trainable; trained in place.
# Qwen3-8B answers answer-only cleanly (~15 tok), so no CoT budget needed.
HERE = Path(__file__).resolve().parent

# env key -> (taskset jsonl, env module file, template factory name)
ENVS = {
    "placement": ("../data/placement_opening_train.trl.jsonl", "catan_placement_env.py", "placement"),
}


def _valid_run(run):
    """A rollout is valid for training only if it actually produced output tokens
    (a 503'd/errored rollout has an empty trace and must be dropped, not trained on)."""
    try:
        toks = sum(
            len(s.output_token_ids)
            for s in run.trace.collect(
                lambda x: x.sample if isinstance(x, AgentStep) and x.sample else None)
            if getattr(s, "output_token_ids", None))
        return toks > 0 and run.reward is not None
    except Exception:
        return run.reward is not None


def _full_prompt(messages):
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return (system + "\n\n" + user) if system else user


def load_taskset(env_key, limit):
    path, _, _ = ENVS[env_key]
    rows = [json.loads(l) for l in open(HERE / path) if l.strip()]
    if limit:
        rows = rows[:limit]
    if env_key == "placement":
        from catan_placement_env import placement
        tasks = [placement(prompt=_full_prompt(r["prompt"]),
                           spot_scores=r["ground_truth"]["spot_scores"],
                           gold=r["ground_truth"]["gold"]) for r in rows]
    else:
        raise ValueError(f"unknown env {env_key}")
    return Taskset(env_key, tasks)


async def main(env_key, steps, group, lr, limit, max_concurrent, max_tokens, rollout_retries):
    _, env_file, _ = ENVS[env_key]
    agent = create_agent(MODEL, completion_kwargs={
        "max_tokens": max_tokens, "extra_body": {"return_token_ids": True}})
    trainer = TrainingClient(MODEL)
    taskset = load_taskset(env_key, limit)
    runtime = LocalRuntime(str(HERE / env_file))
    print(f"env={env_key} model={MODEL} tasks={len(taskset.tasks) if hasattr(taskset,'tasks') else '?'} "
          f"steps={steps} group={group} lr={lr}", flush=True)
    session = await Job.start(f"catan-{env_key}-rl", group=group)
    history = []
    for step in range(steps):
        # Rollout with retry: Tinker can return 503 upstream_overloaded transiently.
        # Retry the whole taskset.run until every rollout in the batch is valid, so
        # GRPO groups stay intact (we never train on a batch with holes).
        t0 = time.perf_counter()
        batch, valid = [], []
        for attempt in range(rollout_retries):
            start = len(session.runs)
            try:
                await taskset.run(agent, runtime=runtime, job=session,
                                  max_concurrent=max_concurrent)
            except Exception as e:
                print(f"  step {step} rollout error: {type(e).__name__}: {e}", flush=True)
            batch = session.runs[start:]
            valid = [r for r in batch if _valid_run(r)]
            if batch and len(valid) == len(batch):
                break
            wait = min(120, 10 * 2 ** attempt)
            print(f"  step {step} attempt {attempt}: {len(valid)}/{len(batch)} valid "
                  f"(Tinker flaky) — retry in {wait}s", flush=True)
            await asyncio.sleep(wait)
        # Train on intact groups only (whole multiple of group_size).
        keep = (len(valid) // group) * group
        if keep < group:
            print(f"  step {step}: only {len(valid)} valid runs (<group {group}); "
                  f"skipping train this step", flush=True)
            continue
        t1 = time.perf_counter()
        train_batch = valid[:keep]
        fb = await trainer.forward_backward(train_batch, loss_fn="importance_sampling",
                                            group_size=group)
        res = await trainer.optim_step(learning_rate=lr)
        t2 = time.perf_counter()
        rewards = [r.reward for r in train_batch]
        mean = sum(rewards) / len(rewards)
        history.append(mean)
        solved = sum(1 for r in rewards if r >= 0.999)   # top-3 hits (reward==1)
        loss = fb.metrics.get("loss:sum", float("nan"))
        trend = "".join(("▲" if history[i] > history[i - 1] else
                         "▼" if history[i] < history[i - 1] else "·")
                        for i in range(1, len(history)))[-24:]
        print(f"step {step:2d} | top3 {mean:.3f} ({solved}/{len(rewards)}) "
              f"| loss {loss:+.4f} | rollout {t1-t0:5.1f}s train {t2-t1:5.1f}s | {trend}",
              flush=True)
    if history:
        print(f"DONE {env_key}: reward {history[0]:.3f} -> {history[-1]:.3f} "
              f"(best {max(history):.3f}) over {len(history)} steps", flush=True)


if __name__ == "__main__":
    load_dotenv(HERE / "../.env")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="placement", choices=list(ENVS))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--limit", type=int, default=0, help="cap #tasks (0=all)")
    p.add_argument("--max-concurrent", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--rollout-retries", type=int, default=6,
                   help="retries per step when Tinker 503s before giving up the step")
    a = p.parse_args()
    asyncio.run(main(a.env, a.steps, a.group, a.learning_rate, a.limit,
                     a.max_concurrent, a.max_tokens, a.rollout_retries))
