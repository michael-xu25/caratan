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
from pathlib import Path

from dotenv import load_dotenv

from hud import TrainingClient
from hud.agents import create_agent
from hud.eval import Job, LocalRuntime, Taskset

MODEL = "catan-grpo-q8b"  # forked Qwen3-8B (Tinker), trainable; trained in place.
# Qwen3-8B answers answer-only cleanly (~15 tok), so no CoT budget needed.
HERE = Path(__file__).resolve().parent

# env key -> (taskset jsonl, env module file, template factory name)
ENVS = {
    "placement": ("../data/placement_opening_train.trl.jsonl", "catan_placement_env.py", "placement"),
}


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


async def main(env_key, steps, group, lr, limit, max_concurrent, max_tokens):
    _, env_file, _ = ENVS[env_key]
    agent = create_agent(MODEL, completion_kwargs={
        "max_tokens": max_tokens, "extra_body": {"return_token_ids": True}})
    trainer = TrainingClient(MODEL)
    taskset = load_taskset(env_key, limit)
    runtime = LocalRuntime(str(HERE / env_file))
    print(f"env={env_key} model={MODEL} tasks={len(taskset.tasks) if hasattr(taskset,'tasks') else '?'} "
          f"steps={steps} group={group} lr={lr}", flush=True)
    session = await Job.start(f"catan-{env_key}-rl", group=group)
    for step in range(steps):
        start = len(session.runs)
        await taskset.run(agent, runtime=runtime, job=session, max_concurrent=max_concurrent)
        batch = session.runs[start:]
        fb = await trainer.forward_backward(batch, loss_fn="importance_sampling", group_size=group)
        res = await trainer.optim_step(learning_rate=lr)
        rewards = [run.reward for run in batch]
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        solved = sum(1 for r in rewards if r >= 0.999)
        loss = fb.metrics.get("loss:sum", float("nan"))
        print(f"step {step:2d} | reward {mean:.3f} | optimal {solved}/{len(rewards)} "
              f"| loss {loss:+.4f} | optim {res.step}", flush=True)


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
    a = p.parse_args()
    asyncio.run(main(a.env, a.steps, a.group, a.learning_rate, a.limit,
                     a.max_concurrent, a.max_tokens))
