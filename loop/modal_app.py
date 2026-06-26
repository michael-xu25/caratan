"""Caratan on Modal — shared app: vLLM serving + (later) GRPO training.

Deploy:   .venv-modal/bin/modal deploy loop/modal_app.py
Serve URL is printed on deploy; the orchestrator/harness hit it OpenAI-style.

Two images on purpose: serving needs vLLM only; training needs torch+trl+peft.
Both share one Volume of state (best LoRA, registry, logs) and one HF cache.
"""
import modal

app = modal.App("caratan")

# --- shared storage -------------------------------------------------------
hf_cache = modal.Volume.from_name("caratan-hf-cache", create_if_missing=True)
state = modal.Volume.from_name("caratan-state", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"
STATE = "/state"                       # best/ (LoRA), registry.json, rounds/, datasets/

BASE_MODEL = "Qwen/Qwen3-8B"
SERVED_NAME = "catan"                  # the model name clients request
BEST_LORA_DIR = f"{STATE}/best"        # current promoted adapter (may not exist yet)

# --- serving image (vLLM) -------------------------------------------------
# vLLM >=0.8.5 supports Qwen3. Pin is tunable if a newer build is needed.
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.8.5", "transformers==4.51.3", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)


@app.function(
    image=vllm_image,
    gpu="L40S",                        # 48GB: 8B(bf16)~16GB + KV cache + LoRA headroom
    volumes={HF_CACHE: hf_cache, STATE: state},
    secrets=[modal.Secret.from_name("caratan", required_keys=["VLLM_API_KEY"])],
    timeout=24 * 60 * 60,
    scaledown_window=300,              # stay warm 5min after last request
    min_containers=0,
)
@modal.concurrent(max_inputs=64)       # high concurrency = fast self-play
@modal.web_server(port=8000, startup_timeout=15 * 60)
def serve():
    """OpenAI-compatible vLLM server. Serves base + the promoted LoRA (if any)
    under one name `catan`, so the orchestrator never changes the client config —
    a freshly promoted adapter is just hot-reloaded into /state/best."""
    import os
    import subprocess

    state.reload()                     # pick up the latest promoted adapter
    # Serve the base model under "base" and (if present) the promoted adapter under
    # "trained" — so self-play can pit them head-to-head from one endpoint.
    cmd = [
        "vllm", "serve", BASE_MODEL,
        "--served-model-name", "base", SERVED_NAME,
        "--port", "8000",
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
        "--enable-prefix-caching",
    ]
    if os.path.isdir(BEST_LORA_DIR) and os.listdir(BEST_LORA_DIR):
        cmd += [
            "--enable-lora",
            "--max-lora-rank", "32",
            "--lora-modules", f"trained={BEST_LORA_DIR}",
        ]
    print("launching:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)


# --- training image (TRL GRPO + vLLM colocate generation) -----------------
# Version matrix is finicky; these are a known-compatible set for Qwen3 + GRPO.
# vLLM matches the serving image so generation behaves identically.
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",
        "trl==0.17.0",
        "peft==0.15.2",
        "accelerate==1.6.0",
        "datasets==3.5.0",
        "vllm==0.8.5",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("graders")   # ship loop/graders.py into the container
)

# env key -> training dataset (the .trl.jsonl prompt+ground_truth files)
DATASETS = {
    "placement": "/state/datasets/placement_opening_train.trl.jsonl",
    "maritime": "/state/datasets/maritime_trade_train.trl.jsonl",
    "build": "/state/datasets/build_train.trl.jsonl",
}


@app.function(
    image=train_image,
    gpu="A100-80GB",
    volumes={HF_CACHE: hf_cache, STATE: state},
    secrets=[modal.Secret.from_name("caratan")],
    timeout=8 * 60 * 60,
)
def train(env_key: str, steps: int = 60, group: int = 8, lr: float = 4e-5,
          warm_start: bool = True, base_env: str | None = None):
    """One GRPO round on an env. Warm-starts from /state/best (the promoted
    adapter) if present, trains, writes the candidate to /state/candidate/<env>/.
    Promotion (eval gate) happens in the orchestrator, NOT here.

    For an AUTONOMOUS env (e.g. env_key='maritime_v2', base_env='maritime'): the
    reward grader is the Claude-authored one at /state/envs/<env_key>.py, but the
    scenarios are base_env's real ones. Promotion is still gated on base_env's
    CANONICAL grader (in the orchestrator) so a generated grader can't be hacked."""
    import json
    import os
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    import graders  # shipped via add_local_python_source

    state.reload()
    data_env = base_env or env_key
    if base_env:                      # autonomous env: load the generated grader
        code = open(f"{STATE}/envs/{env_key}.py").read()
        scorer = graders.load_score_from_code(code)
        reward_fn = graders.make_reward_fn(env_key, scorer=scorer)
        print(f"[{env_key}] using generated grader (base={base_env})", flush=True)
    else:
        reward_fn = graders.make_reward_fn(env_key)
    rows = [json.loads(l) for l in open(DATASETS[data_env]) if l.strip()]

    def nothink(prompt):
        # answer-only rollouts: append /no_think so Qwen3 doesn't burn the 32-token
        # budget reasoning and never reach <answer> (reward would be a flat 0).
        # Identical to the eval format -> train/eval parity.
        msgs = [dict(m) for m in prompt]
        msgs[-1]["content"] = msgs[-1]["content"] + " /no_think"
        return msgs

    ds = Dataset.from_list([
        {"prompt": nothink(r["prompt"]), "ground_truth": r["ground_truth"]} for r in rows
    ])
    print(f"[{env_key}] {len(ds)} training rows", flush=True)

    peft_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    out_dir = f"{STATE}/candidate/{env_key}"
    resume = f"{BEST_LORA_DIR}" if (warm_start and os.path.isdir(BEST_LORA_DIR)
                                    and os.listdir(BEST_LORA_DIR)) else None

    cfg = GRPOConfig(
        output_dir=out_dir,
        model_init_kwargs={"torch_dtype": "bfloat16"},  # else TRL loads fp32 (32GB!) -> OOM
        per_device_train_batch_size=group,
        num_generations=group,                 # GRPO group size
        gradient_accumulation_steps=4,
        learning_rate=lr,                       # 4e-5 — the value that actually climbs
        max_steps=steps,
        max_prompt_length=1280,                 # prompts are ~1.2k tok; no truncation
        max_completion_length=32,               # answer-only (/no_think)
        temperature=1.0,
        beta=0.0,                               # no KL ref model -> saves a full 16GB copy
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,            # trade compute for activation memory
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=False,                         # HF generation (robust; completions
                                                # are 32 tok so this is fine). vLLM
                                                # colocate is a later perf upgrade.
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=BASE_MODEL,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=ds,
        peft_config=peft_cfg,
    )
    if resume:
        print(f"[{env_key}] warm-starting LoRA from {resume}", flush=True)
        trainer.model.load_adapter(resume, adapter_name="default")

    trainer.train()
    trainer.save_model(out_dir)                 # adapter_config.json + safetensors
    state.commit()

    # surface the reward trend so the orchestrator/logs can see it climb
    hist = [h for h in trainer.state.log_history if "reward" in h]
    trend = [round(h["reward"], 3) for h in hist]
    print(f"[{env_key}] reward trend: {trend}", flush=True)
    return {"env": env_key, "candidate": out_dir, "steps": steps,
            "reward_trend": trend,
            "reward_first": trend[0] if trend else None,
            "reward_last": trend[-1] if trend else None}


@app.function(
    image=train_image,
    gpu="A100-80GB",
    volumes={HF_CACHE: hf_cache, STATE: state},
    secrets=[modal.Secret.from_name("caratan")],
    timeout=2 * 60 * 60,
)
def evaluate(env_key: str, adapter: str | None = None, limit: int = 0):
    """Held-out mean reward for an adapter (None = base Qwen3-8B). Deterministic
    (temperature 0), answer-only (/no_think), scored with the SAME graders used in
    training. This IS the promotion gate's measuring stick."""
    import json
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    import graders

    state.reload()
    path = f"{STATE}/datasets/{env_key}_eval.trl.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if limit:
        rows = rows[:limit]

    llm = LLM(model=BASE_MODEL, enable_lora=adapter is not None,
              max_lora_rank=32, max_model_len=4096, gpu_memory_utilization=0.90,
              enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=32)

    def with_nothink(msgs):
        msgs = [dict(m) for m in msgs]
        msgs[-1]["content"] = msgs[-1]["content"] + " /no_think"
        return msgs

    convos = [with_nothink(r["prompt"]) for r in rows]
    lora_req = LoRARequest("cand", 1, adapter) if adapter else None
    outs = llm.chat(convos, sp, lora_request=lora_req, use_tqdm=False)

    scorer = graders.GRADERS[env_key]
    rewards, valid = [], 0
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        reward, reason = scorer(text, r["ground_truth"])
        rewards.append(reward)
        valid += 0 if ("unparseable" in reason or "invalid" in reason) else 1
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    res = {"env": env_key, "adapter": adapter or "base", "n": len(rows),
           "mean_reward": round(mean, 4), "valid_rate": round(valid / len(rows), 4)}
    print(res, flush=True)
    return res


# --- autonomous loop image (CPU: orchestration + env-gen brain) -----------
loop_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("anthropic", "python-dotenv", "fastapi[standard]")
    .add_local_python_source("graders", "weakness", "envgen")
)

REGISTRY_JSON = f"{STATE}/registry.json"
LOOP_JSON = f"{STATE}/loop.json"
PROMOTE_MARGIN = 0.01


def _read_json(path, default):
    import json
    import os
    return json.load(open(path)) if os.path.exists(path) else default


def _round(env, gate_env, steps, group, lr, eval_limit):
    """One round, Modal-native: calls the GPU functions, returns the record.
    Gate is always measured on gate_env's CANONICAL grader (unhackable)."""
    import os
    reg = _read_json(REGISTRY_JSON, {"rounds": [], "best": {}})
    has_best = os.path.isdir(BEST_LORA_DIR) and bool(os.listdir(BEST_LORA_DIR)) \
        if os.path.isdir(BEST_LORA_DIR) else False

    before = evaluate.remote(gate_env, adapter=BEST_LORA_DIR if has_best else None,
                             limit=eval_limit)
    base_env = gate_env if env != gate_env else None
    tr = train.remote(env, steps=steps, group=group, lr=lr, base_env=base_env)
    after = evaluate.remote(gate_env, adapter=f"{STATE}/candidate/{env}", limit=eval_limit)

    gain = after["mean_reward"] - before["mean_reward"]
    promoted = gain >= PROMOTE_MARGIN
    if promoted:
        promote.remote(env)
    rec = {"env": env, "gate_env": gate_env, "steps": steps,
           "before": before["mean_reward"], "after": after["mean_reward"],
           "gain": round(gain, 4), "promoted": promoted,
           "train_first": tr["reward_first"], "train_last": tr["reward_last"]}
    return rec, promoted, after["mean_reward"]


@app.function(image=loop_image, volumes={STATE: state},
              secrets=[modal.Secret.from_name("caratan")], timeout=24 * 60 * 60)
def loop_run(n_rounds: int = 200, steps: int = 40, group: int = 8, lr: float = 4e-5,
             eval_limit: int = 40, envgen_every: int = 4, max_hours: float = 23.0):
    """The autonomous self-improvement loop. Cycles the canonical envs; every
    `envgen_every` rounds the env-gen brain authors a refined grader for the
    weakest env (sanity-gated) and trains on it — gated on the canonical metric.
    Writes registry + loop.json to the Volume each round so the live site updates
    and the loop is resume-safe. Exits after `max_hours` so the daily cron can
    relaunch cleanly with no overlap (state persists on the Volume)."""
    import json
    import os
    import time

    import envgen
    import graders

    state.reload()
    # single-runner lock: refuse to start if another loop's heartbeat is fresh.
    # Prevents the daily cron and a manual run from both promoting to /state/best.
    lock = f"{STATE}/loop.lock"
    if os.path.exists(lock):
        age = time.time() - _read_json(lock, {"ts": 0}).get("ts", 0)
        if age < 30 * 60:
            print(f"[loop] another loop is active (heartbeat {age:.0f}s old); exiting", flush=True)
            return {"skipped": "already running"}

    def _beat():
        json.dump({"ts": time.time()}, open(lock, "w"))
        state.commit()

    _beat()
    reg = _read_json(REGISTRY_JSON, {"rounds": [], "best": {}})
    canonical = ["placement", "maritime", "build"]
    extra_envs = [(e["env_key"], e["base_env"]) for e in reg.get("envgen", [])
                  if e.get("admitted")]   # resume admitted generated envs
    t0 = time.time()

    for i in range(n_rounds):
        if (time.time() - t0) / 3600.0 >= max_hours:
            print(f"[loop] hit max_hours={max_hours}; exiting for cron relaunch", flush=True)
            break
        # choose env: usually cycle canonical; sometimes train an admitted v2 env
        use_extra = extra_envs and (i % 2 == 1)
        if use_extra:
            env, gate_env = extra_envs[i % len(extra_envs)]
        else:
            env = canonical[i % len(canonical)]
            gate_env = env

        rec, promoted, score = _round(env, gate_env, steps, group, lr, eval_limit)
        reg["rounds"].append(rec)
        reg.setdefault("best", {})
        if promoted:
            reg["best"][gate_env] = score
        _write_state(reg)
        _beat()
        print(f"[loop {i+1}/{n_rounds}] {env}: {rec['before']}->{rec['after']} "
              f"{'PROMOTED' if promoted else 'kept'}", flush=True)

        # periodically let the brain invent a refined env for the weakest canonical
        if envgen_every and (i + 1) % envgen_every == 0:
            weakest = min(canonical, key=lambda e: reg["best"].get(e, 0.0))
            report = {"target": weakest, "best_by_env": reg["best"],
                      "recent_rounds": reg["rounds"][-6:],
                      "note": f"{weakest} has the lowest held-out reward; sharpen its reward."}
            try:
                gen = envgen.generate_env(report, weakest, f"{STATE}/envs")
                gen_rec = {"round": i + 1, **gen}
                reg.setdefault("envgen", []).append(gen_rec)
                if gen.get("admitted"):
                    extra_envs.append((gen["env_key"], weakest))
                    print(f"[loop] env-gen ADMITTED {gen['env_key']} (base {weakest})", flush=True)
                else:
                    print(f"[loop] env-gen rejected: {gen.get('reasons')}", flush=True)
                _write_state(reg)
            except Exception as e:
                print(f"[loop] env-gen error (skipped): {e}", flush=True)

    if os.path.exists(lock):      # release so the next cron run can start
        os.remove(lock)
        state.commit()
    return {"rounds_run": len(reg["rounds"]), "best": reg.get("best", {}),
            "admitted_envs": [e[0] for e in extra_envs]}


def _write_state(reg):
    """Persist registry + a compact loop.json (the live feed for the site)."""
    import json
    os_makedirs = __import__("os").makedirs
    os_makedirs(STATE, exist_ok=True)
    json.dump(reg, open(REGISTRY_JSON, "w"), indent=2)
    rounds = reg.get("rounds", [])
    feed = {
        "rounds_run": len(rounds),
        "promotions": sum(1 for r in rounds if r.get("promoted")),
        "best_by_env": reg.get("best", {}),
        "envgen": reg.get("envgen", [])[-6:],
        "recent": rounds[-20:],
    }
    json.dump(feed, open(LOOP_JSON, "w"), indent=2)
    state.commit()


# NOTE: daily cron DISABLED (winding the experiment down). Re-add
# `schedule=modal.Cron("0 7 * * *")` to resume multi-day auto-relaunch.
@app.function(image=loop_image, timeout=300)
def daily_loop():
    """Relaunch the loop each day (it self-exits at max_hours, so no overlap).
    State persists on the Volume, so the model keeps improving across days."""
    loop_run.spawn(n_rounds=200, steps=40, group=8, eval_limit=40,
                   envgen_every=4, max_hours=23.0)
    print("daily_loop: spawned a fresh ~23h loop_run", flush=True)


@app.function(image=train_image, volumes={STATE: state}, timeout=600)
def vol_copy(src: str, dst: str):
    """Copy one Volume dir to another (snapshot/restore best checkpoints).
    Paths are absolute under /state. Overwrites dst."""
    import os
    import shutil
    state.reload()
    if not (os.path.isdir(src) and os.listdir(src)):
        return {"copied": False, "reason": f"empty src {src}"}
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    state.commit()
    print(f"copied {src} -> {dst}", flush=True)
    return {"copied": True, "src": src, "dst": dst}


@app.function(image=loop_image, volumes={STATE: state}, min_containers=0)
@modal.fastapi_endpoint(method="GET")
def status():
    """Public live-state endpoint; the site fetches this client-side (cross-origin,
    so we set permissive CORS)."""
    from fastapi.responses import JSONResponse
    state.reload()
    data = _read_json(LOOP_JSON, {"rounds_run": 0, "recent": [], "best_by_env": {}})
    return JSONResponse(content=data, headers={"Access-Control-Allow-Origin": "*",
                                               "Cache-Control": "no-store"})


@app.function(image=train_image, volumes={STATE: state}, timeout=600)
def promote(env_key: str):
    """Copy a validated candidate adapter -> /state/best (the served adapter).
    Called only after the eval gate confirms improvement. Backs up the prior
    best to /state/history/<env>-<n> so a bad promotion is recoverable."""
    import os
    import shutil

    state.reload()
    cand = f"{STATE}/candidate/{env_key}"
    if not (os.path.isdir(cand) and os.listdir(cand)):
        raise RuntimeError(f"no candidate at {cand}")
    if os.path.isdir(BEST_LORA_DIR) and os.listdir(BEST_LORA_DIR):
        hist = f"{STATE}/history"
        os.makedirs(hist, exist_ok=True)
        n = len(os.listdir(hist))
        shutil.copytree(BEST_LORA_DIR, f"{hist}/{env_key}-{n}")
    if os.path.isdir(BEST_LORA_DIR):
        shutil.rmtree(BEST_LORA_DIR)
    shutil.copytree(cand, BEST_LORA_DIR)
    state.commit()
    print(f"promoted {env_key} -> {BEST_LORA_DIR}", flush=True)
    return {"promoted": env_key, "path": BEST_LORA_DIR}


@app.local_entrypoint()
def smoke():
    """After deploy: prints the serve URL to sanity-check from the laptop."""
    print("serve web url:", serve.get_web_url())
