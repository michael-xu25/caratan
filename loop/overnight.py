"""Overnight self-play-driven improvement run.

Runs in .venv-modal. Until the 4:30am cutoff, each round:
  1. train a candidate on Modal (warm-start from /state/best), promote it,
  2. EVAL BY SELF-PLAY: play a small mirrored batch trained-vs-base (the eval),
  3. mine the model's FAILURES from those self-play games (-> env-gen fuel),
  4. periodically have Claude author a refined env targeting the failures,
  5. keep a snapshot of the best checkpoint *by self-play win-rate*.
The round running when the cutoff passes finishes, then the loop stops.

Then it runs the SAME 100 games vs base on the best checkpoint, and refreshes
caratan-bot.vercel.app — keeping the current 56% as the start of a trajectory so
the night's improvement is visible.

    .venv-modal/bin/python loop/overnight.py        # launch (background it)
"""
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "loop"))

import modal  # noqa: E402
import weakness  # noqa: E402
import envgen  # noqa: E402

SCRATCH = Path("/private/tmp/claude-502/-Users-mxu-Desktop-Caratan/"
               "905a51d7-5cee-476d-b083-f304ec204dbe/scratchpad")
SERVE_URL = (SCRATCH / "serve_url.txt").read_text().strip()
VLLM_KEY = (SCRATCH / "vllm_key.txt").read_text().strip()

VENV = str(REPO / ".venv" / "bin" / "python")          # catanatron harness venv
CUTOFF = datetime.datetime.now().replace(hour=4, minute=30, second=0, microsecond=0)
STATE = Path(REPO / "loop" / "state"); STATE.mkdir(parents=True, exist_ok=True)
PROGRESS = STATE / "overnight.json"

# held-out seeds: first 50 = final 100-game eval; next 8 = per-round self-play gate
GRADER = json.loads((REPO / "dataset/initial/index.json").read_text())
SEEDS = sorted(b["seed"] for b in GRADER["boards"] if b.get("split") == "grader_games")
FULL_SEEDS = SEEDS[:50]
GATE_SEEDS = SEEDS[50:58]
START_WINRATE = 0.561        # the checkpoint we're at right now (today's 100-game run)

train = modal.Function.from_name("caratan", "train")
promote = modal.Function.from_name("caratan", "promote")
vol_copy = modal.Function.from_name("caratan", "vol_copy")

ENVS = ["placement", "maritime", "build"]


def log(m):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


def selfplay(seeds, run_dir, concurrency):
    """Run a mirrored self-play batch trained-vs-base via the harness. Returns
    win-rate dict. This IS the eval."""
    env = dict(os.environ, MODAL_LLM_URL=SERVE_URL, VLLM_API_KEY=VLLM_KEY, MODAL_NO_THINK="1")
    subprocess.run(
        [VENV, "-m", "harness.cli", "--a", "modal:trained", "--b", "modal:base",
         "--seeds", ",".join(map(str, seeds)), "--max-turns", "300",
         "--concurrency", str(concurrency), "--run-dir", run_dir],
        cwd=str(REPO), env=env, check=False)
    return winrate_of(run_dir)


def winrate_of(run_dir):
    tw = bw = dr = 0
    for p in Path(run_dir).glob("*.json"):
        if p.name.endswith(".view.json") or p.name in ("winrate.json", "matchups.json"):
            continue
        try:
            g = json.loads(p.read_text())
        except Exception:
            continue
        win = g.get("winning_color")
        if win is None:
            dr += 1; continue
        spec = g.get("seats", {}).get(win, "")
        tw += int("trained" in spec); bw += int("trained" not in spec)
    dec = tw + bw
    return {"games": tw + bw + dr, "trained_wins": tw, "base_wins": bw,
            "draws": dr, "winrate": (tw / dec) if dec else 0.0}


def pick_envgen_target(report):
    """Choose which base env to refine from the mined failures."""
    los = report.get("losing_decision_samples", [])
    counts = {"placement": 0, "maritime": 0, "build": 0}
    for d in los:
        at = (d.get("action_type") or "")
        if at == "MARITIME_TRADE":
            counts["maritime"] += 1
        elif at.startswith("BUILD_"):
            counts["build" if d.get("phase") != "early" else "placement"] += 1
    return max(counts, key=counts.get) if any(counts.values()) else "placement"


def save(state):
    PROGRESS.write_text(json.dumps(state, indent=2))


def main():
    log(f"overnight run start; cutoff {CUTOFF:%H:%M}; gate seeds {GATE_SEEDS}")
    state = {"start_winrate": START_WINRATE, "rounds": [], "envgen": [],
             "best_overnight_winrate": None, "cutoff": CUTOFF.isoformat()}

    # snapshot current best as the running best-by-self-play, measure its gate win-rate
    vol_copy.remote("/state/best", "/state/best_overnight")
    base_dir = "transcripts/overnight/round0_baseline"
    wr0 = selfplay(GATE_SEEDS, base_dir, concurrency=16)
    state["best_overnight_winrate"] = wr0["winrate"]
    log(f"baseline gate win-rate: {wr0}")
    save(state)

    admitted = []   # (env_key, base_env)
    i = 0
    while datetime.datetime.now() < CUTOFF:
        i += 1
        try:
            # choose env: cycle canonical, sometimes an admitted v2
            if admitted and i % 3 == 0:
                env, base_env = admitted[i % len(admitted)]
            else:
                env, base_env = ENVS[i % len(ENVS)], None
            log(f"round {i}: train {env}" + (f" (base {base_env})" if base_env else ""))
            train.remote(env, steps=40, group=8, lr=4e-5, warm_start=True, base_env=base_env)
            promote.remote(env)

            # EVAL = self-play (serve cold-starts with the new best after the train gap)
            rd = f"transcripts/overnight/round{i}_{env}"
            wr = selfplay(GATE_SEEDS, rd, concurrency=16)
            log(f"round {i}: self-play win-rate {wr['winrate']:.3f} {wr}")

            # systematically review failures from self-play
            report = weakness.mine(rd, model="trained")

            kept = wr["winrate"] >= state["best_overnight_winrate"]
            if kept:
                vol_copy.remote("/state/best", "/state/best_overnight")
                state["best_overnight_winrate"] = wr["winrate"]
            state["rounds"].append({"round": i, "env": env, "base_env": base_env,
                                    "winrate": wr["winrate"], "record": wr,
                                    "kept_as_best": kept,
                                    "fallback_rate": report.get("fallback_rate"),
                                    "action_mix": report.get("action_mix")})
            save(state)

            # periodically invent a refined env from the failures
            if i % 3 == 0:
                tgt = pick_envgen_target(report)
                try:
                    gen = envgen.generate_env(report, tgt, str(REPO / "loop/state/envs"))
                    rec = {"round": i, "target": tgt,
                           **{k: gen[k] for k in ("env_key", "admitted", "reasons") if k in gen}}
                    state["envgen"].append(rec)
                    if gen.get("admitted"):
                        # unique key per round so graders never collide on the volume
                        ekey = f"{tgt}_v2_r{i}"
                        subprocess.run(
                            [str(REPO / ".venv-modal/bin/python"), "-m", "modal", "volume",
                             "put", "caratan-state", gen["grader_path"], f"/envs/{ekey}.py"],
                            cwd=str(REPO), check=False)
                        admitted.append((ekey, tgt))
                        log(f"round {i}: env-gen ADMITTED {ekey} (base {tgt})")
                    else:
                        log(f"round {i}: env-gen rejected {gen.get('reasons')}")
                    save(state)
                except Exception as e:
                    log(f"round {i}: env-gen error {e}")
        except Exception as e:
            log(f"round {i} ERROR (continuing): {e}")
            time.sleep(10)

    log(f"cutoff reached after round {i}; best-overnight gate win-rate "
        f"{state['best_overnight_winrate']:.3f}")

    # use the best-by-self-play checkpoint for the final eval + serving
    vol_copy.remote("/state/best_overnight", "/state/best")
    log("running FINAL 100 games vs base on the best checkpoint...")
    final_dir = "transcripts/overnight-final"
    final = selfplay(FULL_SEEDS, final_dir, concurrency=32)
    state["final_winrate"] = final["winrate"]; state["final_record"] = final
    save(state)
    log(f"FINAL 100-game win-rate: {final}")

    refresh_site(final, state)
    log("DONE. site refreshed with the trajectory.")


def refresh_site(final, state):
    """Build views + runs.json for the new run, set the win-rate TRAJECTORY
    (keep the current 56% as the start point), rebuild + deploy."""
    final_dir = "transcripts/overnight-final"
    subprocess.run([VENV, "scripts/build_viewer_data.py", final_dir], cwd=str(REPO), check=False)
    subprocess.run([str(REPO / ".venv-modal/bin/python"), "loop/build_site_run.py",
                    final_dir, "--run", "overnight-final", "--n", "12"], cwd=str(REPO), check=False)

    wr = round(final["winrate"], 3)
    new_block = {
        "cap_turns": 300, "games": final["games"],
        "record": {k: final[k] for k in ("trained_wins", "base_wins", "draws")},
        "entries": [
            {"label": "After overnight self-play", "sub": f"{final['games']} games",
             "value": wr, "note": f"{final['trained_wins']} wins / {final['base_wins']} losses — best checkpoint, selected by self-play"},
            {"label": "Start of the night", "sub": "previous checkpoint",
             "value": START_WINRATE, "note": "where the model was before the overnight self-play run"},
        ],
        "trajectory": [
            {"label": "Start of night", "winrate": START_WINRATE, "games": 100},
            {"label": "After overnight self-play", "winrate": wr, "games": final["games"]},
        ],
        "footnote": ("Same 100 held-out boards, mirrored, reasoning off, 300-turn cap. "
                     "Tonight's run used SELF-PLAY as the eval: each round the model played the "
                     "base model, its failures were mined, and Claude authored sharper reward "
                     "envs to fix them. The best checkpoint by self-play win-rate is shown; the "
                     "start point is the previous checkpoint, so the trajectory is visible."),
        "headline": {"trained": wr, "baseline": 0.5, "games": final["games"]},
    }
    for p in ["web/app/data/results.json", "web/public/data/results.json"]:
        d = json.loads((REPO / p).read_text()); d["winrate"] = new_block
        (REPO / p).write_text(json.dumps(d, indent=2))

    # matchups viewer page: feature the new run only
    fwr = json.loads((REPO / final_dir / "winrate.json").read_text()) \
        if (REPO / final_dir / "winrate.json").exists() else final
    (REPO / "web/public/viewer/data/matchups.json").write_text(
        json.dumps({"matchups": [fwr]}, indent=2))

    subprocess.run(["npm", "run", "build"], cwd=str(REPO / "web"), check=False)
    token = json.loads(Path("/Users/mxu/Library/Application Support/com.vercel.cli/auth.json")
                       .read_text())["token"]
    subprocess.run(["npx", "vercel", "deploy", "--prod", "--yes", "--token", token],
                   cwd=str(REPO / "web"), check=False)


if __name__ == "__main__":
    main()
