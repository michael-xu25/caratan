"""Human-readable status for a Fireworks RFT job — the readout the dashboard hides.

Fireworks reports `state=RUNNING` for BOTH "provisioning GPUs" and "actually
training", with no label. But the job's progress counters tell them apart:
  RUNNING + totalInputRequests == 0  -> still PROVISIONING (no rollouts yet)
  RUNNING + totalInputRequests  > 0  -> TRAINING (with %, epoch, rollout counts)
This script derives the real phase + a rough ETA so you don't babysit the UI.

Usage:
  python training/rft_status.py <job_id>            # one-shot
  python training/rft_status.py <job_id> --watch    # poll until terminal
  python training/rft_status.py                      # uses $RFT_JOB_ID

Needs FIREWORKS_API_KEY in env (set -a; source .env; set +a).
"""
import json
import os
import sys
import time
import urllib.request

ACCOUNT = "brickedup25"


def _get(job_id: str) -> dict:
    url = f"https://api.fireworks.ai/v1/accounts/{ACCOUNT}/reinforcementFineTuningJobs/{job_id}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
        "Accept": "application/json",
        "User-Agent": "caratan-rft-status/1.0",  # default urllib UA gets 403'd
    })
    return json.load(urllib.request.urlopen(req))


def _fmt(secs: float) -> str:
    secs = int(max(0, secs))
    h, m = secs // 3600, (secs % 3600) // 60
    return f"{h}h{m:02d}m" if h else f"{m}m{secs % 60:02d}s"


def phase_line(j: dict) -> tuple:
    """Return (is_terminal, one-line human status)."""
    state = j["state"]
    p = j.get("jobProgress", {}) or {}
    pct = p.get("percent", 0) or 0
    done = p.get("totalProcessedRequests", 0) or 0
    total = p.get("totalInputRequests", 0) or 0
    epoch = p.get("epoch", 0) or 0
    short = state.replace("JOB_STATE_", "")

    if state == "JOB_STATE_COMPLETED":
        return True, f"✅ COMPLETED — model: {j.get('trainingConfig', {}).get('outputModel', '?')}"
    if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
        return True, f"❌ {short} — {j.get('status', {})}"
    if "DELET" in state:
        return True, f"🗑️  {short}"
    if state in ("JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_CREATING"):
        if total == 0:
            return False, "⏳ PROVISIONING — allocating GPUs / loading base model (no rollouts yet)"
        eta = ""
        if pct > 0:
            # rough: scale elapsed-since-create by remaining fraction
            try:
                import datetime
                ct = j["createTime"].replace("Z", "+00:00")
                elapsed = time.time() - datetime.datetime.fromisoformat(ct).timestamp()
                eta = f", ~{_fmt(elapsed * (100 - pct) / pct)} left"
            except Exception:
                eta = ""
        return False, f"🟢 TRAINING — {pct}% | epoch {epoch} | {done}/{total} rollouts{eta}"
    return False, f"… {short}"


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    watch = "--watch" in argv
    job_id = args[0] if args else os.environ.get("RFT_JOB_ID", "")
    if not job_id:
        print("usage: rft_status.py <job_id> [--watch]"); return 2
    while True:
        try:
            j = _get(job_id)
        except Exception as e:
            print(f"[error fetching job: {e}]"); return 1
        terminal, line = phase_line(j)
        print(f"[{time.strftime('%H:%M:%S')}] {job_id}  {line}")
        if terminal or not watch:
            return 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
