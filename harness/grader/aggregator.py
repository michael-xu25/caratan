"""
aggregator.py — turn per-decision grader verdicts into ranked, discovered weaknesses.

Pipeline position:

    grader (per-decision JSON) ─► [THIS] ─► ranked (criterion, tag) fail-rate table
                                              │
                                              ├─► generation targets  → Michael's env list
                                              └─► before/after scoreboard → the demo metric

WHY THIS STEP EXISTS
    A single failed decision is noise — could be a hard board, a coin-flip, or a grader
    miscall. You can't target it and you can't measure it move. This step converts
    individual mistakes into *systematic* patterns: "criterion C fails X% of the time in
    state Z". A pattern is targetable (generate scenarios of it) and measurable (watch the
    rate drop). The shift from instance → pattern is the whole point.

DESIGN CHOICES (each defends against a specific way the loop can lie to you)
  - fail-RATE, not fail-count: ranking by count just rediscovers which situations are
    COMMON, not which the model is BAD at. Rate = failures / times-scored normalizes for
    frequency so you measure skill deficit. (This is why the grader scores passes too —
    they're the denominator.)
  - group by (criterion, TAG): "bad at robber timing" is too coarse to generate from and
    usually false uniformly. Localizing to the state where it appears is what makes the
    generated scenarios *targeted* — the entire thesis.
  - Wilson lower bound for ranking, not raw rate: a criterion seen 3x failing 3x shows
    100% on pure noise. Wilson discounts small-n high-rates automatically, so the top of
    the list is both high-fail AND well-supported. Raw rate + n stay visible.
  - ONE table serves generation + measurement: train-target and eval-target are identical
    by construction, so you can't "improve" on something you didn't target, or target
    something you don't measure.

TWO POOLS, SAME COMPUTATION
    discovery pool  → find weaknesses, hand top rows to env generation
    held-out pool   → re-measure the SAME rows after training (before vs after)
    Keep the pools strictly separate, or the before/after number measures memorization,
    not skill. Re-measuring on trained instances is the classic self-own.
"""

import json
import math
import glob
import sys
from collections import defaultdict
from pathlib import Path

# Importable as harness.grader.aggregator AND runnable directly (python …/aggregator.py).
try:
    from harness.grader.taxonomy import (
        CRITERIA_BY_TYPE, validate_decision_type, validate_criterion, validate_tags)
except ModuleNotFoundError:  # direct execution: put repo root on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from harness.grader.taxonomy import (
        CRITERIA_BY_TYPE, validate_decision_type, validate_criterion, validate_tags)

MIN_SAMPLES = 15   # floor: a (criterion, tag) seen fewer than this is too noisy to rank
Z = 1.96           # 95% Wilson interval


# ----------------------------------------------------------------------------
# 1. load grader output and flatten into verdict records
# ----------------------------------------------------------------------------
# A "verdict" is the atomic unit the aggregator counts:
#     (decision_type, criterion, tag, failed)
# One grader object (one decision) explodes into one verdict per criterion per tag.
# Doing the per-tag explosion here means a decision tagged [behind, robber_threat]
# that fails `timing` contributes a failure to BOTH (timing, behind) and
# (timing, robber_threat) — giving single-tag weakness rows like "timing-when-behind".

def flatten_grader_object(obj):
    dtype = obj["decision_type"]
    tags = obj.get("state_tags", [])
    if not tags:
        # a decision with no tags can't land in any (criterion, tag) bucket — warn loudly
        print(f"  [warn] decision {obj.get('decision_id')!r} has no state_tags; "
              f"it contributes to no weakness bucket.")
    out = []
    for c in obj["criteria"]:
        for tag in tags:
            out.append({
                "decision_id": obj["decision_id"],
                "decision_type": dtype,
                "criterion": c["name"],
                "tag": tag,
                "failed": bool(c["failed"]),
            })
    return out


def _validate(obj):
    validate_decision_type(obj["decision_type"])
    validate_tags(obj.get("state_tags", []))
    for c in obj["criteria"]:
        validate_criterion(obj["decision_type"], c["name"])


def load_verdicts(path_glob):
    """Load every grader JSON matching a glob -> flat verdict list.
    Each file may hold one object or a list of objects."""
    verdicts = []
    for path in sorted(glob.glob(path_glob)):
        with open(path) as f:
            data = json.load(f)
        objs = data if isinstance(data, list) else [data]
        for obj in objs:
            _validate(obj)               # fail loudly on any off-vocab string
            verdicts.extend(flatten_grader_object(obj))
    return verdicts


# ----------------------------------------------------------------------------
# 2. aggregate verdicts -> ranked weakness rows
# ----------------------------------------------------------------------------

def wilson_lower_bound(fails, n, z=Z):
    """Lower bound of the 95% Wilson score interval for the failure proportion.

    Why rank by this instead of raw fail_rate: it bakes the noise floor into the
    score. 3/3 fails -> ~0.44 (heavily discounted), while 30/35 fails -> ~0.71.
    Small samples get pushed down automatically; you don't chase 3-sample ghosts.
    """
    if n == 0:
        return 0.0
    p = fails / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - margin) / denom


def aggregate(verdicts, min_samples=MIN_SAMPLES):
    """Group by (decision_type, criterion, tag); compute fail_rate, n, Wilson LB.
    Returns rows sorted so the robustly-worst weaknesses are on top."""
    buckets = defaultdict(lambda: {"n": 0, "fails": 0})
    for v in verdicts:
        key = (v["decision_type"], v["criterion"], v["tag"])
        buckets[key]["n"] += 1
        buckets[key]["fails"] += int(v["failed"])

    rows = []
    for (dtype, crit, tag), b in buckets.items():
        n, fails = b["n"], b["fails"]
        rows.append({
            "decision_type": dtype,
            "criterion": crit,
            "tag": tag,
            "fail_rate": round(fails / n, 3),
            "n": n,
            "wilson_lb": round(wilson_lower_bound(fails, n), 3),
            "above_floor": n >= min_samples,
        })

    # rank: above-floor rows first, then by Wilson LB (robustly-bad), then raw rate
    rows.sort(
        key=lambda r: (r["above_floor"], r["wilson_lb"], r["fail_rate"]),
        reverse=True,
    )
    return rows


# ----------------------------------------------------------------------------
# 3. before/after comparison on held-out pools -> the demo scoreboard
# ----------------------------------------------------------------------------

def compare(before_verdicts, after_verdicts, min_samples=MIN_SAMPLES):
    """Join held-out fail-rates before vs after training, per (criterion, tag).
    This is the headline metric: 'timing x behind: 70% -> 22% fail'.
    Sorted by biggest improvement (most negative delta) first."""
    def index(verdicts):
        return {(r["decision_type"], r["criterion"], r["tag"]): r
                for r in aggregate(verdicts, min_samples)}

    b, a = index(before_verdicts), index(after_verdicts)
    rows = []
    for key in set(b) | set(a):
        rb, ra = b.get(key), a.get(key)
        delta = (round(ra["fail_rate"] - rb["fail_rate"], 3)
                 if rb and ra else None)
        rows.append({
            "decision_type": key[0], "criterion": key[1], "tag": key[2],
            "fail_before": rb["fail_rate"] if rb else None,
            "fail_after": ra["fail_rate"] if ra else None,
            "n_before": rb["n"] if rb else 0,
            "n_after": ra["n"] if ra else 0,
            "delta": delta,   # negative = improvement (failure went down)
        })
    rows.sort(key=lambda r: (r["delta"] if r["delta"] is not None else 0.0))
    return rows


# ----------------------------------------------------------------------------
# 4. pretty printers
# ----------------------------------------------------------------------------

def print_weakness_table(rows, top=None):
    rows = [r for r in rows if r["above_floor"]] or rows
    if top:
        rows = rows[:top]
    print(f"{'decision_type':<13} {'criterion':<20} {'tag':<18} "
          f"{'fail%':>6} {'n':>5} {'wilsonLB':>9}")
    print("-" * 76)
    for r in rows:
        flag = "" if r["above_floor"] else "  (below floor)"
        print(f"{r['decision_type']:<13} {r['criterion']:<20} {r['tag']:<18} "
              f"{r['fail_rate']*100:>5.0f}% {r['n']:>5} {r['wilson_lb']:>9.3f}{flag}")


def print_before_after(rows, top=None):
    rows = [r for r in rows if r["delta"] is not None]
    if top:
        rows = rows[:top]
    print(f"{'criterion':<20} {'tag':<18} {'before':>7} {'after':>7} {'delta':>7}")
    print("-" * 62)
    for r in rows:
        print(f"{r['criterion']:<20} {r['tag']:<18} "
              f"{r['fail_before']*100:>6.0f}% {r['fail_after']*100:>6.0f}% "
              f"{r['delta']*100:>+6.0f}%")


# ----------------------------------------------------------------------------
# 5. runnable demo on synthetic data (so you can see the shape immediately)
#    real use: load_verdicts("grader_out/discovery/*.json") etc.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(0)

    def fake_decision(did, dtype, tags, fail_probs):
        """fail_probs: {criterion: P(fail)} for the injected 'true' weaknesses."""
        crits = CRITERIA_BY_TYPE[dtype]
        criteria = []
        for c in crits:
            failed = random.random() < fail_probs.get(c, 0.08)  # 8% baseline noise
            criteria.append({"name": c, "score": 0 if failed else 2,
                             "failed": failed, "reason": "demo"})
        return {"decision_id": did, "decision_type": dtype,
                "state_tags": tags, "criteria": criteria}

    # discovery pool: inject a real weakness — build_spend `timing` fails a lot when `behind`
    discovery = []
    for i in range(400):
        behind = random.random() < 0.4
        tags = ["mid", "behind" if behind else "even"]
        if random.random() < 0.3:
            tags.append("robber_threat")
        fp = {"timing": 0.7} if behind else {"timing": 0.1}
        discovery.append(fake_decision(f"d{i}", "build_spend", tags, fp))

    dv = []
    for obj in discovery:
        _validate(obj)
        dv.extend(flatten_grader_object(obj))

    print("\n=== DISCOVERED WEAKNESSES (discovery pool) ===")
    print_weakness_table(aggregate(dv), top=8)

    # held-out before/after: same weakness, training cut the behind-timing failure rate
    def heldout(timing_fail_when_behind):
        pool = []
        for i in range(300):
            behind = random.random() < 0.5
            tags = ["mid", "behind" if behind else "even"]
            fp = {"timing": timing_fail_when_behind} if behind else {"timing": 0.1}
            pool.append(fake_decision(f"h{i}", "build_spend", tags, fp))
        out = []
        for obj in pool:
            out.extend(flatten_grader_object(obj))
        return out

    before = heldout(0.70)   # untrained
    after = heldout(0.22)    # after targeted training

    print("\n=== BEFORE / AFTER (held-out pool) ===")
    print_before_after(compare(before, after), top=6)
    print("\n(timing x behind is the planted weakness; note it tops discovery and "
          "shows the biggest before/after drop.)\n")
