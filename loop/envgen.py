"""Env-gen brain — the autonomous part. Claude reads the weakness report and
authors a NEW reward grader targeting the discovered weakness, then a hard
sanity gate decides whether it's allowed to train.

Design that makes this safe to run unattended:
  * The generated grader SHAPES training, but promotion is gated on an
    INDEPENDENT trusted metric (canonical held-out eval + self-play win-rate),
    never on the generated grader itself — so a generated grader cannot be
    reward-hacked into a fake "improvement".
  * Generated graders score the EXISTING env ground_truth schema, so scenarios
    keep real, valid ground truth (no hallucinated scenarios).
  * The sanity gate rejects any grader that isn't deterministic, doesn't rank a
    known-good answer above a known-bad one, rewards garbage, is unbounded, or
    touches the filesystem/network/os.

    .venv-modal/bin/python loop/envgen.py --report weakness.json --base-env maritime
"""
import argparse
import json
import os
import re
from pathlib import Path

MODEL = "claude-opus-4-8"

# code the generated grader is NEVER allowed to contain (defense-in-depth; it
# also runs sandboxed on Modal, but we refuse to even admit it)
# Block dangerous imports and BARE builtin calls. Negative lookbehind on [\w.] so
# attribute access (re.compile, x.open) is NOT flagged — only the bare builtins.
# (The restricted exec namespace also lacks these, so this is defense-in-depth.)
_FORBIDDEN = re.compile(
    r"(import\s+(os|sys|subprocess|socket|shutil|pathlib|requests|urllib)\b|"
    r"__import__|__builtins__|"
    r"(?<![\w.])(eval|exec|open|compile|globals|locals|input|getattr|setattr|"
    r"vars|breakpoint)\s*\()", re.I)

GRADER_CONTRACT = '''\
Write a Python function `score(text, gt)` (stdlib only — you MAY `import re`).
- `text` is the model's raw answer string; `gt` is a dict (the row's ground_truth).
- return a tuple (reward: float, reason: str).
- return a NEGATIVE or zero reward with a reason containing "invalid" for
  unparseable / illegal answers.
- be fully deterministic (no randomness, time, io, network, os, files).
- keep reward bounded roughly in [-2, 2].
'''


def _client():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    import anthropic
    return anthropic.Anthropic()


def _existing_grader_src(base_env):
    """Show the brain the current grader for the env it's refining."""
    src = (Path(__file__).resolve().parent / "graders.py").read_text()
    return src


def propose(report, base_env):
    """Ask Claude for a refined grader targeting the mined weakness. Returns the
    parsed proposal dict (grader_code, rationale, sanity exemplars)."""
    client = _client()
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "env_key": {"type": "string"},
            "rationale": {"type": "string"},
            "grader_code": {"type": "string"},
            "sanity": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    # JSON-encoded string (structured output needs closed objects)
                    "ground_truth_json": {"type": "string"},
                    "good_answer": {"type": "string"},
                    "bad_answer": {"type": "string"},
                    "garbage_answer": {"type": "string"},
                },
                "required": ["ground_truth_json", "good_answer", "bad_answer", "garbage_answer"],
            },
        },
        "required": ["env_key", "rationale", "grader_code", "sanity"],
    }
    prompt = (
        f"You tune RL reward graders for a Catan-playing LLM. Self-play surfaced "
        f"this weakness report:\n\n{json.dumps(report, indent=2)}\n\n"
        f"The env you are refining is `{base_env}`. Its ground_truth schema and "
        f"current grader are below — your new grader MUST accept the same `gt` "
        f"schema so existing scenarios stay valid:\n\n```python\n"
        f"{_existing_grader_src(base_env)}\n```\n\n"
        f"Author an IMPROVED `score(text, gt)` that more sharply targets the "
        f"weakness (e.g. if the model over-trades unproductively, widen the gap "
        f"between productive and churning trades). {GRADER_CONTRACT}\n"
        f"Also give a `sanity` block: `ground_truth_json` (a representative gt dict "
        f"encoded as a JSON string), a `good_answer` your grader should score HIGH, "
        f"a `bad_answer` it should score LOWER, and a `garbage_answer` (unparseable) "
        f"it must score <= 0. Set env_key to `{base_env}_v2`."
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return json.loads(text)


def sanity_check(proposal):
    """Hard gate. Returns (ok: bool, reasons: list[str])."""
    reasons = []
    code = proposal.get("grader_code", "")
    m = _FORBIDDEN.search(code)
    if m:
        return False, [f"forbidden construct {m.group(0)!r} in grader_code"]
    if "def score" not in code:
        return False, ["grader_code does not define score()"]

    # exec in a restricted namespace
    safe_builtins = {"len": len, "range": range, "min": min, "max": max,
                     "abs": abs, "float": float, "int": int, "str": str,
                     "sorted": sorted, "sum": sum, "round": round, "bool": bool,
                     "enumerate": enumerate, "zip": zip, "dict": dict,
                     "list": list, "set": set, "tuple": tuple, "any": any,
                     "all": all, "map": map, "filter": filter, "reversed": reversed,
                     "divmod": divmod, "pow": pow, "ord": ord, "chr": chr,
                     "repr": repr, "next": next, "iter": iter,
                     "isinstance": isinstance, "ValueError": ValueError,
                     "TypeError": TypeError, "KeyError": KeyError,
                     "IndexError": IndexError, "ZeroDivisionError": ZeroDivisionError,
                     "Exception": Exception}
    # allow ONLY these stdlib modules to be imported by the generated grader
    import importlib
    _ALLOWED_MODS = {"re", "math", "json", "collections", "itertools", "string"}

    def _safe_import(name, *a, **k):
        root = name.split(".")[0]
        if root not in _ALLOWED_MODS:
            raise ImportError(f"import of {name!r} not allowed")
        return importlib.import_module(name)

    safe_builtins["__import__"] = _safe_import
    ns = {"__builtins__": safe_builtins}
    try:
        import re as _re
        ns["re"] = _re
        exec(code, ns)
        score = ns["score"]
    except Exception as e:
        return False, [f"grader_code failed to exec: {e}"]

    s = proposal["sanity"]
    try:
        gt = json.loads(s["ground_truth_json"])
    except Exception as e:
        return False, [f"ground_truth_json not valid JSON: {e}"]
    try:
        good = score(s["good_answer"], gt)
        bad = score(s["bad_answer"], gt)
        garbage = score(s["garbage_answer"], gt)
        good2 = score(s["good_answer"], gt)   # determinism
    except Exception as e:
        return False, [f"grader raised on sanity inputs: {e}"]

    for label, r in [("good", good), ("bad", bad), ("garbage", garbage)]:
        if not (isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], (int, float))):
            reasons.append(f"{label}: score() must return (float, str), got {r!r}")
    if reasons:
        return False, reasons

    if good[0] != good2[0]:
        reasons.append("non-deterministic (good answer scored differently twice)")
    if not (good[0] > bad[0]):
        reasons.append(f"good ({good[0]}) not > bad ({bad[0]})")
    if garbage[0] > 0:
        reasons.append(f"garbage scored > 0 ({garbage[0]})")
    if not (-2.5 <= good[0] <= 2.5 and -2.5 <= bad[0] <= 2.5):
        reasons.append("reward out of bounds [-2.5,2.5]")
    return (len(reasons) == 0), reasons


def generate_env(report, base_env, out_dir):
    """Full cycle: propose -> sanity gate -> write env if admitted. Returns a
    record (admitted bool, reasons, paths)."""
    proposal = propose(report, base_env)
    ok, reasons = sanity_check(proposal)
    rec = {"env_key": proposal["env_key"], "base_env": base_env,
           "admitted": ok, "reasons": reasons,
           "rationale": proposal.get("rationale", "")[:500]}
    if ok:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{proposal['env_key']}.py").write_text(proposal["grader_code"])
        (out / f"{proposal['env_key']}.meta.json").write_text(json.dumps(proposal, indent=2))
        rec["grader_path"] = str(out / f"{proposal['env_key']}.py")
    return rec


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True, help="weakness report json")
    p.add_argument("--base-env", required=True, choices=["placement", "maritime", "build"])
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "state" / "envs"))
    a = p.parse_args()
    report = json.loads(Path(a.report).read_text())
    rec = generate_env(report, a.base_env, a.out)
    print(json.dumps(rec, indent=2))
