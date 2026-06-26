"""Pure text->reward graders — the single source of truth for the Modal loop.

No hud, no catanatron, stdlib only, so they run inside the minimal TRL training
image AND in eval AND in the grader sanity-gate. Each grader is
`score(text, gt) -> (reward: float, reason: str)`, where `gt` is the row's
`ground_truth` dict. Extracted verbatim from hud_training/catan_*_env.py (which
are now legacy — we are off HUD). Keep these and that file in sync if ever both
are used; going forward THIS is canonical.
"""
import re

# ---------------------------------------------------------------- placement
TOP_K = 3
_PLC_ANS = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _node_id_str(x):
    s = str(x).strip()
    s = s[len("node_"):] if s.startswith("node_") else s
    try:
        return f"node_{int(s)}"
    except (ValueError, TypeError):
        return None


def placement_score(text, gt):
    """reward 1.0 if chosen spot is among the TOP_K highest-scoring legal spots."""
    spot_scores = gt["spot_scores"] if "spot_scores" in gt else gt
    if not spot_scores:
        return 0.0, "no spot_scores"
    m = _PLC_ANS.search(text or "")
    if not m:
        return 0.0, "unparseable"
    chosen = _node_id_str(m.group(1))
    totals = {_node_id_str(k): float(v) for k, v in spot_scores.items()}
    if chosen not in totals:
        return 0.0, f"illegal {chosen}"
    order = sorted(totals.values(), reverse=True)
    threshold = order[min(TOP_K - 1, len(order) - 1)]
    r = 1.0 if totals[chosen] >= threshold else 0.0
    return r, f"{chosen} {'TOP3' if r else 'miss'} (score {totals[chosen]:.2f}, cut {threshold:.2f})"


# ----------------------------------------------------- index parsing (shared)
_ACTION = re.compile(r'"action"\s*:\s*(\d+)')
_IDX_ANS = re.compile(r"<answer>\s*(\d+)\s*</answer>", re.IGNORECASE)
_INT = re.compile(r"\d+")


def _parse_index(text, n):
    for rx in (_ACTION, _IDX_ANS):
        m = rx.search(text or "")
        if m:
            i = int(m.group(1))
            return i if 0 <= i < n else None
    for tok in reversed(_INT.findall(text or "")):
        i = int(tok)
        if 0 <= i < n:
            return i
    return None


# ---------------------------------------------------------------- maritime
_M_BUILD_VALUE = {"BUILD_CITY": 1.0, "BUILD_SETTLEMENT": 0.85,
                  "BUY_DEVELOPMENT_CARD": 0.5, "BUILD_ROAD": 0.4}
_M_CLAMP = (-1.1, 1.2)
_M_INVALID = -1.0


def _maritime_reward(c, w):
    enables = c.get("enables") or []
    if enables:
        r = w["enable"] * max(_M_BUILD_VALUE.get(b, 0.0) for b in enables)
    elif c.get("progresses"):
        r = w["progress"]
    else:
        r = -w["churn"]
    r -= w["scarcity"] * float(c.get("gives_scarce", 0.0))
    lo, hi = _M_CLAMP
    return max(lo, min(hi, r))


def maritime_score(text, gt):
    legal = gt["legal_actions"]
    idx = _parse_index(text, len(legal))
    if idx is None:
        return _M_INVALID, "invalid/illegal index"
    chosen = legal[idx]
    topt = gt["trade_options"]
    if chosen in topt:
        r = _maritime_reward(topt[chosen], gt["weights"])
        return r, f"[{idx}] trade r={r:+.2f} {chosen[:40]}"
    return float(gt.get("no_trade_reward", 0.0)), f"[{idx}] no-trade {chosen[:40]}"


# ---------------------------------------------------------------- build
_B_ROAD_DISCOUNT = 0.6
_B_DEV_VALUE = 1.0
_B_HOARD_OK = 0.3
_B_CLAMP = (-1.5, 2.0)
_B_INVALID = -1.5


def _b_clamp(x):
    lo, hi = _B_CLAMP
    return max(lo, min(hi, x))


def _build_reward(c, w):
    k = c["kind"]
    if k == "settlement":
        r = w["prod"] * c["pip_norm"] + w["div"] * c["diversity"] + w["vp"] * c["vp"]
    elif k == "city":
        r = w["prod"] * c["pip_norm"] + w["vp"] * c["vp"]
    elif k == "road":
        opened = (w["prod"] * c["opens_pip_norm"] + w["div"] * c["opens_diversity"]
                  if c.get("opens_node") is not None else 0.0)
        r = w["road"] * _B_ROAD_DISCOUNT * opened
    elif k == "dev":
        r = w["dev"] * _B_DEV_VALUE
    else:
        r = 0.0
    return _b_clamp(r)


def _hoard_penalty(best_value, w):
    return _b_clamp(-w["hoard"] * max(0.0, best_value - _B_HOARD_OK))


def build_score(text, gt):
    legal = gt["legal_actions"]
    idx = _parse_index(text, len(legal))
    if idx is None:
        return _B_INVALID, "invalid/illegal index"
    chosen = legal[idx]
    opts = gt["build_options"]
    if chosen in opts:
        r = _build_reward(opts[chosen], gt["weights"])
        return r, f"[{idx}] build r={r:+.2f} {chosen[:40]}"
    r = _hoard_penalty(gt["best_value"], gt["weights"])
    return r, f"[{idx}] pass/hoard r={r:+.2f} (forgone best {gt['best_value']:.2f})"


# ---------------------------------------------------------------- registry
GRADERS = {
    "placement": placement_score,
    "maritime": maritime_score,
    "build": build_score,
}


def _completion_text(completion):
    """TRL passes conversational completions as [{'role','content'}]; plain as str."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return str(completion)


def load_score_from_code(code):
    """Exec a generated grader's source and return its `score(text, gt)`.
    Used for autonomous envs (maritime_v2, ...) whose grader was authored by the
    env-gen brain and already passed the sanity gate before admission. Restricted
    builtins as defense-in-depth (it also runs in the Modal sandbox)."""
    import importlib
    allowed = {"re", "math", "json", "collections", "itertools", "string"}

    def _imp(name, *a, **k):
        if name.split(".")[0] not in allowed:
            raise ImportError(f"import {name!r} not allowed")
        return importlib.import_module(name)

    sb = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
          for k in ("len", "range", "min", "max", "abs", "float", "int", "str",
                    "sorted", "sum", "round", "bool", "enumerate", "zip", "dict",
                    "list", "set", "tuple", "any", "all", "map", "filter", "reversed",
                    "divmod", "pow", "ord", "chr", "repr", "next", "iter",
                    "isinstance", "ValueError", "TypeError", "KeyError",
                    "IndexError", "ZeroDivisionError", "Exception")}
    sb["__import__"] = _imp
    ns = {"__builtins__": sb, "re": __import__("re")}
    exec(code, ns)
    return ns["score"]


def make_reward_fn(env_key, scorer=None):
    """Return a TRL GRPOTrainer reward function. Pass `scorer` to use a generated
    grader; otherwise look up the canonical one in the registry.

    TRL calls reward_func(prompts, completions, **cols) where each extra dataset
    column arrives as a batch-aligned list. We require a `ground_truth` column.
    """
    if scorer is None:
        scorer = GRADERS[env_key]

    def reward_func(prompts=None, completions=None, ground_truth=None, **kwargs):
        out = []
        for comp, gt in zip(completions, ground_truth):
            try:
                r, _ = scorer(_completion_text(comp), gt)
            except Exception:
                r = 0.0
            out.append(float(r))
        return out

    reward_func.__name__ = f"reward_{env_key}"
    return reward_func


if __name__ == "__main__":
    # logic check mirroring the legacy env __main__ blocks
    print("placement:", placement_score("<answer>node_5</answer>",
          {"spot_scores": {"node_5": 9.0, "node_2": 3.0, "node_1": 1.0}}))
    print("placement miss:", placement_score("<answer>node_1</answer>",
          {"spot_scores": {"node_5": 9.0, "node_2": 8.0, "node_3": 7.0, "node_1": 1.0}}))
    print("garbage:", placement_score("i dunno", {"spot_scores": {"node_5": 9.0}}))
    print("ALL GOOD")
