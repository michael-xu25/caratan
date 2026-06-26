import re

_M_BUILD_VALUE = {"BUILD_CITY": 1.0, "BUILD_SETTLEMENT": 0.85,
                  "BUY_DEVELOPMENT_CARD": 0.5, "BUILD_ROAD": 0.4}
_M_CLAMP = (-2.0, 2.0)
_M_INVALID = -1.0
_ENABLE_BONUS = 0.25
_CHURN_MULT = 2.0
_SCARCITY_MULT = 1.5

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


def _clamp(x):
    lo, hi = _M_CLAMP
    return max(lo, min(hi, x))


def _maritime_reward(c, w):
    enables = c.get("enables") or []
    if enables:
        best = max(_M_BUILD_VALUE.get(b, 0.0) for b in enables)
        r = w["enable"] * best + _ENABLE_BONUS
    elif c.get("progresses"):
        r = w["progress"]
    else:
        # unproductive churn: penalize hard so the model stops over-trading
        r = -w["churn"] * _CHURN_MULT
    r -= w["scarcity"] * float(c.get("gives_scarce", 0.0)) * _SCARCITY_MULT
    return _clamp(r)


def score(text, gt):
    legal = gt["legal_actions"]
    idx = _parse_index(text, len(legal))
    if idx is None:
        return _M_INVALID, "invalid/illegal index"
    chosen = legal[idx]
    topt = gt["trade_options"]
    if chosen in topt:
        r = _maritime_reward(topt[chosen], gt["weights"])
        c = topt[chosen]
        kind = "enable" if (c.get("enables")) else ("progress" if c.get("progresses") else "churn")
        return r, f"[{idx}] trade({kind}) r={r:+.2f} {str(chosen)[:40]}"
    return float(gt.get("no_trade_reward", 0.0)), f"[{idx}] no-trade {str(chosen)[:40]}"
