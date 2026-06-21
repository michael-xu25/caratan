"""TEMPORARY determinism shim — easy to remove/replace.

Why this exists
---------------
Catanatron draws all randomness from Python's **global** `random` module, and
its action ordering depends on Python's hash seed. So reproducible concurrent
runs need two things:

  1. each game isolated in its own process (independent global RNG), and
  2. a pinned `PYTHONHASHSEED` in the worker processes (stable action ordering).

This module is the ONLY place that knows about either. The runner calls
`make_pool()` and nothing else.

How to remove / replace
-----------------------
When the team's own randomness / balanced-dice system lands and owns
reproducibility, you have one integration point:

  * To drop isolation entirely: make `make_pool` return a
    `ThreadPoolExecutor`, or have the runner stop using a pool.
  * To change the hash-seed policy: edit `HASH_SEED` / `PIN_HASH_SEED`.
  * To delete the shim: remove this file and inline a plain
    `concurrent.futures.ProcessPoolExecutor(max_workers=...)` in runner.py.

Nothing else in the harness depends on the details here.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os

# --- knobs (the whole policy lives here) ----------------------------------
PIN_HASH_SEED = True   # set False to stop pinning the worker hash seed
HASH_SEED = "0"        # value used when pinning
START_METHOD = "spawn"  # spawn => fresh interpreter that reads PYTHONHASHSEED


def make_pool(max_workers: int) -> concurrent.futures.ProcessPoolExecutor:
    """Build the process pool the runner fans games out across.

    Pins PYTHONHASHSEED (so spawned workers get stable hash ordering) and forces
    the spawn start method (so the pin actually takes effect on every platform).
    """
    if PIN_HASH_SEED:
        # Must be set in the env BEFORE workers spawn; spawn re-launches a fresh
        # interpreter that reads PYTHONHASHSEED at startup.
        os.environ["PYTHONHASHSEED"] = HASH_SEED
    ctx = multiprocessing.get_context(START_METHOD)
    return concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
