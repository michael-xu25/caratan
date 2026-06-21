"""HUD-gateway backend — drives the Tinker-trained models deployed on HUD
(`Qwen/Qwen3-8B`, `catan-placement-only`, `catan-grpo-q8b`) as players.

OpenAI-compatible gateway, stdlib-only (urllib + json) so it pickles into the
runner's spawn workers. Same `complete(system, user) -> str` contract as the
other backends, so the harness drives it via a spec flag:

    spec: hud:<model>     (e.g. hud:catan-grpo-q8b, hud:Qwen/Qwen3-8B)

Key: `HUD_API_KEY` in the env (in `.env`). We append `/no_think` to the user
message — same as eval_holdout.py — so the base Qwen3-8B answers directly instead
of reasoning forever and never committing (the trained models already answer
directly, so it's a no-op for them).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from goldilocks_eval.agents.base import LLMBackend

API_URL = "https://inference.beta.hud.ai/v1/chat/completions"
DEFAULT_MODEL = None  # account-scoped; pass an explicit model via the spec


class HudBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 512,
                 temperature: float = None, timeout: float = 90.0, max_retries: int = 4):
        if temperature is None:
            temperature = float(os.environ.get("FIREWORKS_TEMPERATURE", "0.7"))
        key = os.environ.get("HUD_API_KEY")
        if not key:
            raise RuntimeError("HUD_API_KEY not set (add it to .env).")
        if not model:
            raise ValueError("HUD needs an explicit model id, e.g. hud:catan-grpo-q8b")
        self._key = key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.name = f"hud:{model}"

    def complete(self, system: str, user: str) -> str:
        # /no_think so the base model commits to an answer (parity with eval_holdout).
        body = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user + " /no_think"},
            ],
        }).encode()
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "User-Agent": "caratan-eval/1.0",
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(API_URL, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read())
                return payload["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as exc:
                if exc.code in (408, 409, 429) or exc.code >= 500:
                    last_exc = exc
                    time.sleep(2 ** attempt)
                    continue
                detail = exc.read().decode()[:200] if hasattr(exc, "read") else ""
                raise RuntimeError(f"HUD HTTP {exc.code}: {detail}")
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"HUD request failed after {self.max_retries} attempts: {last_exc}")
