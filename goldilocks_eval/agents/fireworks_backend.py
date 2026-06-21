"""Fireworks backend — serves the models we TRAIN (the 'after' in before/after).

Fireworks exposes an OpenAI-compatible chat-completions API, so this is a thin
POST to `/inference/v1/chat/completions`. Implemented on the stdlib (urllib +
json) so it needs no extra dependency and pickles cleanly into the runner's
spawn workers (it holds only strings; the HTTP call happens per `complete`).

Same `complete(system, user) -> str` contract as `ClaudeBackend`, so the agent,
runner, scorer, and transcripts are all unchanged — swapping to a trained model
is just a spec flag: `fireworks:accounts/<acct>/models/<model-id>`.

Key resolution: `FIREWORKS_API_KEY` in the environment (mirror the Claude key
flow: `export FIREWORKS_API_KEY="$(scripts/fireworks_api_key.sh)"`).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from goldilocks_eval.agents.base import LLMBackend

API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
# No public default: our key is account-scoped (it serves the models WE deploy/
# fine-tune, not Fireworks' serverless catalog), so the model id must be passed
# explicitly, e.g. `fireworks:accounts/<acct>/models/<model-id>`.
DEFAULT_MODEL = None


class FireworksBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048,
                 temperature: float = None, timeout: float = 60.0,
                 max_retries: int = 3):
        # Temperature default comes from FIREWORKS_TEMPERATURE when not passed,
        # so it can cross the runner's process-pool boundary (spawned workers
        # inherit the env). Greedy (0.0) is fine for one-shot scoring; self-play
        # wants > 0 so two identical models don't play identically into draws.
        if temperature is None:
            temperature = float(os.environ.get("FIREWORKS_TEMPERATURE", "0.0"))
        key = os.environ.get("FIREWORKS_API_KEY")
        if not key:
            raise RuntimeError(
                "FIREWORKS_API_KEY not set. Run: "
                'export FIREWORKS_API_KEY="$(scripts/fireworks_api_key.sh)"')
        if not model:
            raise ValueError(
                "Fireworks needs an explicit model id (our key serves only the "
                "account's own deployed/fine-tuned models). Pass it via the spec: "
                "fireworks:accounts/<account>/models/<model-id>")
        self._key = key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.name = f"fireworks:{model}"

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Fireworks sits behind Cloudflare, which 403s (error 1010) the
            # default "Python-urllib/x" agent. Any normal UA gets through.
            "User-Agent": "caratan-eval/1.0",
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(API_URL, data=body, headers=headers,
                                         method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read())
                return payload["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                # Retry transient server / rate-limit errors; fail fast on 4xx.
                if exc.code in (408, 409, 429) or exc.code >= 500:
                    last_exc = exc
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Fireworks request failed after "
                           f"{self.max_retries} attempts: {last_exc}")
