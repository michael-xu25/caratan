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

CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
COMPLETIONS_URL = "https://api.fireworks.ai/inference/v1/completions"
# No public default: our key is account-scoped (it serves the models WE deploy/
# fine-tune, not Fireworks' serverless catalog), so the model id must be passed
# explicitly, e.g. `fireworks:accounts/<acct>/models/<model-id>`.
DEFAULT_MODEL = None


class FireworksBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048,
                 temperature: float = 0.0, timeout: float = 60.0,
                 max_retries: int = 3):
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
        # Base/templateless models reject /chat/completions; flip to /completions.
        self._use_completions = False

    def _post(self, url: str, body: dict) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST",
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Fireworks sits behind Cloudflare, which 403s (error 1010) the
                # default "Python-urllib/x" agent. Any normal UA gets through.
                "User-Agent": "caratan-eval/1.0",
            })
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _common(self) -> dict:
        return {"model": self.model, "max_tokens": self.max_tokens,
                "temperature": self.temperature}

    def complete(self, system: str, user: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self._use_completions:
                    # Templateless/base model: one folded prompt, raw completion.
                    prompt = f"{system}\n\n{user}" if system else user
                    payload = self._post(COMPLETIONS_URL, {**self._common(), "prompt": prompt})
                    return payload["choices"][0]["text"]
                payload = self._post(CHAT_URL, {**self._common(), "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}]})
                return payload["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode()[:300] if hasattr(exc, "read") else ""
                # Models without a chat template reject /chat/completions; switch
                # to /completions for this and all subsequent calls.
                if not self._use_completions and exc.code == 400 and "chat template" in detail:
                    self._use_completions = True
                    continue
                if exc.code in (408, 409, 429) or exc.code >= 500:
                    last_exc = exc
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Fireworks HTTP {exc.code}: {detail}")
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Fireworks request failed after "
                           f"{self.max_retries} attempts: {last_exc}")
