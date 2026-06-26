"""Modal backend — our self-hosted vLLM endpoint (the trained Catan model).

OpenAI-compatible (vLLM), so same `complete(system, user) -> str` contract as the
other backends, on the stdlib so it pickles cleanly into self-play workers.
This is what makes self-play fast: the Modal vLLM server answers in well under a
second vs ~15s on the HUD gateway.

    spec: modal[:<model>]      (model defaults to "catan", the served name)

Env:
    MODAL_LLM_URL   base URL of the deployed serve fn (…modal.run), no trailing /v1
    VLLM_API_KEY    the key set in the `caratan` Modal secret
    MODAL_NO_THINK  "1" (default) appends " /no_think" so Qwen3 answers answer-only
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from goldilocks_eval.agents.base import LLMBackend

DEFAULT_MODEL = "catan"


class ModalBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 32,
                 temperature: float = 0.0, timeout: float = 60.0,
                 max_retries: int = 4):
        base = os.environ.get("MODAL_LLM_URL")
        if not base:
            raise RuntimeError("MODAL_LLM_URL not set (the deployed serve URL).")
        key = os.environ.get("VLLM_API_KEY")
        if not key:
            raise RuntimeError("VLLM_API_KEY not set (from the caratan Modal secret).")
        self._url = base.rstrip("/") + "/v1/chat/completions"
        self._key = key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.no_think = os.environ.get("MODAL_NO_THINK", "1") == "1"
        self.name = f"modal:{model}"

    def complete(self, system: str, user: str) -> str:
        if self.no_think:
            user = user + " /no_think"
        body = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        headers = {"Authorization": f"Bearer {self._key}",
                   "Content-Type": "application/json"}
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(self._url, data=body, headers=headers,
                                         method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read())
                return payload["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as exc:
                if exc.code in (408, 409, 429) or exc.code >= 500:
                    last_exc = exc
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Modal request failed after {self.max_retries} attempts: {last_exc}")
