"""OpenAI backend — a grader model (Claude is the other; dual-grader setup).

Same `complete(system, user) -> str` contract as the Claude/Fireworks backends,
on the stdlib (urllib + json) so it needs no extra dependency and pickles
cleanly into worker processes. Used as an LLM judge over transcripts, and usable
as a player via the same spec flag if ever wanted.

    spec: openai:<model>     (e.g. openai:gpt-4o)

Key resolution: `OPENAI_API_KEY` in the environment
(`export OPENAI_API_KEY="$(scripts/openai_api_key.sh)"`).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from goldilocks_eval.agents.base import LLMBackend

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o"


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048,
                 temperature: float = 0.0, timeout: float = 90.0,
                 max_retries: int = 3):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Run: "
                'export OPENAI_API_KEY="$(scripts/openai_api_key.sh)"')
        self._key = key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.name = f"openai:{model}"

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
                if exc.code in (408, 409, 429) or exc.code >= 500:
                    last_exc = exc
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI request failed after "
                           f"{self.max_retries} attempts: {last_exc}")
