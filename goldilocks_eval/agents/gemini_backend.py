"""Google (Gemini / Gemma) backend via the Generative Language API.

Serves both Gemini and **Gemma** models with one `AIza…` key — useful for
running initial samples/tests on Gemma before the Fireworks deployment exists.
Same `complete(system, user) -> str` contract as the other backends, stdlib only.

    spec: gemini:<model>   e.g. gemini:gemma-4-31b-it  or  gemini:gemini-2.5-flash

Gemma quirk: Gemma models on this API reject a separate system instruction, so
for gemma* we fold the system prompt into the user turn; Gemini models use the
native systemInstruction field.

Key resolution: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the environment
(`export GEMINI_API_KEY="$(scripts/gemini_api_key.sh)"`).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from goldilocks_eval.agents.base import LLMBackend

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048,
                 temperature: float = 0.0, timeout: float = 90.0,
                 max_retries: int = 3):
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Run: "
                'export GEMINI_API_KEY="$(scripts/gemini_api_key.sh)"')
        self._key = key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.name = f"gemini:{model}"

    def complete(self, system: str, user: str) -> str:
        is_gemma = self.model.lower().startswith("gemma")
        body = {
            "generationConfig": {"maxOutputTokens": self.max_tokens,
                                 "temperature": self.temperature},
        }
        if is_gemma:
            # Gemma rejects systemInstruction — fold it into the user turn.
            text = f"{system}\n\n{user}" if system else user
            body["contents"] = [{"role": "user", "parts": [{"text": text}]}]
        else:
            body["contents"] = [{"role": "user", "parts": [{"text": user}]}]
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}

        url = f"{BASE}/{self.model}:generateContent?key={self._key}"
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "caratan-eval/1.0"}
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read())
                cands = payload.get("candidates") or []
                if not cands:
                    return ""  # blocked / empty; caller's parse falls back legally
                parts = cands[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts)
            except urllib.error.HTTPError as exc:
                if exc.code in (408, 409, 429) or exc.code >= 500:
                    last_exc = exc
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini request failed after "
                           f"{self.max_retries} attempts: {last_exc}")
