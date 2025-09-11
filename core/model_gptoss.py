from __future__ import annotations

import os
import requests
from dataclasses import dataclass
from typing import Optional


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


@dataclass
class GPTOSSLM:
    """
    Minimal client for GPT-OSS-20B (OpenAI-compatible API).
    Env variables:
        GPTOSS_HOST (default: http://localhost:8000/v1)
        GPTOSS_MODEL (default: gpt-oss-20b)
        GPTOSS_API_KEY (default: none)
        GPTOSS_TIMEOUT (default: 300)
    """

    model: str = os.getenv("GPTOSS_MODEL", "gpt-oss-20b")
    host: str = os.getenv("GPTOSS_HOST", "http://localhost:8000/v1")
    api_key: str = os.getenv("GPTOSS_API_KEY", "none")

    def __post_init__(self) -> None:
        self.timeout_s: int = _env_int("GPTOSS_TIMEOUT", 300)

    @staticmethod
    def _compose_prompt(user: str, system: Optional[str] = None) -> list[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return messages

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        system: Optional[str] = None,
    ) -> str:
        url = f"{self.host}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": self._compose_prompt(prompt, system),
            "temperature": float(temperature),
            "stream": False,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
