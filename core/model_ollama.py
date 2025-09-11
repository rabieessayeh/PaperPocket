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
class OllamaLM:
    """
    Client minimaliste pour Ollama avec réglages de performance.
    - Respecte les variables d'env:
        OLLAMA_HOST (default: http://localhost:11434)
        OLLAMA_MODEL (default: llama3.1:8b-instruct-q4_K_M)
        OLLAMA_NUM_CTX (default: 3072)
        OLLAMA_NUM_PREDICT (default: 384)
        OLLAMA_NUM_THREAD (default: nb coeurs/2)
        OLLAMA_KEEP_ALIVE (default: 30m)
    """

    model: str = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def __post_init__(self) -> None:
        self.num_ctx: int = _env_int("OLLAMA_NUM_CTX", 3072)
        self.num_predict: int = _env_int("OLLAMA_NUM_PREDICT", 384)
        # threads ~ coeurs physiques (approximation)
        self.num_thread: int = _env_int("OLLAMA_NUM_THREAD", max((os.cpu_count() or 2) // 2, 1))
        self.keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        self.timeout_s: int = _env_int("OLLAMA_HTTP_TIMEOUT", 300)

    # -- utils -----------------------------------------------------------------

    @staticmethod
    def _compose_prompt(user: str, system: Optional[str] = None) -> str:
        """
        Concat simple 'system + user' pour modèles instruct. Si ton modèle a un
        format spécial, adapte ici.
        """
        if system:
            return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
        return user

    # -- core ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        system: Optional[str] = None,
    ) -> str:
        """
        Renvoie une unique completion non-streamée (plus simple à intégrer dans Streamlit).
        """
        payload = {
            "model": self.model,
            "prompt": self._compose_prompt(prompt, system),
            "stream": False,
            "temperature": float(temperature),
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "num_thread": self.num_thread,
            },
        }
        url = f"{self.host}/api/generate"
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()
