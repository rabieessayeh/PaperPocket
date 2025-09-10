from __future__ import annotations

from typing import List

from core.model_ollama import OllamaLM

SYS_SUMMARY = (
    "You are a helpful research assistant. Write a concise, factual summary. "
    "Prefer bullet points. Avoid speculation. Keep it under 10 bullet points."
)


def summarize_text(lm: OllamaLM, chunks: List[str], max_chunks: int = 12) -> str:
    """
    Map-Reduce summarization pour réduire le contexte et accélérer.
    - Map: résumé court par chunk (3–5 bullets)
    - Reduce: fusion non redondante (≤ 10 bullets)
    """
    if not chunks:
        return "No text available to summarize."

    parts = chunks[:max_chunks]
    mini = []
    for c in parts:
        c = (c or "")[:1200]  # tronque le contexte pour réduire les tokens
        p = (
            "Summarize this excerpt in 3–5 bullet points, factual only. "
            "Mention methods, key results, and limitations if present.\n\n"
            f"{c}\n\nBullets:"
        )
        mini.append(lm.generate(p, system=SYS_SUMMARY))

    combined = "\n".join(mini)
    p2 = (
        "Combine the bullet points below into a single, non-redundant summary "
        "of ≤10 bullets, ordered from most to least important. Include methods, "
        "key results, and limitations.\n\n"
        f"{combined}\n\nFinal summary:"
    )
    return lm.generate(p2, system=SYS_SUMMARY).strip()
