from __future__ import annotations

import re
from typing import List


def clean_text(s: str) -> str:
    s = s.replace("\x00", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(
    s: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Greedy splitter par caractères avec chevauchement.
    """
    s = clean_text(s)
    if not s:
        return []

    chunk_size = max(200, int(chunk_size))
    chunk_overlap = max(0, int(chunk_overlap))
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    chunks: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append(s[i:j])
        if j == n:
            break
        i = j - chunk_overlap
    return chunks
