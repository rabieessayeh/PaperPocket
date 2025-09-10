from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

from core.model_ollama import OllamaLM

# -----------------------------------------------------------------------------


@dataclass
class SimpleRetriever:
    """
    Retriever minimal avec index dense en RAM (FAISS-like simplifié).
    - passages: liste de textes
    - passage_vecs: np.ndarray (n, d)
    - encode_query: Callable[[str], np.ndarray(d,)]
    - top_k: nombre de résultats
    """
    passages: List[str]
    passage_vecs: np.ndarray
    encode_query: Callable[[str], np.ndarray]
    top_k: int = 5

    def search(self, query: str, k: int | None = None) -> List[Tuple[int, float]]:
        k = int(k or self.top_k)
        q = self.encode_query(query)
        # cosine similarity
        pv = self.passage_vecs
        # normalisation si nécessaire
        denom = (np.linalg.norm(pv, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
        sims = pv @ q / denom
        idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]


# -----------------------------------------------------------------------------


_SYS_QA = (
    "You are a precise scientific assistant. Use ONLY the provided context. "
    "If the context is insufficient, say you don't know."
)


def _build_context_block(passages: Sequence[str]) -> str:
    lines: List[str] = []
    for i, p in enumerate(passages, 1):
        p = (p or "").strip()
        if not p:
            continue
        if len(p) > 1200:
            p = p[:1200] + " …"
        lines.append(f"[{i}] {p}")
    return "\n\n".join(lines)


def answer_with_retriever(
    lm: OllamaLM,
    retriever: SimpleRetriever,
    question: str,
    k: int = 3,
    min_sim: float = 0.25,
) -> str:
    hits = retriever.search(question, k=k)
    hits = [(i, s) for (i, s) in hits if s >= min_sim]
    if not hits:
        return "Could not retrieve any relevant passages from the index."

    ctx_passages = [retriever.passages[i] for i, _ in hits]
    context = _build_context_block(ctx_passages)

    prompt = (
        "You are given a user question and a set of retrieved context passages.\n"
        "Answer the question strictly based on the context. If insufficient, say you don't know.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
    return lm.generate(prompt, temperature=0.2, system=_SYS_QA).strip()


# -----------------------------------------------------------------------------


def _emb_cache_key(passages: Sequence[str]) -> str:
    return hashlib.sha256("\n\n".join(passages).encode("utf-8")).hexdigest()[:16]


def prepare_retriever(
    passages: List[str],
    emb_model: "LocalEmbeddingModel",
    top_k: int = 5,
) -> SimpleRetriever:
    """
    Prépare un retriever avec cache des embeddings sur disque (.npy).
    """
    os.makedirs(".cache_index", exist_ok=True)
    key = _emb_cache_key(passages)
    path = os.path.join(".cache_index", f"{key}.npy")

    if os.path.exists(path):
        passage_vecs = np.load(path, mmap_mode="r")
    else:
        passage_vecs = emb_model.encode_passages(passages)
        np.save(path, passage_vecs)

    def encode_query_fn(q: str) -> np.ndarray:
        return emb_model.encode_queries([q])[0]

    return SimpleRetriever(
        passages=passages,
        passage_vecs=passage_vecs,
        encode_query=encode_query_fn,
        top_k=top_k,
    )
