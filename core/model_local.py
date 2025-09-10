from __future__ import annotations

import os
from typing import Iterable, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def _as_list(x: Iterable[str] | List[str]) -> List[str]:
    return list(x) if isinstance(x, (list, tuple)) else [x]  # pragma: no cover


class LocalEmbeddingModel:
    """
    Encoder local (Sentence-Transformers) optimisé pour RAG.
    - model_name: par défaut 'intfloat/multilingual-e5-small' (rapide & multilingue)
    - normalize: normalise L2 pour cos-sim
    - uses_e5_format: préfixe 'query:' et 'passage:' (schéma e5/bge)
    - device: 'cuda' si dispo, sinon 'cpu'
    - dtype: 'float16' sur GPU pour accélérer/économiser la RAM
    Variables d'env utiles:
      EMBED_BATCH (def=128)
      EMBED_MAX_CHARS (def=4000)  # tronque gentiment les très longs passages
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        normalize: bool = True,
        uses_e5_format: bool = True,
        device: str | None = None,
        dtype: str = "float16",
    ):
        self.normalize = bool(normalize)
        self.uses_e5 = bool(uses_e5_format)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

        # Réduit la séquence max si le modèle le permet (légère accélération)
        try:
            if hasattr(self.model, "max_seq_length"):
                # 512 est souvent suffisant pour des chunks 300–800 tokens
                self.model.max_seq_length = int(os.getenv("EMBED_MAX_TOKENS", "512"))
        except Exception:
            pass

        # Cast en float16 si GPU + demandé
        if dtype == "float16" and self.device.startswith("cuda"):
            try:
                self.model = self.model.half()
            except Exception:
                # Certains backends ne supportent pas .half() → on ignore
                pass

        # Paramètres d'encode
        self.batch_size = int(os.getenv("EMBED_BATCH", "128"))
        self.max_chars = int(os.getenv("EMBED_MAX_CHARS", "4000"))  # coupe textes extrêmes

    # ------------- utils -----------------------------------------------------

    def _prep(self, items: List[tuple[str, str]]) -> List[str]:
        """
        Ajoute les préfixes e5/bge ('query:' vs 'passage:') et tronque si nécessaire.
        """
        out: List[str] = []
        if self.uses_e5:
            for role, text in items:
                txt = (text or "").strip()
                if self.max_chars and len(txt) > self.max_chars:
                    txt = txt[: self.max_chars] + " …"
                prefix = "query:" if role == "q" else "passage:"
                out.append(f"{prefix} {txt}")
        else:
            for _, text in items:
                txt = (text or "").strip()
                if self.max_chars and len(txt) > self.max_chars:
                    txt = txt[: self.max_chars] + " …"
                out.append(txt)
        return out

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)  # dim par défaut; sera remplacée en pratique

        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        # Nettoyage éventuel de NaN/Inf (très rare)
        vecs = np.nan_to_num(vecs, copy=False)
        return vecs

    # ------------- API -------------------------------------------------------

    def encode_passages(self, passages: Iterable[str] | List[str]) -> np.ndarray:
        xs = _as_list(passages)
        prepped = self._prep([("p", t) for t in xs])
        return self._encode(prepped)

    def encode_queries(self, queries: Iterable[str] | List[str]) -> np.ndarray:
        xs = _as_list(queries)
        prepped = self._prep([("q", t) for t in xs])
        return self._encode(prepped)
