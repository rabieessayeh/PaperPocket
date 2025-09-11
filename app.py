from __future__ import annotations

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["DISABLE_TRANSFORMERS_AV"] = "1"

import streamlit as st
from dotenv import load_dotenv

from pathlib import Path
from typing import Optional

# Local imports
from core import extract_text, chunk_text, summarize_text, prepare_retriever, answer_with_retriever
from core import OllamaLM as LLM
# from core import GPTOSSLM as LLM # for gpt-oss-20b model

try:
    from core.model_local import LocalEmbeddingModel
except Exception:
    LocalEmbeddingModel = None


# -----------------------------------------------------------------------------


class AppState:
    def __init__(self) -> None:
        self.full_text: str = ""
        self.chunks: list[str] = []
        self.retriever = None
        self.emb_model = None
        self.lm: Optional[LLM] = None
        # CHANGEMENT: garder une signature du dernier fichier pour éviter les re-indexations inutiles
        self.last_file_sig: Optional[str] = None


# -----------------------------------------------------------------------------


def _init_state() -> AppState:
    if "state" not in st.session_state:
        st.session_state.state = AppState()
    return st.session_state.state


def _ensure_lm(state: AppState) -> LLM:
    if state.lm is None:
        state.lm = LLM()
    return state.lm


def _ensure_embedder(state: AppState):
    if state.emb_model is not None:
        return state.emb_model

    if LocalEmbeddingModel is None:
        st.error("LocalEmbeddingModel introuvable. Ajoute core/model_local.py ou installe sentence-transformers.")
        return None


    model_name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
    device = os.getenv("EMBED_DEVICE", None)
    dtype = os.getenv("EMBED_DTYPE", "float16")

    state.emb_model = LocalEmbeddingModel(
        model_name=model_name,
        normalize=True,
        uses_e5_format=True,
        device=device,
        dtype=dtype,
    )
    return state.emb_model



def main() -> None:
    load_dotenv(override=True)

    st.set_page_config(page_title=" PaperPocket", page_icon="🧪", layout="wide")
    st.title("🧪 PaperPocket")

    state = _init_state()

    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        st.markdown("---")
        st.markdown("**Chunking**")
        chunk_size = st.number_input("chunk_size", 200, 2000, 800, 50)
        chunk_overlap = st.number_input("chunk_overlap", 0, 800, 100, 10)

        st.markdown("---")
        top_k = st.slider("Retriever top_k", 1, 10, 3, 1)
        min_sim = st.slider("Min cosine similarity", 0.0, 1.0, 0.25, 0.01)

    colL, colR = st.columns([1, 2])

    # ---------------- Upload & parsing ---------------------------------------
    with colL:
        st.subheader("📄 Document")
        file = st.file_uploader("Upload PDF", type=["pdf"])
        if file is not None:
            pdf_bytes = file.read()
            # CHANGEMENT: signature simple pour éviter re-traitement sur reruns identiques
            file_sig = f"{file.name}-{len(pdf_bytes)}"

            # Ne retraiter que si nouveau fichier ou taille différente
            if file_sig != state.last_file_sig:
                state.last_file_sig = file_sig

                # Lecture PDF
                text = extract_text(pdf_bytes)
                state.full_text = text or ""

                # Chunking
                state.chunks = chunk_text(
                    state.full_text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                # Reset retriever si nouveau doc
                state.retriever = None


                with st.spinner("Wait…"):
                    emb = _ensure_embedder(state)
                    if emb is None:
                        st.error("Unable to index the document (embedding template not found).")
                    else:
                        state.retriever = prepare_retriever(
                            state.chunks,
                            emb,
                            top_k=top_k
                        )
                        st.success("The article has been successfully uploaded.")



    # ---------------- Summarization ------------------------------------------
    with colR:
        st.subheader("📝 Summary")
        if state.chunks:
            if st.button("⚡ Summarize"):
                lm = _ensure_lm(state)
                with st.spinner("Summarizing…"):
                    summary = summarize_text(lm, state.chunks)
                st.write(summary)

        st.subheader("Your question about the paper…")
        question = st.chat_input("")
        ask = st.button("Ask")
        if ask:
            if not state.retriever:
                st.warning("The index is not available (indexing failed).")
            else:
                lm = _ensure_lm(state)
                with st.spinner("Wait…"):
                    answer = answer_with_retriever(
                        lm,
                        state.retriever,
                        question,
                        k=top_k,
                        min_sim=min_sim
                    )
                st.write(answer)

    st.caption("")


if __name__ == "__main__":
    main()
