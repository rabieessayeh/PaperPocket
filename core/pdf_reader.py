from __future__ import annotations

from typing import Optional
from io import BytesIO


try:
    import PyPDF2
except Exception:
    PyPDF2 = None


def extract_text(pdf_bytes: bytes, max_pages: Optional[int] = None) -> str:
    """
    Extrait le texte d'un PDF depuis un buffer 'bytes'.
    max_pages : limite le nombre de pages (accélère l’aperçu / test).
    """
    if not pdf_bytes:
        return ""

    if PyPDF2 is None:
        # Fallback très simple si PyPDF2 absent
        return ""

    text_parts = []
    with BytesIO(pdf_bytes) as buf:
        reader = PyPDF2.PdfReader(buf)
        pages = reader.pages
        n = len(pages)
        limit = min(n, max_pages) if max_pages is not None else n
        for i in range(limit):
            try:
                page = pages[i]
                text = page.extract_text() or ""
                text_parts.append(text)
            except Exception:
                # On ignore silencieusement les pages problématiques
                continue
    return "\n\n".join(text_parts).strip()
