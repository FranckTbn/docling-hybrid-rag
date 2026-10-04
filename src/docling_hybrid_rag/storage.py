"""Relire les chunks et les vecteurs enregistrés d'un document."""

from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from docling_hybrid_rag.settings import EMBEDDING_MODEL_ID, DENSE_REVISION
from docling_hybrid_rag.store import read_json


def load_chunks(directory: Path):
    bundle = read_json(directory / "chunks.json")
    parents = [Document(**item) for item in bundle["parents"]]
    children = [Document(**item) for item in bundle["children"]]
    return parents, children


def load_vectors(directory: Path, children) -> np.ndarray:
    with np.load(directory / "children-embeddings.npz", allow_pickle=False) as saved:
        if (saved["texts"].tolist() != [c.page_content for c in children]
            or saved["child_ids"].tolist() != [c.metadata["child_id"] for c in children]
            or saved["model"].item() != EMBEDDING_MODEL_ID
            or saved["revision"].item() != DENSE_REVISION):
            raise ValueError("Les embeddings ne correspondent plus aux enfants ; réindexer le document.")
        vectors = saved["vectors"]
    if (vectors.shape != (len(children), 1024) or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
        raise ValueError("Les vecteurs BGE-M3 doivent être complets, finis et normalisés.")
    return vectors
