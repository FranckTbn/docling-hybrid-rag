"""Benchmarks publics : Open RAG Benchmark (articles arXiv) et ViDoRe V3 (rapports techniques).

Chaque exécution enregistre l'environnement qui l'a produite, pour qu'un lecteur puisse refaire
la même mesure avec les mêmes versions.
"""

import platform
from importlib.metadata import PackageNotFoundError, version

TRACKED = ("docling-hybrid-rag", "docling", "docling-core", "transformers", "sentence-transformers",
           "torch", "bm25s", "PyStemmer", "numpy")


def environment(**extra) -> dict:
    """Versions des bibliothèques qui comptent pour un résultat, plus ce que l'appelant ajoute (jeu de données, révision)."""
    versions = {}
    for name in TRACKED:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    from docling_hybrid_rag.settings import DENSE_REVISION, EMBEDDING_MODEL_ID, RERANKER_MODEL_ID, RERANKER_REVISION

    return {"python": platform.python_version(), "versions": versions,
            "models": {"embedding": f"{EMBEDDING_MODEL_ID}@{DENSE_REVISION}",
                       "reranker": f"{RERANKER_MODEL_ID}@{RERANKER_REVISION}"}, **extra}
