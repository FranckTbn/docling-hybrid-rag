"""Mesurer la recherche au niveau de la page, comme les benchmarks de documents (ViDoRe).

Ces benchmarks donnent, pour chaque question, les pages pertinentes avec une note (1 : utile, 2 :
indispensable) et classent les systèmes par nDCG@10. Notre système recherche des enfants. Chaque
enfant sait sur quelles pages il se trouve : le classement des enfants devient un classement de
pages, dans l'ordre de leur meilleur enfant. Les moteurs sont donc comparés sur les mêmes pages.

La formule est celle de `pytrec_eval`, utilisée par MTEB : gain linéaire (la note), remise
logarithmique, normalisation par le meilleur classement possible.
"""

import math
from collections.abc import Hashable, Iterable

from docling_hybrid_rag.retrieval import fuse_parents, hybrid_retrieval, rank_parents, retrieve_bm25, retrieve_dense

PAGE_DEPTH = 100  # enfants gardés par moteur : assez pour obtenir dix pages distinctes
VARIANTS = ("bm25", "dense", "rrf")


def dcg(gains: Iterable[float]) -> float:
    return sum(gain / math.log2(position + 2) for position, gain in enumerate(gains))


def ndcg_at_k(ranking: list[Hashable], relevance: dict[Hashable, float], k: int = 10) -> float:
    """nDCG@k d'un classement de pages face à leurs notes. Zéro s'il n'y a aucune page pertinente."""
    ideal = sorted(relevance.values(), reverse=True)[:k]
    if not ideal:
        return 0.0
    return dcg(relevance.get(page, 0) for page in ranking[:k]) / dcg(ideal)


def recall_at_k(ranking: list[Hashable], relevance: dict[Hashable, float], k: int = 10) -> float:
    """Part des pages pertinentes présentes parmi les k premières (sans tenir compte des notes)."""
    return len(set(ranking[:k]) & set(relevance)) / len(relevance) if relevance else 0.0


def page_keys(metadata: dict) -> list[tuple[str, int]]:
    """Pages d'un enfant ou d'un parent : (titre du document, numéro de page à partir de 0).

    Docling numérote les pages à partir de 1, les benchmarks à partir de 0.
    """
    return [(metadata["document_title"], page - 1) for page in metadata["pages"]]


def pages_from_children(hits: list[dict], children_by_id: dict) -> list[tuple[str, int]]:
    """Classer les pages d'après leurs enfants : une page prend le rang de son meilleur enfant."""
    ranking: dict[tuple[str, int], None] = {}
    for hit in hits:
        for key in page_keys(children_by_id[hit["child_id"]].metadata):
            ranking.setdefault(key, None)
    return list(ranking)


def rank_pages(question: str, vector, knowledge_base, *, depth: int = PAGE_DEPTH) -> dict[str, list]:
    """Classements de pages de BM25 seul, du dense seul et de leur fusion RRF (constante 60).

    La fusion se fait sur les pages, comme celle du pipeline se fait sur les parents.
    """
    children = knowledge_base.children_by_id
    channels = {"bm25": pages_from_children(retrieve_bm25(question, knowledge_base, depth), children),
                "dense": pages_from_children(retrieve_dense(vector, knowledge_base, depth), children)}
    as_hits = lambda pages: [{"parent_id": page} for page in pages]
    channels["rrf"] = fuse_parents(as_hits(channels["bm25"]), as_hits(channels["dense"]))[0]
    return channels


def context_pages(question: str, vector, knowledge_base, *, context_k: int = 3, **options) -> tuple[set, int]:
    """Pages couvertes par les parents que le pipeline donnerait au modèle, et taille de ce contexte en tokens."""
    result = hybrid_retrieval(question, knowledge_base, question_vector=vector, context_k=context_k, **options)
    parents = [knowledge_base.parents_by_id[parent_id] for parent_id in result["parent_ids"]]
    pages = {key for parent in parents for key in page_keys(parent.metadata)}
    return pages, sum(parent.metadata["token_count"] for parent in parents)


def context_recall(pages: set, relevance: dict[Hashable, float]) -> float:
    """Part des notes des pages pertinentes qui se trouvent dans le contexte : une page indispensable compte double."""
    total = sum(relevance.values())
    return sum(score for page, score in relevance.items() if page in pages) / total if total else 0.0


def evaluate_queries(knowledge_base, queries: list[dict], vectors: dict, *, k: int = 10, context_k: int = 3) -> dict:
    """Mesurer chaque question : nDCG@k et rappel@k de chaque moteur, puis le contexte du pipeline.

    `queries` : liste de {"query_id", "question", "relevance": {(document, page): note}, ...}.
    Renvoie une ligne par question et par variante, prêtes à être résumées.
    """
    rows = {variant: [] for variant in VARIANTS}
    context_rows = []
    for query in queries:
        vector = vectors[query["query_id"]]
        for variant, ranking in rank_pages(query["question"], vector, knowledge_base).items():
            rows[variant].append({"query_id": query["query_id"], f"ndcg@{k}": ndcg_at_k(ranking, query["relevance"], k),
                                  f"recall@{k}": recall_at_k(ranking, query["relevance"], k),
                                  f"recall@{PAGE_DEPTH}": recall_at_k(ranking, query["relevance"], PAGE_DEPTH)})
        pages, tokens = context_pages(query["question"], vector, knowledge_base, context_k=context_k)
        context_rows.append({"query_id": query["query_id"], "context_recall": context_recall(pages, query["relevance"]),
                             "context_pages": len(pages), "context_tokens": tokens})
    return {"variants": rows, "context": context_rows}
