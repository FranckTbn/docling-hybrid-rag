"""Mesurer la recherche quand la preuve est un passage de texte, comme dans les benchmarks publics.

Un benchmark comme Open RAG Benchmark donne, pour chaque question, le document et la section
qui la contiennent, mais pas les références Docling de notre parsing. On compare donc les
textes : un parent est pertinent s'il partage assez de suites de quatre mots avec la section
attendue. Cette méthode ne dépend ni de la façon dont le PDF a été parsé ni de la taille des parents.

Mesures, pour les k premiers parents d'un classement :
- `doc_hit` : un parent vient du bon document ;
- `hit` : un parent recouvre la section attendue ;
- `coverage` : part de la section attendue présente dans le contexte donné au modèle ;
- `tokens` : taille de ce contexte, car un grand parent couvre plus sans être plus précis.
"""

import random
import re
from dataclasses import replace
from statistics import mean

from docling_hybrid_rag.retrieval import fuse_parents, hybrid_retrieval, rank_parents, retrieve_bm25, retrieve_dense

SHINGLE_SIZE = 4
OVERLAP_THRESHOLD = 0.5
MIN_SHARED = 3
VARIANTS = ("bm25", "dense", "rrf", "rrf+reranker")


def shingles(text: str, size: int = SHINGLE_SIZE) -> frozenset:
    """Suites de `size` mots consécutifs, sans casse ni ponctuation : la mise en forme du parsing ne compte pas."""
    words = re.findall(r"[a-z0-9àâçéèêëîïôûùüÿœ]+", text.casefold())
    return frozenset(tuple(words[i:i + size]) for i in range(len(words) - size + 1))


def overlaps(gold: frozenset, passage: frozenset) -> bool:
    """Le passage est dans la section attendue, ou la contient : coefficient de recouvrement d'au moins 0,5.

    Diviser par la plus petite des deux listes reconnaît un petit enfant entièrement dans la
    section comme un grand parent qui la contient entière.
    """
    shared = len(gold & passage)
    return shared >= MIN_SHARED and shared >= OVERLAP_THRESHOLD * min(len(gold), len(passage))


def evaluate_ranking(knowledge_base, ranking: list[str], gold: frozenset, gold_document: str, *, ks=(1, 3, 5)) -> dict:
    """Mesurer un classement de parents face à une section attendue."""
    parents = [knowledge_base.parents_by_id[parent_id] for parent_id in ranking]
    texts = [shingles(p.page_content) for p in parents]
    relevant = [overlaps(gold, text) for text in texts]
    from_gold = [p.metadata["document_title"] == gold_document for p in parents]
    first = next((rank for rank, flag in enumerate(relevant, 1) if flag), None)
    row = {"first_rank": first, "reciprocal_rank": 1 / first if first else 0.0}
    for k in ks:
        covered = set().union(*texts[:k]) if texts[:k] else set()
        row[f"hit@{k}"] = float(any(relevant[:k]))
        row[f"doc_hit@{k}"] = float(any(from_gold[:k]))
        row[f"coverage@{k}"] = len(gold & covered) / len(gold)
        row[f"tokens@{k}"] = sum(p.metadata["token_count"] for p in parents[:k])
    return row


def rank_all_variants(question: str, vector, knowledge_base, reranker=None) -> dict[str, list[str]]:
    """Classements de parents du même moteur : BM25 seul, dense seul, RRF, et RRF puis reranker."""
    children = knowledge_base.children_by_id
    bm25 = rank_parents(retrieve_bm25(question, knowledge_base), children)
    dense = rank_parents(retrieve_dense(vector, knowledge_base), children)
    rankings = {"bm25": [h["parent_id"] for h in bm25], "dense": [h["parent_id"] for h in dense],
                "rrf": fuse_parents(bm25, dense)[0]}
    if reranker is not None:
        rankings["rrf+reranker"] = hybrid_retrieval(question, knowledge_base, question_vector=vector,
                                                    reranker=reranker, context_k=20)["parent_ids"]
    return rankings


def bootstrap_interval(values: list[float], *, samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Intervalle à 95 % de la moyenne, par rééchantillonnage : avec peu de questions, il compte autant que la moyenne."""
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def summarize(rows: list[dict]) -> dict:
    """Moyenne et intervalle de chaque mesure, sur les questions dont la section est retrouvable."""
    summary = {"questions": len(rows)}
    for key in (k for k in rows[0] if k not in ("first_rank", "query_id", "source", "type")):
        values = [float(row[key]) for row in rows]
        low, high = bootstrap_interval(values)
        summary[key] = {"mean": mean(values), "low": low, "high": high}
    return summary


def with_parent_scope(knowledge_base, knowledge_dir, scope: str):
    """Même base, mêmes enfants et mêmes vecteurs, mais des parents regroupés autrement (pour comparer)."""
    from docling_hybrid_rag.chunking import build_parent_child_chunks
    from docling_hybrid_rag.ingestion import _namespace_chunks
    from docling_hybrid_rag.parsing import load_document
    from docling_hybrid_rag.settings import get_tokenizer

    parents, children = [], []
    for document_id, record in knowledge_base.records.items():
        document = load_document(knowledge_base.knowledge_dir / "documents" / document_id / "document.json")
        new_parents, new_children = build_parent_child_chunks(document, get_tokenizer(), parent_scope=scope)
        _namespace_chunks(new_parents, new_children, document_id, record["title"])
        parents.extend(new_parents)
        children.extend(new_children)
    if [c.metadata["child_id"] for c in children] != [c.metadata["child_id"] for c in knowledge_base.children]:
        raise ValueError("Les enfants ne sont pas dans le même ordre : les vecteurs ne correspondraient plus.")
    # Les identifiants des parents changent avec la portée ; les enfants pointent vers les nouveaux.
    return replace(knowledge_base, parents=parents, children=children)
