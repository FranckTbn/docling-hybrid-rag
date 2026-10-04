"""Recherche d'une question : BM25 et dense sur les enfants, remontée aux parents,
fusion RRF, puis reranking facultatif des parents candidats."""

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from importlib.metadata import version
from pathlib import Path

import bm25s
import numpy as np

from docling_hybrid_rag.expansion import expand_query
from docling_hybrid_rag.indexing import weighted_scores
from docling_hybrid_rag.settings import get_encoder, CANDIDATE_K, CONTEXT_K, RERANK_K
from docling_hybrid_rag.storage import load_chunks, load_vectors
from docling_hybrid_rag.store import file_sha256, read_catalog, read_json, require_stage


@dataclass
class KnowledgeBase:
    knowledge_dir: Path
    parents: list
    children: list
    vectors: np.ndarray
    bm25_index: object
    records: dict
    lexical_settings: dict

    @cached_property
    def parents_by_id(self):
        return {p.metadata["parent_id"]: p for p in self.parents}

    @cached_property
    def children_by_id(self):
        return {c.metadata["child_id"]: c for c in self.children}


def _check_document(directory: Path, entry: dict) -> None:
    # Chaque fichier doit être celui enregistré par son étape, et chaque étape
    # doit avoir été calculée à partir du fichier actuel de la précédente.
    parsing = require_stage(directory, "parsing")
    chunking = require_stage(directory, "chunking")
    embeddings = require_stage(directory, "embeddings")
    if (file_sha256(directory / "source.pdf") != entry["source_sha256"]
        or parsing["inputs"]["source_sha256"] != entry["source_sha256"]
        or chunking["inputs"]["document_sha256"] != parsing["outputs"]["document.json"]
        or embeddings["inputs"]["chunks_sha256"] != chunking["outputs"]["chunks.json"]):
        raise ValueError(f"Document {directory.name} : les étapes ne se suivent plus. Relancer ingest_document.")


def load_knowledge_base(knowledge_dir: str | Path = "data/knowledge") -> KnowledgeBase:
    """Recharger les deux index une fois pour les questions suivantes."""
    root = Path(knowledge_dir).resolve()
    catalog = read_catalog(root)
    records = {key: entry for key, entry in catalog["documents"].items() if entry["ready"]}
    if not records:
        raise ValueError("Base vide. Appeler ingest_document avec un PDF avant une question documentaire.")
    parents, children, matrices = [], [], []
    for document_id, entry in records.items():
        directory = root / "documents" / document_id
        _check_document(directory, entry)
        doc_parents, doc_children = load_chunks(directory)
        matrices.append(load_vectors(directory, doc_children))
        parents.extend(doc_parents)
        children.extend(doc_children)
    index_dir = root / catalog["bm25"]
    contract = read_json(index_dir / "contract.json")
    if (contract["texts"] != [c.page_content for c in children]
        or contract["child_ids"] != [c.metadata["child_id"] for c in children]
        or contract["versions"] != {name: version(name) for name in ("bm25s", "PyStemmer")}):
        raise ValueError("L'index BM25 ne correspond plus aux enfants ; réindexer la base.")
    index = bm25s.BM25.load(str(index_dir), load_corpus=False, show_progress=False)
    return KnowledgeBase(root, parents, children, np.concatenate(matrices), index, records,
                         contract["settings"])


def retrieve_bm25(query: str | list[dict], knowledge_base: KnowledgeBase, k=CANDIDATE_K):
    """Classer les enfants ; `query` est une question, ou les termes pondérés de `expand_query`."""
    terms = expand_query(query) if isinstance(query, str) else query
    scores = weighted_scores(knowledge_base.bm25_index, terms, knowledge_base.lexical_settings)
    rows = np.argsort(-scores, kind="stable")[:k]
    # Un score nul ne constitue pas une correspondance lexicale.
    return [{"child_id": knowledge_base.children[int(row)].metadata["child_id"], "score": float(scores[row])}
            for row in rows if scores[row] > 0]


def retrieve_dense(question_vector, knowledge_base: KnowledgeBase, k=CANDIDATE_K):
    question_vector = np.asarray(question_vector, dtype=knowledge_base.vectors.dtype)
    if (question_vector.shape != (1024,) or not np.isfinite(question_vector).all()
        or not np.isclose(np.linalg.norm(question_vector), 1, atol=1e-4)):
        raise ValueError("Le vecteur de la question doit être normalisé et avoir 1 024 dimensions.")
    similarities = knowledge_base.vectors @ question_vector
    rows = np.argsort(-similarities, kind="stable")[:k]
    return [{"child_id": knowledge_base.children[int(row)].metadata["child_id"],
             "score": float(similarities[row])} for row in rows]


def rank_parents(hits, children_by_id):
    """Ramener des enfants classés à leurs parents ; chaque parent garde son meilleur enfant."""
    best_by_parent = {}
    # Les enfants arrivent déjà classés par score décroissant.
    for hit in hits:
        parent_id = children_by_id[hit["child_id"]].metadata["parent_id"]
        if parent_id not in best_by_parent:
            best_by_parent[parent_id] = {"parent_id": parent_id, **hit}
    return list(best_by_parent.values())


def fuse_parents(*rankings, constant=60):
    scores = defaultdict(float)
    for ranking in rankings:
        # Un seul vote par parent et par moteur, même en cas de doublon.
        unique_ids = dict.fromkeys(hit["parent_id"] for hit in ranking)
        for rank, parent_id in enumerate(unique_ids, start=1):
            scores[parent_id] += 1 / (constant + rank)
    return sorted(scores, key=lambda parent_id: (-scores[parent_id], parent_id)), dict(scores)


def rerank_parents(question, candidates, best_children, knowledge_base, reranker):
    """Relire le meilleur enfant de chaque canal face à la question ; un parent garde le meilleur score.

    Le reranker reçoit des enfants plutôt que des parents entiers : un parent long
    serait tronqué et jugé sur son seul début.
    """
    pairs, owners = [], []
    for parent_id in candidates:
        for child_id in best_children[parent_id]:
            pairs.append((question, knowledge_base.children_by_id[child_id].page_content))
            owners.append((parent_id, child_id))
    if not pairs:
        return []
    scores = reranker.predict(pairs, show_progress_bar=False)
    best = {}
    for (parent_id, child_id), score in zip(owners, scores):
        if parent_id not in best or score > best[parent_id]["score"]:
            best[parent_id] = {"parent_id": parent_id, "child_id": child_id, "score": float(score)}
    position = {parent_id: index for index, parent_id in enumerate(candidates)}
    # À score égal, l'ordre RRF départage : le classement reste reproductible.
    return sorted(best.values(), key=lambda hit: (-hit["score"], position[hit["parent_id"]]))


def hybrid_retrieval(question: str, knowledge_base: KnowledgeBase, *, encoder=None, question_vector=None,
                     reranker=None, thesaurus=None, policy=None, candidate_k=CANDIDATE_K,
                     rerank_k=RERANK_K, context_k=CONTEXT_K):
    """Retrouver les parents d'une question ; les images ne participent pas au classement.

    `thesaurus` (dictionnaire ou chemin JSON) active l'expansion de la requête pour BM25.
    Le dense garde la question originale : il rapproche déjà les formulations voisines.
    """
    if not question.strip() or min(candidate_k, rerank_k, context_k) < 1:
        raise ValueError("Une question non vide et des budgets positifs sont nécessaires.")
    if question_vector is None:
        question_vector = (encoder or get_encoder()).encode(question, normalize_embeddings=True)
    terms = expand_query(question, thesaurus, policy)
    bm25_hits = retrieve_bm25(terms, knowledge_base, candidate_k)
    dense_hits = retrieve_dense(question_vector, knowledge_base, candidate_k)
    bm25_parent_hits = rank_parents(bm25_hits, knowledge_base.children_by_id)
    dense_parent_hits = rank_parents(dense_hits, knowledge_base.children_by_id)
    fused_ids, rrf_scores = fuse_parents(bm25_parent_hits, dense_parent_hits)

    reranked = None
    ranked_ids = fused_ids
    if reranker is not None:
        candidates = fused_ids[:rerank_k]
        best_children = {parent_id: list(dict.fromkeys(
            hit["child_id"] for hit in (*bm25_parent_hits, *dense_parent_hits) if hit["parent_id"] == parent_id))
            for parent_id in candidates}
        reranked = rerank_parents(question, candidates, best_children, knowledge_base, reranker)
        ranked_ids = [hit["parent_id"] for hit in reranked]

    return {"parent_ids": ranked_ids[:context_k], "expansion": terms,
            "bm25_hits": bm25_hits, "dense_hits": dense_hits,
            "bm25_parent_hits": bm25_parent_hits, "dense_parent_hits": dense_parent_hits,
            "rrf_scores": rrf_scores, "reranked": reranked,
            "best_score": reranked[0]["score"] if reranked else None}
