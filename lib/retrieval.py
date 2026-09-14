"""BM25 sur les parents, dense sur les enfants, RRF sur les parents."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from importlib.metadata import version
from hashlib import sha256

import bm25s
import numpy as np

from lib.settings import get_encoder, CANDIDATE_K, CONTEXT_K
from lib.storage import read_json, load_chunks, load_vectors, pipeline_signature


@dataclass
class KnowledgeBase:
    data_dir: Path
    parents: list
    children: list
    vectors: np.ndarray
    bm25_index: object
    records: dict

    @property
    def parents_by_id(self):
        return {p.metadata["parent_id"]: p for p in self.parents}

    @property
    def children_by_id(self):
        return {c.metadata["child_id"]: c for c in self.children}


def load_knowledge_base(data_dir: str | Path = "data") -> KnowledgeBase:
    """Recharger les deux index une fois pour les questions suivantes."""
    root = Path(data_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Base vide. Appeler ingest_document avec un PDF avant une question documentaire.")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != 1 or not manifest["documents"]:
        raise ValueError("Manifeste vide ou incompatible.")
    parents, children, matrices, records = [], [], [], {}
    signature = pipeline_signature()
    for document_id in manifest["documents"]:
        directory = root / "documents" / document_id
        record = read_json(directory / "record.json")
        if record["document_id"] != document_id or record["pipeline_signature"] != signature:
            raise ValueError("La base ne correspond plus au pipeline courant.")
        if (sha256((directory / "document.json").read_bytes()).hexdigest() != record["canonical_sha256"]
            or sha256((directory / "source.pdf").read_bytes()).hexdigest() != document_id):
            raise ValueError("Le PDF ou le JSON source a changé depuis l'indexation.")
        doc_parents, doc_children = load_chunks(directory)
        matrices.append(load_vectors(directory, doc_children))
        parents.extend(doc_parents)
        children.extend(doc_children)
        records[document_id] = record
    index_dir = root / manifest["bm25"]
    contract = read_json(index_dir / "contract.json")
    if (contract["texts"] != [p.page_content for p in parents]
        or contract["parent_ids"] != [p.metadata["parent_id"] for p in parents]
        or contract["stopwords"] != "fr" or contract["bm25s_version"] != version("bm25s")):
        raise ValueError("L'index BM25 ne correspond plus aux parents ; réindexer la base.")
    index = bm25s.BM25.load(str(index_dir), load_corpus=False)
    return KnowledgeBase(root, parents, children, np.concatenate(matrices), index, records)


def retrieve_bm25(question, knowledge_base, k=CANDIDATE_K):
    query_tokens = bm25s.tokenize([question], stopwords="fr", show_progress=False)
    rows, scores = knowledge_base.bm25_index.retrieve(
        query_tokens, k=min(k, len(knowledge_base.parents)), show_progress=False,
    )
    return [{"parent_id": knowledge_base.parents[int(row)].metadata["parent_id"], "score": float(score)}
            for row, score in zip(rows[0], scores[0]) if score > 0]


def retrieve_dense(question_vector, knowledge_base, k=CANDIDATE_K):
    question_vector = np.asarray(question_vector)
    if (question_vector.shape != (1024,) or not np.isfinite(question_vector).all()
        or not np.isclose(np.linalg.norm(question_vector), 1, atol=1e-4)):
        raise ValueError("Le vecteur de la question doit être normalisé et avoir 1 024 dimensions.")
    similarities = knowledge_base.vectors @ question_vector
    rows = np.argsort(-similarities, kind="stable")[:k]
    return [{"child_id": knowledge_base.children[int(row)].metadata["child_id"],
             "score": float(similarities[row])} for row in rows]


def rank_dense_parents(hits, children_by_id):
    best_by_parent = {}
    # Les enfants arrivent déjà classés par similarité décroissante.
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


def hybrid_retrieval(question: str, knowledge_base: KnowledgeBase, *, encoder=None,
                     candidate_k=CANDIDATE_K, context_k=CONTEXT_K):
    """Retrouver les parents ; les images ne participent pas au classement."""
    if not question.strip() or candidate_k < 1 or context_k < 1:
        raise ValueError("Une question non vide et des budgets positifs sont nécessaires.")
    vector = (encoder or get_encoder()).encode(question, normalize_embeddings=True)
    bm25_hits = retrieve_bm25(question, knowledge_base, candidate_k)
    dense_hits = retrieve_dense(vector, knowledge_base, candidate_k)
    dense_parent_hits = rank_dense_parents(dense_hits, knowledge_base.children_by_id)
    parent_ids, scores = fuse_parents(bm25_hits, dense_parent_hits)
    return {"parent_ids": parent_ids[:context_k], "bm25_hits": bm25_hits,
            "dense_hits": dense_hits, "dense_parent_hits": dense_parent_hits, "rrf_scores": scores}
