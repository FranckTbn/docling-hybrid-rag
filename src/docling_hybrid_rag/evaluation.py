"""Mesurer la recherche sur des questions dont les preuves sont des éléments du document.

Un fichier JSONL décrit une question par ligne :

    {"query_id": "Q01", "question": "...", "evidence": [{"ref": "#/texts/42", "grade": 3}]}

`ref` est la référence Docling (`self_ref`) de l'élément qui contient la réponse, `grade` sa
pertinence (3 : indispensable, 2 : utile). Une preuve est retrouvée lorsque l'élément figure
parmi ceux du parent (texte, tableau, figure ou titre) ou de l'enfant classé. Ces mesures
vérifient la recherche, pas la qualité de la réponse du modèle.
"""

import json
from pathlib import Path

import numpy as np

from docling_hybrid_rag.retrieval import KnowledgeBase, hybrid_retrieval, retrieve_bm25, retrieve_dense
from docling_hybrid_rag.settings import get_encoder

PRIMARY_GRADE = 3


def load_questions(path: str | Path) -> list[dict]:
    questions = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for question in questions:
        if not question.get("evidence"):
            raise ValueError(f"{question.get('query_id')} n'a aucune preuve.")
    return questions


def parent_refs(parent) -> set[str]:
    meta = parent.metadata
    return {*meta["doc_item_refs"], *meta["picture_refs"], *meta["heading_refs"]}


def first_rank(ranked_refs: list[set[str]], evidence: set[str]) -> int | None:
    """Rang (à partir de 1) du premier élément classé qui contient une preuve."""
    return next((rank for rank, refs in enumerate(ranked_refs, 1) if refs & evidence), None)


def recall_at(ranked_refs: list[set[str]], evidence: set[str], k: int) -> float:
    """Part des preuves indispensables couvertes par les k premiers éléments."""
    return len(evidence & set().union(*ranked_refs[:k], set())) / len(evidence)


def recall_within_budget(ranked_refs: list[set[str]], tokens: list[int], evidence: set[str], budget: int) -> float:
    """Part des preuves couvertes en remplissant un contexte de `budget` tokens, parent après parent.

    Le rang seul avantage les grands parents : un parent immense contient presque toujours la preuve.
    À budget égal, on compare ce que chaque découpage met réellement sous les yeux du modèle.
    """
    used, kept = 0, []
    for refs, size in zip(ranked_refs, tokens):
        if kept and used + size > budget:
            break
        used += size
        kept.append(refs)
    return len(evidence & set().union(*kept, set())) / len(evidence)


def evaluate_retrieval(knowledge_base: KnowledgeBase, questions: list[dict], *, encoder=None, reranker=None,
                       thesaurus=None, ks=(1, 3, 5), channel_k=20, budgets=(1500,), **options) -> dict:
    """Classer les parents de chaque question et mesurer où figurent les preuves.

    Renvoie `rows` (une ligne par question) et `summary` (moyennes). `options` est transmis à
    `hybrid_retrieval` (par exemple `rerank_k`). `budgets` donne des tailles de contexte, en tokens,
    pour `recall_within_budget`.
    """
    encoder = encoder or get_encoder()
    vectors = encoder.encode([q["question"] for q in questions], normalize_embeddings=True)
    children_by_id, parents_by_id = knowledge_base.children_by_id, knowledge_base.parents_by_id
    rows = []
    for question, vector in zip(questions, vectors):
        primary = {e["ref"] for e in question["evidence"] if e["grade"] >= PRIMARY_GRADE}
        anything = {e["ref"] for e in question["evidence"]}
        result = hybrid_retrieval(question["question"], knowledge_base, question_vector=vector, reranker=reranker,
                                  thesaurus=thesaurus, context_k=max(ks + (channel_k,)), **options)
        ranked = [parents_by_id[pid] for pid in result["parent_ids"]]
        refs = [parent_refs(p) for p in ranked]
        dense = retrieve_dense(vector, knowledge_base, channel_k)
        bm25 = retrieve_bm25(result["expansion"], knowledge_base, channel_k)
        child_refs = lambda hits: [set(children_by_id[h["child_id"]].metadata["doc_item_refs"]) for h in hits]
        rows.append({
            "query_id": question["query_id"], "kind": question.get("kind", ""),
            "parent_rank": first_rank(refs, anything),
            "recall": {k: recall_at(refs, primary, k) for k in ks},
            "recall_budget": {b: recall_within_budget(refs, [p.metadata["token_count"] for p in ranked], primary, b)
                              for b in budgets},
            "dense_child_rank": first_rank(child_refs(dense), anything),
            "bm25_child_rank": first_rank(child_refs(bm25), anything),
            "blank_in_dense_top": sum(1 for h in dense if not children_by_id[h["child_id"]].page_content.strip()),
            "context_tokens": {k: sum(p.metadata["token_count"] for p in ranked[:k]) for k in ks},
            "best_score": result["best_score"],
        })
    count = len(rows)
    hit = lambda key, k: sum(1 for r in rows if r[key] is not None and r[key] <= k) / count
    summary = {
        "questions": count,
        "parent_hit": {k: hit("parent_rank", k) for k in ks},
        "parent_mrr": sum(1 / r["parent_rank"] for r in rows if r["parent_rank"]) / count,
        "recall": {k: float(np.mean([r["recall"][k] for r in rows])) for k in ks},
        "recall_budget": {b: float(np.mean([r["recall_budget"][b] for r in rows])) for b in budgets},
        "dense_child_hit": {k: hit("dense_child_rank", k) for k in (5, channel_k)},
        "bm25_child_hit": {k: hit("bm25_child_rank", k) for k in (5, channel_k)},
        "blank_in_dense_top": float(np.mean([r["blank_in_dense_top"] for r in rows])),
        "context_tokens": {k: float(np.mean([r["context_tokens"][k] for r in rows])) for k in ks},
    }
    return {"rows": rows, "summary": summary}
