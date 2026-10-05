"""Mesures de recherche : rang de la première preuve, recall, MRR, enfants vides."""

import json
import tempfile
import unittest
from pathlib import Path

import bm25s
import numpy as np
from langchain_core.documents import Document

from docling_hybrid_rag.evaluation import (
    evaluate_retrieval, first_rank, load_questions, recall_at, recall_within_budget,
)
from docling_hybrid_rag.indexing import LEXICAL_SETTINGS, lexical_tokens
from docling_hybrid_rag.retrieval import KnowledgeBase


class UnitEncoder:
    def encode(self, texts, **kwargs):
        rows = np.zeros((len(texts), 1024), dtype=np.float32)
        rows[:, 0] = 1
        return rows


def base():
    texts = ["Mack variance des provisions", "Mack hypothèses", "Chain-Ladder facteurs", " "]
    refs = [["#/texts/1"], ["#/texts/2"], ["#/texts/3"], ["#/texts/4"]]
    children = [Document(page_content=t, metadata={"child_id": f"c{i}", "parent_id": f"p{i // 2}", "doc_item_refs": refs[i]})
                for i, t in enumerate(texts)]
    parents = [Document(page_content="", metadata={
        "parent_id": f"p{i}", "doc_item_refs": [r for c in children[2 * i:2 * i + 2] for r in c.metadata["doc_item_refs"]],
        "picture_refs": ["#/pictures/0"] if i == 1 else [], "heading_refs": [], "token_count": 100}) for i in range(2)]
    index = bm25s.BM25()
    index.index(lexical_tokens(texts[:3] + ["vide"]), show_progress=False)
    vectors = np.zeros((4, 1024), dtype=np.float32)
    vectors[:, 0] = 1
    return KnowledgeBase(Path("."), parents, children, vectors, index, {}, LEXICAL_SETTINGS)


class MetricTests(unittest.TestCase):
    def test_first_rank_and_recall_follow_the_ranking(self):
        ranked = [{"a"}, {"b"}, {"c", "d"}]
        self.assertEqual(first_rank(ranked, {"d"}), 3)
        self.assertIsNone(first_rank(ranked, {"z"}))
        self.assertEqual(recall_at(ranked, {"a", "d"}, 1), .5)
        self.assertEqual(recall_at(ranked, {"a", "d"}, 3), 1.0)

    def test_budget_counts_what_fits_and_always_keeps_the_first_parent(self):
        ranked, sizes = [{"a"}, {"b"}, {"d"}], [800, 800, 800]
        self.assertEqual(recall_within_budget(ranked, sizes, {"a", "b", "d"}, 1700), 2 / 3)
        # Même trop grand pour le budget, le premier parent est toujours transmis.
        self.assertEqual(recall_within_budget(ranked, [5000, 10, 10], {"a", "b"}, 1000), .5)


class EvaluationTests(unittest.TestCase):
    def test_evidence_in_the_top_parent_and_blank_children_are_counted(self):
        questions = [{"query_id": "Q1", "question": "Mack variance", "kind": "texte",
                      "evidence": [{"ref": "#/texts/1", "grade": 3}]},
                     {"query_id": "Q2", "question": "Chain-Ladder facteurs", "kind": "texte",
                      "evidence": [{"ref": "#/texts/3", "grade": 3}, {"ref": "#/pictures/0", "grade": 2}]}]
        result = evaluate_retrieval(base(), questions, encoder=UnitEncoder(), ks=(1, 3), channel_k=4)
        first, second = result["rows"]
        self.assertEqual(first["parent_rank"], 1)
        self.assertEqual(first["bm25_child_rank"], 1)
        self.assertEqual(second["parent_rank"], 1)
        # Une preuve de grade 2 ne compte pas dans le recall, qui porte sur les preuves indispensables.
        self.assertEqual(second["recall"][1], 1.0)
        self.assertEqual(first["blank_in_dense_top"], 1)
        self.assertEqual(result["summary"]["parent_hit"][1], 1.0)
        self.assertEqual(result["summary"]["parent_mrr"], 1.0)
        self.assertEqual(first["context_tokens"][1], 100)

    def test_a_picture_attached_to_a_parent_counts_as_evidence(self):
        questions = [{"query_id": "Q3", "question": "Chain-Ladder", "evidence": [{"ref": "#/pictures/0", "grade": 3}]}]
        row = evaluate_retrieval(base(), questions, encoder=UnitEncoder(), ks=(1, 3), channel_k=4)["rows"][0]
        self.assertEqual(row["parent_rank"], 1)

    def test_questions_are_read_from_jsonl_and_must_have_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_text(json.dumps({"query_id": "Q1", "question": "?", "evidence": [{"ref": "#/texts/1", "grade": 3}]}) + "\n",
                            encoding="utf-8")
            self.assertEqual(load_questions(path)[0]["query_id"], "Q1")
            path.write_text(json.dumps({"query_id": "Q2", "question": "?", "evidence": []}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "aucune preuve"):
                load_questions(path)


if __name__ == "__main__":
    unittest.main()
