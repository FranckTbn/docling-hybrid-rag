"""nDCG@10 par page : valeurs calculées à la main, passage des enfants aux pages, contexte du pipeline."""

import math
import unittest
from types import SimpleNamespace

from docling_hybrid_rag.page_evaluation import (
    context_recall, evaluate_queries, ndcg_at_k, page_keys, pages_from_children, recall_at_k,
)


def child(title, pages):
    return SimpleNamespace(metadata={"document_title": title, "pages": pages})


class NdcgTests(unittest.TestCase):
    def test_value_computed_by_hand(self):
        relevance = {"A": 2, "B": 1, "C": 1}
        # Gains 1, 2, 0, 1 aux positions 1 à 4 ; meilleur classement possible : 2, 1, 1.
        dcg = 1 / math.log2(2) + 2 / math.log2(3) + 0 + 1 / math.log2(5)
        ideal = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(ndcg_at_k(["B", "A", "X", "C"], relevance), dcg / ideal)

    def test_perfect_ranking_scores_one_and_the_order_of_grades_matters(self):
        relevance = {"A": 2, "B": 1}
        self.assertEqual(ndcg_at_k(["A", "B"], relevance), 1.0)
        self.assertLess(ndcg_at_k(["B", "A"], relevance), 1.0)

    def test_no_relevant_page_found_scores_zero_and_no_judgement_scores_zero(self):
        self.assertEqual(ndcg_at_k(["X", "Y"], {"A": 1}), 0.0)
        self.assertEqual(ndcg_at_k(["A"], {}), 0.0)

    def test_only_the_first_k_pages_count(self):
        ranking = [f"x{i}" for i in range(10)] + ["A"]
        self.assertEqual(ndcg_at_k(ranking, {"A": 2}, k=10), 0.0)
        self.assertGreater(ndcg_at_k(ranking, {"A": 2}, k=11), 0.0)

    def test_ideal_ranking_is_cut_at_k_when_there_are_many_relevant_pages(self):
        relevance = {f"p{i}": 1 for i in range(30)}
        self.assertEqual(ndcg_at_k([f"p{i}" for i in range(10)], relevance, k=10), 1.0)

    def test_recall_ignores_grades_and_counts_pages_found_in_the_top_k(self):
        self.assertEqual(recall_at_k(["A", "X", "B"], {"A": 2, "B": 1, "C": 1}, k=3), 2 / 3)
        self.assertEqual(recall_at_k(["A", "X", "B"], {"A": 2, "B": 1, "C": 1}, k=1), 1 / 3)


class PageMappingTests(unittest.TestCase):
    def test_docling_pages_start_at_one_and_benchmark_pages_at_zero(self):
        self.assertEqual(page_keys({"document_title": "doc", "pages": [3, 4]}), [("doc", 2), ("doc", 3)])

    def test_a_page_takes_the_rank_of_its_best_child_without_duplicates(self):
        children = {"c1": child("doc", [1, 2]), "c2": child("doc", [2]), "c3": child("other", [1])}
        hits = [{"child_id": "c2"}, {"child_id": "c1"}, {"child_id": "c3"}]
        # c2 place la page 1 d'abord ; c1 ajoute la page 0 (la page 1 est déjà classée) ; puis l'autre document.
        self.assertEqual(pages_from_children(hits, children), [("doc", 1), ("doc", 0), ("other", 0)])

    def test_context_recall_weights_pages_by_their_grade(self):
        relevance = {("d", 0): 2, ("d", 1): 1}
        self.assertEqual(context_recall({("d", 0)}, relevance), 2 / 3)
        self.assertEqual(context_recall({("d", 0), ("d", 1), ("d", 9)}, relevance), 1.0)
        self.assertEqual(context_recall(set(), relevance), 0.0)
        self.assertEqual(context_recall({("d", 0)}, {}), 0.0)


class EvaluateQueriesTests(unittest.TestCase):
    def test_every_engine_gets_one_row_per_query_and_the_context_is_measured(self):
        from unittest.mock import patch

        queries = [{"query_id": 1, "question": "q", "relevance": {("doc", 0): 2}}]
        rankings = {"bm25": [("doc", 0)], "dense": [("x", 1), ("doc", 0)], "rrf": [("doc", 0), ("x", 1)]}
        with patch("docling_hybrid_rag.page_evaluation.rank_pages", return_value=rankings), \
                patch("docling_hybrid_rag.page_evaluation.context_pages", return_value=({("doc", 0), ("doc", 1)}, 900)):
            result = evaluate_queries(object(), queries, {1: None})
        self.assertEqual(result["variants"]["bm25"][0]["ndcg@10"], 1.0)
        self.assertLess(result["variants"]["dense"][0]["ndcg@10"], 1.0)
        self.assertEqual(result["context"][0], {"query_id": 1, "context_recall": 1.0, "context_pages": 2, "context_tokens": 900})


if __name__ == "__main__":
    unittest.main()
