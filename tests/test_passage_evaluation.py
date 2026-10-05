"""Mesure par passages : recouvrement de textes, couverture du contexte, intervalles de confiance."""

import unittest
from types import SimpleNamespace

from langchain_core.documents import Document

from docling_hybrid_rag.passage_evaluation import (
    bootstrap_interval, evaluate_ranking, overlaps, shingles, summarize,
)

GOLD_TEXT = "la méthode de Mack mesure la variance des provisions pour sinistres à payer en assurance non vie"
OTHER_TEXT = "le chat dort sur le canapé du salon pendant que la pluie tombe doucement sur la ville entière"


def parent(identifier, text, title, tokens=None):
    return Document(page_content=text, metadata={"parent_id": identifier, "document_title": title,
                                                 "token_count": tokens or len(text.split())})


def base(*parents):
    return SimpleNamespace(parents_by_id={p.metadata["parent_id"]: p for p in parents})


class ShinglesTests(unittest.TestCase):
    def test_case_punctuation_and_table_markup_do_not_matter(self):
        plain = shingles("La méthode de Mack mesure la variance")
        marked = shingles("**La** méthode | de | Mack, mesure la variance.")
        self.assertEqual(plain, marked)

    def test_text_shorter_than_one_shingle_is_empty(self):
        self.assertEqual(shingles("trois mots seulement"), frozenset())

    def test_a_small_passage_inside_the_gold_and_a_large_parent_containing_it_both_overlap(self):
        gold = shingles(GOLD_TEXT)
        small = shingles("la méthode de Mack mesure la variance des provisions")
        large = shingles(GOLD_TEXT + " " + OTHER_TEXT * 5)
        self.assertTrue(overlaps(gold, small))
        self.assertTrue(overlaps(gold, large))
        self.assertFalse(overlaps(gold, shingles(OTHER_TEXT)))

    def test_a_few_shared_words_are_not_enough(self):
        gold = shingles(GOLD_TEXT)
        weak = shingles("la méthode de Mack est citée une seule fois dans ce passage sur autre chose entièrement")
        self.assertFalse(overlaps(gold, weak))


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.kb = base(parent("a", OTHER_TEXT, "doc-2"),
                       parent("b", GOLD_TEXT + " et un peu de contexte autour", "doc-1"),
                       parent("c", OTHER_TEXT + " encore", "doc-1"))
        self.gold = shingles(GOLD_TEXT)

    def test_rank_coverage_and_document_hit_follow_the_ranking(self):
        row = evaluate_ranking(self.kb, ["a", "b", "c"], self.gold, "doc-1")
        self.assertEqual(row["first_rank"], 2)
        self.assertEqual(row["reciprocal_rank"], 0.5)
        self.assertEqual((row["hit@1"], row["hit@3"]), (0.0, 1.0))
        self.assertEqual((row["doc_hit@1"], row["doc_hit@3"]), (0.0, 1.0))
        self.assertEqual((row["coverage@1"], row["coverage@3"]), (0.0, 1.0))

    def test_tokens_count_the_context_actually_given_to_the_model(self):
        row = evaluate_ranking(self.kb, ["a", "b"], self.gold, "doc-1", ks=(1, 2))
        self.assertEqual(row["tokens@2"], row["tokens@1"] + len(self.kb.parents_by_id["b"].page_content.split()))

    def test_no_relevant_parent_gives_a_zero_reciprocal_rank(self):
        row = evaluate_ranking(self.kb, ["a", "c"], self.gold, "doc-1")
        self.assertIsNone(row["first_rank"])
        self.assertEqual(row["reciprocal_rank"], 0.0)


class SummaryTests(unittest.TestCase):
    def test_interval_contains_the_mean_and_is_reproducible(self):
        values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        low, high = bootstrap_interval(values)
        self.assertLessEqual(low, 0.7)
        self.assertGreaterEqual(high, 0.7)
        self.assertEqual((low, high), bootstrap_interval(values))

    def test_summary_reports_mean_and_bounds_per_measure(self):
        rows = [{"query_id": "q1", "first_rank": 1, "hit@1": 1.0}, {"query_id": "q2", "first_rank": None, "hit@1": 0.0}]
        summary = summarize(rows)
        self.assertEqual(summary["questions"], 2)
        self.assertEqual(summary["hit@1"]["mean"], 0.5)
        self.assertLessEqual(summary["hit@1"]["low"], 0.5)


if __name__ == "__main__":
    unittest.main()
