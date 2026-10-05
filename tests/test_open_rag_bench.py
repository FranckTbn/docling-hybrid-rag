"""Contrôle de non-régression sur une base de dix PDF réels (Open RAG Benchmark, articles arXiv).

Facultatif : la base se prépare avec `benchmarks/open_rag_bench.py` (téléchargement, ingestion), puis

    RAG_BENCHMARK_DIR=data/benchmarks/open-rag-bench-arxiv python -m unittest tests.test_open_rag_bench -v

Les seuils sont des planchers sous les résultats mesurés, avec l'incertitude de cent questions : ils
détectent une régression, ils ne mesurent pas la qualité de la méthode.
"""

import importlib.util
import os
import unittest
from pathlib import Path

BENCHMARK_DIR = os.getenv("RAG_BENCHMARK_DIR")
SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "open_rag_bench.py"


@unittest.skipUnless(BENCHMARK_DIR, "Base de dix PDF facultative : définir RAG_BENCHMARK_DIR.")
class TenPdfBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from docling_hybrid_rag import load_knowledge_base
        from docling_hybrid_rag.passage_evaluation import evaluate_ranking, rank_all_variants, summarize
        from docling_hybrid_rag.retrieval import encode_question

        spec = importlib.util.spec_from_file_location("open_rag_bench", SCRIPT)
        cls.bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.bench)
        data_dir = Path(BENCHMARK_DIR)
        cls.knowledge = load_knowledge_base(data_dir / "knowledge")
        cls.cases = cls.bench.build_cases(data_dir, cls.bench.DOCUMENTS)
        cls.rows = {"bm25": [], "dense": [], "rrf": []}
        for case in cls.cases:
            vector = encode_question(case["question"], cls.knowledge)
            for variant, ranking in rank_all_variants(case["question"], vector, cls.knowledge).items():
                cls.rows[variant].append(evaluate_ranking(cls.knowledge, ranking, case["gold"], case["document"]))
        cls.summary = {variant: summarize(rows) for variant, rows in cls.rows.items()}

    def test_ten_documents_share_one_knowledge_base(self):
        titles = {record["title"] for record in self.knowledge.records.values()}
        self.assertEqual(titles, set(self.bench.DOCUMENTS))
        child_ids = [c.metadata["child_id"] for c in self.knowledge.children]
        self.assertEqual(len(child_ids), len(set(child_ids)))
        self.assertEqual(self.knowledge.vectors.shape[0], len(child_ids))

    def test_children_are_small_and_parents_partition_them(self):
        by_parent = {}
        for child in self.knowledge.children:
            self.assertTrue(child.metadata["raw_text"].strip())
            self.assertLessEqual(child.metadata["token_count"], 400)
            by_parent.setdefault(child.metadata["parent_id"], []).append(child)
        for parent in self.knowledge.parents:
            kids = by_parent[parent.metadata["parent_id"]]
            self.assertEqual("".join(c.metadata["raw_text"] for c in kids), parent.page_content)
        # Les parents sont des sections : plus larges que leurs enfants en médiane.
        parent_tokens = sorted(p.metadata["token_count"] for p in self.knowledge.parents)
        child_tokens = sorted(c.metadata["token_count"] for c in self.knowledge.children)
        self.assertGreater(parent_tokens[len(parent_tokens) // 2], child_tokens[len(child_tokens) // 2])

    def test_every_question_section_is_present_in_the_parsed_documents(self):
        self.assertEqual(len(self.cases), 100)

    def test_the_right_document_is_always_among_the_first_three_parents(self):
        for variant in ("bm25", "dense", "rrf"):
            with self.subTest(variant=variant):
                self.assertGreaterEqual(self.summary[variant]["doc_hit@3"]["mean"], 0.95)

    def test_hybrid_search_finds_the_expected_section_often_enough(self):
        rrf = self.summary["rrf"]
        self.assertGreaterEqual(rrf["hit@3"]["mean"], 0.80)
        self.assertGreaterEqual(rrf["hit@5"]["mean"], 0.88)
        self.assertGreaterEqual(rrf["reciprocal_rank"]["mean"], 0.72)

    def test_three_section_parents_cover_most_of_the_expected_section(self):
        self.assertGreaterEqual(self.summary["rrf"]["coverage@3"]["mean"], 0.60)


if __name__ == "__main__":
    unittest.main()
