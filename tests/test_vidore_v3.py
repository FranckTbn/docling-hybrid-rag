"""ViDoRe V3 : lecture du jeu, notes de pertinence par page, baseline BM25 et comparaison aux scores publiés.

Un minuscule jeu de la même forme que le vrai (fichiers parquet) est écrit dans un dossier temporaire :
le test ne télécharge rien.
"""

import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None

from docling_hybrid_rag.benchmarks import vidore_v3 as vidore


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def tiny_dataset(root: Path) -> Path:
    target = root / "hr"
    write(target / "documents_metadata" / "test-00000-of-00001.parquet", [
        {"file_name": "rapport-a.pdf", "doc_id": "rapport-a"}, {"file_name": "rapport-b.pdf", "doc_id": "rapport-b"}])
    write(target / "corpus" / "test-00000-of-00001.parquet", [
        {"corpus_id": 0, "doc_id": "rapport-a", "page_number_in_doc": 0, "markdown": "unemployment rate fell in spain"},
        {"corpus_id": 1, "doc_id": "rapport-a", "page_number_in_doc": 1, "markdown": "wages grew slowly across europe"},
        {"corpus_id": 2, "doc_id": "rapport-b", "page_number_in_doc": 0, "markdown": "care work remains undeclared"}])
    write(target / "queries" / "test-00000-of-00001.parquet", [
        {"query_id": 10, "query": "Why did unemployment fall in Spain?", "language": "english",
         "query_types": ["extractive"], "content_type": ["Text"]},
        {"query_id": 11, "query": "Is care work declared?", "language": "english",
         "query_types": ["boolean"], "content_type": ["Text", "Table"]},
        {"query_id": 12, "query": "Pourquoi le chômage a-t-il baissé ?", "language": "french",
         "query_types": ["extractive"], "content_type": ["Text"]},
        {"query_id": 13, "query": "Question sans note de pertinence", "language": "english",
         "query_types": ["extractive"], "content_type": ["Text"]}])
    write(target / "qrels" / "test-00000-of-00001.parquet", [
        {"query_id": 10, "corpus_id": 0, "score": 2}, {"query_id": 10, "corpus_id": 1, "score": 1},
        {"query_id": 11, "corpus_id": 2, "score": 2}, {"query_id": 12, "corpus_id": 0, "score": 2}])
    return target


@unittest.skipIf(pa is None, 'pyarrow absent : pip install "docling-hybrid-rag[benchmarks]"')
class ViDoReReadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.data = Path(cls.directory.name)
        cls.target = tiny_dataset(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_documents_map_their_identifier_to_the_pdf(self):
        documents = vidore.load_documents(self.target)
        self.assertEqual(documents, {"rapport-a": self.target / "pdfs" / "rapport-a.pdf",
                                     "rapport-b": self.target / "pdfs" / "rapport-b.pdf"})

    def test_only_the_dataset_language_and_questions_with_judgements_are_kept(self):
        queries = vidore.load_queries(self.target, "hr")
        self.assertEqual([q["query_id"] for q in queries], [10, 11])
        self.assertEqual(vidore.load_queries(self.target, "physics")[0]["query_id"], 12)

    def test_relevance_is_graded_and_pages_start_at_zero(self):
        first = vidore.load_queries(self.target, "hr")[0]
        self.assertEqual(first["relevance"], {("rapport-a", 0): 2, ("rapport-a", 1): 1})
        self.assertEqual(first["query_types"], ["extractive"])

    def test_bm25_baseline_runs_on_the_provided_page_text(self):
        result = vidore.baseline_bm25("hr", self.data)
        self.assertEqual(result["questions"], 2)
        self.assertGreater(result["ndcg@10"], 50)


class PublishedScoresTests(unittest.TestCase):
    def test_every_dataset_has_a_pinned_revision_and_a_published_column(self):
        for name, spec in vidore.DATASETS.items():
            with self.subTest(dataset=name):
                self.assertEqual(len(spec["revision"]), 40)
                published = vidore.PUBLISHED["english" if spec["language"] == "en" else "french"]
                self.assertTrue(any(spec["column"] in scores for scores in published.values()))

    def test_hr_published_values_are_those_of_the_paper(self):
        english = vidore.PUBLISHED["english"]
        self.assertEqual(english["BM25S (texte)"]["H.R."], 49.6)
        self.assertEqual(english["BGE-M3 (texte)"]["H.R."], 45.3)
        self.assertEqual(english["ColEmbed-3B-v2 (images)"]["H.R."], 65.4)

    def test_comparison_mixes_published_and_measured_scores_best_first(self):
        rows = vidore.comparison("hr", {"docling-hybrid-rag, rrf": 55.0})
        scores = [row["ndcg@10"] for row in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))
        ours = next(row for row in rows if row["source"] == "mesuré ici")
        self.assertEqual(ours["ndcg@10"], 55.0)
        self.assertGreater(len([row for row in rows if row["source"] == "publié"]), 5)


if __name__ == "__main__":
    unittest.main()
