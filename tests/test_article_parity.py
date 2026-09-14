"""Contrôle optionnel avec le vrai guide et les artefacts de l'article, sans reparsing."""

import os
import sys
import unittest
from pathlib import Path

from lib.chunking import build_parent_chunks, build_child_chunks
from lib.ingestion import _namespace_chunks
from lib.retrieval import KnowledgeBase, hybrid_retrieval
from lib.context import build_context, parent_content_blocks


@unittest.skipUnless(os.getenv("RAG_ARTICLE_DIR"), "Contrôle local optionnel avec les artefacts de l'article.")
class ArticleParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        article_dir = Path(os.environ["RAG_ARTICLE_DIR"]).resolve()
        sys.path.insert(0, str(article_dir))
        from test_indexation_article import IndexationArticleTests
        IndexationArticleTests.setUpClass()
        cls.article = IndexationArticleTests.ns
        cls.parents, cls.parts = build_parent_chunks(
            cls.article["doc"], cls.article["canonical_path"], cls.article["tokenizer"])
        cls.children = build_child_chunks(cls.parents, cls.parts,
                                          cls.article["canonical_path"], cls.article["tokenizer"])

    def test_real_guide_parent_text_metadata_and_children_match_article(self):
        self.assertEqual([p.model_dump() for p in self.parents],
                         [p.model_dump() for p in self.article["parents"]])
        self.assertEqual([c.model_dump() for c in self.children],
                         [c.model_dump() for c in self.article["children"]])

    def test_real_guide_rankings_text_images_and_citations_match_article(self):
        parents = [p.model_copy(deep=True) for p in self.parents]
        children = [c.model_copy(deep=True) for c in self.children]
        _namespace_chunks(parents, children, "guide", "Guide de test")
        for item in [*parents, *children]:
            item.metadata["document_path"] = str(self.article["canonical_path"])
        base = KnowledgeBase(Path("."), parents, children, self.article["vectors"],
                             self.article["bm25_index"], {"guide": {"source_url": self.article["SOURCE_URL"]}})
        vector = self.article["question_vector"]
        class RecordedEncoder:
            def encode(self, question, **kwargs):
                return vector
        hits = hybrid_retrieval(self.article["question"], base, encoder=RecordedEncoder())
        self.assertEqual(hits["parent_ids"], ["guide:" + key for key in self.article["selected_parent_ids"]])
        context = build_context(hits["parent_ids"], base)
        blocks = [b for parent in context["parents"] for b in parent_content_blocks(parent)]
        self.assertEqual(blocks, self.article["context_for_model"])
        for key, original in self.article["sources"].items():
            source = context["sources"]["guide:" + key]
            for field in ("pages", "name", "text", "ref"):
                self.assertEqual(source[field], original[field], (key, field))


if __name__ == "__main__":
    unittest.main()
