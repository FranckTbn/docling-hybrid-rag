"""Contrôle optionnel : le chunking du package reproduit celui de l'article sur le vrai guide.

Lancer avec RAG_ARTICLE_DIR pointant vers le dossier de l'article (formations/RAG).
"""

import os
import sys
import unittest
from pathlib import Path

from docling_hybrid_rag.chunking import build_parent_child_chunks


@unittest.skipUnless(os.getenv("RAG_ARTICLE_DIR"), "Contrôle local optionnel avec les artefacts de l'article.")
class ArticleParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        article_dir = Path(os.environ["RAG_ARTICLE_DIR"]).resolve()
        sys.path.insert(0, str(article_dir))
        from test_parent_child_article import ARTICLE, article_namespace, labeled_cells
        cells = labeled_cells(ARTICLE)
        cls.article = article_namespace(cells)
        for label in ("load-parsed-document", "build-parent-chunks", "build-child-chunks"):
            exec(compile(cells[label], label, "exec"), cls.article)
        cls.parents, cls.children = build_parent_child_chunks(
            cls.article["doc"], cls.article["tokenizer"], cls.article["canonical_path"])

    def test_real_guide_parents_and_children_match_article(self):
        self.assertEqual([p.model_dump() for p in self.parents], [p.model_dump() for p in self.article["parents"]])
        self.assertEqual([c.model_dump() for c in self.children], [c.model_dump() for c in self.article["children"]])


if __name__ == "__main__":
    unittest.main()
