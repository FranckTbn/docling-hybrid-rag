"""Contrôle optionnel : les chunks enregistrés de l'article sont ceux que le code produit aujourd'hui.

Lancer avec RAG_ARTICLE_DIR pointant vers le dossier de l'article (formations/RAG). Le test refait le
chunking du vrai guide (sans nouveau parsing) et le compare à `chunks.json`.
"""

import os
import unittest
from pathlib import Path

from docling_hybrid_rag.chunking import build_parent_child_chunks
from docling_hybrid_rag.ingestion import _namespace_chunks
from docling_hybrid_rag.parsing import load_document
from docling_hybrid_rag.settings import get_tokenizer
from docling_hybrid_rag.storage import load_chunks
from docling_hybrid_rag.store import find_document

SOURCE = "https://www.institutdesactuaires.com/global/gene/link.php?doc_id=17739&fg=1"


@unittest.skipUnless(os.getenv("RAG_ARTICLE_DIR"), "Contrôle local optionnel avec la base de l'article.")
class ArticleParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        knowledge_dir = Path(os.environ["RAG_ARTICLE_DIR"]).resolve().parents[1] / "data" / "knowledge"
        directory = find_document(knowledge_dir, SOURCE)
        cls.saved_parents, cls.saved_children = load_chunks(directory)
        document = load_document(directory / "document.json")
        cls.parents, cls.children = build_parent_child_chunks(document, get_tokenizer())
        _namespace_chunks(cls.parents, cls.children, directory.name, cls.saved_parents[0].metadata["document_title"])

    def test_real_guide_parents_and_children_match_saved_chunks(self):
        self.assertEqual([p.model_dump() for p in self.parents], [p.model_dump() for p in self.saved_parents])
        self.assertEqual([c.model_dump() for c in self.children], [c.model_dump() for c in self.saved_children])

    def test_real_guide_has_no_blank_child_and_no_shared_item(self):
        self.assertTrue(all(child.metadata["raw_text"].strip() for child in self.children))
        refs = [ref for parent in self.parents for ref in parent.metadata["doc_item_refs"]]
        self.assertEqual(len(refs), len(set(refs)))


if __name__ == "__main__":
    unittest.main()
