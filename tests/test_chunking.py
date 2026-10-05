"""Invariants du chunking : parents sans chevauchement, enfants jamais vides, formules propres."""

import re
import tempfile
import unittest
from pathlib import Path

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc import (
    BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, Size, TableCell, TableData,
)

from docling_hybrid_rag.chunking import PARENT_ID_KEY, _end_of_match, build_parent_child_chunks
from docling_hybrid_rag.parsing import clean_parsed_document, load_document, save_parsed_document


class WordTokenizer(BaseTokenizer):
    """Un token par mot, avec une limite basse pour forcer le découpage des longs éléments."""
    max_tokens: int = 40

    def count_tokens(self, text):
        return len(text.split())

    def get_max_tokens(self):
        return self.max_tokens

    def get_tokenizer(self):
        # semchunk, que HybridChunker appelle pour couper un long élément, attend un compteur de tokens.
        return lambda text: len(text.split())


def sample_document(rows=24):
    """Deux sections : l'une avec un long tableau (découpé en plusieurs chunks) et une formule."""
    doc = DoclingDocument(name="Guide de test")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    prov = ProvenanceItem(page_no=1, bbox=BoundingBox(l=20, t=20, r=400, b=80), charspan=(0, 10))
    # Éléments à plat, comme dans un PDF parsé : le titre n'est pas le parent de son contenu.
    doc.add_heading(text="1. Méthode de Mack", level=1, prov=prov)
    doc.add_text(label=DocItemLabel.TEXT, text="Mack mesure la variance des provisions. " * 4, prov=prov)
    cells = [TableCell(text=f"ligne{r} colonne{c} valeur", start_row_offset_idx=r, end_row_offset_idx=r + 1,
                       start_col_offset_idx=c, end_col_offset_idx=c + 1, column_header=(r == 0))
             for r in range(rows) for c in range(3)]
    doc.add_table(data=TableData(num_rows=rows, num_cols=3, table_cells=cells), prov=prov)
    doc.add_text(label=DocItemLabel.TEXT, text="Conclusion courte du tableau.", prov=prov)
    doc.add_text(label=DocItemLabel.FORMULA, text=r"MSEP = \sigma^2</formula", prov=prov)
    doc.add_heading(text="2. Chain-Ladder", level=1, prov=prov)
    doc.add_text(label=DocItemLabel.TEXT, text="Chain-Ladder projette les sinistres. " * 3, prov=prov)
    return doc


class ChunkingInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = sample_document()
        clean_parsed_document(cls.doc)
        cls.parents, cls.children = build_parent_child_chunks(cls.doc, WordTokenizer())

    def test_the_table_really_is_split_into_several_chunks(self):
        # Sans ce découpage, le test des parents ne reproduirait pas le cas qui posait problème.
        table_children = [c for c in self.children if "ligne" in c.metadata["raw_text"]]
        self.assertGreater(len(self.children), len(self.parents))
        self.assertGreater(len(table_children), 1)

    def test_no_child_is_blank_and_each_has_tokens(self):
        for child in self.children:
            with self.subTest(child=child.metadata["child_id"]):
                self.assertTrue(child.page_content.strip())
                self.assertTrue(child.metadata["raw_text"].strip())
                self.assertGreater(child.metadata["token_count"], 0)

    def test_children_partition_each_parent_exactly(self):
        for parent in self.parents:
            parent_id = parent.metadata[PARENT_ID_KEY]
            kids = [c for c in self.children if c.metadata[PARENT_ID_KEY] == parent_id]
            self.assertTrue(kids)
            cursor = 0
            for child in kids:
                meta = child.metadata
                self.assertEqual(meta["start"], cursor)
                self.assertEqual(parent.page_content[meta["start"]:meta["end"]], meta["raw_text"])
                cursor = meta["end"]
            self.assertEqual(cursor, len(parent.page_content))
            self.assertEqual("".join(c.metadata["raw_text"] for c in kids), parent.page_content)

    def test_parents_never_share_a_document_item(self):
        seen = {}
        for parent in self.parents:
            for ref in parent.metadata["doc_item_refs"]:
                self.assertNotIn(ref, seen, f"{ref} figure dans {seen.get(ref)} et dans {parent.metadata[PARENT_ID_KEY]}")
                seen[ref] = parent.metadata[PARENT_ID_KEY]

    def test_no_parent_is_contained_in_another(self):
        texts = [re.sub(r"\s+", " ", p.page_content) for p in self.parents]
        for i, a in enumerate(texts):
            for j, b in enumerate(texts):
                if i != j:
                    self.assertNotIn(a, b)

    def test_all_chunks_of_a_split_table_share_one_parent(self):
        owners = {c.metadata[PARENT_ID_KEY] for c in self.children if "ligne" in c.metadata["raw_text"]}
        self.assertEqual(len(owners), 1)

    def test_tables_are_serialized_as_markdown_not_triplets(self):
        table_parent = next(p for p in self.parents if "ligne0" in p.page_content)
        self.assertRegex(table_parent.page_content, r"\|\s*-{3,}")
        self.assertNotRegex(table_parent.page_content, r"ligne\d+ colonne\d+ valeur\.[\w]* =")

    def test_children_keep_the_section_titles_in_their_indexed_text(self):
        mack = [c for c in self.children if c.metadata["heading_path"] == "1. Méthode de Mack"]
        self.assertTrue(mack)
        self.assertTrue(all(c.page_content.startswith("1. Méthode de Mack") for c in mack))

    def test_parents_are_whole_sections_by_default_and_larger_than_their_children(self):
        # Une section = un parent : les deux titres du document donnent deux parents.
        self.assertEqual(len(self.parents), 2)
        self.assertEqual(self.parents[0].metadata["heading_path"], "1. Méthode de Mack")
        self.assertEqual(self.parents[1].metadata["heading_path"], "2. Chain-Ladder")
        mack = self.parents[0]
        kids = [c for c in self.children if c.metadata[PARENT_ID_KEY] == mack.metadata[PARENT_ID_KEY]]
        self.assertGreater(len(kids), 1)
        self.assertGreater(mack.metadata["token_count"], max(c.metadata["token_count"] for c in kids))
        # Le texte avant et après le tableau, et la formule, sont dans le même parent.
        for text in ("Mack mesure la variance", "ligne0", "Conclusion courte", "MSEP"):
            self.assertIn(text, mack.page_content)

    def test_a_long_section_is_split_into_consecutive_parents_without_cutting_a_table(self):
        parents, children = build_parent_child_chunks(self.doc, WordTokenizer(), max_parent_tokens=60)
        mack = [p for p in parents if p.metadata["heading_path"] == "1. Méthode de Mack"]
        self.assertGreater(len(mack), 1)
        # Le tableau, découpé en plusieurs chunks, reste entier dans un seul parent, même au-dessus de la limite.
        with_table = [p for p in mack if "ligne0" in p.page_content]
        self.assertEqual(len(with_table), 1)
        self.assertIn("ligne23", with_table[0].page_content)
        for parent in parents:
            kids = [c for c in children if c.metadata[PARENT_ID_KEY] == parent.metadata[PARENT_ID_KEY]]
            self.assertEqual("".join(c.metadata["raw_text"] for c in kids), parent.page_content)
        refs = [ref for parent in parents for ref in parent.metadata["doc_item_refs"]]
        self.assertEqual(len(refs), len(set(refs)))

    def test_without_a_limit_the_whole_section_is_one_parent(self):
        parents, _ = build_parent_child_chunks(self.doc, WordTokenizer(), max_parent_tokens=None)
        self.assertEqual(len(parents), 2)

    def test_elements_scope_keeps_the_same_invariants_with_more_parents(self):
        parents, children = build_parent_child_chunks(self.doc, WordTokenizer(), parent_scope="elements")
        self.assertGreater(len(parents), len(self.parents))
        for parent in parents:
            kids = [c for c in children if c.metadata[PARENT_ID_KEY] == parent.metadata[PARENT_ID_KEY]]
            self.assertEqual("".join(c.metadata["raw_text"] for c in kids), parent.page_content)
        refs = [ref for parent in parents for ref in parent.metadata["doc_item_refs"]]
        self.assertEqual(len(refs), len(set(refs)))
        self.assertEqual(len(children), len(self.children))
        with self.assertRaisesRegex(ValueError, "parent_scope"):
            build_parent_child_chunks(self.doc, WordTokenizer(), parent_scope="chapitre")

    def test_chunks_without_a_title_do_not_merge_into_one_giant_parent(self):
        doc = DoclingDocument(name="Sans titre")
        doc.add_page(page_no=1, size=Size(width=600, height=800))
        prov = ProvenanceItem(page_no=1, bbox=BoundingBox(l=20, t=20, r=400, b=80), charspan=(0, 10))
        for number in range(4):
            # Chaque paragraphe dépasse la limite de 40 mots : il forme son propre chunk.
            doc.add_text(label=DocItemLabel.TEXT, text=f"Paragraphe{number} " + "mot " * 45, prov=prov)
        parents, children = build_parent_child_chunks(doc, WordTokenizer())
        self.assertGreater(len(parents), 1)
        self.assertEqual(len(parents), len({p.metadata[PARENT_ID_KEY] for p in parents}))

    def test_a_title_repeated_later_does_not_merge_distant_sections(self):
        doc = DoclingDocument(name="Titres répétés")
        doc.add_page(page_no=1, size=Size(width=600, height=800))
        prov = ProvenanceItem(page_no=1, bbox=BoundingBox(l=20, t=20, r=400, b=80), charspan=(0, 10))
        for title, body in (("Résultats", "premier bloc"), ("Discussion", "autre bloc"), ("Résultats", "dernier bloc")):
            doc.add_heading(text=title, level=1, prov=prov)
            doc.add_text(label=DocItemLabel.TEXT, text=f"{body} " + "mot " * 20, prov=prov)
        parents, _ = build_parent_child_chunks(doc, WordTokenizer())
        self.assertEqual([p.metadata["heading_path"] for p in parents], ["Résultats", "Discussion", "Résultats"])
        self.assertIn("premier bloc", parents[0].page_content)
        self.assertNotIn("dernier bloc", parents[0].page_content)


class NestedDocumentTests(unittest.TestCase):
    """Markdown ou DOCX : les paragraphes sont rangés sous leur titre, que l'élargissement ne développe pas."""

    @staticmethod
    def nested_document():
        doc = DoclingDocument(name="Guide imbriqué")
        doc.add_page(page_no=1, size=Size(width=600, height=800))
        prov = ProvenanceItem(page_no=1, bbox=BoundingBox(l=20, t=20, r=400, b=80), charspan=(0, 10))
        heading = doc.add_heading(text="Méthode de Mack", level=1, prov=prov)
        doc.add_text(label=DocItemLabel.TEXT, text="Mack mesure la variance des provisions.", prov=prov, parent=heading)
        doc.add_text(label=DocItemLabel.TEXT, text="Les hypothèses doivent être vérifiées.", prov=prov, parent=heading)
        return doc

    def test_parent_falls_back_to_its_children_and_says_so(self):
        parents, children = build_parent_child_chunks(self.nested_document(), WordTokenizer())
        self.assertEqual({p.metadata["parent_origin"] for p in parents}, {"chunks"})
        for parent in parents:
            kids = [c for c in children if c.metadata[PARENT_ID_KEY] == parent.metadata[PARENT_ID_KEY]]
            self.assertEqual("".join(c.metadata["raw_text"] for c in kids), parent.page_content)
            self.assertIn("Mack mesure la variance des provisions.", parent.page_content)
        self.assertTrue(all(c.page_content.strip() for c in children))

    def test_flat_document_parents_come_from_the_expansion(self):
        parents, _ = build_parent_child_chunks(sample_document(), WordTokenizer())
        self.assertEqual({p.metadata["parent_origin"] for p in parents}, {"tree"})


class AlignmentTests(unittest.TestCase):
    def test_alignment_ignores_blank_line_differences(self):
        self.assertEqual(_end_of_match("alpha\nbeta gamma\ndelta", "alpha\n\nbeta  gamma", 0), 16)

    def test_missing_chunk_fails_loudly_instead_of_falling_back(self):
        with self.assertRaisesRegex(ValueError, "introuvable"):
            _end_of_match("alpha beta", "gamma", 0)


class StrayFormulaTagTests(unittest.TestCase):
    def test_loading_removes_the_unclosed_tag_without_touching_the_saved_json(self):
        doc = sample_document()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "source.pdf").write_bytes(b"%PDF-1.4 fixture")
            # Le JSON enregistré garde la sortie brute de Docling.
            save_parsed_document(doc, directory)
            raw = (directory / "document.json").read_text(encoding="utf-8")
            self.assertIn("</formula", raw)
            loaded = load_document(directory / "document.json")
            formula = next(t for t in loaded.texts if t.label == DocItemLabel.FORMULA)
            self.assertEqual(formula.text, r"MSEP = \sigma^2")
            self.assertEqual(formula.orig, r"MSEP = \sigma^2")
            self.assertEqual((directory / "document.json").read_text(encoding="utf-8"), raw)

    def test_a_formula_without_tag_is_left_alone(self):
        doc = sample_document()
        clean_parsed_document(doc)
        self.assertEqual(clean_parsed_document(doc), 0)


if __name__ == "__main__":
    unittest.main()
