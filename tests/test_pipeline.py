"""Tests hors ligne : vrais fichiers et vrai graphe, modèles remplacés explicitement."""

import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from docling_core.types.doc import (
    DoclingDocument, DocItemLabel, Size, ProvenanceItem, BoundingBox, ImageRef,
)

from lib import ingest_document, load_knowledge_base, hybrid_retrieval, build_context, create_workflow
from lib.answer import SourcedAnswer, validate_answer, make_answer_messages, citation_schema, sourced_answer_markdown
from lib.context import parent_content_blocks
from lib.retrieval import fuse_parents, rank_dense_parents
from lib.storage import read_json, write_json
from tests.test_workflow import ScriptedLLM, decision, conversation


class WordTokenizer:
    def count_tokens(self, text):
        return len(text.split())


class FakeEncoder:
    """Vecteurs contrôlés pour tester les associations, pas la qualité sémantique."""
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            result = np.zeros(1024, dtype=np.float32)
            result[0] = 1
            return result
        result = np.zeros((len(texts), 1024), dtype=np.float32)
        result[:, 0] = 1
        return result


def sample_document():
    doc = DoclingDocument(name="Guide de test")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    prov = ProvenanceItem(page_no=1, bbox=BoundingBox(l=20, t=20, r=400, b=80), charspan=(0, 10))
    heading = doc.add_heading(text="Méthode de Mack", level=1, prov=prov)
    doc.add_text(label=DocItemLabel.TEXT, text="Mack mesure la variance des provisions.", prov=prov, parent=heading)
    doc.add_text(label=DocItemLabel.TEXT, text="Les hypothèses doivent être vérifiées.", prov=prov, parent=heading)
    return doc


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = self.root / "data"
        self.pdf = self.root / "guide.pdf"
        self.pdf.write_bytes(b"%PDF-1.4 fixture A")
        self.parse = self.enterContext(patch("lib.parsing.parse_document", side_effect=lambda *a: sample_document()))
        self.enterContext(patch("lib.parsing.build_document_converter", return_value=object()))
        self.enterContext(patch("lib.ingestion.get_tokenizer", return_value=WordTokenizer()))
        self.encode = self.enterContext(patch("lib.ingestion.get_encoder", return_value=FakeEncoder()))

    def ingest(self, pdf=None):
        return ingest_document(pdf or self.pdf, self.data, title="Guide de test")

    def test_ingestion_reuses_saved_parsing_and_vectors(self):
        first = self.ingest()
        self.ingest()
        self.assertEqual(self.parse.call_count, 1)
        self.assertEqual(self.encode.call_count, 1)
        self.assertEqual(read_json(self.data / "manifest.json")["documents"], [first["document_id"]])
        base = load_knowledge_base(self.data)
        self.assertEqual(base.vectors.shape, (len(base.children), 1024))

    def test_failed_encoding_is_not_published_and_resumes_without_parsing(self):
        with patch("lib.ingestion.get_encoder", side_effect=RuntimeError("interruption simulée")):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                self.ingest()
        self.assertFalse((self.data / "manifest.json").exists())
        self.ingest()
        self.assertEqual(self.parse.call_count, 1)
        self.assertTrue((self.data / "manifest.json").is_file())

    def test_two_documents_keep_distinct_parents_and_sources(self):
        self.ingest()
        other = self.root / "second.pdf"
        other.write_bytes(b"%PDF-1.4 fixture B")
        self.ingest(other)
        base = load_knowledge_base(self.data)
        self.assertEqual(len(base.records), 2)
        self.assertEqual(len(base.parents_by_id), len(base.parents))
        context = build_context(list(base.parents_by_id), base)
        self.assertEqual(len(context["sources"]), len(base.parents))
        self.assertEqual(len({s["source_url"] for s in context["sources"].values()}), 2)
        self.assertTrue(all(s["pages"] == [1] for s in context["sources"].values()))

    def test_children_partition_parents_exactly(self):
        self.ingest()
        base = load_knowledge_base(self.data)
        for parent in base.parents:
            children = [c for c in base.children if c.metadata["parent_id"] == parent.metadata["parent_id"]]
            self.assertEqual("".join(c.metadata["raw_text"] for c in children), parent.page_content)
            for child in children:
                meta = child.metadata
                self.assertEqual(parent.page_content[meta["start"]:meta["end"]], meta["raw_text"])

    def test_stale_vectors_and_changed_pipeline_are_refused(self):
        result = self.ingest()
        path = Path(result["directory"])
        with patch("lib.retrieval.pipeline_signature", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "pipeline"):
                load_knowledge_base(self.data)
        chunks = read_json(path / "chunks.json")
        chunks["children"][0]["page_content"] = "Contenu modifié"
        write_json(path / "chunks.json", chunks)
        with self.assertRaisesRegex(ValueError, "embeddings"):
            load_knowledge_base(self.data)

    def test_hybrid_keeps_ranked_parents_and_correct_context(self):
        self.ingest()
        base = load_knowledge_base(self.data)
        hits = hybrid_retrieval("Mack provisions", base, encoder=FakeEncoder())
        context = build_context(hits["parent_ids"], base)
        self.assertTrue(hits["bm25_hits"])
        self.assertTrue(hits["dense_hits"])
        self.assertLessEqual(len(context["parents"]), 3)
        self.assertEqual(context["parents"][0]["text"], base.parents_by_id[hits["parent_ids"][0]].page_content)

    def test_hybrid_returns_three_five_or_seven_distinct_ranked_parents(self):
        # Huit vrais parents indexés permettent de vérifier le découpage après RRF.
        doc = sample_document()
        prov = doc.texts[0].prov[0]
        for number in range(7):
            heading = doc.add_heading(text=f"Mack section {number}", level=1, prov=prov)
            doc.add_text(label=DocItemLabel.TEXT, text=f"Mack provisions hypothèse {number}",
                         prov=prov, parent=heading)
        self.parse.side_effect = lambda *args: doc
        self.ingest()
        base = load_knowledge_base(self.data)
        complete = hybrid_retrieval("Mack provisions", base, encoder=FakeEncoder(), context_k=8)
        self.assertEqual(len(complete["parent_ids"]), 8)
        for budget in (3, 5, 7):
            result = hybrid_retrieval("Mack provisions", base, encoder=FakeEncoder(), context_k=budget)
            self.assertEqual(result["parent_ids"], complete["parent_ids"][:budget])
            self.assertEqual(len(set(result["parent_ids"])), budget)

    def test_modified_canonical_document_cannot_supply_old_index_citations(self):
        result = self.ingest()
        path = Path(result["directory"]) / "document.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source a changé"):
            load_knowledge_base(self.data)

    def test_lexical_index_is_reloaded_without_rebuilding(self):
        self.ingest()
        with patch("bm25s.BM25.index", side_effect=AssertionError("réindexation interdite")):
            base = load_knowledge_base(self.data)
        self.assertGreater(len(base.parents), 0)

    def test_graph_rag_branch_uses_real_retrieval_and_citations(self):
        self.ingest()
        base = load_knowledge_base(self.data)
        parent_id = base.parents[0].metadata["parent_id"]
        response = SourcedAnswer(paragraphs=[{"text": "Mack mesure la variance des provisions.",
                                              "source_ids": [parent_id]}], missing_information="")
        fake_llm = ScriptedLLM(
            routes=[decision("retrieve", 5, "Comment Mack mesure la variance des provisions ?"),
                    decision("direct", 3, "merci")],
            answers=[response], direct_answers=["Avec plaisir !"],
        )
        graph = create_workflow(self.data, llm=fake_llm, encoder=FakeEncoder())
        config = conversation("guide")
        result = graph.invoke({"question": "Comment Mack mesure la variance des provisions ?"}, config)
        self.assertEqual(result["route"], "retrieve")
        self.assertEqual(result["context_k"], 5)
        self.assertIn("Méthode de Mack", result["answer"])
        self.assertIn("#page=1", result["answer"])
        self.assertNotIn(parent_id, result["answer"])
        self.assertEqual(set(result["sources"]), {parent_id})
        self.assertEqual(result["messages"][-1].content, result["answer"])
        direct = graph.invoke({"question": "merci"}, config)
        self.assertEqual(direct["context"]["parents"], [])
        self.assertEqual(direct["sources"], {})
        self.assertEqual(direct["retrieval"], {})
        self.assertEqual(direct["context_k"], 0)



class ContractTests(unittest.TestCase):
    def test_math_delimiters_remain_renderable_in_markdown(self):
        answer = SourcedAnswer(paragraphs=[{"text": r"Le facteur est \(\lambda_j\).", "source_ids": ["p"]}],
                               missing_information="")
        source = {"pages": [1], "name": "Mack", "document_title": "Guide", "source_url": "https://example.org/guide.pdf"}
        output = sourced_answer_markdown(answer, {"p": source})
        self.assertIn(r"$\lambda_j$", output)
        self.assertIn("#page=1", output)

    def test_schema_limits_references_to_the_actual_context(self):
        sources = {"doc1:parent-001": {}, "doc2:#/pictures/0": {}}
        schema = citation_schema(sources)
        choices = schema["$defs"]["CitedParagraph"]["properties"]["source_ids"]["items"]["enum"]
        self.assertEqual(set(choices), set(sources))
        self.assertEqual(schema["properties"]["paragraphs"]["maxItems"], 2)

    def test_control_characters_in_generated_formulas_are_rejected(self):
        answer = SourcedAnswer(paragraphs=[{"text": "$\x1clambda_j$", "source_ids": ["p"]}], missing_information="")
        with self.assertRaisesRegex(ValueError, "contrôle"):
            validate_answer(answer, {"p": {}})

    def test_rrf_deduplicates_before_assigning_ranks(self):
        ids, scores = fuse_parents([{"parent_id": "a"}, {"parent_id": "a"}, {"parent_id": "b"}],
                                  [{"parent_id": "b"}])
        self.assertEqual(ids, ["b", "a"])
        self.assertAlmostEqual(scores["b"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(scores["a"], 1 / 61)

    def test_best_child_does_not_accumulate_siblings(self):
        children = {key: SimpleNamespace(metadata={"parent_id": parent})
                    for key, parent in (("a1", "a"), ("a2", "a"), ("b1", "b"))}
        hits = [{"child_id": key, "score": score} for key, score in (("a1", .9), ("a2", .8), ("b1", .7))]
        result = rank_dense_parents(hits, children)
        self.assertEqual([h["parent_id"] for h in result], ["a", "b"])
        self.assertEqual(result[0]["child_id"], "a1")

    def test_unknown_citation_and_uncited_assertion_are_rejected(self):
        for ids in ([], ["inconnue"]):
            response = SourcedAnswer(paragraphs=[{"text": "Assertion", "source_ids": ids}], missing_information="")
            with self.assertRaises(ValueError):
                validate_answer(response, {})

    def test_images_and_text_sent_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure.png"
            pixels = b"image fixture bytes"
            path.write_bytes(pixels)
            parent = {"parent_id": "doc:parent-1", "text": "# Titre\n\nTexte du parent",
                      "images": [{"picture_ref": "doc:#/pictures/0", "path": str(path)}]}
            blocks = parent_content_blocks(parent)
            self.assertEqual(base64.b64decode(blocks[1]["image_url"]["url"].split(",")[1]), pixels)
            messages = make_answer_messages("Question", [parent], {})
            self.assertIn(blocks[0], messages[1].content)
            self.assertIn(blocks[1], messages[1].content)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                parent_content_blocks(parent)

    def test_notebook_contains_no_outputs_or_credentials(self):
        import nbformat
        notebook = nbformat.read(Path(__file__).resolve().parents[1] / "demo.ipynb", as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
            self.assertNotIn("sk-proj-", cell.source)


if __name__ == "__main__":
    unittest.main()
