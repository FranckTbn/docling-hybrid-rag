"""Essai de bout en bout avec de vrais modèles : plusieurs PDF dans une même base, sans simulation.

Lent (Docling, BGE-M3) : à lancer explicitement.

    RAG_REAL_MODELS=1 python -m unittest tests.test_real_pdfs -v

Trois petits PDF de sujets très différents sont fabriqués ici, ingérés dans une base temporaire, puis
interrogés. Le test vérifie que chaque question retrouve son document avec BM25, avec le dense et avec
la recherche hybride, que les identifiants ne se mélangent pas entre documents et que les sources citées
nomment le bon PDF.
"""

import os
import tempfile
import unittest
from pathlib import Path

from docling_hybrid_rag import build_context, hybrid_retrieval, ingest_document, load_knowledge_base
from docling_hybrid_rag.retrieval import encode_question, rank_parents, retrieve_bm25, retrieve_dense

TOPICS = {
    "provisions": (
        "Provisionnement des sinistres",
        ["La méthode Chain-Ladder estime la charge ultime à partir des coefficients de passage entre deux années de développement.",
         "Les provisions pour sinistres tardifs, appelées IBNR, couvrent les sinistres survenus mais non encore déclarés."],
        "Comment estimer les sinistres survenus mais non déclarés avec les coefficients de passage ?"),
    "astronomie": (
        "Les exoplanètes",
        ["La méthode des transits détecte une exoplanète lorsque son passage devant l'étoile fait baisser la luminosité observée.",
         "La vitesse radiale mesure le décalage Doppler de la lumière d'une étoile entraînée par une planète massive."],
        "Comment détecter une exoplanète quand elle passe devant son étoile ?"),
    "cuisine": (
        "La pâte à crêpes",
        ["La pâte à crêpes se prépare avec de la farine, des oeufs et du lait, en laissant reposer le mélange une heure.",
         "Une poêle bien chaude et un peu de beurre donnent des crêpes fines qui se retournent d'un geste."],
        "Quels ingrédients faut-il pour préparer la pâte à crêpes ?"),
}


def write_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    """Un PDF d'une page, avec un texte sélectionnable, écrit à la main : aucune bibliothèque de plus."""
    lines = [(title, 18)]
    for paragraph in paragraphs:
        words, line = paragraph.split(), ""
        for word in words:
            if len(line) + len(word) > 80:
                lines.append((line, 11))
                line = ""
            line = f"{line} {word}".strip()
        lines.append((line, 11))
        lines.append(("", 11))
    escape = lambda text: text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = "BT\n"
    y = 780
    for text, size in lines:
        stream += f"/F1 {size} Tf 1 0 0 1 56 {y} Tm ({escape(text)}) Tj\n"
        y -= size + 8
    stream += "ET"
    body = stream.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(body)).encode() + b" >>\nstream\n" + body + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, content in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + content + b"\nendobj\n"
    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


@unittest.skipUnless(os.getenv("RAG_REAL_MODELS"), "Essai lent avec Docling et BGE-M3 : définir RAG_REAL_MODELS=1.")
class ThreePdfKnowledgeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.records = {}
        for key, (title, paragraphs, _) in TOPICS.items():
            pdf = root / f"{key}.pdf"
            write_pdf(pdf, title, paragraphs)
            cls.records[key] = ingest_document(pdf, root / "base", title=title)
        cls.knowledge = load_knowledge_base(root / "base")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_each_document_is_ready_with_its_own_identifier(self):
        self.assertEqual(len({r["document_id"] for r in self.records.values()}), 3)
        self.assertEqual(set(self.knowledge.records), {r["document_id"] for r in self.records.values()})
        for item in [*self.knowledge.parents, *self.knowledge.children]:
            self.assertTrue(item.metadata["document_id"] in self.knowledge.records)
        ids = [c.metadata["child_id"] for c in self.knowledge.children]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_question_finds_its_own_document_with_each_engine(self):
        for key, (title, _, question) in TOPICS.items():
            vector = encode_question(question, self.knowledge)
            children = self.knowledge.children_by_id
            engines = {
                "bm25": rank_parents(retrieve_bm25(question, self.knowledge), children),
                "dense": rank_parents(retrieve_dense(vector, self.knowledge), children),
            }
            hybrid = hybrid_retrieval(question, self.knowledge, question_vector=vector, context_k=3)
            for engine, ranking in engines.items():
                with self.subTest(topic=key, engine=engine):
                    first = self.knowledge.parents_by_id[ranking[0]["parent_id"]]
                    self.assertEqual(first.metadata["document_title"], title)
            with self.subTest(topic=key, engine="hybride"):
                first = self.knowledge.parents_by_id[hybrid["parent_ids"][0]]
                self.assertEqual(first.metadata["document_title"], title)

    def test_context_and_sources_name_the_right_pdf(self):
        title, _, question = TOPICS["astronomie"]
        vector = encode_question(question, self.knowledge)
        hits = hybrid_retrieval(question, self.knowledge, question_vector=vector, context_k=1)
        context = build_context(hits["parent_ids"], self.knowledge)
        self.assertEqual({s["document_title"] for s in context["sources"].values()}, {title})
        self.assertIn("exoplanète", context["parents"][0]["text"])


if __name__ == "__main__":
    unittest.main()
