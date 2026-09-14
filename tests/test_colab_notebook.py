"""Exécuter les garde-fous du vrai notebook avec des réponses de processus simulées."""

from contextlib import redirect_stdout
import ast
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "demo_colab.ipynb").read_text(encoding="utf-8"))
CELLS = {cell["id"]: "".join(cell["source"]) for cell in NOTEBOOK["cells"]}
DOCUMENT_ID = "61417ac1abfe7571938f4dc26d5ba88db0b31f996770e68edb26ffc9f0d0888a"


def summary():
    return {
        "record": {"document_id": DOCUMENT_ID, "parents": 3, "children": 4},
        "pages": 30, "covered_pages": sorted(set(range(1, 31)) - {2, 4, 14}),
        "manifest_ok": True, "tables": 2, "formulas": 1, "pictures": 1,
        "vector_shape": [4, 1024], "elapsed_seconds": 12,
    }


class NotebookTests(unittest.TestCase):
    def test_readme_badge_loads_the_same_branch_as_the_installer(self):
        # Le lecteur doit essayer le code de la version qu'il vient d'ouvrir.
        assignments = {}
        for node in ast.parse(CELLS["install"]).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node.value.value
        repository = assignments["DEPOT"].removeprefix("https://github.com/").removesuffix(".git")
        expected = f"https://colab.research.google.com/github/{repository}/blob/{assignments['REFERENCE']}/demo_colab.ipynb"
        self.assertIn(f"]({expected})", (ROOT / "README.md").read_text(encoding="utf-8"))

    def test_format_syntax_clean_outputs_and_six_form_cells(self):
        nbformat.validate(nbformat.from_dict(NOTEBOOK))
        code_cells = [cell for cell in NOTEBOOK["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 6)
        for cell in code_cells:
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            self.assertEqual(cell["metadata"]["cellView"], "form")
            compile("".join(cell["source"]), cell["id"], "exec")
        source = "\n".join(CELLS.values())
        self.assertIn("Copier sur Drive", source)
        self.assertNotIn("drive.mount", source)
        self.assertNotIn("jupyterlab", source)
        self.assertNotIn("write_text(cle", source)
        self.assertIn('userdata.get("OPENAI_API_KEY")', source)
        self.assertIn("del cle", source)

    def namespace(self, receipt=None):
        calls = []

        def request(operation, **payload):
            calls.append((operation, payload))
            return copy.deepcopy(receipt or summary())

        return {
            "session": SimpleNamespace(request=request), "connexion_verifiee": True,
            "PROJET": Path("/content/example"), "uuid": uuid,
        }, calls

    def test_complete_example_enables_questions_with_real_reported_counts(self):
        namespace, calls = self.namespace()
        output = io.StringIO()
        with redirect_stdout(output):
            exec(CELLS["ingest"], namespace)
        self.assertTrue(namespace["document_pret"])
        self.assertIn("3 parents ; 4 enfants", output.getvalue())
        self.assertEqual(calls[0][0], "ingest")
        self.assertTrue(calls[0][1]["source"].startswith("https://"))

    def test_missing_page_blocks_question_even_when_manifest_exists(self):
        receipt = summary()
        receipt["covered_pages"].remove(21)
        namespace, calls = self.namespace(receipt)
        namespace["document_pret"] = True  # Un ancien succès ne doit pas survivre.
        with self.assertRaisesRegex(RuntimeError, "Extraction à vérifier"):
            exec(CELLS["ingest"], namespace)
        self.assertFalse(namespace["document_pret"])
        with self.assertRaisesRegex(RuntimeError, "cellule 3"):
            exec(CELLS["ask"], namespace)
        self.assertEqual(len(calls), 1)

    def test_changed_pdf_blocks_the_old_reference_answers(self):
        receipt = summary()
        receipt["record"]["document_id"] = "changed"
        namespace, _ = self.namespace(receipt)
        with self.assertRaisesRegex(RuntimeError, "fichier publié a changé"):
            exec(CELLS["ingest"], namespace)
        self.assertFalse(namespace["document_pret"])

    def test_no_connection_prevents_expensive_ingestion(self):
        namespace, calls = self.namespace()
        namespace["connexion_verifiee"] = False
        with self.assertRaisesRegex(RuntimeError, "Connexion OpenAI vérifiée"):
            exec(CELLS["ingest"], namespace)
        self.assertEqual(calls, [])

    def test_failed_reinstallation_cannot_reuse_an_old_python_path(self):
        namespace = {"PYTHON_RAG": Path("/stale/python"), "installation_terminee": False,
                     "connexion_verifiee": True, "document_pret": True}
        with self.assertRaisesRegex(RuntimeError, "Installation terminée"):
            exec(CELLS["connect"], namespace)
        self.assertFalse(namespace["connexion_verifiee"])
        self.assertFalse(namespace["document_pret"])

    def test_direct_answer_inspection_has_no_retrieval_or_api_requirement(self):
        namespace = {"resultat": {"route": "direct", "retrieval": {}}}
        output = io.StringIO()
        with redirect_stdout(output):
            exec(CELLS["inspect"], namespace)
        self.assertIn("aucun passage documentaire", output.getvalue())


if __name__ == "__main__":
    unittest.main()
