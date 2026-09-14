"""Protocole Colab et mémoire réels, modèles et ingestion remplacés explicitement."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from langchain_core.messages import AIMessage

from lib.colab_worker import RESULT_PREFIX, Worker, ingestion_summary, runtime_status, serve
from lib.workflow import create_workflow
from tests.test_workflow import ScriptedLLM, decision


def request(operation, payload=None, identity="test"):
    return {"id": identity, "operation": operation, "payload": payload or {}}


def answer():
    return {"answer": "Bonjour !", "route": "direct", "context_k": 0,
            "route_reason": "Salutation", "search_question": "Bonjour !", "sources": {},
            "context": {"parents": [], "sources": {}}, "retrieval": {},
            "messages": [AIMessage(content="Bonjour !")]}


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"OPENAI_API_KEY": "secret-test-ne-pas-afficher"}))
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.graph = Mock()
        self.graph.invoke.return_value = answer()
        self.factory = self.enterContext(patch("lib.colab_worker.create_workflow", return_value=self.graph))
        self.ingest = self.enterContext(patch("lib.colab_worker.ingest_document"))
        self.worker = Worker()

    def connect(self, **changes):
        payload = {"data_dir": "data", "model": "modele-test", **changes}
        return self.worker.handle(request("hello", payload))

    def test_hello_and_ask_keep_graph_and_environment_only_key(self):
        self.assertTrue(self.connect()["ok"])
        reply = self.worker.handle(request("ask", {"question": "Une suite", "thread_id": "colab"}))
        self.assertTrue(reply["ok"])
        self.factory.assert_called_once_with(data_dir=Path("data").resolve(), env_path=os.devnull, model="modele-test")
        self.assertEqual(self.graph.invoke.call_args_list[0].args[0], {"question": "Bonjour !"})
        self.assertEqual(self.graph.invoke.call_args.args[1], {"configurable": {"thread_id": "colab"}})
        self.assertNotIn(os.environ["OPENAI_API_KEY"], json.dumps(reply))

    def test_real_graph_preserves_conversation_without_any_model(self):
        llm = ScriptedLLM([decision("direct")] * 3, ["Bonjour", "Bonjour Alice", "Alice"])
        self.factory.side_effect = lambda **kwargs: create_workflow(llm=llm)
        self.connect(thread_id="alice")
        self.worker.handle(request("ask", {"question": "Je suis Alice", "thread_id": "alice"}))
        reply = self.worker.handle(request("ask", {"question": "Mon prénom ?", "thread_id": "alice"}))
        self.assertTrue(reply["ok"])
        self.assertEqual([message.content for message in llm.routing_calls[-1][1:]],
                         ["Je suis Alice", "Bonjour Alice", "Mon prénom ?"])
        self.assertEqual(reply["result"]["messages"], [
            {"type": "ai", "text": "Bonjour Alice"}, {"type": "human", "text": "Mon prénom ?"},
            {"type": "ai", "text": "Alice"},
        ])
        self.factory.assert_called_once()

    def test_configuration_change_replaces_graph(self):
        self.connect()
        self.connect()
        self.assertEqual(self.factory.call_count, 1)
        self.connect(model="autre-modele")
        self.assertEqual(self.factory.call_count, 2)

    def test_ingestion_success_invalidates_graph_before_summary(self):
        self.connect()
        self.ingest.return_value = {"document_id": "doc"}
        payload = {"source": "https://example.org/guide.pdf", "title": "Guide", "data_dir": "new-data"}
        with patch("lib.colab_worker.ingestion_summary", return_value={"pages": 30}):
            reply = self.worker.handle(request("ingest", payload))
        self.assertTrue(reply["ok"])
        self.assertIsNone(self.worker.graph)
        self.ingest.assert_called_once_with(payload["source"], Path("new-data").resolve(), title="Guide")
        self.worker.handle(request("ask", {"question": "Une question", "thread_id": "colab"}))
        self.assertEqual(self.factory.call_count, 2)
        self.assertEqual(self.factory.call_args.kwargs["data_dir"], Path("new-data").resolve())
        with patch("lib.colab_worker.ingestion_summary", side_effect=ValueError("secret")):
            self.assertFalse(self.worker.handle(request("ingest", payload))["ok"])
        self.assertIsNone(self.worker.graph)

    def test_failed_ingestion_preserves_existing_graph(self):
        self.connect()
        self.ingest.side_effect = RuntimeError("secret-test-ne-pas-afficher")
        reply = self.worker.handle(request("ingest", {
            "source": "https://example.org/guide.pdf", "title": "Guide", "data_dir": "data",
        }))
        self.assertFalse(reply["ok"])
        self.assertIs(self.worker.graph, self.graph)

    def test_unknown_operations_and_invalid_payloads_never_call_pipeline(self):
        bad = [[], request("exec", {"code": "secret"}), request("status", {"extra": "secret"}),
               request("ask", {"question": "Bonjour", "thread_id": []}),
               request("ask", {"question": "   ", "thread_id": "a"}),
               request("hello", {"data_dir": "data", "api_key": "secret"}),
               request("hello", {"data_dir": "data\x00"}), request("close", identity=True),
               request("ingest", {"source": "file:///tmp/a.pdf", "title": "PDF", "data_dir": "data"}),
               request("ingest", {"source": "https://user:secret@example.org/a.pdf", "title": "PDF", "data_dir": "data"})]
        for value in bad:
            with self.subTest(value=value):
                reply = self.worker.handle(value)
                self.assertFalse(reply["ok"])
                self.assertNotIn("secret", reply["error"])
        self.ingest.assert_not_called()
        self.factory.assert_not_called()

    def test_missing_key_and_unconfigured_worker_produce_safe_help(self):
        reply = self.worker.handle(request("ask", {"question": "Bonjour", "thread_id": "a"}))
        self.assertFalse(reply["ok"])
        self.assertIn("connexion", reply["error"])
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            reply = self.connect()
        self.assertFalse(reply["ok"])
        self.assertIn("OPENAI_API_KEY", reply["error"])
        self.factory.assert_not_called()

    def test_errors_are_useful_without_exception_message_or_traceback(self):
        self.connect()
        for status, hint in ((401, "Clé"), (403, "Accès"), (429, "Quota"), (404, "Modèle"), (400, "options"), (500, "compatibilité")):
            error = RuntimeError("secret-test-ne-pas-afficher Authorization: Bearer private")
            error.status_code = status
            self.graph.invoke.side_effect = error
            reply = self.worker.handle(request("ask", {"question": "Bonjour", "thread_id": "a"}))
            self.assertFalse(reply["ok"])
            self.assertEqual(reply["error_type"], "RuntimeError")
            self.assertIn(hint, reply["error"])
            self.assertNotIn("secret-test", json.dumps(reply))
            self.assertNotIn("Traceback", json.dumps(reply))

    def test_answer_serializes_numpy_paths_and_only_public_message_fields(self):
        result = answer()
        result["retrieval"] = {"scores": np.array([0.25]), "count": np.int64(1)}
        result["context"] = {"parents": [{"images": [{"path": Path("image.png")}]}]}
        result["sources"] = {"doc:parent-1": {"pages": [1], "source_url": "https://example.org/a.pdf"}}
        result["messages"] = [AIMessage(content="Réponse", response_metadata={"private": "secret"})]
        self.graph.invoke.return_value = result
        reply = self.connect()["result"]
        self.assertEqual(reply["retrieval"], {"scores": [0.25], "count": 1})
        self.assertEqual(reply["context"]["parents"][0]["images"][0]["path"], "image.png")
        self.assertEqual(reply["messages"], [{"type": "ai", "text": "Réponse"}])
        self.assertEqual(reply["sources"], result["sources"])
        json.dumps(reply, allow_nan=False)

    def test_nonfinite_or_unknown_objects_are_not_stringified(self):
        for value in (float("nan"), object()):
            result = answer()
            result["retrieval"] = {"bad": value}
            self.graph.invoke.return_value = result
            reply = self.connect()
            self.assertFalse(reply["ok"])
            self.assertNotIn("object at", json.dumps(reply))

    def test_close_clears_graph_and_refuses_further_operations(self):
        self.connect()
        self.assertEqual(self.worker.handle(request("close"))["result"], {"closed": True})
        self.assertIsNone(self.worker.graph)
        self.assertFalse(self.worker.handle(request("status"))["ok"])

    def test_progress_messages_do_not_echo_request_data(self):
        self.connect(data_dir="private-directory", model="private-model")
        self.worker.handle(request("ask", {"question": "private-question", "thread_id": "private-thread"}))
        with patch("lib.colab_worker.ingestion_summary", return_value={}):
            self.worker.handle(request("ingest", {"source": "https://example.org/private.pdf",
                                                   "title": "private-title", "data_dir": "private-directory"}))
        logs = self.output.getvalue()
        self.assertIn("test de salutation en cours", logs)
        self.assertIn("indexation du PDF en cours", logs)
        self.assertIn("préparation de la réponse en cours", logs)
        self.assertNotIn("private", logs)
        self.assertNotIn(os.environ["OPENAI_API_KEY"], logs)
        self.assertNotIn(RESULT_PREFIX, logs)


class ArtifactAndProtocolTests(unittest.TestCase):
    def test_summary_reads_persisted_counts_coverage_vectors_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "documents" / "doc"
            directory.mkdir(parents=True)
            covered = [page for page in range(1, 31) if page not in {2, 4, 14}]
            document = {"pages": {str(page): {} for page in range(1, 31)},
                        "texts": [{"label": "formula" if page == 3 else "text", "prov": [{"page_no": page}]} for page in covered],
                        "tables": [{"prov": [{"page_no": 21}]}], "pictures": []}
            (directory / "document.json").write_text(json.dumps(document), encoding="utf-8")
            np.savez(directory / "children-embeddings.npz", vectors=np.zeros((2, 1024), dtype=np.float32))
            (root / "manifest.json").write_text(json.dumps({"schema": 1, "documents": ["doc"], "bm25": "indexes/bm25/test"}), encoding="utf-8")
            contract = root / "indexes/bm25/test/contract.json"
            contract.parent.mkdir(parents=True)
            contract.write_text("{}", encoding="utf-8")
            record = {"directory": str(directory), "document_id": "doc", "source_url": "https://example.org/a.pdf"}
            summary = ingestion_summary(record, root, 12.5)
            self.assertEqual(summary, {"record": record, "pages": 30, "covered_pages": covered,
                                      "tables": 1, "formulas": 1, "pictures": 0, "vector_shape": [2, 1024],
                                      "source_url": record["source_url"], "elapsed_seconds": 12.5, "manifest_ok": True})
            contract.unlink()
            self.assertFalse(ingestion_summary(record, root, 0)["manifest_ok"])

    def test_status_queries_cuda_without_loading_a_model(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "GPU simulé"),
                                     version=SimpleNamespace(cuda="12.x"))
        with patch.dict("sys.modules", {"torch": fake_torch}):
            status = runtime_status()
        self.assertTrue(status["python_version"])
        self.assertTrue(status["python_executable"])
        self.assertTrue(status["cuda_available"])
        self.assertEqual(status["cuda_device"], "GPU simulé")
        self.assertEqual(status["cuda_version"], "12.x")

    def test_stream_protocol_survives_bad_json_logs_and_stops_after_close(self):
        source = io.StringIO("{broken secret\n" + "\n".join(json.dumps(item) for item in [
            request("unknown", identity=2), request("status", identity=3), request("close", identity=4),
            request("status", identity=5),
        ]))
        output = io.StringIO()
        def status_with_log():
            # Simuler un log de progression qui n'a pas terminé sa ligne.
            print("Journal ordinaire", end="")
            return {"cuda_available": False}
        with patch("lib.colab_worker.runtime_status", side_effect=status_with_log), redirect_stdout(output):
            serve(source, output)
        lines = output.getvalue().splitlines()
        replies = [json.loads(line[len(RESULT_PREFIX):]) for line in lines if line.startswith(RESULT_PREFIX)]
        self.assertIn("Journal ordinaire", lines)
        self.assertEqual([reply["id"] for reply in replies], [None, 2, 3, 4])
        self.assertEqual([reply["ok"] for reply in replies], [False, False, True, True])
        self.assertNotIn("secret", output.getvalue())


if __name__ == "__main__":
    unittest.main()
