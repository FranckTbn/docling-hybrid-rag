"""Transport réel entre deux processus, sans modèle ni accès réseau."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("colab_support", ROOT / "colab_support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)

FAKE_WORKER = '''
import json, sys
for line in sys.stdin:
    item = json.loads(line)
    operation = item['operation']
    print('Progression du calcul', flush=True)
    if operation == 'die':
        break
    reply = {'id': item['id'], 'ok': True, 'result': {'question': item['payload'].get('question')}}
    if operation == 'bad_id':
        reply['id'] = 'another-request'
    if operation == 'error':
        reply = {'id': item['id'], 'ok': False, 'error': 'Message utile au lecteur'}
    print('RAG_COLAB_RESULT ' + json.dumps(reply), flush=True)
'''


class TransportTests(unittest.TestCase):
    def start_session(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        script = Path(temporary.name) / "fake_worker.py"
        script.write_text(FAKE_WORKER, encoding="utf-8")
        real_popen = subprocess.Popen

        def launch(command, **kwargs):
            return real_popen([sys.executable, "-u", str(script)], **kwargs)

        with patch.object(support.subprocess, "Popen", side_effect=launch) as mocked:
            session = support.ColabSession(sys.executable, temporary.name, api_key="test-secret-value")
        self.addCleanup(session.close)
        return session, mocked.call_args

    def test_streamed_logs_and_multiple_requests_keep_one_process(self):
        session, call = self.start_session()
        captured = io.StringIO()
        with redirect_stdout(captured):
            first = session.request("ask", question="Première question ?")
            second = session.request("ask", question="Et la suite ?")
        self.assertEqual(first["question"], "Première question ?")
        self.assertEqual(second["question"], "Et la suite ?")
        self.assertIn("Progression du calcul", captured.getvalue())
        self.assertNotIn("test-secret-value", captured.getvalue() + repr(session) + repr(call.args))
        self.assertEqual(call.kwargs["env"]["OPENAI_API_KEY"], "test-secret-value")
        self.assertEqual(call.kwargs["env"]["DOCLING_DEVICE"], "cuda")
        self.assertIsNone(session._process.poll())

    def test_unexpected_response_stops_process(self):
        session, _ = self.start_session()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "inattendue"):
            session.request("bad_id")
        self.assertIsNotNone(session._process.poll())

    def test_reader_error_is_visible_and_process_can_be_reused(self):
        session, _ = self.start_session()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Message utile"):
            session.request("error")
        self.assertIsNone(session._process.poll())

    def test_dead_process_does_not_wait_forever(self):
        session, _ = self.start_session()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "sans résultat"):
            session.request("die")
        self.assertIsNotNone(session._process.poll())

    def test_interruption_terminates_child_and_reports_memory_loss(self):
        session, _ = self.start_session()
        with patch.object(session, "_read_response", side_effect=KeyboardInterrupt):
            with self.assertRaisesRegex(RuntimeError, "conversation précédente a été effacée"):
                session.request("ask", question="Question")
        self.assertIsNotNone(session._process.poll())


class InstallationTests(unittest.TestCase):
    @patch.object(support.sys, "platform", "linux")
    def test_installation_stays_in_venv_and_reuses_successful_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / "lib").mkdir()
            (project / "lib/colab_worker.py").write_text("", encoding="utf-8")
            (project / "pyproject.toml").write_text("[project]\nname='example'\n", encoding="utf-8")
            python = project / ".venv-colab/bin/python"

            def run(command, **kwargs):
                if "venv" in command:
                    python.parent.mkdir(parents=True)
                    python.touch()
                return subprocess.CompletedProcess(command, 0)

            with patch.object(support.subprocess, "run", side_effect=run) as runner, redirect_stdout(io.StringIO()):
                self.assertEqual(support.prepare_environment(project), python)
                first_calls = list(runner.call_args_list)
                runner.reset_mock()
                support.prepare_environment(project)
                self.assertEqual(runner.call_count, 1)
                self.assertEqual(runner.call_args.args[0][-1], "check")
            for call in first_calls:
                command = call.args[0]
                if "install" in command or "check" in command:
                    self.assertIn("--python", command)
                    self.assertIn(str(python), command)
                self.assertNotIn("--system-site-packages", command)
            marker = json.loads((project / ".venv-colab/rag-installation.json").read_text())
            self.assertEqual(len(marker["fingerprint"]), 64)


if __name__ == "__main__":
    unittest.main()
