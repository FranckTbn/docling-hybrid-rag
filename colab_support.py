"""Installation isolée et échanges locaux entre Colab et le processus RAG.

Ce module n'importe que la bibliothèque standard : le noyau Colab conserve
ses propres versions de NumPy, Torch et de ses bibliothèques d'affichage.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid


RESULT_PREFIX = "RAG_COLAB_RESULT "
TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"


def prepare_environment(project_dir: str | Path) -> Path:
    """Installer dans une venv, sans modifier les paquets du noyau Colab."""
    project = Path(project_dir).resolve()
    if sys.version_info < (3, 12):
        raise RuntimeError("Choisissez un environnement Colab récent, avec Python 3.12 ou supérieur.")
    if sys.platform != "linux":
        raise RuntimeError("Cette installation est destinée à Google Colab. Pour votre PC, suivre le README local.")
    requirements = project / "pyproject.toml"
    if not requirements.is_file() or not (project / "lib/colab_worker.py").is_file():
        raise RuntimeError("Le téléchargement du compagnon est incomplet. Relancez la cellule 1.")

    environment = project / ".venv-colab"
    python = environment / "bin/python"
    # pip pilote directement la venv ; aucune activation de terminal n'est nécessaire.
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(environment), "--without-pip"], check=True)
    pip = [sys.executable, "-m", "pip", "--python", str(python)]
    constraints = environment / "cuda-constraints.txt"
    constraints.write_text(
        f"torch=={TORCH_VERSION}\ntorchvision=={TORCHVISION_VERSION}\n", encoding="utf-8"
    )
    fingerprint = hashlib.sha256(
        requirements.read_bytes() + constraints.read_bytes() + sys.version.encode("utf-8")
    ).hexdigest()
    marker = environment / "rag-installation.json"
    previous = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
    if previous.get("fingerprint") != fingerprint:
        print("Installation des bibliothèques GPU, puis du compagnon. Cette étape peut prendre plusieurs minutes.")
        subprocess.run(
            [*pip, "install", "--quiet", "--index-url", "https://download.pytorch.org/whl/cu128",
             f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}"], check=True,
        )
        subprocess.run([*pip, "install", "--quiet", "-c", str(constraints), "-e", str(project)], check=True)
        subprocess.run([*pip, "check"], check=True)
        marker.write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")
    else:
        subprocess.run([*pip, "check"], check=True)
        print("Installation déjà disponible dans cette session.")
    return python


class ColabSession:
    """Un processus par lecteur garde les modèles et la mémoire entre les cellules."""

    def __init__(self, python: str | Path, project_dir: str | Path, *, api_key: str):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Ajoutez OPENAI_API_KEY dans Secrets, puis autorisez ce notebook à le lire.")
        environment = os.environ.copy()
        # La clé reste dans la mémoire des processus, pas dans .env ni les arguments.
        environment.update(
            OPENAI_API_KEY=api_key.strip(), DOCLING_DEVICE="cuda", PYTHONUNBUFFERED="1",
            PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
        )
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [str(python), "-u", "-m", "lib.colab_worker"], cwd=str(Path(project_dir).resolve()),
            env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", bufsize=1,
        )

    def __repr__(self) -> str:
        return f"ColabSession(active={self._process.poll() is None})"

    def request(self, operation: str, **payload):
        """Lire aussi les journaux pour ne pas bloquer un long parsing sur un tube plein."""
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("La session RAG est arrêtée. Relancez la cellule 2, puis la cellule 3.")
            request_id = uuid.uuid4().hex
            message = {"id": request_id, "operation": operation, "payload": payload}
            try:
                self._process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
                return self._read_response(request_id)
            except KeyboardInterrupt:
                self.close()
                raise RuntimeError(
                    "Calcul interrompu et processus arrêté. Relancez les cellules 2 et 3 pour reprendre. "
                    "La conversation précédente a été effacée."
                ) from None
            except (BrokenPipeError, OSError):
                self.close()
                raise RuntimeError("La session s'est arrêtée. Relancez la cellule 2, puis la cellule 3.") from None

    def _read_response(self, request_id: str):
        for line in self._process.stdout:
            if not line.startswith(RESULT_PREFIX):
                print(line, end="", flush=True)
                continue
            try:
                response = json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                self.close()
                raise RuntimeError("Réponse du processus illisible. Relancez la cellule 2.") from None
            if not isinstance(response, dict) or response.get("id") != request_id:
                self.close()
                raise RuntimeError("Réponse inattendue du processus. Relancez la cellule 2.")
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Le calcul n'a pas abouti."))
            return response["result"]
        self.close()
        raise RuntimeError("Le processus RAG s'est arrêté sans résultat. Relancez la cellule 2.")

    def close(self) -> None:
        """Interrompre aussi le calcul enfant quand le lecteur arrête une cellule."""
        process = self._process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()
