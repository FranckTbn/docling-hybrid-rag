"""Processus RAG persistant, piloté par des requêtes JSON depuis Colab."""

import json
import math
import os
import platform
import sys
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import numpy as np

from lib.ingestion import ingest_document
from lib.settings import ANSWER_MODEL
from lib.workflow import create_workflow

RESULT_PREFIX = "RAG_COLAB_RESULT "
_FIELDS = {
    "status": (set(), set()),
    "hello": ({"data_dir"}, {"model", "thread_id"}),
    "ingest": ({"source", "title", "data_dir"}, set()),
    "ask": ({"question", "thread_id"}, set()),
    "close": (set(), set()),
}
_REQUEST_ERRORS = {
    "request": "Requête invalide : fournir id, operation et payload en JSON.",
    "operation": "Opération inconnue. Utiliser status, hello, ingest, ask ou close.",
    "payload": "Paramètres invalides pour cette opération. Vérifiez les cellules du notebook.",
    "source": "La source doit être une URL publique HTTP(S), sans identifiants dans l'adresse.",
    "configuration": "Exécutez la cellule de connexion ou d'ingestion avant de poser une question.",
    "key": "Ajoutez OPENAI_API_KEY aux secrets Colab et relancez la cellule de connexion.",
    "closed": "Le processus est fermé. Relancez la cellule de connexion.",
}
_API_ERRORS = {
    401: "Clé API refusée. Vérifiez le secret OPENAI_API_KEY puis reconnectez le notebook.",
    403: "Accès API refusé. Vérifiez les droits de la clé et l'accès au modèle choisi.",
    429: "Quota ou limite API atteint. Vérifiez la facturation et les limites du compte avant de réessayer.",
    404: "Modèle ou ressource API introuvable. Vérifiez le nom du modèle et son accès.",
    400: "Requête API refusée. Vérifiez que le modèle accepte les options du compagnon.",
}
_API_STATUS = {"AuthenticationError": 401, "PermissionDeniedError": 403,
               "RateLimitError": 429, "NotFoundError": 404, "BadRequestError": 400}
_GENERIC_ERROR = "Opération interrompue. Vérifiez la connexion, les fichiers et la compatibilité de l'environnement."


class RequestError(ValueError):
    """Un code interne permet d'afficher un message sûr sans reprendre le payload."""


def safe_error(error):
    """Ne jamais sérialiser l'exception API : elle peut contenir une clé ou une URL privée."""
    kind = type(error).__name__
    if isinstance(error, RequestError):
        message = _REQUEST_ERRORS.get(error.args[0], _REQUEST_ERRORS["request"])
    else:
        status = _API_STATUS.get(kind, getattr(error, "status_code", None))
        message = _API_ERRORS.get(status, _GENERIC_ERROR) if isinstance(status, int) else _GENERIC_ERROR
    return {"ok": False, "error_type": kind, "error": message}


def json_value(value):
    """Conserver les données utiles, sans repr() implicite d'objets ou de clients API."""
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nombre non fini.")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return _json_collection(value)


def _json_collection(value):
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: json_value(item) for key, item in value.items()}
    raise TypeError("Objet non sérialisable.")


def runtime_status():
    # Interroger CUDA ne construit ni modèle Docling ni encodeur BGE-M3.
    import torch

    available = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "cuda_available": available, "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if available else None,
    }


def ingestion_summary(record, data_dir, elapsed):
    """Lire les artefacts produits, sans reparsing ni nouvel encodage."""
    directory = Path(record["directory"])
    document = json.loads((directory / "document.json").read_text(encoding="utf-8"))
    with np.load(directory / "children-embeddings.npz", allow_pickle=False) as saved:
        vector_shape = list(saved["vectors"].shape)
    return {
        "record": record, **_document_statistics(document), "vector_shape": vector_shape,
        "source_url": record.get("source_url"), "elapsed_seconds": elapsed,
        "manifest_ok": _manifest_ready(data_dir, record["document_id"]),
    }


def _document_statistics(document):
    items = [item for name in ("texts", "tables", "pictures") for item in document.get(name, [])]
    covered = sorted({prov["page_no"] for item in items for prov in item.get("prov", [])})
    return {"pages": len(document["pages"]), "covered_pages": covered,
            "tables": len(document.get("tables", [])), "pictures": len(document.get("pictures", [])),
            "formulas": sum(item.get("label") == "formula" for item in document.get("texts", []))}


def _manifest_ready(data_dir, document_id):
    manifest_path = Path(data_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    index = Path(data_dir) / manifest.get("bm25", "") / "contract.json"
    return manifest.get("schema") == 1 and document_id in manifest.get("documents", []) and index.is_file()


def _valid_id(identity):
    return isinstance(identity, (int, str)) and not isinstance(identity, bool)


def _validate_payload(operation, payload):
    required, optional = _FIELDS[operation]
    if not isinstance(payload, dict) or not required <= payload.keys() or payload.keys() - required - optional:
        raise RequestError("payload")
    if any(not isinstance(value, str) or not value.strip() or "\x00" in value for value in payload.values()):
        raise RequestError("payload")


def _validate_source(source):
    url = urlsplit(source)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise RequestError("source")


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {"id", "operation", "payload"}:
        raise RequestError("request")
    if not _valid_id(request["id"]):
        raise RequestError("request")
    operation, payload = request["operation"], request["payload"]
    if not isinstance(operation, str) or operation not in _FIELDS:
        raise RequestError("operation")
    _validate_payload(operation, payload)
    if operation == "ingest":
        _validate_source(payload["source"])
    return operation, payload


class Worker:
    """Un graphe par configuration, maintenu jusqu'à une ingestion réussie ou une fermeture."""

    def __init__(self):
        self.graph = None
        self.data_dir = None
        self.model = os.getenv("OPENAI_MODEL") or ANSWER_MODEL
        self.closed = False

    def _configure(self, data_dir, model):
        directory = Path(data_dir).expanduser().resolve()
        if (directory, model) != (self.data_dir, self.model):
            self.graph = None
        self.data_dir, self.model = directory, model

    def _graph(self):
        if self.data_dir is None:
            raise RequestError("configuration")
        if os.getenv("OPENAI_API_KEY", "").strip() in {"", "your-openai-api-key"}:
            raise RequestError("key")
        if self.graph is None:
            # os.devnull empêche un ancien .env local de remplacer le secret Colab.
            self.graph = create_workflow(data_dir=self.data_dir, env_path=os.devnull, model=self.model)
        return self.graph

    def _hello(self, payload):
        print("Connexion au modèle et test de salutation en cours...", flush=True)
        self._configure(payload["data_dir"], payload.get("model", self.model))
        return self._invoke_question({"question": "Bonjour !", "thread_id": payload.get("thread_id", "colab")})

    def _ingest(self, payload):
        directory = Path(payload["data_dir"]).expanduser().resolve()
        started = perf_counter()
        print("Analyse et indexation du PDF en cours...", flush=True)
        record = ingest_document(payload["source"], directory, title=payload["title"])
        # Le graphe peut déjà avoir chargé les anciens index : il faut le recréer.
        self.graph, self.data_dir = None, directory
        return ingestion_summary(record, directory, perf_counter() - started)

    def _ask(self, payload):
        print("Traitement de la question et préparation de la réponse en cours...", flush=True)
        return self._invoke_question(payload)

    def _invoke_question(self, payload):
        result = self._graph().invoke(
            {"question": payload["question"]}, {"configurable": {"thread_id": payload["thread_id"]}},
        )
        fields = ("answer", "route", "context_k", "route_reason", "search_question",
                  "sources", "context", "retrieval")
        response = {field: result[field] for field in fields}
        # La mémoire visible ne contient ni métadonnées API, ni prompts système.
        response["messages"] = [{"type": message.type, "text": message.text}
                                for message in result.get("messages", [])]
        return response

    def _close(self, payload):
        self.graph, self.closed = None, True
        return {"closed": True}

    def handle(self, request):
        identity = request.get("id") if isinstance(request, dict) else None
        if not _valid_id(identity):
            identity = None
        try:
            if self.closed:
                raise RequestError("closed")
            operation, payload = validate_request(request)
            handlers = {"status": lambda _: runtime_status(), "hello": self._hello,
                        "ingest": self._ingest, "ask": self._ask, "close": self._close}
            return {"id": identity, "ok": True, "result": json_value(handlers[operation](payload))}
        except (Exception, KeyboardInterrupt) as error:
            return {"id": identity, **safe_error(error)}


def serve(input_stream, output_stream, worker=None):
    worker = worker or Worker()
    for line in input_stream:
        try:
            request = json.loads(line)
            response = worker.handle(request)
        except (ValueError, TypeError):
            response = {"id": None, **safe_error(RequestError("request"))}
        # Une barre de progression peut laisser un log sans saut de ligne : séparer le protocole.
        output_stream.write("\n" + RESULT_PREFIX + json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
        output_stream.flush()
        if worker.closed:
            break


def main():
    # Le protocole est UTF-8, y compris lorsque les tests tournent sous Windows.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
