"""Base de connaissance : un catalogue et un dossier par document.

Le catalogue relie chaque PDF ajouté à son dossier. Dans ce dossier, le
manifeste retient, pour chaque étape coûteuse, les entrées qui ont produit
ses fichiers et l'empreinte de ces fichiers. Une étape n'est réutilisée que
si ses entrées n'ont pas changé et si ses fichiers sont intacts.
"""

import inspect
import json
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SCHEMA = 2
CATALOG = "catalog.json"
MANIFEST = "manifest.json"
MAX_PDF_BYTES = 512 * 1024 * 1024


def write_json(path: Path, value) -> None:
    # Écriture atomique : un fichier à moitié écrit ne remplace jamais l'ancien.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_fingerprint(*objects) -> str:
    # Modifier le code qui produit une étape invalide ses résultats enregistrés.
    source = "\n".join(inspect.getsource(item) for item in objects)
    return sha256(source.encode("utf-8")).hexdigest()


def read_pdf(source: str | Path) -> bytes:
    if isinstance(source, str) and urlparse(source).scheme in {"https", "http"}:
        request = Request(source, headers={"User-Agent": "docling-hybrid-rag"})
        with urlopen(request, timeout=90) as response:
            content = response.read(MAX_PDF_BYTES + 1)
    else:
        path = Path(source).expanduser()
        if path.stat().st_size > MAX_PDF_BYTES:
            raise ValueError("PDF supérieur à 512 Mo.")
        content = path.read_bytes()
    if len(content) > MAX_PDF_BYTES or b"%PDF-" not in content[:1024]:
        raise ValueError("La source doit être un PDF de moins de 512 Mo, pas une page HTML.")
    return content


def read_catalog(knowledge_dir: str | Path) -> dict:
    path = Path(knowledge_dir) / CATALOG
    if not path.exists():
        return {"schema": SCHEMA, "documents": {}, "bm25": None}
    catalog = read_json(path)
    if catalog.get("schema") != SCHEMA:
        raise ValueError("Catalogue d'une ancienne version du package : reconstruire la base.")
    return catalog


def add_source(source: str | Path, knowledge_dir: str | Path, *, title: str | None = None) -> Path:
    """Copier le PDF dans la base et l'inscrire au catalogue. Même PDF, même dossier."""
    root = Path(knowledge_dir).resolve()
    content = read_pdf(source)
    source_sha256 = sha256(content).hexdigest()
    # Le dossier est nommé d'après le contenu du PDF, pas d'après son nom de fichier.
    document_id = source_sha256[:16]
    directory = root / "documents" / document_id
    if not (directory / "source.pdf").exists():
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "source.pdf").write_bytes(content)

    catalog = read_catalog(root)
    entry = catalog["documents"].setdefault(document_id, {"ready": False})
    is_url = isinstance(source, str) and urlparse(source).scheme in {"http", "https"}
    entry.update(
        title=title or entry.get("title") or Path(urlparse(str(source)).path).stem or "Document PDF",
        source_url=source if is_url else entry.get("source_url"),
        source_sha256=source_sha256,
    )
    write_json(root / CATALOG, catalog)
    return directory


def find_document(knowledge_dir: str | Path, source: str) -> Path:
    """Retrouver le dossier d'un document déjà ajouté, sans relire sa source."""
    root = Path(knowledge_dir).resolve()
    for document_id, entry in read_catalog(root)["documents"].items():
        if source in (entry.get("source_url"), document_id):
            return root / "documents" / document_id
    raise ValueError(f"{source} n'est pas dans la base : l'ajouter avec add_source.")


def read_manifest(directory: Path) -> dict:
    path = Path(directory) / MANIFEST
    return read_json(path) if path.exists() else {"schema": SCHEMA, "stages": {}}


def _as_json(value):
    # Les entrées sont comparées telles qu'elles seront relues depuis le manifeste.
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def record_stage(directory: Path, stage: str, inputs: dict, outputs: list[str]) -> None:
    """Enregistrer une étape, après l'écriture de ses fichiers."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    manifest["stages"][stage] = {
        "inputs": _as_json(inputs),
        "outputs": {name: file_sha256(directory / name) for name in outputs},
    }
    write_json(directory / MANIFEST, manifest)


def stage_problem(directory: Path, stage: str, inputs: dict | None = None) -> str | None:
    """Dire pourquoi une étape ne peut pas être réutilisée ; None si elle peut l'être."""
    directory = Path(directory)
    record = read_manifest(directory)["stages"].get(stage)
    if record is None:
        return f"l'étape {stage} n'a pas été exécutée"
    if inputs is not None and record["inputs"] != _as_json(inputs):
        changed = sorted(key for key in {*record["inputs"], *inputs}
                         if record["inputs"].get(key) != _as_json(inputs).get(key))
        return f"les entrées de l'étape {stage} ont changé ({', '.join(changed)})"
    for name, digest in record["outputs"].items():
        path = directory / name
        if not path.is_file() or file_sha256(path) != digest:
            return f"{name} a changé depuis l'étape {stage}"
    return None


def require_stage(directory: Path, stage: str, inputs: dict | None = None) -> dict:
    """Renvoyer l'enregistrement d'une étape réutilisable, ou expliquer quoi relancer."""
    problem = stage_problem(directory, stage, inputs)
    if problem:
        raise ValueError(f"Document {Path(directory).name} : {problem}. Relancer cette étape.")
    return read_manifest(directory)["stages"][stage]
