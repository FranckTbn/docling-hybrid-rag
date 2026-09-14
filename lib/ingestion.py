"""Ajouter un PDF local ou distant sans refaire les documents déjà ingérés."""

from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

from lib.settings import get_encoder, get_tokenizer, EMBEDDING_MODEL_ID, DENSE_REVISION
from lib.storage import read_json, write_json, pipeline_signature, load_chunks, load_vectors, save_lexical_index


def read_pdf(source: str | Path) -> bytes:
    if isinstance(source, str) and urlparse(source).scheme in {"https", "http"}:
        request = Request(source, headers={"User-Agent": "docling-hybrid-rag/0.1"})
        with urlopen(request, timeout=90) as response:
            content = response.read(512 * 1024 * 1024 + 1)
    else:
        path = Path(source).expanduser()
        if path.stat().st_size > 512 * 1024 * 1024:
            raise ValueError("PDF supérieur à 512 Mo.")
        content = path.read_bytes()
    if len(content) > 512 * 1024 * 1024 or b"%PDF-" not in content[:1024]:
        raise ValueError("La source doit être un PDF de moins de 512 Mo, pas une page HTML.")
    return content


def _namespace_chunks(parents, children, document_id, title):
    # Un self_ref Docling est local à un document. Les identifiants d'index doivent être globaux.
    for item in [*parents, *children]:
        meta = item.metadata
        meta["document_id"] = document_id
        meta["document_path"] = f"documents/{document_id}/document.json"
        meta["document_title"] = title
        meta["parent_id"] = f'{document_id}:{meta["parent_id"]}'
        if "child_id" in meta:
            meta["child_id"] = f'{document_id}:{meta["child_id"]}'


def ingest_document(source: str | Path, data_dir: str | Path = "data", *, title: str | None = None):
    """Parser, découper et indexer un PDF. Même contenu + même pipeline = reprise sans modèle."""
    from lib.parsing import build_document_converter, parse_document, save_parsed_document, load_document
    from lib.chunking import build_parent_chunks, build_child_chunks

    root = Path(data_dir).resolve()
    content = read_pdf(source)
    document_id = sha256(content).hexdigest()
    directory = root / "documents" / document_id
    signature = pipeline_signature()
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / "record.json"
    if record_path.exists():
        record = read_json(record_path)
        if record["pipeline_signature"] != signature:
            raise ValueError("Pipeline modifié. Utiliser un nouveau dossier data pour reconstruire la base.")
    else:
        pdf_path = directory / "source.pdf"
        pdf_path.write_bytes(content)
        record = dict(document_id=document_id, pipeline_signature=signature,
                      title=title or Path(urlparse(str(source)).path).stem or "Document PDF",
                      source_url=str(source) if urlparse(str(source)).scheme in {"http", "https"} else None)
        write_json(record_path, record)

    canonical_path = directory / "document.json"
    if not canonical_path.exists():
        doc = parse_document(directory / "source.pdf", build_document_converter())
        save_parsed_document(doc, directory)
    digest = sha256(canonical_path.read_bytes()).hexdigest()
    if record.get("canonical_sha256", digest) != digest:
        raise ValueError("Le JSON Docling a changé ; reconstruire les chunks et les index dans une nouvelle base.")
    doc = load_document(canonical_path)
    record["canonical_sha256"] = digest
    write_json(record_path, record)
    if not (directory / "chunks.json").exists():
        tokenizer = get_tokenizer()
        parents, parts = build_parent_chunks(doc, canonical_path, tokenizer)
        children = build_child_chunks(parents, parts, canonical_path, tokenizer)
        if not parents or not children:
            raise ValueError("Aucun chunk textuel exploitable dans ce PDF.")
        _namespace_chunks(parents, children, document_id, record["title"])
        write_json(directory / "chunks.json", {
            "parents": [p.model_dump() for p in parents], "children": [c.model_dump() for c in children],
        })
    parents, children = load_chunks(directory)
    if not (directory / "children-embeddings.npz").exists():
        texts = [c.page_content for c in children]
        vectors = get_encoder().encode(texts, batch_size=2, normalize_embeddings=True, show_progress_bar=True)
        np.savez_compressed(directory / "children-embeddings.npz", vectors=vectors, texts=texts,
                            child_ids=[c.metadata["child_id"] for c in children],
                            model=EMBEDDING_MODEL_ID, revision=DENSE_REVISION)
    load_vectors(directory, children)
    # Le manifeste ne référence le document qu'après la réussite de toutes les étapes.
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"schema": 1, "documents": []}
    if document_id not in manifest["documents"]:
        manifest["documents"].append(document_id)
    manifest["bm25"] = save_lexical_index(root, manifest["documents"])
    write_json(manifest_path, manifest)
    return {**record, "parents": len(parents), "children": len(children), "directory": str(directory)}
