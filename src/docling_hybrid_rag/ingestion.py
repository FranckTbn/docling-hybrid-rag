"""Ajouter un PDF à la base. Une étape déjà faite avec les mêmes entrées est réutilisée."""

from importlib.metadata import version
from pathlib import Path

import numpy as np

from docling_hybrid_rag.indexing import DEFAULT_LANGUAGE, lexical_settings, save_lexical_index
from docling_hybrid_rag.settings import get_encoder, get_tokenizer, EMBEDDING_MODEL_ID, DENSE_REVISION, CHILD_MAX_TOKENS
from docling_hybrid_rag.storage import children_sha256, load_chunks, load_vectors
from docling_hybrid_rag.store import (
    CATALOG, add_source, code_fingerprint, file_sha256, read_catalog, record_stage, stage_problem, write_json,
)

CHUNKING = "chunking"
EMBEDDINGS = "embeddings"


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


def _title(directory: Path, title: str | None) -> str:
    return title or read_catalog(directory.parents[1])["documents"][directory.name]["title"]


def ensure_chunks(directory: str | Path, doc, *, title: str | None = None, parent_scope: str = "section"):
    """Renvoyer les parents et les enfants d'un document, en les construisant seulement si nécessaire.

    Les chunks enregistrés sont rechargés tant que le document, le tokenizer, la portée des
    parents, les versions de Docling et le code du chunking sont ceux qui les ont produits.
    """
    from docling_hybrid_rag import chunking
    from docling_hybrid_rag.parsing import clean_parsed_document

    directory = Path(directory).resolve()
    title = _title(directory, title)
    # Chaque étape dépend de l'empreinte du fichier produit par la précédente :
    # un nouveau parsing invalide donc les chunks, puis les embeddings.
    inputs = {
        "document_sha256": file_sha256(directory / "document.json"), "title": title,
        "tokenizer": {"model": EMBEDDING_MODEL_ID, "revision": DENSE_REVISION, "max_tokens": CHILD_MAX_TOKENS},
        "parent_scope": parent_scope,
        "versions": {name: version(name) for name in ("docling-core", "transformers")},
        # Le nettoyage du document chargé fait partie de ce qui produit les chunks.
        "code": code_fingerprint(chunking, _namespace_chunks, clean_parsed_document),
    }
    if stage_problem(directory, CHUNKING, inputs):
        parents, children = chunking.build_parent_child_chunks(doc, get_tokenizer(), parent_scope=parent_scope)
        if not parents or not children:
            raise ValueError("Aucun chunk textuel exploitable dans ce PDF.")
        _namespace_chunks(parents, children, directory.name, title)
        write_json(directory / "chunks.json", {
            "parents": [p.model_dump() for p in parents], "children": [c.model_dump() for c in children],
        })
        record_stage(directory, CHUNKING, inputs, ["chunks.json"])
    return load_chunks(directory)


def ensure_embeddings(directory: str | Path, *, encoder=None) -> np.ndarray:
    """Renvoyer les vecteurs des enfants, en les encodant seulement si leurs textes ont changé.

    Seuls les enfants sont encodés : changer la portée des parents ne force pas un nouvel encodage.
    """
    directory = Path(directory).resolve()
    _, children = load_chunks(directory)
    inputs = {
        "children_sha256": children_sha256(children),
        "model": EMBEDDING_MODEL_ID, "revision": DENSE_REVISION,
        "versions": {name: version(name) for name in ("sentence-transformers", "transformers")},
    }
    if stage_problem(directory, EMBEDDINGS, inputs):
        texts = [c.page_content for c in children]
        vectors = (encoder or get_encoder()).encode(texts, batch_size=2, normalize_embeddings=True,
                                                    show_progress_bar=True)
        np.savez_compressed(directory / "children-embeddings.npz", vectors=vectors, texts=texts,
                            child_ids=[c.metadata["child_id"] for c in children],
                            model=EMBEDDING_MODEL_ID, revision=DENSE_REVISION)
        record_stage(directory, EMBEDDINGS, inputs, ["children-embeddings.npz"])
    return load_vectors(directory, children)


def ingest_document(source: str | Path, knowledge_dir: str | Path = "data/knowledge", *,
                    title: str | None = None, pdf_options=None, parent_scope: str = "section",
                    language: str | None = None):
    """Parser, découper et indexer un PDF, puis le rendre interrogeable.

    `language` (`fr`, `en`, `de`, `es`, `it`, `pt`) règle les mots courants et la racinisation de BM25.
    Elle est fixée à la création de la base (français par défaut) et se retrouve ensuite d'elle-même.
    """
    from docling_hybrid_rag.parsing import (
        PARSING, build_document_converter, load_parsed_document, parse_document, parsing_inputs,
        save_parsed_document,
    )

    root = Path(knowledge_dir).resolve()
    known = read_catalog(root).get("language")
    if language and known and language != known:
        raise ValueError(f"Cette base est en {known!r} : BM25 ne peut pas mélanger {language!r}. Créer une autre base.")
    settings = lexical_settings(language or known or DEFAULT_LANGUAGE)
    directory = add_source(source, root, title=title)
    document_id = directory.name

    if stage_problem(directory, PARSING, parsing_inputs(directory, pdf_options)):
        doc = parse_document(directory / "source.pdf", build_document_converter(pdf_options))
        save_parsed_document(doc, directory, pdf_options)
    doc = load_parsed_document(directory, pdf_options)

    parents, children = ensure_chunks(directory, doc, parent_scope=parent_scope)
    ensure_embeddings(directory)

    # Le document n'est interrogeable qu'après la réussite de toutes les étapes.
    catalog = read_catalog(root)
    catalog["documents"][document_id]["ready"] = True
    ready = [key for key, entry in catalog["documents"].items() if entry["ready"]]
    catalog["language"] = language or known or DEFAULT_LANGUAGE
    catalog["bm25"] = save_lexical_index(root, ready, settings)
    write_json(root / CATALOG, catalog)
    return {"document_id": document_id, **catalog["documents"][document_id],
            "parents": len(parents), "children": len(children), "directory": str(directory)}
