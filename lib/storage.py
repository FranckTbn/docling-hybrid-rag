"""Fichiers locaux portables et reprise des étapes coûteuses."""

import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import bm25s
from langchain_core.documents import Document

from lib.settings import EMBEDDING_MODEL_ID, DENSE_REVISION, CHILD_MAX_TOKENS


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pipeline_signature() -> str:
    # Une modification du parsing, du chunking ou du modèle exige un nouveau calcul explicite.
    code = {name: Path(__file__).with_name(name).read_text(encoding="utf-8")
            for name in ("parsing.py", "chunking.py")}
    config = dict(code=code, model=EMBEDDING_MODEL_ID, revision=DENSE_REVISION,
                  child_tokens=CHILD_MAX_TOKENS,
                  versions={name: version(name) for name in
                            ("docling", "docling-core", "sentence-transformers", "transformers")})
    return sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def load_chunks(directory: Path):
    bundle = read_json(directory / "chunks.json")
    parents = [Document(**item) for item in bundle["parents"]]
    children = [Document(**item) for item in bundle["children"]]
    return parents, children


def load_vectors(directory: Path, children) -> np.ndarray:
    with np.load(directory / "children-embeddings.npz", allow_pickle=False) as saved:
        if (saved["texts"].tolist() != [c.page_content for c in children]
            or saved["child_ids"].tolist() != [c.metadata["child_id"] for c in children]
            or saved["model"].item() != EMBEDDING_MODEL_ID
            or saved["revision"].item() != DENSE_REVISION):
            raise ValueError("Les embeddings ne correspondent plus aux enfants ; réindexer le document.")
        vectors = saved["vectors"]
    if (vectors.shape != (len(children), 1024) or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
        raise ValueError("Les vecteurs BGE-M3 doivent être complets, finis et normalisés.")
    return vectors


def save_lexical_index(root: Path, document_ids: list[str]) -> str:
    # BM25 calcule la rareté des termes sur toute la base, pas séparément par PDF.
    parents = [p for document_id in document_ids
               for p in load_chunks(root / "documents" / document_id)[0]]
    contract = {"texts": [p.page_content for p in parents],
                "parent_ids": [p.metadata["parent_id"] for p in parents],
                "stopwords": "fr", "bm25s_version": version("bm25s")}
    identity = sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    relative_path = f"indexes/bm25/{identity}"
    directory = root / relative_path
    if not (directory / "contract.json").exists():
        tokens = bm25s.tokenize(contract["texts"], stopwords="fr", show_progress=False)
        index = bm25s.BM25()
        index.index(tokens, show_progress=False)
        index.save(str(directory), show_progress=False)
        write_json(directory / "contract.json", contract)
    return relative_path
