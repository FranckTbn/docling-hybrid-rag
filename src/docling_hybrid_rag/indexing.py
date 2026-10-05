"""Index lexical BM25 sur les enfants contextualisés de toute la base.

Le texte d'un enfant contient déjà les titres de sa section : un nom de méthode
présent dans le titre est donc retrouvé dans chacun de ses enfants. La
racinisation rapproche les variantes d'un mot (« provisions », « provision ») et les
mots courants de la langue de la base sont ignorés (français par défaut).
"""

import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

from docling_hybrid_rag.storage import load_chunks
from docling_hybrid_rag.store import write_json

DEFAULT_LANGUAGE = "fr"
# Mots courants et racinisation par langue de la base. BM25 ne traduit rien : une base se tient dans la
# langue de ses documents, sinon la racinisation d'une langue abîme les mots de l'autre.
LANGUAGES = {
    "fr": {"stopwords": "fr", "stemmer": "french"},
    "en": {"stopwords": "en", "stemmer": "english"},
    "de": {"stopwords": "de", "stemmer": "german"},
    "es": {"stopwords": "es", "stemmer": "spanish"},
    "it": {"stopwords": "it", "stemmer": "italian"},
    "pt": {"stopwords": "pt", "stemmer": "portuguese"},
}
LEXICAL_SETTINGS = LANGUAGES[DEFAULT_LANGUAGE]


def lexical_settings(language: str = DEFAULT_LANGUAGE) -> dict:
    """Règles de tokenisation BM25 d'une langue (`fr`, `en`, `de`, `es`, `it`, `pt`)."""
    if language not in LANGUAGES:
        raise ValueError(f"Langue inconnue : {language!r}. Choisir parmi {', '.join(LANGUAGES)}.")
    return dict(LANGUAGES[language])


def lexical_tokens(texts, settings=LEXICAL_SETTINGS, *, return_ids=True):
    """Tokeniser corpus et questions avec exactement les mêmes règles."""
    stemmer = Stemmer.Stemmer(settings["stemmer"]) if settings.get("stemmer") else None
    return bm25s.tokenize(texts, stopwords=settings["stopwords"], stemmer=stemmer,
                          return_ids=return_ids, show_progress=False)


def save_lexical_index(root: Path, document_ids: list[str], settings=LEXICAL_SETTINGS) -> str:
    """Indexer les enfants de tous les documents prêts ; renvoyer le chemin relatif de l'index."""
    # BM25 calcule la rareté des termes sur toute la base, pas séparément par PDF.
    children = [child for document_id in document_ids
                for child in load_chunks(root / "documents" / document_id)[1]]
    contract = {"texts": [child.page_content for child in children],
                "child_ids": [child.metadata["child_id"] for child in children],
                "settings": settings,
                "versions": {name: version(name) for name in ("bm25s", "PyStemmer")}}
    identity = sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    relative_path = f"indexes/bm25/{identity}"
    directory = root / relative_path
    if not (directory / "contract.json").exists():
        index = bm25s.BM25()
        index.index(lexical_tokens(contract["texts"], settings), show_progress=False)
        index.save(str(directory), show_progress=False)
        write_json(directory / "contract.json", contract)
    return relative_path


def weighted_scores(index, terms: list[dict], settings=LEXICAL_SETTINGS) -> np.ndarray:
    """Score BM25 pondéré de l'article : somme des contributions, chacune multipliée par son poids.

    Les tokens de la requête originale gardent le poids 1, répétitions comprises.
    Un token ajouté par l'expansion n'est compté qu'une fois, avec son poids maximal,
    et jamais s'il figure déjà dans la requête originale.
    """
    original = [token for item in terms if item["source"] == "original"
                for token in lexical_tokens(item["term"], settings, return_ids=False)[0]]
    added: dict[str, float] = {}
    for item in terms:
        if item["source"] == "original":
            continue
        for token in lexical_tokens(item["term"], settings, return_ids=False)[0]:
            if token not in original:
                added[token] = max(added.get(token, 0.0), item["weight"])

    scores = np.zeros(index.scores["num_docs"], dtype=float)
    for token, weight in [*((token, 1.0) for token in original), *added.items()]:
        if token in index.vocab_dict:
            scores += weight * index.get_scores([token])
    return scores
