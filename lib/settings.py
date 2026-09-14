"""Les mêmes budgets et modèles que dans l'article."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values

EMBEDDING_MODEL_ID = "BAAI/bge-m3"
DENSE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
CHILD_MAX_TOKENS = 400
CANDIDATE_K = 20
CONTEXT_K = 3
ANSWER_MODEL = "gpt-5.6-luna"


@lru_cache(maxsize=1)
def get_encoder():
    # Téléchargement au premier usage ; le même modèle sert aux enfants et aux questions.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_ID, revision=DENSE_REVISION, device="cpu")


@lru_cache(maxsize=1)
def get_tokenizer():
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer
    return HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL_ID, revision=DENSE_REVISION),
        max_tokens=CHILD_MAX_TOKENS,
    )


def get_llm(env_path: str | Path = ".env", model: str | None = None):
    from langchain_openai import ChatOpenAI

    # La clé locale remplace une ancienne clé héritée du terminal, sans jamais être affichée.
    local = dotenv_values(env_path, interpolate=False)
    key = local.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key or key == "your-openai-api-key":
        raise ValueError("Renseigner OPENAI_API_KEY dans le fichier .env à la racine du dépôt.")
    model = model or local.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or ANSWER_MODEL
    return ChatOpenAI(
        model=model, api_key=key, base_url="https://api.openai.com/v1",
        use_responses_api=True, reasoning={"effort": "medium"}, max_tokens=6000,
        timeout=180, max_retries=0, store=False,
    )
