"""Les entrées du RAG présenté dans l'article."""

from importlib.metadata import version

from docling_hybrid_rag.answer import plain_citation, source_link
from docling_hybrid_rag.ingestion import ingest_document, reindex
from docling_hybrid_rag.retrieval import document_passages, hybrid_retrieval, load_knowledge_base
from docling_hybrid_rag.settings import release_models
from docling_hybrid_rag.usage import total_usage
from docling_hybrid_rag.context import build_context
from docling_hybrid_rag.expansion import expand_query, load_thesaurus
from docling_hybrid_rag.workflow import create_workflow

__version__ = version("docling-hybrid-rag")
__all__ = ["ingest_document", "reindex", "load_knowledge_base", "document_passages", "hybrid_retrieval",
           "build_context", "expand_query", "load_thesaurus", "create_workflow", "plain_citation", "source_link",
           "release_models", "total_usage"]
