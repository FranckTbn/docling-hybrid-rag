"""Les entrées du RAG présenté dans l'article."""

from importlib.metadata import version

from docling_hybrid_rag.ingestion import ingest_document
from docling_hybrid_rag.retrieval import hybrid_retrieval, load_knowledge_base
from docling_hybrid_rag.context import build_context
from docling_hybrid_rag.expansion import expand_query, load_thesaurus
from docling_hybrid_rag.workflow import create_workflow

__version__ = version("docling-hybrid-rag")
__all__ = ["ingest_document", "load_knowledge_base", "hybrid_retrieval", "build_context",
           "expand_query", "load_thesaurus", "create_workflow"]
