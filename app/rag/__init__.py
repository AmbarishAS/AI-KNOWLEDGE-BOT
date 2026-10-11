"""RAG: prompts, citations, and the grounded generation pipeline."""

from app.rag.citations import build_citations, extract_markers, is_grounded
from app.rag.pipeline import RAGConfig, RAGPipeline, deduplicate, fit_token_budget
from app.rag.prompts import PROMPT_VERSION, build_rag_messages, format_context

__all__ = [
    "RAGPipeline",
    "RAGConfig",
    "deduplicate",
    "fit_token_budget",
    "build_citations",
    "extract_markers",
    "is_grounded",
    "build_rag_messages",
    "format_context",
    "PROMPT_VERSION",
]
