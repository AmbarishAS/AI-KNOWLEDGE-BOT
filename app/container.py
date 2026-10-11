"""Composition root / dependency container.

Builds and wires the singletons (embedder, vector store, LLM, retriever, pipeline,
services) from :class:`Settings`. This is our lightweight dependency-injection approach:
construct graph once, hand typed objects to the API layer. Tests can build their own
container or reset it.
"""

from __future__ import annotations

from app.agents import Agent, build_default_registry
from app.config import Settings, get_settings
from app.embeddings import create_embedding_provider
from app.llm import create_llm
from app.rag import RAGPipeline
from app.retrieval import Retriever, create_reranker
from app.services.chat_service import ChatService
from app.services.ingestion import IngestionService
from app.vectorstore import create_vector_store


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = create_embedding_provider(self.settings)
        self.vector_store = create_vector_store(self.settings, self.embedder.dim)
        self.llm = create_llm(self.settings)
        self.reranker = create_reranker()
        self.retriever = Retriever(self.vector_store, self.embedder, self.settings)
        self.pipeline = RAGPipeline(self.retriever, self.llm, self.settings, self.reranker)
        self.ingestion = IngestionService(self.embedder, self.vector_store, self.settings)
        self.chat = ChatService(self.pipeline, self.settings)
        self.tool_registry = build_default_registry(self.retriever)
        self.agent = Agent(self.tool_registry, self.pipeline, self.settings)


_container: Container | None = None


def get_container() -> Container:
    global _container
    if _container is None:
        _container = Container()
    return _container


def reset_container() -> None:
    global _container
    _container = None
