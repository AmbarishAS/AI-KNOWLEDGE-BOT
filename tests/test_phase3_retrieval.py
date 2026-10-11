"""Phase 3 — embeddings, vector store, dense/sparse/hybrid retrieval, reranking."""

from __future__ import annotations

import pytest

from app.config import RetrievalMode, get_settings
from app.domain import Principal
from app.embeddings import cosine_similarity, create_embedding_provider
from app.retrieval import Retriever, apply_threshold, create_reranker
from app.services.ingestion import IngestionService
from app.vectorstore import create_vector_store, reset_vector_store

DOC_A = b"""# Vacation Policy

Full-time employees accrue 20 days of paid vacation per year. Unused vacation
rolls over up to a maximum of 5 days into the next calendar year.
"""

DOC_B = b"""# Expense Reimbursement

Submit expense reports within 30 days. Meals are reimbursed up to 50 dollars per day
during business travel. Receipts are required for all claims over 25 dollars.
"""


@pytest.fixture
async def ingested():
    get_settings.cache_clear()
    settings = get_settings()
    await reset_vector_store()
    embedder = create_embedding_provider(settings)
    store = create_vector_store(settings, embedder.dim)
    svc = IngestionService(embedder, store, settings)
    principal = Principal(tenant="acme")
    await svc.ingest("vacation.md", DOC_A, "text/markdown", principal)
    await svc.ingest("expense.md", DOC_B, "text/markdown", principal)
    retriever = Retriever(store, embedder, settings)
    yield retriever, store, settings
    await reset_vector_store()


async def test_mock_embeddings_are_deterministic_and_similar_for_shared_words():
    emb = create_embedding_provider(get_settings())
    v1 = await emb.embed_query("vacation policy accrual")
    v2 = await emb.embed_query("vacation policy accrual")
    assert v1 == v2  # deterministic
    related = cosine_similarity(v1, await emb.embed_query("vacation accrual days"))
    unrelated = cosine_similarity(v1, await emb.embed_query("database sharding kafka"))
    assert related > unrelated


async def test_dense_retrieval_finds_relevant_doc(ingested):
    retriever, _store, _s = ingested
    results = await retriever.retrieve("how many vacation days do I get", "acme", top_k=1, mode=RetrievalMode.DENSE)
    assert results
    assert "vacation" in results[0].chunk.text.lower()


async def test_sparse_retrieval_matches_exact_terms(ingested):
    retriever, _store, _s = ingested
    results = await retriever.retrieve("reimbursed receipts expense", "acme", top_k=1, mode=RetrievalMode.SPARSE)
    assert results
    assert results[0].chunk.filename == "expense.md"


async def test_hybrid_retrieval_fuses_rankings(ingested):
    retriever, _store, _s = ingested
    results = await retriever.retrieve("vacation rollover days", "acme", top_k=2, mode=RetrievalMode.HYBRID)
    assert results
    assert results[0].source == "hybrid"


async def test_tenant_isolation_in_retrieval(ingested):
    retriever, _store, _s = ingested
    results = await retriever.retrieve("vacation", "other-tenant", top_k=5)
    assert results == []


async def test_metadata_filter(ingested):
    retriever, _store, _s = ingested
    results = await retriever.retrieve("policy", "acme", top_k=5, filters={"filename": "expense.md"})
    assert results
    assert all(r.chunk.filename == "expense.md" for r in results)


async def test_reranker_orders_by_relevance(ingested):
    retriever, _store, _s = ingested
    candidates = await retriever.retrieve("expense receipts meals", "acme", top_k=5)
    reranked = create_reranker().rerank("expense receipts meals", candidates, top_k=2)
    assert reranked
    assert reranked[0].chunk.filename == "expense.md"
    assert reranked[0].source == "rerank"


async def test_threshold_enables_abstention(ingested):
    retriever, _store, _s = ingested
    results = await retriever.retrieve("quantum teleportation protocol", "acme", top_k=5)
    filtered = apply_threshold(results, threshold=0.99)
    assert filtered == []


async def test_delete_document_removes_from_store(ingested):
    retriever, store, settings = ingested
    svc = IngestionService(create_embedding_provider(settings), store, settings)
    docs_before = await store.count()
    # Delete the vacation doc by finding its id via a chunk.
    chunk = (await retriever.retrieve("vacation", "acme", top_k=1))[0].chunk
    await svc.delete(Principal(tenant="acme"), chunk.document_id)
    assert await store.count() < docs_before
    assert await retriever.retrieve("vacation rollover", "acme", top_k=5) == [] or all(
        r.chunk.document_id != chunk.document_id for r in await retriever.retrieve("vacation rollover", "acme", top_k=5)
    )
