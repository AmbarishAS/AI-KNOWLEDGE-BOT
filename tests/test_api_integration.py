"""API integration tests using FastAPI TestClient (mock mode, in-process)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.container import reset_container
from app.main import create_app
from app.vectorstore import reset_vector_store

HEADERS = {"X-API-Key": "test-key", "X-Tenant": "acme"}


@pytest.fixture
async def client():
    await reset_vector_store()
    reset_container()
    app = create_app()
    with TestClient(app) as c:
        yield c
    await reset_vector_store()
    reset_container()


def _upload(client: TestClient, name: str, text: str):
    return client.post(
        "/api/v1/documents",
        headers=HEADERS,
        files={"file": (name, text.encode(), "text/markdown")},
    )


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready").json()
    assert ready["database"] is True


def test_auth_required(client):
    r = client.get("/api/v1/documents")  # no key
    assert r.status_code == 401


def test_upload_list_get_delete_document(client):
    r = _upload(client, "vacation.md", "# Vacation\n\nEmployees get 20 vacation days per year.")
    assert r.status_code == 201, r.text
    doc_id = r.json()["document_id"]

    lst = client.get("/api/v1/documents", headers=HEADERS).json()
    assert lst["total"] == 1

    got = client.get(f"/api/v1/documents/{doc_id}", headers=HEADERS)
    assert got.status_code == 200

    dele = client.delete(f"/api/v1/documents/{doc_id}", headers=HEADERS)
    assert dele.status_code == 204
    assert client.get("/api/v1/documents", headers=HEADERS).json()["total"] == 0


def test_upload_rejects_bad_extension(client):
    r = client.post(
        "/api/v1/documents",
        headers=HEADERS,
        files={"file": ("x.exe", b"data", "application/octet-stream")},
    )
    assert r.status_code == 422


def test_chat_returns_grounded_answer_with_citations(client):
    _upload(client, "vacation.md", "# Vacation\n\nEmployees accrue 20 vacation days per year.")
    r = client.post(
        "/api/v1/chat",
        headers=HEADERS,
        json={"question": "How many vacation days do employees accrue?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["abstained"] is False
    assert body["citations"]
    assert body["conversation_id"]


def test_chat_abstains_when_no_docs(client):
    r = client.post("/api/v1/chat", headers=HEADERS, json={"question": "What is the capital of France?"})
    assert r.json()["abstained"] is True


def test_conversation_persistence_and_tenant_isolation(client):
    _upload(client, "vacation.md", "# Vacation\n\nEmployees accrue 20 vacation days per year.")
    r = client.post("/api/v1/chat", headers=HEADERS, json={"question": "How many vacation days?"})
    cid = r.json()["conversation_id"]

    convo = client.get(f"/api/v1/conversations/{cid}", headers=HEADERS)
    assert convo.status_code == 200
    assert len(convo.json()["messages"]) == 2

    # Another tenant cannot read this conversation.
    other = client.get(f"/api/v1/conversations/{cid}", headers={"X-API-Key": "test-key", "X-Tenant": "evil"})
    assert other.status_code == 404


def test_agent_tools_listed_and_run(client):
    tools = client.get("/api/v1/agent/tools", headers=HEADERS).json()["tools"]
    assert any(t["name"] == "calculator" for t in tools)
    r = client.post("/api/v1/agent/run", headers=HEADERS, json={"question": "What is 6 * 7?"})
    assert "42" in r.json()["answer"]


def test_evaluation_endpoint_runs(client):
    r = client.post("/api/v1/evaluations/run", headers=HEADERS, json={})
    assert r.status_code == 200, r.text
    report = r.json()
    assert "retrieval" in report and "generation" in report
