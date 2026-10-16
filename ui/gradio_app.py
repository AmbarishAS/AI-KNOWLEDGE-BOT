"""Gradio demo UI for the Enterprise AI Knowledge Copilot.

Talks to the FastAPI backend over HTTP. Supports: document upload, document listing, chat
with visible citations, conversation history, and a simple evaluation results view.

Run:
    python ui/gradio_app.py
Environment:
    COPILOT_API_URL (default http://localhost:8000)
    COPILOT_API_KEY (default changeme-dev-key)
    GRADIO_SERVER_PORT (default 7860)
"""

from __future__ import annotations

import os

import gradio as gr
import httpx

API_URL = os.getenv("COPILOT_API_URL", "http://localhost:8000")
API_KEY = os.getenv("COPILOT_API_KEY", "changeme-dev-key")
SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))


def _headers(api_key: str, tenant: str) -> dict:
    return {"X-API-Key": api_key or API_KEY, "X-Tenant": tenant or "public"}


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=60.0)


def check_health() -> str:
    try:
        with _client() as c:
            health = c.get("/health").json()
        return f"✅ API up · mode={health.get('mode')} · v{health.get('version')}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ API unreachable at {API_URL}: {exc}"


def list_documents(api_key: str, tenant: str) -> str:
    try:
        with _client() as c:
            docs = c.get("/api/v1/documents", headers=_headers(api_key, tenant)).json()
        lines = [f"**{docs['total']} document(s)** in tenant `{tenant or 'public'}`", ""]
        for d in docs["items"]:
            lines.append(f"- {d['filename']} — {d['num_chunks']} chunks ({d['chunk_strategy']})")
        if not docs["items"]:
            lines.append("_No documents yet._")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return "_No documents yet._"


def ingest_document(file_path: str | None, api_key: str, tenant: str) -> tuple[str, str]:
    if not file_path:
        return "⚠️ Please select a file first.", list_documents(api_key, tenant)
    try:
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as fh, _client() as c:
            resp = c.post(
                "/api/v1/documents",
                headers=_headers(api_key, tenant),
                files={"file": (filename, fh.read())},
            )
        if resp.status_code == 201:
            status = f"✅ Ingested {filename} ({resp.json()['num_chunks']} chunks)"
        else:
            status = f"❌ {resp.status_code}: {resp.text}"
    except Exception as exc:  # noqa: BLE001
        status = f"❌ Upload failed: {exc}"
    return status, list_documents(api_key, tenant)


def chat(
    message: str,
    history: list[dict],
    conversation_id: str | None,
    api_key: str,
    tenant: str,
) -> tuple[list[dict], str | None]:
    history = history or []
    if not message.strip():
        return history, conversation_id

    history.append({"role": "user", "content": message})
    try:
        with _client() as c:
            resp = c.post(
                "/api/v1/chat",
                headers=_headers(api_key, tenant),
                json={"question": message, "conversation_id": conversation_id},
            )
    except Exception as exc:  # noqa: BLE001
        history.append({"role": "assistant", "content": f"❌ Request failed: {exc}"})
        return history, conversation_id

    if resp.status_code != 200:
        history.append({"role": "assistant", "content": f"❌ {resp.status_code}: {resp.text}"})
        return history, conversation_id

    data = resp.json()
    conversation_id = data["conversation_id"]

    parts = [data["answer"]]
    if data["abstained"]:
        parts.append("\n> ℹ️ The assistant abstained (insufficient grounded evidence).")
    if data["citations"]:
        parts.append("\n**Citations**")
        for cit in data["citations"]:
            parts.append(f"- {cit['marker']} **{cit['filename']}** — {cit['snippet'][:160]}…")
    parts.append(
        "\n`prompt_tokens={pt} · completion_tokens={ct} · cost=${cost:.6f} · latency={lat:.0f}ms`".format(
            pt=data["prompt_tokens"],
            ct=data["completion_tokens"],
            cost=data["estimated_cost_usd"],
            lat=data["latency_ms"],
        )
    )
    history.append({"role": "assistant", "content": "\n".join(parts)})
    return history, conversation_id


def run_evaluation(api_key: str, tenant: str) -> str:
    try:
        with _client() as c:
            resp = c.post("/api/v1/evaluations/run", headers=_headers(api_key, tenant), json={})
    except Exception as exc:  # noqa: BLE001
        return f"❌ Request failed: {exc}"
    if resp.status_code != 200:
        return f"❌ {resp.status_code}: {resp.text}"
    report = resp.json()
    import json

    lines = [
        "### Retrieval (baseline dense vs improved hybrid+rerank)",
        "```json",
        json.dumps(
            {
                "baseline_dense": report["retrieval"]["baseline_dense"],
                "improved_hybrid_rerank": report["retrieval"]["improved_hybrid_rerank"],
            },
            indent=2,
        ),
        "```",
        "### Generation",
        "```json",
        json.dumps(report["generation"], indent=2),
        "```",
    ]
    return "\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Knowledge Copilot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 AI Knowledge Copilot")
        conversation_id = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                health_md = gr.Markdown(check_health())
                refresh_health = gr.Button("Check API health", size="sm")
                api_key_box = gr.Textbox(label="API key", value=API_KEY, type="password")
                tenant_box = gr.Textbox(label="Tenant", value="public")

                gr.Markdown("### 📄 Documents")
                file_box = gr.File(
                    label="Upload (pdf/docx/md/txt)",
                    file_types=[".pdf", ".docx", ".md", ".txt"],
                    type="filepath",
                )
                ingest_btn = gr.Button("Ingest", variant="primary")
                ingest_status = gr.Markdown()
                docs_md = gr.Markdown(list_documents(API_KEY, "public"))
                refresh_docs = gr.Button("Refresh document list", size="sm")

            with gr.Column(scale=2):
                with gr.Tab("💬 Chat"):
                    chatbot = gr.Chatbot(label="Ask about your documents", type="messages", height=480)
                    msg_box = gr.Textbox(label="Your question", placeholder="Type a question and press Enter")
                    clear_btn = gr.Button("Clear conversation", size="sm")

                with gr.Tab("📊 Evaluation"):
                    gr.Markdown("Retrieval & generation evaluation (ablation)")
                    eval_btn = gr.Button("Run evaluation", variant="primary")
                    eval_out = gr.Markdown()

        # ---- Wiring ----
        refresh_health.click(check_health, outputs=health_md)
        refresh_docs.click(list_documents, inputs=[api_key_box, tenant_box], outputs=docs_md)
        ingest_btn.click(
            ingest_document,
            inputs=[file_box, api_key_box, tenant_box],
            outputs=[ingest_status, docs_md],
        )

        def _submit(message, history, conv_id, api_key, tenant):
            history, conv_id = chat(message, history, conv_id, api_key, tenant)
            return history, conv_id, ""

        msg_box.submit(
            _submit,
            inputs=[msg_box, chatbot, conversation_id, api_key_box, tenant_box],
            outputs=[chatbot, conversation_id, msg_box],
        )

        def _clear():
            return [], None

        clear_btn.click(_clear, outputs=[chatbot, conversation_id])
        eval_btn.click(run_evaluation, inputs=[api_key_box, tenant_box], outputs=eval_out)

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=SERVER_PORT)

