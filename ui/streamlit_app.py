"""Streamlit demo UI for the Enterprise AI Knowledge Copilot.

Talks to the FastAPI backend over HTTP. Supports: document upload, document listing, chat
with visible citations, conversation history, and a simple evaluation results view.

Run:
    streamlit run ui/streamlit_app.py
Environment:
    COPILOT_API_URL (default http://localhost:8000)
    COPILOT_API_KEY (default changeme-dev-key)
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("COPILOT_API_URL", "http://localhost:8000")
API_KEY = os.getenv("COPILOT_API_KEY", "changeme-dev-key")

st.set_page_config(page_title="AI Knowledge Copilot", page_icon="🤖", layout="wide")


def headers() -> dict:
    return {"X-API-Key": st.session_state.get("api_key", API_KEY), "X-Tenant": st.session_state.get("tenant", "public")}


def client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=60.0)


# ---- Session state ----
st.session_state.setdefault("conversation_id", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("api_key", API_KEY)
st.session_state.setdefault("tenant", "public")

# ---- Sidebar: connection + documents ----
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state["api_key"] = st.text_input("API key", st.session_state["api_key"], type="password")
    st.session_state["tenant"] = st.text_input("Tenant", st.session_state["tenant"])

    try:
        with client() as c:
            health = c.get("/health").json()
        st.success(f"API up · mode={health.get('mode')} · v{health.get('version')}")
    except Exception as exc:
        st.error(f"API unreachable at {API_URL}: {exc}")

    st.divider()
    st.header("📄 Documents")
    uploaded = st.file_uploader("Upload (pdf/docx/md/txt)", type=["pdf", "docx", "md", "txt"])
    if uploaded and st.button("Ingest"):
        with client() as c:
            resp = c.post(
                "/api/v1/documents",
                headers=headers(),
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            )
        if resp.status_code == 201:
            st.success(f"Ingested {uploaded.name} ({resp.json()['num_chunks']} chunks)")
        else:
            st.error(f"{resp.status_code}: {resp.text}")

    if st.button("Refresh document list"):
        pass
    try:
        with client() as c:
            docs = c.get("/api/v1/documents", headers=headers()).json()
        st.caption(f"{docs['total']} document(s) in tenant '{st.session_state['tenant']}'")
        for d in docs["items"]:
            st.write(f"• {d['filename']} — {d['num_chunks']} chunks ({d['chunk_strategy']})")
    except Exception:
        st.caption("No documents yet.")

# ---- Main: tabs ----
chat_tab, eval_tab = st.tabs(["💬 Chat", "📊 Evaluation"])

with chat_tab:
    st.subheader("Ask about your documents")
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            for cit in m.get("citations", []):
                st.caption(f"{cit['marker']} {cit['filename']} — {cit['snippet'][:120]}…")

    if prompt := st.chat_input("Your question"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with client() as c:
            resp = c.post(
                "/api/v1/chat",
                headers=headers(),
                json={"question": prompt, "conversation_id": st.session_state["conversation_id"]},
            )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state["conversation_id"] = data["conversation_id"]
            with st.chat_message("assistant"):
                st.markdown(data["answer"])
                if data["abstained"]:
                    st.info("The assistant abstained (insufficient grounded evidence).")
                for cit in data["citations"]:
                    st.caption(f"{cit['marker']} **{cit['filename']}** — {cit['snippet'][:160]}…")
                cols = st.columns(4)
                cols[0].metric("Prompt tok", data["prompt_tokens"])
                cols[1].metric("Completion tok", data["completion_tokens"])
                cols[2].metric("Cost $", f"{data['estimated_cost_usd']:.6f}")
                cols[3].metric("Latency ms", f"{data['latency_ms']:.0f}")
            st.session_state["messages"].append(
                {"role": "assistant", "content": data["answer"], "citations": data["citations"]}
            )
        else:
            st.error(f"{resp.status_code}: {resp.text}")

with eval_tab:
    st.subheader("Retrieval & generation evaluation (ablation)")
    if st.button("Run evaluation"):
        with st.spinner("Running evaluation…"), client() as c:
            resp = c.post("/api/v1/evaluations/run", headers=headers(), json={})
        if resp.status_code == 200:
            report = resp.json()
            st.write("**Retrieval (baseline dense vs improved hybrid+rerank)**")
            st.table(
                {
                    "baseline_dense": report["retrieval"]["baseline_dense"],
                    "improved_hybrid_rerank": report["retrieval"]["improved_hybrid_rerank"],
                }
            )
            st.write("**Generation**")
            st.json(report["generation"])
        else:
            st.error(f"{resp.status_code}: {resp.text}")
