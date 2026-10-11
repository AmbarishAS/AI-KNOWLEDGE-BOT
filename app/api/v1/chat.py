"""Chat endpoints: grounded RAG answer + streaming + conversation retrieval."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response, StreamingResponse

from app.api.deps import container_dep, principal_dep
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationResponse,
    ConversationResponse,
    MessageResponse,
    RetrievedPassage,
)
from app.container import Container
from app.domain import Principal
from app.rag import RAGConfig

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _to_config(req: ChatRequest) -> RAGConfig:
    return RAGConfig(rerank=req.rerank, multi_query=req.multi_query, top_k=req.top_k)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    principal: Principal = Depends(principal_dep),
    container: Container = Depends(container_dep),
) -> ChatResponse:
    cid, answer = await container.chat.chat(
        principal,
        req.question,
        conversation_id=req.conversation_id,
        use_memory=req.use_memory,
        config=_to_config(req),
    )
    return ChatResponse(
        conversation_id=cid,
        answer=answer.text,
        abstained=answer.abstained,
        citations=[CitationResponse.from_citation(c) for c in answer.citations],
        retrieved=[RetrievedPassage.from_scored(s) for s in answer.retrieved],
        provider=answer.provider,
        mode=answer.mode,
        prompt_tokens=answer.usage.prompt_tokens,
        completion_tokens=answer.usage.completion_tokens,
        estimated_cost_usd=answer.estimated_cost_usd,
        latency_ms=round(answer.latency_ms, 1),
    )


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    principal: Principal = Depends(principal_dep),
    container: Container = Depends(container_dep),
) -> StreamingResponse:
    """Server-sent-events style streaming.

    We compute the grounded answer first (so citations stay valid), then stream it token by
    token followed by a final metadata event. This keeps streaming compatible with citation
    validation (we never stream ungrounded tokens).
    """
    cid, answer = await container.chat.chat(
        principal,
        req.question,
        conversation_id=req.conversation_id,
        use_memory=req.use_memory,
        config=_to_config(req),
    )

    async def event_gen():
        yield f"data: {json.dumps({'conversation_id': cid})}\n\n"
        for token in answer.text.split(" "):
            yield f"data: {json.dumps({'token': token + ' '})}\n\n"
        meta = {
            "done": True,
            "abstained": answer.abstained,
            "citations": [CitationResponse.from_citation(c).model_dump() for c in answer.citations],
            "prompt_tokens": answer.usage.prompt_tokens,
            "completion_tokens": answer.usage.completion_tokens,
            "estimated_cost_usd": answer.estimated_cost_usd,
        }
        yield f"data: {json.dumps(meta)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    principal: Principal = Depends(principal_dep),
    container: Container = Depends(container_dep),
) -> ConversationResponse:
    messages = await container.chat.get_history(principal, conversation_id)
    if not messages:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found or empty")
    return ConversationResponse(
        conversation_id=conversation_id,
        messages=[MessageResponse(role=m.role, content=m.content) for m in messages],
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_conversation(
    conversation_id: str,
    principal: Principal = Depends(principal_dep),
    container: Container = Depends(container_dep),
):
    deleted = await container.chat.delete_conversation(principal, conversation_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
