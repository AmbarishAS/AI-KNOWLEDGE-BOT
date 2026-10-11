"""Chat service: conversation-aware RAG.

Responsibilities:
- ensure/track a conversation for the principal (tenant-scoped),
- load **bounded** prior history (not the whole transcript — see memory discussion in
  ``docs/ARCHITECTURE.md`` and ``docs/INTERVIEW_GUIDE.md``),
- call the RAG pipeline,
- persist the user turn and grounded assistant turn with citations.
"""

from __future__ import annotations

from app.config import Settings
from app.domain import Answer, Message, Principal, Role
from app.persistence.db import session_scope
from app.persistence.repositories import ConversationRepository
from app.rag import RAGConfig, RAGPipeline

# Keep only the last N turns in the prompt to bound token cost and reduce memory
# contamination / drift. Older context lives in the DB but is not blindly replayed.
MAX_HISTORY_MESSAGES = 6


class ChatService:
    def __init__(self, pipeline: RAGPipeline, settings: Settings) -> None:
        self._pipeline = pipeline
        self._settings = settings

    async def chat(
        self,
        principal: Principal,
        question: str,
        conversation_id: str | None = None,
        use_memory: bool = True,
        config: RAGConfig | None = None,
    ) -> tuple[str, Answer]:
        async with session_scope() as session:
            convo = ConversationRepository(session)
            cid = await convo.ensure(principal.tenant, principal.subject, conversation_id)
            history: list[Message] = []
            if use_memory:
                history = await convo.get_messages(principal.tenant, cid, limit=MAX_HISTORY_MESSAGES)

        answer = await self._pipeline.answer(question, principal.tenant, history=history, config=config)

        async with session_scope() as session:
            convo = ConversationRepository(session)
            await convo.add_message(principal.tenant, cid, Message(role=Role.USER, content=question))
            await convo.add_message(
                principal.tenant,
                cid,
                Message(role=Role.ASSISTANT, content=answer.text),
                citations=[c.model_dump() for c in answer.citations],
            )
        return cid, answer

    async def get_history(self, principal: Principal, conversation_id: str) -> list[Message]:
        async with session_scope() as session:
            return await ConversationRepository(session).get_messages(principal.tenant, conversation_id)

    async def delete_conversation(self, principal: Principal, conversation_id: str) -> bool:
        async with session_scope() as session:
            return await ConversationRepository(session).delete(principal.tenant, conversation_id)
