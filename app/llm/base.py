"""LLM provider interface and result types.

Programming against this interface (not a vendor SDK) is what lets us swap mock ↔ real and
even swap providers for cost/latency reasons without touching business logic (ADR 0001).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import BaseModel

from app.domain import Message, TokenUsage


class ChatResult(BaseModel):
    text: str
    usage: TokenUsage
    provider: str
    model: str
    finish_reason: str = "stop"


class LLMProvider(Protocol):
    name: str
    model: str

    async def chat(
        self,
        messages: list[Message],
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> ChatResult: ...

    async def stream(
        self,
        messages: list[Message],
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> AsyncIterator[str]: ...
