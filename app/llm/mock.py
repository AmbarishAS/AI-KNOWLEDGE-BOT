"""Deterministic mock LLM.

It produces a *grounded* answer by quoting the numbered passages injected into the prompt,
emitting citation markers like ``[1]``. This lets the whole RAG pipeline — including
citation validation and abstention — be tested offline. Responses are tagged
``provider='mock'`` so they are never mistaken for real model output.

It also understands a couple of control markers used by the pipeline:
- ``NO_CONTEXT`` in the prompt → the model abstains.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from app.domain import Message, Role, TokenUsage
from app.llm.base import ChatResult
from app.llm.tokens import count_tokens

_PASSAGE = re.compile(r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\Z)", re.DOTALL)
_FIRST_SENTENCE = re.compile(r"(.+?[.!?])(\s|$)")

ABSTAIN = "I don't have enough information in the provided documents to answer that confidently."


def _first_sentence(text: str) -> str:
    text = text.strip().replace("\n", " ")
    m = _FIRST_SENTENCE.match(text)
    return (m.group(1) if m else text)[:300].strip()


class MockLLM:
    name = "mock"

    def __init__(self, model: str = "mock") -> None:
        self.model = model

    def _answer(self, messages: list[Message]) -> str:
        user_text = "\n".join(m.content for m in messages if m.role == Role.USER)
        if "NO_CONTEXT" in user_text:
            return ABSTAIN
        passages = _PASSAGE.findall(user_text)
        passages = [(n, t) for n, t in passages if t.strip()]
        if not passages:
            return ABSTAIN
        parts = []
        for num, text in passages[:2]:
            parts.append(f"{_first_sentence(text)} [{num}]")
        return "Based on the documents: " + " ".join(parts)

    async def chat(self, messages: list[Message], temperature: float = 0.1, max_tokens: int = 800) -> ChatResult:
        text = self._answer(messages)
        prompt_tokens = sum(count_tokens(m.content, self.model) for m in messages)
        completion_tokens = count_tokens(text, self.model)
        return ChatResult(
            text=text,
            usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            provider=self.name,
            model=self.model,
        )

    async def stream(
        self, messages: list[Message], temperature: float = 0.1, max_tokens: int = 800
    ) -> AsyncIterator[str]:
        text = self._answer(messages)
        for token in text.split(" "):
            yield token + " "
