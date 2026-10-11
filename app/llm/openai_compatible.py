"""OpenAI-compatible chat LLM (real mode).

httpx client against ``/chat/completions`` with bounded retries, timeout, and streaming.
Only used when ``APP_MODE=real``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from app.domain import Message, TokenUsage
from app.llm.base import ChatResult
from app.logging_config import get_logger

logger = get_logger(__name__)


class OpenAICompatibleLLM:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout
        self._max_retries = max_retries

    def _payload(self, messages, temperature, max_tokens, stream=False):
        return {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    async def chat(self, messages: list[Message], temperature: float = 0.1, max_tokens: int = 800) -> ChatResult:
        payload = self._payload(messages, temperature, max_tokens)
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(f"{self._base_url}/chat/completions", json=payload, headers=self._headers)
                    if resp.status_code in (429, 500, 502, 503):
                        raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                    resp.raise_for_status()
                    data = resp.json()
                    choice = data["choices"][0]
                    usage = data.get("usage", {})
                    return ChatResult(
                        text=choice["message"]["content"],
                        usage=TokenUsage(
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                        ),
                        provider=self.name,
                        model=self.model,
                        finish_reason=choice.get("finish_reason", "stop"),
                    )
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_exc = exc
                logger.warning(f"llm retry {attempt + 1}/{self._max_retries}: {exc}")
                await asyncio.sleep((2**attempt) * 0.1)
        raise RuntimeError(f"LLM request failed after retries: {last_exc}")

    async def stream(
        self, messages: list[Message], temperature: float = 0.1, max_tokens: int = 800
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, temperature, max_tokens, stream=True)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", f"{self._base_url}/chat/completions", json=payload, headers=self._headers
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[len("data: ") :]
                    if chunk.strip() == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
