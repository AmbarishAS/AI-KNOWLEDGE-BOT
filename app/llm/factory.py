"""LLM factory (mock vs real), driven by Settings."""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.mock import MockLLM


def create_llm(settings: Settings) -> LLMProvider:
    if settings.is_mock:
        return MockLLM(model="mock")
    settings.require_real_credentials()
    from app.llm.openai_compatible import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.llm_model,
        timeout=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
    )
