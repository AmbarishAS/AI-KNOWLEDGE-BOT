"""Application configuration using Pydantic settings.

All configuration is read from environment variables (or a local ``.env`` file). This is
the single, typed source of truth for runtime behaviour, which keeps secrets out of code
and makes provider selection explicit (see ``docs/ARCHITECTURE.md`` section 7).
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppMode(str, Enum):
    """Determines whether real external providers are used."""

    MOCK = "mock"
    REAL = "real"


class RetrievalMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    STRUCTURE = "structure"
    SEMANTIC = "semantic"
    PARENT_CHILD = "parent_child"


class Settings(BaseSettings):
    """Typed application settings loaded from the environment."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False)

    # Core
    app_mode: AppMode = AppMode.MOCK
    app_env: str = "local"
    log_level: str = "INFO"
    log_json: bool = True

    # Security / API
    api_auth_enabled: bool = True
    demo_api_key: str = "changeme-dev-key"
    default_tenant: str = "public"
    max_upload_mb: int = 10
    allowed_upload_extensions: str = ".pdf,.docx,.md,.txt"

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/copilot.db"

    # Vector store
    vector_store: str = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "copilot_chunks"

    # LLM / embeddings (real mode)
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 800
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Retrieval / RAG
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = 5
    rerank_enabled: bool = True
    score_threshold: float = 0.15
    enable_query_rewrite: bool = False

    # Agent
    agent_max_steps: int = 6
    agent_max_retries: int = 2

    # Provider reliability
    http_timeout_seconds: float = 30.0
    http_max_retries: int = 3

    @field_validator("allowed_upload_extensions")
    @classmethod
    def _normalize_exts(cls, v: str) -> str:
        return ",".join(e.strip().lower() for e in v.split(",") if e.strip())

    @property
    def allowed_extensions(self) -> set[str]:
        return {e if e.startswith(".") else f".{e}" for e in self.allowed_upload_extensions.split(",")}

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_mock(self) -> bool:
        return self.app_mode == AppMode.MOCK

    def require_real_credentials(self) -> None:
        """Fail fast if real mode is requested without credentials."""
        if self.app_mode == AppMode.REAL and not self.openai_api_key:
            raise ValueError(
                "APP_MODE=real requires OPENAI_API_KEY to be set. " "Set APP_MODE=mock for a no-credential demo."
            )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (one per process)."""
    return Settings()
