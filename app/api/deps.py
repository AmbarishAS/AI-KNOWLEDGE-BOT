"""FastAPI dependencies: settings, auth principal, and the DI container."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings
from app.container import Container, get_container
from app.core.security import AuthError, authenticate
from app.domain import Principal


def settings_dep() -> Settings:
    return get_settings()


def container_dep() -> Container:
    return get_container()


def principal_dep(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_tenant: str | None = Header(default=None, alias="X-Tenant"),
    settings: Settings = Depends(settings_dep),
) -> Principal:
    """Authenticate the request and resolve the calling principal + tenant scope."""
    try:
        return authenticate(x_api_key, x_tenant, settings)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "ApiKey"},
        ) from exc
