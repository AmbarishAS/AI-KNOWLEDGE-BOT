"""Security primitives: API-key auth, tenant scoping, and upload validation.

This is intentionally simple for a local demo, with a clearly marked extension point for
OAuth2/OIDC. The goal is to demonstrate the *boundaries* (authn, authz, tenant isolation,
safe file handling) rather than ship a full IdP integration.
"""

from __future__ import annotations

import os
import re
import unicodedata

from app.config import Settings
from app.domain import Principal

# ---- Authentication -------------------------------------------------------------------


class AuthError(Exception):
    """Raised when authentication fails."""


def authenticate(api_key: str | None, tenant: str | None, settings: Settings) -> Principal:
    """Validate the demo API key and return the calling principal.

    Extension point: replace this with OAuth2/OIDC token validation. The rest of the app
    only depends on the returned :class:`Principal`, so swapping the mechanism is isolated.
    """
    if not settings.api_auth_enabled:
        return Principal(tenant=tenant or settings.default_tenant)
    if not api_key or api_key != settings.demo_api_key:
        raise AuthError("Invalid or missing API key")
    return Principal(subject="demo-user", tenant=tenant or settings.default_tenant)


# ---- Safe filename handling -----------------------------------------------------------

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(filename: str) -> str:
    """Return a filesystem-safe, path-traversal-proof filename.

    Strips directory components, normalizes unicode, and removes unsafe characters.
    """
    # Drop any path component (prevents ../../ traversal).
    base = os.path.basename(filename.replace("\\", "/"))
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    base = _SAFE_CHARS.sub("_", base).strip("._")
    if not base:
        base = "upload"
    return base[:200]


# ---- Upload validation ----------------------------------------------------------------


class UploadValidationError(Exception):
    """Raised when an uploaded file fails validation."""


def validate_upload(filename: str, size_bytes: int, settings: Settings) -> str:
    """Validate extension and size; return the sanitized filename.

    Raises :class:`UploadValidationError` with a clear message on failure.
    """
    safe = safe_filename(filename)
    ext = os.path.splitext(safe)[1].lower()
    if ext not in settings.allowed_extensions:
        raise UploadValidationError(f"Unsupported file type '{ext}'. Allowed: {sorted(settings.allowed_extensions)}")
    if size_bytes <= 0:
        raise UploadValidationError("Empty upload")
    if size_bytes > settings.max_upload_bytes:
        raise UploadValidationError(f"File too large ({size_bytes} bytes). Limit is {settings.max_upload_mb} MB.")
    return safe
