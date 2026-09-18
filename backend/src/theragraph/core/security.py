"""Authentication, authorisation and secret handling.

TheraGraph supports two credential types:

* **API keys** for programmatic/service access (constant-time compared, never logged).
* **JWT bearer tokens** for the web UI.

Auth is *off* by default for local development and *required* when
``AUTH_ENABLED=true`` (the deployed demo). A production boot with auth enabled and a
default secret refuses to start -- failing closed is the only safe default.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

import jwt

from theragraph.core.clock import utcnow
from theragraph.core.errors import AuthenticationError, AuthorizationError
from theragraph.core.logging import get_logger

log = get_logger(__name__)

Role = Literal["viewer", "researcher", "admin"]
_ROLE_RANK: dict[str, int] = {"viewer": 0, "researcher": 1, "admin": 2}

DEFAULT_INSECURE_SECRET = "dev-only-insecure-secret-change-me"

_SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret|password|token)\b\s*[=:]\s*\S+"),
)


@dataclass(frozen=True)
class Principal:
    """The authenticated caller."""

    subject: str
    role: Role = "researcher"
    auth_method: Literal["anonymous", "api_key", "jwt"] = "anonymous"

    @property
    def is_anonymous(self) -> bool:
        return self.auth_method == "anonymous"

    def require_role(self, minimum: Role) -> None:
        if _ROLE_RANK[self.role] < _ROLE_RANK[minimum]:
            raise AuthorizationError(
                f"role '{self.role}' cannot perform this action (requires '{minimum}')"
            )


ANONYMOUS = Principal(subject="anonymous", role="researcher", auth_method="anonymous")


def hash_api_key(key: str) -> str:
    """Stable fingerprint of an API key, safe to log and store."""
    return "tgk_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def verify_api_key(candidate: str, allowed: list[str]) -> Principal:
    """Constant-time API key check."""
    candidate_digest = hashlib.sha256(candidate.encode()).digest()
    for key in allowed:
        if hmac.compare_digest(candidate_digest, hashlib.sha256(key.encode()).digest()):
            return Principal(
                subject=hash_api_key(candidate), role="researcher", auth_method="api_key"
            )
    raise AuthenticationError("invalid API key")


def issue_token(
    subject: str,
    *,
    secret: str,
    algorithm: str = "HS256",
    ttl_s: int = 86_400,
    role: Role = "researcher",
) -> str:
    now = utcnow()
    payload = {
        "sub": subject,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl_s)).timestamp()),
        "iss": "theragraph",
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def verify_token(token: str, *, secret: str, algorithms: list[str] | None = None) -> Principal:
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            secret,
            algorithms=algorithms or ["HS256"],
            issuer="theragraph",
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError(f"invalid token: {exc}") from exc

    role = payload.get("role", "viewer")
    if role not in _ROLE_RANK:
        raise AuthenticationError(f"unknown role in token: {role!r}")
    return Principal(subject=str(payload["sub"]), role=role, auth_method="jwt")


def assert_secrets_are_safe(*, environment: str, auth_enabled: bool, jwt_secret: str) -> None:
    """Refuse to serve production traffic with development secrets."""
    if environment != "production":
        return
    if auth_enabled and jwt_secret == DEFAULT_INSECURE_SECRET:
        raise RuntimeError(
            "JWT_SECRET is still the development default; set a strong secret before "
            "running in production."
        )


def redact(text: str, *, max_len: int = 4000) -> str:
    """Strip anything that looks like a credential before logging or persisting."""
    cleaned = text
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("[REDACTED]", cleaned)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + f"... [truncated {len(cleaned) - max_len} chars]"
    return cleaned


def redact_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact obviously sensitive keys in a dict."""
    sensitive = {"api_key", "apikey", "authorization", "password", "secret", "token"}
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in sensitive:
            out[key] = "[REDACTED]"
        elif isinstance(value, dict):
            out[key] = redact_mapping(value)
        elif isinstance(value, str):
            out[key] = redact(value)
        else:
            out[key] = value
    return out
