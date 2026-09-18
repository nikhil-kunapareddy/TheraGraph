"""Shared service context.

One object carries the collaborators every service needs -- settings, database session,
cache, inference service, object store -- so services take a single argument instead of
six, and tests can build a context with in-memory implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.config import Settings
from theragraph.core.cache import Cache
from theragraph.core.security import ANONYMOUS, Principal
from theragraph.core.storage import ObjectStore
from theragraph.ml.inference import InferenceService


@dataclass
class ServiceContext:
    settings: Settings
    session: AsyncSession
    cache: Cache | None = None
    store: ObjectStore | None = None
    inference: InferenceService | None = None
    principal: Principal = ANONYMOUS
    request_id: str | None = None

    def require_inference(self) -> InferenceService:
        if self.inference is None:
            self.inference = InferenceService(self.settings, cache=self.cache, store=self.store)
        return self.inference

    def with_session(self, session: AsyncSession) -> ServiceContext:
        return ServiceContext(
            settings=self.settings,
            session=session,
            cache=self.cache,
            store=self.store,
            inference=self.inference,
            principal=self.principal,
            request_id=self.request_id,
        )

    def audit(self) -> dict[str, Any]:
        return {
            "principal": self.principal.subject,
            "auth_method": self.principal.auth_method,
            "request_id": self.request_id,
        }
