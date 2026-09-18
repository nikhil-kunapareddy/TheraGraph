"""Object storage for model artifacts, dataset snapshots and rendered reports.

Two interchangeable backends behind one protocol:

* :class:`FilesystemObjectStore` -- default for local dev/tests (no MinIO needed).
* :class:`S3ObjectStore` -- S3 or MinIO in deployed environments.

Keys are POSIX-style paths (``models/solubility/rf/2026.09.1/model.joblib``). boto3 is
synchronous, so S3 calls are dispatched to a thread pool to keep the event loop free.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from theragraph.core.errors import NotFoundError
from theragraph.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class StoredObject:
    key: str
    size_bytes: int
    uri: str


def _validate_key(key: str) -> str:
    """Reject traversal and absolute keys -- these become filesystem paths."""
    normalized = key.strip().lstrip("/")
    if not normalized:
        raise ValueError("object key must not be empty")
    parts = Path(normalized).parts
    if any(part in ("..", "~") for part in parts):
        raise ValueError(f"unsafe object key: {key!r}")
    return normalized


class ObjectStore(Protocol):
    async def put_bytes(
        self, key: str, data: bytes, *, content_type: str = ...
    ) -> StoredObject: ...
    async def get_bytes(self, key: str) -> bytes: ...
    async def put_file(self, key: str, path: Path) -> StoredObject: ...
    async def get_file(self, key: str, destination: Path) -> Path: ...
    async def exists(self, key: str) -> bool: ...
    async def list_prefix(self, prefix: str) -> list[str]: ...
    async def delete(self, key: str) -> None: ...
    def uri_for(self, key: str) -> str: ...


class FilesystemObjectStore:
    """Stores objects under a root directory. Durable enough for local dev."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / _validate_key(key)

    async def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> StoredObject:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # write-then-rename for atomicity
        tmp = path.with_suffix(path.suffix + ".tmp")
        await asyncio.to_thread(tmp.write_bytes, data)
        await asyncio.to_thread(tmp.replace, path)
        return StoredObject(key=_validate_key(key), size_bytes=len(data), uri=self.uri_for(key))

    async def get_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise NotFoundError(f"object not found: {key}")
        return await asyncio.to_thread(path.read_bytes)

    async def put_file(self, key: str, path: Path) -> StoredObject:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, path, target)
        return StoredObject(
            key=_validate_key(key), size_bytes=target.stat().st_size, uri=self.uri_for(key)
        )

    async def get_file(self, key: str, destination: Path) -> Path:
        source = self._path(key)
        if not source.exists():
            raise NotFoundError(f"object not found: {key}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, source, destination)
        return destination

    async def exists(self, key: str) -> bool:
        return self._path(key).exists()

    async def list_prefix(self, prefix: str) -> list[str]:
        base = self.root / prefix.strip("/")
        search_root = base if base.is_dir() else base.parent
        if not search_root.exists():
            return []
        return sorted(
            str(p.relative_to(self.root))
            for p in search_root.rglob("*")
            if p.is_file() and not p.name.endswith(".tmp")
        )

    async def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            await asyncio.to_thread(path.unlink)

    def uri_for(self, key: str) -> str:
        return f"file://{self._path(key)}"


class S3ObjectStore:
    """S3 / MinIO backed store."""

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
    ) -> None:
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self._client: Any = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(
                retries={"max_attempts": 4, "mode": "standard"},
                signature_version="s3v4",
            ),
        )

    async def ensure_bucket(self) -> None:
        def _ensure() -> None:
            from botocore.exceptions import ClientError

            try:
                self._client.head_bucket(Bucket=self.bucket)
            except ClientError:
                self._client.create_bucket(Bucket=self.bucket)

        await asyncio.to_thread(_ensure)

    async def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> StoredObject:
        key = _validate_key(key)
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return StoredObject(key=key, size_bytes=len(data), uri=self.uri_for(key))

    async def get_bytes(self, key: str) -> bytes:
        key = _validate_key(key)

        def _get() -> bytes:
            from botocore.exceptions import ClientError

            try:
                response = self._client.get_object(Bucket=self.bucket, Key=key)
            except ClientError as exc:
                raise NotFoundError(f"object not found: {key}") from exc
            return response["Body"].read()  # type: ignore[no-any-return]

        return await asyncio.to_thread(_get)

    async def put_file(self, key: str, path: Path) -> StoredObject:
        key = _validate_key(key)

        def _upload() -> int:
            self._client.upload_file(str(path), self.bucket, key)
            return path.stat().st_size

        size_bytes = await asyncio.to_thread(_upload)
        return StoredObject(key=key, size_bytes=size_bytes, uri=self.uri_for(key))

    async def get_file(self, key: str, destination: Path) -> Path:
        key = _validate_key(key)

        def _download() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self.bucket, key, str(destination))

        await asyncio.to_thread(_download)
        return destination

    async def exists(self, key: str) -> bool:
        def _head() -> bool:
            from botocore.exceptions import ClientError

            try:
                self._client.head_object(Bucket=self.bucket, Key=_validate_key(key))
                return True
            except ClientError:
                return False

        return await asyncio.to_thread(_head)

    async def list_prefix(self, prefix: str) -> list[str]:
        def _list() -> list[str]:
            paginator = self._client.get_paginator("list_objects_v2")
            keys: list[str] = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                keys.extend(obj["Key"] for obj in page.get("Contents", []))
            return sorted(keys)

        return await asyncio.to_thread(_list)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(
            self._client.delete_object, Bucket=self.bucket, Key=_validate_key(key)
        )

    def uri_for(self, key: str) -> str:
        return f"s3://{self.bucket}/{_validate_key(key)}"


def build_object_store(
    *,
    backend: str,
    root: Path,
    bucket: str,
    endpoint_url: str | None,
    access_key: str,
    secret_key: str,
    region: str,
) -> ObjectStore:
    if backend == "s3":
        log.info("storage.backend", backend="s3", bucket=bucket, endpoint=endpoint_url)
        return S3ObjectStore(
            bucket,
            endpoint_url=endpoint_url,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
        )
    log.info("storage.backend", backend="filesystem", root=str(root))
    return FilesystemObjectStore(root)
