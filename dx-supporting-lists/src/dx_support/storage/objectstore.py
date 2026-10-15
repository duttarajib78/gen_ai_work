"""A tiny object-store abstraction: bytes in, bytes out, under validated keys.

Two implementations -- the local file system (development, tests, a single-machine batch run) and Amazon S3 (production: shared by every
instance of the endpoint and by batch jobs) -- behind one interface, so the pipeline never knows which it is using.

Safety properties
  * keys are validated (letters, digits, ``._-=`` and ``/`` separators; no empty, ``.`` or ``..`` segments), so a key built from a request value
    can never escape the store root;
  * local writes are atomic (temp file + rename): a crash never leaves a half-written object;
  * ``put_if_absent`` is atomic on both back ends (``O_EXCL`` locally, S3 conditional ``If-None-Match: *`` writes), which is what the run
    lock needs;
  * the S3 store sends server-side-encryption headers only when a KMS key is configured; otherwise it relies on the bucket's default
    encryption, which the provided CloudFormation template enforces.
"""
from __future__ import annotations

import contextlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import unquote, urlparse

_KEY_RE = re.compile(r"^[A-Za-z0-9._=\-]+(?:/[A-Za-z0-9._=\-]+)*/?$")
MAX_KEY_CHARS = 512


class StoreError(RuntimeError):
    pass


def validate_key(key: str) -> str:
    if not isinstance(key, str) or not key or len(key) > MAX_KEY_CHARS or not _KEY_RE.match(key):
        raise StoreError(f"invalid object key {key!r}")
    if any(seg in (".", "..") for seg in key.split("/")):
        raise StoreError(f"invalid object key {key!r}")
    return key


class ObjectStore(Protocol):
    uri: str

    def get(self, key: str) -> Optional[bytes]: ...
    def put(self, key: str, data: bytes) -> None: ...
    def put_if_absent(self, key: str, data: bytes) -> bool: ...
    def delete(self, key: str) -> None: ...
    def list(self, prefix: str) -> list[str]: ...


class LocalObjectStore:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.uri = self.root.as_uri()

    def _path(self, key: str) -> Path:
        p = (self.root / validate_key(key)).resolve()
        if self.root != p and self.root not in p.parents:
            raise StoreError(f"key {key!r} escapes the store root")
        return p

    def get(self, key: str) -> Optional[bytes]:
        try:
            return self._path(key).read_bytes()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, p)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

    def put_if_absent(self, key: str, data: bytes) -> bool:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return True

    def delete(self, key: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._path(key).unlink()

    def list(self, prefix: str) -> list[str]:
        prefix = prefix.lstrip("/")
        if prefix:
            validate_key(prefix)
        base_dir = self.root / (prefix.rsplit("/", 1)[0] if "/" in prefix else "")
        if not base_dir.exists():
            return []
        out = []
        for dirpath, _dirs, files in os.walk(base_dir):
            for name in files:
                if name.startswith(".tmp-"):
                    continue
                key = (Path(dirpath) / name).relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    out.append(key)
        return sorted(out)


class S3ObjectStore:
    def __init__(self, bucket: str, prefix: str = "", client: Any = None, region: Optional[str] = None, kms_key_id: Optional[str] = None):
        if not re.match(r"^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$", bucket or ""):
            raise StoreError(f"invalid S3 bucket name {bucket!r}")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.kms_key_id = kms_key_id
        self.region = region
        self._client = client
        self.uri = f"s3://{bucket}" + (f"/{self.prefix}" if self.prefix else "")

    def _s3(self) -> Any:
        if self._client is None:
            import boto3
            from botocore.config import Config
            cfg = Config(region_name=self.region, retries={"max_attempts": 5, "mode": "standard"}, connect_timeout=5, read_timeout=30,
                         max_pool_connections=32)
            self._client = boto3.session.Session().client("s3", config=cfg)
        return self._client

    def _key(self, key: str) -> str:
        validate_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    @staticmethod
    def _code(exc: BaseException) -> str:
        return str(((getattr(exc, "response", None) or {}).get("Error") or {}).get("Code") or "")

    def _sse(self) -> dict:
        return {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": self.kms_key_id} if self.kms_key_id else {}

    def get(self, key: str) -> Optional[bytes]:
        try:
            return self._s3().get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except Exception as e:                        # noqa: BLE001
            if self._code(e) in ("NoSuchKey", "404", "NotFound"):
                return None
            raise

    def put(self, key: str, data: bytes) -> None:
        self._s3().put_object(Bucket=self.bucket, Key=self._key(key), Body=data, ContentType="application/json", **self._sse())

    def put_if_absent(self, key: str, data: bytes) -> bool:
        try:
            self._s3().put_object(Bucket=self.bucket, Key=self._key(key), Body=data, ContentType="application/json", IfNoneMatch="*", **self._sse())
            return True
        except Exception as e:                        # noqa: BLE001
            if self._code(e) in ("PreconditionFailed", "412", "ConditionalRequestConflict", "409"):
                return False
            raise

    def delete(self, key: str) -> None:
        self._s3().delete_object(Bucket=self.bucket, Key=self._key(key))

    def list(self, prefix: str) -> list[str]:
        if prefix:
            validate_key(prefix)
        full = f"{self.prefix}/{prefix}" if self.prefix else prefix
        strip = len(self.prefix) + 1 if self.prefix else 0
        out: list[str] = []
        for page in self._s3().get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=full):
            out.extend(o["Key"][strip:] for o in page.get("Contents", []))
        return sorted(out)


class PrefixedStore:
    """A view of another store under a fixed key prefix (a namespace)."""

    def __init__(self, inner: ObjectStore, prefix: str):
        self.inner = inner
        self.prefix = validate_key(prefix).strip("/")
        self.uri = f"{inner.uri.rstrip('/')}/{self.prefix}"

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{validate_key(key)}"

    def get(self, key: str) -> Optional[bytes]:
        return self.inner.get(self._k(key))

    def put(self, key: str, data: bytes) -> None:
        self.inner.put(self._k(key), data)

    def put_if_absent(self, key: str, data: bytes) -> bool:
        return self.inner.put_if_absent(self._k(key), data)

    def delete(self, key: str) -> None:
        self.inner.delete(self._k(key))

    def list(self, prefix: str) -> list[str]:
        strip = len(self.prefix) + 1
        return [k[strip:] for k in self.inner.list(f"{self.prefix}/{prefix}")]


def open_store(uri: str, *, region: Optional[str] = None, kms_key_id: Optional[str] = None) -> ObjectStore:
    """`s3://bucket/prefix`, `file:///absolute/path`, or a plain directory path."""
    if not uri:
        raise StoreError("no store URI configured (set DX_STORE_URI)")
    if uri.startswith("s3://"):
        u = urlparse(uri)
        return S3ObjectStore(u.netloc, u.path, region=region, kms_key_id=kms_key_id)
    if uri.startswith("file://"):
        return LocalObjectStore(Path(unquote(urlparse(uri).path)))
    if "://" in uri:
        raise StoreError("DX_STORE_URI must be s3://bucket/prefix or a local path")
    return LocalObjectStore(Path(uri))


def namespace(store: ObjectStore, *parts: str) -> ObjectStore:
    """A namespaced view: `namespace(store, "serve", fingerprint)`."""
    ns = "/".join(p.strip("/") for p in parts if p)
    return PrefixedStore(store, ns) if ns else store
