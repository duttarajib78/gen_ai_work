"""Model artifact bundles.

A *serving bundle* is the directory SageMaker extracts from ``model.tar.gz``::

    model.safetensors     encoder.* and classifier.* tensors (safetensors: no code execution on load)
    encoder_config.json   encoder architecture hyper-parameters (incl. vocab_size, pad_token_id)
    classifier_config.json classifier architecture (hidden_dim)
    vocab.json            token -> id
    label_maps.json       version / mdc / base / split label -> index
    inference_config.json decision threshold, request limits, runtime defaults
    manifest.json         SHA-256 + size of every file above, tokenization spec, provenance

Security: weights are never unpickled; every file is hash-verified against the manifest at load
time and loading fails closed on any mismatch, missing file, or malformed content.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drg_audit import __version__
from drg_audit.errors import ArtifactError
from drg_audit.features import CLS, MASK, PAD, SEP, SPECIAL_TOKENS, UNK
from drg_audit.schemas import LabelMaps, TokenizationSpec

if TYPE_CHECKING:  # pragma: no cover
    import torch

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "model.safetensors"
ENCODER_CONFIG_NAME = "encoder_config.json"
CLASSIFIER_CONFIG_NAME = "classifier_config.json"
VOCAB_NAME = "vocab.json"
LABELS_NAME = "label_maps.json"
INFERENCE_CONFIG_NAME = "inference_config.json"
ENCODER_WEIGHTS_NAME = "encoder.safetensors"
METRICS_NAME = "metrics.json"

BUNDLE_FILES = (WEIGHTS_NAME, ENCODER_CONFIG_NAME, CLASSIFIER_CONFIG_NAME, VOCAB_NAME, LABELS_NAME,
                INFERENCE_CONFIG_NAME)
ENCODER_CHECKPOINT_FILES = (ENCODER_WEIGHTS_NAME, ENCODER_CONFIG_NAME, VOCAB_NAME)
_MAX_JSON_BYTES = 64 * 1024 * 1024
_TRACKED_LIBRARIES = ("torch", "transformers", "safetensors", "numpy")


@dataclass(frozen=True)
class ModelBundle:
    root: Path
    manifest: dict[str, Any]
    encoder_config: dict[str, Any]
    classifier_config: dict[str, Any]
    vocab: dict[str, int]
    labels: LabelMaps
    inference_config: dict[str, Any]
    tokenization: TokenizationSpec

    @property
    def model_version(self) -> str:
        return str(self.manifest["bundle_id"])

    @property
    def weights_path(self) -> Path:
        return self.root / WEIGHTS_NAME


@dataclass(frozen=True)
class EncoderCheckpoint:
    root: Path
    manifest: dict[str, Any]
    encoder_config: dict[str, Any]
    vocab: dict[str, int]
    tokenization: TokenizationSpec

    @property
    def weights_path(self) -> Path:
        return self.root / ENCODER_WEIGHTS_NAME


# --------------------------------------------------------------------------------------
# hashing and atomic writes
# --------------------------------------------------------------------------------------
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _json_writer(data: Any, compact: bool = False) -> Callable[[Path], None]:
    def write(path: Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            if compact:
                json.dump(data, handle, sort_keys=True, separators=(",", ":"))
            else:
                json.dump(data, handle, sort_keys=True, indent=2)

    return write


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def tensors_writer(tensors: Mapping[str, torch.Tensor]) -> Callable[[Path], None]:
    """Writer that stores tensors with safetensors (imported lazily)."""

    def write(path: Path) -> None:
        from safetensors.torch import save_file

        save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(path),
                  metadata={"format": "pt"})

    return write


# --------------------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------------------
def _write_artifact_dir(
    output_dir: str | Path,
    writers: Mapping[str, Callable[[Path], None]],
    *,
    kind: str,
    tokenization: TokenizationSpec,
    extra_manifest: Mapping[str, Any] | None = None,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for name, writer in writers.items():
        target = root / name
        _atomic_write(target, writer)
        files[name] = {"sha256": sha256_file(target), "bytes": target.stat().st_size}
    fingerprint = hashlib.sha256("".join(f"{n}:{files[n]['sha256']}" for n in sorted(files)).encode()).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "bundle_id": fingerprint[:16],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "library_versions": _library_versions(),
        "tokenization": asdict(tokenization),
        "files": files,
        **(dict(extra_manifest) if extra_manifest else {}),
    }
    _atomic_write(root / MANIFEST_NAME, _json_writer(manifest))
    return root


def write_bundle(
    output_dir: str | Path,
    *,
    write_weights: Callable[[Path], None],
    encoder_config: Mapping[str, Any],
    classifier_config: Mapping[str, Any],
    vocab: Mapping[str, int],
    labels: LabelMaps,
    inference_config: Mapping[str, Any],
    tokenization: TokenizationSpec,
    training_info: Mapping[str, Any] | None = None,
) -> Path:
    writers = {
        WEIGHTS_NAME: write_weights,
        ENCODER_CONFIG_NAME: _json_writer(dict(encoder_config)),
        CLASSIFIER_CONFIG_NAME: _json_writer(dict(classifier_config)),
        VOCAB_NAME: _json_writer(dict(vocab), compact=True),
        LABELS_NAME: _json_writer(labels.to_dict()),
        INFERENCE_CONFIG_NAME: _json_writer(dict(inference_config)),
    }
    extra = {"training": dict(training_info)} if training_info else None
    return _write_artifact_dir(output_dir, writers, kind="serving_bundle", tokenization=tokenization,
                               extra_manifest=extra)


def write_encoder_checkpoint(
    output_dir: str | Path,
    *,
    write_weights: Callable[[Path], None],
    encoder_config: Mapping[str, Any],
    vocab: Mapping[str, int],
    tokenization: TokenizationSpec,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    writers: dict[str, Callable[[Path], None]] = {
        ENCODER_WEIGHTS_NAME: write_weights,
        ENCODER_CONFIG_NAME: _json_writer(dict(encoder_config)),
        VOCAB_NAME: _json_writer(dict(vocab), compact=True),
    }
    if metrics:
        writers[METRICS_NAME] = _json_writer(dict(metrics))
    return _write_artifact_dir(output_dir, writers, kind="encoder_checkpoint", tokenization=tokenization)


# --------------------------------------------------------------------------------------
# reading and verification
# --------------------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            raise ArtifactError(f"{path.name} exceeds the maximum accepted size")
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise ArtifactError(f"required artifact file is missing: {path.name}") from None
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"artifact file is unreadable or not valid JSON: {path.name}") from exc


def _verified_manifest(root: Path, required: tuple[str, ...], kind: str, verify: bool) -> dict[str, Any]:
    manifest = _read_json(root / MANIFEST_NAME)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError("unsupported or malformed manifest.json")
    if manifest.get("kind") != kind:
        raise ArtifactError(f"expected an artifact of kind '{kind}'")
    files = manifest.get("files")
    if not isinstance(files, dict) or not set(required) <= set(files):
        raise ArtifactError("manifest does not list all required artifact files")
    if verify:
        for name, info in files.items():
            if Path(name).name != name:
                raise ArtifactError("manifest contains an unsafe file path")
            path = root / name
            if not path.is_file():
                raise ArtifactError(f"artifact file listed in manifest is missing: {name}")
            if not hmac.compare_digest(sha256_file(path), str(info.get("sha256", ""))):
                raise ArtifactError(f"integrity check failed for {name}")
    return manifest


def _validated_vocab(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict) or not raw:
        raise ArtifactError("vocab.json must be a non-empty object")
    vocab = {str(k): int(v) for k, v in raw.items()}
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ArtifactError("vocab ids must be unique and contiguous")
    for idx, token in enumerate(SPECIAL_TOKENS):
        if vocab.get(token) != idx:
            raise ArtifactError("special tokens must occupy ids 0-4 in canonical order")
    return vocab


def validate_vocab(raw: Any) -> dict[str, int]:
    """Public wrapper used by converters: contiguous ids, special tokens at ids 0-4."""
    return _validated_vocab(raw)


def _tokenization(manifest: Mapping[str, Any]) -> TokenizationSpec:
    try:
        spec = TokenizationSpec(**manifest["tokenization"])
        spec.validate()
        return spec
    except (KeyError, TypeError) as exc:
        raise ArtifactError("manifest tokenization spec is missing or malformed") from exc


def load_bundle(root: str | Path, *, verify: bool = True) -> ModelBundle:
    path = Path(root)
    manifest = _verified_manifest(path, BUNDLE_FILES, "serving_bundle", verify)
    vocab = _validated_vocab(_read_json(path / VOCAB_NAME))
    encoder_config = _read_json(path / ENCODER_CONFIG_NAME)
    if not isinstance(encoder_config, dict) or encoder_config.get("vocab_size") != len(vocab):
        raise ArtifactError("encoder_config.vocab_size does not match vocab.json")
    classifier_config = _read_json(path / CLASSIFIER_CONFIG_NAME)
    hidden_dim = classifier_config.get("hidden_dim") if isinstance(classifier_config, dict) else None
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
        raise ArtifactError("classifier_config.hidden_dim must be a positive integer")
    labels = LabelMaps.from_mapping(_read_json(path / LABELS_NAME))
    inference_config = _read_json(path / INFERENCE_CONFIG_NAME)
    if not isinstance(inference_config, dict):
        raise ArtifactError("inference_config.json must be an object")
    return ModelBundle(path, manifest, encoder_config, classifier_config, vocab, labels, inference_config,
                       _tokenization(manifest))


def load_encoder_checkpoint(root: str | Path, *, verify: bool = True) -> EncoderCheckpoint:
    path = Path(root)
    manifest = _verified_manifest(path, ENCODER_CHECKPOINT_FILES, "encoder_checkpoint", verify)
    vocab = _validated_vocab(_read_json(path / VOCAB_NAME))
    encoder_config = _read_json(path / ENCODER_CONFIG_NAME)
    if not isinstance(encoder_config, dict) or encoder_config.get("vocab_size") != len(vocab):
        raise ArtifactError("encoder_config.vocab_size does not match vocab.json")
    return EncoderCheckpoint(path, manifest, encoder_config, vocab, _tokenization(manifest))


def load_tensors(path: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load a safetensors file (imported lazily)."""
    from safetensors.torch import load_file

    return load_file(str(path), device=device)


__all__ = [
    "BUNDLE_FILES", "CLS", "MASK", "PAD", "SEP", "UNK", "ModelBundle", "EncoderCheckpoint",
    "load_bundle", "load_encoder_checkpoint", "load_tensors", "sha256_file", "tensors_writer",
    "validate_vocab", "write_bundle", "write_encoder_checkpoint",
]
