"""Strict, dependency-free conversion of plain mappings (YAML / JSON) into dataclasses.

* unknown keys are rejected (typos in config must fail loudly, not be ignored)
* types are checked and lightly coerced (``"1e-5"`` -> float, because PyYAML parses ``1e-5`` as a string)
* nested dataclasses, ``Optional``, ``list`` and ``dict`` are supported
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from drg_audit.errors import ConfigError

T = TypeVar("T")
_NONE_TYPE = type(None)


def dataclass_from_mapping(cls: type[T], data: Mapping[str, Any] | None, *, path: str = "") -> T:
    label = path or cls.__name__
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{label}: expected a mapping, got {type(data).__name__}")
    fields = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - fields)
    if unknown:
        raise ConfigError(f"{label}: unknown configuration keys {unknown}")
    hints = typing.get_type_hints(cls)
    kwargs = {
        name: _coerce(hints[name], value, f"{path}.{name}" if path else name) for name, value in data.items()
    }
    try:
        obj = cls(**kwargs)
    except TypeError as exc:  # missing required fields
        raise ConfigError(f"{label}: {exc}") from exc
    validate = getattr(obj, "validate", None)
    if callable(validate):
        validate()
    return obj


def _coerce(tp: Any, value: Any, path: str) -> Any:
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if tp is Any:
        return value
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not _NONE_TYPE]
        if value is None:
            if len(non_none) != len(args):
                return None
            raise ConfigError(f"{path}: null is not allowed")
        if len(non_none) == 1:
            return _coerce(non_none[0], value, path)
        raise ConfigError(f"{path}: unsupported union type")
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return dataclass_from_mapping(tp, value, path=path)
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected a boolean")
    if tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected an integer")
    if tp is float:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected a number")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as exc:
                raise ConfigError(f"{path}: expected a number") from exc
        raise ConfigError(f"{path}: expected a number")
    if tp is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{path}: expected a string")
    if origin in (list, tuple, Sequence):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ConfigError(f"{path}: expected a list")
        item_tp = args[0] if args else Any
        items = [_coerce(item_tp, v, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping")
        val_tp = args[1] if len(args) == 2 else Any
        return {str(k): _coerce(val_tp, v, f"{path}.{k}") for k, v in value.items()}
    return value
