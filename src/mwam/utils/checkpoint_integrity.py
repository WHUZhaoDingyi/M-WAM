"""Fail-closed helpers for immutable floating-point checkpoint assets."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

FLOATING_STATE_DICT_SCHEMA_VERSION = "fastwam_floating_state_dict_schema_v1"
TENSOR_STATE_DICT_SCHEMA_VERSION = "fastwam_tensor_state_dict_schema_v1"
CANONICAL_JSON_CONTRACT_SCHEMA_VERSION = "fastwam_canonical_json_contract_v1"


def _validate_canonical_json_value(value: Any, *, name: str) -> None:
    """Reject values whose JSON representation would be lossy or ambiguous."""

    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"`{name}` contains NaN or Inf.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"`{name}` mapping keys must be strings, "
                    f"got {type(key).__name__}."
                )
            _validate_canonical_json_value(
                item,
                name=f"{name}[{key!r}]",
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_canonical_json_value(
                item,
                name=f"{name}[{index}]",
            )
        return
    raise TypeError(
        f"`{name}` must contain only canonical JSON values, "
        f"got {type(value).__name__}."
    )


def canonical_json_contract(
    value: Any,
    *,
    domain: str,
    name: str,
) -> dict[str, Any]:
    """Freeze one JSON contract together with a domain-separated SHA256.

    Tuples are normalized to JSON arrays. Non-string mapping keys, non-finite
    floats, tensors, and other implicit encoder conversions are rejected so
    the recorded value and digest have one unambiguous cross-process meaning.
    """

    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("`domain` must be a non-empty string.")
    if "\x00" in domain:
        raise ValueError("`domain` may not contain NUL.")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("`name` must be a non-empty string.")
    _validate_canonical_json_value(value, name=name)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    normalized = json.loads(encoded.decode("utf-8"))
    digest = hashlib.sha256(
        b"fastwam-canonical-json-contract-v1\0"
        + domain.encode("utf-8")
        + b"\0"
        + encoded
    ).hexdigest()
    return {
        "schema_version": CANONICAL_JSON_CONTRACT_SCHEMA_VERSION,
        "domain": domain,
        "canonical_json_bytes": len(encoded),
        "sha256": digest,
        "contract": normalized,
    }


def checkpoint_file_identity(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Hash one regular file and reject replacement while it is being read."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint is not a regular file: {resolved}")
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = resolved.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"checkpoint changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def validate_floating_state_dict(
    state: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    """Validate that every state value is a finite real floating tensor."""

    if not isinstance(state, Mapping):
        raise TypeError(f"`{name}` must be a mapping.")
    tensor_count = 0
    numel = 0
    dtypes: set[str] = set()
    for key, value in state.items():
        if not isinstance(key, str):
            raise TypeError(f"`{name}` keys must be strings, got {type(key).__name__}.")
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"`{name}[{key}]` must be torch.Tensor, "
                f"got {type(value).__name__}."
            )
        if not torch.is_floating_point(value):
            raise TypeError(
                f"`{name}[{key}]` must have a real floating dtype, got {value.dtype}."
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"`{name}[{key}]` contains NaN or Inf.")
        tensor_count += 1
        numel += int(value.numel())
        dtypes.add(str(value.dtype).removeprefix("torch."))
    return {
        "tensor_count": tensor_count,
        "numel": numel,
        "dtypes": sorted(dtypes),
    }


def floating_state_dict_schema(
    state: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    """Hash the exact sorted key/shape/dtype schema of one finite state dict."""

    integrity = validate_floating_state_dict(state, name=name)
    entries = [
        {
            "key": key,
            "shape": list(state[key].shape),
            "dtype": str(state[key].dtype).removeprefix("torch."),
            "numel": int(state[key].numel()),
        }
        for key in sorted(state)
    ]
    encoded = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    schema_sha256 = hashlib.sha256(
        b"fastwam-floating-state-dict-schema-v1\0" + encoded
    ).hexdigest()
    return {
        "schema_version": FLOATING_STATE_DICT_SCHEMA_VERSION,
        "tensor_count": int(integrity["tensor_count"]),
        "numel": int(integrity["numel"]),
        "dtypes": list(integrity["dtypes"]),
        "schema_sha256": schema_sha256,
    }


def tensor_state_dict_schema(
    state: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    """Return explicit tensor names, shapes, and dtypes without content hashes."""

    if not isinstance(state, Mapping):
        raise TypeError(f"`{name}` must be a mapping.")
    entries: list[dict[str, Any]] = []
    for key in sorted(state):
        if not isinstance(key, str):
            raise TypeError(
                f"`{name}` keys must be strings, got {type(key).__name__}."
            )
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"`{name}[{key}]` must be torch.Tensor, "
                f"got {type(value).__name__}."
            )
        entries.append(
            {
                "name": key,
                "shape": [int(dimension) for dimension in value.shape],
                "dtype": str(value.dtype).removeprefix("torch."),
            }
        )
    if not entries:
        raise ValueError(f"`{name}` must contain at least one tensor.")
    return {
        "schema_version": TENSOR_STATE_DICT_SCHEMA_VERSION,
        "tensor_count": len(entries),
        "tensors": entries,
    }


__all__ = [
    "CANONICAL_JSON_CONTRACT_SCHEMA_VERSION",
    "FLOATING_STATE_DICT_SCHEMA_VERSION",
    "TENSOR_STATE_DICT_SCHEMA_VERSION",
    "canonical_json_contract",
    "checkpoint_file_identity",
    "floating_state_dict_schema",
    "tensor_state_dict_schema",
    "validate_floating_state_dict",
]
