"""Immutable mmap cache for the compact RoboTwin V1 runtime index.

The cache stores actual compact address arrays, not only an index receipt.  A
single CPU producer publishes it atomically; distributed training ranks then
reconstruct the same V1SampleIndex from read-only NumPy memory maps without
opening every episode HDF5 during startup.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import secrets
import shutil
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from mwam.utils.atomic_directory_publish import publish_directory_noreplace

from .v1_index import (
    V1_INDEX_AUDIT_KEYS,
    V1_INDEX_AUDIT_SCHEMA_VERSION,
    CompactSampleRefSequence,
    V1EpisodeRecord,
    V1IndexConfig,
    V1SampleIndex,
    _CompactEpisodeWindows,
    _compact_episode_index_binding,
    _index_config_payload,
    _runtime_index_binding_from_parts,
    _sha256_json,
    _valid_sha256,
    _validated_index_audit,
    project_v1_sample_index_future_contract,
)


V1_RUNTIME_INDEX_CACHE_SCHEMA_VERSION = (
    "robotwin_momap_v1_runtime_index_cache_v1"
)
V1_RUNTIME_INDEX_CACHE_SERIALIZATION = (
    "field_dtype_npy_pools_and_episode_slices_v1"
)
V1_RUNTIME_INDEX_CACHE_PLANS_SCHEMA_VERSION = (
    "robotwin_momap_v1_runtime_index_cache_plans_v1"
)
V1_RUNTIME_INDEX_CACHE_COMPLETE_SCHEMA_VERSION = (
    "robotwin_momap_v1_runtime_index_cache_complete_v1"
)
CACHE_MANIFEST_NAME = "manifest.json"
CACHE_PLANS_NAME = "plans.json"
CACHE_COMPLETE_NAME = "COMPLETE"

_ARRAY_FIELDS = (
    "source_indices",
    "target_indices",
    "action_indices",
    "action_query_time_base_sec",
    "dense_action_query_times_sec",
)
_UNSIGNED_FIELDS = frozenset(
    {"source_indices", "target_indices", "action_indices"}
)
_FLOAT_FIELDS = frozenset(
    {"action_query_time_base_sec", "dense_action_query_times_sec"}
)
_MANIFEST_DIGEST_PREFIX = b"robotwin-momap-v1-runtime-index-cache-v1"
_COMPLETE_DIGEST_PREFIX = b"robotwin-momap-v1-runtime-index-cache-complete-v1"
_EPISODE_FIELDS = tuple(field.name for field in dataclasses.fields(V1EpisodeRecord))
_PATH_EPISODE_FIELDS = frozenset(
    {"episode_path", "source_manifest", "candidate_report"}
)


class V1RuntimeIndexCacheError(ValueError):
    """An immutable runtime-index cache failed strict validation."""


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V1RuntimeIndexCacheError(
                f"duplicate JSON object key: {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise V1RuntimeIndexCacheError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise V1RuntimeIndexCacheError(
            f"value is not canonical JSON: {exc}"
        ) from exc


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V1RuntimeIndexCacheError(
            f"{label} is not strict JSON: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise V1RuntimeIndexCacheError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(value: Mapping[str, Any]) -> str:
    payload = {
        key: item for key, item in value.items() if key != "manifest_sha256"
    }
    return _sha256_json(payload, prefix=_MANIFEST_DIGEST_PREFIX)


def _complete_digest(value: Mapping[str, Any]) -> str:
    payload = {
        key: item for key, item in value.items() if key != "complete_sha256"
    }
    return _sha256_json(payload, prefix=_COMPLETE_DIGEST_PREFIX)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (_canonical_json(value) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _safe_cache_path(raw_path: str | Path, *, must_exist: bool) -> Path:
    supplied = Path(raw_path).expanduser()
    if supplied.is_symlink():
        raise V1RuntimeIndexCacheError(
            f"runtime-index cache must not be a symlink: {supplied}"
        )
    parent = supplied.parent.resolve()
    resolved = parent / supplied.name
    if must_exist:
        try:
            metadata = resolved.lstat()
        except FileNotFoundError as exc:
            raise V1RuntimeIndexCacheError(
                f"runtime-index cache does not exist: {resolved}"
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise V1RuntimeIndexCacheError(
                f"runtime-index cache is not a real directory: {resolved}"
            )
    return resolved


def _regular_external_path(raw_path: str | Path, *, label: str) -> Path:
    supplied = Path(raw_path).expanduser()
    if supplied.is_symlink():
        raise V1RuntimeIndexCacheError(
            f"{label} must not be a symlink: {supplied}"
        )
    resolved = supplied.resolve()
    try:
        metadata = resolved.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise V1RuntimeIndexCacheError(
            f"{label} is unavailable: {resolved}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise V1RuntimeIndexCacheError(
            f"{label} is not a regular file: {resolved}"
        )
    return resolved


def _regular_child(root: Path, name: str, *, label: str) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise V1RuntimeIndexCacheError(
            f"{label} has an unsafe relative path: {name!r}"
        )
    path = root / name
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise V1RuntimeIndexCacheError(
            f"{label} is missing: {path}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise V1RuntimeIndexCacheError(
            f"{label} is not a regular non-symlink file: {path}"
        )
    return path


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise V1RuntimeIndexCacheError(
            f"cache payload is not a regular file: {path}"
        )
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _validate_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _valid_sha256(value):
        raise V1RuntimeIndexCacheError(
            f"{label} must be lowercase SHA-256 hex"
        )
    return value


def _episode_payload(episode: V1EpisodeRecord) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in _EPISODE_FIELDS:
        value = getattr(episode, name)
        if isinstance(value, Path):
            value = str(value)
        result[name] = value
    return result


def _episode_from_payload(value: Any) -> V1EpisodeRecord:
    if not isinstance(value, Mapping) or set(value) != set(_EPISODE_FIELDS):
        raise V1RuntimeIndexCacheError(
            "cached V1EpisodeRecord fields drifted"
        )
    kwargs = dict(value)
    for name in _PATH_EPISODE_FIELDS:
        if kwargs[name] is not None:
            if not isinstance(kwargs[name], str) or not Path(
                kwargs[name]
            ).is_absolute():
                raise V1RuntimeIndexCacheError(
                    f"cached episode {name} must be an absolute path"
                )
            kwargs[name] = Path(kwargs[name])
    try:
        episode = V1EpisodeRecord(**kwargs)
    except (TypeError, ValueError) as exc:
        raise V1RuntimeIndexCacheError(
            f"cached episode record is invalid: {exc}"
        ) from exc
    if _episode_payload(episode) != dict(value):
        raise V1RuntimeIndexCacheError(
            "cached episode record is not canonical"
        )
    return episode


def _episode_from_training_payload(value: Any) -> V1EpisodeRecord:
    """Parse only real episode identity; ignore historical provenance fields."""

    if not isinstance(value, Mapping):
        raise V1RuntimeIndexCacheError("cached episode must be a mapping")
    raw_path = value.get("episode_path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise V1RuntimeIndexCacheError(
            "cached episode_path must be an absolute path"
        )
    episode_path = _regular_external_path(
        raw_path, label="training episode HDF5"
    )
    optional_fields = (
        "episode_id",
        "task_name",
        "task_config",
        "variant",
        "scope_id",
        "candidate_ordinal",
        "seed",
        "expected_frame_count",
        "expected_episode_number",
        "expected_hdf5_sha256",
    )
    kwargs = {
        name: value[name]
        for name in optional_fields
        if name in value and value[name] is not None
    }
    if "expected_hdf5_sha256" in kwargs:
        kwargs["expected_hdf5_sha256"] = _validate_sha(
            kwargs["expected_hdf5_sha256"],
            label="cached training episode HDF5 SHA",
        )
    try:
        return V1EpisodeRecord(episode_path=episode_path, **kwargs)
    except (TypeError, ValueError) as exc:
        raise V1RuntimeIndexCacheError(
            f"cached training episode record is invalid: {exc}"
        ) from exc


def _pool_id(field: str, dtype: np.dtype[Any]) -> str:
    token = dtype.str.replace("<", "le").replace(">", "be").replace("|", "na")
    if not token or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        for character in token
    ):
        raise V1RuntimeIndexCacheError(
            f"unsupported NumPy dtype token: {dtype.str!r}"
        )
    return f"{field}.{token}.npy"


def _source_bindings_payload(
    values: Sequence[tuple[str, str, str | None]],
) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "file_sha256": file_sha256,
            "semantic_sha256": semantic_sha256,
        }
        for path, file_sha256, semantic_sha256 in values
    ]


def _source_bindings_from_payload(
    value: Any,
) -> tuple[tuple[str, str, str | None], ...]:
    if not isinstance(value, list):
        raise V1RuntimeIndexCacheError(
            "source_manifest_bindings must be a list"
        )
    result: list[tuple[str, str, str | None]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "file_sha256",
            "semantic_sha256",
        }:
            raise V1RuntimeIndexCacheError(
                "source manifest binding fields drifted"
            )
        path = item["path"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise V1RuntimeIndexCacheError(
                "source manifest binding path must be absolute"
            )
        file_sha = _validate_sha(
            item["file_sha256"], label="source manifest file SHA"
        )
        semantic = item["semantic_sha256"]
        if semantic is not None:
            semantic = _validate_sha(
                semantic, label="source manifest semantic SHA"
            )
        result.append((path, file_sha, semantic))
    if len(result) != len(set(result)):
        raise V1RuntimeIndexCacheError(
            "source manifest bindings contain duplicates"
        )
    return tuple(result)


def _validate_file_descriptor(
    value: Any,
    *,
    label: str,
    pool: bool,
) -> dict[str, Any]:
    required = (
        {"path", "size_bytes", "sha256", "dtype", "shape"}
        if pool
        else {"path", "size_bytes", "sha256"}
    )
    if not isinstance(value, Mapping) or set(value) != required:
        raise V1RuntimeIndexCacheError(
            f"{label} descriptor fields drifted"
        )
    path = value["path"]
    if not isinstance(path, str) or Path(path).name != path:
        raise V1RuntimeIndexCacheError(f"{label} path is unsafe")
    size = value["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise V1RuntimeIndexCacheError(f"{label} size is invalid")
    _validate_sha(value["sha256"], label=f"{label} SHA")
    if pool:
        try:
            dtype = np.dtype(value["dtype"])
        except (TypeError, ValueError) as exc:
            raise V1RuntimeIndexCacheError(
                f"{label} dtype is invalid"
            ) from exc
        shape = value["shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item <= 0
                for item in shape
            )
        ):
            raise V1RuntimeIndexCacheError(
                f"{label} shape is invalid"
            )
        if dtype.hasobject:
            raise V1RuntimeIndexCacheError(
                f"{label} object dtype is forbidden"
            )
    return dict(value)


def _validate_manifest(
    value: Any,
    *,
    verify_digests: bool = True,
) -> dict[str, Any]:
    if not isinstance(verify_digests, bool):
        raise TypeError("verify_digests must be bool")
    required = {
        "schema_version",
        "status",
        "passed",
        "serialization",
        "formal_split",
        "language_manifest",
        "build_contract",
        "index",
        "plans_file",
        "pool_files",
        "manifest_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise V1RuntimeIndexCacheError(
            "runtime-index cache manifest fields drifted"
        )
    if (
        value["schema_version"] != V1_RUNTIME_INDEX_CACHE_SCHEMA_VERSION
        or value["serialization"] != V1_RUNTIME_INDEX_CACHE_SERIALIZATION
        or (
            verify_digests
            and (value["status"] != "PASS" or value["passed"] is not True)
        )
    ):
        raise V1RuntimeIndexCacheError(
            "runtime-index cache manifest policy mismatch"
        )
    if verify_digests:
        claimed = _validate_sha(
            value["manifest_sha256"], label="cache manifest semantic SHA"
        )
        if claimed != _manifest_digest(value):
            raise V1RuntimeIndexCacheError(
                "runtime-index cache manifest semantic SHA mismatch"
            )
    formal = value["formal_split"]
    if not isinstance(formal, Mapping) or set(formal) != {
        "path",
        "file_sha256",
        "semantic_sha256",
        "split_name",
    }:
        raise V1RuntimeIndexCacheError(
            "cache Formal split binding fields drifted"
        )
    language = value["language_manifest"]
    if not isinstance(language, Mapping) or set(language) != {
        "path",
        "file_sha256",
    }:
        raise V1RuntimeIndexCacheError(
            "cache language binding fields drifted"
        )
    contract = value["build_contract"]
    if not isinstance(contract, Mapping) or set(contract) != {
        "require_raw_v1",
        "verify_manifest_hashes",
        "verify_hdf5_hashes",
        "verify_source_artifact_hashes",
    } or any(not isinstance(item, bool) for item in contract.values()):
        raise V1RuntimeIndexCacheError(
            "cache build contract fields drifted"
        )
    index = value["index"]
    if not isinstance(index, Mapping) or set(index) != {
        "index_config",
        "index_config_sha256",
        "index_sha256",
        "sample_count",
        "episode_count",
        "index_audit",
        "source_manifest_bindings",
    }:
        raise V1RuntimeIndexCacheError(
            "cache index identity fields drifted"
        )
    _validate_file_descriptor(
        value["plans_file"], label="plans file", pool=False
    )
    pools = value["pool_files"]
    if not isinstance(pools, Mapping) or not pools:
        raise V1RuntimeIndexCacheError(
            "cache pool_files must be a non-empty mapping"
        )
    for pool_id, descriptor in pools.items():
        if not isinstance(pool_id, str) or descriptor.get("path") != pool_id:
            raise V1RuntimeIndexCacheError(
                "cache pool ID/path mismatch"
            )
        _validate_file_descriptor(
            descriptor, label=f"pool {pool_id}", pool=True
        )
    return dict(value)


def _validate_training_file_descriptor(
    value: Any,
    *,
    label: str,
    pool: bool,
) -> dict[str, Any]:
    """Keep only real file-layout fields; ignore publication digests."""

    if not isinstance(value, Mapping):
        raise V1RuntimeIndexCacheError(f"{label} descriptor must be a mapping")
    path = value.get("path")
    if not isinstance(path, str) or Path(path).name != path:
        raise V1RuntimeIndexCacheError(f"{label} path is unsafe")
    size = value.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise V1RuntimeIndexCacheError(f"{label} size is invalid")
    result: dict[str, Any] = {"path": path, "size_bytes": size}
    if pool:
        try:
            dtype = np.dtype(value.get("dtype"))
        except (TypeError, ValueError) as exc:
            raise V1RuntimeIndexCacheError(
                f"{label} dtype is invalid"
            ) from exc
        shape = value.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item <= 0
                for item in shape
            )
            or dtype.hasobject
        ):
            raise V1RuntimeIndexCacheError(
                f"{label} shape/dtype is invalid"
            )
        result["dtype"] = dtype.str
        result["shape"] = list(shape)
    return result


def _load_training_manifest(cache_dir: Path) -> dict[str, Any]:
    """Parse only the mmap layout and sampling fields training consumes."""

    manifest_path = _regular_child(
        cache_dir, CACHE_MANIFEST_NAME, label="cache manifest"
    )
    value = _read_json(manifest_path, label="cache manifest")
    if (
        value.get("schema_version") != V1_RUNTIME_INDEX_CACHE_SCHEMA_VERSION
        or value.get("serialization")
        != V1_RUNTIME_INDEX_CACHE_SERIALIZATION
    ):
        raise V1RuntimeIndexCacheError(
            "runtime-index cache schema/serialization mismatch"
        )

    raw_index = value.get("index")
    if not isinstance(raw_index, Mapping):
        raise V1RuntimeIndexCacheError("cache index layout is missing")
    index_config = raw_index.get("index_config")
    if not isinstance(index_config, Mapping):
        raise V1RuntimeIndexCacheError("cache index_config is missing")
    sample_count = raw_index.get("sample_count")
    episode_count = raw_index.get("episode_count")
    index_audit = raw_index.get("index_audit")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or isinstance(episode_count, bool)
        or not isinstance(episode_count, int)
        or episode_count <= 0
        or not isinstance(index_audit, Mapping)
    ):
        raise V1RuntimeIndexCacheError(
            "cache sample/episode counts or index audit are invalid"
        )

    plans = _validate_training_file_descriptor(
        value.get("plans_file"), label="plans file", pool=False
    )
    raw_pools = value.get("pool_files")
    if not isinstance(raw_pools, Mapping) or not raw_pools:
        raise V1RuntimeIndexCacheError(
            "cache pool_files must be a non-empty mapping"
        )
    pools: dict[str, dict[str, Any]] = {}
    for pool_id, raw_descriptor in raw_pools.items():
        if not isinstance(pool_id, str):
            raise V1RuntimeIndexCacheError("cache pool ID must be text")
        descriptor = _validate_training_file_descriptor(
            raw_descriptor,
            label=f"pool {pool_id}",
            pool=True,
        )
        if descriptor["path"] != pool_id:
            raise V1RuntimeIndexCacheError("cache pool ID/path mismatch")
        pools[pool_id] = descriptor

    raw_split = value.get("formal_split")
    split_path = None
    recorded_split_name = None
    if isinstance(raw_split, Mapping):
        raw_path = raw_split.get("path")
        if isinstance(raw_path, str) and raw_path:
            split_path = raw_path
        raw_name = raw_split.get("split_name")
        if isinstance(raw_name, str) and raw_name:
            recorded_split_name = raw_name
    raw_contract = value.get("build_contract")
    recorded_raw_v1 = None
    if isinstance(raw_contract, Mapping) and isinstance(
        raw_contract.get("require_raw_v1"), bool
    ):
        recorded_raw_v1 = raw_contract["require_raw_v1"]

    return {
        "formal_split": {
            "path": split_path,
            "split_name": recorded_split_name,
        },
        "build_contract": {"require_raw_v1": recorded_raw_v1},
        "index": {
            "index_config": dict(index_config),
            "sample_count": sample_count,
            "episode_count": episode_count,
            "index_audit": dict(index_audit),
        },
        "plans_file": plans,
        "pool_files": pools,
    }


def _load_manifest(
    cache_dir: Path,
    *,
    verify_digests: bool = True,
) -> dict[str, Any]:
    if not isinstance(verify_digests, bool):
        raise TypeError("verify_digests must be bool")
    manifest_path = _regular_child(
        cache_dir, CACHE_MANIFEST_NAME, label="cache manifest"
    )
    if not verify_digests:
        del manifest_path
        return _load_training_manifest(cache_dir)
    complete_path = _regular_child(
        cache_dir, CACHE_COMPLETE_NAME, label="cache completion marker"
    )
    complete = _read_json(complete_path, label="cache completion marker")
    if set(complete) != {
        "schema_version",
        "manifest_file_sha256",
        "manifest_sha256",
        "complete_sha256",
    } or complete["schema_version"] != V1_RUNTIME_INDEX_CACHE_COMPLETE_SCHEMA_VERSION:
        raise V1RuntimeIndexCacheError(
            "cache completion marker fields drifted"
        )
    claimed_complete = _validate_sha(
        complete["complete_sha256"], label="cache COMPLETE SHA"
    )
    if verify_digests and claimed_complete != _complete_digest(complete):
        raise V1RuntimeIndexCacheError(
            "cache completion marker semantic SHA mismatch"
        )
    expected_manifest_file_sha = _validate_sha(
        complete["manifest_file_sha256"],
        label="cache manifest file SHA",
    )
    if (
        verify_digests
        and _sha256_file(manifest_path) != expected_manifest_file_sha
    ):
        raise V1RuntimeIndexCacheError(
            "cache manifest file SHA differs from COMPLETE"
        )
    manifest = _validate_manifest(
        _read_json(manifest_path, label="cache manifest"),
        verify_digests=verify_digests,
    )
    complete_manifest_sha = _validate_sha(
        complete["manifest_sha256"],
        label="cache manifest semantic SHA in COMPLETE",
    )
    if (
        verify_digests
        and manifest["manifest_sha256"] != complete_manifest_sha
    ):
        raise V1RuntimeIndexCacheError(
            "cache manifest semantic SHA differs from COMPLETE"
        )
    expected_names = {
        CACHE_MANIFEST_NAME,
        CACHE_COMPLETE_NAME,
        manifest["plans_file"]["path"],
        *(
            descriptor["path"]
            for descriptor in manifest["pool_files"].values()
        ),
    }
    observed_names = set()
    for child in cache_dir.iterdir():
        metadata = child.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise V1RuntimeIndexCacheError(
                f"cache contains a non-regular entry: {child}"
            )
        observed_names.add(child.name)
    if observed_names != expected_names:
        raise V1RuntimeIndexCacheError(
            "cache directory inventory differs from manifest: "
            f"missing={sorted(expected_names - observed_names)} "
            f"extra={sorted(observed_names - expected_names)}"
        )
    return manifest


def publish_v1_runtime_index_cache(
    index: V1SampleIndex,
    cache_dir: str | Path,
    *,
    formal_split_manifest_path: str | Path,
    formal_split_file_sha256: str,
    formal_split_semantic_sha256: str,
    split_name: str,
    language_manifest_path: str | Path,
    source_artifact_hashes_verified: bool,
    hdf5_hashes_verified: bool,
) -> dict[str, Any]:
    """Atomically publish one exact compact index as mmap-compatible pools."""

    if not isinstance(index.sample_refs, CompactSampleRefSequence):
        raise TypeError(
            "runtime-index cache requires CompactSampleRefSequence"
        )
    if index.index_sha256 is None:
        raise ValueError(
            "runtime-index cache requires a computed index_sha256"
        )
    if not isinstance(source_artifact_hashes_verified, bool) or not isinstance(
        hdf5_hashes_verified, bool
    ):
        raise TypeError("cache verification flags must be bool")
    if not isinstance(split_name, str) or not split_name.strip():
        raise ValueError("split_name must be non-empty")

    split_path = _regular_external_path(
        formal_split_manifest_path,
        label="Formal split manifest",
    )
    language_path = _regular_external_path(
        language_manifest_path,
        label="language manifest",
    )
    split_file_sha = _validate_sha(
        formal_split_file_sha256, label="Formal split file SHA"
    )
    split_semantic_sha = _validate_sha(
        formal_split_semantic_sha256,
        label="Formal split semantic SHA",
    )
    if _sha256_file(split_path) != split_file_sha:
        raise V1RuntimeIndexCacheError(
            "Formal split file differs from cache publication anchor"
        )

    final_dir = _safe_cache_path(cache_dir, must_exist=False)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists() or final_dir.is_symlink():
        existing = _load_v1_runtime_index_cache(
            final_dir,
            config=index.config,
            formal_split_manifest_path=split_path,
            formal_split_file_sha256=split_file_sha,
            formal_split_semantic_sha256=split_semantic_sha,
            split_name=split_name,
            language_manifest_path=language_path,
            require_source_artifact_hashes_verified=(
                source_artifact_hashes_verified
            ),
            require_hdf5_hashes_verified=hdf5_hashes_verified,
            verify_payload_hashes=True,
        )
        if existing.runtime_index_binding() != index.runtime_index_binding():
            raise FileExistsError(
                "existing runtime-index cache has different indexed samples: "
                f"{final_dir}"
            )
        return _load_manifest(final_dir)

    pending = final_dir.parent / (
        f".{final_dir.name}.pending.{os.getpid()}.{secrets.token_hex(12)}"
    )
    pending.mkdir(mode=0o700)
    try:
        plans = index.sample_refs.plans
        if (
            tuple(plan.episode for plan in plans) != index.episodes
            or tuple(plan.episode_ordinal for plan in plans)
            != tuple(range(len(plans)))
        ):
            raise V1RuntimeIndexCacheError(
                "cache plans do not exactly match index episodes/order"
            )
        pool_specs: dict[str, dict[str, Any]] = {}
        for plan in plans:
            for field in _ARRAY_FIELDS:
                array = getattr(plan, field)
                if array is None:
                    continue
                dtype = np.dtype(array.dtype)
                if (
                    field in _UNSIGNED_FIELDS and dtype.kind != "u"
                ) or (
                    field in _FLOAT_FIELDS and dtype != np.dtype(np.float64)
                ):
                    raise V1RuntimeIndexCacheError(
                        f"cache field has unsupported dtype: {field}={dtype.str}"
                    )
                pool_id = _pool_id(field, dtype)
                tail_shape = tuple(int(item) for item in array.shape[1:])
                spec = pool_specs.setdefault(
                    pool_id,
                    {
                        "field": field,
                        "dtype": dtype,
                        "tail_shape": tail_shape,
                        "rows": 0,
                    },
                )
                if (
                    spec["field"] != field
                    or spec["dtype"] != dtype
                    or spec["tail_shape"] != tail_shape
                ):
                    raise V1RuntimeIndexCacheError(
                        f"incompatible arrays mapped to pool {pool_id}"
                    )
                spec["rows"] += int(array.shape[0])

        pool_maps: dict[str, np.memmap] = {}
        for pool_id in sorted(pool_specs):
            spec = pool_specs[pool_id]
            shape = (int(spec["rows"]), *spec["tail_shape"])
            pool_maps[pool_id] = open_memmap(
                pending / pool_id,
                mode="w+",
                dtype=spec["dtype"],
                shape=shape,
            )

        pool_cursors = {pool_id: 0 for pool_id in pool_maps}
        plan_entries: list[dict[str, Any]] = []
        episode_bindings: list[dict[str, Any]] = []
        for plan in plans:
            arrays: dict[str, Any] = {}
            for field in _ARRAY_FIELDS:
                array = getattr(plan, field)
                if array is None:
                    arrays[field] = None
                    continue
                pool_id = _pool_id(field, np.dtype(array.dtype))
                start = pool_cursors[pool_id]
                stop = start + int(array.shape[0])
                pool_maps[pool_id][start:stop] = array
                pool_cursors[pool_id] = stop
                arrays[field] = {
                    "pool_id": pool_id,
                    "row_start": start,
                    "row_stop": stop,
                    "shape": [int(item) for item in array.shape],
                    "dtype": np.dtype(array.dtype).str,
                }
            episode_binding = _compact_episode_index_binding(plan)
            episode_bindings.append(episode_binding)
            plan_entries.append(
                {
                    "episode": _episode_payload(plan.episode),
                    "episode_ordinal": int(plan.episode_ordinal),
                    "episode_id": plan.episode_id,
                    "index_audit": dict(plan.index_audit),
                    "arrays": arrays,
                    "episode_index_sha256": episode_binding[
                        "episode_index_sha256"
                    ],
                }
            )

        for pool_id, mmap_array in pool_maps.items():
            if pool_cursors[pool_id] != int(mmap_array.shape[0]):
                raise RuntimeError(
                    f"cache pool row accounting failed: {pool_id}"
                )
            mmap_array.flush()
        pool_maps.clear()

        for pool_id in sorted(pool_specs):
            descriptor = os.open(pending / pool_id, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        plans_payload = {
            "schema_version": V1_RUNTIME_INDEX_CACHE_PLANS_SCHEMA_VERSION,
            "plans": plan_entries,
        }
        plans_path = pending / CACHE_PLANS_NAME
        _write_json(plans_path, plans_payload)

        index_audit = dict(index.index_audit)
        composed = _runtime_index_binding_from_parts(
            index_config_sha256=_sha256_json(
                _index_config_payload(index.config),
                prefix=b"robotwin-momap-v1-index-config-v1",
            ),
            episode_indices=episode_bindings,
            index_audit=index_audit,
        )
        if (
            composed["index_sha256"] != index.index_sha256
            or composed["legal_sample_count"] != len(index)
            or composed["episode_count"] != len(index.episodes)
        ):
            raise V1RuntimeIndexCacheError(
                "cache publication index identity differs from source index"
            )

        pool_files: dict[str, Any] = {}
        for pool_id in sorted(pool_specs):
            path = pending / pool_id
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
            spec = pool_specs[pool_id]
            expected_shape = (int(spec["rows"]), *spec["tail_shape"])
            if (
                not isinstance(loaded, np.memmap)
                or loaded.dtype != spec["dtype"]
                or tuple(loaded.shape) != expected_shape
            ):
                raise V1RuntimeIndexCacheError(
                    f"published cache pool cannot be mmap loaded: {pool_id}"
                )
            pool_files[pool_id] = {
                "path": pool_id,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
                "dtype": loaded.dtype.str,
                "shape": [int(item) for item in loaded.shape],
            }

        manifest: dict[str, Any] = {
            "schema_version": V1_RUNTIME_INDEX_CACHE_SCHEMA_VERSION,
            "status": "PASS",
            "passed": True,
            "serialization": V1_RUNTIME_INDEX_CACHE_SERIALIZATION,
            "formal_split": {
                "path": str(split_path),
                "file_sha256": split_file_sha,
                "semantic_sha256": split_semantic_sha,
                "split_name": split_name,
            },
            "language_manifest": {
                "path": str(language_path),
                "file_sha256": _sha256_file(language_path),
            },
            "build_contract": {
                "require_raw_v1": bool(index.config.require_raw_v1),
                "verify_manifest_hashes": bool(
                    index.config.verify_manifest_hashes
                ),
                "verify_hdf5_hashes": bool(index.config.verify_hdf5_hashes),
                "verify_source_artifact_hashes": (
                    source_artifact_hashes_verified
                ),
            },
            "index": {
                "index_config": _index_config_payload(index.config),
                "index_config_sha256": composed["index_config_sha256"],
                "index_sha256": composed["index_sha256"],
                "sample_count": len(index),
                "episode_count": len(index.episodes),
                "index_audit": index_audit,
                "source_manifest_bindings": _source_bindings_payload(
                    index.source_manifest_bindings
                ),
            },
            "plans_file": {
                "path": CACHE_PLANS_NAME,
                "size_bytes": int(plans_path.stat().st_size),
                "sha256": _sha256_file(plans_path),
            },
            "pool_files": pool_files,
            "manifest_sha256": "",
        }
        manifest["manifest_sha256"] = _manifest_digest(manifest)
        manifest_path = pending / CACHE_MANIFEST_NAME
        _write_json(manifest_path, manifest)
        complete = {
            "schema_version": V1_RUNTIME_INDEX_CACHE_COMPLETE_SCHEMA_VERSION,
            "manifest_file_sha256": _sha256_file(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "complete_sha256": "",
        }
        complete["complete_sha256"] = _complete_digest(complete)
        _write_json(pending / CACHE_COMPLETE_NAME, complete)
        for child in pending.iterdir():
            metadata = child.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise V1RuntimeIndexCacheError(
                    f"pending cache contains non-regular entry: {child}"
                )
            os.chmod(child, 0o444, follow_symlinks=False)
        os.chmod(pending, 0o755)
        _fsync_directory(pending)

        publish_directory_noreplace(pending, final_dir)
        os.chmod(final_dir, 0o555)
        _fsync_directory(final_dir.parent)
        return manifest
    except Exception:
        if pending.exists() and not pending.is_symlink():
            shutil.rmtree(pending)
            _fsync_directory(pending.parent)
        raise


def _load_v1_runtime_index_cache(
    cache_dir: str | Path,
    *,
    config: V1IndexConfig,
    split_name: str,
    formal_split_manifest_path: str | Path | None = None,
    formal_split_file_sha256: str | None = None,
    formal_split_semantic_sha256: str | None = None,
    language_manifest_path: str | Path | None = None,
    require_source_artifact_hashes_verified: bool = False,
    require_hdf5_hashes_verified: bool = False,
    verify_payload_hashes: bool = False,
    verification_mode: str = "strict",
) -> V1SampleIndex:
    """Reconstruct one V1SampleIndex from immutable read-only mmap pools.

    ``strict`` preserves the verifier-bound digest contract. ``lightweight``
    keeps path, schema, shape, dtype, count and sampling-contract checks but
    deliberately does not recompute asset digests during training startup.
    """

    if verification_mode not in {"strict", "lightweight"}:
        raise ValueError(
            "verification_mode must be 'strict' or 'lightweight'"
        )
    strict = verification_mode == "strict"

    for name, value in (
        (
            "require_source_artifact_hashes_verified",
            require_source_artifact_hashes_verified,
        ),
        ("require_hdf5_hashes_verified", require_hdf5_hashes_verified),
        ("verify_payload_hashes", verify_payload_hashes),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool")

    root = _safe_cache_path(cache_dir, must_exist=True)
    manifest = _load_manifest(root, verify_digests=strict)
    formal = manifest["formal_split"]
    split_path = None
    if formal_split_manifest_path is not None:
        split_path = _regular_external_path(
            formal_split_manifest_path,
            label="training split manifest",
        )
    elif strict:
        raise V1RuntimeIndexCacheError(
            "offline verification requires the source split manifest"
        )
    if (
        (
            formal.get("split_name") is not None
            and formal["split_name"] != split_name
        )
        or (
            split_path is not None
            and formal.get("path") is not None
            and formal["path"] != str(split_path)
        )
    ):
        raise V1RuntimeIndexCacheError(
            "runtime-index cache split path/name differs"
        )
    if strict:
        assert split_path is not None
        split_file_sha = _validate_sha(
            formal_split_file_sha256,
            label="expected Formal split file SHA",
        )
        split_semantic_sha = _validate_sha(
            formal_split_semantic_sha256,
            label="expected Formal split semantic SHA",
        )
        if (
            formal["file_sha256"] != split_file_sha
            or formal["semantic_sha256"] != split_semantic_sha
            or _sha256_file(split_path) != split_file_sha
        ):
            raise V1RuntimeIndexCacheError(
                "runtime-index cache Formal split digest binding differs"
            )
    if strict:
        if language_manifest_path is None:
            raise V1RuntimeIndexCacheError(
                "offline verification requires the source language manifest"
            )
        language_path = _regular_external_path(
            language_manifest_path,
            label="language manifest",
        )
        language = manifest["language_manifest"]
        if language["path"] != str(language_path):
            raise V1RuntimeIndexCacheError(
                "runtime-index cache language manifest path differs"
            )
        if _sha256_file(language_path) != language["file_sha256"]:
            raise V1RuntimeIndexCacheError(
                "runtime-index cache language manifest digest binding differs"
            )

    contract = manifest["build_contract"]
    recorded_raw_v1 = contract.get("require_raw_v1")
    if (
        (
            recorded_raw_v1 is not None
            and recorded_raw_v1 != bool(config.require_raw_v1)
        )
        or (
            strict
            and (
                contract["verify_manifest_hashes"]
                != bool(config.verify_manifest_hashes)
                or contract["verify_hdf5_hashes"]
                != bool(config.verify_hdf5_hashes)
                or (
                    require_source_artifact_hashes_verified
                    and not contract["verify_source_artifact_hashes"]
                )
                or (
                    require_hdf5_hashes_verified
                    and not contract["verify_hdf5_hashes"]
                )
            )
        )
    ):
        raise V1RuntimeIndexCacheError(
            "runtime-index cache build contract differs"
        )

    index_identity = manifest["index"]
    expected_config_payload = _index_config_payload(config)
    if index_identity["index_config"] != expected_config_payload:
        raise V1RuntimeIndexCacheError(
            "runtime-index cache V1IndexConfig differs"
        )
    expected_config_sha = None
    if strict:
        expected_config_sha = _sha256_json(
            expected_config_payload,
            prefix=b"robotwin-momap-v1-index-config-v1",
        )
        if index_identity["index_config_sha256"] != expected_config_sha:
            raise V1RuntimeIndexCacheError(
                "runtime-index cache V1IndexConfig digest differs"
            )
        _validate_sha(
            index_identity["index_sha256"], label="cached runtime index SHA"
        )
    sample_count = index_identity["sample_count"]
    episode_count = index_identity["episode_count"]
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or isinstance(episode_count, bool)
        or not isinstance(episode_count, int)
        or episode_count <= 0
    ):
        raise V1RuntimeIndexCacheError(
            "cached sample/episode counts are invalid"
        )
    audit = _validated_index_audit(
        {
            "schema_version": "robotwin_momap_v1_index_build_audit_v1",
            "available": True,
            "counts": index_identity["index_audit"],
        }
    )["counts"]
    source_bindings = (
        _source_bindings_from_payload(index_identity["source_manifest_bindings"])
        if strict
        else ()
    )

    plans_descriptor = (
        _validate_file_descriptor(
            manifest["plans_file"], label="plans file", pool=False
        )
        if strict
        else manifest["plans_file"]
    )
    plans_path = _regular_child(
        root, plans_descriptor["path"], label="cache plans file"
    )
    if plans_path.stat().st_size != plans_descriptor["size_bytes"]:
        raise V1RuntimeIndexCacheError(
            "cache plans file size mismatch"
        )
    if strict and _sha256_file(plans_path) != plans_descriptor["sha256"]:
        raise V1RuntimeIndexCacheError(
            "cache plans file SHA mismatch"
        )
    plans_payload = _read_json(plans_path, label="cache plans file")
    if set(plans_payload) != {"schema_version", "plans"} or (
        plans_payload["schema_version"]
        != V1_RUNTIME_INDEX_CACHE_PLANS_SCHEMA_VERSION
    ):
        raise V1RuntimeIndexCacheError(
            "cache plans payload fields drifted"
        )
    plan_entries = plans_payload["plans"]
    if not isinstance(plan_entries, list) or len(plan_entries) != episode_count:
        raise V1RuntimeIndexCacheError(
            "cache plan count differs from manifest"
        )
    pools: dict[str, np.memmap[Any, Any]] = {}
    pool_identities: dict[str, tuple[int, int, int, int]] = {}
    for pool_id, raw_descriptor in manifest["pool_files"].items():
        descriptor = (
            _validate_file_descriptor(
                raw_descriptor, label=f"pool {pool_id}", pool=True
            )
            if strict
            else raw_descriptor
        )
        path = _regular_child(
            root, descriptor["path"], label=f"cache pool {pool_id}"
        )
        identity = _file_identity(path)
        if identity[2] != descriptor["size_bytes"]:
            raise V1RuntimeIndexCacheError(
                f"cache pool size mismatch: {pool_id}"
            )
        if strict and verify_payload_hashes and _sha256_file(path) != descriptor["sha256"]:
            raise V1RuntimeIndexCacheError(
                f"cache pool SHA mismatch: {pool_id}"
            )
        try:
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise V1RuntimeIndexCacheError(
                f"cache pool cannot be mmap loaded: {pool_id}: {exc}"
            ) from exc
        if (
            not isinstance(loaded, np.memmap)
            or loaded.dtype.str != descriptor["dtype"]
            or list(loaded.shape) != descriptor["shape"]
            or loaded.flags.writeable
        ):
            raise V1RuntimeIndexCacheError(
                f"cache pool dtype/shape/mmap contract differs: {pool_id}"
            )
        pools[pool_id] = loaded
        pool_identities[pool_id] = identity

    pool_cursors = {pool_id: 0 for pool_id in pools}
    plans: list[_CompactEpisodeWindows] = []
    episode_bindings: list[dict[str, Any]] = []
    for expected_ordinal, raw_entry in enumerate(plan_entries):
        required_plan_fields = {
            "episode",
            "episode_ordinal",
            "episode_id",
            "index_audit",
            "arrays",
        }
        if (
            not isinstance(raw_entry, Mapping)
            or not required_plan_fields.issubset(raw_entry)
            or (
                strict
                and set(raw_entry)
                != required_plan_fields | {"episode_index_sha256"}
            )
        ):
            raise V1RuntimeIndexCacheError(
                "cache plan entry fields drifted"
            )
        if raw_entry["episode_ordinal"] != expected_ordinal:
            raise V1RuntimeIndexCacheError(
                "cache episode ordinals are not contiguous from zero"
            )
        episode = (
            _episode_from_payload(raw_entry["episode"])
            if strict
            else _episode_from_training_payload(raw_entry["episode"])
        )
        episode_id = raw_entry["episode_id"]
        if not isinstance(episode_id, str) or not episode_id:
            raise V1RuntimeIndexCacheError(
                "cache episode_id must be non-empty"
            )
        plan_audit = _validated_index_audit(
            {
                "schema_version": "robotwin_momap_v1_index_build_audit_v1",
                "available": True,
                "counts": raw_entry["index_audit"],
            }
        )["counts"]
        count = plan_audit["included_windows"]
        arrays_payload = raw_entry["arrays"]
        if not isinstance(arrays_payload, Mapping) or set(
            arrays_payload
        ) != set(_ARRAY_FIELDS):
            raise V1RuntimeIndexCacheError(
                "cache plan array descriptor fields drifted"
            )

        resolved_arrays: dict[str, np.ndarray | None] = {}
        for field in _ARRAY_FIELDS:
            descriptor = arrays_payload[field]
            if descriptor is None:
                if field != "dense_action_query_times_sec":
                    raise V1RuntimeIndexCacheError(
                        f"cache plan is missing required array {field}"
                    )
                resolved_arrays[field] = None
                continue
            if not isinstance(descriptor, Mapping) or set(descriptor) != {
                "pool_id",
                "row_start",
                "row_stop",
                "shape",
                "dtype",
            }:
                raise V1RuntimeIndexCacheError(
                    f"cache plan {field} descriptor fields drifted"
                )
            pool_id = descriptor["pool_id"]
            if pool_id not in pools:
                raise V1RuntimeIndexCacheError(
                    f"cache plan references unknown pool: {pool_id}"
                )
            start = descriptor["row_start"]
            stop = descriptor["row_stop"]
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(stop, bool)
                or not isinstance(stop, int)
                or start != pool_cursors[pool_id]
                or stop != start + count
            ):
                raise V1RuntimeIndexCacheError(
                    f"cache plan row range is invalid for {field}"
                )
            shape = descriptor["shape"]
            if (
                not isinstance(shape, list)
                or not shape
                or shape[0] != count
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item <= 0
                    for item in shape
                )
            ):
                raise V1RuntimeIndexCacheError(
                    f"cache plan shape is invalid for {field}"
                )
            pool = pools[pool_id]
            view = pool[start:stop]
            if (
                list(view.shape) != shape
                or view.dtype.str != descriptor["dtype"]
                or view.flags.writeable
            ):
                raise V1RuntimeIndexCacheError(
                    f"cache plan mmap view differs for {field}"
                )
            if field in _UNSIGNED_FIELDS and (
                view.dtype.kind != "u" or view.dtype.itemsize not in {2, 4, 8}
            ):
                raise V1RuntimeIndexCacheError(
                    f"cache plan {field} must use uint16/32/64"
                )
            if field in _FLOAT_FIELDS and view.dtype != np.dtype(np.float64):
                raise V1RuntimeIndexCacheError(
                    f"cache plan {field} must use float64"
                )
            pool_cursors[pool_id] = stop
            resolved_arrays[field] = view

        try:
            plan = _CompactEpisodeWindows(
                episode=episode,
                episode_ordinal=expected_ordinal,
                episode_id=episode_id,
                source_indices=resolved_arrays["source_indices"],
                target_indices=resolved_arrays["target_indices"],
                action_indices=resolved_arrays["action_indices"],
                action_query_time_base_sec=resolved_arrays[
                    "action_query_time_base_sec"
                ],
                dense_action_query_times_sec=resolved_arrays[
                    "dense_action_query_times_sec"
                ],
                index_audit=tuple(sorted(plan_audit.items())),
            )
        except (TypeError, ValueError) as exc:
            raise V1RuntimeIndexCacheError(
                f"cache compact plan is invalid: {exc}"
            ) from exc
        if strict:
            recorded_episode_sha = _validate_sha(
                raw_entry["episode_index_sha256"],
                label="cached episode index SHA",
            )
            episode_binding = _compact_episode_index_binding(plan)
            if episode_binding["episode_index_sha256"] != recorded_episode_sha:
                raise V1RuntimeIndexCacheError(
                    "cache episode index SHA differs from mmap arrays"
                )
            episode_bindings.append(episode_binding)
        plans.append(plan)

    for pool_id, pool in pools.items():
        if pool_cursors[pool_id] != int(pool.shape[0]):
            raise V1RuntimeIndexCacheError(
                f"cache pool has unreferenced or missing rows: {pool_id}"
            )
        descriptor = manifest["pool_files"][pool_id]
        path = root / descriptor["path"]
        if _file_identity(path) != pool_identities[pool_id]:
            raise V1RuntimeIndexCacheError(
                f"cache pool changed while loading: {pool_id}"
            )

    index_sha256 = None
    if strict:
        assert expected_config_sha is not None
        composed = _runtime_index_binding_from_parts(
            index_config_sha256=expected_config_sha,
            episode_indices=episode_bindings,
            index_audit=audit,
        )
        if (
            composed["index_sha256"] != index_identity["index_sha256"]
            or composed["legal_sample_count"] != sample_count
            or composed["episode_count"] != episode_count
        ):
            raise V1RuntimeIndexCacheError(
                "cache global runtime index identity differs"
            )
        index_sha256 = composed["index_sha256"]
    if (
        audit["episode_count"] != episode_count
        or audit["included_windows"] != sample_count
    ):
        raise V1RuntimeIndexCacheError(
            "cache index audit counts differ from sample/episode counts"
        )
    sequence = CompactSampleRefSequence(plans, config=config)
    if len(sequence) != sample_count:
        raise V1RuntimeIndexCacheError(
            "cache sequence length differs from manifest"
        )
    return V1SampleIndex(
        episodes=tuple(plan.episode for plan in plans),
        sample_refs=sequence,
        config=config,
        index_sha256=index_sha256,
        source_manifest_bindings=source_bindings,
        index_audit=tuple(sorted(audit.items())),
    )

def load_v1_runtime_index_cache(
    cache_dir: str | Path,
    *,
    config: V1IndexConfig,
    split_name: str,
    projected_future_offsets: Sequence[int] | None = None,
    projected_future_horizons_seconds: Sequence[float] | None = None,
) -> V1SampleIndex:
    """Load real mmap plans with path/schema/shape/dtype checks only.

    This is the only loader exposed to ordinary training. It never reads a
    completion marker, historical language bundle, or digest-qualified input,
    and it never computes file SHA256 values.
    """

    projection_requested = (
        projected_future_offsets is not None
        or projected_future_horizons_seconds is not None
    )
    if projection_requested and (
        projected_future_offsets is None
        or projected_future_horizons_seconds is None
    ):
        raise ValueError(
            "projected future offsets and horizons must be provided together"
        )
    index = _load_v1_runtime_index_cache(
        cache_dir,
        config=config,
        split_name=split_name,
        verification_mode="lightweight",
    )
    if not projection_requested:
        return index
    assert projected_future_offsets is not None
    assert projected_future_horizons_seconds is not None
    return project_v1_sample_index_future_contract(
        index,
        future_offsets=projected_future_offsets,
        future_horizons_seconds=projected_future_horizons_seconds,
        # The ordinary training loader deliberately performs only lightweight
        # cache checks.  Re-hashing every projected mmap payload here would
        # force each DDP rank to reread the full multi-million-window index.
        # Offline callers retain the public projection helper's strict default.
        compute_index_sha256=False,
    )


def load_verified_v1_runtime_index_cache(
    cache_dir: str | Path,
    **kwargs: Any,
) -> V1SampleIndex:
    """Strictly verify and return an immutable runtime-index cache.

    Unlike the training hot-path loader, this offline asset API verifies the
    completion receipt, source bindings and every serialized payload digest.
    """

    return _load_v1_runtime_index_cache(
        cache_dir,
        verify_payload_hashes=True,
        verification_mode="strict",
        **kwargs,
    )



def verify_v1_runtime_index_cache(
    cache_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fully hash and semantically verify a cache, returning a small summary."""

    index = _load_v1_runtime_index_cache(
        cache_dir,
        verify_payload_hashes=True,
        **kwargs,
    )
    return {
        "schema_version": V1_RUNTIME_INDEX_CACHE_SCHEMA_VERSION,
        "status": "PASS",
        "passed": True,
        "cache_dir": str(_safe_cache_path(cache_dir, must_exist=True)),
        "episode_count": len(index.episodes),
        "sample_count": len(index),
        "index_sha256": index.index_sha256,
        "compact_index_nbytes": index.compact_index_nbytes,
    }


__all__ = [
    "CACHE_COMPLETE_NAME",
    "CACHE_MANIFEST_NAME",
    "CACHE_PLANS_NAME",
    "V1_RUNTIME_INDEX_CACHE_SCHEMA_VERSION",
    "V1RuntimeIndexCacheError",
    "load_v1_runtime_index_cache",
    "load_verified_v1_runtime_index_cache",
    "publish_v1_runtime_index_cache",
    "verify_v1_runtime_index_cache",
]
