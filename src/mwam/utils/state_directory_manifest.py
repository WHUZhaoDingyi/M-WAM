"""Lightweight structural validation for Accelerate training checkpoints.

Checkpoint payloads are written to a private sibling directory and published
with one atomic rename. The persisted metadata describes only state required
to resume training. It deliberately contains no file digests, data identity,
prompt identity, qualification evidence, or completion receipt.

Historical state_manifest.json, COMPLETE, and rank_receipts paths are ignored
when reading an older checkpoint. Their presence, absence, or contents never
decides whether the underlying model/optimizer/scheduler/RNG state can resume.

Receipt-based publication remains available only through the explicit legacy
helpers exported at the bottom of this module. The ordinary audit path never
auto-detects or validates those governance files.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_METADATA_SCHEMA_VERSION = "fastwam_checkpoint_metadata_v1"
CHECKPOINT_METADATA_FILENAME = "checkpoint_meta.json"
CHECKPOINT_RANK_STATE_SCHEMA_VERSION = "fastwam_checkpoint_rank_state_v1"

# Historical public constants retained for offline tooling compatibility.
# The ordinary checkpoint writer and auditor do not consume these values.
STATE_DIRECTORY_MANIFEST_SCHEMA_VERSION = (
    "fastwam_accelerate_state_manifest_v1"
)
CHECKPOINT_PAYLOAD_MANIFEST_SCHEMA_VERSION = (
    "fastwam_accelerate_checkpoint_payload_manifest_v1"
)
CHECKPOINT_COMPLETE_SCHEMA_VERSION = (
    "fastwam_accelerate_checkpoint_complete_v1"
)
CHECKPOINT_RANK_RECEIPT_SCHEMA_VERSION = (
    "fastwam_accelerate_checkpoint_rank_receipt_v1"
)
CHECKPOINT_PAYLOAD_MANIFEST_FILENAME = "state_manifest.json"
CHECKPOINT_COMPLETE_FILENAME = "COMPLETE"
CHECKPOINT_RANK_RECEIPT_DIRNAME = "rank_receipts"

_IGNORED_LEGACY_FILES = frozenset(
    {"state_manifest.json", "COMPLETE"}
)
_IGNORED_LEGACY_DIRECTORIES = frozenset(
    {"rank_receipts"}
)


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    pending = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        descriptor = os.open(
            pending,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError(f"short write while publishing {path.name}")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(pending, path)
        _fsync_directory(path.parent)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass


def _regular_file_stat(path: Path, *, relative_path: str) -> os.stat_result:
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise ValueError(
            f"checkpoint state entry is not a regular file: {relative_path}"
        )
    if value.st_size <= 0:
        raise ValueError(
            f"checkpoint state file is absent or empty: {relative_path}"
        )
    return value


def _read_regular_bytes(path: Path, *, relative_path: str) -> bytes:
    before_lstat = _regular_file_stat(path, relative_path=relative_path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before_fstat = os.fstat(descriptor)
        if _file_identity(before_lstat) != _file_identity(before_fstat):
            raise RuntimeError(
                f"checkpoint state file changed before read: {relative_path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fstat = os.fstat(descriptor)
        if _file_identity(after_fstat) != _file_identity(before_fstat):
            raise RuntimeError(
                f"checkpoint state file changed during read: {relative_path}"
            )
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _read_json(path: Path, *, relative_path: str) -> dict[str, Any]:
    raw = _read_regular_bytes(path, relative_path=relative_path)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"checkpoint state file is not valid UTF-8 JSON: {relative_path}"
        ) from exc
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint JSON root must be an object: {relative_path}")
    return value


def _scan_state_files(root: Path) -> list[dict[str, Any]]:
    """Collect cheap file metadata without reading or hashing tensor assets."""

    entries: list[dict[str, Any]] = []
    for current_root, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_root)
        kept_directories: list[str] = []
        for directory_name in sorted(directory_names):
            relative = (current / directory_name).relative_to(root).as_posix()
            if relative in _IGNORED_LEGACY_DIRECTORIES:
                continue
            candidate = current / directory_name
            candidate_lstat = candidate.lstat()
            if stat.S_ISLNK(candidate_lstat.st_mode) or not stat.S_ISDIR(
                candidate_lstat.st_mode
            ):
                raise ValueError(
                    "checkpoint state directory is not a real directory: "
                    f"{relative}"
                )
            kept_directories.append(directory_name)
        directory_names[:] = kept_directories

        for file_name in sorted(file_names):
            candidate = current / file_name
            relative = candidate.relative_to(root).as_posix()
            if relative in _IGNORED_LEGACY_FILES:
                continue
            value = _regular_file_stat(candidate, relative_path=relative)
            entries.append(
                {
                    "path": relative,
                    "size": int(value.st_size),
                    "device": int(value.st_dev),
                    "inode": int(value.st_ino),
                    "mtime_ns": int(value.st_mtime_ns),
                }
            )
    entries.sort(key=lambda item: item["path"])
    return entries


def _positive_world_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("checkpoint world_size must be a positive integer")
    return int(value)


def _non_negative_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("checkpoint global_step must be a non-negative integer")
    return int(value)


def expected_rng_state_paths(world_size: int) -> list[str]:
    world_size = _positive_world_size(world_size)
    return [f"random_states_{rank}.pkl" for rank in range(world_size)]


def _observed_rng_paths(paths: set[str]) -> list[str]:
    observed: list[tuple[int, str]] = []
    for relative in paths:
        if "/" in relative or not relative.startswith("random_states_"):
            continue
        if not relative.endswith(".pkl"):
            raise ValueError(
                f"checkpoint contains an invalid RNG state filename: {relative}"
            )
        rank_text = relative[len("random_states_") : -len(".pkl")]
        if not rank_text.isdigit():
            raise ValueError(
                f"checkpoint contains an invalid RNG state filename: {relative}"
            )
        observed.append((int(rank_text), relative))
    return [path for _, path in sorted(observed)]


def _validate_checkpoint_metadata(value: object) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "global_step",
        "world_size",
        "deepspeed_required",
        "scaler_required",
        "expected_rng_state_files",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("checkpoint metadata has an invalid structure")
    if value.get("schema_version") != CHECKPOINT_METADATA_SCHEMA_VERSION:
        raise ValueError("checkpoint metadata has an unsupported schema")
    global_step = _non_negative_step(value.get("global_step"))
    world_size = _positive_world_size(value.get("world_size"))
    deepspeed_required = value.get("deepspeed_required")
    scaler_required = value.get("scaler_required")
    if not isinstance(deepspeed_required, bool):
        raise ValueError("checkpoint metadata deepspeed_required must be boolean")
    if not isinstance(scaler_required, bool):
        raise ValueError("checkpoint metadata scaler_required must be boolean")
    expected_rng = expected_rng_state_paths(world_size)
    if value.get("expected_rng_state_files") != expected_rng:
        raise ValueError("checkpoint metadata RNG state filenames drifted")
    return {
        "schema_version": CHECKPOINT_METADATA_SCHEMA_VERSION,
        "global_step": global_step,
        "world_size": world_size,
        "deepspeed_required": deepspeed_required,
        "scaler_required": scaler_required,
        "expected_rng_state_files": expected_rng,
    }


def _validate_rank_state(
    value: object,
    *,
    expected_rank: int,
    expected_world_size: int,
    expected_global_step: int,
    deepspeed_required: bool,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "rank",
        "world_size",
        "global_step",
        "scheduler_last_epoch",
        "deepspeed_global_steps",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("checkpoint rank state has an invalid structure")
    expected_values = {
        "schema_version": CHECKPOINT_RANK_STATE_SCHEMA_VERSION,
        "rank": expected_rank,
        "world_size": expected_world_size,
        "global_step": expected_global_step,
        "scheduler_last_epoch": expected_global_step,
    }
    for key, expected in expected_values.items():
        if value.get(key) != expected:
            raise ValueError(
                f"checkpoint rank state {key} disagrees with trainer state: "
                f"{value.get(key)!r} != {expected!r}"
            )
    engine_step = value.get("deepspeed_global_steps")
    if deepspeed_required:
        if engine_step != expected_global_step:
            raise ValueError(
                "checkpoint rank state DeepSpeed global_steps disagrees with "
                "global_step"
            )
    elif engine_step is not None:
        raise ValueError(
            "non-DeepSpeed checkpoint rank state must have null engine step"
        )
    return dict(value)


def validate_checkpoint_rank_state(
    value: object,
    *,
    global_step: int,
    world_size: int,
    deepspeed_required: bool,
) -> list[dict[str, Any]]:
    """Validate the rank cursor list embedded in a new trainer state."""

    global_step = _non_negative_step(global_step)
    world_size = _positive_world_size(world_size)
    if not isinstance(value, list) or len(value) != world_size:
        raise ValueError(
            "trainer_state rank_state must contain one record per rank"
        )
    return [
        _validate_rank_state(
            record,
            expected_rank=rank,
            expected_world_size=world_size,
            expected_global_step=global_step,
            deepspeed_required=deepspeed_required,
        )
        for rank, record in enumerate(value)
    ]


def _detect_deepspeed(paths: set[str]) -> bool:
    return bool(
        "latest" in paths
        or "pytorch_model/mp_rank_00_model_states.pt" in paths
        or any(path.endswith("_optim_states.pt") for path in paths)
    )


def _validate_deepspeed_payload(
    root: Path,
    *,
    paths: set[str],
    world_size: int,
) -> None:
    required = {
        "latest",
        "scheduler.bin",
        "pytorch_model/mp_rank_00_model_states.pt",
    }
    missing = sorted(required.difference(paths))
    if missing:
        raise ValueError(
            f"DeepSpeed checkpoint is missing required state files: {missing}"
        )
    latest = _read_regular_bytes(root / "latest", relative_path="latest")
    if latest.decode("ascii", errors="strict").strip() != "pytorch_model":
        raise ValueError("DeepSpeed latest tag must select pytorch_model")

    expected_optimizer_paths: list[str] = []
    for rank in range(world_size):
        candidates = (
            "pytorch_model/"
            f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
            "pytorch_model/"
            f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
        )
        present = [path for path in candidates if path in paths]
        if len(present) != 1:
            raise ValueError(
                "DeepSpeed checkpoint must contain exactly one optimizer "
                f"shard for rank {rank}: candidates={list(candidates)}"
            )
        expected_optimizer_paths.append(present[0])
    actual_optimizer_paths = sorted(
        path for path in paths if path.endswith("_optim_states.pt")
    )
    if actual_optimizer_paths != sorted(expected_optimizer_paths):
        raise ValueError(
            "DeepSpeed checkpoint optimizer shard set is not exact: "
            f"expected={sorted(expected_optimizer_paths)}, "
            f"actual={actual_optimizer_paths}"
        )


def _validate_non_deepspeed_payload(paths: set[str]) -> None:
    model_files = sorted(
        path
        for path in paths
        if "/" not in path
        and (
            (path.startswith("model") and path.endswith(".safetensors"))
            or (path.startswith("pytorch_model") and path.endswith(".bin"))
        )
    )
    if not model_files:
        raise ValueError(
            "non-DeepSpeed checkpoint is missing model state_dict files"
        )
    for required in ("optimizer.bin", "scheduler.bin"):
        if required not in paths:
            raise ValueError(
                f"non-DeepSpeed checkpoint is missing required state file: {required}"
            )


def _validate_payload(
    root: Path,
    *,
    entries: list[dict[str, Any]],
    trainer_state: Mapping[str, Any],
    checkpoint_meta: Mapping[str, Any] | None,
    require_step_directory_name: bool = True,
) -> dict[str, Any]:
    paths = {str(entry["path"]) for entry in entries}
    if len(paths) != len(entries):
        raise ValueError("checkpoint contains duplicate relative paths")
    if "trainer_state.json" not in paths:
        raise FileNotFoundError(
            f"checkpoint lacks required trainer state: {root / 'trainer_state.json'}"
        )

    global_step = _non_negative_step(trainer_state.get("global_step"))
    if (
        require_step_directory_name
        and root.name != f"step_{global_step:06d}"
    ):
        raise ValueError(
            "checkpoint directory name disagrees with trainer global_step"
        )

    observed_rng = _observed_rng_paths(paths)
    if not observed_rng:
        raise FileNotFoundError("checkpoint contains no per-rank RNG state")
    inferred_world_size = len(observed_rng)
    if observed_rng != expected_rng_state_paths(inferred_world_size):
        raise ValueError(
            "checkpoint does not contain a contiguous per-rank RNG state set"
        )

    detected_deepspeed = _detect_deepspeed(paths)
    scaler_present = "scaler.pt" in paths
    if checkpoint_meta is None:
        world_size = inferred_world_size
        deepspeed_required = detected_deepspeed
        scaler_required = scaler_present
    else:
        normalized = _validate_checkpoint_metadata(checkpoint_meta)
        world_size = normalized["world_size"]
        deepspeed_required = normalized["deepspeed_required"]
        scaler_required = normalized["scaler_required"]
        if normalized["global_step"] != global_step:
            raise ValueError(
                "checkpoint metadata global_step disagrees with trainer_state"
            )
        if normalized["expected_rng_state_files"] != observed_rng:
            raise ValueError(
                "checkpoint RNG state set disagrees with checkpoint metadata"
            )
        if detected_deepspeed != deepspeed_required:
            raise ValueError(
                "checkpoint backend files disagree with checkpoint metadata"
            )
    if inferred_world_size != world_size:
        raise ValueError(
            "checkpoint RNG state world size disagrees with checkpoint metadata"
        )
    if scaler_required and not scaler_present:
        raise ValueError("checkpoint is missing the required AMP scaler state")

    if deepspeed_required:
        _validate_deepspeed_payload(root, paths=paths, world_size=world_size)
    else:
        _validate_non_deepspeed_payload(paths)

    return {
        "global_step": global_step,
        "world_size": world_size,
        "deepspeed_required": deepspeed_required,
        "scaler_required": scaler_required,
        "expected_rng_state_files": expected_rng_state_paths(world_size),
        "format": "metadata_v1" if checkpoint_meta is not None else "legacy",
    }


def _fsync_payload(root: Path, entries: list[dict[str, Any]]) -> None:
    directories = {root}
    for entry in entries:
        relative = str(entry["path"])
        descriptor = os.open(
            root / relative,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        current = (root / relative).parent
        while current != root:
            directories.add(current)
            current = current.parent
    for directory in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def write_checkpoint_metadata(
    state_dir: Path,
    *,
    global_step: int,
    world_size: int,
    deepspeed_required: bool,
    scaler_required: bool,
) -> dict[str, Any]:
    """Validate a pending payload and write non-digest resume metadata."""

    global_step = _non_negative_step(global_step)
    world_size = _positive_world_size(world_size)
    if not isinstance(deepspeed_required, bool):
        raise TypeError("deepspeed_required must be boolean")
    if not isinstance(scaler_required, bool):
        raise TypeError("scaler_required must be boolean")
    root = state_dir.resolve(strict=True)
    metadata_path = root / CHECKPOINT_METADATA_FILENAME
    if metadata_path.exists() or metadata_path.is_symlink():
        raise FileExistsError(
            f"checkpoint metadata already exists: {metadata_path}"
        )

    trainer_state = _read_json(
        root / "trainer_state.json",
        relative_path="trainer_state.json",
    )
    entries = _scan_state_files(root)
    observed = _validate_payload(
        root,
        entries=entries,
        trainer_state=trainer_state,
        checkpoint_meta=None,
        require_step_directory_name=False,
    )
    expected = {
        "global_step": global_step,
        "world_size": world_size,
        "deepspeed_required": deepspeed_required,
        "scaler_required": scaler_required,
    }
    for key, value in expected.items():
        if observed[key] != value:
            raise ValueError(
                f"pending checkpoint {key} disagrees with runtime: "
                f"{observed[key]!r} != {value!r}"
            )

    _fsync_payload(root, entries)
    stable_entries = _scan_state_files(root)
    if stable_entries != entries:
        raise RuntimeError(
            "checkpoint payload changed while being made durable"
        )

    metadata = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA_VERSION,
        "global_step": global_step,
        "world_size": world_size,
        "deepspeed_required": deepspeed_required,
        "scaler_required": scaler_required,
        "expected_rng_state_files": expected_rng_state_paths(world_size),
    }
    _atomic_write_json(metadata_path, metadata)
    return metadata


def publish_checkpoint(pending_dir: Path, final_dir: Path) -> None:
    """Atomically publish one fully written private checkpoint directory."""

    if pending_dir.parent != final_dir.parent:
        raise ValueError("pending and final checkpoint directories must be siblings")
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(f"final checkpoint state already exists: {final_dir}")
    metadata_path = pending_dir / CHECKPOINT_METADATA_FILENAME
    _read_json(metadata_path, relative_path=CHECKPOINT_METADATA_FILENAME)
    os.rename(pending_dir, final_dir)
    _fsync_directory(final_dir.parent)


def audit_state_directory(state_dir: Path) -> dict[str, Any]:
    """Validate real resume state without hashing tensor assets."""

    requested = state_dir.expanduser()
    if requested.is_symlink():
        raise ValueError(
            f"checkpoint root must be a real directory, not a symlink: {requested}"
        )
    root = requested.resolve(strict=True)
    root_lstat = root.lstat()
    if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
        raise ValueError(f"checkpoint root must be a real directory: {root}")
    root_identity = _file_identity(root_lstat)

    trainer_state = _read_json(
        root / "trainer_state.json",
        relative_path="trainer_state.json",
    )
    metadata_path = root / CHECKPOINT_METADATA_FILENAME
    checkpoint_meta = (
        _read_json(
            metadata_path,
            relative_path=CHECKPOINT_METADATA_FILENAME,
        )
        if metadata_path.exists() or metadata_path.is_symlink()
        else None
    )
    entries = _scan_state_files(root)
    resume_state = _validate_payload(
        root,
        entries=entries,
        trainer_state=trainer_state,
        checkpoint_meta=checkpoint_meta,
    )
    if _file_identity(root.lstat()) != root_identity:
        raise RuntimeError(f"checkpoint root changed while being audited: {root}")
    return {
        "path": str(root),
        "trainer_state": trainer_state,
        "checkpoint_meta": (
            _validate_checkpoint_metadata(checkpoint_meta)
            if checkpoint_meta is not None
            else None
        ),
        "resume_state": resume_state,
        # Runtime-only snapshot used to catch concurrent mutation around
        # Accelerator.load_state. It is never written into trainer state or
        # propagated into resume lineage.
        "runtime_file_state": entries,
    }


def expected_rank_receipt_paths(world_size: int) -> list[str]:
    """Return historical receipt paths for explicit offline workflows."""

    from .legacy_state_directory_manifest import (
        expected_rank_receipt_paths as _legacy_expected_paths,
    )

    return _legacy_expected_paths(world_size)


def write_rank_checkpoint_receipt(
    state_dir: Path,
    receipt: Mapping[str, Any],
) -> Path:
    """Write one historical rank receipt for an explicit offline workflow."""

    from .legacy_state_directory_manifest import (
        write_rank_checkpoint_receipt as _legacy_write_receipt,
    )

    return _legacy_write_receipt(state_dir, receipt)


def write_checkpoint_completion_metadata(
    state_dir: Path,
    *,
    global_step: int,
    world_size: int,
    deepspeed_required: bool,
) -> dict[str, Any]:
    """Write historical digest metadata for an explicit offline workflow."""

    from .legacy_state_directory_manifest import (
        write_checkpoint_completion_metadata as _legacy_write_metadata,
    )

    return _legacy_write_metadata(
        state_dir,
        global_step=global_step,
        world_size=world_size,
        deepspeed_required=deepspeed_required,
    )


def publish_completed_checkpoint(
    pending_dir: Path,
    final_dir: Path,
) -> None:
    """Publish historical receipt-gated state in an explicit offline flow."""

    from .legacy_state_directory_manifest import (
        publish_completed_checkpoint as _legacy_publish,
    )

    _legacy_publish(pending_dir, final_dir)


def audit_legacy_state_directory(state_dir: Path) -> dict[str, Any]:
    """Hash and validate historical governance metadata for offline use only.

    Unlike :func:`audit_state_directory`, this explicit compatibility audit
    requires valid ``state_manifest.json``, ``COMPLETE``, and per-rank receipt
    files and hashes the entire checkpoint payload. Runtime resume must use
    :func:`audit_state_directory` instead.
    """

    from .legacy_state_directory_manifest import (
        audit_state_directory as _legacy_audit,
    )

    return _legacy_audit(state_dir)


__all__ = [
    "CHECKPOINT_COMPLETE_FILENAME",
    "CHECKPOINT_COMPLETE_SCHEMA_VERSION",
    "CHECKPOINT_METADATA_FILENAME",
    "CHECKPOINT_METADATA_SCHEMA_VERSION",
    "CHECKPOINT_PAYLOAD_MANIFEST_FILENAME",
    "CHECKPOINT_PAYLOAD_MANIFEST_SCHEMA_VERSION",
    "CHECKPOINT_RANK_RECEIPT_DIRNAME",
    "CHECKPOINT_RANK_RECEIPT_SCHEMA_VERSION",
    "CHECKPOINT_RANK_STATE_SCHEMA_VERSION",
    "STATE_DIRECTORY_MANIFEST_SCHEMA_VERSION",
    "audit_legacy_state_directory",
    "audit_state_directory",
    "expected_rank_receipt_paths",
    "expected_rng_state_paths",
    "publish_completed_checkpoint",
    "publish_checkpoint",
    "validate_checkpoint_rank_state",
    "write_checkpoint_completion_metadata",
    "write_checkpoint_metadata",
    "write_rank_checkpoint_receipt",
]
