"""Fail-closed Accelerate/DeepSpeed checkpoint publication and auditing.

A resumable state directory has two layers:

* the payload written collectively by Accelerate/DeepSpeed plus one explicit
  receipt and RNG file per global rank; and
* ``state_manifest.json`` and ``COMPLETE``, written by rank zero only after
  every payload file is present and durable.

The caller must build the payload under a private sibling directory, call
``write_checkpoint_completion_metadata`` there, and finally publish it with
``publish_completed_checkpoint``. Readers accept only the atomically renamed
``step_NNNNNN`` directory and re-hash the complete payload before returning any
trainer state.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


STATE_DIRECTORY_MANIFEST_SCHEMA_VERSION = "fastwam_accelerate_state_manifest_v1"
CHECKPOINT_PAYLOAD_MANIFEST_SCHEMA_VERSION = (
    "fastwam_accelerate_checkpoint_payload_manifest_v1"
)
CHECKPOINT_COMPLETE_SCHEMA_VERSION = "fastwam_accelerate_checkpoint_complete_v1"
CHECKPOINT_RANK_RECEIPT_SCHEMA_VERSION = (
    "fastwam_accelerate_checkpoint_rank_receipt_v1"
)
CHECKPOINT_PAYLOAD_MANIFEST_FILENAME = "state_manifest.json"
CHECKPOINT_COMPLETE_FILENAME = "COMPLETE"
CHECKPOINT_RANK_RECEIPT_DIRNAME = "rank_receipts"
_METADATA_PATHS = frozenset(
    {
        CHECKPOINT_PAYLOAD_MANIFEST_FILENAME,
        CHECKPOINT_COMPLETE_FILENAME,
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_file(
    path: Path,
    *,
    relative_path: str,
    capture: bool = False,
) -> tuple[dict[str, Any], bytes | None]:
    before_lstat = path.lstat()
    if stat.S_ISLNK(before_lstat.st_mode) or not stat.S_ISREG(
        before_lstat.st_mode
    ):
        raise ValueError(
            f"state inventory entry is not a regular file: {relative_path}"
        )
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] | None = [] if capture else None
    digest = hashlib.sha256()
    size = 0
    try:
        before_fstat = os.fstat(descriptor)
        if _file_identity(before_lstat) != _file_identity(before_fstat):
            raise RuntimeError(
                f"state file changed before inventory read: {relative_path}"
            )
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after_fstat = os.fstat(descriptor)
        if (
            _file_identity(after_fstat) != _file_identity(before_fstat)
            or size != before_fstat.st_size
        ):
            raise RuntimeError(
                f"state file changed during inventory read: {relative_path}"
            )
    finally:
        os.close(descriptor)
    return (
        {
            "path": relative_path,
            "size": size,
            "sha256": digest.hexdigest(),
        },
        b"".join(chunks) if chunks is not None else None,
    )


def _inventory_directory(
    root: Path,
    *,
    excluded_paths: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], bytes]:
    excluded = frozenset(excluded_paths)
    root_lstat = root.lstat()
    if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
        raise ValueError("state root must be a real directory, not a symlink")

    entries: list[dict[str, Any]] = []
    trainer_state_bytes: bytes | None = None
    for current_root, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_root)
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            candidate = current / directory_name
            candidate_lstat = candidate.lstat()
            if stat.S_ISLNK(candidate_lstat.st_mode) or not stat.S_ISDIR(
                candidate_lstat.st_mode
            ):
                relative = candidate.relative_to(root).as_posix()
                raise ValueError(
                    "state inventory directory is not a real directory: "
                    f"{relative}"
                )
        for file_name in file_names:
            candidate = current / file_name
            relative = candidate.relative_to(root).as_posix()
            if relative in excluded:
                continue
            entry, captured = _read_regular_file(
                candidate,
                relative_path=relative,
                capture=relative == "trainer_state.json",
            )
            entries.append(entry)
            if captured is not None:
                trainer_state_bytes = captured

    entries.sort(key=lambda item: item["path"])
    if trainer_state_bytes is None:
        raise FileNotFoundError(
            f"strict state inventory requires {root / 'trainer_state.json'}"
        )
    return entries, trainer_state_bytes


def _compact_manifest(entries: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_hasher = hashlib.sha256()
    manifest_hasher.update(
        f"{STATE_DIRECTORY_MANIFEST_SCHEMA_VERSION}\0".encode("ascii")
    )
    for entry in entries:
        encoded = _canonical_json(entry).encode("ascii")
        manifest_hasher.update(len(encoded).to_bytes(8, "big"))
        manifest_hasher.update(encoded)
    trainer_entries = [
        item for item in entries if item["path"] == "trainer_state.json"
    ]
    if len(trainer_entries) != 1:
        raise ValueError(
            "state inventory must contain exactly one trainer_state.json"
        )
    return {
        "schema_version": STATE_DIRECTORY_MANIFEST_SCHEMA_VERSION,
        "file_count": len(entries),
        "total_size_bytes": sum(item["size"] for item in entries),
        "manifest_sha256": manifest_hasher.hexdigest(),
        "trainer_state_sha256": trainer_entries[0]["sha256"],
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> bytes:
    encoded = (_canonical_json(dict(payload)) + "\n").encode("ascii")
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
    return encoded


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} root must be an object")
    return payload


def _read_regular_bytes(path: Path, *, relative_path: str) -> bytes:
    _, raw = _read_regular_file(
        path,
        relative_path=relative_path,
        capture=True,
    )
    if raw is None:
        raise AssertionError(f"failed to capture regular file {relative_path}")
    return raw


def expected_rank_receipt_paths(world_size: int) -> list[str]:
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("checkpoint world_size must be a positive integer")
    return [
        f"{CHECKPOINT_RANK_RECEIPT_DIRNAME}/rank_{rank:05d}.json"
        for rank in range(world_size)
    ]


def expected_rng_state_paths(world_size: int) -> list[str]:
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("checkpoint world_size must be a positive integer")
    return [f"random_states_{rank}.pkl" for rank in range(world_size)]


def _validate_rank_receipt(
    receipt: Mapping[str, Any],
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
    if set(receipt) != expected_keys:
        raise ValueError("checkpoint rank receipt has an invalid structure")
    expected_values = {
        "schema_version": CHECKPOINT_RANK_RECEIPT_SCHEMA_VERSION,
        "rank": expected_rank,
        "world_size": expected_world_size,
        "global_step": expected_global_step,
        "scheduler_last_epoch": expected_global_step,
    }
    for key, expected in expected_values.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f"checkpoint rank receipt {key} disagrees with publication: "
                f"{receipt.get(key)!r} != {expected!r}"
            )
    engine_step = receipt.get("deepspeed_global_steps")
    if deepspeed_required:
        if engine_step != expected_global_step:
            raise ValueError(
                "checkpoint rank receipt DeepSpeed global_steps disagrees "
                "with global_step"
            )
    elif engine_step is not None:
        raise ValueError(
            "non-DeepSpeed checkpoint rank receipt must record null engine step"
        )
    return dict(receipt)


def write_rank_checkpoint_receipt(
    state_dir: Path,
    receipt: Mapping[str, Any],
) -> Path:
    """Atomically publish one rank's cursor receipt inside a pending state."""

    rank = receipt.get("rank")
    world_size = receipt.get("world_size")
    global_step = receipt.get("global_step")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (rank, global_step)
    ):
        raise ValueError("checkpoint rank/global_step must be non-negative integers")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or rank >= world_size
    ):
        raise ValueError("checkpoint rank receipt world_size/rank is invalid")
    deepspeed_required = receipt.get("deepspeed_global_steps") is not None
    normalized = _validate_rank_receipt(
        receipt,
        expected_rank=rank,
        expected_world_size=world_size,
        expected_global_step=global_step,
        deepspeed_required=deepspeed_required,
    )
    receipt_dir = state_dir / CHECKPOINT_RANK_RECEIPT_DIRNAME
    receipt_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
    receipt_lstat = receipt_dir.lstat()
    if stat.S_ISLNK(receipt_lstat.st_mode) or not stat.S_ISDIR(receipt_lstat.st_mode):
        raise ValueError("checkpoint rank receipt root must be a real directory")
    path = receipt_dir / f"rank_{rank:05d}.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"checkpoint rank receipt already exists: {path}")
    _atomic_write_json(path, normalized)
    return path


def _validate_payload_inventory(
    root: Path,
    entries: list[dict[str, Any]],
    *,
    global_step: int,
    world_size: int,
    deepspeed_required: bool,
) -> None:
    by_path = {entry["path"]: entry for entry in entries}
    if len(by_path) != len(entries):
        raise ValueError("checkpoint payload contains duplicate relative paths")

    expected_receipts = expected_rank_receipt_paths(world_size)
    actual_receipts = sorted(
        path
        for path in by_path
        if path.startswith(f"{CHECKPOINT_RANK_RECEIPT_DIRNAME}/")
    )
    if actual_receipts != expected_receipts:
        raise ValueError(
            "checkpoint payload does not contain the exact per-rank receipt set: "
            f"expected={expected_receipts}, actual={actual_receipts}"
        )
    expected_rng = expected_rng_state_paths(world_size)
    # Lexicographic sorting puts random_states_10.pkl before
    # random_states_2.pkl.  The expected list is rank-numeric, so normalize
    # the observed names by their integer rank before comparing.  This matters
    # for the first formal 16-rank checkpoint (world_size >= 10).
    actual_rng_entries: list[tuple[int, str]] = []
    for path in by_path:
        if (
            "/" not in path
            and path.startswith("random_states_")
            and path.endswith(".pkl")
        ):
            raw_rank = path[len("random_states_") : -len(".pkl")]
            if not raw_rank.isdigit():
                raise ValueError(
                    "checkpoint payload contains an invalid RNG state name: "
                    f"{path}"
                )
            actual_rng_entries.append((int(raw_rank), path))
    actual_rng = [
        path for _, path in sorted(actual_rng_entries, key=lambda item: item[0])
    ]
    if actual_rng != expected_rng:
        raise ValueError(
            "checkpoint payload does not contain the exact per-rank RNG set: "
            f"expected={expected_rng}, actual={actual_rng}"
        )
    for relative in (*expected_receipts, *expected_rng, "trainer_state.json"):
        if relative not in by_path or by_path[relative]["size"] <= 0:
            raise ValueError(f"checkpoint payload file is absent or empty: {relative}")

    for rank, relative in enumerate(expected_receipts):
        raw = _read_regular_bytes(root / relative, relative_path=relative)
        receipt = _strict_json_bytes(raw, label=relative)
        _validate_rank_receipt(
            receipt,
            expected_rank=rank,
            expected_world_size=world_size,
            expected_global_step=global_step,
            deepspeed_required=deepspeed_required,
        )

    if not deepspeed_required:
        return

    required_common = (
        "latest",
        "scheduler.bin",
        "pytorch_model/mp_rank_00_model_states.pt",
    )
    for relative in required_common:
        if relative not in by_path or by_path[relative]["size"] <= 0:
            raise ValueError(
                f"DeepSpeed checkpoint payload is missing required file: {relative}"
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
        present = [path for path in candidates if path in by_path]
        if len(present) != 1 or by_path[present[0]]["size"] <= 0:
            raise ValueError(
                "DeepSpeed checkpoint payload must contain exactly one optimizer "
                f"shard for rank {rank}: candidates={list(candidates)}"
            )
        expected_optimizer_paths.append(present[0])
    actual_optimizer_paths = sorted(
        path for path in by_path if path.endswith("_optim_states.pt")
    )
    if actual_optimizer_paths != sorted(expected_optimizer_paths):
        raise ValueError(
            "DeepSpeed checkpoint optimizer shard set is not exact: "
            f"expected={sorted(expected_optimizer_paths)}, "
            f"actual={actual_optimizer_paths}"
        )


def _fsync_payload(root: Path, entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        path = root / entry["path"]
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = {root}
    for entry in entries:
        current = (root / entry["path"]).parent
        while current != root:
            directories.add(current)
            current = current.parent
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def write_checkpoint_completion_metadata(
    state_dir: Path,
    *,
    global_step: int,
    world_size: int,
    deepspeed_required: bool,
) -> dict[str, Any]:
    """Write the hash manifest and the COMPLETE marker, in that order."""

    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError("checkpoint global_step must be a non-negative integer")
    expected_rank_receipt_paths(world_size)
    if not isinstance(deepspeed_required, bool):
        raise TypeError("deepspeed_required must be boolean")
    for relative in _METADATA_PATHS:
        candidate = state_dir / relative
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError(
                f"checkpoint completion metadata already exists: {candidate}"
            )

    entries, _ = _inventory_directory(
        state_dir,
        excluded_paths=_METADATA_PATHS,
    )
    _validate_payload_inventory(
        state_dir,
        entries,
        global_step=global_step,
        world_size=world_size,
        deepspeed_required=deepspeed_required,
    )
    _fsync_payload(state_dir, entries)
    stable_entries, _ = _inventory_directory(
        state_dir,
        excluded_paths=_METADATA_PATHS,
    )
    if stable_entries != entries:
        raise RuntimeError("checkpoint payload changed while being made durable")

    payload_manifest = {
        "schema_version": CHECKPOINT_PAYLOAD_MANIFEST_SCHEMA_VERSION,
        "global_step": global_step,
        "world_size": world_size,
        "deepspeed_required": deepspeed_required,
        "expected_rank_receipts": expected_rank_receipt_paths(world_size),
        "expected_rng_state_files": expected_rng_state_paths(world_size),
        "payload_inventory": {
            **_compact_manifest(entries),
            "files": entries,
        },
    }
    manifest_bytes = _atomic_write_json(
        state_dir / CHECKPOINT_PAYLOAD_MANIFEST_FILENAME,
        payload_manifest,
    )
    complete = {
        "schema_version": CHECKPOINT_COMPLETE_SCHEMA_VERSION,
        "global_step": global_step,
        "world_size": world_size,
        "manifest_path": CHECKPOINT_PAYLOAD_MANIFEST_FILENAME,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    _atomic_write_json(state_dir / CHECKPOINT_COMPLETE_FILENAME, complete)
    return {"payload_manifest": payload_manifest, "complete": complete}


def publish_completed_checkpoint(pending_dir: Path, final_dir: Path) -> None:
    """Atomically rename a completed private directory into its final name."""

    if pending_dir.parent != final_dir.parent:
        raise ValueError("pending and final checkpoint directories must be siblings")
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(f"final checkpoint state already exists: {final_dir}")
    if not (pending_dir / CHECKPOINT_COMPLETE_FILENAME).is_file():
        raise FileNotFoundError("pending checkpoint lacks COMPLETE marker")
    os.rename(pending_dir, final_dir)
    _fsync_directory(final_dir.parent)


def audit_state_directory(state_dir: Path) -> dict[str, Any]:
    """Verify a complete, atomically published state and return exact evidence."""

    requested = state_dir.expanduser()
    if requested.is_symlink():
        raise ValueError(
            f"state root must be a real directory, not a symlink: {requested}"
        )
    root = requested.resolve(strict=True)
    root_lstat = root.lstat()
    if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
        raise ValueError(f"state root must be a real directory: {root}")
    root_identity = _file_identity(root_lstat)

    complete_path = root / CHECKPOINT_COMPLETE_FILENAME
    manifest_path = root / CHECKPOINT_PAYLOAD_MANIFEST_FILENAME
    if not complete_path.exists():
        raise FileNotFoundError(
            f"strict resume requires completion marker {complete_path}"
        )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"strict resume requires payload manifest {manifest_path}"
        )
    complete_bytes = _read_regular_bytes(
        complete_path,
        relative_path=CHECKPOINT_COMPLETE_FILENAME,
    )
    manifest_bytes = _read_regular_bytes(
        manifest_path,
        relative_path=CHECKPOINT_PAYLOAD_MANIFEST_FILENAME,
    )
    complete = _strict_json_bytes(
        complete_bytes,
        label=CHECKPOINT_COMPLETE_FILENAME,
    )
    payload_manifest = _strict_json_bytes(
        manifest_bytes,
        label=CHECKPOINT_PAYLOAD_MANIFEST_FILENAME,
    )
    if set(complete) != {
        "schema_version",
        "global_step",
        "world_size",
        "manifest_path",
        "manifest_sha256",
    }:
        raise ValueError("checkpoint COMPLETE marker has an invalid structure")
    if complete.get("schema_version") != CHECKPOINT_COMPLETE_SCHEMA_VERSION:
        raise ValueError("checkpoint COMPLETE marker has an unsupported schema")
    if complete.get("manifest_path") != CHECKPOINT_PAYLOAD_MANIFEST_FILENAME:
        raise ValueError("checkpoint COMPLETE marker selects an invalid manifest")
    if complete.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("checkpoint payload manifest SHA does not match COMPLETE")

    expected_manifest_keys = {
        "schema_version",
        "global_step",
        "world_size",
        "deepspeed_required",
        "expected_rank_receipts",
        "expected_rng_state_files",
        "payload_inventory",
    }
    if set(payload_manifest) != expected_manifest_keys:
        raise ValueError("checkpoint payload manifest has an invalid structure")
    if (
        payload_manifest.get("schema_version")
        != CHECKPOINT_PAYLOAD_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("checkpoint payload manifest has an unsupported schema")
    global_step = complete.get("global_step")
    world_size = complete.get("world_size")
    deepspeed_required = payload_manifest.get("deepspeed_required")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError("checkpoint COMPLETE global_step is invalid")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise ValueError("checkpoint COMPLETE world_size is invalid")
    if not isinstance(deepspeed_required, bool):
        raise ValueError("checkpoint payload deepspeed_required is invalid")
    if payload_manifest.get("global_step") != global_step:
        raise ValueError("checkpoint manifest and COMPLETE global_step disagree")
    if payload_manifest.get("world_size") != world_size:
        raise ValueError("checkpoint manifest and COMPLETE world_size disagree")
    if root.name != f"step_{global_step:06d}":
        raise ValueError(
            "completed checkpoint directory name disagrees with global_step"
        )
    if payload_manifest.get("expected_rank_receipts") != expected_rank_receipt_paths(
        world_size
    ):
        raise ValueError("checkpoint manifest expected rank receipts drifted")
    if payload_manifest.get("expected_rng_state_files") != expected_rng_state_paths(
        world_size
    ):
        raise ValueError("checkpoint manifest expected RNG files drifted")

    payload_entries, trainer_state_bytes = _inventory_directory(
        root,
        excluded_paths=_METADATA_PATHS,
    )
    live_payload_inventory = {
        **_compact_manifest(payload_entries),
        "files": payload_entries,
    }
    if payload_manifest.get("payload_inventory") != live_payload_inventory:
        raise ValueError(
            "checkpoint payload inventory differs from the signed manifest"
        )
    _validate_payload_inventory(
        root,
        payload_entries,
        global_step=global_step,
        world_size=world_size,
        deepspeed_required=deepspeed_required,
    )

    trainer_state = _strict_json_bytes(
        trainer_state_bytes,
        label="trainer_state.json",
    )
    if trainer_state.get("global_step") != global_step:
        raise ValueError(
            "trainer_state global_step disagrees with checkpoint completion"
        )

    all_entries, _ = _inventory_directory(root)
    by_path = {entry["path"]: entry for entry in all_entries}
    if by_path[CHECKPOINT_COMPLETE_FILENAME]["sha256"] != hashlib.sha256(
        complete_bytes
    ).hexdigest():
        raise RuntimeError("checkpoint COMPLETE changed during audit")
    if by_path[CHECKPOINT_PAYLOAD_MANIFEST_FILENAME]["sha256"] != hashlib.sha256(
        manifest_bytes
    ).hexdigest():
        raise RuntimeError("checkpoint payload manifest changed during audit")
    if _file_identity(root.lstat()) != root_identity:
        raise RuntimeError(f"state root changed while being audited: {root}")

    state_manifest = _compact_manifest(all_entries)
    state_files = [
        {
            "relpath": item["path"],
            "size_bytes": item["size"],
            "sha256": item["sha256"],
        }
        for item in all_entries
    ]
    return {
        "path": str(root),
        "trainer_state": trainer_state,
        "state_manifest": state_manifest,
        "state_files": state_files,
        "state_files_manifest_sha256": hashlib.sha256(
            _canonical_json(state_files).encode("utf-8")
        ).hexdigest(),
        "checkpoint_payload_manifest": payload_manifest,
        "checkpoint_complete": complete,
    }
