"""Atomic completion marker for one canonical RoboTwin raw episode.

The HDF5 writer and JSON manifest are published independently.  Neither file
alone is therefore a safe dataset-indexing signal: a process may die after one
artifact is visible but before the remaining validation reports are durable.
This module makes ``<episode>_SUCCESS.json`` the final, create-only commit
record.  The marker binds all required artifacts by SHA-256 and is never
overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import h5py

from .raw_episode_hdf5 import RAW_EPISODE_SCHEMA_VERSION
from .production_schema_policy import require_production_raw_episode_handle
from .source_scene_identity import validate_complete_scene_identity


EPISODE_SUCCESS_SCHEMA_VERSION = "robotwin_momap_episode_success_v0"


def sha256_path(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def success_marker_path(episode_hdf5: str | Path) -> Path:
    path = Path(episode_hdf5)
    if path.suffix != ".h5":
        raise ValueError(f"canonical episode must end in .h5: {path}")
    return path.with_name(f"{path.stem}_SUCCESS.json")


def _strict_existing_file(raw_path: str | Path, *, label: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_marker_output(raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_symlink() or candidate.exists():
        raise FileExistsError(f"refusing to overwrite SUCCESS marker: {candidate}")
    return candidate.resolve()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _strict_existing_file(path, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {resolved}")
    return value


def _artifact_record(path: Path) -> dict[str, Any]:
    resolved = _strict_existing_file(path, label="SUCCESS artifact")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_path(resolved),
    }


def _capture_gate_passed(manifest: Mapping[str, Any]) -> bool:
    summary = manifest.get("capture_buffer_rgb")
    if not isinstance(summary, Mapping) or not summary:
        return False
    return all(
        isinstance(value, Mapping)
        and int(value.get("frames", 0)) > 0
        and value.get("all_bitwise_equal") is True
        and int(value.get("max_abs_error", -1)) == 0
        for value in summary.values()
    )


def _validate_registry_report(report: Mapping[str, Any]) -> None:
    if report.get("passed") is not True:
        raise ValueError("segmentation registry report did not pass")
    status = report.get("status")
    if not isinstance(status, Mapping):
        raise ValueError("segmentation registry report lacks status object")
    required = {
        "data_internal_consistency": "PASS",
        "live_sapien_channel_semantics": "PASS",
        "live_link_level_semantics": "PASS",
        "production_readiness": "PASS",
    }
    mismatches = {
        key: status.get(key)
        for key, expected in required.items()
        if status.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "segmentation registry report has incomplete live gates: "
            f"{mismatches}"
        )


def _write_new_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create ``path`` atomically without ever replacing an old marker."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite SUCCESS marker: {path}")
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, default=str
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link creation is an atomic create-if-absent operation.  Unlike
        # os.replace(), it cannot clobber a previously successful publication.
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_episode_success(
    episode_hdf5: str | Path,
    manifest_path: str | Path,
    registry_report_path: str | Path,
    *,
    marker_path: str | Path | None = None,
) -> Path:
    """Validate all production gates, then atomically publish SUCCESS.

    Validation failures occur before marker creation.  If a marker already
    exists it is never removed or replaced, even when new candidate artifacts
    fail validation.
    """

    episode = _strict_existing_file(episode_hdf5, label="canonical HDF5")
    manifest_file = _strict_existing_file(manifest_path, label="episode manifest")
    registry_file = _strict_existing_file(
        registry_report_path, label="segmentation registry report"
    )
    marker = _resolve_marker_output(
        marker_path if marker_path is not None else success_marker_path(episode)
    )

    manifest = _read_json_object(manifest_file, label="episode manifest")
    registry_report = _read_json_object(
        registry_file, label="segmentation registry report"
    )
    _validate_registry_report(registry_report)
    if manifest.get("status") != "complete":
        raise ValueError(f"manifest status is not complete: {manifest.get('status')!r}")
    if manifest.get("task_success") is not True:
        raise ValueError("manifest task_success is not true")
    if not _capture_gate_passed(manifest):
        raise ValueError("same-capture-buffer RGB gate did not pass")
    if manifest.get("source_scene_identity_exact") is not True:
        raise ValueError("manifest source_scene_identity_exact is not true")
    manifest_scene_identity = validate_complete_scene_identity(
        manifest.get("source_scene_identity")
    )

    if not episode.is_file():
        raise FileNotFoundError(f"canonical HDF5 does not exist: {episode}")
    with h5py.File(episode, "r") as handle:
        physical_schema = require_production_raw_episode_handle(
            handle,
            expected_camera_names=tuple(manifest.get("camera_names", ())),
        )
        if "metadata/episode_json" not in handle:
            raise ValueError("canonical HDF5 lacks metadata/episode_json")
        raw_metadata = handle["metadata/episode_json"][()]
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        try:
            hdf5_metadata = json.loads(str(raw_metadata))
        except json.JSONDecodeError as exc:
            raise ValueError("canonical HDF5 episode_json is invalid") from exc
        if hdf5_metadata.get("source_scene_identity_exact") is not True:
            raise ValueError("HDF5 source_scene_identity_exact is not true")
        hdf5_scene_identity = validate_complete_scene_identity(
            hdf5_metadata.get("source_scene_identity")
        )
        if hdf5_scene_identity != manifest_scene_identity:
            raise ValueError("HDF5 and manifest source scene identity differ")
        raw_schema = str(handle.attrs.get("schema_version", ""))
        if raw_schema != RAW_EPISODE_SCHEMA_VERSION:
            raise ValueError(
                "production SUCCESS requires raw_episode_v0; "
                f"candidate schema {raw_schema!r} is read-only"
            )
        hdf5_status = str(handle.attrs.get("status", ""))
        frame_count = int(handle.attrs.get("num_frames", -1))
    if hdf5_status != "complete" or frame_count <= 0:
        raise ValueError(
            f"canonical HDF5 is not complete: status={hdf5_status!r}, "
            f"num_frames={frame_count}"
        )
    expected = int(manifest.get("expected_num_frames", -1))
    captured = int(manifest.get("captured_frames", -1))
    report_frames = int(
        registry_report.get("source", {}).get("num_frames", -1)
    )
    if len({frame_count, expected, captured, report_frames}) != 1:
        raise ValueError(
            "frame counts disagree: "
            f"hdf5={frame_count}, expected={expected}, captured={captured}, "
            f"registry_report={report_frames}"
        )

    payload = {
        "schema_version": EPISODE_SUCCESS_SCHEMA_VERSION,
        "status": "SUCCESS",
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_episode_schema_version": raw_schema,
        "production_schema_policy_sha256": physical_schema["policy_sha256"],
        "manifest_schema_version": manifest.get("schema_version"),
        "registry_report_schema_version": registry_report.get(
            "report_schema_version"
        ),
        "frame_count": frame_count,
        "camera_names": list(manifest.get("camera_names", [])),
        "source_scene_identity": {
            "schema_version": manifest_scene_identity["schema_version"],
            "episode_key": manifest_scene_identity["episode_key"],
            "expected_sha256": manifest_scene_identity["expected_sha256"],
            "actual_sha256": manifest_scene_identity["actual_sha256"],
            "comparison": manifest_scene_identity["comparison"],
            "passed": True,
        },
        "gates": {
            "hdf5_complete": True,
            "manifest_complete": True,
            "task_success": True,
            "same_capture_buffer_rgb_bitwise": True,
            "segmentation_registry_internal": True,
            "live_sapien_raw_channel": True,
            "live_link_level_pose_lookup": True,
            "source_scene_identity_exact": True,
        },
        "artifacts": {
            "episode_hdf5": _artifact_record(episode),
            "manifest": _artifact_record(manifest_file),
            "segmentation_registry_report": _artifact_record(registry_file),
        },
    }
    _write_new_atomic_json(marker, payload)
    return marker


def require_episode_success(
    episode_hdf5: str | Path,
    *,
    marker_path: str | Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Load and validate the final SUCCESS record for a consumer."""

    episode = _strict_existing_file(episode_hdf5, label="canonical HDF5")
    marker = _strict_existing_file(
        marker_path if marker_path is not None else success_marker_path(episode),
        label="episode SUCCESS marker",
    )
    value = _read_json_object(marker, label="episode SUCCESS marker")
    if value.get("schema_version") != EPISODE_SUCCESS_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported SUCCESS schema: {value.get('schema_version')!r}"
        )
    if value.get("raw_episode_schema_version") != RAW_EPISODE_SCHEMA_VERSION:
        raise ValueError(
            "SUCCESS marker is not bound to production raw_episode_v0"
        )
    if value.get("status") != "SUCCESS":
        raise ValueError(f"SUCCESS marker status is {value.get('status')!r}")
    gates = value.get("gates")
    if not isinstance(gates, Mapping) or not gates or not all(
        gate is True for gate in gates.values()
    ):
        raise ValueError(f"SUCCESS marker contains failed/missing gates: {gates!r}")
    episode_record = value.get("artifacts", {}).get("episode_hdf5")
    if not isinstance(episode_record, Mapping):
        raise ValueError("SUCCESS marker lacks episode_hdf5 artifact")
    recorded_episode = _strict_existing_file(
        str(episode_record.get("path", "")), label="SUCCESS episode artifact"
    )
    if recorded_episode != episode:
        raise ValueError("SUCCESS marker points to a different canonical HDF5")
    with h5py.File(episode, "r") as handle:
        require_production_raw_episode_handle(
            handle,
            expected_camera_names=tuple(value.get("camera_names", ())),
        )
    if verify_hashes:
        for name, record in value.get("artifacts", {}).items():
            if not isinstance(record, Mapping):
                raise ValueError(f"invalid artifact record {name!r}")
            path = _strict_existing_file(
                str(record.get("path", "")), label=f"SUCCESS artifact {name}"
            )
            if int(record.get("size_bytes", -1)) != path.stat().st_size:
                raise ValueError(f"SUCCESS artifact size changed: {path}")
            if str(record.get("sha256", "")) != sha256_path(path):
                raise ValueError(f"SUCCESS artifact hash changed: {path}")
    return value


__all__ = [
    "EPISODE_SUCCESS_SCHEMA_VERSION",
    "publish_episode_success",
    "require_episode_success",
    "sha256_path",
    "success_marker_path",
]
