"""Optional O(T) sidecar cache for low-resolution MoMap source grids.

The canonical raw episode remains the source of truth.  This cache stores only
per-frame simulator geometry/state sampled at fixed uniform cell centres; it
never stores frame-pair flow and therefore grows linearly with episode length.
Target-frame full-resolution buffers are intentionally absent, so visibility
continues to use the canonical raw HDF5 z-buffer inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .raw_episode_hdf5 import RawEpisodeReader


SOURCE_GRID_CACHE_SCHEMA_VERSION = "robotwin_momap_source_grid_cache_v1"


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("h5py is required for the source-grid cache") from exc
    return h5py


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _uniform_cell_center_indices(length: int, count: int) -> np.ndarray:
    if count <= 0 or count > length:
        raise ValueError(f"grid dimension {count} is outside [1,{length}]")
    if count == length:
        return np.arange(length, dtype=np.int64)
    return np.floor(
        (np.arange(count, dtype=np.float64) + 0.5) * length / count
    ).astype(np.int64)


def _normalize_grids(grids: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for grid in grids:
        if len(grid) != 2:
            raise ValueError(f"grid must be (H,W), got {grid!r}")
        height, width = (int(value) for value in grid)
        if height <= 0 or width <= 0:
            raise ValueError(f"grid dimensions must be positive, got {grid!r}")
        item = (height, width)
        if item not in normalized:
            normalized.append(item)
    if not normalized:
        raise ValueError("at least one grid is required")
    return tuple(normalized)


def _source_fingerprint(reader: RawEpisodeReader) -> str:
    """Fingerprint immutable episode identity without hashing all image bytes."""

    stat = reader.path.stat()
    payload = {
        "raw_schema_version": reader.schema_version,
        "geometry_encoding": reader.geometry_encoding,
        "geometry_contract_sha256": reader.geometry_contract_hash,
        "resolved_path": str(reader.path.resolve()),
        "file_size": int(stat.st_size),
        "file_mtime_ns": int(stat.st_mtime_ns),
        "num_frames": reader.num_frames,
        "camera_names": list(reader.camera_names),
        "metadata": reader.metadata,
        "entity_registry": reader.entity_registry,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _create_dataset(
    group: Any,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
    *,
    compression: str | None,
    compression_opts: int | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "shape": shape,
        "dtype": dtype,
        "chunks": (1, *shape[1:]),
    }
    if compression is not None:
        kwargs.update(
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
        )
    return group.create_dataset(name, **kwargs)


def build_source_grid_cache(
    episode: str | Path,
    output: str | Path,
    *,
    cameras: Sequence[str] | None = None,
    grids: Iterable[tuple[int, int]] = ((16, 16), (32, 32)),
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
) -> dict[str, Any]:
    """Build one atomic source-grid sidecar and return a size report."""

    h5py = _require_h5py()
    episode_path = Path(episode)
    output_path = Path(output)
    partial_path = output_path.with_name(output_path.name + ".partial")
    if output_path.exists():
        raise FileExistsError(f"source-grid cache already exists: {output_path}")
    if partial_path.exists():
        raise FileExistsError(f"partial source-grid cache exists: {partial_path}")
    normalized_grids = _normalize_grids(grids)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with RawEpisodeReader(episode_path) as reader:
        selected_cameras = (
            tuple(reader.camera_names) if cameras is None else tuple(str(v) for v in cameras)
        )
        if not selected_cameras or len(set(selected_cameras)) != len(selected_cameras):
            raise ValueError("cameras must be a non-empty unique sequence")
        unknown = set(selected_cameras) - set(reader.camera_names)
        if unknown:
            raise KeyError(f"unknown cameras: {sorted(unknown)}")
        frame_count = reader.num_frames
        fingerprint = _source_fingerprint(reader)

        try:
            with h5py.File(partial_path, "x", libver="latest") as cache:
                cache.attrs.update(
                    schema_version=SOURCE_GRID_CACHE_SCHEMA_VERSION,
                    status="partial",
                    raw_episode_path=str(episode_path.resolve()),
                    raw_episode_fingerprint=fingerprint,
                    raw_episode_size_bytes=int(episode_path.stat().st_size),
                    raw_episode_mtime_ns=int(episode_path.stat().st_mtime_ns),
                    num_frames=frame_count,
                    complexity="O(T)",
                    contains_frame_pair_flow=np.uint8(0),
                )
                timeline_group = cache.create_group("timeline")
                timestamp = _create_dataset(
                    timeline_group,
                    "timestamp_sec",
                    (frame_count,),
                    np.float64,
                    compression=None,
                    compression_opts=None,
                )
                raw_handle = reader._ensure_open()  # internal, read-only schema inspection
                entity_count = int(raw_handle["entities/pose_to_world"].shape[1])
                entities_group = cache.create_group("entities")
                pose = _create_dataset(
                    entities_group,
                    "pose_to_world",
                    (frame_count, entity_count, 4, 4),
                    np.float32,
                    compression=compression,
                    compression_opts=compression_opts,
                )
                pose_valid = _create_dataset(
                    entities_group,
                    "pose_valid",
                    (frame_count, entity_count),
                    np.uint8,
                    compression=compression,
                    compression_opts=compression_opts,
                )

                camera_datasets: dict[str, dict[str, Any]] = {}
                cameras_group = cache.create_group("cameras")
                for camera_name in selected_cameras:
                    raw_geometry = raw_handle[f"private_geometry/{camera_name}"]
                    source_height, source_width = reader.camera_image_shape(camera_name)
                    camera_group = cameras_group.create_group(camera_name)
                    camera_group.attrs.update(
                        source_height=source_height,
                        source_width=source_width,
                    )
                    camera_pose = _create_dataset(
                        camera_group,
                        "T_cam_to_world",
                        (frame_count, 4, 4),
                        np.float32,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    grids_group = camera_group.create_group("grids")
                    per_camera: dict[str, Any] = {"T_cam_to_world": camera_pose}
                    for grid_height, grid_width in normalized_grids:
                        if grid_height > source_height or grid_width > source_width:
                            raise ValueError(
                                f"grid {(grid_height, grid_width)} exceeds camera "
                                f"{camera_name} shape {(source_height, source_width)}"
                            )
                        grid_key = f"{grid_height}x{grid_width}"
                        grid_group = grids_group.create_group(grid_key)
                        rows = _uniform_cell_center_indices(source_height, grid_height)
                        columns = _uniform_cell_center_indices(source_width, grid_width)
                        grid_group.create_dataset("rows", data=rows, dtype=np.int64)
                        grid_group.create_dataset("columns", data=columns, dtype=np.int64)
                        seg_dtype = raw_geometry["actor_seg_raw"].dtype
                        per_camera[grid_key] = {
                            "rows": rows,
                            "columns": columns,
                            "position_xyz": _create_dataset(
                                grid_group,
                                "position_xyz",
                                (frame_count, grid_height, grid_width, 3),
                                np.float32,
                                compression=compression,
                                compression_opts=compression_opts,
                            ),
                            "position_valid": _create_dataset(
                                grid_group,
                                "position_valid",
                                (frame_count, grid_height, grid_width),
                                np.uint8,
                                compression=compression,
                                compression_opts=compression_opts,
                            ),
                            "actor_seg_raw": _create_dataset(
                                grid_group,
                                "actor_seg_raw",
                                (frame_count, grid_height, grid_width),
                                seg_dtype,
                                compression=compression,
                                compression_opts=compression_opts,
                            ),
                        }
                    camera_datasets[camera_name] = per_camera

                for index in range(frame_count):
                    state = reader.read_frame_fields(
                        index,
                        camera_names=(),
                        observation_fields=(),
                        private_geometry_fields=(),
                        timeline_fields=("timestamp_sec",),
                        robot_state_fields=(),
                        entity_fields=("pose_to_world", "pose_valid"),
                    )
                    timestamp[index] = state["timeline"]["timestamp_sec"]
                    pose[index] = state["entities"]["pose_to_world"]
                    pose_valid[index] = state["entities"]["pose_valid"]
                    for camera_name in selected_cameras:
                        projected = reader.read_frame_fields(
                            index,
                            camera_names=(camera_name,),
                            observation_fields=(),
                            private_geometry_fields=(
                                "position_xyz",
                                "position_valid",
                                "actor_seg_raw",
                                "T_cam_to_world",
                            ),
                            timeline_fields=(),
                            robot_state_fields=(),
                            entity_fields=(),
                        )["private_geometry"][camera_name]
                        per_camera = camera_datasets[camera_name]
                        per_camera["T_cam_to_world"][index] = projected["T_cam_to_world"]
                        for grid_height, grid_width in normalized_grids:
                            grid_key = f"{grid_height}x{grid_width}"
                            destination = per_camera[grid_key]
                            rows = destination["rows"]
                            columns = destination["columns"]
                            for field_name in (
                                "position_xyz",
                                "position_valid",
                                "actor_seg_raw",
                            ):
                                destination[field_name][index] = np.asarray(
                                    projected[field_name]
                                )[rows[:, None], columns[None, :]]

                cache.attrs["status"] = "complete"
                cache.flush()
            os.replace(partial_path, output_path)
        except BaseException:
            # Preserve a partial file as evidence; callers may inspect or remove it.
            raise

    logical_bytes = 0
    stored_bytes = 0
    with h5py.File(output_path, "r") as cache:
        def visit(_name: str, value: Any) -> None:
            nonlocal logical_bytes, stored_bytes
            if isinstance(value, h5py.Dataset):
                logical_bytes += int(value.size * value.dtype.itemsize)
                stored_bytes += int(value.id.get_storage_size())
        cache.visititems(visit)
    file_bytes = int(output_path.stat().st_size)
    return {
        "schema_version": SOURCE_GRID_CACHE_SCHEMA_VERSION,
        "episode": str(episode_path),
        "cache": str(output_path),
        "cameras": list(selected_cameras),
        "grids": [list(grid) for grid in normalized_grids],
        "frames": frame_count,
        "raw_episode_bytes": int(episode_path.stat().st_size),
        "cache_file_bytes": file_bytes,
        "cache_logical_bytes": logical_bytes,
        "cache_dataset_storage_bytes": stored_bytes,
        "cache_to_raw_ratio": file_bytes / max(1, int(episode_path.stat().st_size)),
        "contains_frame_pair_flow": False,
        "complexity": "O(T)",
    }


class SourceGridCacheReader:
    """PID-aware lazy reader for an immutable source-grid sidecar."""

    def __init__(self, path: str | Path, episode: str | Path) -> None:
        self.path = Path(path)
        self.episode = Path(episode)
        self._file: Any | None = None
        self._owner_pid: int | None = None

    def _ensure_open(self) -> Any:
        pid = os.getpid()
        if self._file is not None and self._owner_pid != pid:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        if self._file is None:
            h5py = _require_h5py()
            self._file = h5py.File(self.path, "r", swmr=True, libver="latest")
            self._owner_pid = pid
            try:
                if str(self._file.attrs.get("schema_version", "")) != SOURCE_GRID_CACHE_SCHEMA_VERSION:
                    raise ValueError(f"unsupported source-grid cache schema: {self.path}")
                if str(self._file.attrs.get("status", "")) != "complete":
                    raise ValueError(f"source-grid cache is not complete: {self.path}")
                with RawEpisodeReader(self.episode) as reader:
                    expected = _source_fingerprint(reader)
                actual = str(self._file.attrs.get("raw_episode_fingerprint", ""))
                if actual != expected:
                    raise ValueError(
                        "source-grid cache does not match the canonical raw episode; "
                        f"cache={actual}, episode={expected}"
                    )
            except BaseException:
                self._file.close()
                self._file = None
                self._owner_pid = None
                raise
        return self._file

    @property
    def is_open(self) -> bool:
        return self._file is not None

    def available_grids(self, camera_name: str) -> tuple[tuple[int, int], ...]:
        handle = self._ensure_open()
        path = f"cameras/{camera_name}/grids"
        if path not in handle:
            return ()
        return tuple(
            tuple(int(v) for v in key.split("x")) for key in handle[path].keys()
        )

    def source_image_shape(self, camera_name: str) -> tuple[int, int]:
        handle = self._ensure_open()
        group = handle[f"cameras/{camera_name}"]
        return int(group.attrs["source_height"]), int(group.attrs["source_width"])

    def has_grid(self, camera_name: str, grid_size: tuple[int, int]) -> bool:
        handle = self._ensure_open()
        key = f"{int(grid_size[0])}x{int(grid_size[1])}"
        return f"cameras/{camera_name}/grids/{key}" in handle

    def read_state(self, index: int) -> dict[str, Any]:
        handle = self._ensure_open()
        return {
            "timeline": {"timestamp_sec": handle["timeline/timestamp_sec"][index].item()},
            "entities": {
                "pose_to_world": handle["entities/pose_to_world"][index],
                "pose_valid": handle["entities/pose_valid"][index],
            },
        }

    def read_camera_pose(self, camera_name: str, index: int) -> np.ndarray:
        return self._ensure_open()[f"cameras/{camera_name}/T_cam_to_world"][index]

    def read_source_grid(
        self, camera_name: str, index: int, grid_size: tuple[int, int]
    ) -> dict[str, np.ndarray]:
        handle = self._ensure_open()
        key = f"{int(grid_size[0])}x{int(grid_size[1])}"
        group = handle[f"cameras/{camera_name}/grids/{key}"]
        return {
            "position_xyz": group["position_xyz"][index],
            "position_valid": group["position_valid"][index],
            "actor_seg_raw": group["actor_seg_raw"][index],
            "T_cam_to_world": self.read_camera_pose(camera_name, index),
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._owner_pid = None

    def __enter__(self) -> "SourceGridCacheReader":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        self.close()
        return False

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_file"] = None
        state["_owner_pid"] = None
        return state

    def __del__(self) -> None:  # pragma: no cover - best effort at shutdown.
        try:
            self.close()
        except Exception:
            pass


