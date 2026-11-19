"""Streaming HDF5 storage for a canonical RoboTwin raw episode.

This module intentionally keeps simulator-private geometry separate from model
visible observations.  It contains no RoboTwin/SAPIEN imports, so readers can
be used in CPU-only data-loader workers.

The writer always targets ``<episode>.h5.partial`` first.  ``finalize()`` is
the only operation that atomically renames that file to ``<episode>.h5``.
Failed or interrupted exports therefore cannot be mistaken for complete
episodes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .se3 import require_se3

try:  # Keep importing the rest of fastwam possible in minimal environments.
    import h5py
except ImportError as exc:  # pragma: no cover - exercised through monkeypatch.
    h5py = None
    _H5PY_IMPORT_ERROR: BaseException | None = exc
else:
    _H5PY_IMPORT_ERROR = None


RAW_EPISODE_SCHEMA_VERSION = "robotwin_momap_raw_episode_v0"
RAW_EPISODE_SCHEMA_V1 = "robotwin_momap_raw_episode_v1"
# Experimental read-only schema used by the exact-position residual study.
# Keeping it outside SUPPORTED_RAW_EPISODE_SCHEMAS prevents generic structural
# consumers from mistaking readability for production authorization.
RAW_EPISODE_SCHEMA_V2_EXACT_POSITION = (
    "robotwin_momap_raw_episode_v2_exact_position_candidate"
)
SUPPORTED_RAW_EPISODE_SCHEMAS = (
    RAW_EPISODE_SCHEMA_VERSION,
    RAW_EPISODE_SCHEMA_V1,
)
READABLE_RAW_EPISODE_SCHEMAS = (
    *SUPPORTED_RAW_EPISODE_SCHEMAS,
    RAW_EPISODE_SCHEMA_V2_EXACT_POSITION,
)
RAW_EPISODE_V1_GEOMETRY_ENCODING = (
    "depth_float32_plus_explicit_ray_xy_float32_v1"
)
RAW_EPISODE_V2_EXACT_POSITION_GEOMETRY_ENCODING = (
    "depth_ray_float32_plus_position_bitxor_v2"
)
RAW_EPISODE_V2_POSITION_XY_XOR_FIELD = "position_xy_xor"
RAW_EPISODE_V2_POSITION_Z_XOR_FIELD = "position_z_xor"
CONTROL_STREAM_SCHEMA_VERSION = "robotwin_momap_control_stream_v0"
SUPPORTED_SEGMENTATION_DTYPES = (np.dtype("uint16"), np.dtype("uint32"))

TIMELINE_FIELDS = (
    "sim_step",
    "control_step",
    "render_step",
    "save_index",
    "video_frame_index",
    "action_index",
)


def _require_h5py() -> Any:
    if h5py is None:
        detail = f": {_H5PY_IMPORT_ERROR}" if _H5PY_IMPORT_ERROR else ""
        raise RuntimeError(
            "Raw RoboTwin HDF5 support requires h5py; install h5py before "
            f"creating or reading an episode{detail}"
        )
    return h5py


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def raw_episode_v1_geometry_contract() -> dict[str, Any]:
    """Canonical physical/logical contract shared by the HDF5 episode writers."""

    return {
        "schema_version": RAW_EPISODE_SCHEMA_V1,
        "geometry_encoding": RAW_EPISODE_V1_GEOMETRY_ENCODING,
        "depth": {
            "field": "private_geometry/<camera>/depth_m",
            "dtype": "float32",
            "shape": "[T,H,W]",
            "definition": (
                "-position_xyz[...,2] where source position_valid=1 and "
                "actor_seg_raw!=0; 0 otherwise"
            ),
            "unit": "meter",
        },
        "ray": {
            "field": "private_geometry/<camera>/ray_xy",
            "dtype": "float32",
            "shape": "[H,W,2]",
            "definition": (
                "[X/depth,Y/depth] from first valid nonzero-actor renderer "
                "Position"
            ),
            "episode_constant": True,
        },
        "reconstruction": "[ray_x*depth,ray_y*depth,-depth] in float32",
        "invalid_semantics": (
            "position_valid=0; actor_seg_raw=0 renderer no-hit pixels are "
            "forced invalid; reconstructed XYZ is unspecified"
        ),
        "K": "episode-static float32[3,3] copied exactly from v0 frame 0",
    }


def raw_episode_v1_geometry_contract_sha256() -> str:
    """Stable digest binding direct and converted HDF5 episodes."""

    return hashlib.sha256(
        _canonical_json(raw_episode_v1_geometry_contract()).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CameraSpec:
    """Static storage choices for one camera in an episode."""

    name: str
    height: int
    width: int
    segmentation_dtype: str | np.dtype = "uint32"
    store_rgb: bool = True
    store_mesh_segmentation: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("camera name must not be empty")
        if "/" in self.name:
            raise ValueError(f"camera name must not contain '/': {self.name!r}")
        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                f"camera {self.name!r} has invalid image shape "
                f"({self.height}, {self.width})"
            )
        dtype = np.dtype(self.segmentation_dtype)
        if dtype not in SUPPORTED_SEGMENTATION_DTYPES:
            raise ValueError(
                "segmentation_dtype must be uint16 or uint32, "
                f"got {dtype} for camera {self.name!r}"
            )
        object.__setattr__(self, "segmentation_dtype", dtype)


@dataclass(frozen=True)
class CameraFrame:
    """One capture buffer split into observable and private fields."""

    position_xyz: np.ndarray
    position_valid: np.ndarray
    actor_seg_raw: np.ndarray
    K: np.ndarray
    T_cam_to_world: np.ndarray
    rgb: np.ndarray | None = None
    mesh_seg_raw: np.ndarray | None = None


@dataclass(frozen=True)
class TimelineFrame:
    """Explicit mapping between simulator, control, render, and saved time."""

    sim_step: int
    control_step: int
    render_step: int
    save_index: int
    video_frame_index: int
    action_index: int
    timestamp_sec: float


@dataclass(frozen=True)
class RobotStateFrame:
    """Robot state and command arrays whose meanings must remain separate."""

    qpos_real: np.ndarray
    drive_target: np.ndarray
    action_target: np.ndarray
    action_target_valid: bool = True


@dataclass(frozen=True)
class EntityFrame:
    """World poses in the stable episode-level registry order."""

    pose_to_world: np.ndarray
    pose_valid: np.ndarray


def _normalise_camera_specs(
    camera_specs: Sequence[CameraSpec],
) -> tuple[CameraSpec, ...]:
    specs = tuple(camera_specs)
    if not specs:
        raise ValueError("at least one camera spec is required")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate camera names: {names}")
    return specs


def _validate_registry(
    registry: Sequence[Mapping[str, Any]], entity_count: int
) -> list[dict[str, Any]]:
    if entity_count <= 0:
        raise ValueError(f"entity_count must be positive, got {entity_count}")
    required = {
        "raw_seg_id",
        "seg_channel",
        "entity_index",
        "entity_type",
        "entity_name",
        "pose_row",
        "supported",
    }
    normalised: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    for item_index, item in enumerate(registry):
        missing = required - set(item)
        if missing:
            raise ValueError(
                f"registry entry {item_index} is missing fields: {sorted(missing)}"
            )
        entry = dict(item)
        raw_seg_id = int(entry["raw_seg_id"])
        pose_row = int(entry["pose_row"])
        entity_index = int(entry["entity_index"])
        seg_channel = str(entry["seg_channel"])
        if raw_seg_id < 0:
            raise ValueError(f"registry raw_seg_id must be non-negative: {entry}")
        if not (0 <= pose_row < entity_count):
            raise ValueError(
                f"registry pose_row={pose_row} is outside [0, {entity_count}): "
                f"{entry}"
            )
        if entity_index < 0:
            raise ValueError(f"registry entity_index must be non-negative: {entry}")
        key = (seg_channel, raw_seg_id)
        if key in seen_keys:
            raise ValueError(f"duplicate registry segmentation key: {key}")
        seen_keys.add(key)
        entry.update(
            raw_seg_id=raw_seg_id,
            pose_row=pose_row,
            entity_index=entity_index,
            seg_channel=seg_channel,
            entity_type=str(entry["entity_type"]),
            entity_name=str(entry["entity_name"]),
            supported=bool(entry["supported"]),
        )
        if "pose_key" in entry:
            entry["pose_key"] = str(entry["pose_key"])
        normalised.append(entry)
    return normalised


def _as_exact_shape(
    value: Any,
    shape: tuple[int, ...],
    *,
    name: str,
    dtype: np.dtype | str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} shape={array.shape}, expected={shape}")
    return array.astype(dtype, copy=False)


def _as_segmentation(
    value: Any,
    shape: tuple[int, int],
    *,
    name: str,
    dtype: np.dtype,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} shape={array.shape}, expected={shape}")
    if array.dtype.kind not in "iub":
        raise TypeError(f"{name} must contain integer IDs, got {array.dtype}")
    if array.size:
        minimum = int(array.min())
        maximum = int(array.max())
        limit = int(np.iinfo(dtype).max)
        if minimum < 0 or maximum > limit:
            raise OverflowError(
                f"{name} ID range [{minimum}, {maximum}] does not fit {dtype}; "
                "select uint32 instead of silently truncating IDs"
            )
    return array.astype(dtype, copy=False)


def _as_binary_mask(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} shape={array.shape}, expected={shape}")
    if array.size and not np.logical_or(array == 0, array == 1).all():
        raise ValueError(f"{name} must contain only 0/1 values")
    return array.astype(np.uint8, copy=False)


def _create_stream_dataset(
    group: Any,
    name: str,
    row_shape: tuple[int, ...],
    dtype: np.dtype | str,
    *,
    compression: str | None,
    compression_opts: int | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "shape": (0, *row_shape),
        "maxshape": (None, *row_shape),
        "chunks": (1, *row_shape),
        "dtype": dtype,
    }
    if compression is not None:
        kwargs.update(
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
        )
    return group.create_dataset(name, **kwargs)


class RawEpisodeWriter:
    """Append-only writer for one canonical raw episode.

    Parameters are fixed at construction so an episode cannot silently change
    image shape, segmentation width, entity ordering, or robot dimensions.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        camera_specs: Sequence[CameraSpec],
        entity_registry: Sequence[Mapping[str, Any]],
        entity_count: int,
        qpos_dim: int,
        drive_target_dim: int,
        action_dim: int,
        metadata: Mapping[str, Any] | None = None,
        compression: str | None = "gzip",
        compression_opts: int | None = 1,
        geometry_encoding: str = "position_xyz_float32_v0",
        max_reconstruction_error_m: float = 1.0e-6,
    ) -> None:
        h5 = _require_h5py()
        self.final_path = Path(path)
        if self.final_path.name.endswith(".partial"):
            raise ValueError("final episode path must not end with .partial")
        self.partial_path = self.final_path.with_name(self.final_path.name + ".partial")
        if self.final_path.exists():
            raise FileExistsError(f"final episode already exists: {self.final_path}")
        if self.partial_path.exists():
            raise FileExistsError(
                f"Partial episode already exists: {self.partial_path}"
            )

        self.camera_specs = _normalise_camera_specs(camera_specs)
        self._camera_by_name = {spec.name: spec for spec in self.camera_specs}
        self.geometry_encoding = str(geometry_encoding)
        if self.geometry_encoding not in {
            "position_xyz_float32_v0",
            RAW_EPISODE_V1_GEOMETRY_ENCODING,
        }:
            raise ValueError(
                "geometry_encoding must be position_xyz_float32_v0 or "
                f"{RAW_EPISODE_V1_GEOMETRY_ENCODING!r}, got "
                f"{self.geometry_encoding!r}"
            )
        self.max_reconstruction_error_m = float(max_reconstruction_error_m)
        if (
            not np.isfinite(self.max_reconstruction_error_m)
            or self.max_reconstruction_error_m <= 0
        ):
            raise ValueError("max_reconstruction_error_m must be finite and positive")
        self._v1_ray_xy = {
            spec.name: np.zeros((spec.height, spec.width, 2), dtype=np.float32)
            for spec in self.camera_specs
        }
        self._v1_ray_valid = {
            spec.name: np.zeros((spec.height, spec.width), dtype=bool)
            for spec in self.camera_specs
        }
        self._v1_K: dict[str, np.ndarray | None] = {
            spec.name: None for spec in self.camera_specs
        }
        self._v1_validated_vector_count = {
            spec.name: 0 for spec in self.camera_specs
        }
        self._v1_max_xyz_error_m = {
            spec.name: 0.0 for spec in self.camera_specs
        }
        self.entity_count = int(entity_count)
        self.entity_registry = _validate_registry(entity_registry, self.entity_count)
        dimensions = {
            "qpos_real": int(qpos_dim),
            "drive_target": int(drive_target_dim),
            "action_target": int(action_dim),
        }
        for name, dimension in dimensions.items():
            if dimension <= 0:
                raise ValueError(f"{name} dimension must be positive, got {dimension}")
        self._robot_dimensions = dimensions
        self._closed = False
        self._finalized = False

        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5.File(self.partial_path, "x", libver="latest")
        try:
            self._initialise_file(
                metadata=dict(metadata or {}),
                compression=compression,
                compression_opts=compression_opts,
            )
        except BaseException:
            self._file.close()
            raise

    @property
    def num_frames(self) -> int:
        self._ensure_open()
        return int(self._file["timeline/save_index"].shape[0])

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("raw episode writer is closed")

    def _initialise_file(
        self,
        *,
        metadata: Mapping[str, Any],
        compression: str | None,
        compression_opts: int | None,
    ) -> None:
        root = self._file
        schema_version = (
            RAW_EPISODE_SCHEMA_V1
            if self.geometry_encoding == RAW_EPISODE_V1_GEOMETRY_ENCODING
            else RAW_EPISODE_SCHEMA_VERSION
        )
        root.attrs.update(
            schema_version=schema_version,
            geometry_encoding=self.geometry_encoding,
            status="partial",
            num_frames=0,
            simulator_private_fields_model_visible=np.uint8(0),
        )
        if schema_version == RAW_EPISODE_SCHEMA_V1:
            root.attrs.update(
                streaming_writer="RawEpisodeWriter.direct_v1",
                geometry_contract_sha256=(
                    raw_episode_v1_geometry_contract_sha256()
                ),
                candidate_only=np.uint8(1),
                production_authorized=np.uint8(0),
                formal_index_eligible=np.uint8(0),
                v1_reconstruction_error_limit_m=np.float64(
                    self.max_reconstruction_error_m
                ),
            )
        metadata_group = root.create_group("metadata")
        metadata_group.attrs["role"] = "provenance_and_schema"
        string_dtype = _require_h5py().string_dtype(encoding="utf-8")
        metadata_group.create_dataset(
            "episode_json", data=_canonical_json(metadata), dtype=string_dtype
        )

        observations = root.create_group("observations")
        observations.attrs.update(
            role="model_visible_observations",
            model_visible=np.uint8(1),
        )
        private = root.create_group("private_geometry")
        private.attrs.update(
            role="private_simulator_geometry_for_target_construction",
            model_visible=np.uint8(0),
            oracle_only=np.uint8(1),
        )

        for spec in self.camera_specs:
            obs_camera = observations.create_group(spec.name)
            private_camera = private.create_group(spec.name)
            obs_camera.attrs.update(
                height=spec.height,
                width=spec.width,
                role="model_visible_observation",
            )
            private_camera.attrs.update(
                height=spec.height,
                width=spec.width,
                role="private_simulator_geometry",
                segmentation_dtype=str(spec.segmentation_dtype),
            )
            if spec.store_rgb:
                rgb = _create_stream_dataset(
                    obs_camera,
                    "rgb",
                    (spec.height, spec.width, 3),
                    np.uint8,
                    compression=compression,
                    compression_opts=compression_opts,
                )
                rgb.attrs.update(role="model_visible_observation", color_order="RGB")

            if self.geometry_encoding == RAW_EPISODE_V1_GEOMETRY_ENCODING:
                fields = (
                    ("depth_m", (spec.height, spec.width), np.float32),
                    ("position_valid", (spec.height, spec.width), np.uint8),
                    ("actor_seg_raw", (spec.height, spec.width), spec.segmentation_dtype),
                    ("T_cam_to_world", (4, 4), np.float32),
                )
            else:
                fields = (
                    ("position_xyz", (spec.height, spec.width, 3), np.float32),
                    ("position_valid", (spec.height, spec.width), np.uint8),
                    ("actor_seg_raw", (spec.height, spec.width), spec.segmentation_dtype),
                    ("K", (3, 3), np.float32),
                    ("T_cam_to_world", (4, 4), np.float32),
                )
            for name, row_shape, dtype in fields:
                dataset = _create_stream_dataset(
                    private_camera,
                    name,
                    row_shape,
                    dtype,
                    compression=compression,
                    compression_opts=compression_opts,
                )
                dataset.attrs["role"] = "private_simulator_geometry"
            if self.geometry_encoding == RAW_EPISODE_V1_GEOMETRY_ENCODING:
                fixed_kwargs: dict[str, Any] = {}
                if compression is not None:
                    fixed_kwargs.update(
                        compression=compression,
                        compression_opts=compression_opts,
                        shuffle=True,
                    )
                ray = private_camera.create_dataset(
                    "ray_xy",
                    shape=(spec.height, spec.width, 2),
                    dtype=np.float32,
                    chunks=(spec.height, spec.width, 2),
                    **fixed_kwargs,
                )
                ray.attrs.update(
                    role="private_simulator_geometry_calibration",
                    semantics="episode_constant_[X/depth,Y/depth]",
                )
                ray_valid = private_camera.create_dataset(
                    "ray_valid",
                    shape=(spec.height, spec.width),
                    dtype=np.uint8,
                    chunks=(spec.height, spec.width),
                    **fixed_kwargs,
                )
                ray_valid.attrs["semantics"] = (
                    "1 iff at least one valid nonzero-actor Position "
                    "established this ray"
                )
                K = private_camera.create_dataset("K", shape=(3, 3), dtype=np.float32)
                K.attrs.update(
                    role="private_simulator_geometry_calibration",
                    episode_constant=np.uint8(1),
                )
                private_camera["depth_m"].attrs.update(
                    unit="meter",
                    semantics="negative_opengl_camera_z_on_valid_pixels_else_zero",
                )
            else:
                private_camera["position_xyz"].attrs.update(
                    unit="meter",
                    coordinate_convention="declared_in_episode_metadata",
                )
            if self.geometry_encoding == RAW_EPISODE_V1_GEOMETRY_ENCODING:
                private_camera["position_valid"].attrs["semantics"] = (
                    "1 iff renderer Position is valid and actor_seg_raw is nonzero"
                )
            else:
                private_camera["position_valid"].attrs["semantics"] = (
                    "1 iff the renderer Position value is valid"
                )
            if spec.store_mesh_segmentation:
                mesh = _create_stream_dataset(
                    private_camera,
                    "mesh_seg_raw",
                    (spec.height, spec.width),
                    spec.segmentation_dtype,
                    compression=compression,
                    compression_opts=compression_opts,
                )
                mesh.attrs["role"] = "private_simulator_debug_geometry"

        timeline = root.create_group("timeline")
        timeline.attrs.update(
            role="cross_modal_time_alignment",
            units_timestamp_sec="second",
        )
        for name in TIMELINE_FIELDS:
            _create_stream_dataset(
                timeline,
                name,
                (),
                np.int64,
                compression=None,
                compression_opts=None,
            )
        _create_stream_dataset(
            timeline,
            "timestamp_sec",
            (),
            np.float64,
            compression=None,
            compression_opts=None,
        )

        robot = root.create_group("robot_state")
        robot.attrs["role"] = "proprioception_and_action_alignment"
        for name, dimension in self._robot_dimensions.items():
            dataset = _create_stream_dataset(
                robot,
                name,
                (dimension,),
                np.float32,
                compression=compression,
                compression_opts=compression_opts,
            )
            if name == "qpos_real":
                dataset.attrs.update(role="model_visible_proprioception", semantics="measured_qpos")
            elif name == "drive_target":
                dataset.attrs.update(role="private_control_state", semantics="joint_drive_target")
            else:
                dataset.attrs.update(role="training_target", semantics="replay_action_command")
        action_valid = _create_stream_dataset(
            robot,
            "action_target_valid",
            (),
            np.uint8,
            compression=None,
            compression_opts=None,
        )
        action_valid.attrs["semantics"] = "1 iff action_target is aligned and finite"


        entities = root.create_group("entities")
        entities.attrs.update(
            role="private_simulator_geometry",
            model_visible=np.uint8(0),
            entity_count=self.entity_count,
        )
        registry_strings = np.asarray(
            [_canonical_json(item) for item in self.entity_registry], dtype=object
        )
        entities.create_dataset(
            "registry_json", data=registry_strings, dtype=string_dtype
        )
        pose = _create_stream_dataset(
            entities,
            "pose_to_world",
            (self.entity_count, 4, 4),
            np.float32,
            compression=compression,
            compression_opts=compression_opts,
        )
        pose.attrs.update(
            role="private_simulator_geometry",
            semantics="T_entity_to_world_in_registry_pose_row_order",
        )
        pose_valid = _create_stream_dataset(
            entities,
            "pose_valid",
            (self.entity_count,),
            np.uint8,
            compression=compression,
            compression_opts=compression_opts,
        )
        pose_valid.attrs["semantics"] = "1 iff pose_to_world row is valid"
        root.flush()

    def _prepare_camera_frame(
        self, spec: CameraSpec, frame: CameraFrame
    ) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
        shape = (spec.height, spec.width)
        position = _as_exact_shape(
            frame.position_xyz,
            (*shape, 3),
            name=f"{spec.name}.position_xyz",
            dtype=np.float32,
        )
        valid = _as_binary_mask(
            frame.position_valid, shape, name=f"{spec.name}.position_valid"
        )
        actor_seg = _as_segmentation(
            frame.actor_seg_raw,
            shape,
            name=f"{spec.name}.actor_seg_raw",
            dtype=spec.segmentation_dtype,
        )
        if self.geometry_encoding == RAW_EPISODE_V1_GEOMETRY_ENCODING:
            # SAPIEN uses actor id 0 for renderer no-hit pixels.  Their finite
            # Position payload is not physical scene geometry and can change
            # discontinuously as silhouettes cross the pixel, so the episode export must
            # not use it to establish or validate an episode-static ray.
            valid = np.logical_and(valid.astype(bool), actor_seg != 0).astype(
                np.uint8
            )
        valid_bool = valid.astype(bool)
        if not np.isfinite(position[valid_bool]).all():
            raise ValueError(f"{spec.name}.position_xyz has NaN/Inf at valid pixels")
        K = _as_exact_shape(
            frame.K,
            (3, 3),
            name=f"{spec.name}.K",
            dtype=np.float32,
        )
        camera_pose = _as_exact_shape(
            frame.T_cam_to_world,
            (4, 4),
            name=f"{spec.name}.T_cam_to_world",
            dtype=np.float32,
        )
        if not np.isfinite(K).all():
            raise ValueError(f"{spec.name}.K contains NaN/Inf")
        camera_pose = require_se3(
            camera_pose, f"{spec.name}.T_cam_to_world"
        )

        static_update: dict[str, Any] | None = None
        if self.geometry_encoding == RAW_EPISODE_V1_GEOMETRY_ENCODING:
            depth = -position[..., 2]
            bad_depth = valid_bool & (~np.isfinite(depth) | (depth <= 0))
            if bad_depth.any():
                raise ValueError(
                    f"{spec.name} v1 depth has {int(bad_depth.sum())} invalid "
                    "values at position_valid pixels"
                )
            existing_K = self._v1_K[spec.name]
            if existing_K is not None and not np.array_equal(existing_K, K):
                raise ValueError(f"{spec.name}.K changed within a v1 episode")
            known = self._v1_ray_valid[spec.name]
            ray = self._v1_ray_xy[spec.name]
            new_mask = valid_bool & ~known
            new_values = (
                position[new_mask, :2] / depth[new_mask, None]
                if new_mask.any()
                else np.empty((0, 2), dtype=np.float32)
            )
            current_ray = ray[valid_bool].copy()
            if new_mask.any():
                current_ray[~known[valid_bool]] = new_values
            current_depth = depth[valid_bool]
            reconstructed = np.empty_like(position[valid_bool])
            np.multiply(
                current_ray,
                current_depth[:, None],
                out=reconstructed[:, :2],
            )
            np.negative(current_depth, out=reconstructed[:, 2])
            error = np.linalg.norm(
                reconstructed - position[valid_bool], axis=-1
            )
            max_error = float(error.max(initial=0.0))
            if max_error > self.max_reconstruction_error_m:
                raise ValueError(
                    f"{spec.name} ray contract reconstruction failed: "
                    f"max_xyz_error_m={max_error:.9g}, "
                    f"limit={self.max_reconstruction_error_m:.9g}"
                )
            prepared = {
                "depth_m": np.where(valid_bool, depth, 0.0).astype(np.float32),
                "position_valid": valid,
                "actor_seg_raw": actor_seg,
                "T_cam_to_world": camera_pose,
            }
            static_update = {
                "K": K.copy(),
                "new_mask": new_mask,
                "new_values": np.asarray(new_values, dtype=np.float32),
                "validated_vector_count": int(valid_bool.sum()),
                "max_xyz_error_m": max_error,
            }
        else:
            prepared = {
                "position_xyz": position,
                "position_valid": valid,
                "actor_seg_raw": actor_seg,
                "K": K,
                "T_cam_to_world": camera_pose,
            }

        if spec.store_rgb:
            if frame.rgb is None:
                raise ValueError(f"{spec.name}.rgb is required by CameraSpec")
            prepared["rgb"] = _as_exact_shape(
                frame.rgb,
                (*shape, 3),
                name=f"{spec.name}.rgb",
                dtype=np.uint8,
            )
        elif frame.rgb is not None:
            raise ValueError(
                f"{spec.name}.rgb was provided but store_rgb=false; this likely "
                "indicates a schema/configuration mismatch"
            )

        if spec.store_mesh_segmentation:
            if frame.mesh_seg_raw is None:
                raise ValueError(
                    f"{spec.name}.mesh_seg_raw is required by CameraSpec"
                )
            prepared["mesh_seg_raw"] = _as_segmentation(
                frame.mesh_seg_raw,
                shape,
                name=f"{spec.name}.mesh_seg_raw",
                dtype=spec.segmentation_dtype,
            )
        elif frame.mesh_seg_raw is not None:
            raise ValueError(
                f"{spec.name}.mesh_seg_raw was provided but "
                "store_mesh_segmentation=false"
            )
        return prepared, static_update

    def _prepare_frame(
        self,
        *,
        cameras: Mapping[str, CameraFrame],
        timeline: TimelineFrame,
        robot_state: RobotStateFrame,
        entities: EntityFrame,
    ) -> dict[str, Any]:
        expected_cameras = set(self._camera_by_name)
        actual_cameras = set(cameras)
        if actual_cameras != expected_cameras:
            raise ValueError(
                f"camera set={sorted(actual_cameras)}, "
                f"expected={sorted(expected_cameras)}"
            )
        current_index = self.num_frames
        if int(timeline.save_index) != current_index:
            raise ValueError(
                f"timeline.save_index={timeline.save_index}, expected={current_index}"
            )
        timeline_values = {
            name: int(getattr(timeline, name)) for name in TIMELINE_FIELDS
        }
        timestamp = float(timeline.timestamp_sec)
        if not np.isfinite(timestamp):
            raise ValueError("timeline.timestamp_sec must be finite")
        if current_index:
            previous_timestamp = float(self._file["timeline/timestamp_sec"][-1])
            if timestamp < previous_timestamp:
                raise ValueError(
                    f"timestamp_sec moved backwards: {timestamp} < {previous_timestamp}"
                )
            for name in ("sim_step", "control_step", "render_step"):
                previous = int(self._file[f"timeline/{name}"][-1])
                if timeline_values[name] < previous:
                    raise ValueError(
                        f"timeline.{name} moved backwards: "
                        f"{timeline_values[name]} < {previous}"
                    )

        prepared_cameras: dict[str, dict[str, np.ndarray]] = {}
        camera_static_updates: dict[str, dict[str, Any] | None] = {}
        for name, frame in cameras.items():
            prepared_camera, static_update = self._prepare_camera_frame(
                self._camera_by_name[name], frame
            )
            prepared_cameras[name] = prepared_camera
            camera_static_updates[name] = static_update
        prepared_robot = {}
        for name, dimension in self._robot_dimensions.items():
            prepared_robot[name] = _as_exact_shape(
                getattr(robot_state, name),
                (dimension,),
                name=f"robot_state.{name}",
                dtype=np.float32,
            )
            if name != "action_target" or bool(robot_state.action_target_valid):
                if not np.isfinite(prepared_robot[name]).all():
                    raise ValueError(f"robot_state.{name} contains NaN/Inf")
        prepared_robot["action_target_valid"] = np.uint8(robot_state.action_target_valid)

        pose = _as_exact_shape(
            entities.pose_to_world,
            (self.entity_count, 4, 4),
            name="entities.pose_to_world",
            dtype=np.float32,
        )
        pose_valid = _as_binary_mask(
            entities.pose_valid,
            (self.entity_count,),
            name="entities.pose_valid",
        )
        pose = require_se3(
            pose,
            "entities.pose_to_world",
            valid_mask=pose_valid.astype(bool),
        )
        return {
            "cameras": prepared_cameras,
            "camera_static_updates": camera_static_updates,
            "timeline": {**timeline_values, "timestamp_sec": timestamp},
            "robot_state": prepared_robot,
            "pose_to_world": pose,
            "pose_valid": pose_valid,
        }

    def append_frame(
        self,
        *,
        cameras: Mapping[str, CameraFrame],
        timeline: TimelineFrame,
        robot_state: RobotStateFrame,
        entities: EntityFrame,
    ) -> int:
        """Validate and append exactly one aligned saved frame.

        Validation happens before any dataset is resized.  Unexpected HDF5
        write failures roll all first dimensions back to their old length.
        """

        self._ensure_open()
        prepared = self._prepare_frame(
            cameras=cameras,
            timeline=timeline,
            robot_state=robot_state,
            entities=entities,
        )
        row_index = self.num_frames
        datasets: list[tuple[Any, np.ndarray | int | float]] = []
        for camera_name, fields in prepared["cameras"].items():
            for field_name, value in fields.items():
                root_group = (
                    "observations" if field_name == "rgb" else "private_geometry"
                )
                datasets.append(
                    (self._file[f"{root_group}/{camera_name}/{field_name}"], value)
                )
        for name, value in prepared["timeline"].items():
            datasets.append((self._file[f"timeline/{name}"], value))
        for name, value in prepared["robot_state"].items():
            datasets.append((self._file[f"robot_state/{name}"], value))
        datasets.extend(
            (
                (self._file["entities/pose_to_world"], prepared["pose_to_world"]),
                (self._file["entities/pose_valid"], prepared["pose_valid"]),
            )
        )

        resized: list[Any] = []
        try:
            for dataset, value in datasets:
                dataset.resize(row_index + 1, axis=0)
                resized.append(dataset)
                dataset[row_index] = value
            self._file.attrs["num_frames"] = row_index + 1
            self._file.flush()
            if self.geometry_encoding == RAW_EPISODE_V1_GEOMETRY_ENCODING:
                for camera_name, update in prepared["camera_static_updates"].items():
                    assert update is not None
                    if self._v1_K[camera_name] is None:
                        self._v1_K[camera_name] = update["K"]
                    new_mask = update["new_mask"]
                    if new_mask.any():
                        self._v1_ray_xy[camera_name][new_mask] = update["new_values"]
                        self._v1_ray_valid[camera_name][new_mask] = True
                    self._v1_validated_vector_count[camera_name] += update[
                        "validated_vector_count"
                    ]
                    self._v1_max_xyz_error_m[camera_name] = max(
                        self._v1_max_xyz_error_m[camera_name],
                        update["max_xyz_error_m"],
                    )
        except BaseException:
            for dataset in reversed(resized):
                dataset.resize(row_index, axis=0)
            self._file.attrs["num_frames"] = row_index
            self._file.flush()
            raise
        return row_index

    def update_episode_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Replace the scalar metadata snapshot while the episode is partial.

        This binds post-replay gates that are unavailable when the first frame
        creates the streaming HDF5 file. It changes no frame-aligned dataset.
        """

        self._ensure_open()
        self._file["metadata/episode_json"][()] = _canonical_json(dict(metadata))
        self._file.flush()

    def write_control_stream(
        self,
        *,
        drive_target: np.ndarray,
        stream_index: np.ndarray,
        control_step: np.ndarray,
        sim_step_before: np.ndarray,
        sim_step_after: np.ndarray,
        timestamp_before_sec: np.ndarray,
        timestamp_after_sec: np.ndarray,
        saved_frame_applied_start: np.ndarray,
        saved_frame_applied_stop: np.ndarray,
        contract: Mapping[str, Any],
    ) -> None:
        """Write the complete successful-step command stream once.

        The stream is small relative to camera geometry, so it is buffered in
        memory and written as fixed-size datasets immediately before episode
        publication. Older readers safely ignore this optional group.
        """

        self._ensure_open()
        if "control_stream" in self._file:
            raise RuntimeError("control_stream was already written")

        drive = np.asarray(drive_target, dtype=np.float32)
        expected_dim = self._robot_dimensions["drive_target"]
        if drive.ndim != 2 or drive.shape[1] != expected_dim:
            raise ValueError(
                "control_stream.drive_target must have shape "
                f"[C, {expected_dim}], got {drive.shape}"
            )
        count = int(drive.shape[0])
        if count <= 0:
            raise ValueError("control_stream must contain a successful step")
        if not np.isfinite(drive).all():
            raise ValueError("control_stream.drive_target contains NaN/Inf")

        def vector(value: Any, name: str, dtype: Any) -> np.ndarray:
            result = np.asarray(value, dtype=dtype)
            if result.shape != (count,):
                raise ValueError(
                    f"control_stream.{name} shape={result.shape}, "
                    f"expected={(count,)}"
                )
            return result

        indices = vector(stream_index, "stream_index", np.int64)
        controls = vector(control_step, "control_step", np.int64)
        before = vector(sim_step_before, "sim_step_before", np.int64)
        after = vector(sim_step_after, "sim_step_after", np.int64)
        time_before = vector(
            timestamp_before_sec, "timestamp_before_sec", np.float64
        )
        time_after = vector(timestamp_after_sec, "timestamp_after_sec", np.float64)
        if not np.array_equal(indices, np.arange(count, dtype=np.int64)):
            raise ValueError("control_stream.stream_index must start at zero")
        if not np.array_equal(after, before + 1):
            raise ValueError("each successful control must advance sim_step by one")
        if count > 1 and (
            np.any(np.diff(controls) <= 0)
            or np.any(np.diff(before) <= 0)
            or np.any(np.diff(after) <= 0)
        ):
            raise ValueError("control/simulator steps must be strictly increasing")
        if not np.isfinite(time_before).all() or not np.isfinite(time_after).all():
            raise ValueError("control_stream timestamps contain NaN/Inf")
        if np.any(time_after <= time_before) or (
            count > 1 and np.any(np.diff(time_after) <= 0)
        ):
            raise ValueError("successful-step timestamps must advance strictly")

        frame_count = self.num_frames
        starts = np.asarray(saved_frame_applied_start, dtype=np.int64)
        stops = np.asarray(saved_frame_applied_stop, dtype=np.int64)
        if starts.shape != (frame_count,) or stops.shape != (frame_count,):
            raise ValueError(
                "saved-frame spans require one row per frame: "
                f"starts={starts.shape}, stops={stops.shape}, frames={frame_count}"
            )
        if (
            np.any(starts < 0)
            or np.any(stops < starts)
            or np.any(stops > count)
            or int(starts[0]) != 0
            or int(stops[0]) != 0
            or (
                frame_count > 1
                and not np.array_equal(starts[1:], stops[:-1])
            )
        ):
            raise ValueError("saved-frame control spans are not contiguous")
        saved_sim_steps = np.asarray(
            self._file["timeline/sim_step"][:], dtype=np.int64
        )
        if not np.array_equal(saved_sim_steps, stops):
            raise ValueError(
                "saved-frame control stop pointers must equal captured sim_step"
            )

        saved_action = np.asarray(self._file["robot_state/action_target"][:])
        saved_valid = np.asarray(
            self._file["robot_state/action_target_valid"][:], dtype=bool
        )
        expected_valid = stops > 0
        if not np.array_equal(saved_valid, expected_valid):
            raise ValueError(
                "saved-frame action validity does not match control endpoint"
            )
        if expected_valid.any():
            expected = drive[stops[expected_valid] - 1]
            if not np.array_equal(saved_action[expected_valid], expected):
                raise ValueError(
                    "saved-frame past action differs from last applied command"
                )

        group = self._file.create_group("control_stream")
        group.attrs.update(
            schema_version=CONTROL_STREAM_SCHEMA_VERSION,
            role="low_level_control_label_source_and_alignment",
            model_visible=np.uint8(0),
            high_level_policy_action=np.uint8(0),
        )
        string_dtype = _require_h5py().string_dtype(encoding="utf-8")
        group.create_dataset(
            "contract_json",
            data=_canonical_json(dict(contract)),
            dtype=string_dtype,
        )
        datasets = {
            "drive_target": drive,
            "stream_index": indices,
            "control_step": controls,
            "sim_step_before": before,
            "sim_step_after": after,
            "timestamp_before_sec": time_before,
            "timestamp_after_sec": time_after,
            "saved_frame_applied_start": starts,
            "saved_frame_applied_stop": stops,
        }
        for name, value in datasets.items():
            kwargs: dict[str, Any] = {}
            if value.ndim > 1:
                kwargs.update(compression="gzip", compression_opts=1, shuffle=True)
            dataset = group.create_dataset(name, data=value, **kwargs)
            dataset.attrs["role"] = (
                "frame_to_past_control_alignment"
                if name.startswith("saved_frame_")
                else "successful_scene_step_control_stream"
            )
        group["drive_target"].attrs.update(
            semantics=(
                "low-level joint drive target sampled immediately before "
                "scene.step and committed only after that step succeeds"
            ),
            unit="RoboTwin joint-drive convention",
        )
        group["saved_frame_applied_start"].attrs["semantics"] = (
            "inclusive start of commands applied after the previous saved frame"
        )
        group["saved_frame_applied_stop"].attrs["semantics"] = (
            "exclusive endpoint already reflected by the observation; "
            "future low-level labels begin at this pointer"
        )
        self._file.attrs["control_stream_schema_version"] = (
            CONTROL_STREAM_SCHEMA_VERSION
        )
        self._file.flush()

    def _all_stream_lengths(self) -> dict[str, int]:
        lengths: dict[str, int] = {}

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, _require_h5py().Dataset) and obj.maxshape:
                if obj.maxshape[0] is None:
                    lengths[name] = int(obj.shape[0])

        self._file.visititems(visit)
        return lengths

    def finalize(self) -> Path:
        """Validate, close, and atomically publish the completed episode."""

        self._ensure_open()
        frame_count = self.num_frames
        if frame_count <= 0:
            raise RuntimeError("refusing to finalize an empty raw episode")
        if self.geometry_encoding == RAW_EPISODE_V1_GEOMETRY_ENCODING:
            for camera_name in self._camera_by_name:
                K = self._v1_K[camera_name]
                if K is None:
                    raise RuntimeError(f"v1 camera {camera_name!r} has no calibration")
                group = self._file[f"private_geometry/{camera_name}"]
                group["ray_xy"][...] = self._v1_ray_xy[camera_name]
                group["ray_valid"][...] = self._v1_ray_valid[camera_name].astype(np.uint8)
                group["K"][...] = K
                group.attrs.update(
                    v1_validated_vector_count=np.int64(
                        self._v1_validated_vector_count[camera_name]
                    ),
                    v1_max_xyz_error_m=np.float64(
                        self._v1_max_xyz_error_m[camera_name]
                    ),
                    v1_reconstruction_error_limit_m=np.float64(
                        self.max_reconstruction_error_m
                    ),
                )
            self._file.attrs.update(
                v1_validated_vector_count=np.int64(
                    sum(self._v1_validated_vector_count.values())
                ),
                v1_max_xyz_error_m=np.float64(
                    max(self._v1_max_xyz_error_m.values(), default=0.0)
                ),
            )
            self._file.flush()
        lengths = self._all_stream_lengths()
        bad_lengths = {name: size for name, size in lengths.items() if size != frame_count}
        if bad_lengths:
            raise RuntimeError(
                f"stream dataset lengths do not match num_frames={frame_count}: "
                f"{bad_lengths}"
            )
        self._file.attrs.update(status="complete", num_frames=frame_count)
        self._file.flush()
        self._file.close()
        self._closed = True
        file_fd = os.open(self.partial_path, os.O_RDONLY)
        try:
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.replace(self.partial_path, self.final_path)
        directory_fd = os.open(
            self.final_path.parent, os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self._finalized = True
        return self.final_path

    def abort(self, reason: str) -> Path:
        """Close a failed export while deliberately leaving only ``.partial``."""

        if self._closed:
            return self.partial_path
        self._file.attrs.update(status="failed", failure_reason=str(reason))
        self._file.flush()
        self._file.close()
        self._closed = True
        return self.partial_path

    def close(self) -> None:
        """Close without publishing; the file remains status=partial."""

        if not self._closed:
            self._file.flush()
            self._file.close()
            self._closed = True

    def __enter__(self) -> "RawEpisodeWriter":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        if exc is not None:
            self.abort(f"{type(exc).__name__}: {exc}")
        elif not self._finalized:
            self.close()
        return False


class RawEpisodeReader:
    """PID-aware lazy HDF5 reader suitable for DataLoader workers.

    Construction and pickling never hold an h5py handle.  If a process forks
    after the file has been opened, the child detects the PID change and opens
    its own handle instead of reusing the inherited one.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        require_complete: bool = True,
        require_success: bool = False,
        verify_success_hashes: bool = True,
        swmr: bool = True,
    ) -> None:
        _require_h5py()
        self.path = Path(path)
        self.require_complete = bool(require_complete)
        self.require_success = bool(require_success)
        self.verify_success_hashes = bool(verify_success_hashes)
        self.swmr = bool(swmr)
        self._file: Any | None = None
        self._owner_pid: int | None = None
        self._ray_cache: dict[str, np.ndarray] = {}
        self._ray_cache_owner_pid: int | None = None

    @property
    def is_open(self) -> bool:
        return self._file is not None

    def _ensure_open(self) -> Any:
        current_pid = os.getpid()
        if self._file is not None and self._owner_pid != current_pid:
            # Do not trust an inherited libhdf5 handle after fork.
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._ray_cache.clear()
            self._ray_cache_owner_pid = None
        if self._file is None:
            h5 = _require_h5py()
            if not self.path.is_file():
                raise FileNotFoundError(f"raw episode does not exist: {self.path}")
            if self.require_success:
                from .episode_publication import require_episode_success

                require_episode_success(
                    self.path,
                    verify_hashes=self.verify_success_hashes,
                )
            self._file = h5.File(self.path, "r", swmr=self.swmr, libver="latest")
            self._owner_pid = current_pid
            try:
                schema_version = str(self._file.attrs.get("schema_version", ""))
                if schema_version not in READABLE_RAW_EPISODE_SCHEMAS:
                    raise ValueError(
                        f"unsupported raw episode schema {schema_version!r}; "
                        f"readable={READABLE_RAW_EPISODE_SCHEMAS!r}"
                    )
                if schema_version in {
                    RAW_EPISODE_SCHEMA_V1,
                    RAW_EPISODE_SCHEMA_V2_EXACT_POSITION,
                }:
                    self._validate_depth_ray_layout(self._file)
                status = str(self._file.attrs.get("status", ""))
                if self.require_complete and status != "complete":
                    raise ValueError(
                        f"raw episode is not complete: status={status!r}, path={self.path}"
                    )
            except BaseException:
                self._file.close()
                self._file = None
                self._owner_pid = None
                raise
        return self._file

    @property
    def num_frames(self) -> int:
        handle = self._ensure_open()
        return int(handle.attrs["num_frames"])

    @property
    def camera_names(self) -> tuple[str, ...]:
        handle = self._ensure_open()
        return tuple(handle["private_geometry"].keys())

    @property
    def schema_version(self) -> str:
        """Physical raw-episode schema stored in the HDF5 file."""

        return str(self._ensure_open().attrs["schema_version"])

    @property
    def geometry_encoding(self) -> str:
        handle = self._ensure_open()
        if self.schema_version == RAW_EPISODE_SCHEMA_VERSION:
            return "position_xyz_float32_v0"
        return str(handle.attrs["geometry_encoding"])

    @property
    def geometry_contract_hash(self) -> str:
        handle = self._ensure_open()
        value = handle.attrs.get("geometry_contract_sha256")
        if value is not None:
            return str(value)
        import hashlib

        descriptor = _canonical_json(
            {
                "schema_version": self.schema_version,
                "geometry_encoding": self.geometry_encoding,
            }
        )
        return hashlib.sha256(descriptor.encode("utf-8")).hexdigest()

    def camera_image_shape(self, camera_name: str) -> tuple[int, int]:
        handle = self._ensure_open()
        if camera_name not in self.camera_names:
            raise KeyError(f"unknown camera: {camera_name!r}")
        group = handle[f"private_geometry/{camera_name}"]
        height = int(group.attrs.get("height", 0))
        width = int(group.attrs.get("width", 0))
        if height > 0 and width > 0:
            return height, width
        field = "position_xyz" if "position_xyz" in group else "depth_m"
        return tuple(int(value) for value in group[field].shape[1:3])

    @staticmethod
    def _validate_depth_ray_layout(handle: Any) -> None:
        schema_version = str(handle.attrs.get("schema_version", ""))
        expected_encoding = {
            RAW_EPISODE_SCHEMA_V1: RAW_EPISODE_V1_GEOMETRY_ENCODING,
            RAW_EPISODE_SCHEMA_V2_EXACT_POSITION: (
                RAW_EPISODE_V2_EXACT_POSITION_GEOMETRY_ENCODING
            ),
        }.get(schema_version)
        if expected_encoding is None:
            raise ValueError(
                f"depth-ray layout does not support schema={schema_version!r}"
            )
        encoding = str(handle.attrs.get("geometry_encoding", ""))
        if encoding != expected_encoding:
            raise ValueError(
                f"raw schema {schema_version!r} geometry_encoding={encoding!r}, "
                f"expected {expected_encoding!r}"
            )
        if schema_version == RAW_EPISODE_SCHEMA_V2_EXACT_POSITION:
            contract_hash = str(handle.attrs.get("geometry_contract_sha256", ""))
            if len(contract_hash) != 64 or any(
                character not in "0123456789abcdef" for character in contract_hash
            ):
                raise ValueError(
                    "raw exact-position candidate requires a lowercase hex "
                    "geometry_contract_sha256"
                )
            if int(handle.attrs.get("candidate_only", -1)) != 1:
                raise ValueError("raw exact-position candidate_only must be true")
            if int(handle.attrs.get("production_authorized", -1)) != 0:
                raise ValueError(
                    "raw exact-position production_authorized must be false"
                )
            if str(handle.attrs.get("converted_from_schema", "")) != (
                RAW_EPISODE_SCHEMA_VERSION
            ):
                raise ValueError("raw exact-position source schema must be v0")
        frame_count = int(handle.attrs.get("num_frames", -1))
        for camera_name, geometry in handle["private_geometry"].items():
            if "position_xyz" in geometry or "depth_m" not in geometry:
                raise ValueError(
                    f"exported camera {camera_name!r} must contain depth_m and "
                    "must not physically contain position_xyz"
                )
            depth = geometry["depth_m"]
            valid = geometry.get("position_valid")
            if valid is None or depth.shape != valid.shape:
                raise ValueError(
                    f"exported camera {camera_name!r} depth/valid shape mismatch"
                )
            if depth.shape[0] != frame_count or depth.dtype != np.dtype("float32"):
                raise ValueError(
                    f"exported camera {camera_name!r} depth_m must be "
                    f"float32[T,H,W], got {depth.shape} {depth.dtype}"
                )
            if "ray_xy" not in geometry or "ray_valid" not in geometry:
                raise ValueError(
                    f"exported camera {camera_name!r} is missing static ray data"
                )
            if "K" not in geometry or geometry["K"].shape != (3, 3):
                raise ValueError(
                    f"exported camera {camera_name!r} K must be episode-static [3,3]"
                )
            if geometry["K"].dtype != np.dtype("float32"):
                raise ValueError(
                    f"exported camera {camera_name!r} K must be float32"
                )
            expected_shape = tuple(int(value) for value in depth.shape[1:])
            ray = geometry["ray_xy"]
            ray_valid = geometry["ray_valid"]
            if ray.shape != (*expected_shape, 2) or ray.dtype != np.dtype("float32"):
                raise ValueError(
                    f"exported camera {camera_name!r} ray_xy has invalid layout"
                )
            if ray_valid.shape != expected_shape or ray_valid.dtype != np.dtype("uint8"):
                raise ValueError(
                    f"exported camera {camera_name!r} ray_valid has invalid layout"
                )

            residual_fields = {
                RAW_EPISODE_V2_POSITION_XY_XOR_FIELD,
                RAW_EPISODE_V2_POSITION_Z_XOR_FIELD,
            }
            present_residuals = residual_fields & set(geometry.keys())
            if schema_version == RAW_EPISODE_SCHEMA_V1:
                if present_residuals:
                    raise ValueError(
                        f"exported camera {camera_name!r} unexpectedly contains "
                        f"exact-position residuals: {sorted(present_residuals)}"
                    )
                continue
            if present_residuals != residual_fields:
                raise ValueError(
                    f"raw exact-position camera {camera_name!r} requires "
                    f"residual fields {sorted(residual_fields)}"
                )
            xy_xor = geometry[RAW_EPISODE_V2_POSITION_XY_XOR_FIELD]
            z_xor = geometry[RAW_EPISODE_V2_POSITION_Z_XOR_FIELD]
            if (
                xy_xor.shape != (frame_count, *expected_shape, 2)
                or xy_xor.dtype != np.dtype("uint32")
            ):
                raise ValueError(
                    f"raw exact-position camera {camera_name!r} "
                    f"{RAW_EPISODE_V2_POSITION_XY_XOR_FIELD} has invalid layout"
                )
            if (
                z_xor.shape != (frame_count, *expected_shape)
                or z_xor.dtype != np.dtype("uint32")
            ):
                raise ValueError(
                    f"raw exact-position camera {camera_name!r} "
                    f"{RAW_EPISODE_V2_POSITION_Z_XOR_FIELD} has invalid layout"
                )

    def _v1_ray_xy(self, camera_name: str) -> np.ndarray:
        current_pid = os.getpid()
        if self._ray_cache_owner_pid != current_pid:
            self._ray_cache.clear()
            self._ray_cache_owner_pid = current_pid
        if camera_name not in self._ray_cache:
            ray = np.asarray(
                self._ensure_open()[f"private_geometry/{camera_name}/ray_xy"][()],
                dtype=np.float32,
            )
            ray.setflags(write=False)
            self._ray_cache[camera_name] = ray
        return self._ray_cache[camera_name]

    def _depth_ray_position_xyz(self, camera_name: str, index: int) -> np.ndarray:
        depth = np.asarray(
            self._ensure_open()[f"private_geometry/{camera_name}/depth_m"][index],
            dtype=np.float32,
        )
        ray = self._v1_ray_xy(camera_name)
        result = np.empty((*depth.shape, 3), dtype=np.float32)
        np.multiply(ray, depth[..., None], out=result[..., :2])
        np.negative(depth, out=result[..., 2])
        if self.schema_version == RAW_EPISODE_SCHEMA_V2_EXACT_POSITION:
            geometry = self._ensure_open()[f"private_geometry/{camera_name}"]
            xy_xor = np.asarray(
                geometry[RAW_EPISODE_V2_POSITION_XY_XOR_FIELD][index],
                dtype=np.uint32,
            )
            z_xor = np.asarray(
                geometry[RAW_EPISODE_V2_POSITION_Z_XOR_FIELD][index],
                dtype=np.uint32,
            )
            xy_bits = result[..., :2].view(np.uint32)
            z_bits = result[..., 2].view(np.uint32)
            np.bitwise_xor(xy_bits, xy_xor, out=xy_bits)
            np.bitwise_xor(z_bits, z_xor, out=z_bits)
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        handle = self._ensure_open()
        value = handle["metadata/episode_json"][()]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(str(value))

    @property
    def entity_registry(self) -> list[dict[str, Any]]:
        handle = self._ensure_open()
        result = []
        for value in handle["entities/registry_json"][:]:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            result.append(json.loads(str(value)))
        return result

    @property
    def has_control_stream(self) -> bool:
        """Whether this episode carries the optional complete control stream."""

        return "control_stream" in self._ensure_open()

    @property
    def control_stream_count(self) -> int:
        handle = self._ensure_open()
        if "control_stream" not in handle:
            return 0
        return int(handle["control_stream/drive_target"].shape[0])

    @property
    def control_stream_contract(self) -> dict[str, Any]:
        handle = self._ensure_open()
        if "control_stream" not in handle:
            raise ValueError(
                "canonical episode has no complete control_stream; do not infer "
                "future actions from saved-frame drive_target"
            )
        value = handle["control_stream/contract_json"][()]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(str(value))

    def read_control_stream_fields(
        self,
        start: int = 0,
        stop: int | None = None,
        *,
        fields: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        handle = self._ensure_open()
        if "control_stream" not in handle:
            raise ValueError("canonical episode has no control_stream")
        group = handle["control_stream"]
        available = {
            name: dataset
            for name, dataset in group.items()
            if name
            not in {
                "contract_json",
                "saved_frame_applied_start",
                "saved_frame_applied_stop",
            }
        }
        selected = self._selected_fields(
            available, fields, group_name="control_stream"
        )
        count = int(group["drive_target"].shape[0])
        begin = int(start)
        end = count if stop is None else int(stop)
        if begin < 0:
            begin += count
        if end < 0:
            end += count
        if not (0 <= begin <= end <= count):
            raise IndexError(
                f"control stream slice [{begin}:{end}] outside [0:{count}]"
            )
        return {name: available[name][begin:end] for name in selected}

    def saved_frame_control_span(self, index: int) -> tuple[int, int]:
        handle = self._ensure_open()
        if "control_stream" not in handle:
            raise ValueError("canonical episode has no control_stream")
        frame_count = int(handle.attrs["num_frames"])
        normalized = int(index)
        if normalized < 0:
            normalized += frame_count
        if not 0 <= normalized < frame_count:
            raise IndexError(
                f"frame index {normalized} out of range for {frame_count} frames"
            )
        group = handle["control_stream"]
        return (
            int(group["saved_frame_applied_start"][normalized]),
            int(group["saved_frame_applied_stop"][normalized]),
        )

    def read_index_alignment_fields(self) -> dict[str, np.ndarray]:
        """Read the three per-frame index-alignment columns in one HDF5 slice."""

        handle = self._ensure_open()
        if "control_stream" not in handle:
            raise ValueError("canonical episode has no control_stream")
        control_group = handle["control_stream"]
        timeline_group = handle["timeline"]
        return {
            "timestamp_sec": np.asarray(
                timeline_group["timestamp_sec"][:],
            ),
            "saved_frame_applied_start": np.asarray(
                control_group["saved_frame_applied_start"][:],
            ),
            "saved_frame_applied_stop": np.asarray(
                control_group["saved_frame_applied_stop"][:],
            ),
        }

    @staticmethod
    def _selected_fields(
        group: Any,
        requested: Sequence[str] | None,
        *,
        group_name: str,
    ) -> tuple[str, ...]:
        """Validate a field projection against one HDF5 group.

        ``None`` means every dataset in the group, while an empty sequence
        means no datasets. Rejecting unknown and duplicate names keeps typoed
        projections from silently falling back to expensive full-frame reads.
        """

        if requested is None:
            return tuple(group.keys())
        if isinstance(requested, str):
            values = (requested,)
        else:
            values = tuple(str(value) for value in requested)
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate fields requested from {group_name}: {values}")
        unknown = set(values) - set(group.keys())
        if unknown:
            raise KeyError(
                f"unknown fields in {group_name}: {sorted(unknown)}; "
                f"available={sorted(group.keys())}"
            )
        return values

    def read_frame_fields(
        self,
        index: int,
        *,
        camera_names: Sequence[str] | None = None,
        observation_fields: Sequence[str] | None = (),
        private_geometry_fields: Sequence[str] | None = (),
        timeline_fields: Sequence[str] | None = (),
        robot_state_fields: Sequence[str] | None = (),
        entity_fields: Sequence[str] | None = (),
    ) -> dict[str, Any]:
        """Read an explicit projection of one frame.

        This is the low-I/O counterpart of :meth:`read_frame`. Callers must
        opt into every returned field. Passing ``None`` for a field group
        requests all datasets in that group; the default empty tuple reads
        none. The returned nesting matches :meth:`read_frame`.
        """

        handle = self._ensure_open()
        frame_count = int(handle.attrs["num_frames"])
        index = int(index)
        if index < 0:
            index += frame_count
        if not (0 <= index < frame_count):
            raise IndexError(f"frame index {index} out of range for {frame_count} frames")

        selected = self.camera_names if camera_names is None else tuple(camera_names)
        if len(selected) != len(set(selected)):
            raise ValueError(f"duplicate camera names requested: {selected}")
        unknown_cameras = set(selected) - set(self.camera_names)
        if unknown_cameras:
            raise KeyError(f"unknown cameras: {sorted(unknown_cameras)}")

        observations: dict[str, dict[str, np.ndarray]] = {}
        private_geometry: dict[str, dict[str, np.ndarray]] = {}
        for name in selected:
            obs_group = handle[f"observations/{name}"]
            obs_names = self._selected_fields(
                obs_group,
                observation_fields,
                group_name=f"observations/{name}",
            )
            observations[name] = {
                field: obs_group[field][index] for field in obs_names
            }

            geometry_group = handle[f"private_geometry/{name}"]
            if self.schema_version in {
                RAW_EPISODE_SCHEMA_V1,
                RAW_EPISODE_SCHEMA_V2_EXACT_POSITION,
            }:
                logical_geometry = {
                    field: dataset
                    for field, dataset in geometry_group.items()
                    if field
                    not in {
                        "depth_m",
                        "ray_xy",
                        "ray_valid",
                        RAW_EPISODE_V2_POSITION_XY_XOR_FIELD,
                        RAW_EPISODE_V2_POSITION_Z_XOR_FIELD,
                    }
                }
                logical_geometry["position_xyz"] = None
                # Source-time oracle conditions need the stored depth and the
                # exact episode-static ray used to reconstruct it.  Keep
                # ``position_xyz`` as the existing logical convenience field,
                # but allow explicit low-I/O projections of the physical
                # depth/ray datasets as well.
                logical_geometry["depth_m"] = geometry_group["depth_m"]
                logical_geometry["ray_xy"] = geometry_group["ray_xy"]
                logical_geometry["ray_valid"] = geometry_group["ray_valid"]
                geometry_names = self._selected_fields(
                    logical_geometry,
                    private_geometry_fields,
                    group_name=f"private_geometry/{name}",
                )
                private_geometry[name] = {
                    field: (
                        self._depth_ray_position_xyz(name, index)
                        if field == "position_xyz"
                        else (
                            geometry_group[field][()]
                            if field in {"K", "ray_xy", "ray_valid"}
                            else geometry_group[field][index]
                        )
                    )
                    for field in geometry_names
                }
            else:
                geometry_names = self._selected_fields(
                    geometry_group,
                    private_geometry_fields,
                    group_name=f"private_geometry/{name}",
                )
                private_geometry[name] = {
                    field: geometry_group[field][index] for field in geometry_names
                }
            if "T_cam_to_world" in private_geometry[name]:
                private_geometry[name]["T_cam_to_world"] = require_se3(
                    private_geometry[name]["T_cam_to_world"],
                    f"{name}.T_cam_to_world",
                )

        timeline_group = handle["timeline"]
        selected_timeline = self._selected_fields(
            timeline_group, timeline_fields, group_name="timeline"
        )
        timeline = {
            name: timeline_group[name][index].item() for name in selected_timeline
        }

        robot_group = handle["robot_state"]
        selected_robot = self._selected_fields(
            robot_group, robot_state_fields, group_name="robot_state"
        )
        robot_state = {
            name: robot_group[name][index] for name in selected_robot
        }

        entities_group = handle["entities"]
        # registry_json is episode-level, not frame-indexed.
        available_entity_fields = {
            name: entities_group[name]
            for name in ("pose_to_world", "pose_valid")
            if name in entities_group
        }
        selected_entities = self._selected_fields(
            available_entity_fields, entity_fields, group_name="entities"
        )
        entities = {
            name: available_entity_fields[name][index] for name in selected_entities
        }
        if "pose_to_world" in entities:
            validation_mask = (
                entities["pose_valid"]
                if "pose_valid" in entities
                else entities_group["pose_valid"][index]
            )
            entities["pose_to_world"] = require_se3(
                entities["pose_to_world"],
                "entities.pose_to_world",
                valid_mask=np.asarray(validation_mask, dtype=bool),
            )
        return {
            "observations": observations,
            "private_geometry": private_geometry,
            "timeline": timeline,
            "robot_state": robot_state,
            "entities": entities,
        }

    def read_frame(
        self,
        index: int,
        *,
        camera_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        # Preserve the legacy contract, including the historical convention
        # that an empty camera list means "all cameras".
        selected = tuple(camera_names or self.camera_names)
        return self.read_frame_fields(
            index,
            camera_names=selected,
            observation_fields=None,
            private_geometry_fields=None,
            timeline_fields=None,
            robot_state_fields=None,
            entity_fields=None,
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._owner_pid = None
            self._ray_cache.clear()
            self._ray_cache_owner_pid = None

    def __enter__(self) -> "RawEpisodeReader":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        self.close()
        return False

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_file"] = None
        state["_owner_pid"] = None
        state["_ray_cache"] = {}
        state["_ray_cache_owner_pid"] = None
        return state

    def __del__(self) -> None:  # pragma: no cover - best effort during shutdown.
        try:
            self.close()
        except Exception:
            pass
