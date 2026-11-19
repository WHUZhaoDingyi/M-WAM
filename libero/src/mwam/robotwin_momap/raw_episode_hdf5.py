"""Read canonical episode geometry in CPU data-loader workers."""

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
    """Canonical physical/logical contract shared by all raw-v1 writers."""

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
    """Stable digest binding direct and converted raw-v1 episodes."""

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
        swmr: bool = True,
    ) -> None:
        _require_h5py()
        self.path = Path(path)
        self.require_complete = bool(require_complete)
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
                    f"raw v1 camera {camera_name!r} must contain depth_m and "
                    "must not physically contain position_xyz"
                )
            depth = geometry["depth_m"]
            valid = geometry.get("position_valid")
            if valid is None or depth.shape != valid.shape:
                raise ValueError(
                    f"raw v1 camera {camera_name!r} depth/valid shape mismatch"
                )
            if depth.shape[0] != frame_count or depth.dtype != np.dtype("float32"):
                raise ValueError(
                    f"raw v1 camera {camera_name!r} depth_m must be "
                    f"float32[T,H,W], got {depth.shape} {depth.dtype}"
                )
            if "ray_xy" not in geometry or "ray_valid" not in geometry:
                raise ValueError(
                    f"raw v1 camera {camera_name!r} is missing static ray data"
                )
            if "K" not in geometry or geometry["K"].shape != (3, 3):
                raise ValueError(
                    f"raw v1 camera {camera_name!r} K must be episode-static [3,3]"
                )
            if geometry["K"].dtype != np.dtype("float32"):
                raise ValueError(
                    f"raw v1 camera {camera_name!r} K must be float32"
                )
            expected_shape = tuple(int(value) for value in depth.shape[1:])
            ray = geometry["ray_xy"]
            ray_valid = geometry["ray_valid"]
            if ray.shape != (*expected_shape, 2) or ray.dtype != np.dtype("float32"):
                raise ValueError(
                    f"raw v1 camera {camera_name!r} ray_xy has invalid layout"
                )
            if ray_valid.shape != expected_shape or ray_valid.dtype != np.dtype("uint8"):
                raise ValueError(
                    f"raw v1 camera {camera_name!r} ray_valid has invalid layout"
                )

            residual_fields = {
                RAW_EPISODE_V2_POSITION_XY_XOR_FIELD,
                RAW_EPISODE_V2_POSITION_Z_XOR_FIELD,
            }
            present_residuals = residual_fields & set(geometry.keys())
            if schema_version == RAW_EPISODE_SCHEMA_V1:
                if present_residuals:
                    raise ValueError(
                        f"raw v1 camera {camera_name!r} unexpectedly contains "
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
