"""Lazy, CPU-only on-demand MoMap targets from canonical raw episodes.

The public result deliberately contains only training targets and alignment
metadata.  Simulator-private Position/segmentation/entity poses and robot
control state are consumed internally and are never exposed to a model-facing
caller.  The implementation is NumPy-only: DataLoader workers using this
module never create a CUDA context.
"""

from __future__ import annotations

import copy
import os
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .flow_builder import VisibilityMasks, compute_source_to_targets
from .raw_episode_hdf5 import RawEpisodeReader
from .visibility import (
    evaluate_target_visibility,
    project_future_points_to_target_image,
)


if TYPE_CHECKING:
    from .source_grid_cache import SourceGridCacheReader

GridSpec = str | int | tuple[int, int]

_SOURCE_GEOMETRY_FIELDS = (
    "position_xyz",
    "position_valid",
    "actor_seg_raw",
    "T_cam_to_world",
)
_TARGET_FLOW_GEOMETRY_FIELDS = ("T_cam_to_world",)
_TARGET_IMAGE_FLOW_GEOMETRY_FIELDS = ("K", "T_cam_to_world")
_TARGET_VISIBILITY_GEOMETRY_FIELDS = (
    "position_xyz",
    "position_valid",
    "actor_seg_raw",
    "K",
    "T_cam_to_world",
)

# ``canonical_dxyz`` is the historical target and remains the API default.
# ``source_aligned_image_flow_px`` is deliberately explicit about both the
# source-grid alignment and native-pixel unit of its first two channels.
MOMAP_REPRESENTATION_CANONICAL_DXYZ = "canonical_dxyz"
MOMAP_REPRESENTATION_SOURCE_ALIGNED_IMAGE_FLOW_PX = (
    "source_aligned_image_flow_px"
)
MOMAP_REPRESENTATIONS = frozenset(
    {
        MOMAP_REPRESENTATION_CANONICAL_DXYZ,
        MOMAP_REPRESENTATION_SOURCE_ALIGNED_IMAGE_FLOW_PX,
    }
)
_STATE_ENTITY_FIELDS = ("pose_to_world", "pose_valid")
_STATE_TIMELINE_FIELDS = ("timestamp_sec",)

_COORDINATE_METADATA_FIELDS = (
    "coordinate_convention",
    "position_coordinate",
    "position_unit",
    "position_front_axis",
    "flow_coordinate",
    "matrix_convention",
    "pixel_convention",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_representation(representation: str) -> str:
    if not isinstance(representation, str) or not representation.strip():
        raise TypeError("representation must be a non-empty string")
    normalized = representation.strip()
    if normalized not in MOMAP_REPRESENTATIONS:
        raise ValueError(
            "representation must be one of "
            f"{sorted(MOMAP_REPRESENTATIONS)}, got {representation!r}"
        )
    return normalized


@dataclass(frozen=True)
class MomapCacheKey:
    """Stable identity for a derived target; no cache backend is implied."""

    schema_version: str
    episode_descriptor_sha256: str
    camera_name: str
    source_index: int
    target_indices: tuple[int, ...]
    grid_size: tuple[int, int]
    dtype: str
    coordinate_hash: str
    reference_camera_name: str | None = None
    visibility_mode: str = "unevaluated"
    depth_tolerance_m: float | None = None
    pixel_boundary_ambiguity_px: float = 0.0
    representation: str = MOMAP_REPRESENTATION_CANONICAL_DXYZ

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_descriptor_sha256": self.episode_descriptor_sha256,
            "camera_name": self.camera_name,
            "source_index": self.source_index,
            "target_indices": list(self.target_indices),
            "grid_size": list(self.grid_size),
            "dtype": self.dtype,
            "coordinate_hash": self.coordinate_hash,
            "reference_camera_name": self.reference_camera_name,
            "visibility_mode": self.visibility_mode,
            "depth_tolerance_m": self.depth_tolerance_m,
            "pixel_boundary_ambiguity_px": self.pixel_boundary_ambiguity_px,
            "representation": self.representation,
        }

    @property
    def digest(self) -> str:
        return _sha256_json(self.as_dict())


@dataclass(frozen=True)
class TargetVisibility:
    """Target-render masks in channels-first ``[K,1,H,W]`` layout.

    With the default ``evaluate_visibility=False``, ``evaluated`` is entirely
    false.  The other false masks must therefore not be interpreted as a
    visibility classification.
    """

    evaluated: np.ndarray
    sampling_ambiguous: np.ndarray
    visible: np.ndarray
    occluded: np.ndarray
    out_of_frame: np.ndarray
    depth_conflict: np.ndarray


@dataclass(frozen=True)
class OnDemandMomapTarget:
    """Supervision result without raw simulator-private or action fields."""

    momap: np.ndarray
    geom_valid: np.ndarray
    component_valid: np.ndarray
    unsupported: np.ndarray
    unmapped: np.ndarray
    visibility: TargetVisibility
    frame_offsets: np.ndarray
    dt_seconds: np.ndarray
    source_index: int
    target_indices: np.ndarray
    camera_name: str
    grid_size: tuple[int, int]
    representation: str
    cache_key: MomapCacheKey
    source_gt_background: np.ndarray | None = None


def _parse_grid_size(grid: GridSpec, image_shape: tuple[int, int]) -> tuple[int, int]:
    image_height, image_width = image_shape
    if isinstance(grid, str):
        normalized = grid.strip().lower()
        if normalized == "full":
            return image_shape
        if "x" in normalized:
            parts = normalized.split("x")
            if len(parts) != 2:
                raise ValueError(
                    "grid string must be 'full', 'N', or 'HxW', "
                    f"got {grid!r}"
                )
            try:
                grid = (int(parts[0]), int(parts[1]))
            except ValueError as exc:
                raise ValueError(
                    "grid string must be 'full', 'N', or 'HxW', "
                    f"got {grid!r}"
                ) from exc
        else:
            try:
                grid = int(normalized)
            except ValueError as exc:
                raise ValueError(
                    "grid string must be 'full', 'N', or 'HxW', "
                    f"got {grid!r}"
                ) from exc

    if isinstance(grid, (bool, np.bool_)):
        raise TypeError("grid must be 'full', a positive integer, or (H,W)")
    if isinstance(grid, (int, np.integer)):
        grid_height = grid_width = int(grid)
    elif isinstance(grid, (tuple, list)) and len(grid) == 2:
        if any(isinstance(value, (bool, np.bool_)) for value in grid):
            raise TypeError("grid dimensions must be integers, not booleans")
        if not all(isinstance(value, (int, np.integer)) for value in grid):
            raise TypeError(f"grid dimensions must be integers, got {grid!r}")
        grid_height, grid_width = (int(value) for value in grid)
    else:
        raise TypeError(
            "grid must be 'full', a positive integer, or a two-element (H,W)"
        )

    if grid_height <= 0 or grid_width <= 0:
        raise ValueError(f"grid dimensions must be positive, got {(grid_height, grid_width)}")
    if grid_height > image_height or grid_width > image_width:
        raise ValueError(
            f"grid {(grid_height, grid_width)} exceeds source image {image_shape}; "
            "upsampling would duplicate source pixels"
        )
    return grid_height, grid_width


def uniform_cell_center_indices(
    image_shape: tuple[int, int],
    grid: GridSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return uniformly spaced cell-center row and column indices.

    ``grid='full'`` returns every source pixel.  A scalar creates a square
    grid, while ``(H,W)`` (or ``"HxW"``) creates a rectangular grid.  No
    resize/interpolation is performed: each cell selects its source-pixel
    centre using ``floor((cell + 0.5) * image_size / grid_size)``.
    """

    if len(image_shape) != 2:
        raise ValueError(f"image_shape must be (H,W), got {image_shape!r}")
    image_height, image_width = (int(value) for value in image_shape)
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"image dimensions must be positive, got {image_shape!r}")
    grid_height, grid_width = _parse_grid_size(
        grid, (image_height, image_width)
    )
    if (grid_height, grid_width) == (image_height, image_width):
        return (
            np.arange(image_height, dtype=np.int64),
            np.arange(image_width, dtype=np.int64),
        )
    rows = np.floor(
        (np.arange(grid_height, dtype=np.float64) + 0.5)
        * image_height
        / grid_height
    ).astype(np.int64)
    columns = np.floor(
        (np.arange(grid_width, dtype=np.float64) + 0.5)
        * image_width
        / grid_width
    ).astype(np.int64)
    return rows, columns


def _normalize_source_index(source_index: int, frame_count: int) -> int:
    if isinstance(source_index, (bool, np.bool_)) or not isinstance(
        source_index, (int, np.integer)
    ):
        raise TypeError(f"source_index must be an integer, got {source_index!r}")
    normalized = int(source_index)
    if not 0 <= normalized < frame_count:
        raise IndexError(
            f"source_index {normalized} outside episode with {frame_count} frames"
        )
    return normalized


def _normalize_offsets(
    offsets: Sequence[int], source_index: int, frame_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    try:
        values = tuple(offsets)
    except TypeError as exc:
        raise TypeError("offsets must be a non-empty sequence of integers") from exc
    if not values:
        raise ValueError("offsets must not be empty")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise TypeError(f"offsets must contain only integers, got {values!r}")
    normalized = tuple(int(value) for value in values)
    if any(value <= 0 for value in normalized):
        raise ValueError(f"offsets must be strictly positive, got {normalized}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"offsets must be unique, got {normalized}")
    targets = tuple(source_index + value for value in normalized)
    invalid = [target for target in targets if target >= frame_count]
    if invalid:
        raise IndexError(
            f"target indices {invalid} exceed episode boundary {frame_count - 1}"
        )
    return normalized, targets


def _normalize_target_indices(
    target_indices: Sequence[int], source_index: int, frame_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate explicit in-episode targets while preserving caller order.

    Explicit targets may be before or after the source.  Self-pairs are
    excluded because their trivial zero flow can hide accidental duplicate
    work in all-pairs exports.
    """

    try:
        values = tuple(target_indices)
    except TypeError as exc:
        raise TypeError(
            "target_indices must be a non-empty sequence of integers"
        ) from exc
    if not values:
        raise ValueError("target_indices must not be empty")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise TypeError(
            f"target_indices must contain only integers, got {values!r}"
        )
    normalized = tuple(int(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"target_indices must be unique, got {normalized}")
    invalid = [target for target in normalized if not 0 <= target < frame_count]
    if invalid:
        raise IndexError(
            f"target indices {invalid} outside episode [0,{frame_count - 1}]"
        )
    if source_index in normalized:
        raise ValueError(
            f"target_indices must not contain source_index {source_index}"
        )
    offsets = tuple(target - source_index for target in normalized)
    return offsets, normalized


def _output_dtype(value: str | np.dtype | type[np.floating[Any]]) -> np.dtype:
    dtype = np.dtype(value)
    if dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError(
            "target dtype must be float16 or float32; geometry is always "
            f"computed in float32, got {dtype}"
        )
    return dtype


GT_BACKGROUND_ENTITY_NAMES = ("background", "ground", "table", "wall")


def source_gt_background_mask(actor_seg: np.ndarray, registry: Mapping) -> np.ndarray:
    """Supervision only: exact episode-local identities; unknown names kept."""
    ids = [raw_id for raw_id, entry in registry.items()
           if entry.get("entity_name") in GT_BACKGROUND_ENTITY_NAMES]
    return np.isin(actor_seg, ids)


def _actor_registry(entries: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if str(entry.get("seg_channel")) != "actor_id":
            continue
        raw_seg_id = int(entry["raw_seg_id"])
        if raw_seg_id in result:
            raise ValueError(f"duplicate actor_id registry entry {raw_seg_id}")
        result[raw_seg_id] = dict(entry)
    if not result:
        raise ValueError("episode registry contains no actor_id entries")
    return result


def _episode_descriptor(metadata: Mapping[str, Any], path: Path) -> dict[str, Any]:
    provenance_keys = (
        "dataset_version",
        "source_dataset",
        "source_episode",
        "source_hashes",
        "task_name",
        "task_config",
        "task_config_sha256",
        "episode_id",
        "seed",
        "left_joint_path_hash",
        "right_joint_path_hash",
        "task_config_hash",
        "assets_hash",
        "robotwin_commit",
        "fastwam_commit",
        "exporter_commit",
    )
    # Hash complete canonical metadata as well as common provenance fields.
    # Thus two replays sharing task/config/seed but carrying different source
    # hashes or code revisions cannot silently share cached targets.  The path
    # is a legacy safeguard; canonical producers should emit source_hashes and
    # commit/config hashes so identity remains meaningful across copies.
    return {
        "provenance": {
            key: metadata[key] for key in provenance_keys if key in metadata
        },
        "canonical_metadata_hash": _sha256_json(dict(metadata)),
        "episode_path": str(path.resolve()),
    }


def episode_descriptor_sha256(metadata: Mapping[str, Any], path: str | Path) -> str:
    """Hash logical canonical identity: metadata provenance plus absolute path.

    This is intentionally distinct from the canonical HDF5 file SHA-256.
    """

    return _sha256_json(_episode_descriptor(metadata, Path(path)))


def _coordinate_descriptor(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: metadata[key]
        for key in _COORDINATE_METADATA_FIELDS
        if key in metadata
    }
    if not result:
        result["coordinate_convention"] = "unspecified"
    return result


def _sample_grid(array: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim < 2:
        raise ValueError(f"grid source must have at least two dimensions, got {value.shape}")
    return value[rows[:, None], columns[None, :], ...]


def _camera_geometry(
    frame: Mapping[str, Any],
    camera_name: str,
    required_fields: Sequence[str],
) -> Mapping[str, np.ndarray]:
    try:
        geometry = frame["private_geometry"][camera_name]
    except KeyError as exc:
        raise KeyError(f"frame does not contain private geometry for {camera_name!r}") from exc
    missing = set(required_fields) - set(geometry)
    if missing:
        raise KeyError(
            f"camera {camera_name!r} is missing canonical geometry fields: "
            f"{sorted(missing)}"
        )
    return geometry


class MomapTargetService:
    """Bind one canonical episode to a lazy, fork-safe CPU target service."""

    def __init__(
        self,
        episode: str | PathLike[str] | RawEpisodeReader,
        *,
        require_complete: bool = True,
        frame_cache_size: int = 32,
        source_grid_cache: str | PathLike[str] | SourceGridCacheReader | None = None,
    ) -> None:
        if isinstance(episode, RawEpisodeReader):
            self._reader = episode
            self._owns_reader = False
        else:
            self._reader = RawEpisodeReader(
                Path(episode), require_complete=require_complete
            )
            self._owns_reader = True
        if source_grid_cache is None:
            self._source_grid_cache = None
            self._owns_source_grid_cache = False
        else:
            from .source_grid_cache import SourceGridCacheReader

            if isinstance(source_grid_cache, SourceGridCacheReader):
                self._source_grid_cache = source_grid_cache
                self._owns_source_grid_cache = False
            else:
                self._source_grid_cache = SourceGridCacheReader(
                    Path(source_grid_cache), self._reader.path
                )
                self._owns_source_grid_cache = True
        self._metadata: dict[str, Any] | None = None
        self._registry: dict[int, dict[str, Any]] | None = None
        self._episode_descriptor_sha256: str | None = None
        self._coordinate_hash: str | None = None
        if (
            isinstance(frame_cache_size, (bool, np.bool_))
            or not isinstance(frame_cache_size, (int, np.integer))
            or int(frame_cache_size) < 0
        ):
            raise ValueError("frame_cache_size must be a non-negative integer")
        self._motion_cache = None
        self._frame_cache_size = int(frame_cache_size)
        self._state_cache: OrderedDict[int, dict[str, dict[str, Any]]] = OrderedDict()
        self._camera_cache: OrderedDict[
            tuple[str, int, tuple[int, int] | None], dict[str, np.ndarray]
        ] = OrderedDict()

    def _trim_frame_caches(self) -> None:
        while len(self._state_cache) > self._frame_cache_size:
            self._state_cache.popitem(last=False)
        while len(self._camera_cache) > self._frame_cache_size:
            self._camera_cache.popitem(last=False)

    def _cached_frame(
        self,
        index: int,
        camera_name: str,
        geometry_fields: Sequence[str],
        *,
        source_grid: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Read and cache only missing state/camera fields for one frame."""

        state = self._state_cache.get(
            index, {"timeline": {}, "entities": {}}
        )
        camera_key = (camera_name, index, source_grid)
        camera_geometry = self._camera_cache.get(camera_key, {})
        missing_geometry = tuple(
            field for field in geometry_fields if field not in camera_geometry
        )
        missing_timeline = tuple(
            field
            for field in _STATE_TIMELINE_FIELDS
            if field not in state["timeline"]
        )
        missing_entities = tuple(
            field
            for field in _STATE_ENTITY_FIELDS
            if field not in state["entities"]
        )
        if self._source_grid_cache is not None:
            if missing_timeline or missing_entities:
                cached_state = self._source_grid_cache.read_state(index)
                state["timeline"].update(cached_state["timeline"])
                state["entities"].update(cached_state["entities"])
                missing_timeline = tuple(
                    field
                    for field in _STATE_TIMELINE_FIELDS
                    if field not in state["timeline"]
                )
                missing_entities = tuple(
                    field
                    for field in _STATE_ENTITY_FIELDS
                    if field not in state["entities"]
                )
            if (
                source_grid is not None
                and self._source_grid_cache.has_grid(camera_name, source_grid)
                and missing_geometry
            ):
                cached_geometry = self._source_grid_cache.read_source_grid(
                    camera_name, index, source_grid
                )
                camera_geometry.update(
                    {
                        field: cached_geometry[field]
                        for field in missing_geometry
                        if field in cached_geometry
                    }
                )
                missing_geometry = tuple(
                    field for field in geometry_fields if field not in camera_geometry
                )
            elif (
                "T_cam_to_world" in missing_geometry
                and self._source_grid_cache.available_grids(camera_name)
            ):
                camera_geometry["T_cam_to_world"] = (
                    self._source_grid_cache.read_camera_pose(camera_name, index)
                )
                missing_geometry = tuple(
                    field for field in geometry_fields if field not in camera_geometry
                )

        if missing_geometry or missing_timeline or missing_entities:
            projected = self._reader.read_frame_fields(
                index,
                camera_names=(camera_name,) if missing_geometry else (),
                observation_fields=(),
                private_geometry_fields=missing_geometry,
                timeline_fields=missing_timeline,
                robot_state_fields=(),
                entity_fields=missing_entities,
            )
            state["timeline"].update(projected["timeline"])
            state["entities"].update(projected["entities"])
            if missing_geometry:
                camera_geometry.update(
                    projected["private_geometry"][camera_name]
                )

        if self._frame_cache_size:
            self._state_cache[index] = state
            self._state_cache.move_to_end(index)
            self._camera_cache[camera_key] = camera_geometry
            self._camera_cache.move_to_end(camera_key)
            self._trim_frame_caches()
        return {
            "observations": {camera_name: {}},
            "private_geometry": {camera_name: camera_geometry},
            "timeline": state["timeline"],
            "robot_state": {},
            "entities": state["entities"],
        }

    @property
    def is_open(self) -> bool:
        return self._reader.is_open

    @property
    def episode_path(self) -> Path:
        return self._reader.path

    def _context(
        self,
    ) -> tuple[dict[str, Any], dict[int, dict[str, Any]], str, str]:
        if self._metadata is None:
            self._metadata = self._reader.metadata
        if self._registry is None:
            self._registry = _actor_registry(self._reader.entity_registry)
        if self._episode_descriptor_sha256 is None:
            self._episode_descriptor_sha256 = episode_descriptor_sha256(self._metadata, self._reader.path)
        if self._coordinate_hash is None:
            self._coordinate_hash = _sha256_json(
                {
                    "coordinate_descriptor": _coordinate_descriptor(self._metadata),
                    "raw_schema_version": self._reader.schema_version,
                    "geometry_encoding": self._reader.geometry_encoding,
                    "geometry_contract_sha256": self._reader.geometry_contract_hash,
                }
            )
        return (
            self._metadata,
            self._registry,
            self._episode_descriptor_sha256,
            self._coordinate_hash,
        )

    def get_momap(
        self,
        camera_name: str,
        source_index: int,
        offsets: Sequence[int],
        grid: GridSpec = "full",
        *,
        dtype: str | np.dtype | type[np.floating[Any]] = np.float32,
        evaluate_visibility: bool = False,
        depth_tolerance_m: float = 0.005,
        pixel_boundary_ambiguity_px: float = 0.0,
        reference_camera_name: str | None = None,
        representation: str = MOMAP_REPRESENTATION_CANONICAL_DXYZ,
    ) -> OnDemandMomapTarget:
        """Construct source-aligned GT flow for positive future offsets.

        This training-oriented API intentionally keeps its historical
        positive-offset contract.  Use :meth:`get_momap_between` for explicit
        earlier or later target frames.  ``reference_camera_name=None`` keeps
        the historical per-view source-camera coordinates.  Supplying a
        source-time camera rotates the final displacement vectors into that
        common frame without applying translation.  The default
        ``representation='canonical_dxyz'`` preserves that historical XYZ
        target exactly.  ``'source_aligned_image_flow_px'`` instead returns
        native-pixel target-image displacements on the source grid.
        """

        frame_count = self._reader.num_frames
        source_index = _normalize_source_index(source_index, frame_count)
        normalized_offsets, target_indices = _normalize_offsets(
            offsets, source_index, frame_count
        )
        return self._build_momap_for_targets(
            camera_name,
            source_index,
            normalized_offsets,
            target_indices,
            grid,
            dtype=dtype,
            evaluate_visibility=evaluate_visibility,
            depth_tolerance_m=depth_tolerance_m,
            pixel_boundary_ambiguity_px=pixel_boundary_ambiguity_px,
            reference_camera_name=reference_camera_name,
            representation=representation,
        )

    def get_momap_between(
        self,
        camera_name: str,
        source_index: int,
        target_indices: Sequence[int],
        grid: GridSpec = "full",
        *,
        dtype: str | np.dtype | type[np.floating[Any]] = np.float32,
        evaluate_visibility: bool = False,
        depth_tolerance_m: float = 0.005,
        pixel_boundary_ambiguity_px: float = 0.0,
        reference_camera_name: str | None = None,
        representation: str = MOMAP_REPRESENTATION_CANONICAL_DXYZ,
    ) -> OnDemandMomapTarget:
        """Construct source-aligned GT flow to explicit in-episode targets.

        Targets may lie before or after ``source_index`` and are returned in
        caller order.  With ``reference_camera_name=None``, every flow remains
        expressed in the source view's camera coordinates.  Supplying a
        reference camera rotates it into that camera's source-time axes.  A
        reverse pair is recomputed from that reverse source and is not merely
        the negation of a forward image-space field. Self-pairs and duplicate
        targets are rejected.
        """

        frame_count = self._reader.num_frames
        source_index = _normalize_source_index(source_index, frame_count)
        frame_offsets, normalized_targets = _normalize_target_indices(
            target_indices, source_index, frame_count
        )
        return self._build_momap_for_targets(
            camera_name,
            source_index,
            frame_offsets,
            normalized_targets,
            grid,
            dtype=dtype,
            evaluate_visibility=evaluate_visibility,
            depth_tolerance_m=depth_tolerance_m,
            pixel_boundary_ambiguity_px=pixel_boundary_ambiguity_px,
            reference_camera_name=reference_camera_name,
            representation=representation,
        )

    def _build_momap_for_targets(
        self,
        camera_name: str,
        source_index: int,
        normalized_offsets: tuple[int, ...],
        target_indices: tuple[int, ...],
        grid: GridSpec,
        *,
        dtype: str | np.dtype | type[np.floating[Any]],
        evaluate_visibility: bool,
        depth_tolerance_m: float,
        pixel_boundary_ambiguity_px: float,
        reference_camera_name: str | None,
        representation: str,
    ) -> OnDemandMomapTarget:
        """Shared implementation for positive-offset and explicit-pair APIs."""

        if camera_name not in self._reader.camera_names:
            raise KeyError(
                f"unknown camera {camera_name!r}; available={self._reader.camera_names}"
            )
        representation = _normalize_representation(representation)
        if reference_camera_name is not None:
            if (
                not isinstance(reference_camera_name, str)
                or not reference_camera_name.strip()
            ):
                raise TypeError(
                    "reference_camera_name must be a non-empty string or None"
                )
            if reference_camera_name not in self._reader.camera_names:
                raise KeyError(
                    f"unknown reference camera {reference_camera_name!r}; "
                    f"available={self._reader.camera_names}"
                )
        output_dtype = _output_dtype(dtype)
        if not isinstance(evaluate_visibility, (bool, np.bool_)):
            raise TypeError("evaluate_visibility must be boolean")
        if evaluate_visibility and (
            not np.isfinite(depth_tolerance_m) or depth_tolerance_m < 0
        ):
            raise ValueError("depth_tolerance_m must be finite and non-negative")
        if (
            not np.isfinite(pixel_boundary_ambiguity_px)
            or pixel_boundary_ambiguity_px < 0
        ):
            raise ValueError(
                "pixel_boundary_ambiguity_px must be finite and non-negative"
            )

        metadata, registry, episode_descriptor_sha256, coordinate_hash = self._context()
        cached_source_grid: tuple[int, int] | None = None
        cached_source_image_shape: tuple[int, int] | None = None
        if self._source_grid_cache is not None:
            available = self._source_grid_cache.available_grids(camera_name)
            if available:
                cached_source_image_shape = self._source_grid_cache.source_image_shape(
                    camera_name
                )
                cached_rows, cached_columns = uniform_cell_center_indices(
                    cached_source_image_shape, grid
                )
                candidate = (len(cached_rows), len(cached_columns))
                if self._source_grid_cache.has_grid(camera_name, candidate):
                    cached_source_grid = candidate
        source_frame = self._cached_frame(
            source_index,
            camera_name,
            _SOURCE_GEOMETRY_FIELDS,
            source_grid=cached_source_grid,
        )
        target_fields = (
            _TARGET_VISIBILITY_GEOMETRY_FIELDS
            if evaluate_visibility
            else (
                _TARGET_IMAGE_FLOW_GEOMETRY_FIELDS
                if representation
                == MOMAP_REPRESENTATION_SOURCE_ALIGNED_IMAGE_FLOW_PX
                else _TARGET_FLOW_GEOMETRY_FIELDS
            )
        )
        target_frames = [
            self._cached_frame(index, camera_name, target_fields)
            for index in target_indices
        ]
        source_geometry = _camera_geometry(
            source_frame, camera_name, _SOURCE_GEOMETRY_FIELDS
        )
        reference_camera_pose: np.ndarray | None = None
        if representation == MOMAP_REPRESENTATION_CANONICAL_DXYZ:
            if reference_camera_name is None or reference_camera_name == camera_name:
                reference_camera_pose = np.asarray(
                    source_geometry["T_cam_to_world"], dtype=np.float32
                )
            else:
                reference_frame = self._cached_frame(
                    source_index,
                    reference_camera_name,
                    ("T_cam_to_world",),
                )
                reference_camera_pose = np.asarray(
                    _camera_geometry(
                        reference_frame,
                        reference_camera_name,
                        ("T_cam_to_world",),
                    )["T_cam_to_world"],
                    dtype=np.float32,
                )
        target_geometry = [
            _camera_geometry(frame, camera_name, target_fields)
            for frame in target_frames
        ]

        source_position = np.asarray(source_geometry["position_xyz"])
        if source_position.ndim != 3 or source_position.shape[-1] < 3:
            raise ValueError(
                f"source Position must be [H,W,>=3], got {source_position.shape}"
            )
        if cached_source_grid is not None:
            image_shape = cached_source_image_shape
            assert image_shape is not None
            grid_size = cached_source_grid
            source_rows = cached_rows
            source_columns = cached_columns
            sampled_position = source_position
            sampled_valid = np.asarray(source_geometry["position_valid"]).astype(
                bool, copy=False
            )
            sampled_actor_seg = np.asarray(source_geometry["actor_seg_raw"])
        else:
            image_shape = tuple(int(value) for value in source_position.shape[:2])
            rows, columns = uniform_cell_center_indices(image_shape, grid)
            grid_size = (len(rows), len(columns))
            source_rows = rows
            source_columns = columns
            sampled_position = _sample_grid(source_position, rows, columns)
            sampled_valid = _sample_grid(
                np.asarray(source_geometry["position_valid"]), rows, columns
            ).astype(bool, copy=False)
            sampled_actor_seg = _sample_grid(
                np.asarray(source_geometry["actor_seg_raw"]), rows, columns
            )

        source_entity_poses = np.asarray(
            source_frame["entities"]["pose_to_world"], dtype=np.float32
        )
        source_entity_valid = np.asarray(
            source_frame["entities"]["pose_valid"], dtype=bool
        )
        target_entity_poses = np.stack(
            [frame["entities"]["pose_to_world"] for frame in target_frames]
        ).astype(np.float32, copy=False)
        target_entity_valid = np.stack(
            [frame["entities"]["pose_valid"] for frame in target_frames]
        ).astype(bool, copy=False)
        target_camera_poses = np.stack(
            [geometry["T_cam_to_world"] for geometry in target_geometry]
        ).astype(np.float32, copy=False)

        # UV eligibility and metric XYZ supervision share the same source-space
        # rigid trajectories. Keep only the most recent computation, per service.
        # Snapshots prevent caller mutation of a returned target from corrupting
        # the next representation; PID is part of the key for fork isolation.
        motion_key = (os.getpid(), camera_name, source_index, target_indices,
                      grid_size, coordinate_hash)
        if self._motion_cache is not None and self._motion_cache[0] == motion_key:
            flow_result = copy.deepcopy(self._motion_cache[1])
        else:
            flow_result = compute_source_to_targets(
                source_position=sampled_position,
                source_segmentation=sampled_actor_seg,
                T_source_camera_to_world=source_geometry["T_cam_to_world"],
                source_entity_poses=source_entity_poses,
                target_entity_poses=target_entity_poses,
                registry=registry,
                source_position_valid=sampled_valid,
                source_entity_pose_valid=source_entity_valid,
                target_entity_pose_valid=target_entity_valid,
                T_target_camera_to_world=target_camera_poses,
            )
            if self._frame_cache_size:
                self._motion_cache = (motion_key, copy.deepcopy(flow_result))

        visibility: VisibilityMasks | Any = flow_result.visibility
        visibility_mode = "unevaluated"
        cache_tolerance: float | None = None
        coordinate_convention: str | None = None
        if (
            evaluate_visibility
            or representation == MOMAP_REPRESENTATION_SOURCE_ALIGNED_IMAGE_FLOW_PX
        ):
            convention = metadata.get("coordinate_convention")
            if convention not in {"sapien_opengl_camera", "opencv_camera"}:
                raise ValueError(
                    "target projection requires metadata.coordinate_convention to be "
                    "'sapien_opengl_camera' or 'opencv_camera', "
                    f"got {convention!r}"
                )
            coordinate_convention = str(convention)
        if evaluate_visibility:
            assert coordinate_convention is not None
            visibility = evaluate_target_visibility(
                flow_result.future_xyz_ref,
                target_camera_poses,
                np.stack([geometry["K"] for geometry in target_geometry]),
                target_position=np.stack(
                    [geometry["position_xyz"] for geometry in target_geometry]
                ),
                target_valid=np.stack(
                    [geometry["position_valid"] for geometry in target_geometry]
                ),
                source_entity_ids=sampled_actor_seg,
                target_actor_seg=np.stack(
                    [geometry["actor_seg_raw"] for geometry in target_geometry]
                ),
                geom_valid=flow_result.geom_valid,
                future_coordinate="source_camera",
                T_source_camera_to_world=source_geometry["T_cam_to_world"],
                coordinate_convention=coordinate_convention,
                depth_tolerance_m=float(depth_tolerance_m),
                pixel_boundary_ambiguity_px=float(pixel_boundary_ambiguity_px),
            )
            visibility_mode = "target_zbuffer"
            cache_tolerance = float(depth_tolerance_m)

        source_timestamp = float(source_frame["timeline"]["timestamp_sec"])
        target_timestamps = np.asarray(
            [frame["timeline"]["timestamp_sec"] for frame in target_frames],
            dtype=np.float64,
        )
        dt_seconds = target_timestamps - source_timestamp
        offsets_array = np.asarray(normalized_offsets, dtype=np.int64)
        order_mismatch = ((offsets_array > 0) & (dt_seconds < 0)) | (
            (offsets_array < 0) & (dt_seconds > 0)
        )
        if (
            not np.isfinite(dt_seconds).all()
            or np.any(order_mismatch)
        ):
            raise ValueError(
                "source/target frame order requires finite timeline timestamps "
                "with nondecreasing frame order; distinct saved frames at the "
                "same simulator step may legitimately have dt_seconds=0; "
                f"offsets={list(normalized_offsets)}, "
                f"dt_seconds={dt_seconds.tolist()}"
            )

        cache_key = MomapCacheKey(
            schema_version=self._reader.schema_version,
            episode_descriptor_sha256=episode_descriptor_sha256,
            camera_name=camera_name,
            source_index=source_index,
            target_indices=target_indices,
            grid_size=grid_size,
            dtype=output_dtype.name,
            coordinate_hash=coordinate_hash,
            reference_camera_name=reference_camera_name,
            visibility_mode=visibility_mode,
            depth_tolerance_m=cache_tolerance,
            pixel_boundary_ambiguity_px=float(pixel_boundary_ambiguity_px),
            representation=representation,
        )
        output_geom_valid = flow_result.geom_valid
        if representation == MOMAP_REPRESENTATION_CANONICAL_DXYZ:
            output_flow = flow_result.flow_xyz_ref
            if reference_camera_name is not None:
                # Flow is a displacement, so only the relative camera rotation
                # is applied. Translation cancels and stationary points remain
                # exactly zero. This is the historical default path.
                assert reference_camera_pose is not None
                source_camera_pose = np.asarray(
                    source_geometry["T_cam_to_world"], dtype=np.float32
                )
                view_to_reference = (
                    source_camera_pose[:3, :3].T
                    @ reference_camera_pose[:3, :3]
                )
                output_flow = np.einsum(
                    "...i,ij->...j",
                    flow_result.flow_xyz_ref,
                    view_to_reference,
                    optimize=True,
                )
            component_valid = np.broadcast_to(
                output_geom_valid[:, None],
                (len(target_indices), 3, *grid_size),
            )
        else:
            assert coordinate_convention is not None
            projection = project_future_points_to_target_image(
                flow_result.future_xyz_ref,
                target_camera_poses,
                np.stack([geometry["K"] for geometry in target_geometry]),
                target_image_shape=image_shape,
                future_coordinate="source_camera",
                T_source_camera_to_world=source_geometry["T_cam_to_world"],
                coordinate_convention=coordinate_convention,
            )
            source_u, source_v = np.meshgrid(
                source_columns.astype(np.float32, copy=False),
                source_rows.astype(np.float32, copy=False),
            )
            source_foreground = np.asarray(sampled_actor_seg) != 0
            image_flow_valid = (
                flow_result.geom_valid
                & source_foreground[None]
                & projection.in_frame
            )
            delta_uv = projection.uv - np.stack((source_u, source_v), axis=-1)[None]
            output_flow = np.full_like(flow_result.flow_xyz_ref, np.nan)
            output_flow[..., 0] = np.where(
                image_flow_valid,
                delta_uv[..., 0],
                np.nan,
            )
            output_flow[..., 1] = np.where(
                image_flow_valid,
                delta_uv[..., 1],
                np.nan,
            )
            output_flow[..., 2] = np.where(image_flow_valid, 0.0, np.nan)
            component_valid = np.zeros(
                (len(target_indices), 3, *grid_size),
                dtype=bool,
            )
            component_valid[:, 0] = image_flow_valid
            component_valid[:, 1] = image_flow_valid

        return OnDemandMomapTarget(
            momap=np.ascontiguousarray(
                np.moveaxis(output_flow, -1, 1),
                dtype=output_dtype,
            ),
            geom_valid=np.ascontiguousarray(output_geom_valid[:, None]),
            component_valid=np.ascontiguousarray(component_valid),
            unsupported=np.ascontiguousarray(
                np.broadcast_to(
                    flow_result.unsupported[None, None],
                    (len(target_indices), 1, *grid_size),
                )
            ),
            unmapped=np.ascontiguousarray(
                np.broadcast_to(
                    flow_result.unmapped[None, None],
                    (len(target_indices), 1, *grid_size),
                )
            ),
            visibility=TargetVisibility(
                evaluated=np.ascontiguousarray(visibility.evaluated[:, None]),
                sampling_ambiguous=np.ascontiguousarray(
                    (
                        visibility.sampling_ambiguous
                        if evaluate_visibility
                        else np.zeros_like(visibility.evaluated)
                    )[:, None]
                ),
                visible=np.ascontiguousarray(visibility.visible[:, None]),
                occluded=np.ascontiguousarray(visibility.occluded[:, None]),
                out_of_frame=np.ascontiguousarray(
                    visibility.out_of_frame[:, None]
                ),
                depth_conflict=np.ascontiguousarray(
                    (
                        visibility.evaluated
                        & ~(
                            visibility.visible
                            | visibility.occluded
                            | visibility.out_of_frame
                        )
                    )[:, None]
                ),
            ),
            frame_offsets=np.asarray(normalized_offsets, dtype=np.int64),
            dt_seconds=dt_seconds,
            source_index=source_index,
            target_indices=np.asarray(target_indices, dtype=np.int64),
            camera_name=camera_name,
            grid_size=grid_size,
            representation=representation,
            cache_key=cache_key,
            source_gt_background=source_gt_background_mask(sampled_actor_seg, registry),
        )

    def close(self) -> None:
        self._motion_cache = None
        self._state_cache.clear()
        self._camera_cache.clear()
        if self._owns_source_grid_cache and self._source_grid_cache is not None:
            self._source_grid_cache.close()
        if self._owns_reader:
            self._reader.close()

    def __enter__(self) -> "MomapTargetService":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        self.close()
        return False


def get_momap(
    episode: str | PathLike[str] | RawEpisodeReader | MomapTargetService,
    camera_name: str,
    source_index: int,
    offsets: Sequence[int],
    grid: GridSpec = "full",
    **kwargs: Any,
) -> OnDemandMomapTarget:
    """One-shot convenience wrapper around :class:`MomapTargetService`."""

    if isinstance(episode, MomapTargetService):
        return episode.get_momap(
            camera_name, source_index, offsets, grid, **kwargs
        )
    service = MomapTargetService(episode)
    try:
        return service.get_momap(
            camera_name, source_index, offsets, grid, **kwargs
        )
    finally:
        service.close()


def get_momap_between(
    episode: str | PathLike[str] | RawEpisodeReader | MomapTargetService,
    camera_name: str,
    source_index: int,
    target_indices: Sequence[int],
    grid: GridSpec = "full",
    **kwargs: Any,
) -> OnDemandMomapTarget:
    """One-shot explicit-pair wrapper supporting earlier and later targets."""

    if isinstance(episode, MomapTargetService):
        return episode.get_momap_between(
            camera_name, source_index, target_indices, grid, **kwargs
        )
    service = MomapTargetService(episode)
    try:
        return service.get_momap_between(
            camera_name, source_index, target_indices, grid, **kwargs
        )
    finally:
        service.close()


