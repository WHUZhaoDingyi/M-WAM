"""Vectorized piecewise-rigid GT 3D flow construction.

This module deliberately operates only on exported arrays.  It does not import
RoboTwin or SAPIEN, so a raw replay can be converted into flow targets on a CPU
machine.  Flow is Lagrangian and source-pixel aligned: future points are always
expressed in the *source* camera coordinate system before subtraction.

The production implementation maps raw segmentation IDs to dense pose rows and
then transforms all pixels/targets in tensor batches.  It never loops over
pixels or entities; an optional loop over coarse point chunks only bounds peak
working memory.  The former entity-grouped implementation remains available as
an explicit numerical reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .se3 import require_se3


STATIC_ENTITY_TYPES = frozenset({"background", "static"})
SUPPORTED_DYNAMIC_ENTITY_TYPES = frozenset(
    {"rigid_actor", "articulation_link", "robot_link"}
)


@dataclass(frozen=True)
class FlowEntity:
    """Episode-local mapping from a raw segmentation ID to one pose row.

    ``pose_row`` indexes dense ``T_entity_to_world`` arrays.  Static world
    entities do not need a pose row.  Raw renderer IDs are intentionally not
    treated as cross-episode semantic IDs.
    """

    entity_index: int
    entity_type: str
    pose_row: int | None
    name: str = ""
    supported: bool = True

    @property
    def is_static(self) -> bool:
        return self.entity_type in STATIC_ENTITY_TYPES


@dataclass
class VisibilityMasks:
    """Target-render visibility state for a flow batch.

    The geometry builder does not have target depth/segmentation, so it returns
    an explicitly *unevaluated* structure.  Consumers must consult ``evaluated``
    and must not interpret the other all-false masks as "not visible".
    """

    evaluated: np.ndarray
    visible: np.ndarray
    occluded: np.ndarray
    out_of_frame: np.ndarray

    @classmethod
    def unevaluated(cls, shape: tuple[int, ...]) -> "VisibilityMasks":
        return cls(
            evaluated=np.zeros(shape, dtype=bool),
            visible=np.zeros(shape, dtype=bool),
            occluded=np.zeros(shape, dtype=bool),
            out_of_frame=np.zeros(shape, dtype=bool),
        )


@dataclass
class FlowBatchResult:
    """Flow from one source frame to multiple target frames.

    Shapes use ``K`` target frames followed by the source spatial dimensions.
    Invalid ``flow_xyz_ref`` and ``future_xyz_ref`` entries are NaN on purpose;
    callers must apply ``geom_valid`` rather than accidentally treating an
    unsupported or unmapped point as zero motion.
    """

    flow_xyz_ref: np.ndarray
    future_xyz_ref: np.ndarray
    geom_valid: np.ndarray
    source_entity_index: np.ndarray
    unmapped: np.ndarray
    unsupported: np.ndarray
    visibility: VisibilityMasks


def inverse_se3(T: np.ndarray) -> np.ndarray:
    """Analytically invert one or a batch of rigid 4x4 transforms.

    Matrices follow the column-vector convention ``x_out = T @ x_in``.  The
    returned dtype is float32; this routine never performs a generic matrix
    inverse and never downcasts the computation to float16.
    """

    matrix = require_se3(T, "T")

    rotation_t = np.swapaxes(matrix[..., :3, :3], -1, -2)
    translation = matrix[..., :3, 3]
    result = np.zeros_like(matrix, dtype=np.float32)
    result[..., :3, :3] = rotation_t
    result[..., :3, 3] = -np.einsum(
        "...ij,...j->...i", rotation_t, translation, optimize=True
    )
    result[..., 3, 3] = 1.0
    return result


def transform_points_se3(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Transform points with one SE(3) matrix, preserving point dimensions."""

    matrix = np.asarray(T, dtype=np.float32)
    points = np.asarray(xyz, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"T must have shape [4,4], got {matrix.shape}")
    if points.shape[-1:] != (3,):
        raise ValueError(f"xyz must end in [3], got {points.shape}")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _normalize_registry_entry(value: FlowEntity | Mapping[str, object]) -> FlowEntity:
    if isinstance(value, FlowEntity):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "registry values must be FlowEntity or mappings, "
            f"got {type(value).__name__}"
        )
    entity_type = str(value.get("entity_type", ""))
    pose_row_value = value.get("pose_row")
    pose_row = None if pose_row_value is None else int(pose_row_value)
    entity_index_value = value.get("entity_index", pose_row if pose_row is not None else -1)
    return FlowEntity(
        entity_index=int(entity_index_value),
        entity_type=entity_type,
        pose_row=pose_row,
        name=str(value.get("entity_name", value.get("name", ""))),
        supported=bool(value.get("supported", True)),
    )


def _normalize_registry(
    registry: Mapping[int | str, FlowEntity | Mapping[str, object]],
) -> dict[int, FlowEntity]:
    normalized: dict[int, FlowEntity] = {}
    for raw_seg_id, value in registry.items():
        normalized[int(raw_seg_id)] = _normalize_registry_entry(value)
    return normalized


def _pose_validity(poses: np.ndarray) -> np.ndarray:
    return np.isfinite(poses).all(axis=(-2, -1))


def compute_source_to_targets(
    source_position: np.ndarray,
    source_segmentation: np.ndarray,
    T_source_camera_to_world: np.ndarray,
    source_entity_poses: np.ndarray,
    target_entity_poses: np.ndarray,
    registry: Mapping[int | str, FlowEntity | Mapping[str, object]],
    *,
    source_position_valid: np.ndarray | None = None,
    source_entity_pose_valid: np.ndarray | None = None,
    target_entity_pose_valid: np.ndarray | None = None,
    T_target_camera_to_world: np.ndarray | None = None,
) -> FlowBatchResult:
    """Compute source-reference 3D flow to several target frames.

    Args:
        source_position: Source camera-space XYZ, shape ``[*S, >=3]``.
        source_segmentation: Episode-local raw IDs, shape ``[*S]``.
        T_source_camera_to_world: Source camera pose, shape ``[4,4]``.
        source_entity_poses: Entity-to-world poses, shape ``[N,4,4]``.
        target_entity_poses: Target entity-to-world poses, shape ``[K,N,4,4]``.
        registry: Raw segmentation ID to :class:`FlowEntity` mapping.
        source_position_valid: Optional explicit validity mask, shape ``[*S]``.
        source_entity_pose_valid: Optional pose mask, shape ``[N]``.
        target_entity_pose_valid: Optional pose mask, shape ``[K,N]``.
        T_target_camera_to_world: Optional ``[K,4,4]`` target camera poses.
            Geometry flow does not use them because output coordinates are fixed
            to the source camera.  They are validated here for the subsequent
            visibility stage, which projects into target cameras.

    Returns:
        A :class:`FlowBatchResult` with arrays shaped ``[K,*S,...]``.
    """

    position = np.asarray(source_position, dtype=np.float32)
    segmentation = np.asarray(source_segmentation)
    source_camera_pose = np.asarray(T_source_camera_to_world, dtype=np.float32)
    source_poses = np.asarray(source_entity_poses, dtype=np.float32)
    target_poses = np.asarray(target_entity_poses, dtype=np.float32)

    if position.ndim < 2 or position.shape[-1] < 3:
        raise ValueError(f"source_position must be [*S,>=3], got {position.shape}")
    spatial_shape = position.shape[:-1]
    if segmentation.shape != spatial_shape:
        raise ValueError(
            "source_segmentation shape must match source_position spatial shape, "
            f"got {segmentation.shape} and {spatial_shape}"
        )
    if source_camera_pose.shape != (4, 4):
        raise ValueError(
            "T_source_camera_to_world must be [4,4], "
            f"got {source_camera_pose.shape}"
        )
    source_camera_pose = require_se3(
        source_camera_pose, "T_source_camera_to_world"
    )
    if source_poses.ndim != 3 or source_poses.shape[-2:] != (4, 4):
        raise ValueError(
            f"source_entity_poses must be [N,4,4], got {source_poses.shape}"
        )
    if target_poses.ndim != 4 or target_poses.shape[-2:] != (4, 4):
        raise ValueError(
            f"target_entity_poses must be [K,N,4,4], got {target_poses.shape}"
        )
    if target_poses.shape[1] != source_poses.shape[0]:
        raise ValueError(
            "source/target entity counts differ: "
            f"{source_poses.shape[0]} vs {target_poses.shape[1]}"
        )

    target_count, entity_count = target_poses.shape[:2]
    if T_target_camera_to_world is not None:
        target_camera_poses = np.asarray(T_target_camera_to_world, dtype=np.float32)
        if target_camera_poses.shape != (target_count, 4, 4):
            raise ValueError(
                "T_target_camera_to_world must be [K,4,4], "
                f"got {target_camera_poses.shape}"
            )
        target_camera_poses = require_se3(
            target_camera_poses, "T_target_camera_to_world"
        )

    xyz_source = position[..., :3]
    flat_xyz = xyz_source.reshape(-1, 3)
    flat_segmentation = segmentation.reshape(-1)
    flat_position_valid = np.isfinite(flat_xyz).all(axis=-1)
    if source_position_valid is not None:
        explicit_position_valid = np.asarray(source_position_valid, dtype=bool)
        if explicit_position_valid.shape != spatial_shape:
            raise ValueError(
                "source_position_valid shape must match source spatial shape, "
                f"got {explicit_position_valid.shape} and {spatial_shape}"
            )
        flat_position_valid &= explicit_position_valid.reshape(-1)

    inferred_source_pose_valid = _pose_validity(source_poses)
    if source_entity_pose_valid is not None:
        explicit_source_pose_valid = np.asarray(source_entity_pose_valid, dtype=bool)
        if explicit_source_pose_valid.shape != (entity_count,):
            raise ValueError(
                "source_entity_pose_valid must be [N], "
                f"got {explicit_source_pose_valid.shape}"
            )
        require_se3(
            source_poses,
            "source_entity_poses",
            valid_mask=explicit_source_pose_valid,
        )
        inferred_source_pose_valid &= explicit_source_pose_valid
    else:
        require_se3(
            source_poses,
            "source_entity_poses",
            valid_mask=inferred_source_pose_valid,
        )

    inferred_target_pose_valid = _pose_validity(target_poses)
    if target_entity_pose_valid is not None:
        explicit_target_pose_valid = np.asarray(target_entity_pose_valid, dtype=bool)
        if explicit_target_pose_valid.shape != (target_count, entity_count):
            raise ValueError(
                "target_entity_pose_valid must be [K,N], "
                f"got {explicit_target_pose_valid.shape}"
            )
        require_se3(
            target_poses,
            "target_entity_poses",
            valid_mask=explicit_target_pose_valid,
        )
        inferred_target_pose_valid &= explicit_target_pose_valid
    else:
        require_se3(
            target_poses,
            "target_entity_poses",
            valid_mask=inferred_target_pose_valid,
        )

    output_shape = (target_count, *spatial_shape)
    flow = np.full((*output_shape, 3), np.nan, dtype=np.float32)
    future = np.full((*output_shape, 3), np.nan, dtype=np.float32)
    geom_valid = np.zeros(output_shape, dtype=bool)
    source_entity_index = np.full(spatial_shape, -1, dtype=np.int32)
    unmapped = np.zeros(spatial_shape, dtype=bool)
    unsupported = np.zeros(spatial_shape, dtype=bool)

    flat_flow = flow.reshape(target_count, -1, 3)
    flat_future = future.reshape(target_count, -1, 3)
    flat_geom_valid = geom_valid.reshape(target_count, -1)
    flat_entity_index = source_entity_index.reshape(-1)
    flat_unmapped = unmapped.reshape(-1)
    flat_unsupported = unsupported.reshape(-1)
    normalized_registry = _normalize_registry(registry)

    T_world_to_source_camera = inverse_se3(source_camera_pose)
    source_world = transform_points_se3(source_camera_pose, flat_xyz)

    # The only Python loop is over entity IDs.  Pixel transforms and all target
    # frames for one entity are vectorized below.
    for raw_seg_id in np.unique(flat_segmentation[flat_position_valid]):
        pixel_indices = np.flatnonzero(
            flat_position_valid & (flat_segmentation == raw_seg_id)
        )
        entity = normalized_registry.get(int(raw_seg_id))
        if entity is None:
            flat_unmapped[pixel_indices] = True
            continue

        flat_entity_index[pixel_indices] = entity.entity_index
        recognized_type = (
            entity.is_static
            or entity.entity_type in SUPPORTED_DYNAMIC_ENTITY_TYPES
        )
        if not entity.supported or not recognized_type:
            flat_unsupported[pixel_indices] = True
            continue

        if entity.is_static:
            flat_future[:, pixel_indices, :] = flat_xyz[pixel_indices][None]
            flat_flow[:, pixel_indices, :] = 0.0
            flat_geom_valid[:, pixel_indices] = True
            continue

        if entity.pose_row is None:
            raise ValueError(
                f"dynamic entity {entity.name!r} ({entity.entity_type}) "
                "has no pose_row"
            )
        pose_row = entity.pose_row
        if not 0 <= pose_row < entity_count:
            raise ValueError(
                f"entity {entity.name!r} pose_row={pose_row} is outside [0,{entity_count})"
            )
        if not inferred_source_pose_valid[pose_row]:
            continue

        source_entity_inverse = inverse_se3(source_poses[pose_row])
        local_points = transform_points_se3(
            source_entity_inverse, source_world[pixel_indices]
        )
        valid_targets = np.flatnonzero(inferred_target_pose_valid[:, pose_row])
        if not len(valid_targets):
            continue

        target_transforms = target_poses[valid_targets, pose_row]
        future_world = np.einsum(
            "kij,pj->kpi",
            target_transforms[:, :3, :3],
            local_points,
            optimize=True,
        )
        future_world += target_transforms[:, None, :3, 3]
        future_ref = np.einsum(
            "ij,kpj->kpi",
            T_world_to_source_camera[:3, :3],
            future_world,
            optimize=True,
        )
        future_ref += T_world_to_source_camera[None, None, :3, 3]
        flat_future[valid_targets[:, None], pixel_indices[None, :], :] = future_ref
        flat_flow[valid_targets[:, None], pixel_indices[None, :], :] = (
            future_ref - flat_xyz[pixel_indices][None]
        )
        flat_geom_valid[valid_targets[:, None], pixel_indices[None, :]] = True

    return FlowBatchResult(
        flow_xyz_ref=flow,
        future_xyz_ref=future,
        geom_valid=geom_valid,
        source_entity_index=source_entity_index,
        unmapped=unmapped,
        unsupported=unsupported,
        visibility=VisibilityMasks.unevaluated(output_shape),
    )

# Keep the former entity-grouped implementation as an explicit numerical
# reference.  Production callers below use the dense pose-row implementation.
compute_source_to_targets_grouped_reference = compute_source_to_targets


_DENSE_UNMAPPED = np.int8(0)
_DENSE_STATIC = np.int8(1)
_DENSE_DYNAMIC = np.int8(2)
_DENSE_UNSUPPORTED = np.int8(3)
_DEFAULT_DENSE_WORKING_SET_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class CompiledFlowRegistry:
    """Sorted episode-local registry arrays used by dense gather operations.

    This compilation loops only over the small metadata registry, never over
    pixels or target frames.  ``np.searchsorted``/``torch.searchsorted`` maps a
    segmentation image to these rows in one vectorized operation.
    """

    raw_seg_ids: np.ndarray
    entity_indices: np.ndarray
    pose_rows: np.ndarray
    kinds: np.ndarray


def compile_flow_registry(
    registry: Mapping[int | str, FlowEntity | Mapping[str, object]],
) -> CompiledFlowRegistry:
    """Compile a registry into sorted numeric lookup arrays."""

    normalized = _normalize_registry(registry)
    raw_ids = np.asarray(sorted(normalized), dtype=np.int64)
    entity_indices = np.empty(len(raw_ids), dtype=np.int32)
    pose_rows = np.full(len(raw_ids), -1, dtype=np.int64)
    kinds = np.full(len(raw_ids), _DENSE_UNSUPPORTED, dtype=np.int8)
    for index, raw_id in enumerate(raw_ids):
        entity = normalized[int(raw_id)]
        entity_indices[index] = entity.entity_index
        if entity.pose_row is not None:
            pose_rows[index] = entity.pose_row
        if not entity.supported:
            continue
        if entity.is_static:
            kinds[index] = _DENSE_STATIC
        elif entity.entity_type in SUPPORTED_DYNAMIC_ENTITY_TYPES:
            kinds[index] = _DENSE_DYNAMIC
    return CompiledFlowRegistry(
        raw_seg_ids=raw_ids,
        entity_indices=entity_indices,
        pose_rows=pose_rows,
        kinds=kinds,
    )


def estimate_dense_flow_memory_bytes(
    point_count: int,
    target_count: int,
    *,
    chunk_point_count: int | None = None,
) -> dict[str, int]:
    """Conservative FP32 memory estimate for the dense NumPy/Torch kernel.

    Output memory is unavoidable and scales as ``O(K*P)``.  Temporary memory is
    bounded by ``chunk_point_count`` and scales as ``O(K*C)``.  The estimate is
    deliberately conservative because advanced indexing materializes gathered
    rotations/translations in both NumPy and PyTorch.
    """

    points = int(point_count)
    targets = int(target_count)
    if points < 0 or targets < 0:
        raise ValueError("point_count and target_count must be non-negative")
    chunk = points if chunk_point_count is None else int(chunk_point_count)
    if chunk < 0:
        raise ValueError("chunk_point_count must be non-negative")
    chunk = min(points, chunk)
    # flow + future XYZ (FP32) and geom-valid (bool)
    outputs = targets * points * (2 * 3 * 4 + 1)
    # Source-world/mapping state retained for all points.
    persistent = points * (3 * 4 + 8 + 8 + 4 + 1 + 1 + 1)
    # Gathered source R/t/local and target R/t plus two XYZ intermediates.
    temporary = chunk * (15 * 4) + targets * chunk * (18 * 4 + 1)
    return {
        "output_bytes": int(outputs),
        "persistent_bytes": int(persistent),
        "temporary_bytes": int(temporary),
        "estimated_peak_bytes": int(outputs + persistent + temporary),
    }


def _dense_chunk_point_count(
    point_count: int,
    target_count: int,
    max_working_set_bytes: int | None,
) -> int:
    if point_count == 0:
        return 0
    if max_working_set_bytes is None:
        return point_count
    budget = int(max_working_set_bytes)
    if budget <= 0:
        raise ValueError("max_working_set_bytes must be positive or None")
    # Temporary bytes per point from estimate_dense_flow_memory_bytes().
    per_point = 15 * 4 + max(1, int(target_count)) * (18 * 4 + 1)
    return min(point_count, max(1, budget // per_point))


def _map_segmentation_numpy(
    flat_segmentation: np.ndarray,
    compiled: CompiledFlowRegistry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map raw IDs to registry arrays without a pixel/entity Python loop."""

    segmentation = np.asarray(flat_segmentation).astype(np.int64, copy=False)
    count = segmentation.size
    if not len(compiled.raw_seg_ids):
        return (
            np.zeros(count, dtype=bool),
            np.full(count, -1, dtype=np.int32),
            np.full(count, -1, dtype=np.int64),
            np.full(count, _DENSE_UNMAPPED, dtype=np.int8),
        )
    insertion = np.searchsorted(compiled.raw_seg_ids, segmentation)
    safe = np.minimum(insertion, len(compiled.raw_seg_ids) - 1)
    matched = (insertion < len(compiled.raw_seg_ids)) & (
        compiled.raw_seg_ids[safe] == segmentation
    )
    entity_indices = np.where(matched, compiled.entity_indices[safe], -1).astype(
        np.int32, copy=False
    )
    pose_rows = np.where(matched, compiled.pose_rows[safe], -1).astype(
        np.int64, copy=False
    )
    kinds = np.where(matched, compiled.kinds[safe], _DENSE_UNMAPPED).astype(
        np.int8, copy=False
    )
    return matched, entity_indices, pose_rows, kinds


def compute_source_to_targets_dense(
    source_position: np.ndarray,
    source_segmentation: np.ndarray,
    T_source_camera_to_world: np.ndarray,
    source_entity_poses: np.ndarray,
    target_entity_poses: np.ndarray,
    registry: Mapping[int | str, FlowEntity | Mapping[str, object]],
    *,
    source_position_valid: np.ndarray | None = None,
    source_entity_pose_valid: np.ndarray | None = None,
    target_entity_pose_valid: np.ndarray | None = None,
    T_target_camera_to_world: np.ndarray | None = None,
    max_working_set_bytes: int | None = _DEFAULT_DENSE_WORKING_SET_BYTES,
) -> FlowBatchResult:
    """Dense, source-aligned, piecewise-rigid flow for one source and K targets.

    Raw segmentation IDs are mapped to pose rows once with ``searchsorted``.
    Every point then gathers its source/target pose and participates in batched
    SE(3) arithmetic.  The only data-dependent Python loop is over coarse,
    contiguous point chunks used to cap temporary ``[K,C,...]`` tensors; there
    is no loop over pixels, entities, or target frames.

    ``max_working_set_bytes`` bounds temporary gather/intermediate tensors, not
    the required ``[K,P,3]`` outputs.  Pass ``None`` for one fully dense chunk.
    """

    position = np.asarray(source_position, dtype=np.float32)
    segmentation = np.asarray(source_segmentation)
    source_camera_pose = np.asarray(T_source_camera_to_world, dtype=np.float32)
    source_poses = np.asarray(source_entity_poses, dtype=np.float32)
    target_poses = np.asarray(target_entity_poses, dtype=np.float32)

    if position.ndim < 2 or position.shape[-1] < 3:
        raise ValueError(f"source_position must be [*S,>=3], got {position.shape}")
    spatial_shape = position.shape[:-1]
    if segmentation.shape != spatial_shape:
        raise ValueError(
            "source_segmentation shape must match source_position spatial shape, "
            f"got {segmentation.shape} and {spatial_shape}"
        )
    if source_camera_pose.shape != (4, 4):
        raise ValueError(
            "T_source_camera_to_world must be [4,4], "
            f"got {source_camera_pose.shape}"
        )
    source_camera_pose = require_se3(
        source_camera_pose, "T_source_camera_to_world"
    )
    if source_poses.ndim != 3 or source_poses.shape[-2:] != (4, 4):
        raise ValueError(
            f"source_entity_poses must be [N,4,4], got {source_poses.shape}"
        )
    if target_poses.ndim != 4 or target_poses.shape[-2:] != (4, 4):
        raise ValueError(
            f"target_entity_poses must be [K,N,4,4], got {target_poses.shape}"
        )
    if target_poses.shape[1] != source_poses.shape[0]:
        raise ValueError(
            "source/target entity counts differ: "
            f"{source_poses.shape[0]} vs {target_poses.shape[1]}"
        )

    target_count, entity_count = target_poses.shape[:2]
    if T_target_camera_to_world is not None:
        target_camera_poses = np.asarray(T_target_camera_to_world, dtype=np.float32)
        if target_camera_poses.shape != (target_count, 4, 4):
            raise ValueError(
                "T_target_camera_to_world must be [K,4,4], "
                f"got {target_camera_poses.shape}"
            )
        target_camera_poses = require_se3(
            target_camera_poses, "T_target_camera_to_world"
        )

    xyz_source = position[..., :3]
    flat_xyz = xyz_source.reshape(-1, 3)
    flat_segmentation = segmentation.reshape(-1)
    flat_position_valid = np.isfinite(flat_xyz).all(axis=-1)
    if source_position_valid is not None:
        explicit_position_valid = np.asarray(source_position_valid, dtype=bool)
        if explicit_position_valid.shape != spatial_shape:
            raise ValueError(
                "source_position_valid shape must match source spatial shape, "
                f"got {explicit_position_valid.shape} and {spatial_shape}"
            )
        flat_position_valid &= explicit_position_valid.reshape(-1)

    inferred_source_pose_valid = _pose_validity(source_poses)
    if source_entity_pose_valid is not None:
        explicit_source_pose_valid = np.asarray(source_entity_pose_valid, dtype=bool)
        if explicit_source_pose_valid.shape != (entity_count,):
            raise ValueError(
                "source_entity_pose_valid must be [N], "
                f"got {explicit_source_pose_valid.shape}"
            )
        require_se3(
            source_poses,
            "source_entity_poses",
            valid_mask=explicit_source_pose_valid,
        )
        inferred_source_pose_valid &= explicit_source_pose_valid
    else:
        require_se3(
            source_poses,
            "source_entity_poses",
            valid_mask=inferred_source_pose_valid,
        )

    inferred_target_pose_valid = _pose_validity(target_poses)
    if target_entity_pose_valid is not None:
        explicit_target_pose_valid = np.asarray(target_entity_pose_valid, dtype=bool)
        if explicit_target_pose_valid.shape != (target_count, entity_count):
            raise ValueError(
                "target_entity_pose_valid must be [K,N], "
                f"got {explicit_target_pose_valid.shape}"
            )
        require_se3(
            target_poses,
            "target_entity_poses",
            valid_mask=explicit_target_pose_valid,
        )
        inferred_target_pose_valid &= explicit_target_pose_valid
    else:
        require_se3(
            target_poses,
            "target_entity_poses",
            valid_mask=inferred_target_pose_valid,
        )

    compiled = compile_flow_registry(registry)
    matched, mapped_entity_index, pose_rows, kinds = _map_segmentation_numpy(
        flat_segmentation, compiled
    )
    active = flat_position_valid
    dynamic = active & (kinds == _DENSE_DYNAMIC)
    static = active & (kinds == _DENSE_STATIC)
    invalid_dynamic_row = dynamic & ((pose_rows < 0) | (pose_rows >= entity_count))
    if invalid_dynamic_row.any():
        bad_raw_id = int(flat_segmentation[np.flatnonzero(invalid_dynamic_row)[0]])
        raise ValueError(
            f"visible dynamic raw_seg_id={bad_raw_id} has a missing/out-of-range "
            f"pose_row for entity_count={entity_count}"
        )

    output_shape = (target_count, *spatial_shape)
    flow = np.full((*output_shape, 3), np.nan, dtype=np.float32)
    future = np.full((*output_shape, 3), np.nan, dtype=np.float32)
    geom_valid = np.zeros(output_shape, dtype=bool)
    source_entity_index = np.where(
        active & matched, mapped_entity_index, -1
    ).reshape(spatial_shape).astype(np.int32, copy=False)
    unmapped = (active & ~matched).reshape(spatial_shape)
    unsupported = (
        active & matched & (kinds == _DENSE_UNSUPPORTED)
    ).reshape(spatial_shape)
    flat_flow = flow.reshape(target_count, -1, 3)
    flat_future = future.reshape(target_count, -1, 3)
    flat_geom_valid = geom_valid.reshape(target_count, -1)

    if static.any():
        flat_future[:, static, :] = flat_xyz[static][None]
        flat_flow[:, static, :] = 0.0
        flat_geom_valid[:, static] = True
    if not dynamic.any() or target_count == 0:
        return FlowBatchResult(
            flow_xyz_ref=flow,
            future_xyz_ref=future,
            geom_valid=geom_valid,
            source_entity_index=source_entity_index,
            unmapped=unmapped,
            unsupported=unsupported,
            visibility=VisibilityMasks.unevaluated(output_shape),
        )

    source_world = (
        flat_xyz @ source_camera_pose[:3, :3].T + source_camera_pose[:3, 3]
    )
    point_count = len(flat_xyz)
    chunk_points = _dense_chunk_point_count(
        point_count, target_count, max_working_set_bytes
    )
    for start in range(0, point_count, chunk_points):
        stop = min(point_count, start + chunk_points)
        chunk_dynamic = dynamic[start:stop]
        # Non-dynamic points use row zero only as a safe gather placeholder and
        # are masked before writing.  A dynamic point was range-checked above.
        rows = np.where(chunk_dynamic, pose_rows[start:stop], 0)
        source_rotations = source_poses[rows, :3, :3]
        source_translations = source_poses[rows, :3, 3]
        local = np.einsum(
            "pi,pij->pj",
            source_world[start:stop] - source_translations,
            source_rotations,
            optimize=True,
        )
        # ``np.take(..., axis=1)`` preserves [K,C,...] ordering.  Mixed
        # basic/advanced indexing on the original 4-D array can otherwise move
        # the point dimension ahead of K when the final translation index is
        # scalar.
        target_rotations = np.take(target_poses[..., :3, :3], rows, axis=1)
        target_translations = np.take(target_poses[..., :3, 3], rows, axis=1)
        future_world = np.einsum(
            "kpij,pj->kpi", target_rotations, local, optimize=True
        )
        future_world += target_translations
        future_ref = np.einsum(
            "kpi,ij->kpj",
            future_world - source_camera_pose[None, None, :3, 3],
            source_camera_pose[:3, :3],
            optimize=True,
        )
        candidate_valid = (
            chunk_dynamic[None]
            & inferred_source_pose_valid[rows][None]
            & inferred_target_pose_valid[:, rows]
        )
        previous_future = flat_future[:, start:stop]
        previous_flow = flat_flow[:, start:stop]
        flat_future[:, start:stop] = np.where(
            candidate_valid[..., None], future_ref, previous_future
        )
        flat_flow[:, start:stop] = np.where(
            candidate_valid[..., None],
            future_ref - flat_xyz[None, start:stop],
            previous_flow,
        )
        flat_geom_valid[:, start:stop] |= candidate_valid

    return FlowBatchResult(
        flow_xyz_ref=flow,
        future_xyz_ref=future,
        geom_valid=geom_valid,
        source_entity_index=source_entity_index,
        unmapped=unmapped,
        unsupported=unsupported,
        visibility=VisibilityMasks.unevaluated(output_shape),
    )


# Public production API: retain the same required arguments and result schema.
compute_source_to_targets = compute_source_to_targets_dense
