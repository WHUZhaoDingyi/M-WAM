"""Vectorized target-render visibility checks for GT 3D flow endpoints.

Geometry flow and render visibility are deliberately separate concepts.  A
source point can have a perfectly valid future 3D position even when that point
is occluded or outside the target image.  This module projects those future
positions into the *target* camera and compares them with the target z-buffer.

Depth is the primary visibility criterion.  Actor segmentation is only an
auxiliary consistency signal: an ID mismatch does not by itself make a
depth-consistent point invisible, and a matching ID does not hide
self-occlusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


CoordinateConvention = Literal[
    "sapien_opengl_camera",
    "opencv_camera",
]
FutureCoordinate = Literal[
    "world",
    "source_camera",
    "reference_camera",
]


@dataclass(frozen=True)
class VisibilityResult:
    """Target-render classification for one or more target frames.

    All mask and scalar arrays have shape ``[K, *S]``, where ``K`` is the
    number of target frames and ``*S`` is the source point/grid shape.

    ``evaluated`` is true for points that were conclusively classified as
    out-of-frame or had a valid target depth sample.  An in-frame point landing
    on an invalid target depth remains unevaluated.  A depth surface farther
    away than the predicted point is an evaluated depth conflict and is
    intentionally neither ``visible`` nor ``occluded``.
    """

    evaluated: np.ndarray
    sampling_ambiguous: np.ndarray
    visible: np.ndarray
    occluded: np.ndarray
    out_of_frame: np.ndarray
    id_mismatch: np.ndarray
    depth_residual_m: np.ndarray
    uv: np.ndarray
    projected_depth_m: np.ndarray
    sampled_depth_m: np.ndarray
    sampled_pixel_xy: np.ndarray


@dataclass(frozen=True)
class TargetImageProjection:
    """Projection-only result for source-aligned image-flow supervision.

    Arrays have shape ``[K, *S]`` except ``uv``, which has shape
    ``[K, *S, 2]``.  Unlike :class:`VisibilityResult`, this contains no target
    z-buffer classification: it answers only whether a future endpoint has a
    finite, front-facing projection inside the target image.
    """

    uv: np.ndarray
    projected_depth_m: np.ndarray
    finite: np.ndarray
    in_front: np.ndarray
    in_frame: np.ndarray


def inverse_se3(T: np.ndarray) -> np.ndarray:
    """Analytically invert one or a batch of rigid transforms.

    Matrices use the column-vector convention ``x_out = T @ x_in``.  The
    calculation is always performed in float32 and never uses a generic matrix
    inverse.
    """

    matrix = np.asarray(T, dtype=np.float32)
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"T must end in [4,4], got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("T contains NaN or Inf")
    expected_bottom = np.zeros_like(matrix[..., 3, :])
    expected_bottom[..., 3] = 1.0
    if not np.allclose(matrix[..., 3, :], expected_bottom, atol=1e-5):
        raise ValueError("T must be an SE(3) matrix with bottom row [0,0,0,1]")

    rotation_t = np.swapaxes(matrix[..., :3, :3], -1, -2)
    translation = matrix[..., :3, 3]
    result = np.zeros_like(matrix, dtype=np.float32)
    result[..., :3, :3] = rotation_t
    result[..., :3, 3] = -np.einsum(
        "...ij,...j->...i", rotation_t, translation, optimize=True
    )
    result[..., 3, 3] = 1.0
    return result


def _broadcast_camera_matrices(
    value: np.ndarray,
    target_count: int,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == shape:
        array = np.broadcast_to(array, (target_count, *shape))
    elif array.shape != (target_count, *shape):
        raise ValueError(
            f"{name} must be {shape} or [K,{shape[0]},{shape[1]}], "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return np.asarray(array, dtype=np.float32)


def _normalize_target_image(
    value: np.ndarray,
    target_count: int,
    *,
    channels: bool,
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    minimum_ndim = 3 if channels else 2
    if array.ndim == minimum_ndim:
        if target_count != 1:
            raise ValueError(
                f"{name} without a K dimension is only valid for K=1, "
                f"got K={target_count}"
            )
        array = array[None]
    expected_ndim = minimum_ndim + 1
    if array.ndim != expected_ndim or array.shape[0] != target_count:
        suffix = "[K,H,W,C]" if channels else "[K,H,W]"
        raise ValueError(f"{name} must be {suffix}, got {array.shape}")
    return array


def _normalize_point_mask(
    value: np.ndarray | None,
    target_count: int,
    spatial_shape: tuple[int, ...],
    *,
    default: bool,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.full((target_count, *spatial_shape), default, dtype=bool)
    array = np.asarray(value, dtype=bool)
    if array.shape == spatial_shape:
        array = np.broadcast_to(array, (target_count, *spatial_shape))
    elif array.shape != (target_count, *spatial_shape):
        raise ValueError(
            f"{name} must be [*S] or [K,*S], got {array.shape}; "
            f"expected {spatial_shape} or {(target_count, *spatial_shape)}"
        )
    return np.asarray(array, dtype=bool)


def _normalize_source_ids(
    source_entity_ids: np.ndarray,
    target_count: int,
    spatial_shape: tuple[int, ...],
) -> np.ndarray:
    ids = np.asarray(source_entity_ids)
    if ids.shape == spatial_shape:
        ids = np.broadcast_to(ids, (target_count, *spatial_shape))
    elif ids.shape != (target_count, *spatial_shape):
        raise ValueError(
            "source_entity_ids must be [*S] or [K,*S], "
            f"got {ids.shape}"
        )
    return ids


def _camera_depth_and_projection_coordinates(
    xyz_target_camera: np.ndarray,
    convention: CoordinateConvention,
) -> tuple[np.ndarray, np.ndarray]:
    if convention == "sapien_opengl_camera":
        # SAPIEN Position: x right, y up, visible z negative.  K uses the
        # equivalent CV image axes, hence [x, -y, -z] before projection.
        projection_xyz = xyz_target_camera.copy()
        projection_xyz[..., 1] *= -1.0
        projection_xyz[..., 2] *= -1.0
    elif convention == "opencv_camera":
        # OpenCV: x right, y down, visible z positive.
        projection_xyz = xyz_target_camera
    else:
        raise ValueError(
            "coordinate_convention must be 'sapien_opengl_camera' or "
            f"'opencv_camera', got {convention!r}"
        )
    return projection_xyz[..., 2], projection_xyz


def project_future_points_to_target_image(
    future_xyz: np.ndarray,
    T_target_camera_to_world: np.ndarray,
    K_target: np.ndarray,
    *,
    target_image_shape: tuple[int, int],
    future_coordinate: FutureCoordinate = "world",
    T_source_camera_to_world: np.ndarray | None = None,
    coordinate_convention: CoordinateConvention = "sapien_opengl_camera",
    near_plane_epsilon_m: float = 1e-8,
) -> TargetImageProjection:
    """Project future endpoints without reading or classifying a z-buffer.

    This is the minimal geometry primitive for source-aligned image flow.  It
    deliberately does *not* decide occlusion: callers may require only a
    finite, front-facing, in-frame target correspondence and avoid loading a
    target Position/depth texture.

    Args mirror :func:`evaluate_target_visibility` for coordinate conversion.
    ``target_image_shape`` is the native target camera ``(height, width)`` and
    defines the same pixel-centre bounds used by visibility:
    ``[-0.5, W-0.5) x [-0.5, H-0.5)``.
    """

    points = np.asarray(future_xyz, dtype=np.float32)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(f"future_xyz must be [K,*S,3], got {points.shape}")
    target_count = int(points.shape[0])
    if target_count <= 0:
        raise ValueError("future_xyz must contain at least one target frame")
    spatial_shape = tuple(int(value) for value in points.shape[1:-1])
    point_count = int(np.prod(spatial_shape, dtype=np.int64))
    flat_points = points.reshape(target_count, point_count, 3)

    if (
        not isinstance(target_image_shape, tuple)
        or len(target_image_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in target_image_shape
        )
    ):
        raise TypeError("target_image_shape must be an integer (height, width) pair")
    height, width = (int(target_image_shape[0]), int(target_image_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("target_image_shape values must be positive")
    if not np.isfinite(near_plane_epsilon_m) or near_plane_epsilon_m < 0:
        raise ValueError("near_plane_epsilon_m must be finite and non-negative")

    target_poses = _broadcast_camera_matrices(
        T_target_camera_to_world,
        target_count,
        (4, 4),
        "T_target_camera_to_world",
    )
    intrinsics = _broadcast_camera_matrices(
        K_target,
        target_count,
        (3, 3),
        "K_target",
    )

    if future_coordinate == "world":
        future_world = flat_points
    elif future_coordinate in {"source_camera", "reference_camera"}:
        if T_source_camera_to_world is None:
            raise ValueError(
                "T_source_camera_to_world is required for source/reference "
                "camera future coordinates"
            )
        source_pose = np.asarray(T_source_camera_to_world, dtype=np.float32)
        if source_pose.shape != (4, 4):
            raise ValueError(
                "T_source_camera_to_world must be [4,4], "
                f"got {source_pose.shape}"
            )
        inverse_se3(source_pose)
        future_world = np.einsum(
            "ij,kpj->kpi", source_pose[:3, :3], flat_points, optimize=True
        )
        future_world += source_pose[None, None, :3, 3]
    else:
        raise ValueError(
            "future_coordinate must be 'world', 'source_camera', or "
            f"'reference_camera', got {future_coordinate!r}"
        )

    world_to_target = inverse_se3(target_poses)
    future_target = np.einsum(
        "kij,kpj->kpi",
        world_to_target[:, :3, :3],
        future_world,
        optimize=True,
    )
    future_target += world_to_target[:, None, :3, 3]
    projected_depth, projection_xyz = _camera_depth_and_projection_coordinates(
        future_target,
        coordinate_convention,
    )
    homogeneous_image = np.einsum(
        "kij,kpj->kpi", intrinsics, projection_xyz, optimize=True
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = homogeneous_image[..., :2] / homogeneous_image[..., 2:3]

    finite = np.isfinite(future_target).all(axis=-1) & np.isfinite(uv).all(axis=-1)
    in_front = projected_depth > float(near_plane_epsilon_m)
    inside = (
        (uv[..., 0] >= -0.5)
        & (uv[..., 0] < width - 0.5)
        & (uv[..., 1] >= -0.5)
        & (uv[..., 1] < height - 0.5)
    )
    in_frame = finite & in_front & inside
    output_shape = (target_count, *spatial_shape)
    return TargetImageProjection(
        uv=uv.reshape(*output_shape, 2).astype(np.float32, copy=False),
        projected_depth_m=projected_depth.reshape(output_shape).astype(
            np.float32,
            copy=False,
        ),
        finite=finite.reshape(output_shape),
        in_front=in_front.reshape(output_shape),
        in_frame=in_frame.reshape(output_shape),
    )


def evaluate_target_visibility(
    future_xyz: np.ndarray,
    T_target_camera_to_world: np.ndarray,
    K_target: np.ndarray,
    *,
    target_position: np.ndarray | None = None,
    target_depth_m: np.ndarray | None = None,
    target_valid: np.ndarray | None = None,
    source_entity_ids: np.ndarray | None = None,
    target_actor_seg: np.ndarray | None = None,
    geom_valid: np.ndarray | None = None,
    future_coordinate: FutureCoordinate = "world",
    T_source_camera_to_world: np.ndarray | None = None,
    coordinate_convention: CoordinateConvention = "sapien_opengl_camera",
    depth_tolerance_m: float = 0.005,
    near_plane_epsilon_m: float = 1e-8,
    pixel_boundary_ambiguity_px: float = 0.0,
) -> VisibilityResult:
    """Evaluate future 3D endpoints against target camera render buffers.

    Args:
        future_xyz: Future points shaped ``[K,*S,3]``.  They are world-space
            by default; pass ``future_coordinate='source_camera'`` (or
            ``'reference_camera'``) together with
            ``T_source_camera_to_world`` for source/reference-camera points.
        T_target_camera_to_world: One ``[4,4]`` pose shared by all targets or
            target-specific poses shaped ``[K,4,4]``.
        K_target: One ``[3,3]`` intrinsic matrix or ``[K,3,3]``.
        target_position: Optional target Position textures, ``[K,H,W,>=3]``.
            For SAPIEN/OpenGL the metric depth is ``-z``; for OpenCV it is
            ``+z``.  Exactly one of this and ``target_depth_m`` is required.
        target_depth_m: Optional positive metric z-buffer, ``[K,H,W]``.
        target_valid: Optional explicit target z-buffer validity ``[K,H,W]``.
        source_entity_ids: Optional raw source entity IDs, ``[*S]`` or
            ``[K,*S]``.  Must be supplied together with ``target_actor_seg``.
        target_actor_seg: Optional raw target actor IDs, ``[K,H,W]``.  They are
            sampled only as an auxiliary check and never override depth.
        geom_valid: Optional source geometry validity, ``[*S]`` or ``[K,*S]``.
        depth_tolerance_m: Absolute z-buffer tolerance in metres.
        pixel_boundary_ambiguity_px: Optional nearest-pixel boundary band in
            pixels. Points within this distance of a half-integer pixel
            boundary remain explicitly unevaluated. The default ``0``
            disables this candidate-only ambiguity policy and preserves the
            historical exact rounding behaviour.

    Notes:
        Projection uses nearest pixel centres.  A subpixel coordinate is inside
        the image on ``[-0.5, W-0.5) x [-0.5, H-0.5)`` and is rounded with
        ``floor(x + 0.5)``.  This avoids Python point loops while making edge
        behaviour explicit.
    """

    points = np.asarray(future_xyz, dtype=np.float32)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(f"future_xyz must be [K,*S,3], got {points.shape}")
    target_count = points.shape[0]
    spatial_shape = points.shape[1:-1]
    point_count = int(np.prod(spatial_shape, dtype=np.int64))
    flat_points = points.reshape(target_count, point_count, 3)

    if not np.isfinite(depth_tolerance_m) or depth_tolerance_m < 0:
        raise ValueError("depth_tolerance_m must be finite and non-negative")
    if not np.isfinite(near_plane_epsilon_m) or near_plane_epsilon_m < 0:
        raise ValueError("near_plane_epsilon_m must be finite and non-negative")
    if (
        not np.isfinite(pixel_boundary_ambiguity_px)
        or pixel_boundary_ambiguity_px < 0
    ):
        raise ValueError(
            "pixel_boundary_ambiguity_px must be finite and non-negative"
        )
    if (target_position is None) == (target_depth_m is None):
        raise ValueError(
            "exactly one of target_position and target_depth_m must be supplied"
        )
    if (source_entity_ids is None) != (target_actor_seg is None):
        raise ValueError(
            "source_entity_ids and target_actor_seg must be supplied together"
        )

    target_poses = _broadcast_camera_matrices(
        T_target_camera_to_world,
        target_count,
        (4, 4),
        "T_target_camera_to_world",
    )
    intrinsics = _broadcast_camera_matrices(
        K_target, target_count, (3, 3), "K_target"
    )

    if future_coordinate == "world":
        future_world = flat_points
    elif future_coordinate in {"source_camera", "reference_camera"}:
        if T_source_camera_to_world is None:
            raise ValueError(
                "T_source_camera_to_world is required for source/reference "
                "camera future coordinates"
            )
        source_pose = np.asarray(T_source_camera_to_world, dtype=np.float32)
        if source_pose.shape != (4, 4):
            raise ValueError(
                "T_source_camera_to_world must be [4,4], "
                f"got {source_pose.shape}"
            )
        # Validate SE(3) and use its forward rigid transform.
        inverse_se3(source_pose)
        future_world = np.einsum(
            "ij,kpj->kpi", source_pose[:3, :3], flat_points, optimize=True
        )
        future_world += source_pose[None, None, :3, 3]
    else:
        raise ValueError(
            "future_coordinate must be 'world', 'source_camera', or "
            f"'reference_camera', got {future_coordinate!r}"
        )

    world_to_target = inverse_se3(target_poses)
    future_target = np.einsum(
        "kij,kpj->kpi",
        world_to_target[:, :3, :3],
        future_world,
        optimize=True,
    )
    future_target += world_to_target[:, None, :3, 3]

    projected_depth, projection_xyz = _camera_depth_and_projection_coordinates(
        future_target, coordinate_convention
    )
    homogeneous_image = np.einsum(
        "kij,kpj->kpi", intrinsics, projection_xyz, optimize=True
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = homogeneous_image[..., :2] / homogeneous_image[..., 2:3]

    if target_position is not None:
        position = _normalize_target_image(
            target_position,
            target_count,
            channels=True,
            name="target_position",
        )
        if position.shape[-1] < 3:
            raise ValueError(
                "target_position must have at least three channels, "
                f"got {position.shape}"
            )
        position_xyz = np.asarray(position[..., :3], dtype=np.float32)
        if coordinate_convention == "sapien_opengl_camera":
            depth_image = -position_xyz[..., 2]
        else:
            depth_image = position_xyz[..., 2]
        inferred_target_valid = np.isfinite(position_xyz).all(axis=-1)
    else:
        depth_image = np.asarray(
            _normalize_target_image(
                target_depth_m,
                target_count,
                channels=False,
                name="target_depth_m",
            ),
            dtype=np.float32,
        )
        inferred_target_valid = np.isfinite(depth_image)

    if depth_image.ndim != 3:
        raise AssertionError("normalized target depth must be [K,H,W]")
    height, width = depth_image.shape[1:]
    inferred_target_valid &= np.isfinite(depth_image)
    inferred_target_valid &= depth_image > near_plane_epsilon_m
    if target_valid is not None:
        explicit_target_valid = _normalize_target_image(
            target_valid,
            target_count,
            channels=False,
            name="target_valid",
        ).astype(bool, copy=False)
        if explicit_target_valid.shape != depth_image.shape:
            raise ValueError(
                "target_valid shape must match target depth, "
                f"got {explicit_target_valid.shape} and {depth_image.shape}"
            )
        inferred_target_valid &= explicit_target_valid

    target_segmentation: np.ndarray | None = None
    source_ids: np.ndarray | None = None
    if target_actor_seg is not None:
        target_segmentation = _normalize_target_image(
            target_actor_seg,
            target_count,
            channels=False,
            name="target_actor_seg",
        )
        if target_segmentation.shape != depth_image.shape:
            raise ValueError(
                "target_actor_seg shape must match target depth, "
                f"got {target_segmentation.shape} and {depth_image.shape}"
            )
        source_ids = _normalize_source_ids(
            source_entity_ids, target_count, spatial_shape
        ).reshape(target_count, point_count)

    flat_geom_valid = _normalize_point_mask(
        geom_valid,
        target_count,
        spatial_shape,
        default=True,
        name="geom_valid",
    ).reshape(target_count, point_count)

    finite_target_point = np.isfinite(future_target).all(axis=-1)
    finite_uv = np.isfinite(uv).all(axis=-1)
    in_front = projected_depth > near_plane_epsilon_m
    sampling_ambiguous = np.zeros_like(flat_geom_valid, dtype=bool)
    if pixel_boundary_ambiguity_px > 0:
        epsilon = float(pixel_boundary_ambiguity_px)
        safe_uv_for_fraction = np.where(np.isfinite(uv), uv, 0.0)
        fractional = safe_uv_for_fraction - np.floor(safe_uv_for_fraction)
        near_half_pixel = np.any(
            np.abs(fractional - 0.5) <= epsilon,
            axis=-1,
        )
        near_image = (
            (uv[..., 0] >= -0.5 - epsilon)
            & (uv[..., 0] < width - 0.5 + epsilon)
            & (uv[..., 1] >= -0.5 - epsilon)
            & (uv[..., 1] < height - 0.5 + epsilon)
        )
        sampling_ambiguous = (
            flat_geom_valid
            & finite_target_point
            & in_front
            & finite_uv
            & near_image
            & near_half_pixel
        )
    inside = (
        (uv[..., 0] >= -0.5)
        & (uv[..., 0] < width - 0.5)
        & (uv[..., 1] >= -0.5)
        & (uv[..., 1] < height - 0.5)
    )
    in_frame = (
        flat_geom_valid
        & finite_target_point
        & in_front
        & finite_uv
        & inside
        & ~sampling_ambiguous
    )
    out_of_frame = (flat_geom_valid & finite_target_point & (
        (~in_front) | (in_front & finite_uv & ~inside)
    )) & ~sampling_ambiguous

    # Use safe indices for a single vectorized gather.  Values for points that
    # are not in-frame are ignored by the masks below.
    safe_uv = np.where(np.isfinite(uv), uv, 0.0)
    pixel_x = np.floor(safe_uv[..., 0] + 0.5).astype(np.int64)
    pixel_y = np.floor(safe_uv[..., 1] + 0.5).astype(np.int64)
    pixel_x = np.clip(pixel_x, 0, max(width - 1, 0))
    pixel_y = np.clip(pixel_y, 0, max(height - 1, 0))
    target_indices = np.arange(target_count, dtype=np.int64)[:, None]
    sampled_depth = depth_image[target_indices, pixel_y, pixel_x]
    sampled_valid = inferred_target_valid[target_indices, pixel_y, pixel_x]
    depth_evaluable = in_frame & sampled_valid

    depth_residual = np.full((target_count, point_count), np.nan, dtype=np.float32)
    absolute_residual = np.abs(sampled_depth - projected_depth)
    depth_residual[depth_evaluable] = absolute_residual[depth_evaluable]
    visible = depth_evaluable & (absolute_residual <= depth_tolerance_m)
    occluded = depth_evaluable & (
        sampled_depth < projected_depth - depth_tolerance_m
    )
    evaluated = out_of_frame | depth_evaluable

    id_mismatch = np.zeros((target_count, point_count), dtype=bool)
    if target_segmentation is not None and source_ids is not None:
        sampled_ids = target_segmentation[target_indices, pixel_y, pixel_x]
        # Segmentation is auxiliary.  It is meaningful only at a valid sampled
        # surface and therefore never makes an otherwise invalid depth sample
        # evaluated.
        id_mismatch = depth_evaluable & (sampled_ids != source_ids)

    sampled_depth_output = np.full(
        (target_count, point_count), np.nan, dtype=np.float32
    )
    sampled_depth_output[depth_evaluable] = sampled_depth[depth_evaluable]
    sampled_pixel = np.full((target_count, point_count, 2), -1, dtype=np.int32)
    sampled_pixel[in_frame, 0] = pixel_x[in_frame]
    sampled_pixel[in_frame, 1] = pixel_y[in_frame]

    output_shape = (target_count, *spatial_shape)
    return VisibilityResult(
        evaluated=evaluated.reshape(output_shape),
        sampling_ambiguous=sampling_ambiguous.reshape(output_shape),
        visible=visible.reshape(output_shape),
        occluded=occluded.reshape(output_shape),
        out_of_frame=out_of_frame.reshape(output_shape),
        id_mismatch=id_mismatch.reshape(output_shape),
        depth_residual_m=depth_residual.reshape(output_shape),
        uv=uv.reshape(*output_shape, 2).astype(np.float32, copy=False),
        projected_depth_m=projected_depth.reshape(output_shape).astype(
            np.float32, copy=False
        ),
        sampled_depth_m=sampled_depth_output.reshape(output_shape),
        sampled_pixel_xy=sampled_pixel.reshape(*output_shape, 2),
    )
