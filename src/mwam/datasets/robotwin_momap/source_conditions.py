"""Source-time oracle geometry conditions for the standalone MoMap dataset.

The functions in this module are deliberately NumPy/CPU-only.  They consume
only the selected source frame, sample every modality at the same cell-centre
pixels, and express all three views in one fixed reference frame.  Future
geometry remains owned by the target builder and never enters this API.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mwam.robotwin_momap.se3 import require_se3
from mwam.robotwin_momap.target_service import uniform_cell_center_indices


COMMON_REFERENCE_CAMERA = "head_camera"
COMMON_REFERENCE_FRAME = "head_camera@source_t0"
SOURCE_CONDITION_SCHEMA_VERSION = "robotwin_momap_source_condition_v1"
SOURCE_CONDITION_AUGMENTED_SCHEMA_VERSION = (
    "robotwin_momap_source_condition_quality_seg_depth_noise_v1"
)
SOURCE_CONDITION_EDGE_AUGMENTED_SCHEMA_VERSION = (
    "robotwin_momap_source_condition_quality_seg_depth_edge_noise_v2"
)
SOURCE_SEG_ENCODING = "episode_scoped_sha256_rgb24_v1"
SOURCE_SEG_QUALITY_MIX_ENCODING = (
    "gt_episode_or_pred_episode_camera_scoped_sha256_rgb24_v1"
)
SOURCE_SEG_COLOR_SEED = "fastwam_momap_seg_condition_v1"
ROLE_TRACK_SEGMENTATION_ENCODING = (
    "role_linear_0_to_1_track_sha256_rg_episode_camera_v1"
)
ROLE_TRACK_SEGMENTATION_COLOR_SEED = "fastwam_momap_role_track_seg_condition_v1"
PREDICTED_TRACK_SEGMENTATION_ENCODING = (
    "pred_track_episode_camera_scoped_sha256_rgb24_v1"
)
# This is the frozen V3 GroundingDINO/SAM2 role vocabulary:
# 1=manipulated_target, 2=destination_or_reference, 3=tool, 4=robot.
# Zero is reserved exclusively for background.
ROLE_TRACK_SEGMENTATION_MAX_ROLE_ID = 4
# V3 role rasters occasionally contain a few zero-valued boundary pixels while
# the paired track map remains valid and role-prefixed.  Repair only that tiny
# redundant drift in memory; large role-map loss remains a hard error.
ROLE_TRACK_SPARSE_REPAIR_MAX_FRACTION = 1.0e-3
SOURCE_DEPTH_SEMANTICS = "gt_source_only_xyz_derivation_not_prediction_target"
SOURCE_DEPTH_AUGMENTED_SEMANTICS = (
    "gt_source_depth_sensor_noise_augmented_xyz_derivation_v1"
)
SOURCE_DEPTH_EDGE_AUGMENTED_SEMANTICS = (
    "gt_source_depth_sensor_noise_edge_erode_dilate_xyz_derivation_v2"
)


@dataclass(frozen=True)
class SourceConditionTensors:
    """Model-visible, source-only conditions before batch collation."""

    source_rgb: torch.Tensor
    source_depth_m: torch.Tensor
    source_depth_valid: torch.Tensor
    source_seg_condition: torch.Tensor
    source_xyz_ref: torch.Tensor
    source_condition_valid: torch.Tensor


def transform_points_to_reference(
    points_view: np.ndarray,
    T_view_to_world: np.ndarray,
    T_reference_to_world: np.ndarray,
) -> np.ndarray:
    """Transform row-vector XYZ from one camera into the fixed reference."""

    points = np.asarray(points_view, dtype=np.float32)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError(f"points_view must be [*,3], got {points.shape}")
    view = require_se3(T_view_to_world, "T_view_to_world")
    reference = require_se3(
        T_reference_to_world, "T_reference_to_world"
    )
    world = points @ view[:3, :3].T + view[:3, 3]
    return np.asarray(
        (world - reference[:3, 3]) @ reference[:3, :3],
        dtype=np.float32,
    )


def rotate_vectors_to_reference(
    vectors_view: np.ndarray,
    T_view_to_world: np.ndarray,
    T_reference_to_world: np.ndarray,
) -> np.ndarray:
    """Rotate displacement vectors between camera frames without translation."""

    vectors = np.asarray(vectors_view, dtype=np.float32)
    if vectors.ndim < 1 or vectors.shape[-1] != 3:
        raise ValueError(f"vectors_view must be [*,3], got {vectors.shape}")
    view = require_se3(T_view_to_world, "T_view_to_world")
    reference = require_se3(
        T_reference_to_world, "T_reference_to_world"
    )
    return np.asarray(
        vectors @ view[:3, :3].T @ reference[:3, :3],
        dtype=np.float32,
    )


def reconstruct_xyz_from_depth_ray(
    depth_m: np.ndarray,
    ray_xy: np.ndarray,
) -> np.ndarray:
    """Reconstruct OpenGL-camera XYZ as ``[ray_x*d, ray_y*d, -d]``."""

    depth = np.asarray(depth_m, dtype=np.float32)
    ray = np.asarray(ray_xy, dtype=np.float32)
    if ray.shape != (*depth.shape, 2):
        raise ValueError(
            f"ray_xy shape={ray.shape}, expected={(*depth.shape, 2)}"
        )
    xyz = np.empty((*depth.shape, 3), dtype=np.float32)
    np.multiply(ray, depth[..., None], out=xyz[..., :2])
    np.negative(depth, out=xyz[..., 2])
    return xyz


def encode_episode_scoped_segmentation(
    actor_seg_raw: np.ndarray,
    *,
    episode_identity: str,
    color_seed: str = SOURCE_SEG_COLOR_SEED,
) -> np.ndarray:
    """Encode instances as deterministic RGB features in ``[-1,1]``.

    Raw actor IDs are categorical and episode-local, so they must not be fed
    to a network as continuous magnitudes.  The episode identity is part of
    the hash: equal IDs are stable across views of one episode, while no color
    identity is promised across episodes.  Raw ID zero is always black.
    """

    segmentation = np.asarray(actor_seg_raw)
    if segmentation.ndim < 1 or not np.issubdtype(
        segmentation.dtype, np.integer
    ):
        raise TypeError("actor_seg_raw must be an integer array")
    if not str(episode_identity).strip():
        raise ValueError("episode_identity must not be empty")
    if not str(color_seed).strip():
        raise ValueError("color_seed must not be empty")

    encoded = np.zeros((*segmentation.shape, 3), dtype=np.float32)
    for raw_id in np.unique(segmentation):
        integer_id = int(raw_id)
        if integer_id == 0:
            continue
        digest = hashlib.sha256(
            (
                f"{color_seed}\0{episode_identity}\0{integer_id}"
            ).encode("utf-8")
        ).digest()
        rgb = np.frombuffer(digest[:3], dtype=np.uint8).copy()
        # Preserve black exclusively for background, even in the 1/2^24 hash
        # collision case.
        if not bool(rgb.any()):
            rgb[0] = 1
        encoded[segmentation == raw_id] = rgb.astype(np.float32) / 127.5 - 1.0
    return encoded


def validate_role_track_segmentation_ids(
    role_id: np.ndarray,
    track_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one V3 role/track categorical segmentation pair.

    ``role_id`` is semantic and uses the fixed IDs 0..4.  ``track_id`` is the
    temporally associated, camera-local slot ``role_id * 1000 + occurrence``.
    A pixel is background only when both values are zero.  Some V3 caches have
    sparse role-raster boundary holes: ``role_id==0`` while the non-zero track
    still carries an unambiguous role prefix.  Repair only those few redundant
    role values in memory; every other inconsistency remains fail-closed.
    """

    role = np.asarray(role_id)
    track = np.asarray(track_id)
    if role.ndim < 1 or track.ndim < 1:
        raise ValueError("role_id and track_id must have at least one dimension")
    if role.shape != track.shape:
        raise ValueError(
            "role_id/track_id shape mismatch: "
            f"{role.shape} != {track.shape}"
        )
    if not np.issubdtype(role.dtype, np.integer):
        raise TypeError("role_id must be an integer array")
    if not np.issubdtype(track.dtype, np.integer):
        raise TypeError("track_id must be an integer array")
    if np.issubdtype(role.dtype, np.signedinteger) and bool((role < 0).any()):
        raise ValueError("role_id must be non-negative")
    if np.issubdtype(track.dtype, np.signedinteger) and bool((track < 0).any()):
        raise ValueError("track_id must be non-negative")
    if int(role.max(initial=0)) > ROLE_TRACK_SEGMENTATION_MAX_ROLE_ID:
        raise ValueError(
            "role_id exceeds the V3 role vocabulary maximum "
            f"{ROLE_TRACK_SEGMENTATION_MAX_ROLE_ID}"
        )

    role_is_background = role == 0
    track_is_background = track == 0
    missing_role = role_is_background & ~track_is_background
    missing_role_count = int(missing_role.sum())
    if missing_role_count:
        max_repair_pixels = max(1, int(role.size * ROLE_TRACK_SPARSE_REPAIR_MAX_FRACTION))
        if missing_role_count > max_repair_pixels:
            raise ValueError(
                "role_id/track_id mismatch exceeds the sparse V3 boundary repair limit"
            )
        missing_track = track[missing_role].astype(np.int64, copy=False)
        inferred_role = missing_track // 1000
        if bool((inferred_role < 1).any()) or bool(
            (inferred_role > ROLE_TRACK_SEGMENTATION_MAX_ROLE_ID).any()
        ) or bool((missing_track % 1000 == 0).any()):
            raise ValueError(
                "non-background track_id cannot infer a valid V3 role prefix"
            )
        role = np.array(role, copy=True)
        role[missing_role] = inferred_role.astype(role.dtype, copy=False)
        role_is_background = role == 0
    if bool((~role_is_background & track_is_background).any()):
        raise ValueError("non-background role_id requires a non-zero track_id")
    foreground = ~track_is_background
    if bool(foreground.any()):
        foreground_track = track[foreground].astype(np.int64, copy=False)
        foreground_role = role[foreground].astype(np.int64, copy=False)
        if bool((foreground_track % 1000 == 0).any()):
            raise ValueError("foreground track_id must include a positive slot")
        if not bool(np.array_equal(foreground_track // 1000, foreground_role)):
            raise ValueError(
                "foreground track_id role prefix does not match role_id"
            )
    return role, track


def encode_role_track_segmentation(
    role_id: np.ndarray,
    track_id: np.ndarray,
    *,
    episode_identity: str,
    camera_name: str,
    color_seed: str = ROLE_TRACK_SEGMENTATION_COLOR_SEED,
) -> np.ndarray:
    """Encode a V3 role/track map as three finite features in ``[-1,1]``.

    Channel 0 is semantic role information, linearly encoded as
    ``role_id / 4``.  Channels 1--2 are a deterministic two-byte hash of the
    camera-local track identity, keeping separate instances distinguishable
    without treating a raw categorical ID as an ordinal value.  Background
    (``role_id == track_id == 0``) is exactly ``[0, 0, 0]``.

    The camera name is intentionally part of the track hash because V3 track
    slots are scoped to an episode *and* physical camera.  The role channel is
    nevertheless stable across all episodes and views.
    """

    role, track = validate_role_track_segmentation_ids(role_id, track_id)
    if not str(episode_identity).strip():
        raise ValueError("episode_identity must not be empty")
    if not str(camera_name).strip():
        raise ValueError("camera_name must not be empty")
    if not str(color_seed).strip():
        raise ValueError("color_seed must not be empty")

    encoded = np.zeros((*role.shape, 3), dtype=np.float32)
    foreground = role != 0
    if not bool(foreground.any()):
        return encoded

    role_float = role.astype(np.float32, copy=False)
    encoded[..., 0][foreground] = (
        role_float[foreground] / float(ROLE_TRACK_SEGMENTATION_MAX_ROLE_ID)
    )
    # There are at most a few live role-local tracks in one V3 cache frame, so
    # a two-byte hash is ample and keeps the Seg condition at three channels.
    for raw_track_id in np.unique(track[foreground]):
        integer_track_id = int(raw_track_id)
        mask = track == raw_track_id
        integer_role_id = int(role[mask][0])
        digest = hashlib.sha256(
            (
                f"{color_seed}\0{episode_identity}\0{camera_name}\0"
                f"{integer_role_id}\0{integer_track_id}"
            ).encode("utf-8")
        ).digest()
        track_rg = np.frombuffer(digest[:2], dtype=np.uint8).astype(
            np.float32
        )
        encoded[mask, 1:] = track_rg / 127.5 - 1.0
    return encoded


def encode_predicted_track_segmentation(
    role_id: np.ndarray,
    track_id: np.ndarray,
    *,
    episode_identity: str,
    camera_name: str,
    color_seed: str = SOURCE_SEG_COLOR_SEED,
) -> np.ndarray:
    """Encode a validated predicted track map without exposing role semantics.

    The V3 cache's ``role_id`` is used only to fail closed on corrupt
    role/track pairs.  Model-visible RGB is derived solely from ``track_id``
    with the same categorical SHA256 mapping used by oracle actor-instance
    segmentation.  Camera identity is included because predicted track slots
    are camera-local.  Background remains exactly ``[0, 0, 0]``.
    """

    _, track = validate_role_track_segmentation_ids(role_id, track_id)
    if not str(camera_name).strip():
        raise ValueError("camera_name must not be empty")
    return encode_episode_scoped_segmentation(
        track,
        episode_identity=(
            f"{episode_identity}\0{camera_name}\0predicted-track"
        ),
        color_seed=color_seed,
    )


def _sample_cell_centres(
    value: np.ndarray,
    grid_size: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 2:
        raise ValueError(f"cell-centre input must have H,W axes, got {array.shape}")
    rows, columns = uniform_cell_center_indices(
        (int(array.shape[0]), int(array.shape[1])),
        grid_size,
    )
    return np.asarray(array[np.ix_(rows, columns)])


def _normalize_rgb(rgb: np.ndarray, mode: str) -> np.ndarray:
    value = np.asarray(rgb)
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError(f"source RGB must be uint8[H,W,3], got {value.shape} {value.dtype}")
    result = value.astype(np.float32)
    if mode == "minus_one_one":
        return result / 127.5 - 1.0
    if mode == "zero_one":
        return result / 255.0
    raise ValueError(f"unsupported image normalization: {mode!r}")


def build_source_condition_tensors(
    frame: Mapping[str, Any],
    *,
    camera_names: Sequence[str],
    episode_identity: str,
    grid_size: tuple[int, int],
    reference_camera_name: str = COMMON_REFERENCE_CAMERA,
    image_normalization: str = "minus_one_one",
    segmentation_override_by_camera: Mapping[str, np.ndarray] | None = None,
) -> SourceConditionTensors:
    """Build aligned RGB, GT depth/seg and canonical XYZ from one source frame."""

    cameras = tuple(str(name) for name in camera_names)
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("camera_names must be non-empty and unique")
    if reference_camera_name not in cameras:
        raise ValueError(
            f"reference camera {reference_camera_name!r} is not in {cameras!r}"
        )
    if (
        not isinstance(grid_size, tuple)
        or len(grid_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in grid_size
        )
    ):
        raise ValueError("grid_size must be a pair of positive integers")

    observations = frame.get("observations")
    private = frame.get("private_geometry")
    if not isinstance(observations, Mapping) or not isinstance(private, Mapping):
        raise ValueError("source frame must contain observations/private_geometry")
    reference_geometry = private.get(reference_camera_name)
    if not isinstance(reference_geometry, Mapping):
        raise ValueError(f"source frame lacks geometry for {reference_camera_name}")
    reference_pose = require_se3(
        reference_geometry.get("T_cam_to_world"),
        f"{reference_camera_name}.T_cam_to_world",
    )

    rgb_values: list[np.ndarray] = []
    depth_values: list[np.ndarray] = []
    depth_valid_values: list[np.ndarray] = []
    seg_values: list[np.ndarray] = []
    xyz_values: list[np.ndarray] = []
    condition_valid_values: list[np.ndarray] = []

    for camera_name in cameras:
        camera_obs = observations.get(camera_name)
        geometry = private.get(camera_name)
        if not isinstance(camera_obs, Mapping) or not isinstance(geometry, Mapping):
            raise ValueError(f"source frame lacks camera payload for {camera_name}")

        rgb = np.asarray(camera_obs.get("rgb"))
        depth = np.asarray(geometry.get("depth_m"), dtype=np.float32)
        ray = np.asarray(geometry.get("ray_xy"), dtype=np.float32)
        ray_valid = np.asarray(geometry.get("ray_valid"), dtype=bool)
        position_valid = np.asarray(geometry.get("position_valid"), dtype=bool)
        use_predicted_seg = segmentation_override_by_camera is not None
        if use_predicted_seg:
            if camera_name not in segmentation_override_by_camera:
                raise ValueError(
                    "segmentation override must contain every configured camera"
                )
            segmentation = np.asarray(
                segmentation_override_by_camera[camera_name]
            )
        else:
            segmentation = np.asarray(geometry.get("actor_seg_raw"))
        pose = require_se3(
            geometry.get("T_cam_to_world"),
            f"{camera_name}.T_cam_to_world",
        )
        image_shape = tuple(int(value) for value in rgb.shape[:2])
        if depth.shape != image_shape:
            raise ValueError(f"{camera_name} depth/RGB shape mismatch")
        if ray.shape != (*image_shape, 2):
            raise ValueError(f"{camera_name} ray/RGB shape mismatch")
        for name, value in (
            ("ray_valid", ray_valid),
            ("position_valid", position_valid),
            ("actor_seg_raw", segmentation),
        ):
            if value.shape != image_shape:
                raise ValueError(f"{camera_name} {name}/RGB shape mismatch")
        if not np.issubdtype(segmentation.dtype, np.integer):
            raise TypeError(f"{camera_name}.actor_seg_raw must be integer")

        sampled_rgb = _sample_cell_centres(rgb, grid_size)
        sampled_depth = _sample_cell_centres(depth, grid_size)
        sampled_ray = _sample_cell_centres(ray, grid_size)
        sampled_ray_valid = _sample_cell_centres(ray_valid, grid_size).astype(bool)
        sampled_position_valid = _sample_cell_centres(
            position_valid, grid_size
        ).astype(bool)
        sampled_seg = _sample_cell_centres(segmentation, grid_size)

        depth_valid = (
            sampled_ray_valid
            & sampled_position_valid
            & np.isfinite(sampled_depth)
            & (sampled_depth > 0.0)
            & np.isfinite(sampled_ray).all(axis=-1)
        )
        # Geometric validity is independent of the categorical actor ID.
        # Background/static scene pixels may use ID zero while still carrying
        # valid depth and canonical XYZ.  Their segmentation feature remains
        # neutral zero, but the geometry must not be discarded.
        condition_valid = depth_valid.copy()
        xyz_view = reconstruct_xyz_from_depth_ray(sampled_depth, sampled_ray)
        xyz_ref = transform_points_to_reference(xyz_view, pose, reference_pose)
        condition_valid &= np.isfinite(xyz_ref).all(axis=-1)

        finite_depth = np.where(depth_valid, sampled_depth, 0.0).astype(
            np.float32, copy=False
        )
        finite_xyz = np.where(
            condition_valid[..., None], xyz_ref, 0.0
        ).astype(np.float32, copy=False)
        seg_rgb = encode_episode_scoped_segmentation(
            sampled_seg,
            episode_identity=(
                f"{episode_identity}\0{camera_name}\0predicted-track"
                if use_predicted_seg
                else episode_identity
            ),
        )
        seg_rgb[~condition_valid] = 0.0

        rgb_values.append(np.moveaxis(
            _normalize_rgb(sampled_rgb, image_normalization), -1, 0
        ))
        depth_values.append(finite_depth[None])
        depth_valid_values.append(depth_valid[None])
        seg_values.append(np.moveaxis(seg_rgb, -1, 0))
        xyz_values.append(np.moveaxis(finite_xyz, -1, 0))
        condition_valid_values.append(condition_valid[None])

    return SourceConditionTensors(
        source_rgb=torch.from_numpy(
            np.ascontiguousarray(np.stack(rgb_values), dtype=np.float32)
        ),
        source_depth_m=torch.from_numpy(
            np.ascontiguousarray(np.stack(depth_values), dtype=np.float32)
        ),
        source_depth_valid=torch.from_numpy(
            np.ascontiguousarray(np.stack(depth_valid_values), dtype=np.bool_)
        ),
        source_seg_condition=torch.from_numpy(
            np.ascontiguousarray(np.stack(seg_values), dtype=np.float32)
        ),
        source_xyz_ref=torch.from_numpy(
            np.ascontiguousarray(np.stack(xyz_values), dtype=np.float32)
        ),
        source_condition_valid=torch.from_numpy(
            np.ascontiguousarray(np.stack(condition_valid_values), dtype=np.bool_)
        ),
    )


__all__ = [
    "COMMON_REFERENCE_CAMERA",
    "COMMON_REFERENCE_FRAME",
    "SOURCE_CONDITION_AUGMENTED_SCHEMA_VERSION",
    "SOURCE_CONDITION_EDGE_AUGMENTED_SCHEMA_VERSION",
    "SOURCE_CONDITION_SCHEMA_VERSION",
    "SOURCE_DEPTH_AUGMENTED_SEMANTICS",
    "SOURCE_DEPTH_EDGE_AUGMENTED_SEMANTICS",
    "SOURCE_DEPTH_SEMANTICS",
    "SOURCE_SEG_COLOR_SEED",
    "SOURCE_SEG_ENCODING",
    "SOURCE_SEG_QUALITY_MIX_ENCODING",
    "ROLE_TRACK_SEGMENTATION_COLOR_SEED",
    "ROLE_TRACK_SEGMENTATION_ENCODING",
    "ROLE_TRACK_SEGMENTATION_MAX_ROLE_ID",
    "PREDICTED_TRACK_SEGMENTATION_ENCODING",
    "SourceConditionTensors",
    "build_source_condition_tensors",
    "encode_episode_scoped_segmentation",
    "encode_predicted_track_segmentation",
    "encode_role_track_segmentation",
    "reconstruct_xyz_from_depth_ray",
    "rotate_vectors_to_reference",
    "transform_points_to_reference",
    "validate_role_track_segmentation_ids",
]
