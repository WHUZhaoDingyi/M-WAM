"""Typed boundary between canonical RoboTwin data and a future MoMap-WAM.

This module defines shapes, ownership, and auditable sample semantics.  The V1
index/factory chooses concrete action and temporal contracts; model loss-mask
policy remains outside the DataLoader boundary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


SOURCE_CONDITION_KEYS = (
    "source_rgb",
    "source_depth_m",
    "source_depth_valid",
    "source_seg_condition",
    "source_xyz_ref",
    "source_condition_valid",
)
MODEL_INPUT_KEYS = (
    *SOURCE_CONDITION_KEYS,
    "context",
    "context_mask",
    "proprio",
)
ACTION_TARGET_KEYS = ("action", "action_is_pad")
MOMAP_MASK_KEYS = (
    "momap_geom_valid",
    "momap_visibility_evaluated",
    "momap_visible",
    "momap_occluded",
    "momap_out_of_frame",
    "momap_depth_conflict",
    "momap_unsupported",
    "momap_unmapped",
    "momap_sampling_ambiguous",
)
MOMAP_TARGET_KEYS = ("momap", *MOMAP_MASK_KEYS, "momap_dt_seconds")
TENSOR_SAMPLE_KEYS = (*MODEL_INPUT_KEYS, *ACTION_TARGET_KEYS, *MOMAP_TARGET_KEYS)
HEAD_SEG_CANDIDATE_KEYS = (
    "source_head_seg_gt_condition",
    "source_head_seg_pred_condition",
)
MOMAP_COMPONENT_VALID_KEY = "momap_component_valid"
MOMAP_LOSS_VALID_KEY = "momap_loss_valid"
OPTIONAL_TENSOR_SAMPLE_KEYS = (
    *HEAD_SEG_CANDIDATE_KEYS,
    MOMAP_COMPONENT_VALID_KEY,
    MOMAP_LOSS_VALID_KEY,
)
METADATA_KEY = "metadata"
# ``FINAL_ALLOWED_KEYS`` historically described the exact V1 sample schema and
# is kept as the required/base set for callers which compare an old sample
# against it.  New DataLoader features are opt-in and live in
# ``FINAL_OPTIONAL_KEYS``; validators use ``FINAL_PERMITTED_KEYS`` as the true
# unknown-field allow-list.
FINAL_ALLOWED_KEYS = frozenset((*TENSOR_SAMPLE_KEYS, METADATA_KEY))
FINAL_REQUIRED_KEYS = FINAL_ALLOWED_KEYS
FINAL_OPTIONAL_KEYS = frozenset(OPTIONAL_TENSOR_SAMPLE_KEYS)
FINAL_PERMITTED_KEYS = FINAL_REQUIRED_KEYS | FINAL_OPTIONAL_KEYS
LANGUAGE_PROVENANCE_FIELDS = frozenset(
    {
        "language_source_kind",
        "language_scope_id",
        "language_binding_sha256",
        "language_instruction_sha256",
        "language_prompt_sha256",
        "text_embedding_cache_key",
    }
)
SOURCE_CONDITION_PROVENANCE_FIELDS = frozenset(
    {
        "source_condition_schema_version",
        "source_condition_reference_camera",
        "source_condition_grid_size",
        "source_seg_encoding",
        "source_seg_color_seed",
        "source_depth_semantics",
    }
)

METADATA_FIELDS = frozenset(
    {
        "episode_id",
        "episode_path",
        "source_index",
        "target_indices",
        "future_offsets",
        "future_horizons_seconds",
        "future_offset_basis",
        "camera_names",
        "action_start",
        "action_stop",
        "coordinate_frame",
        "action_indices",
        "action_query_times_sec",
        "action_sampling_mode",
        "action_rate_hz",
        "action_contract_schema_version",
        "action_label_kind",
        "action_is_high_level_policy_action",
        "action_normalization",
        "proprio_mode",
        "proprio_normalization",
        "momap_normalization",
        *SOURCE_CONDITION_PROVENANCE_FIELDS,
        *LANGUAGE_PROVENANCE_FIELDS,
    }
)

LEGACY_SAVED_FRAME_OFFSET_BASIS = "legacy_saved_frame"
FASTWAM_CLIP_FRAME_15HZ_OFFSET_BASIS = "fastwam_clip_frame@15hz"
KNOWN_FUTURE_OFFSET_BASES = frozenset(
    {
        LEGACY_SAVED_FRAME_OFFSET_BASIS,
        FASTWAM_CLIP_FRAME_15HZ_OFFSET_BASIS,
    }
)

# These names are legal inside ``PrivateGeometry`` only.  The final sample has
# an exact top-level and metadata whitelist, while this recursive blacklist
# provides a clearer error if a future adapter accidentally nests one of them.
PRIVATE_FIELD_NAMES = frozenset(
    {
        "private_geometry",
        "position_xyz",
        "depth_m",
        "ray_xy",
        "ray_valid",
        "raw_segmentation",
        "actor_seg_raw",
        "mesh_seg_raw",
        "camera_pose",
        "t_cam_to_world",
        "entity_pose",
        "pose_to_world",
        "entity_registry",
        "registry",
        "registry_index",
        "source_position",
        "source_segmentation",
        "source_camera_pose",
        "target_camera_pose",
        "source_entity_pose",
        "target_entity_pose",
    }
)


def _require_tensor(value: Any, name: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tuple(value.shape)}")
    return value


def _require_float(value: torch.Tensor, name: str) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype, got {value.dtype}")


def _require_bool(value: torch.Tensor, name: str) -> None:
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool, got {value.dtype}")


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains NaN/Inf")


@dataclass(frozen=True)
class SampleRef:
    """An already-resolved address for one training sample.

    The component that builds these references owns all temporal semantics.
    This class validates consistency but never infers offsets or action starts.
    """

    episode_id: str
    episode_path: str | Path
    source_index: int
    target_indices: tuple[int, ...]
    future_offsets: tuple[int, ...]
    future_horizons_seconds: tuple[float, ...] | None
    future_offset_basis: str
    camera_names: tuple[str, ...]
    action_start: int
    action_stop: int
    coordinate_frame: str
    action_indices: tuple[int, ...]
    action_query_times_sec: tuple[float, ...]
    action_sampling_mode: str
    action_rate_hz: float | None
    action_contract_schema_version: str
    action_label_kind: str
    action_is_high_level_policy_action: bool
    action_normalization: str
    proprio_mode: str
    proprio_normalization: str
    momap_normalization: str

    def __post_init__(self) -> None:
        if not str(self.episode_id).strip():
            raise ValueError("episode_id must not be empty")
        if not str(self.episode_path).strip():
            raise ValueError("episode_path must not be empty")
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")
        if not isinstance(self.target_indices, tuple) or any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in self.target_indices
        ):
            raise TypeError("target_indices must be a tuple of integers")
        if not self.target_indices:
            raise ValueError("target_indices must not be empty")
        if any(index <= self.source_index for index in self.target_indices):
            raise ValueError("target_indices must be strictly after source_index")
        if any(
            right <= left
            for left, right in zip(self.target_indices, self.target_indices[1:])
        ):
            raise ValueError("target_indices must be strictly increasing")
        if not isinstance(self.future_offsets, tuple) or any(
            not isinstance(offset, int) or isinstance(offset, bool)
            for offset in self.future_offsets
        ):
            raise TypeError("future_offsets must be a tuple of integers")
        if not self.future_offsets or any(
            offset <= 0 for offset in self.future_offsets
        ):
            raise ValueError("future_offsets must contain positive values")
        if any(
            right <= left
            for left, right in zip(self.future_offsets, self.future_offsets[1:])
        ):
            raise ValueError("future_offsets must be strictly increasing")
        if len(self.future_offsets) != len(self.target_indices):
            raise ValueError("future_offsets and target_indices must have the same K")
        if not isinstance(self.future_offset_basis, str):
            raise TypeError("future_offset_basis must be a string")
        if self.future_offset_basis not in KNOWN_FUTURE_OFFSET_BASES:
            raise ValueError(
                "future_offset_basis must be one of "
                f"{sorted(KNOWN_FUTURE_OFFSET_BASES)}, got "
                f"{self.future_offset_basis!r}"
            )
        if self.future_horizons_seconds is None:
            horizons: tuple[float, ...] | None = None
        else:
            if not isinstance(self.future_horizons_seconds, tuple) or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in self.future_horizons_seconds
            ):
                raise TypeError(
                    "future_horizons_seconds must be a tuple of numbers or None"
                )
            horizons = tuple(float(value) for value in self.future_horizons_seconds)
            if len(horizons) != len(self.future_offsets):
                raise ValueError(
                    "future_horizons_seconds and future_offsets must have the same K"
                )
            if any(not math.isfinite(value) or value <= 0 for value in horizons):
                raise ValueError(
                    "future_horizons_seconds must contain finite positive values"
                )
            if any(
                right <= left for left, right in zip(horizons, horizons[1:])
            ):
                raise ValueError("future_horizons_seconds must be strictly increasing")
            object.__setattr__(self, "future_horizons_seconds", horizons)
        if self.future_offset_basis == LEGACY_SAVED_FRAME_OFFSET_BASIS:
            if horizons is not None:
                raise ValueError(
                    "legacy_saved_frame offsets must not also define physical horizons"
                )
            actual_offsets = tuple(
                target - self.source_index for target in self.target_indices
            )
            if actual_offsets != self.future_offsets:
                raise ValueError(
                    "legacy_saved_frame target_indices must equal "
                    "source_index + future_offsets"
                )
        elif horizons is None:
            raise ValueError(
                "fastwam_clip_frame@15hz offsets require future_horizons_seconds"
            )
        if not self.camera_names or any(not name.strip() for name in self.camera_names):
            raise ValueError("camera_names must contain non-empty names")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must not contain duplicates")
        if self.action_start < 0 or self.action_stop <= self.action_start:
            raise ValueError("action range must satisfy 0 <= start < stop")
        if not self.action_indices:
            raise ValueError("action_indices must not be empty")
        if any(
            right <= left
            for left, right in zip(self.action_indices, self.action_indices[1:])
        ):
            raise ValueError("action_indices must be strictly increasing")
        if (
            self.action_start != self.action_indices[0]
            or self.action_stop != self.action_indices[-1] + 1
        ):
            raise ValueError("action range must tightly enclose action_indices")
        if len(self.action_query_times_sec) != len(self.action_indices):
            raise ValueError("action index/query-time length mismatch")
        if any(not math.isfinite(float(value)) for value in self.action_query_times_sec):
            raise ValueError("action query times contain NaN/Inf")
        if not self.action_sampling_mode.strip():
            raise ValueError("action_sampling_mode must not be empty")
        for name, value in (
            ("action_contract_schema_version", self.action_contract_schema_version),
            ("action_label_kind", self.action_label_kind),
            ("action_normalization", self.action_normalization),
            ("proprio_mode", self.proprio_mode),
            ("proprio_normalization", self.proprio_normalization),
            ("momap_normalization", self.momap_normalization),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if not isinstance(self.action_is_high_level_policy_action, bool):
            raise TypeError("action_is_high_level_policy_action must be bool")
        if self.action_rate_hz is not None and (
            not math.isfinite(float(self.action_rate_hz))
            or float(self.action_rate_hz) <= 0
        ):
            raise ValueError("action_rate_hz must be finite and positive when set")
        if not self.coordinate_frame.strip():
            raise ValueError("coordinate_frame must not be empty")

    def as_metadata(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "episode_path": str(self.episode_path),
            "source_index": self.source_index,
            "target_indices": tuple(self.target_indices),
            "future_offsets": tuple(self.future_offsets),
            "future_horizons_seconds": (
                None
                if self.future_horizons_seconds is None
                else tuple(self.future_horizons_seconds)
            ),
            "future_offset_basis": self.future_offset_basis,
            "camera_names": tuple(self.camera_names),
            "action_start": self.action_start,
            "action_stop": self.action_stop,
            "coordinate_frame": self.coordinate_frame,
            "action_indices": tuple(self.action_indices),
            "action_query_times_sec": tuple(self.action_query_times_sec),
            "action_sampling_mode": self.action_sampling_mode,
            "action_rate_hz": self.action_rate_hz,
            "action_contract_schema_version": self.action_contract_schema_version,
            "action_label_kind": self.action_label_kind,
            "action_is_high_level_policy_action": (
                self.action_is_high_level_policy_action
            ),
            "action_normalization": self.action_normalization,
            "proprio_mode": self.proprio_mode,
            "proprio_normalization": self.proprio_normalization,
            "momap_normalization": self.momap_normalization,
        }


@dataclass(frozen=True)
class ModelInputs:
    source_rgb: torch.Tensor
    source_depth_m: torch.Tensor
    source_depth_valid: torch.Tensor
    source_seg_condition: torch.Tensor
    source_xyz_ref: torch.Tensor
    source_condition_valid: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    proprio: torch.Tensor
    provenance: Mapping[str, object]
    condition_provenance: Mapping[str, object]
    source_head_seg_gt_condition: torch.Tensor | None = None
    source_head_seg_pred_condition: torch.Tensor | None = None

    def __post_init__(self) -> None:
        source_rgb = _require_tensor(self.source_rgb, "source_rgb", 4)
        source_depth_m = _require_tensor(
            self.source_depth_m, "source_depth_m", 4
        )
        source_depth_valid = _require_tensor(
            self.source_depth_valid, "source_depth_valid", 4
        )
        source_seg_condition = _require_tensor(
            self.source_seg_condition, "source_seg_condition", 4
        )
        source_xyz_ref = _require_tensor(
            self.source_xyz_ref, "source_xyz_ref", 4
        )
        source_condition_valid = _require_tensor(
            self.source_condition_valid, "source_condition_valid", 4
        )
        context = _require_tensor(self.context, "context", 2)
        context_mask = _require_tensor(self.context_mask, "context_mask", 1)
        proprio = _require_tensor(self.proprio, "proprio", 2)
        if source_rgb.shape[0] <= 0 or source_rgb.shape[1] != 3:
            raise ValueError(
                "source_rgb must be [V,3,Hg,Wg], got "
                f"{tuple(source_rgb.shape)}"
            )
        view_count = source_rgb.shape[0]
        grid_shape = tuple(source_rgb.shape[-2:])
        expected_channels = {
            "source_rgb": 3,
            "source_depth_m": 1,
            "source_depth_valid": 1,
            "source_seg_condition": 3,
            "source_xyz_ref": 3,
            "source_condition_valid": 1,
        }
        source_values = {
            "source_rgb": source_rgb,
            "source_depth_m": source_depth_m,
            "source_depth_valid": source_depth_valid,
            "source_seg_condition": source_seg_condition,
            "source_xyz_ref": source_xyz_ref,
            "source_condition_valid": source_condition_valid,
        }
        if grid_shape[0] <= 0 or grid_shape[1] <= 0:
            raise ValueError("source condition grid must be non-empty")
        for name, tensor in source_values.items():
            expected = (
                view_count,
                expected_channels[name],
                *grid_shape,
            )
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} shape={tuple(tensor.shape)}, expected={expected}"
                )
        if context.shape[0] != context_mask.shape[0]:
            raise ValueError("context and context_mask length mismatch")
        if proprio.shape[0] <= 0 or proprio.shape[1] <= 0:
            raise ValueError(f"proprio must be non-empty [Tp,Dp], got {tuple(proprio.shape)}")
        for name, tensor in (
            ("source_rgb", source_rgb),
            ("source_depth_m", source_depth_m),
            ("source_seg_condition", source_seg_condition),
            ("source_xyz_ref", source_xyz_ref),
            ("context", context),
            ("proprio", proprio),
        ):
            _require_float(tensor, name)
            _require_finite(tensor, name)
        _require_bool(context_mask, "context_mask")
        _require_bool(source_depth_valid, "source_depth_valid")
        _require_bool(source_condition_valid, "source_condition_valid")
        if bool((source_condition_valid & ~source_depth_valid).any().item()):
            raise ValueError(
                "source_condition_valid must be a subset of source_depth_valid"
            )
        if bool((source_depth_m[source_depth_valid] <= 0).any().item()):
            raise ValueError("valid source depth must be strictly positive")
        if bool((source_depth_m[~source_depth_valid] != 0).any().item()):
            raise ValueError("invalid source depth must be finiteized to zero")
        condition_valid_xyz = source_condition_valid.expand_as(source_xyz_ref)
        if bool((source_xyz_ref[~condition_valid_xyz] != 0).any().item()):
            raise ValueError("invalid source XYZ must be finiteized to zero")
        condition_valid_seg = source_condition_valid.expand_as(
            source_seg_condition
        )
        if bool((source_seg_condition[~condition_valid_seg] != 0).any().item()):
            raise ValueError(
                "background/invalid source segmentation must be neutral zero"
            )
        if bool(
            ((source_seg_condition < -1.0) | (source_seg_condition > 1.0))
            .any()
            .item()
        ):
            raise ValueError("source_seg_condition must lie in [-1,1]")
        _validate_language_provenance(self.provenance)
        _validate_source_condition_provenance(
            self.condition_provenance,
            expected_grid_size=grid_shape,
        )
        head_candidates = (
            self.source_head_seg_gt_condition,
            self.source_head_seg_pred_condition,
        )
        if (head_candidates[0] is None) != (head_candidates[1] is None):
            raise ValueError(
                "Head GT and predicted segmentation candidates must either "
                "both be present or both be absent"
            )
        for name, candidate in zip(HEAD_SEG_CANDIDATE_KEYS, head_candidates):
            if candidate is None:
                continue
            candidate = _require_tensor(candidate, name, 3)
            expected = (3, *grid_shape)
            if tuple(candidate.shape) != expected:
                raise ValueError(
                    f"{name} shape={tuple(candidate.shape)}, expected={expected}"
                )
            _require_float(candidate, name)
            _require_finite(candidate, name)
            if bool(((candidate < -1.0) | (candidate > 1.0)).any().item()):
                raise ValueError(f"{name} must lie in [-1,1]")


@dataclass(frozen=True)
class ActionTarget:
    action: torch.Tensor
    action_is_pad: torch.Tensor

    def __post_init__(self) -> None:
        action = _require_tensor(self.action, "action", 2)
        action_is_pad = _require_tensor(self.action_is_pad, "action_is_pad", 1)
        if action.shape[0] <= 0 or action.shape[1] <= 0:
            raise ValueError(f"action must be non-empty [Ta,Da], got {tuple(action.shape)}")
        if action.shape[0] != action_is_pad.shape[0]:
            raise ValueError("action and action_is_pad horizon mismatch")
        _require_float(action, "action")
        _require_finite(action, "action")
        _require_bool(action_is_pad, "action_is_pad")


@dataclass(frozen=True)
class PrivateGeometry:
    """Opaque teacher-only payload that must never be merged into a sample."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("PrivateGeometry.payload must be a mapping")


@dataclass(frozen=True)
class MoMapTarget:
    momap: torch.Tensor
    geom_valid: torch.Tensor
    visibility_evaluated: torch.Tensor
    visible: torch.Tensor
    occluded: torch.Tensor
    out_of_frame: torch.Tensor
    depth_conflict: torch.Tensor
    unsupported: torch.Tensor
    unmapped: torch.Tensor
    sampling_ambiguous: torch.Tensor
    dt_seconds: torch.Tensor
    component_valid: torch.Tensor | None = None
    loss_valid: torch.Tensor | None = None

    def __post_init__(self) -> None:
        momap = _require_tensor(self.momap, "momap", 5)
        if momap.shape[0] <= 0 or momap.shape[1] <= 0 or momap.shape[2] != 3:
            raise ValueError(f"momap must be [V,K,3,H,W], got {tuple(momap.shape)}")
        _require_float(momap, "momap")
        mask_shape = (momap.shape[0], momap.shape[1], 1, momap.shape[3], momap.shape[4])
        for name, mask in self.mask_items():
            _require_tensor(mask, name, 5)
            _require_bool(mask, name)
            if tuple(mask.shape) != mask_shape:
                raise ValueError(f"{name} shape={tuple(mask.shape)}, expected={mask_shape}")
        dt_seconds = _require_tensor(self.dt_seconds, "momap_dt_seconds", 1)
        if tuple(dt_seconds.shape) != (momap.shape[1],):
            raise ValueError(
                "momap_dt_seconds must be [K], got "
                f"{tuple(dt_seconds.shape)} for K={momap.shape[1]}"
            )
        _require_float(dt_seconds, "momap_dt_seconds")
        _require_finite(dt_seconds, "momap_dt_seconds")
        if self.component_valid is not None:
            component_valid = _require_tensor(
                self.component_valid,
                MOMAP_COMPONENT_VALID_KEY,
                5,
            )
            _require_bool(component_valid, MOMAP_COMPONENT_VALID_KEY)
            if tuple(component_valid.shape) != tuple(momap.shape):
                raise ValueError(
                    f"{MOMAP_COMPONENT_VALID_KEY} "
                    f"shape={tuple(component_valid.shape)}, "
                    f"expected={tuple(momap.shape)}"
                )
            if bool(
                (component_valid & ~self.geom_valid.expand_as(component_valid))
                .any()
                .item()
            ):
                raise ValueError(
                    f"{MOMAP_COMPONENT_VALID_KEY} must be a subset of "
                    "momap_geom_valid expanded across components"
                )
        if self.loss_valid is not None:
            if self.component_valid is None:
                raise ValueError(
                    f"{MOMAP_LOSS_VALID_KEY} requires "
                    f"{MOMAP_COMPONENT_VALID_KEY}"
                )
            loss_valid = _require_tensor(
                self.loss_valid,
                MOMAP_LOSS_VALID_KEY,
                5,
            )
            _require_bool(loss_valid, MOMAP_LOSS_VALID_KEY)
            if tuple(loss_valid.shape) != tuple(momap.shape):
                raise ValueError(
                    f"{MOMAP_LOSS_VALID_KEY} "
                    f"shape={tuple(loss_valid.shape)}, "
                    f"expected={tuple(momap.shape)}"
                )
            if bool((loss_valid & ~self.component_valid).any().item()):
                raise ValueError(
                    f"{MOMAP_LOSS_VALID_KEY} must be a subset of "
                    f"{MOMAP_COMPONENT_VALID_KEY}"
                )

    def mask_items(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("momap_geom_valid", self.geom_valid),
            ("momap_visibility_evaluated", self.visibility_evaluated),
            ("momap_visible", self.visible),
            ("momap_occluded", self.occluded),
            ("momap_out_of_frame", self.out_of_frame),
            ("momap_depth_conflict", self.depth_conflict),
            ("momap_unsupported", self.unsupported),
            ("momap_unmapped", self.unmapped),
            ("momap_sampling_ambiguous", self.sampling_ambiguous),
        )


def _walk_mapping_keys(value: Any, path: str = "sample") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized.startswith("private_") or normalized in PRIVATE_FIELD_NAMES:
                raise ValueError(f"private geometry field leaked at {path}.{key}")
            _walk_mapping_keys(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _walk_mapping_keys(child, f"{path}[{index}]")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_language_provenance(provenance: Mapping[str, object]) -> None:
    if not isinstance(provenance, Mapping):
        raise TypeError("language provenance must be a mapping")
    keys = set(provenance)
    if keys != LANGUAGE_PROVENANCE_FIELDS:
        raise KeyError(
            "language provenance fields mismatch: "
            f"missing={sorted(LANGUAGE_PROVENANCE_FIELDS - keys)}, "
            f"unexpected={sorted(keys - LANGUAGE_PROVENANCE_FIELDS)}"
        )
    for field in ("language_source_kind", "language_scope_id"):
        value = provenance[field]
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{field} must be a non-empty string")
    for field in ("language_instruction_sha256", "language_prompt_sha256"):
        if not _is_sha256(provenance[field]):
            raise ValueError(f"{field} must be a lowercase SHA-256")
    binding_sha = provenance["language_binding_sha256"]
    if binding_sha is not None and not _is_sha256(binding_sha):
        raise ValueError("language_binding_sha256 must be SHA-256 or None")
    cache_key = provenance["text_embedding_cache_key"]
    if cache_key is not None and (
        not isinstance(cache_key, str) or not cache_key.strip()
    ):
        raise TypeError("text_embedding_cache_key must be non-empty string or None")


def _validate_source_condition_provenance(
    provenance: Mapping[str, object],
    *,
    expected_grid_size: tuple[int, int] | None = None,
) -> None:
    if not isinstance(provenance, Mapping):
        raise TypeError("source condition provenance must be a mapping")
    keys = set(provenance)
    if keys != SOURCE_CONDITION_PROVENANCE_FIELDS:
        raise KeyError(
            "source condition provenance fields mismatch: "
            f"missing={sorted(SOURCE_CONDITION_PROVENANCE_FIELDS - keys)}, "
            f"unexpected={sorted(keys - SOURCE_CONDITION_PROVENANCE_FIELDS)}"
        )
    for field in (
        "source_condition_schema_version",
        "source_condition_reference_camera",
        "source_seg_encoding",
        "source_seg_color_seed",
        "source_depth_semantics",
    ):
        value = provenance[field]
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{field} must be a non-empty string")
    grid = provenance["source_condition_grid_size"]
    if (
        not isinstance(grid, tuple)
        or len(grid) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in grid
        )
    ):
        raise TypeError(
            "source_condition_grid_size must be a pair of positive integers"
        )
    if expected_grid_size is not None and tuple(grid) != tuple(expected_grid_size):
        raise ValueError(
            "source_condition_grid_size disagrees with source tensors: "
            f"metadata={grid}, tensors={expected_grid_size}"
        )


def _validate_metadata_types(metadata: Mapping[str, Any]) -> None:
    for field in (
        "episode_id",
        "episode_path",
        "coordinate_frame",
        "future_offset_basis",
        "action_sampling_mode",
        "action_contract_schema_version",
        "action_label_kind",
        "action_normalization",
        "proprio_mode",
        "proprio_normalization",
        "momap_normalization",
    ):
        if not isinstance(metadata[field], str):
            raise TypeError(f"metadata.{field} must be a string")

    for field in ("source_index", "action_start", "action_stop"):
        field_value = metadata[field]
        if not isinstance(field_value, int) or isinstance(field_value, bool):
            raise TypeError(f"metadata.{field} must be an integer")

    target_indices = metadata["target_indices"]
    if not isinstance(target_indices, tuple) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in target_indices
    ):
        raise TypeError("metadata.target_indices must be a tuple of integers")

    future_offsets = metadata["future_offsets"]
    if not isinstance(future_offsets, tuple) or any(
        not isinstance(item, int) or isinstance(item, bool)
        for item in future_offsets
    ):
        raise TypeError("metadata.future_offsets must be a tuple of integers")

    future_horizons = metadata["future_horizons_seconds"]
    if future_horizons is not None and (
        not isinstance(future_horizons, tuple)
        or any(
            not isinstance(item, (int, float)) or isinstance(item, bool)
            for item in future_horizons
        )
    ):
        raise TypeError(
            "metadata.future_horizons_seconds must be a tuple of numbers or None"
        )

    action_indices = metadata["action_indices"]
    if not isinstance(action_indices, tuple) or any(
        not isinstance(item, int) or isinstance(item, bool)
        for item in action_indices
    ):
        raise TypeError("metadata.action_indices must be a tuple of integers")

    action_query_times = metadata["action_query_times_sec"]
    if not isinstance(action_query_times, tuple) or any(
        not isinstance(item, (int, float)) or isinstance(item, bool)
        for item in action_query_times
    ):
        raise TypeError("metadata.action_query_times_sec must be a tuple of numbers")

    action_rate = metadata["action_rate_hz"]
    if action_rate is not None and (
        not isinstance(action_rate, (int, float)) or isinstance(action_rate, bool)
    ):
        raise TypeError("metadata.action_rate_hz must be a number or None")

    if not isinstance(metadata["action_is_high_level_policy_action"], bool):
        raise TypeError(
            "metadata.action_is_high_level_policy_action must be a bool"
        )

    _validate_language_provenance(
        {field: metadata[field] for field in LANGUAGE_PROVENANCE_FIELDS}
    )
    _validate_source_condition_provenance(
        {
            field: metadata[field]
            for field in SOURCE_CONDITION_PROVENANCE_FIELDS
        }
    )

    camera_names = metadata["camera_names"]
    if not isinstance(camera_names, tuple) or any(
        not isinstance(item, str) for item in camera_names
    ):
        raise TypeError("metadata.camera_names must be a tuple of strings")


def assert_no_private_or_unknown_fields(sample: Mapping[str, Any]) -> None:
    if not isinstance(sample, Mapping):
        raise TypeError("training sample must be a mapping")
    actual = set(sample)
    missing = FINAL_REQUIRED_KEYS - actual
    unexpected = actual - FINAL_PERMITTED_KEYS
    if missing:
        raise KeyError(f"training sample is missing fields: {sorted(missing)}")
    if unexpected:
        raise KeyError(f"training sample has unexpected fields: {sorted(unexpected)}")
    present_head_candidates = actual & set(HEAD_SEG_CANDIDATE_KEYS)
    if present_head_candidates and present_head_candidates != set(
        HEAD_SEG_CANDIDATE_KEYS
    ):
        raise KeyError(
            "Head GT and predicted segmentation candidate fields must be "
            "present together"
        )
    metadata = sample[METADATA_KEY]
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    metadata_keys = set(metadata)
    if metadata_keys != METADATA_FIELDS:
        raise KeyError(
            "metadata fields mismatch: "
            f"missing={sorted(METADATA_FIELDS - metadata_keys)}, "
            f"unexpected={sorted(metadata_keys - METADATA_FIELDS)}"
        )
    _validate_metadata_types(metadata)
    _walk_mapping_keys(sample)


def training_sample_to_dict(
    sample_ref: SampleRef,
    model_inputs: ModelInputs,
    action_target: ActionTarget,
    momap_target: MoMapTarget,
) -> dict[str, object]:
    metadata = sample_ref.as_metadata()
    metadata.update(dict(model_inputs.provenance))
    metadata.update(dict(model_inputs.condition_provenance))
    sample: dict[str, object] = {
        "source_rgb": model_inputs.source_rgb,
        "source_depth_m": model_inputs.source_depth_m,
        "source_depth_valid": model_inputs.source_depth_valid,
        "source_seg_condition": model_inputs.source_seg_condition,
        "source_xyz_ref": model_inputs.source_xyz_ref,
        "source_condition_valid": model_inputs.source_condition_valid,
        "context": model_inputs.context,
        "context_mask": model_inputs.context_mask,
        "proprio": model_inputs.proprio,
        "action": action_target.action,
        "action_is_pad": action_target.action_is_pad,
        "momap": momap_target.momap,
        **dict(momap_target.mask_items()),
        "momap_dt_seconds": momap_target.dt_seconds,
        METADATA_KEY: metadata,
    }
    if model_inputs.source_head_seg_gt_condition is not None:
        sample["source_head_seg_gt_condition"] = (
            model_inputs.source_head_seg_gt_condition
        )
        sample["source_head_seg_pred_condition"] = (
            model_inputs.source_head_seg_pred_condition
        )
    if momap_target.component_valid is not None:
        sample[MOMAP_COMPONENT_VALID_KEY] = momap_target.component_valid
    if momap_target.loss_valid is not None:
        sample[MOMAP_LOSS_VALID_KEY] = momap_target.loss_valid
    return sample


def validate_training_sample(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    assert_no_private_or_unknown_fields(sample)
    tensor_keys = (*TENSOR_SAMPLE_KEYS, *(
        key for key in OPTIONAL_TENSOR_SAMPLE_KEYS if key in sample
    ))
    for key in tensor_keys:
        if not isinstance(sample[key], torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")

    model_inputs = ModelInputs(
        source_rgb=sample["source_rgb"],
        source_depth_m=sample["source_depth_m"],
        source_depth_valid=sample["source_depth_valid"],
        source_seg_condition=sample["source_seg_condition"],
        source_xyz_ref=sample["source_xyz_ref"],
        source_condition_valid=sample["source_condition_valid"],
        context=sample["context"],
        context_mask=sample["context_mask"],
        proprio=sample["proprio"],
        provenance={
            field: sample[METADATA_KEY][field]
            for field in LANGUAGE_PROVENANCE_FIELDS
        },
        condition_provenance={
            field: sample[METADATA_KEY][field]
            for field in SOURCE_CONDITION_PROVENANCE_FIELDS
        },
        source_head_seg_gt_condition=sample.get(
            "source_head_seg_gt_condition"
        ),
        source_head_seg_pred_condition=sample.get(
            "source_head_seg_pred_condition"
        ),
    )
    action_target = ActionTarget(
        action=sample["action"],
        action_is_pad=sample["action_is_pad"],
    )
    momap_target = MoMapTarget(
        momap=sample["momap"],
        geom_valid=sample["momap_geom_valid"],
        visibility_evaluated=sample["momap_visibility_evaluated"],
        visible=sample["momap_visible"],
        occluded=sample["momap_occluded"],
        out_of_frame=sample["momap_out_of_frame"],
        depth_conflict=sample["momap_depth_conflict"],
        unsupported=sample["momap_unsupported"],
        unmapped=sample["momap_unmapped"],
        sampling_ambiguous=sample["momap_sampling_ambiguous"],
        dt_seconds=sample["momap_dt_seconds"],
        component_valid=sample.get(MOMAP_COMPONENT_VALID_KEY),
        loss_valid=sample.get(MOMAP_LOSS_VALID_KEY),
    )
    metadata = sample[METADATA_KEY]
    sample_ref = SampleRef(
        episode_id=str(metadata["episode_id"]),
        episode_path=str(metadata["episode_path"]),
        source_index=int(metadata["source_index"]),
        target_indices=tuple(int(value) for value in metadata["target_indices"]),
        future_offsets=tuple(int(value) for value in metadata["future_offsets"]),
        future_horizons_seconds=(
            None
            if metadata["future_horizons_seconds"] is None
            else tuple(
                float(value) for value in metadata["future_horizons_seconds"]
            )
        ),
        future_offset_basis=str(metadata["future_offset_basis"]),
        camera_names=tuple(str(value) for value in metadata["camera_names"]),
        action_start=int(metadata["action_start"]),
        action_stop=int(metadata["action_stop"]),
        coordinate_frame=str(metadata["coordinate_frame"]),
        action_indices=tuple(int(value) for value in metadata["action_indices"]),
        action_query_times_sec=tuple(
            float(value) for value in metadata["action_query_times_sec"]
        ),
        action_sampling_mode=str(metadata["action_sampling_mode"]),
        action_rate_hz=(
            None
            if metadata["action_rate_hz"] is None
            else float(metadata["action_rate_hz"])
        ),
        action_contract_schema_version=str(
            metadata["action_contract_schema_version"]
        ),
        action_label_kind=str(metadata["action_label_kind"]),
        action_is_high_level_policy_action=bool(
            metadata["action_is_high_level_policy_action"]
        ),
        action_normalization=str(metadata["action_normalization"]),
        proprio_mode=str(metadata["proprio_mode"]),
        proprio_normalization=str(metadata["proprio_normalization"]),
        momap_normalization=str(metadata["momap_normalization"]),
    )
    if model_inputs.source_rgb.shape[0] != len(sample_ref.camera_names):
        raise ValueError("source condition V does not match metadata camera_names")
    if momap_target.momap.shape[0] != len(sample_ref.camera_names):
        raise ValueError("momap V does not match metadata camera_names")
    if tuple(model_inputs.source_rgb.shape[-2:]) != tuple(
        momap_target.momap.shape[-2:]
    ):
        raise ValueError(
            "source condition grid must match the MoMap target grid"
        )
    if (
        sample_ref.coordinate_frame
        != f"{model_inputs.condition_provenance['source_condition_reference_camera']}"
        "@source_t0"
    ):
        raise ValueError(
            "metadata.coordinate_frame must identify the source condition "
            "reference camera at source_t0"
        )
    if momap_target.momap.shape[1] != len(sample_ref.target_indices):
        raise ValueError("momap K does not match metadata target_indices")
    if momap_target.momap.shape[1] != len(sample_ref.future_offsets):
        raise ValueError("momap K does not match metadata future_offsets")
    expected_action_horizon = len(sample_ref.action_indices)
    if action_target.action.shape[0] != expected_action_horizon:
        raise ValueError(
            "action Ta does not match metadata action_indices: "
            f"Ta={action_target.action.shape[0]}, expected={expected_action_horizon}"
        )

    for key in tensor_keys:
        value = sample[key]
        if value.is_floating_point():
            _require_finite(value, key)
    return sample
