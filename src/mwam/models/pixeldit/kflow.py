"""Canonical K-flow tensor contract and RoboTwin batch adaptation.

The only supported channel layout is horizon-major XYZ:

``[k0.dx, k0.dy, k0.dz, k1.dx, k1.dy, k1.dz, ...]``.

Keeping these operations in one module prevents the DataLoader adapter, model,
loss, sampler, and checkpoint code from silently disagreeing about ``K``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .conditioning import CANONICAL_REFERENCE_FRAME, WORLD_CONDITIONING_V2


KFLOW_CHANNEL_ORDER = "k_major_xyz"
KFLOW_TARGET_REPRESENTATION_CANONICAL_DXYZ = (
    "canonical_dxyz_all_views_v1"
)
KFLOW_TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV = (
    "all_views_dxyz_matched_uv_valid_v1"
)
KFLOW_TARGET_REPRESENTATIONS = frozenset(
    {
        KFLOW_TARGET_REPRESENTATION_CANONICAL_DXYZ,
        KFLOW_TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV,
    }
)
KFLOW_UNIT_CANONICAL_METER = "meter"
KFLOW_UNIT_HEAD_METER_WRIST_TILE_PIXEL = (
    "all_views_canonical_meter"
)
SOURCE_CONDITION_PROVENANCE_KEYS = (
    "source_condition_schema_version",
    "source_seg_encoding",
    "source_depth_semantics",
)
SUPPORTED_SOURCE_CONDITION_PROVENANCE = frozenset(
    {
        (
            "robotwin_momap_source_condition_v1",
            "episode_scoped_sha256_rgb24_v1",
            "gt_source_only_xyz_derivation_not_prediction_target",
        ),
        (
            "robotwin_momap_source_condition_quality_seg_depth_noise_v1",
            "gt_episode_or_pred_episode_camera_scoped_sha256_rgb24_v1",
            "gt_source_depth_sensor_noise_augmented_xyz_derivation_v1",
        ),
        (
            "robotwin_momap_source_condition_quality_seg_depth_edge_noise_v2",
            "gt_episode_or_pred_episode_camera_scoped_sha256_rgb24_v1",
            "gt_source_depth_sensor_noise_edge_erode_dilate_xyz_derivation_v2",
        ),
    }
)
_HEAD_SEG_CANDIDATE_ENCODING = (
    "head_gt_actor_rgb_plus_pred_"
    "role_linear_0_to_1_track_sha256_rg_episode_camera_v1_candidates_v1"
)
_STRICT_PRED_HEAD_TRACK_ENCODING = (
    "pred_track_episode_camera_scoped_sha256_rgb24_v1"
)
SUPPORTED_SOURCE_CONDITION_PROVENANCE = (
    SUPPORTED_SOURCE_CONDITION_PROVENANCE
    | frozenset(
        {
            (
                "robotwin_momap_source_condition_quality_seg_depth_noise_v1",
                _HEAD_SEG_CANDIDATE_ENCODING,
                "gt_source_depth_sensor_noise_augmented_xyz_derivation_v1",
            ),
            (
                "robotwin_momap_source_condition_quality_seg_depth_edge_noise_v2",
                _HEAD_SEG_CANDIDATE_ENCODING,
                "gt_source_depth_sensor_noise_edge_erode_dilate_xyz_derivation_v2",
            ),
            (
                "robotwin_momap_source_condition_quality_seg_depth_noise_v1",
                _HEAD_SEG_CANDIDATE_ENCODING,
                "gt_source_only_xyz_derivation_not_prediction_target",
            ),
            (
                "robotwin_momap_source_condition_quality_seg_depth_noise_v1",
                _STRICT_PRED_HEAD_TRACK_ENCODING,
                "gt_source_depth_sensor_noise_augmented_xyz_derivation_v1",
            ),
            (
                "robotwin_momap_source_condition_quality_seg_depth_edge_noise_v2",
                _STRICT_PRED_HEAD_TRACK_ENCODING,
                "gt_source_depth_sensor_noise_edge_erode_dilate_xyz_derivation_v2",
            ),
        }
    )
)
# Kept for exact replay of pre-full-resolution checkpoints.
KFLOW_ARCHITECTURE_VERSION = "momap_wam_rgb_seg_xyz_offset_26x26_v2"
# Kept for exact replay of the first full-resolution topology, whose 24-D
# velocity head passed through a 16-D local pixel state.
LEGACY_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_26x26_v3"
)
LEGACY_DPIX24_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_26x26_dpix24_v4"
)
FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_26x26_lgf128_v5"
)
FASTWAM_FULLRES_KFLOW_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_26x26_lgf128_raw24res_v6"
)
# K=1 experiment: retain V6's raw-flow residual topology while making the
FASTWAM_FULLRES_KFLOW_K1_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_26x26_lgf128_raw3res_k1_v1"
)
FASTWAM_FULLRES_KFLOW_LOCAL24_FUSION_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_26x26_lgf128_dpix24_v7"
)
FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_h8s4_g26p4_a30_"
    "lgf128_raw24res_dpix24_v7"
)
# Same V7 H8-S4/Local24/raw-flow-residual World, with the Action30 tail
# consuming one corresponding PiT compact K/V sequence and no concatenated
# final-global sequence.
FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_h8s4_g26p4_a30pitkv_"
    "lgf128_raw24res_dpix24_v8"
)
# Same V7 World, but every Global/PiT native QK-normalized/RoPE K and native V
# is independently adapted and mixed once with the corresponding Action layer.
FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_h8s4_g26p4_a30allnativekvmot_"
    "lgf128_raw24res_dpix24_v10"
)
# Matched no-residual counterpart: keep Local24 and the all-native K/V MoT,
# while preserving the Stage-1 local_global_fusion_v1 World output head.
FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_h8s4_g26p4_a30allnativekvmot_"
    "lgf128_nores_dpix24_v13"
)
# Frozen-World Action redesign: fourteen actuator tokens each carry the full
# 32-step trajectory and use PixelDiT-width native K/V without adapters.
FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_h8s4_g26p4_"
    "a30linear14alignednativekv_lgf128_nores_dpix24_v14"
)
# Paired control: identical V7 H8-S4/Local24/raw-flow-residual World, but only
# Action0..25 exist and consume Global0..25; PiT remains World-only.
FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_h8s4_g26p4_a26global_"
    "lgf128_raw24res_dpix24_v8ablation"
)
# Action26 counterpart of the all-native MoT path: Global0..25 native K/V
# feed Action0..25, while PiT0..3 remain strictly World/KFlow-only.
FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_h8s4_g26p4_"
    "a26globalnativekvmot_lgf128_raw24res_dpix24_v12"
)
FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_h32s1_g26p4_a30_"
    "lgf128_raw96res_dpix96_v7"
)
# Same H32-S1/Local96/Flow96 World, with all 30 Action layers consuming the
# corresponding Global/PiT native K/V through independent 1152->3072 adapters.
FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_h32s1_g26p4_a30allnativekvmot_"
    "lgf128_raw96res_dpix96_v11"
)
FASTWAM_FULLRES_KFLOW_H8S4_LOCAL16_RAW_FLOW_RESIDUAL_ABLATION_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_h8s4_g26p4_a30_"
    "lgf128_raw24res_dpix16_v7ablation"
)
FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION = (
    "momap_wam_rgb_seg_xyz_offset_384x320_h8s4_g26p4_a30_"
    "lgf128_nores_dpix24_v7ablation"
)
# Fresh K=4 C/W Hybrid model: Head carries four dUV horizons while the two
# wrist tiles carry four dUV horizons.  The packed carrier remains the explicit
# horizon-major 3*K=12 channels used at the Dataset/adapter boundary.
FASTWAM_FULLRES_KFLOW_K4_HYBRID_CW_ACTION_ARCHITECTURE_VERSION = (
    "momap_wam_cw_hybrid_allxyz_384x320_k4s8_g26p4_a30_v15"
)
SUPPORTED_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSIONS = frozenset(
    {
        LEGACY_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
        LEGACY_DPIX24_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_K1_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_LOCAL24_FUSION_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H8S4_LOCAL16_RAW_FLOW_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_K4_HYBRID_CW_ACTION_ARCHITECTURE_VERSION,
    }
)
ROBOTWIN_CAMERA_ORDER = ("head_camera", "left_camera", "right_camera")
ROBOTWIN_FASTWAM_VIEW_MODE = "robotwin_fastwam_mosaic_384x320_v1"
ROBOTWIN_NATIVE_CAMERA_SIZE = (240, 320)
ROBOTWIN_FASTWAM_HEAD_SIZE = (256, 320)
ROBOTWIN_FASTWAM_WRIST_SIZE = (128, 160)
ROBOTWIN_FASTWAM_MOSAIC_SIZE = (384, 320)
_TIMING_ATOL_SECONDS = 1.0e-7
_H8S4_FUTURE_OFFSETS = (4, 8, 12, 16, 20, 24, 28, 32)
_K4S8_HYBRID_FUTURE_OFFSETS = (8, 16, 24, 32)
_H32S1_FUTURE_OFFSETS = tuple(range(1, 33))


def _positive_increasing_offsets(values: tuple[int, ...]) -> tuple[int, ...]:
    offsets = tuple(int(value) for value in values)
    if not offsets:
        raise ValueError("`future_offsets` must contain at least one offset.")
    if any(value <= 0 for value in offsets):
        raise ValueError(
            f"`future_offsets` must contain positive frame offsets, got {offsets}."
        )
    if any(left >= right for left, right in zip(offsets, offsets[1:])):
        raise ValueError(
            f"`future_offsets` must be strictly increasing, got {offsets}."
        )
    return offsets


def _positive_increasing_horizons(values: Sequence[float]) -> tuple[float, ...]:
    horizons = tuple(float(value) for value in values)
    if not horizons:
        raise ValueError("`future_horizons_seconds` must contain at least one horizon.")
    if any(not math.isfinite(value) or value <= 0.0 for value in horizons):
        raise ValueError(
            "`future_horizons_seconds` must be finite and positive, "
            f"got {horizons}."
        )
    if any(left >= right for left, right in zip(horizons, horizons[1:])):
        raise ValueError(
            "`future_horizons_seconds` must be strictly increasing, "
            f"got {horizons}."
        )
    return horizons


@dataclass(frozen=True)
class KFlowContract:
    """Resolved, checkpointed semantics for one MoMap-WAM architecture."""

    future_offsets: tuple[int, ...]
    future_horizons_seconds: tuple[float, ...]
    future_offset_basis: str
    view_mode: str
    target_representation: str = KFLOW_TARGET_REPRESENTATION_CANONICAL_DXYZ
    source_coordinate_frame: str = CANONICAL_REFERENCE_FRAME
    source_normalization: str = "none"
    coordinate_frame: str = CANONICAL_REFERENCE_FRAME
    unit: str = KFLOW_UNIT_CANONICAL_METER
    channel_order: str = KFLOW_CHANNEL_ORDER
    camera_order: tuple[str, ...] = ROBOTWIN_CAMERA_ORDER
    architecture_version: str = KFLOW_ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "future_offsets",
            _positive_increasing_offsets(tuple(self.future_offsets)),
        )
        object.__setattr__(
            self,
            "future_horizons_seconds",
            _positive_increasing_horizons(self.future_horizons_seconds),
        )
        if len(self.future_horizons_seconds) != len(self.future_offsets):
            raise ValueError(
                "future offset/horizon length mismatch: "
                f"{len(self.future_offsets)} offsets vs "
                f"{len(self.future_horizons_seconds)} horizons."
            )
        object.__setattr__(
            self,
            "camera_order",
            tuple(str(name) for name in self.camera_order),
        )
        supported_view_modes = {
            "head",
            "robotwin_mosaic",
            ROBOTWIN_FASTWAM_VIEW_MODE,
        }
        if self.view_mode not in supported_view_modes:
            raise ValueError(
                f"`view_mode` must be one of {sorted(supported_view_modes)}, "
                f"got {self.view_mode!r}."
            )
        if (
            self.view_mode in {"robotwin_mosaic", ROBOTWIN_FASTWAM_VIEW_MODE}
            and self.camera_order != ROBOTWIN_CAMERA_ORDER
        ):
            raise ValueError(
                f"{self.view_mode} requires camera order "
                f"{ROBOTWIN_CAMERA_ORDER}, got {self.camera_order}."
            )
        if self.channel_order != KFLOW_CHANNEL_ORDER:
            raise ValueError(
                f"Unsupported K-flow channel order {self.channel_order!r}; "
                f"expected {KFLOW_CHANNEL_ORDER!r}."
            )
        if self.target_representation not in KFLOW_TARGET_REPRESENTATIONS:
            raise ValueError(
                "Unsupported K-flow target representation "
                f"{self.target_representation!r}; expected one of "
                f"{sorted(KFLOW_TARGET_REPRESENTATIONS)}."
            )
        if (
            self.target_representation
            == KFLOW_TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV
            and self.view_mode == "head"
        ):
            raise ValueError(
                "Head/Wrist hybrid Flow requires a multi-view mosaic."
            )
        if not self.coordinate_frame:
            raise ValueError("`coordinate_frame` must be non-empty.")
        if self.source_normalization != "none":
            raise ValueError(
                "Raw RoboTwin MoMap must use source_normalization='none', "
                f"got {self.source_normalization!r}."
            )
        expected_architectures = (
            SUPPORTED_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSIONS
            if self.view_mode == ROBOTWIN_FASTWAM_VIEW_MODE
            else frozenset({KFLOW_ARCHITECTURE_VERSION})
        )
        if self.architecture_version not in expected_architectures:
            raise ValueError(
                f"KFlow view mode {self.view_mode!r} requires architecture "
                f"in {sorted(expected_architectures)!r}, got "
                f"{self.architecture_version!r}."
            )
        if (
            self.architecture_version
            == FASTWAM_FULLRES_KFLOW_K1_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION
            and self.k != 1
        ):
            raise ValueError(
                "The K1/raw3 residual architecture requires exactly one "
                f"future horizon, got K={self.k}."
            )
        if (
            self.architecture_version in {
                FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
                FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION,
                FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
                FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
                FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION,
                FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
                FASTWAM_FULLRES_KFLOW_H8S4_LOCAL16_RAW_FLOW_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
                FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
            }
            and self.future_offsets != _H8S4_FUTURE_OFFSETS
        ):
            raise ValueError(
                "The V7 H8-S4 architecture requires exact future_offsets "
                f"{_H8S4_FUTURE_OFFSETS}, got {self.future_offsets}."
            )
        if (
            self.architecture_version
            in {
                FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
                FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
            }
            and self.future_offsets != _H32S1_FUTURE_OFFSETS
        ):
            raise ValueError(
                "The V7 H32-S1 architecture requires exact future_offsets "
                f"{_H32S1_FUTURE_OFFSETS}, got {self.future_offsets}."
            )
        if (
            self.architecture_version
            == FASTWAM_FULLRES_KFLOW_K4_HYBRID_CW_ACTION_ARCHITECTURE_VERSION
        ):
            if self.future_offsets != _K4S8_HYBRID_FUTURE_OFFSETS:
                raise ValueError(
                    "The K4-S8 Hybrid C/W architecture requires exact "
                    f"future_offsets {_K4S8_HYBRID_FUTURE_OFFSETS}, got "
                    f"{self.future_offsets}."
                )
            if (
                self.target_representation
                != KFLOW_TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV
            ):
                raise ValueError(
                    "The K4-S8 Hybrid C/W architecture requires the explicit "
                    "All-view-dXYZ target representation."
                )
        frame_pair = (self.source_coordinate_frame, self.coordinate_frame)
        expected_frames = (
            CANONICAL_REFERENCE_FRAME,
            CANONICAL_REFERENCE_FRAME,
        )
        if frame_pair != expected_frames:
            raise ValueError(
                f"KFlow architecture {self.architecture_version!r} requires "
                f"source/packed frames {expected_frames}, got {frame_pair}."
            )
        if not str(self.future_offset_basis).strip():
            raise ValueError("`future_offset_basis` must be non-empty.")
        expected_unit = (
            KFLOW_UNIT_HEAD_METER_WRIST_TILE_PIXEL
            if self.target_representation
            == KFLOW_TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV
            else KFLOW_UNIT_CANONICAL_METER
        )
        if self.unit != expected_unit:
            raise ValueError(
                "MoMap unit/representation mismatch: "
                f"representation={self.target_representation!r} requires "
                f"unit={expected_unit!r}, got {self.unit!r}."
            )
        if not self.architecture_version:
            raise ValueError("`architecture_version` must be non-empty.")

    @property
    def k(self) -> int:
        return len(self.future_offsets)

    @property
    def flow_channels(self) -> int:
        return 3 * self.k

    @property
    def conditioning_mode(self) -> str:
        return WORLD_CONDITIONING_V2

    @property
    def uses_common_reference(self) -> bool:
        return True

    @property
    def requires_full_rank_pixel_path(self) -> bool:
        return self.architecture_version in {
            LEGACY_DPIX24_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_LOCAL24_FUSION_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_K4_HYBRID_CW_ACTION_ARCHITECTURE_VERSION,
        }

    @property
    def requires_full_rank_output_path(self) -> bool:
        return self.architecture_version in {
            LEGACY_DPIX24_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_K1_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_LOCAL24_FUSION_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL16_RAW_FLOW_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_K4_HYBRID_CW_ACTION_ARCHITECTURE_VERSION,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "future_offsets": list(self.future_offsets),
            "future_horizons_seconds": list(self.future_horizons_seconds),
            "future_offset_basis": self.future_offset_basis,
            "k": self.k,
            "flow_channels": self.flow_channels,
            "view_mode": self.view_mode,
            "target_representation": self.target_representation,
            "camera_order": list(self.camera_order),
            "source_coordinate_frame": self.source_coordinate_frame,
            "source_normalization": self.source_normalization,
            "coordinate_frame": self.coordinate_frame,
            "unit": self.unit,
            "channel_order": self.channel_order,
            "architecture_version": self.architecture_version,
            "conditioning_mode": self.conditioning_mode,
        }


def pack_kflow(flow: torch.Tensor, *, expected_k: int | None = None) -> torch.Tensor:
    """Pack ``[B,K,3,H,W]`` into horizon-major ``[B,3K,H,W]``."""

    if not isinstance(flow, torch.Tensor):
        raise TypeError(f"`flow` must be a torch.Tensor, got {type(flow).__name__}.")
    if flow.ndim != 5 or flow.shape[2] != 3:
        raise ValueError(
            f"`flow` must have shape [B,K,3,H,W], got {tuple(flow.shape)}."
        )
    k = int(flow.shape[1])
    if k <= 0:
        raise ValueError("K-flow tensors must contain at least one horizon.")
    if expected_k is not None and k != int(expected_k):
        raise ValueError(f"K-flow horizon mismatch: tensor K={k}, expected {expected_k}.")
    return flow.contiguous().view(flow.shape[0], 3 * k, flow.shape[3], flow.shape[4])


def unpack_kflow(flow: torch.Tensor, *, k: int) -> torch.Tensor:
    """Unpack horizon-major ``[B,3K,H,W]`` into ``[B,K,3,H,W]``."""

    if not isinstance(flow, torch.Tensor):
        raise TypeError(f"`flow` must be a torch.Tensor, got {type(flow).__name__}.")
    if flow.ndim != 4:
        raise ValueError(f"`flow` must have shape [B,3K,H,W], got {tuple(flow.shape)}.")
    if isinstance(k, bool) or int(k) != k or int(k) <= 0:
        raise ValueError(f"`k` must be a positive integer, got {k!r}.")
    k = int(k)
    if flow.shape[1] != 3 * k:
        raise ValueError(
            f"Packed K-flow has {flow.shape[1]} channels, expected {3 * k} for K={k}."
        )
    return flow.contiguous().view(flow.shape[0], k, 3, flow.shape[2], flow.shape[3])


def _cell_center_indices(length: int, count: int, device: torch.device) -> torch.Tensor:
    if length <= 0 or count <= 0 or count > length:
        raise ValueError(
            f"Invalid cell-center selection length={length}, count={count}."
        )
    index = torch.floor(
        (torch.arange(count, device=device, dtype=torch.float64) + 0.5)
        * (float(length) / float(count))
    ).to(dtype=torch.long)
    return index.clamp_(0, length - 1)


def _resize_cell_center_indices(
    source_length: int,
    target_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return PyTorch ``nearest-exact`` indices, including for upsampling."""

    if source_length <= 0 or target_length <= 0:
        raise ValueError(
            "Spatial resize dimensions must be positive, got "
            f"{source_length}->{target_length}."
        )
    index = torch.floor(
        (torch.arange(target_length, device=device, dtype=torch.float64) + 0.5)
        * (float(source_length) / float(target_length))
    ).to(dtype=torch.long)
    return index.clamp_(0, source_length - 1)


def _source_point_resize(
    value: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    rows = _resize_cell_center_indices(
        int(value.shape[-2]),
        int(size[0]),
        value.device,
    )
    columns = _resize_cell_center_indices(
        int(value.shape[-1]),
        int(size[1]),
        value.device,
    )
    return value.index_select(-2, rows).index_select(-1, columns)


def _continuous_rgb_resize(
    value: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    if not value.is_floating_point():
        raise TypeError("Continuous RGB resize requires a floating tensor.")
    source_shape = tuple(value.shape)
    flat = value.reshape(-1, 1, source_shape[-2], source_shape[-1])
    resized = F.interpolate(
        flat,
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.reshape(*source_shape[:-2], *size)


def robotwin_view_mosaic(views: torch.Tensor) -> torch.Tensor:
    """Place head above two half-resolution wrist views without interpolation.

    ``views`` may have arbitrary dimensions between ``V`` and ``H,W``:
    ``[B,3,...,H,W]`` becomes ``[B,...,H+H/2,W]``.  Wrist samples use the
    same deterministic cell-center rule for RGB, MoMap, and masks.
    """

    if not isinstance(views, torch.Tensor):
        raise TypeError(f"`views` must be a torch.Tensor, got {type(views).__name__}.")
    if views.ndim < 5 or views.shape[1] != 3:
        raise ValueError(
            "`views` must be [B,3,...,H,W] with the canonical three cameras, "
            f"got {tuple(views.shape)}."
        )
    height, width = (int(views.shape[-2]), int(views.shape[-1]))
    if height % 2 or width % 2:
        raise ValueError(
            "robotwin_mosaic requires even source-grid dimensions, "
            f"got {(height, width)}."
        )
    rows = _cell_center_indices(height, height // 2, views.device)
    columns = _cell_center_indices(width, width // 2, views.device)
    head = views[:, 0]
    left = views[:, 1].index_select(-2, rows).index_select(-1, columns)
    right = views[:, 2].index_select(-2, rows).index_select(-1, columns)
    bottom = torch.cat([left, right], dim=-1)
    return torch.cat([head, bottom], dim=-2)


def robotwin_fastwam_view_mosaic(
    views: torch.Tensor,
    *,
    sampling: str,
) -> torch.Tensor:
    """Pack native RoboTwin cameras into FastWAM's exact 384x320 layout.

    The Dataset must provide all native ``240x320`` camera cells. RGB follows
    the original FastWAM bilinear+antialias transform. Segmentation, canonical
    XYZ, KFlow, and masks use one nearest-exact source-point map so categorical
    identities and geometric boundaries are never blended.
    """

    if not isinstance(views, torch.Tensor):
        raise TypeError(f"`views` must be a torch.Tensor, got {type(views).__name__}.")
    if views.ndim < 5 or views.shape[1] != 3:
        raise ValueError(
            "`views` must be [B,3,...,H,W] with the canonical three cameras, "
            f"got {tuple(views.shape)}."
        )
    if tuple(views.shape[-2:]) != ROBOTWIN_NATIVE_CAMERA_SIZE:
        raise ValueError(
            f"{ROBOTWIN_FASTWAM_VIEW_MODE} requires native camera size "
            f"{ROBOTWIN_NATIVE_CAMERA_SIZE}, got {tuple(views.shape[-2:])}."
        )
    if sampling == "continuous_rgb":
        resize = _continuous_rgb_resize
    elif sampling == "source_point":
        resize = _source_point_resize
    else:
        raise ValueError(
            "`sampling` must be 'continuous_rgb' or 'source_point', "
            f"got {sampling!r}."
        )

    head = resize(views[:, 0], ROBOTWIN_FASTWAM_HEAD_SIZE)
    left = resize(views[:, 1], ROBOTWIN_FASTWAM_WRIST_SIZE)
    right = resize(views[:, 2], ROBOTWIN_FASTWAM_WRIST_SIZE)
    bottom = torch.cat([left, right], dim=-1)
    result = torch.cat([head, bottom], dim=-2)
    if tuple(result.shape[-2:]) != ROBOTWIN_FASTWAM_MOSAIC_SIZE:
        raise RuntimeError(
            "FastWAM RoboTwin mosaic produced an unexpected spatial size: "
            f"{tuple(result.shape[-2:])}."
        )
    return result


class MoMapBatchAdapter:
    """Convert the strict V1 batch into the whitelist model/loss contract."""

    _COMMON_REFERENCE_SOURCE_KEYS = {
        "source_rgb",
        "source_seg_condition",
        "source_xyz_ref",
        "source_condition_valid",
    }

    def __init__(
        self,
        contract: KFlowContract,
        *,
        action_label_kind: str,
        action_is_high_level_policy_action: bool,
        action_normalization: str,
        proprio_mode: str,
        proprio_normalization: str,
    ) -> None:
        self.contract = contract
        semantic_strings = {
            "action_label_kind": action_label_kind,
            "action_normalization": action_normalization,
            "proprio_mode": proprio_mode,
            "proprio_normalization": proprio_normalization,
        }
        for name, value in semantic_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`{name}` must be a non-empty string.")
        if not isinstance(action_is_high_level_policy_action, bool):
            raise TypeError("`action_is_high_level_policy_action` must be bool.")
        if action_is_high_level_policy_action:
            raise ValueError(
                "MoMap-WAM requires low-level actions; "
                "action_is_high_level_policy_action must be false."
            )
        if action_normalization != "none" or proprio_normalization != "none":
            raise ValueError(
                "MoMapBatchAdapter requires raw action/proprio metadata with "
                "normalization='none'; per-sample normalization is forbidden."
            )
        self.action_label_kind = action_label_kind
        self.action_is_high_level_policy_action = False
        self.action_normalization = action_normalization
        self.proprio_mode = proprio_mode
        self.proprio_normalization = proprio_normalization

    def semantic_contract(self) -> dict[str, Any]:
        return {
            "action_label_kind": self.action_label_kind,
            "action_is_high_level_policy_action": False,
            "action_normalization": self.action_normalization,
            "proprio_mode": self.proprio_mode,
            "proprio_normalization": self.proprio_normalization,
        }

    @staticmethod
    def _tensor(sample: Mapping[str, Any], key: str) -> torch.Tensor:
        value = sample.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"sample[{key!r}] must be a torch.Tensor.")
        return value

    def _common_reference_sources(
        self,
        sample: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        missing = self._COMMON_REFERENCE_SOURCE_KEYS - set(sample)
        if missing:
            raise KeyError(
                "Common-reference source observation is missing fields: "
                f"{sorted(missing)}."
            )
        rgb = self._tensor(sample, "source_rgb")
        segmentation = self._tensor(sample, "source_seg_condition")
        xyz = self._tensor(sample, "source_xyz_ref")
        valid = self._tensor(sample, "source_condition_valid")
        if rgb.ndim != 5 or rgb.shape[1:3] != (3, 3):
            raise ValueError(
                f"`source_rgb` must be [B,3,3,H,W], got {tuple(rgb.shape)}."
            )
        if segmentation.shape != rgb.shape or xyz.shape != rgb.shape:
            raise ValueError(
                "source segmentation/XYZ must match source_rgb [B,3,3,H,W]."
            )
        expected_valid = (rgb.shape[0], 3, 1, rgb.shape[-2], rgb.shape[-1])
        if tuple(valid.shape) != expected_valid or valid.dtype != torch.bool:
            raise ValueError(
                "`source_condition_valid` must be bool "
                f"{expected_valid}, got {tuple(valid.shape)} {valid.dtype}."
            )
        for name, value in (
            ("source_rgb", rgb),
            ("source_seg_condition", segmentation),
            ("source_xyz_ref", xyz),
        ):
            if not value.is_floating_point():
                raise TypeError(f"`{name}` must be floating point.")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"`{name}` contains NaN or Inf.")
        for name, value in (
            ("source_rgb", rgb),
            ("source_seg_condition", segmentation),
        ):
            if (
                float(value.detach().amin().item()) < -1.0001
                or float(value.detach().amax().item()) > 1.0001
            ):
                raise ValueError(f"`{name}` must lie in [-1,1].")
        invalid = ~valid.expand_as(segmentation)
        if bool((segmentation[invalid] != 0).any().item()) or bool(
            (xyz[invalid] != 0).any().item()
        ):
            raise ValueError(
                "Invalid source segmentation and XYZ must be exact neutral zero."
            )
        return rgb, segmentation, xyz, valid

    def _pack_common_reference_sources(
        self,
        sample: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        rgb, segmentation, xyz, valid = self._common_reference_sources(sample)
        if self.contract.view_mode == "head":
            packed = (rgb[:, 0], segmentation[:, 0], xyz[:, 0], valid[:, 0])
        elif self.contract.view_mode == "robotwin_mosaic":
            packed = tuple(
                robotwin_view_mosaic(value)
                for value in (rgb, segmentation, xyz, valid)
            )
        else:
            packed = (
                robotwin_fastwam_view_mosaic(
                    rgb,
                    sampling="continuous_rgb",
                ),
                robotwin_fastwam_view_mosaic(
                    segmentation,
                    sampling="source_point",
                ),
                robotwin_fastwam_view_mosaic(
                    xyz,
                    sampling="source_point",
                ),
                robotwin_fastwam_view_mosaic(
                    valid,
                    sampling="source_point",
                ),
            )
        packed_rgb, packed_segmentation, packed_xyz, packed_valid = packed
        packed_segmentation = torch.where(
            packed_valid, packed_segmentation, torch.zeros_like(packed_segmentation)
        )
        packed_xyz = torch.where(
            packed_valid, packed_xyz, torch.zeros_like(packed_xyz)
        )
        result = {
            "source_rgb": packed_rgb,
            "source_seg_condition": packed_segmentation,
            "source_xyz_ref": packed_xyz,
            "source_condition_valid": packed_valid,
        }
        head_candidate_keys = (
            "source_head_seg_gt_condition",
            "source_head_seg_pred_condition",
        )
        present_candidates = [key in sample for key in head_candidate_keys]
        if any(present_candidates) and not all(present_candidates):
            raise KeyError(
                "Head GT/Pred segmentation candidates must be present together."
            )
        for key in head_candidate_keys:
            if key not in sample:
                continue
            candidate = self._tensor(sample, key)
            expected = (rgb.shape[0], 3, rgb.shape[-2], rgb.shape[-1])
            if tuple(candidate.shape) != expected:
                raise ValueError(
                    f"`{key}` must be {expected}, got {tuple(candidate.shape)}."
                )
            if not candidate.is_floating_point() or not bool(
                torch.isfinite(candidate).all().item()
            ):
                raise ValueError(f"`{key}` must be finite floating point.")
            candidate_views = torch.zeros_like(segmentation)
            candidate_views[:, 0] = candidate
            if self.contract.view_mode == "head":
                packed_candidate = candidate
            elif self.contract.view_mode == "robotwin_mosaic":
                packed_candidate = robotwin_view_mosaic(candidate_views)
            else:
                packed_candidate = robotwin_fastwam_view_mosaic(
                    candidate_views,
                    sampling="source_point",
                )
            result[key] = packed_candidate
        return result

    def _common_metadata(
        self,
        sample: Mapping[str, Any],
        *,
        batch_size: int,
        grid_size: tuple[int, int],
    ) -> tuple[dict[str, Any], ...]:
        raw = sample.get("metadata")
        if not isinstance(raw, (list, tuple)) or len(raw) != batch_size:
            raise TypeError(
                f"`metadata` must be list[dict] with B={batch_size} entries."
            )
        metadata: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise TypeError(f"`metadata[{index}]` must be a mapping.")
            metadata.append(dict(item))
        exact = {
            "future_offsets": self.contract.future_offsets,
            "future_horizons_seconds": self.contract.future_horizons_seconds,
            "future_offset_basis": self.contract.future_offset_basis,
            "coordinate_frame": self.contract.coordinate_frame,
            "camera_names": self.contract.camera_order,
            "action_label_kind": self.action_label_kind,
            "action_is_high_level_policy_action": False,
            "action_normalization": self.action_normalization,
            "proprio_mode": self.proprio_mode,
            "proprio_normalization": self.proprio_normalization,
            "momap_normalization": self.contract.source_normalization,
            "source_condition_reference_camera": "head_camera",
            "source_seg_color_seed": "fastwam_momap_seg_condition_v1",
            "source_condition_grid_size": tuple(int(value) for value in grid_size),
        }
        for index, item in enumerate(metadata):
            missing = sorted(
                (set(exact) | set(SOURCE_CONDITION_PROVENANCE_KEYS)) - set(item)
            )
            if missing:
                raise KeyError(f"`metadata[{index}]` missing {missing}.")
            source_condition_provenance = tuple(
                item[key] for key in SOURCE_CONDITION_PROVENANCE_KEYS
            )
            if source_condition_provenance not in SUPPORTED_SOURCE_CONDITION_PROVENANCE:
                actual = {
                    key: item[key] for key in SOURCE_CONDITION_PROVENANCE_KEYS
                }
                raise ValueError(
                    f"`metadata[{index}]` source condition provenance mismatch "
                    f"for {SOURCE_CONDITION_PROVENANCE_KEYS}: {actual!r}; expected "
                    "one complete supported provenance tuple."
                )
            for key, expected in exact.items():
                actual = item[key]
                if key in {
                    "future_offsets",
                    "future_horizons_seconds",
                    "camera_names",
                    "source_condition_grid_size",
                }:
                    actual = tuple(actual) if actual is not None else None
                if actual != expected:
                    raise ValueError(
                        f"`metadata[{index}].{key}` mismatch: "
                        f"{actual!r} vs {expected!r}."
                    )
        return tuple(metadata)

    def _adapt_common_observation(
        self,
        sample: Mapping[str, Any],
    ) -> dict[str, Any]:
        sources = self._pack_common_reference_sources(sample)
        output: dict[str, Any] = {
            **sources,
            "context": self._tensor(sample, "context"),
        }
        if "proprio" in sample:
            output["proprio_mode"] = self.proprio_mode
            output["proprio_normalization"] = self.proprio_normalization
        for key in ("context_mask", "proprio"):
            if key in sample:
                output[key] = sample[key]
        return output

    def _adapt_common_training(
        self,
        sample: Mapping[str, Any],
    ) -> dict[str, Any]:
        required = {
            *self._COMMON_REFERENCE_SOURCE_KEYS,
            "momap",
            "momap_geom_valid",
            "momap_dt_seconds",
            "action",
            "action_is_pad",
            "context",
            "metadata",
        }
        missing = required - set(sample)
        if missing:
            raise KeyError(f"Common-reference batch is missing {sorted(missing)}.")
        sources = self._pack_common_reference_sources(sample)
        rgb_views = self._tensor(sample, "source_rgb")
        momap_views = self._tensor(sample, "momap")
        valid_views = self._tensor(sample, "momap_geom_valid")
        component_valid_views = sample.get("momap_component_valid")
        loss_valid_views = sample.get("momap_loss_valid")
        hybrid_target = (
            self.contract.target_representation
            == KFLOW_TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV
        )
        if hybrid_target != (component_valid_views is not None):
            raise ValueError(
                "KFlow target representation and component-valid presence "
                "disagree."
            )
        requires_loss_valid = (
            self.contract.architecture_version
            == FASTWAM_FULLRES_KFLOW_K4_HYBRID_CW_ACTION_ARCHITECTURE_VERSION
        )
        if requires_loss_valid != (loss_valid_views is not None):
            raise ValueError(
                "KFlow architecture and momap_loss_valid presence disagree."
            )
        batch_size = int(rgb_views.shape[0])
        metadata = self._common_metadata(
            sample,
            batch_size=batch_size,
            grid_size=(int(rgb_views.shape[-2]), int(rgb_views.shape[-1])),
        )
        if (
            momap_views.ndim != 6
            or momap_views.shape[1:4] != (3, self.contract.k, 3)
            or momap_views.shape[0] != batch_size
            or momap_views.shape[-2:] != rgb_views.shape[-2:]
        ):
            raise ValueError(
                "`momap` must be [B,3,K,3,H,W] on the source grid, got "
                f"{tuple(momap_views.shape)}."
            )
        expected_valid = (
            batch_size,
            3,
            self.contract.k,
            1,
            momap_views.shape[-2],
            momap_views.shape[-1],
        )
        if tuple(valid_views.shape) != expected_valid or valid_views.dtype != torch.bool:
            raise ValueError(
                f"`momap_geom_valid` must be bool {expected_valid}, got "
                f"{tuple(valid_views.shape)} {valid_views.dtype}."
            )
        if not momap_views.is_floating_point() or not bool(
            torch.isfinite(momap_views).all().item()
        ):
            raise ValueError("`momap` must be finite floating point.")
        if component_valid_views is not None:
            if not isinstance(component_valid_views, torch.Tensor):
                raise TypeError("`momap_component_valid` must be a tensor.")
            if (
                tuple(component_valid_views.shape) != tuple(momap_views.shape)
                or component_valid_views.dtype != torch.bool
            ):
                raise ValueError(
                    "`momap_component_valid` must be bool and match "
                    f"`momap` {tuple(momap_views.shape)}, got "
                    f"{tuple(component_valid_views.shape)} "
                    f"{component_valid_views.dtype}."
                )
            if bool(
                (
                    component_valid_views
                    & ~valid_views.expand_as(component_valid_views)
                )
                .any()
                .item()
            ):
                raise ValueError(
                    "`momap_component_valid` must be a subset of "
                    "`momap_geom_valid`."
                )
            # Spatial resizing selects source points; metric XYZ values do not scale.
        if loss_valid_views is not None:
            if not isinstance(loss_valid_views, torch.Tensor):
                raise TypeError("`momap_loss_valid` must be a tensor.")
            if (
                tuple(loss_valid_views.shape) != tuple(momap_views.shape)
                or loss_valid_views.dtype != torch.bool
            ):
                raise ValueError(
                    "`momap_loss_valid` must be bool and match `momap` "
                    f"{tuple(momap_views.shape)}, got "
                    f"{tuple(loss_valid_views.shape)} "
                    f"{loss_valid_views.dtype}."
                )
            assert component_valid_views is not None
            if bool(
                (loss_valid_views & ~component_valid_views).any().item()
            ):
                raise ValueError(
                    "`momap_loss_valid` must be a subset of "
                    "`momap_component_valid`."
                )
        if self.contract.view_mode == "head":
            structured_flow = momap_views[:, 0]
            structured_valid = valid_views[:, 0]
            structured_component_valid = (
                None
                if component_valid_views is None
                else component_valid_views[:, 0]
            )
            structured_loss_valid = (
                None
                if loss_valid_views is None
                else loss_valid_views[:, 0]
            )
        elif self.contract.view_mode == "robotwin_mosaic":
            structured_flow = robotwin_view_mosaic(momap_views)
            structured_valid = robotwin_view_mosaic(valid_views)
            structured_component_valid = (
                None
                if component_valid_views is None
                else robotwin_view_mosaic(component_valid_views)
            )
            structured_loss_valid = (
                None
                if loss_valid_views is None
                else robotwin_view_mosaic(loss_valid_views)
            )
        else:
            structured_flow = robotwin_fastwam_view_mosaic(
                momap_views,
                sampling="source_point",
            )
            structured_valid = robotwin_fastwam_view_mosaic(
                valid_views,
                sampling="source_point",
            )
            structured_component_valid = (
                None
                if component_valid_views is None
                else robotwin_fastwam_view_mosaic(
                    component_valid_views,
                    sampling="source_point",
                )
            )
            structured_loss_valid = (
                None
                if loss_valid_views is None
                else robotwin_fastwam_view_mosaic(
                    loss_valid_views,
                    sampling="source_point",
                )
            )
        effective_valid = (
            structured_valid
            if structured_component_valid is None
            else structured_component_valid
        )
        structured_flow = torch.where(
            effective_valid,
            structured_flow,
            torch.zeros_like(structured_flow),
        )

        horizons = torch.tensor(
            self.contract.future_horizons_seconds,
            device=momap_views.device,
            dtype=torch.float64,
        ).unsqueeze(0).expand(batch_size, -1)
        dt_seconds = self._tensor(sample, "momap_dt_seconds")
        if (
            dt_seconds.shape != (batch_size, self.contract.k)
            or not dt_seconds.is_floating_point()
            or not bool(torch.isfinite(dt_seconds).all().item())
        ):
            raise ValueError("`momap_dt_seconds` must be finite floating [B,K].")
        raw_lateness = dt_seconds.to(dtype=torch.float64) - horizons
        if bool((raw_lateness < -_TIMING_ATOL_SECONDS).any().item()):
            raise ValueError("MoMap dt precedes the configured physical horizon.")
        lateness = raw_lateness.clamp_min(0.0).to(dtype=dt_seconds.dtype)
        frame_offsets = torch.tensor(
            [
                tuple(
                    int(target) - int(item["source_index"])
                    for target in item["target_indices"]
                )
                for item in metadata
            ],
            device=dt_seconds.device,
            dtype=torch.long,
        )
        self._check_actual_frame_offsets(frame_offsets, batch_size)

        output: dict[str, Any] = {
            **sources,
            "target_flow": pack_kflow(
                structured_flow,
                expected_k=self.contract.k,
            ),
            "flow_valid": structured_valid,
            "future_offsets": self.contract.future_offsets,
            "future_horizons_seconds": self.contract.future_horizons_seconds,
            "future_offset_basis": self.contract.future_offset_basis,
            "coordinate_frame": self.contract.coordinate_frame,
            "target_representation": self.contract.target_representation,
            "unit": self.contract.unit,
            "source_normalization": self.contract.source_normalization,
            "action_label_kind": self.action_label_kind,
            "action_is_high_level_policy_action": False,
            "action_normalization": self.action_normalization,
            "proprio_mode": self.proprio_mode,
            "proprio_normalization": self.proprio_normalization,
            "frame_offsets": frame_offsets,
            "momap_dt_seconds": dt_seconds,
            "momap_lateness_seconds": lateness,
            "action": self._tensor(sample, "action"),
            "action_is_pad": self._tensor(sample, "action_is_pad"),
            "context": self._tensor(sample, "context"),
        }
        if structured_component_valid is not None:
            output["flow_component_valid"] = structured_component_valid
        if structured_loss_valid is not None:
            output["flow_loss_valid"] = structured_loss_valid
        for key in ("context_mask", "proprio"):
            if key in sample:
                output[key] = sample[key]
        return output

    def _check_actual_frame_offsets(self, offsets: torch.Tensor, batch_size: int) -> None:
        if offsets.shape != (batch_size, self.contract.k):
            raise ValueError(
                f"`momap_frame_offsets` must be [B,K], got {tuple(offsets.shape)}."
            )
        if offsets.dtype == torch.bool or offsets.is_floating_point():
            raise TypeError("`momap_frame_offsets` must use an integer dtype.")
        offsets = offsets.to(dtype=torch.long)
        if bool((offsets <= 0).any().item()) or bool(
            (offsets[:, 1:] <= offsets[:, :-1]).any().item()
        ):
            raise ValueError("`momap_frame_offsets` must be positive and increasing.")

    def adapt_observation(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Build V2 inference conditions without reading future target fields."""

        return self._adapt_common_observation(sample)

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        return self._adapt_common_training(sample)


__all__ = [
    "FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_LOCAL24_FUSION_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL16_RAW_FLOW_RESIDUAL_ABLATION_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_K4_HYBRID_CW_ACTION_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION",
    "KFLOW_ARCHITECTURE_VERSION",
    "KFLOW_CHANNEL_ORDER",
    "KFLOW_TARGET_REPRESENTATION_CANONICAL_DXYZ",
    "KFLOW_TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV",
    "KFLOW_TARGET_REPRESENTATIONS",
    "KFLOW_UNIT_CANONICAL_METER",
    "KFLOW_UNIT_HEAD_METER_WRIST_TILE_PIXEL",
    "KFlowContract",
    "LEGACY_DPIX24_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION",
    "LEGACY_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION",
    "MoMapBatchAdapter",
    "ROBOTWIN_CAMERA_ORDER",
    "ROBOTWIN_FASTWAM_HEAD_SIZE",
    "ROBOTWIN_FASTWAM_MOSAIC_SIZE",
    "ROBOTWIN_FASTWAM_VIEW_MODE",
    "ROBOTWIN_FASTWAM_WRIST_SIZE",
    "ROBOTWIN_NATIVE_CAMERA_SIZE",
    "SUPPORTED_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSIONS",
    "pack_kflow",
    "robotwin_fastwam_view_mosaic",
    "robotwin_view_mosaic",
    "unpack_kflow",
]
