from __future__ import annotations

import copy
import math
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from mwam.utils.checkpoint_integrity import checkpoint_file_identity

from ..wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from .conditioning import WORLD_CONDITIONING_V2
from .geometry_normalization import (
    FixedGlobalSourceP99FlowXYZMeanStdGeometryNormalizationContract,
    GeometryNormalization,
    GeometryNormalizationContract,
    PerSampleP99GeometryNormalizationContract,
    SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract,
)
from .hybrid_normalization import HybridHeadWristNormalizationContract
from .kflow import (
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
    ROBOTWIN_FASTWAM_HEAD_SIZE,
    ROBOTWIN_FASTWAM_MOSAIC_SIZE,
    KFlowContract,
    MoMapBatchAdapter,
    unpack_kflow,
)
from .method_contract import (
    FORMAL_METHOD_CONTRACT_V1,
    FORMAL_METHOD_CONTRACT_V2,
    FORMAL_METHOD_CONTRACT_V3,
    FORMAL_METHOD_CONTRACT_V4,
    FORMAL_METHOD_CONTRACT_V5,
    resolve_formal_method_contract,
    FORMAL_METHOD_CONTRACT_V6,
    FORMAL_METHOD_CONTRACT_V7,
    FORMAL_METHOD_CONTRACT_V8,
    FORMAL_METHOD_CONTRACT_V9,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2,
)
from .model import (
    MOMAP_LOCAL_GLOBAL_FUSION_OUTPUT_HEAD,
    MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD,
)
from .momap_flow import MoMapFlowMatchingObjective
from .normalization import FixedFeatureNormalizationContract
from .pixeldit_action_mot import (
    WorldNativeKVAdapter,
    PixelDiTActionMoT,
    WorldToActionAdapter,
)
from .pixeldit_aligned_action import (
    COSINE_ANNEAL_NATIVE_KV_FUSION_VERSION,
    LEGACY_NATIVE_KV_FUSION_VERSION,
    PixelDiTAlignedActionDiT,
    PixelDiTTemporalActionDiT,
)
from .pixeldit_aligned_action_mot import PixelDiTAlignedActionMoT
from .hybrid_cw_action_mot import HYBRID_CW_NATIVE_KV_FUSION_VERSION
from .action_world_initialization import (
    ACTION14_RANDOM_INITIALIZATION,
    ACTION14_WORLD_TRANSFER_INITIALIZATION,
    initialize_pixeldit_aligned_action,
)


def _atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    """Write one torch payload atomically in the destination directory."""

    destination = Path(path)
    if not destination.name:
        raise ValueError("checkpoint destination must name a file")
    parent = destination.parent
    parent_lstat = parent.lstat()
    if stat.S_ISLNK(parent_lstat.st_mode) or not stat.S_ISDIR(
        parent_lstat.st_mode
    ):
        raise ValueError(
            "checkpoint destination parent must be an existing real directory"
        )
    directory_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    parent_fstat = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(parent_fstat.st_mode)
        or (parent_lstat.st_dev, parent_lstat.st_ino)
        != (parent_fstat.st_dev, parent_fstat.st_ino)
    ):
        os.close(directory_fd)
        raise RuntimeError(
            "checkpoint destination parent changed during validation"
        )

    temporary_fd = -1
    temporary_name_in_parent: str | None = None
    try:
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(parent),
        )
        temporary_name_in_parent = Path(temporary_name).name
        handle = os.fdopen(temporary_fd, "wb")
        temporary_fd = -1
        with handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_name_in_parent,
            destination.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name_in_parent = None
        os.fsync(directory_fd)
    except BaseException:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_name_in_parent is not None:
            try:
                os.unlink(temporary_name_in_parent, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        raise
    finally:
        os.close(directory_fd)


FORMAL_NUMERIC_PRECISION_POLICY_V1: dict[str, str] = {
    "schema_version": "fastwam_momap_numeric_precision_policy_v1",
    "pre_normalized_rgb_seg": (
        "fp32_finite_range_validation_then_model_entry_dtype_v1"
    ),
    "canonical_xyz": "fp32_normalize_and_validity_mask_then_model_entry_dtype_v1",
    "target_kflow": "fp32_normalize_then_scheduler_sample_dtype_v1",
    "action": "fp32_normalize_then_scheduler_sample_dtype_v1",
    "proprio": "fp32_normalize_then_model_entry_dtype_v1",
    "validity_masks": "boolean_masks_never_numeric_normalized_v1",
    "training_timesteps": (
        "fp32_scheduler_coordinate_and_pixel_clean_transform_v1"
    ),
    "scheduler_samples": (
        "model_entry_dtype_allowed_after_fp32_normalization_v1"
    ),
    "loss_accumulation": (
        "fp32_residual_square_mask_weight_and_reduction_v1"
    ),
    "model_entry_dtype": (
        "module_parameter_dtype_bound_by_checkpoint_state_schema_v1"
    ),
}


FLOW_SUPERVISION_ALL_MOSAIC = "all_mosaic"
FLOW_SUPERVISION_HEAD_TILE_ONLY = "head_tile_only"
FLOW_SUPERVISION_SCOPES = frozenset(
    {FLOW_SUPERVISION_ALL_MOSAIC, FLOW_SUPERVISION_HEAD_TILE_ONLY}
)


def apply_flow_supervision_scope(
    target_flow: torch.Tensor,
    flow_valid: torch.Tensor,
    *,
    scope: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the frozen spatial MoMap supervision boundary before FM noising.

    Head-only training keeps the three-camera source observation untouched.  It
    removes side-camera GT from both the velocity loss and the clean Flow tensor
    that is mixed with Gaussian noise, so side GT cannot become a hidden model
    input through the flow-matching state.
    """

    if scope not in FLOW_SUPERVISION_SCOPES:
        raise ValueError(
            "`flow_supervision_scope` must be one of "
            f"{sorted(FLOW_SUPERVISION_SCOPES)}, got {scope!r}."
        )
    if scope == FLOW_SUPERVISION_ALL_MOSAIC:
        return target_flow, flow_valid
    if target_flow.ndim != 4:
        raise ValueError("`target_flow` must be packed [B,3K,H,W].")
    if flow_valid.ndim != 5 or flow_valid.shape[2] != 1:
        raise ValueError("`flow_valid` must be [B,K,1,H,W].")
    if flow_valid.dtype != torch.bool:
        raise TypeError("`flow_valid` must use torch.bool.")
    batch_size, packed_channels, height, width = target_flow.shape
    if (height, width) != ROBOTWIN_FASTWAM_MOSAIC_SIZE:
        raise ValueError(
            "Head-only Flow supervision requires the canonical mosaic "
            f"{ROBOTWIN_FASTWAM_MOSAIC_SIZE}, got {(height, width)}."
        )
    horizon_count = int(flow_valid.shape[1])
    if (
        flow_valid.shape[0] != batch_size
        or tuple(flow_valid.shape[-2:]) != (height, width)
        or packed_channels != 3 * horizon_count
    ):
        raise ValueError("`target_flow` and `flow_valid` contracts do not align.")
    head_height, head_width = ROBOTWIN_FASTWAM_HEAD_SIZE
    if head_width != width or not 0 < head_height < height:
        raise RuntimeError("FastWAM head/mosaic geometry constants are inconsistent.")

    scoped_valid = flow_valid.clone()
    scoped_valid[..., head_height:, :] = False
    scoped_target = target_flow.reshape(
        batch_size,
        horizon_count,
        3,
        height,
        width,
    ).clone()
    scoped_target[..., head_height:, :] = 0
    return scoped_target.reshape_as(target_flow), scoped_valid


class FastWAMPixelDiT(nn.Module):
    """Strict V2 training and target-free sampling wrapper for MoMap-WAM."""

    CHECKPOINT_FORMAT = "fastwam_pixeldit_momap_v2"
    GENERATION_MODE = "directional_momap_to_action_flow_matching"
    TIMESTEP_COUPLINGS = frozenset({"independent", "shared"})
    TRAINING_STAGES = frozenset({"action_only", "joint", "world_only"})

    _resolve_method_contract = staticmethod(resolve_formal_method_contract)

    _TRAIN_REQUIRED_KEYS = frozenset(
        {
            "source_rgb",
            "source_seg_condition",
            "source_xyz_ref",
            "source_condition_valid",
            "target_flow",
            "flow_valid",
            "future_offsets",
            "future_horizons_seconds",
            "future_offset_basis",
            "coordinate_frame",
            "unit",
            "source_normalization",
            "target_representation",
            "frame_offsets",
            "momap_dt_seconds",
            "momap_lateness_seconds",
            "action",
            "action_is_pad",
            "action_label_kind",
            "action_is_high_level_policy_action",
            "action_normalization",
            "proprio_mode",
            "proprio_normalization",
            "context",
        }
    )
    _TRAIN_OPTIONAL_KEYS = frozenset(
        {
            "context_mask",
            "proprio",
            "flow_component_valid",
            "flow_loss_valid",
        }
    )
    _RAW_OBSERVATION_REQUIRED_KEYS = frozenset(
        {
            "source_rgb",
            "source_seg_condition",
            "source_xyz_ref",
            "source_condition_valid",
            "context",
        }
    )
    _RAW_OBSERVATION_OPTIONAL_KEYS = frozenset(
        {
            "context_mask",
            "proprio",
            "proprio_mode",
            "proprio_normalization",
            "source_depth_m",
            "source_depth_valid",
        }
    )
    _OBSERVATION_REQUIRED_KEYS = frozenset(
        {
            "source_rgb",
            "source_seg_condition",
            "source_xyz_ref",
            "source_condition_valid",
            "context",
        }
    )
    _OBSERVATION_OPTIONAL_KEYS = frozenset(
        {
            "context_mask",
            "proprio",
            "proprio_mode",
            "proprio_normalization",
        }
    )
    _INFERENCE_FORBIDDEN_KEYS = frozenset(
        {
            "metadata",
            "target_flow",
            "flow_valid",
            "momap",
            "momap_geom_valid",
            "momap_future_offsets",
            "momap_future_horizons_seconds",
            "momap_future_offset_basis",
            "momap_frame_offsets",
            "momap_dt_seconds",
            "momap_visibility_evaluated",
            "momap_visible",
            "momap_occluded",
            "momap_out_of_frame",
            "momap_depth_conflict",
            "momap_unsupported",
            "momap_unmapped",
            "momap_sampling_ambiguous",
            "target_indices",
            "momap_lateness_seconds",
            "future_offsets",
            "future_horizons_seconds",
            "future_offset_basis",
            "frame_offsets",
            "action",
            "action_is_pad",
        }
    )

    def __init__(
        self,
        dit: PixelDiTActionMoT,
        *,
        kflow_contract: KFlowContract,
        normalization_contract: GeometryNormalization,
        action_horizon: int,
        action_normalization_contract: FixedFeatureNormalizationContract,
        text_dim: int,
        batch_adapter: Optional[MoMapBatchAdapter] = None,
        proprio_dim: Optional[int] = None,
        proprio_normalization_contract: Optional[
            FixedFeatureNormalizationContract
        ] = None,
        train_world_scheduler: Optional[WanContinuousFlowMatchScheduler] = None,
        infer_world_scheduler: Optional[WanContinuousFlowMatchScheduler] = None,
        train_action_scheduler: Optional[WanContinuousFlowMatchScheduler] = None,
        infer_action_scheduler: Optional[WanContinuousFlowMatchScheduler] = None,
        timestep_coupling: str = "independent",
        loss_lambda_momap: float = 1.0,
        loss_lambda_action: float = 1.0,
        flow_supervision_scope: str = FLOW_SUPERVISION_ALL_MOSAIC,
        flow_statistics_provenance_contract: Optional[
            Mapping[str, Any]
        ] = None,
        method_contract: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        if not isinstance(dit, PixelDiTActionMoT):
            raise TypeError(
                f"`dit` must be PixelDiTActionMoT, got {type(dit).__name__}."
            )
        if not isinstance(kflow_contract, KFlowContract):
            raise TypeError(
                "`kflow_contract` must be KFlowContract, got "
                f"{type(kflow_contract).__name__}."
            )
        if not isinstance(
            normalization_contract,
            (
                GeometryNormalizationContract,
                PerSampleP99GeometryNormalizationContract,
                FixedGlobalSourceP99FlowXYZMeanStdGeometryNormalizationContract,
                SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract,
                HybridHeadWristNormalizationContract,
            ),
        ):
            raise TypeError(
                "`normalization_contract` must be a supported geometry "
                "normalization contract, got "
                f"{type(normalization_contract).__name__}."
            )
        if not kflow_contract.uses_common_reference:
            raise ValueError(
                "FastWAMPixelDiT production wrapper requires the common-reference "
                f"{WORLD_CONDITIONING_V2!r} contract."
            )
        if isinstance(text_dim, bool) or int(text_dim) != text_dim or text_dim <= 0:
            raise ValueError(
                f"`text_dim` must be a positive integer, got {text_dim!r}."
            )

        if not isinstance(
            action_normalization_contract,
            FixedFeatureNormalizationContract,
        ):
            raise TypeError(
                "`action_normalization_contract` must be "
                "FixedFeatureNormalizationContract."
            )
        if proprio_normalization_contract is not None and not isinstance(
            proprio_normalization_contract,
            FixedFeatureNormalizationContract,
        ):
            raise TypeError(
                "`proprio_normalization_contract` must be "
                "FixedFeatureNormalizationContract or None."
            )

        self.dit = dit
        self.method_contract = self._resolve_method_contract(
            FORMAL_METHOD_CONTRACT_V1
            if method_contract is None
            else method_contract
        )
        self.training_stage = "joint"
        self.kflow_contract = kflow_contract
        self.normalization_contract = normalization_contract
        self.action_horizon = self._positive_int(
            action_horizon,
            name="action_horizon",
        )
        if self.action_horizon != 32:
            raise ValueError(
                "Formal MoMap-WAM action_horizon is fixed at 32, got "
                f"{self.action_horizon}."
            )
        self.action_normalization_contract = action_normalization_contract
        self.text_dim = int(text_dim)
        self.proprio_dim = (
            None
            if proprio_dim is None
            else self._positive_int(proprio_dim, name="proprio_dim")
        )
        if self.proprio_dim is None:
            if proprio_normalization_contract is not None:
                raise ValueError(
                    "`proprio_normalization_contract` requires `proprio_dim`."
                )
            self.proprio_normalization_contract = None
        else:
            if proprio_normalization_contract is None:
                raise ValueError(
                    "`proprio_normalization_contract` is required when "
                    "`proprio_dim` is configured."
                )
            self.proprio_normalization_contract = proprio_normalization_contract

        self.batch_adapter = (
            MoMapBatchAdapter(
                kflow_contract,
                action_label_kind=action_normalization_contract.semantic_kind,
                action_is_high_level_policy_action=False,
                action_normalization=action_normalization_contract.source_normalization,
                proprio_mode=(
                    "none"
                    if self.proprio_normalization_contract is None
                    else self.proprio_normalization_contract.semantic_kind
                ),
                proprio_normalization=(
                    "none"
                    if self.proprio_normalization_contract is None
                    else self.proprio_normalization_contract.source_normalization
                ),
            )
            if batch_adapter is None
            else batch_adapter
        )
        if not isinstance(self.batch_adapter, MoMapBatchAdapter):
            raise TypeError(
                "`batch_adapter` must be MoMapBatchAdapter, got "
                f"{type(self.batch_adapter).__name__}."
            )
        self.proprio_encoder: Optional[nn.Linear]
        if self.proprio_dim is None:
            self.proprio_encoder = None
        else:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim)

        self.train_world_scheduler = self._scheduler_or_default(train_world_scheduler)
        self.infer_world_scheduler = self._scheduler_or_default(infer_world_scheduler)
        self.train_action_scheduler = self._scheduler_or_default(train_action_scheduler)
        self.infer_action_scheduler = self._scheduler_or_default(infer_action_scheduler)
        if timestep_coupling not in self.TIMESTEP_COUPLINGS:
            raise ValueError(
                "`timestep_coupling` must be one of "
                f"{sorted(self.TIMESTEP_COUPLINGS)}, got "
                f"{timestep_coupling!r}."
            )
        self.timestep_coupling = timestep_coupling
        if (
            self.timestep_coupling
            != self.method_contract["train_timestep_coupling"]
        ):
            raise ValueError(
                "`timestep_coupling` does not match the frozen method_contract."
            )

        loss_weights = {
            "loss_lambda_momap": float(loss_lambda_momap),
            "loss_lambda_action": float(loss_lambda_action),
        }
        if any(
            not math.isfinite(value) or value < 0.0 for value in loss_weights.values()
        ):
            raise ValueError(
                "Loss weights must be finite and non-negative, got " f"{loss_weights}."
            )
        self.loss_lambda_momap = loss_weights["loss_lambda_momap"]
        self.loss_lambda_action = loss_weights["loss_lambda_action"]
        if flow_supervision_scope not in FLOW_SUPERVISION_SCOPES:
            raise ValueError(
                "`flow_supervision_scope` must be one of "
                f"{sorted(FLOW_SUPERVISION_SCOPES)}, got "
                f"{flow_supervision_scope!r}."
            )
        self.flow_supervision_scope = flow_supervision_scope
        if flow_statistics_provenance_contract is None:
            self.flow_statistics_provenance_contract = None
        else:
            if not isinstance(flow_statistics_provenance_contract, Mapping):
                raise TypeError(
                    "`flow_statistics_provenance_contract` must be a mapping."
                )
            self.flow_statistics_provenance_contract = copy.deepcopy(
                dict(flow_statistics_provenance_contract)
            )
        if (
            self.flow_supervision_scope == FLOW_SUPERVISION_HEAD_TILE_ONLY
            and self.flow_statistics_provenance_contract is None
        ):
            raise ValueError(
                "Head-only formal training requires a bound Flow statistics "
                "provenance contract."
            )

        self._validate_contract_alignment()
        if self.timestep_coupling == "shared":
            self._validate_shared_scheduler_contract()
        self.momap_objective = MoMapFlowMatchingObjective(
            self.train_world_scheduler,
            k=self.kflow_contract.k,
        )

    def set_training_stage(self, training_stage: str) -> None:
        """Select the runtime loss graph without changing checkpoint math."""

        if not isinstance(training_stage, str):
            raise TypeError("`training_stage` must be a string.")
        resolved = training_stage.strip().lower()
        if resolved not in self.TRAINING_STAGES:
            raise ValueError(
                "`training_stage` must be one of "
                f"{sorted(self.TRAINING_STAGES)}, got {training_stage!r}."
            )
        self.training_stage = resolved

    @staticmethod
    def _positive_int(value: Any, *, name: str) -> int:
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(f"`{name}` must be a positive integer, got {value!r}.")
        return int(value)

    @staticmethod
    def _scheduler_or_default(
        scheduler: Optional[WanContinuousFlowMatchScheduler],
    ) -> WanContinuousFlowMatchScheduler:
        if scheduler is None:
            scheduler = WanContinuousFlowMatchScheduler(
                num_train_timesteps=1000,
                shift=5.0,
            )
        if not isinstance(scheduler, WanContinuousFlowMatchScheduler):
            raise TypeError(
                "Flow-matching schedulers must be "
                "WanContinuousFlowMatchScheduler, got "
                f"{type(scheduler).__name__}."
            )
        values = {
            "shift": float(scheduler.shift),
            "eps": float(scheduler.eps),
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError(
                f"Scheduler shift/eps must be finite and positive, got {values}."
            )
        if int(scheduler.num_train_timesteps) <= 0:
            raise ValueError("Scheduler num_train_timesteps must be positive.")
        return scheduler

    @staticmethod
    def _scheduler_contract(
        scheduler: WanContinuousFlowMatchScheduler,
    ) -> dict[str, Any]:
        contract: dict[str, Any] = {
            "class": type(scheduler).__name__,
            "num_train_timesteps": int(scheduler.num_train_timesteps),
            "shift": float(scheduler.shift),
            "eps": float(scheduler.eps),
        }
        policy_provider = getattr(scheduler, "training_policy_contract", None)
        if policy_provider is None:
            return contract
        if not callable(policy_provider):
            raise TypeError(
                "Scheduler training_policy_contract must be callable when present."
            )
        raw_policy = policy_provider()
        if not isinstance(raw_policy, Mapping):
            raise TypeError("Scheduler training policy contract must be a mapping.")
        contract["training_policy"] = copy.deepcopy(dict(raw_policy))
        return contract

    @staticmethod
    def _scheduler_weighting_contract(
        train_world: Mapping[str, Any],
        train_action: Mapping[str, Any],
    ) -> str:
        world_policy = train_world.get("training_policy")
        action_policy = train_action.get("training_policy")
        if world_policy is None and action_policy is None:
            return "per_sample_then_batch_mean"
        if not isinstance(world_policy, Mapping) or not isinstance(
            action_policy,
            Mapping,
        ):
            raise ValueError(
                "World and Action training schedulers must both bind one policy."
            )
        if dict(world_policy) != dict(action_policy):
            raise ValueError(
                "World and Action training scheduler policies must match exactly."
            )
        if world_policy.get("timestep_weighting") != "uniform_one_v1":
            raise ValueError(
                "Unsupported explicit scheduler timestep-weighting contract."
            )
        return "uniform_one_per_sample_then_batch_mean_v1"

    @staticmethod
    def _pixel_clean_timestep(
        world_timestep: torch.Tensor,
        *,
        scheduler: WanContinuousFlowMatchScheduler,
    ) -> torch.Tensor:
        """Map local noise time ``s`` to PixelDiT's pretrained clean time."""

        if not isinstance(world_timestep, torch.Tensor):
            raise TypeError("`world_timestep` must be a torch.Tensor.")
        if not world_timestep.is_floating_point():
            raise TypeError("`world_timestep` must be floating point.")
        if not bool(torch.isfinite(world_timestep).all().item()):
            raise ValueError("`world_timestep` contains NaN or Inf.")
        world_timestep_fp32 = world_timestep.to(dtype=torch.float32)
        noise_fraction = world_timestep_fp32 / float(
            scheduler.num_train_timesteps
        )
        tolerance = 1.0e-6
        if bool(
            (
                (noise_fraction < -tolerance)
                | (noise_fraction > 1.0 + tolerance)
            )
            .any()
            .item()
        ):
            raise ValueError(
                "World scheduler noise fraction must stay in [0,1]."
            )
        noise_fraction = noise_fraction.clamp(0.0, 1.0)
        return 1.0 - noise_fraction

    @staticmethod
    def _action_noise_fraction_timestep(
        action_timestep: torch.Tensor,
        *,
        scheduler: WanContinuousFlowMatchScheduler,
    ) -> torch.Tensor:
        """Map scheduler coordinates to the V8 Action model's [0,1] input."""

        if not isinstance(action_timestep, torch.Tensor):
            raise TypeError("action_timestep must be a torch.Tensor.")
        if not action_timestep.is_floating_point():
            raise TypeError("action_timestep must be floating point.")
        fraction = action_timestep.float() / float(
            scheduler.num_train_timesteps
        )
        tolerance = 1.0e-6
        if not bool(torch.isfinite(fraction).all().item()):
            raise ValueError("Action scheduler timestep contains NaN or Inf.")
        if bool(
            ((fraction < -tolerance) | (fraction > 1.0 + tolerance))
            .any()
            .item()
        ):
            raise ValueError("Action scheduler noise fraction must stay in [0,1].")
        return fraction.clamp(0.0, 1.0)

    def _validate_shared_scheduler_contract(self) -> None:
        train_world = self._scheduler_contract(self.train_world_scheduler)
        train_action = self._scheduler_contract(self.train_action_scheduler)
        infer_world = self._scheduler_contract(self.infer_world_scheduler)
        infer_action = self._scheduler_contract(self.infer_action_scheduler)
        if train_world != train_action or infer_world != infer_action:
            raise ValueError(
                "`timestep_coupling='shared'` requires identical world/action "
                "scheduler contracts for both train and inference: "
                f"train_world={train_world}, train_action={train_action}, "
                f"infer_world={infer_world}, infer_action={infer_action}."
            )

    def _validate_contract_alignment(self) -> None:
        if self.batch_adapter.contract != self.kflow_contract:
            raise ValueError(
                "MoMapBatchAdapter/KFlowContract mismatch: "
                f"adapter={self.batch_adapter.contract.as_dict()}, "
                f"wrapper={self.kflow_contract.as_dict()}."
            )
        expected_semantics = {
            "action_label_kind": self.action_normalization_contract.semantic_kind,
            "action_is_high_level_policy_action": False,
            "action_normalization": (
                self.action_normalization_contract.source_normalization
            ),
            "proprio_mode": (
                "none"
                if self.proprio_normalization_contract is None
                else self.proprio_normalization_contract.semantic_kind
            ),
            "proprio_normalization": (
                "none"
                if self.proprio_normalization_contract is None
                else self.proprio_normalization_contract.source_normalization
            ),
        }
        if self.batch_adapter.semantic_contract() != expected_semantics:
            raise ValueError(
                "MoMapBatchAdapter action/proprio semantic contract mismatch: "
                f"adapter={self.batch_adapter.semantic_contract()}, "
                f"wrapper={expected_semantics}."
            )
        if self.action_normalization_contract.feature_kind != "action":
            raise ValueError(
                "Action normalization contract feature_kind must be 'action'."
            )
        if self.action_normalization_contract.feature_dim != 14:
            raise ValueError("Action normalization contract must be fixed 14-D.")
        if int(getattr(self.action_expert, "action_dim", -1)) != 14:
            raise ValueError("Formal MoMap-WAM ActionDiT action_dim must be 14.")
        if self.proprio_dim is not None:
            assert self.proprio_normalization_contract is not None
            if self.proprio_dim != 14:
                raise ValueError("Formal MoMap-WAM proprio_dim must be 14.")
            if self.proprio_normalization_contract.feature_kind != "proprio":
                raise ValueError(
                    "Proprio normalization contract feature_kind must be 'proprio'."
                )
            if self.proprio_normalization_contract.feature_dim != self.proprio_dim:
                raise ValueError("Proprio normalization feature dimension mismatch.")
        if self.normalization_contract.k != self.kflow_contract.k:
            raise ValueError(
                "Normalization/K-flow K mismatch: "
                f"{self.normalization_contract.k} vs "
                f"{self.kflow_contract.k}."
            )
        geometry_binding = {
            "future_offsets": self.normalization_contract.future_offsets,
            "reference_frame": self.normalization_contract.reference_frame,
            "source_unit": self.normalization_contract.source_unit,
            "channel_order": self.normalization_contract.channel_order,
        }
        expected_geometry_binding = {
            "future_offsets": self.kflow_contract.future_offsets,
            "reference_frame": self.kflow_contract.coordinate_frame,
            "source_unit": self.kflow_contract.unit,
            "channel_order": self.kflow_contract.channel_order,
        }
        if geometry_binding != expected_geometry_binding:
            raise ValueError(
                "Geometry normalization/KFlow binding mismatch: "
                f"{geometry_binding} vs {expected_geometry_binding}."
            )

        world = self.world_expert
        action = self.action_expert
        expected_flow_channels = self.kflow_contract.flow_channels
        world_contract = {
            "clean_channels": int(getattr(world, "clean_channels", -1)),
            "conditioning_mode": getattr(getattr(world, "conditioning_contract", None), "mode", None),
            "k": int(getattr(world, "k", -1)),
            "flow_channels": int(getattr(world, "flow_channels", -1)),
            "out_channels": int(getattr(world, "out_channels", -1)),
            "in_channels": int(getattr(world, "in_channels", -1)),
            "patch_depth": int(getattr(world, "patch_depth", -1)),
            "pixel_depth": int(getattr(world, "pixel_depth", -1)),
        }
        expected_world_contract = {
            "clean_channels": 9,
            "conditioning_mode": WORLD_CONDITIONING_V2,
            "k": self.kflow_contract.k,
            "flow_channels": expected_flow_channels,
            "out_channels": expected_flow_channels,
            "in_channels": 9 + expected_flow_channels,
            "patch_depth": 26,
            "pixel_depth": 4,
        }
        if world_contract != expected_world_contract:
            raise ValueError(
                "PixelDiT/K-flow formal architecture mismatch: "
                f"model={world_contract}, expected={expected_world_contract}."
            )
        topology_version = getattr(self.dit, "topology_version", None)
        expected_topology = self.method_contract.get(
            "topology",
            PixelDiTActionMoT.LEGACY_TOPOLOGY_VERSION,
        )
        if topology_version != expected_topology:
            raise ValueError(
                "PixelDiTActionMoT topology/method_contract mismatch: "
                f"model={topology_version!r}, expected={expected_topology!r}."
            )
        is_aligned_action = isinstance(self.dit, PixelDiTAlignedActionMoT)
        if is_aligned_action != (
            self.method_contract
            in (
                FORMAL_METHOD_CONTRACT_V8,
                FORMAL_METHOD_CONTRACT_V9,
                FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW,
                FORMAL_METHOD_CONTRACT_K4_HYBRID_AC,
                FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2,
                FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2,
                FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2,
            )
        ):
            raise ValueError(
                "PixelDiT-aligned Action wrapper and matching method contract must "
                "be selected together."
            )
        expected_action_depth = (
            30
            if (
                is_aligned_action
                or topology_version in PixelDiTActionMoT.ACTION30_TOPOLOGY_VERSIONS
            )
            else 26
        )
        if int(getattr(self.dit, "world_depth", -1)) != 26:
            raise ValueError("Formal wrapper requires exactly 26 world patch blocks.")
        if int(getattr(self.dit, "action_depth", -1)) != expected_action_depth:
            raise ValueError(
                "Formal wrapper ActionDiT depth mismatch: expected "
                f"{expected_action_depth}."
            )
        if int(getattr(self.dit, "pixel_depth", -1)) != 4:
            raise ValueError("Formal wrapper requires exactly four pixel modules.")
        if len(action.blocks) != expected_action_depth:
            raise ValueError(
                "ActionDiT must contain exactly "
                f"{expected_action_depth} blocks."
            )
        if int(getattr(action, "text_dim", -1)) != self.text_dim:
            raise ValueError(
                "Wrapper text_dim must match ActionDiT.text_dim, got "
                f"{self.text_dim} vs {getattr(action, 'text_dim', None)}."
            )

        if is_aligned_action:
            if not isinstance(action, PixelDiTAlignedActionDiT):
                raise TypeError("Aligned Action wrapper requires a PixelDiT-aligned Action DiT.")
            if self.method_contract in (
                FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW,
                FORMAL_METHOD_CONTRACT_K4_HYBRID_AC,
                FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2,
                FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2,
                FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2,
            ):
                expected_native_kv_fusion = HYBRID_CW_NATIVE_KV_FUSION_VERSION
            elif self.method_contract == FORMAL_METHOD_CONTRACT_V9:
                expected_native_kv_fusion = (
                    COSINE_ANNEAL_NATIVE_KV_FUSION_VERSION
                )
            else:
                expected_native_kv_fusion = LEGACY_NATIVE_KV_FUSION_VERSION
            common_action = {
                "action_horizon": int(action.action_horizon),
                "action_dim": int(action.action_dim),
                "hidden_size": int(action.hidden_size),
                "num_heads": int(action.num_heads),
                "head_dim": int(action.attn_head_dim),
                "num_layers": len(action.blocks),
                "text_dim": int(action.text_dim),
                "native_kv_fusion": action.native_kv_fusion_version,
            }
            if isinstance(action, PixelDiTTemporalActionDiT):
                if self.method_contract != FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2:
                    raise ValueError("Temporal32 Action requires its dedicated Hybrid method contract.")
                actual_action = {
                    **common_action,
                    "encoder_shape": list(action.action_encoder.weight.shape),
                    "decoder_shape": list(action.action_decoder.weight.shape),
                    "rope_shape": list(action.temporal_rope.shape),
                    "token_semantics": action.TOKEN_SEMANTICS_VERSION,
                    "position_encoding": action.POSITION_ENCODING_VERSION,
                }
                expected_action = {
                    "action_horizon": 32,
                    "action_dim": 14,
                    "hidden_size": 1152,
                    "num_heads": 16,
                    "head_dim": 72,
                    "num_layers": 30,
                    "text_dim": self.text_dim,
                    "encoder_shape": [1152, 14],
                    "decoder_shape": [14, 1152],
                    "rope_shape": [32, 36],
                    "token_semantics": "time32_per_step_action14_tokens_v1",
                    "position_encoding": "temporal_1d_rope_index_0_31_v1",
                    "native_kv_fusion": HYBRID_CW_NATIVE_KV_FUSION_VERSION,
                }
                if "temporal_rope" in action.state_dict():
                    raise ValueError("Temporal Action RoPE must be non-persistent.")
            else:
                if self.method_contract == FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2:
                    raise ValueError("Temporal32 Hybrid method contract requires temporal Action tokens.")
                actual_action = {
                    **common_action,
                    "encoder_shape": list(action.action_encoder.weight.shape),
                    "decoder_shape": list(action.action_decoder.weight.shape),
                    "position_shape": list(action.joint_pos_embed.shape),
                    "token_semantics": action.TOKEN_SEMANTICS_VERSION,
                    "position_encoding": action.POSITION_ENCODING_VERSION,
                }
                expected_action = {
                    "action_horizon": 32,
                    "action_dim": 14,
                    "hidden_size": 1152,
                    "num_heads": 16,
                    "head_dim": 72,
                    "num_layers": 30,
                    "text_dim": self.text_dim,
                    "encoder_shape": [1152, 32],
                    "decoder_shape": [32, 1152],
                    "position_shape": [1, 14, 1152],
                    "token_semantics": "joint14_full_horizon32_spatial_tokens_v1",
                    "position_encoding": (
                        "fixed_joint_index_1d_sincos_no_action_rope_v1"
                    ),
                    "native_kv_fusion": expected_native_kv_fusion,
                }
                if "joint_pos_embed" in action.state_dict():
                    raise ValueError("Fixed joint position encoding must be non-persistent.")
            if actual_action != expected_action:
                raise ValueError(
                    "Aligned Action architecture contract mismatch: "
                    f"{actual_action} vs {expected_action}."
                )
            for name in (
                "world_to_action_adapters",
                "world_native_kv_adapters",
                "pit_native_kv_adapters",
            ):
                modules = getattr(self.dit, name, None)
                if not isinstance(modules, nn.ModuleList) or len(modules) != 0:
                    raise ValueError(f"Aligned Action requires empty {name}.")
            if tuple(self.dit.layer_map) != tuple(range(30)):
                raise ValueError(
                    "Aligned Action bridge layer map must be identity 0..29."
                )
            if len(tuple(self.dit.bridge_source_schedule)) != 30:
                raise ValueError(
                    "Aligned Action requires exactly 30 native-K/V bridge sources."
                )
            return

        adapters = getattr(self.dit, "world_to_action_adapters", None)
        expected_global_adapter_count = (
            0
            if self.method_contract in (
                FORMAL_METHOD_CONTRACT_V6,
                FORMAL_METHOD_CONTRACT_V7,
            )
            else 26
            if self.method_contract == FORMAL_METHOD_CONTRACT_V5
            else expected_action_depth
        )
        if (
            not isinstance(adapters, nn.ModuleList)
            or len(adapters) != expected_global_adapter_count
        ):
            raise ValueError(
                "Formal MoMap-WAM requires exactly "
                f"{expected_global_adapter_count} WorldToActionAdapter instances."
            )
        if len({id(adapter) for adapter in adapters}) != expected_global_adapter_count:
            raise ValueError("WorldToActionAdapter instances must be independent.")
        for index, adapter in enumerate(adapters):
            if not isinstance(adapter, WorldToActionAdapter):
                raise TypeError(f"Bridge {index} is not WorldToActionAdapter.")
            expected_adapter = {
                "world_hidden_size": int(world.hidden_size),
                "action_hidden_size": int(action.hidden_dim),
                "mot_dim": int(self.dit.mot_dim),
                "mot_num_heads": int(self.dit.mot_num_heads),
            }
            actual_adapter = {
                key: int(getattr(adapter, key)) for key in expected_adapter
            }
            if actual_adapter != expected_adapter:
                raise ValueError(
                    f"Bridge {index} architecture mismatch: "
                    f"{actual_adapter} vs {expected_adapter}."
                )

        native_adapters = getattr(self.dit, "world_native_kv_adapters", None)
        expected_native_adapter_count = (
            30
            if self.method_contract == FORMAL_METHOD_CONTRACT_V6
            else 26
            if self.method_contract == FORMAL_METHOD_CONTRACT_V7
            else 0
        )
        if (
            not isinstance(native_adapters, nn.ModuleList)
            or len(native_adapters) != expected_native_adapter_count
        ):
            raise ValueError(
                "Formal MoMap-WAM World native-K/V adapter count mismatch: "
                f"{getattr(native_adapters, '__len__', lambda: -1)()} "
                f"vs {expected_native_adapter_count}."
            )
        if (
            len({id(adapter) for adapter in native_adapters})
            != expected_native_adapter_count
        ):
            raise ValueError("WorldNativeKVAdapter instances must be independent.")
        world_blocks = (
            list(world.patch_blocks)
            if self.method_contract == FORMAL_METHOD_CONTRACT_V7
            else [*world.patch_blocks, *world.pixel_blocks]
        )
        for index, adapter in enumerate(native_adapters):
            if not isinstance(adapter, WorldNativeKVAdapter):
                raise TypeError(
                    f"World native-K/V adapter {index} has the wrong type."
                )
            expected_native_adapter = {
                "world_attn_dim": int(world_blocks[index].attn.dim),
                "action_attn_dim": int(
                    action.blocks[index].self_attn.attn_hidden_dim
                ),
            }
            actual_native_adapter = {
                key: int(getattr(adapter, key))
                for key in expected_native_adapter
            }
            if actual_native_adapter != expected_native_adapter:
                raise ValueError(
                    f"World native-K/V adapter {index} architecture mismatch: "
                    f"{actual_native_adapter} vs {expected_native_adapter}."
                )


        tail_native_adapters = getattr(self.dit, "pit_native_kv_adapters", None)
        expected_tail_native_adapter_count = (
            4 if self.method_contract == FORMAL_METHOD_CONTRACT_V5 else 0
        )
        if (
            not isinstance(tail_native_adapters, nn.ModuleList)
            or len(tail_native_adapters) != expected_tail_native_adapter_count
        ):
            raise ValueError(
                "Formal MoMap-WAM PiT tail native-K/V adapter count mismatch."
            )
        if (
            len({id(adapter) for adapter in tail_native_adapters})
            != expected_tail_native_adapter_count
        ):
            raise ValueError("PiT tail native-K/V adapters must be independent.")
        for pixel_index, adapter in enumerate(tail_native_adapters):
            if not isinstance(adapter, WorldNativeKVAdapter):
                raise TypeError(
                    f"PiT tail native-K/V adapter {pixel_index} has the wrong type."
                )
            expected_tail_adapter = {
                "world_attn_dim": int(world.pixel_blocks[pixel_index].attn.dim),
                "action_attn_dim": int(
                    action.blocks[26 + pixel_index].self_attn.attn_hidden_dim
                ),
            }
            actual_tail_adapter = {
                key: int(getattr(adapter, key)) for key in expected_tail_adapter
            }
            if actual_tail_adapter != expected_tail_adapter:
                raise ValueError(
                    f"PiT tail native-K/V adapter {pixel_index} mismatch: "
                    f"{actual_tail_adapter} vs {expected_tail_adapter}."
                )

        if tuple(self.dit.layer_map) != tuple(range(expected_action_depth)):
            raise ValueError(
                "Formal bridge layer map must be the ActionDiT identity map."
            )
        if self.method_contract == FORMAL_METHOD_CONTRACT_V6:
            expected_bridge_schedule = tuple(
                ("global_native_kv_mixed_with_action_self_kv", index)
                for index in range(26)
            ) + tuple(
                ("pit_native_kv_mixed_with_action_self_kv", index)
                for index in range(4)
            )
        elif self.method_contract == FORMAL_METHOD_CONTRACT_V7:
            expected_bridge_schedule = tuple(
                ("global_native_kv_mixed_with_action_self_kv", index)
                for index in range(26)
            )
        else:
            expected_bridge_schedule = tuple(
                ("global", index) for index in range(26)
            )
            if self.method_contract == FORMAL_METHOD_CONTRACT_V2:
                expected_bridge_schedule += tuple(
                    ("final_global_concat_pit_compact", index)
                    for index in range(4)
                )
            elif self.method_contract == FORMAL_METHOD_CONTRACT_V3:
                expected_bridge_schedule += tuple(
                    ("pit_compact", index) for index in range(4)
                )
            elif self.method_contract == FORMAL_METHOD_CONTRACT_V5:
                expected_bridge_schedule += tuple(
                    ("pit_native_kv_mixed_with_action_self_kv", index)
                    for index in range(4)
                )
            elif self.method_contract not in (
                FORMAL_METHOD_CONTRACT_V1,
                FORMAL_METHOD_CONTRACT_V4,
            ):
                raise ValueError(
                    "Unsupported formal bridge source schedule contract."
                )
        if tuple(self.dit.bridge_source_schedule) != expected_bridge_schedule:
            raise ValueError(
                "Formal World-to-Action bridge source schedule mismatch."
            )

    @property
    def world_expert(self) -> nn.Module:
        return self.dit.world_expert

    @property
    def action_expert(self) -> nn.Module:
        return self.dit.action_expert

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration as exc:  # pragma: no cover - construction invariant
            raise RuntimeError("MoMap-WAM contains no parameters.") from exc

    @property
    def torch_dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration as exc:  # pragma: no cover - construction invariant
            raise RuntimeError("MoMap-WAM contains no parameters.") from exc

    @staticmethod
    def _required_tensor(
        sample: Mapping[str, Any],
        key: str,
    ) -> torch.Tensor:
        value = sample.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"sample[{key!r}] must be a torch.Tensor.")
        return value

    @staticmethod
    def _require_float_finite(value: torch.Tensor, *, name: str) -> None:
        if not value.is_floating_point():
            raise TypeError(f"`{name}` must be floating point, got {value.dtype}.")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"`{name}` contains NaN or Inf.")

    @staticmethod
    def _strict_keyset(
        sample: Mapping[str, Any],
        *,
        required: frozenset[str],
        optional: frozenset[str],
        name: str,
    ) -> None:
        keys = set(sample)
        missing = sorted(required - keys)
        unexpected = sorted(keys - required - optional)
        if missing or unexpected:
            raise ValueError(
                f"{name} key contract mismatch: missing={missing}, "
                f"unexpected={unexpected}."
            )

    @staticmethod
    def _sequence_tuple(value: Any, *, name: str) -> tuple[Any, ...]:
        if isinstance(value, torch.Tensor):
            if value.ndim != 1:
                raise ValueError(f"`{name}` must be a one-dimensional sequence.")
            return tuple(value.detach().cpu().tolist())
        if isinstance(value, (tuple, list)):
            return tuple(value)
        raise TypeError(f"`{name}` must be a tuple, list, or 1-D tensor.")

    def _validate_adapted_feature_metadata(
        self,
        sample: Mapping[str, Any],
    ) -> None:
        expected = self.batch_adapter.semantic_contract()
        for key, expected_value in expected.items():
            value = sample[key]
            if key == "action_is_high_level_policy_action":
                if not isinstance(value, bool):
                    raise TypeError(
                        "Adapted action_is_high_level_policy_action must be bool."
                    )
                if value is not expected_value:
                    raise ValueError(
                        "Adapted action_is_high_level_policy_action mismatch."
                    )
            else:
                if not isinstance(value, str):
                    raise TypeError(f"Adapted `{key}` must be a string.")
                if value != expected_value:
                    raise ValueError(
                        f"Adapted `{key}` mismatch: expected {expected_value!r}, "
                        f"got {value!r}."
                    )

    def _validate_adapted_metadata(
        self,
        sample: Mapping[str, Any],
        *,
        batch_size: int,
    ) -> torch.Tensor:
        offsets = tuple(
            int(value)
            for value in self._sequence_tuple(
                sample["future_offsets"],
                name="future_offsets",
            )
        )
        if offsets != self.kflow_contract.future_offsets:
            raise ValueError(
                "Adapted future_offsets do not match KFlowContract: "
                f"{offsets} vs {self.kflow_contract.future_offsets}."
            )
        horizons = tuple(
            float(value)
            for value in self._sequence_tuple(
                sample["future_horizons_seconds"],
                name="future_horizons_seconds",
            )
        )
        expected_horizons = self.kflow_contract.future_horizons_seconds
        if len(horizons) != len(expected_horizons) or any(
            not math.isclose(
                actual,
                expected,
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            )
            for actual, expected in zip(horizons, expected_horizons)
        ):
            raise ValueError(
                "Adapted future_horizons_seconds do not match KFlowContract."
            )
        exact_metadata = {
            "future_offset_basis": self.kflow_contract.future_offset_basis,
            "coordinate_frame": self.kflow_contract.coordinate_frame,
            "unit": self.kflow_contract.unit,
            "source_normalization": self.kflow_contract.source_normalization,
            "target_representation": self.kflow_contract.target_representation,
        }
        for key, expected in exact_metadata.items():
            if sample[key] != expected:
                raise ValueError(
                    f"Adapted `{key}` mismatch: expected {expected!r}, "
                    f"got {sample[key]!r}."
                )

        frame_offsets = self._required_tensor(sample, "frame_offsets")
        if frame_offsets.shape != (batch_size, self.kflow_contract.k):
            raise ValueError(
                "`frame_offsets` must be [B,K], got " f"{tuple(frame_offsets.shape)}."
            )
        if frame_offsets.dtype == torch.bool or frame_offsets.is_floating_point():
            raise TypeError("`frame_offsets` must use an integer dtype.")
        frame_offsets_long = frame_offsets.to(dtype=torch.long)
        if bool((frame_offsets_long <= 0).any().item()) or bool(
            (frame_offsets_long[:, 1:] <= frame_offsets_long[:, :-1]).any().item()
        ):
            raise ValueError("`frame_offsets` must be positive and increasing.")

        dt_seconds = self._required_tensor(sample, "momap_dt_seconds")
        if dt_seconds.shape != (batch_size, self.kflow_contract.k):
            raise ValueError(
                "`momap_dt_seconds` must be [B,K], got " f"{tuple(dt_seconds.shape)}."
            )
        self._require_float_finite(dt_seconds, name="momap_dt_seconds")
        if bool((dt_seconds <= 0).any().item()):
            raise ValueError("`momap_dt_seconds` must be strictly positive.")
        expected = torch.tensor(
            expected_horizons,
            device=dt_seconds.device,
            dtype=dt_seconds.dtype,
        ).unsqueeze(0)
        if bool((dt_seconds + 1.0e-7 < expected).any().item()):
            raise ValueError(
                "`momap_dt_seconds` precedes a requested physical horizon."
            )
        lateness = self._required_tensor(
            sample,
            "momap_lateness_seconds",
        )
        if lateness.shape != (batch_size, self.kflow_contract.k):
            raise ValueError(
                "`momap_lateness_seconds` must be [B,K], got "
                f"{tuple(lateness.shape)}."
            )
        self._require_float_finite(
            lateness,
            name="momap_lateness_seconds",
        )
        expected_lateness = dt_seconds - expected
        if bool((lateness < -1.0e-7).any().item()) or not torch.allclose(
            lateness,
            expected_lateness,
            rtol=1.0e-5,
            atol=1.0e-7,
        ):
            raise ValueError(
                "`momap_lateness_seconds` must equal "
                "momap_dt_seconds - configured horizon."
            )
        return lateness.to(device=self.device, dtype=torch.float32)

    def _append_proprio(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None:
            if proprio is not None:
                raise ValueError(
                    "Input provides proprio but wrapper has `proprio_dim=None`."
                )
            return context, context_mask
        if proprio is None:
            raise ValueError(
                "Wrapper has a proprio encoder but input has no `proprio`."
            )
        if proprio.ndim == 3:
            if proprio.shape[1] < 1:
                raise ValueError("`proprio` sequence must not be empty.")
            proprio_now = proprio[:, 0]
        elif proprio.ndim == 2:
            proprio_now = proprio
        else:
            raise ValueError(
                f"`proprio` must be [B,D] or [B,T,D], got " f"{tuple(proprio.shape)}."
            )
        if proprio_now.shape != (context.shape[0], self.proprio_dim):
            raise ValueError(
                "Proprio shape mismatch: expected "
                f"{(context.shape[0], self.proprio_dim)}, got "
                f"{tuple(proprio_now.shape)}."
            )
        self._require_float_finite(proprio_now, name="proprio")
        proprio_now = proprio_now.to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        token = self.proprio_encoder(proprio_now).unsqueeze(1)
        context = torch.cat([context, token], dim=1)
        context_mask = torch.cat(
            [
                context_mask,
                torch.ones(
                    (context.shape[0], 1),
                    device=context_mask.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        return context, context_mask

    def _prepare_context(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
        *,
        batch_size: int,
        include_proprio: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if context.ndim != 3 or context.shape[0] != batch_size:
            raise ValueError(
                f"`context` must be [B,L,D] with B={batch_size}, got "
                f"{tuple(context.shape)}."
            )
        if context.shape[2] != self.text_dim:
            raise ValueError(
                f"`context` last dim must be {self.text_dim}, got "
                f"{context.shape[2]}."
            )
        self._require_float_finite(context, name="context")
        context = context.to(device=self.device, dtype=self.torch_dtype)
        if context_mask is None:
            prepared_mask = torch.ones(
                context.shape[:2],
                device=self.device,
                dtype=torch.bool,
            )
        else:
            if context_mask.shape != context.shape[:2]:
                raise ValueError(
                    "`context_mask` must be [B,L], got " f"{tuple(context_mask.shape)}."
                )
            if context_mask.dtype != torch.bool:
                raise TypeError("`context_mask` must use torch.bool.")
            prepared_mask = context_mask.to(device=self.device)
        world_context = context
        world_context_mask = prepared_mask
        if not bool(world_context_mask.any(dim=1).all().item()):
            raise ValueError(
                "Every sample must contain at least one valid language token "
                "before the Action-only proprio token is appended."
            )
        if not include_proprio:
            return (
                world_context,
                world_context_mask,
                world_context,
                world_context_mask,
            )
        action_context, action_context_mask = self._append_proprio(
            context,
            prepared_mask,
            proprio,
        )
        if not bool(action_context_mask.any(dim=1).all().item()):
            raise ValueError(
                "Every sample must contain at least one valid context token."
            )
        return (
            action_context,
            action_context_mask,
            world_context,
            world_context_mask,
        )

    def _prepare_source_support_mask(
        self,
        value: Optional[torch.Tensor],
        *,
        batch_size: int,
        height: int,
        width: int,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            raise TypeError("`source_support_mask` must be a torch.Tensor.")
        if value.dtype != torch.bool:
            raise TypeError("`source_support_mask` must use torch.bool.")
        if value.ndim == 3:
            value = value.unsqueeze(1)
        if value.shape != (batch_size, 1, height, width):
            raise ValueError(
                "`source_support_mask` must be [B,1,H,W], got " f"{tuple(value.shape)}."
            )
        if not bool(value.flatten(1).any(dim=1).all().item()):
            raise ValueError(
                "Every sample must expose at least one source-supported pixel."
            )
        return value.to(device=self.device)

    def _derive_source_geometry_scale(
        self,
        xyz_meters: torch.Tensor,
        valid: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if isinstance(
            self.normalization_contract,
            PerSampleP99GeometryNormalizationContract,
        ):
            return self.normalization_contract.derive_scale(
                xyz_meters,
                valid=valid,
            )
        return None

    def _normalize_source_xyz(
        self,
        xyz_meters: torch.Tensor,
        *,
        valid: torch.Tensor,
        sample_scale_meters: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(
            self.normalization_contract,
            PerSampleP99GeometryNormalizationContract,
        ):
            if sample_scale_meters is None:
                raise RuntimeError("Per-sample XYZ normalization lost its source scale.")
            return self.normalization_contract.normalize_xyz(
                xyz_meters,
                valid=valid,
                sample_scale_meters=sample_scale_meters,
            )
        if sample_scale_meters is not None:
            raise RuntimeError("Fixed geometry normalization received a sample scale.")
        return self.normalization_contract.normalize_xyz(
            xyz_meters,
            valid=valid,
        )

    def _normalize_geometry_offset(
        self,
        offset_meters: torch.Tensor,
        *,
        sample_scale_meters: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(
            self.normalization_contract,
            PerSampleP99GeometryNormalizationContract,
        ):
            if sample_scale_meters is None:
                raise RuntimeError("Per-sample KFlow normalization lost its source scale.")
            return self.normalization_contract.normalize_offset(
                offset_meters,
                sample_scale_meters=sample_scale_meters,
            )
        if sample_scale_meters is not None:
            raise RuntimeError("Fixed KFlow normalization received a sample scale.")
        return self.normalization_contract.normalize_offset(offset_meters)

    def _normalize_target_flow_for_training(
        self,
        target_flow_meters: torch.Tensor,
        *,
        sample_scale_meters: Optional[torch.Tensor],
        inputs: Mapping[str, Any],
    ) -> torch.Tensor:
        """Normalize clean Flow and remove unsupported components before noising.

        A pooled mean/std contract maps raw physical zero to ``-mean/std``.
        ``flow_component_valid`` is not fed to the model; it only ensures that
        clean unsupported components remain zero before diffusion noise is
        added.  Legacy scale-only contracts are unchanged because their
        unsupported raw values are already zero.
        """

        normalized = self._normalize_geometry_offset(
            target_flow_meters,
            sample_scale_meters=sample_scale_meters,
        )
        normalized, _ = apply_flow_supervision_scope(
            normalized,
            inputs["flow_valid"],
            scope=self.flow_supervision_scope,
        )
        component_valid = inputs.get("flow_component_valid")
        if component_valid is None:
            return normalized
        if not isinstance(component_valid, torch.Tensor):
            raise TypeError("`flow_component_valid` must be a torch.Tensor.")
        batch_size, packed_channels, height, width = normalized.shape
        expected = (batch_size, packed_channels // 3, 3, height, width)
        if (
            packed_channels % 3 != 0
            or tuple(component_valid.shape) != expected
            or component_valid.dtype != torch.bool
        ):
            raise ValueError(
                "`flow_component_valid` must be bool "
                f"[B,K,3,H,W]={expected}, got "
                f"{tuple(component_valid.shape)} {component_valid.dtype}."
            )
        packed_valid = component_valid.to(device=normalized.device).reshape_as(
            normalized
        )
        return torch.where(packed_valid, normalized, torch.zeros_like(normalized))

    def _denormalize_geometry_offset(
        self,
        offset_normalized: torch.Tensor,
        *,
        sample_scale_meters: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(
            self.normalization_contract,
            PerSampleP99GeometryNormalizationContract,
        ):
            if sample_scale_meters is None:
                raise RuntimeError(
                    "Per-sample KFlow denormalization lost its source scale."
                )
            return self.normalization_contract.denormalize_offset(
                offset_normalized,
                sample_scale_meters=sample_scale_meters,
            )
        if sample_scale_meters is not None:
            raise RuntimeError("Fixed KFlow denormalization received a sample scale.")
        return self.normalization_contract.denormalize_offset(offset_normalized)

    def _prepare_world_condition(
        self,
        sample: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        rgb = self._required_tensor(sample, "source_rgb")
        segmentation = self._required_tensor(sample, "source_seg_condition")
        xyz = self._required_tensor(sample, "source_xyz_ref")
        valid = self._required_tensor(sample, "source_condition_valid")
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(
                f"`source_rgb` must be [B,3,H,W], got {tuple(rgb.shape)}."
            )
        if segmentation.shape != rgb.shape or xyz.shape != rgb.shape:
            raise ValueError(
                "source segmentation and canonical XYZ must match source_rgb."
            )
        batch_size, _, height, width = rgb.shape
        if valid.shape != (batch_size, 1, height, width):
            raise ValueError(
                "`source_condition_valid` must be [B,1,H,W], got "
                f"{tuple(valid.shape)}."
            )
        if valid.dtype != torch.bool:
            raise TypeError("`source_condition_valid` must use torch.bool.")
        for name, value in (
            ("source_rgb", rgb),
            ("source_seg_condition", segmentation),
            ("source_xyz_ref", xyz),
        ):
            self._require_float_finite(value, name=name)
        rgb_fp32 = rgb.to(device=self.device, dtype=torch.float32)
        segmentation_fp32 = segmentation.to(
            device=self.device,
            dtype=torch.float32,
        )
        for name, value in (
            ("source_rgb", rgb_fp32),
            ("source_seg_condition", segmentation_fp32),
        ):
            if (
                float(value.amin().item()) < -1.0001
                or float(value.amax().item()) > 1.0001
            ):
                raise ValueError(f"`{name}` must use [-1,1].")
        valid = self._prepare_source_support_mask(
            valid,
            batch_size=batch_size,
            height=height,
            width=width,
        )
        assert valid is not None
        rgb = rgb_fp32.to(dtype=self.torch_dtype)
        xyz_fp32 = xyz.to(device=self.device, dtype=torch.float32)
        sample_scale_meters = self._derive_source_geometry_scale(
            xyz_fp32,
            valid,
        )
        segmentation_fp32 = torch.where(
            valid,
            segmentation_fp32,
            torch.zeros_like(segmentation_fp32),
        )
        segmentation = segmentation_fp32.to(dtype=self.torch_dtype)
        xyz_normalized_fp32 = self._normalize_source_xyz(
            xyz_fp32,
            valid=valid,
            sample_scale_meters=sample_scale_meters,
        )
        xyz_normalized = xyz_normalized_fp32.to(dtype=self.torch_dtype)
        condition = torch.cat([rgb, segmentation, xyz_normalized], dim=1)
        if condition.shape[1] != 9 or not bool(torch.isfinite(condition).all().item()):
            raise ValueError("World condition must be finite RGB3+Seg3+XYZ3.")
        return condition, valid, sample_scale_meters

    def _prepare_training_sample(
        self,
        sample: Mapping[str, Any],
        *,
        include_action: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(sample, Mapping):
            raise TypeError("Training sample must be a mapping.")
        if "source_support_mask" in sample:
            raise ValueError(
                "Formal wrapper rejects externally supplied source_support_mask "
                "until a source-observation-only producer and provenance contract "
                "are implemented."
            )
        if "metadata" in sample:
            adapted = dict(self.batch_adapter(sample))
        else:
            adapted = dict(sample)
        self._strict_keyset(
            adapted,
            required=self._TRAIN_REQUIRED_KEYS,
            optional=self._TRAIN_OPTIONAL_KEYS,
            name="Adapted training sample",
        )

        (
            world_condition,
            source_support_mask,
            source_geometry_scale_meters,
        ) = self._prepare_world_condition(adapted)
        target_flow = self._required_tensor(adapted, "target_flow")
        context = self._required_tensor(adapted, "context")
        batch_size, _, height, width = world_condition.shape
        expected_flow_shape = (
            batch_size,
            self.kflow_contract.flow_channels,
            height,
            width,
        )
        if target_flow.shape != expected_flow_shape:
            raise ValueError(
                f"`target_flow` must be packed [B,3K,H,W]="
                f"{expected_flow_shape}, got {tuple(target_flow.shape)}."
            )
        self._require_float_finite(target_flow, name="target_flow")
        action: Optional[torch.Tensor] = None
        if include_action:
            action = self._required_tensor(adapted, "action")
            expected_action_shape = (batch_size, self.action_horizon, self.action_expert.action_dim)
            if action.ndim != 3 or tuple(action.shape) != expected_action_shape:
                raise ValueError(
                    f"`action` must match {expected_action_shape}, got "
                    f"{tuple(action.shape)}."
                )
            self._require_float_finite(action, name="action_raw")
        self._validate_adapted_feature_metadata(adapted)
        momap_lateness_seconds = self._validate_adapted_metadata(
            adapted,
            batch_size=batch_size,
        )

        flow_valid = self._required_tensor(adapted, "flow_valid")
        expected_valid_shape = (
            batch_size,
            self.kflow_contract.k,
            1,
            height,
            width,
        )
        if flow_valid.shape != expected_valid_shape:
            raise ValueError(
                f"`flow_valid` must be [B,K,1,H,W]={expected_valid_shape}, "
                f"got {tuple(flow_valid.shape)}."
            )
        if flow_valid.dtype != torch.bool:
            raise TypeError("`flow_valid` must use torch.bool.")
        target_flow, flow_valid = apply_flow_supervision_scope(
            target_flow,
            flow_valid,
            scope=self.flow_supervision_scope,
        )
        flow_valid_for_target = flow_valid.to(device=target_flow.device)
        target_flow_by_horizon = target_flow.reshape(
            batch_size,
            self.kflow_contract.k,
            3,
            height,
            width,
        )
        invalid_nonzero = (~flow_valid_for_target).expand(
            -1, -1, 3, -1, -1
        ) & target_flow_by_horizon.ne(0)
        if bool(invalid_nonzero.any().item()):
            invalid_count = int(invalid_nonzero.sum().item())
            raise ValueError(
                "Formal target_flow must be exactly zero wherever "
                f"flow_valid is false; found {invalid_count} nonzero values."
            )
        flow_valid = flow_valid.to(device=self.device)
        component_masks: dict[str, torch.Tensor] = {}
        expected_component_shape = (
            batch_size,
            self.kflow_contract.k,
            3,
            height,
            width,
        )
        for key in ("flow_component_valid", "flow_loss_valid"):
            value = adapted.get(key)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"`{key}` must be a torch.Tensor.")
            if tuple(value.shape) != expected_component_shape:
                raise ValueError(
                    f"`{key}` must be [B,K,3,H,W]={expected_component_shape}, "
                    f"got {tuple(value.shape)}."
                )
            if value.dtype != torch.bool:
                raise TypeError(f"`{key}` must use torch.bool.")
            component_masks[key] = value.to(device=self.device)
        if "flow_loss_valid" in component_masks and "flow_component_valid" in component_masks:
            if bool(
                (
                    component_masks["flow_loss_valid"]
                    & ~component_masks["flow_component_valid"]
                ).any().item()
            ):
                raise ValueError(
                    "`flow_loss_valid` must be a subset of `flow_component_valid`."
                )

        raw_context_mask = adapted.get("context_mask")
        if raw_context_mask is not None and not isinstance(
            raw_context_mask,
            torch.Tensor,
        ):
            raise TypeError("`context_mask` must be a torch.Tensor.")
        raw_proprio = adapted.get("proprio") if include_action else None
        proprio_normalized: Optional[torch.Tensor] = None
        if not include_action:
            pass
        elif self.proprio_dim is None:
            if raw_proprio is not None:
                raise ValueError(
                    "Input provides proprio but wrapper has `proprio_dim=None`."
                )
        else:
            if not isinstance(raw_proprio, torch.Tensor):
                raise TypeError("`proprio` must be a torch.Tensor.")
            expected_proprio_shape = (batch_size, 1, self.proprio_dim)
            if tuple(raw_proprio.shape) != expected_proprio_shape:
                raise ValueError(
                    "Training proprio must be the current raw observation "
                    f"{expected_proprio_shape}, got {tuple(raw_proprio.shape)}."
                )
            self._require_float_finite(raw_proprio, name="proprio_raw")
            assert self.proprio_normalization_contract is not None
            proprio_fp32 = raw_proprio.to(
                device=self.device,
                dtype=torch.float32,
            )
            proprio_normalized = (
                self.proprio_normalization_contract.normalize(proprio_fp32).to(
                    dtype=self.torch_dtype
                )
            )
        (
            context,
            context_mask,
            world_context,
            world_context_mask,
        ) = self._prepare_context(
            context,
            raw_context_mask,
            proprio_normalized,
            batch_size=batch_size,
            include_proprio=include_action,
        )

        target_flow = target_flow.to(
            device=self.device,
            dtype=torch.float32,
        )
        for name, value in (
            ("world_condition", world_condition),
            ("target_flow", target_flow),
            ("context", context),
        ):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"`{name}` became non-finite after dtype conversion.")
        prepared = {
            "world_condition": world_condition,
            "source_geometry_scale_meters": source_geometry_scale_meters,
            "target_flow_meters": target_flow,
            "flow_valid": flow_valid,
            "momap_lateness_seconds": momap_lateness_seconds,
            "context": context,
            "context_mask": context_mask,
            "world_context": world_context,
            "world_context_mask": world_context_mask,
            "source_support_mask": source_support_mask,
            **component_masks,
        }
        if not include_action:
            return prepared

        assert action is not None
        action_is_pad = self._required_tensor(adapted, "action_is_pad")
        if action_is_pad.shape != action.shape[:2]:
            raise ValueError(
                "`action_is_pad` must match [B,32], got "
                f"{tuple(action_is_pad.shape)}."
            )
        if action_is_pad.dtype != torch.bool:
            raise TypeError("`action_is_pad` must use torch.bool.")
        if bool(action_is_pad.any().item()):
            padded_count = int(action_is_pad.sum().item())
            raise ValueError(
                "Formal training requires all 32 action targets; "
                f"action_is_pad must be all false, found {padded_count} pads."
            )
        action_is_pad = action_is_pad.to(device=self.device)
        action_raw = action.to(device=self.device, dtype=torch.float32)
        action_normalized = self.action_normalization_contract.normalize(
            action_raw
        ).to(dtype=self.torch_dtype)
        if not bool(torch.isfinite(action_normalized).all().item()):
            raise ValueError(
                "`action_normalized` became non-finite after dtype conversion."
            )
        prepared.update(
            {
                "action_raw": action_raw,
                "action_normalized": action_normalized,
                "proprio_raw": raw_proprio,
                "proprio_normalized": proprio_normalized,
                "action_is_pad": action_is_pad,
            }
        )
        return prepared

    def _coerce_noise(
        self,
        value: Optional[torch.Tensor],
        *,
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if value is None:
            return torch.randn_like(reference)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"`{name}` must be a torch.Tensor.")
        if value.shape != reference.shape:
            raise ValueError(
                f"`{name}` shape must be {tuple(reference.shape)}, got "
                f"{tuple(value.shape)}."
            )
        self._require_float_finite(value, name=name)
        return value.to(device=reference.device, dtype=reference.dtype)

    def _coerce_timestep(
        self,
        value: Optional[torch.Tensor],
        *,
        scheduler: WanContinuousFlowMatchScheduler,
        batch_size: int,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor:
        if value is None:
            return scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=dtype,
            )
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"`{name}` must be a torch.Tensor.")
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(
                f"`{name}` must be [B]={batch_size}, got " f"{tuple(value.shape)}."
            )
        self._require_float_finite(value, name=name)
        value = value.to(device=self.device, dtype=dtype)
        max_t = float(scheduler.num_train_timesteps)
        if bool(((value < 0) | (value > max_t)).any().item()):
            raise ValueError(f"`{name}` must lie in [0,{max_t}].")
        return value

    def _resolve_training_timesteps(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        momap_timestep: Optional[torch.Tensor],
        action_timestep: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.timestep_coupling == "independent":
            return (
                self._coerce_timestep(
                    momap_timestep,
                    scheduler=self.train_world_scheduler,
                    batch_size=batch_size,
                    dtype=dtype,
                    name="momap_timestep",
                ),
                self._coerce_timestep(
                    action_timestep,
                    scheduler=self.train_action_scheduler,
                    batch_size=batch_size,
                    dtype=dtype,
                    name="action_timestep",
                ),
            )

        shared_input = momap_timestep if momap_timestep is not None else action_timestep
        world_timestep = self._coerce_timestep(
            shared_input,
            scheduler=self.train_world_scheduler,
            batch_size=batch_size,
            dtype=dtype,
            name="shared_timestep",
        )
        action_resolved = self._coerce_timestep(
            action_timestep if action_timestep is not None else world_timestep,
            scheduler=self.train_action_scheduler,
            batch_size=batch_size,
            dtype=dtype,
            name="shared_action_timestep",
        )
        if momap_timestep is not None and action_timestep is not None:
            momap_resolved = self._coerce_timestep(
                momap_timestep,
                scheduler=self.train_world_scheduler,
                batch_size=batch_size,
                dtype=dtype,
                name="shared_momap_timestep",
            )
            if not torch.allclose(
                momap_resolved.float(),
                action_resolved.float(),
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError("Shared timestep overrides must be exactly equal.")
            world_timestep = momap_resolved
        return world_timestep, action_resolved

    @staticmethod
    def _action_losses(
        prediction: torch.Tensor,
        target_velocity: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        timestep_weight: torch.Tensor,
        *,
        feature_scale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        error = prediction.float() - target_velocity.float()
        if feature_scale is not None:
            scale = feature_scale.to(device=error.device, dtype=error.dtype)
            if scale.ndim != 1 or scale.shape[0] != error.shape[-1]:
                raise ValueError(
                    "Action feature scale must be [D], got "
                    f"{tuple(scale.shape)} for D={error.shape[-1]}."
                )
            error = error * scale.view(1, 1, -1)
        token_loss = error.square().mean(dim=2)
        if action_is_pad is None:
            per_sample = token_loss.mean(dim=1)
        else:
            valid = (~action_is_pad).to(
                device=token_loss.device,
                dtype=token_loss.dtype,
            )
            valid_count = valid.sum(dim=1)
            # Preserve the original FastWAM action-loss semantics exactly:
            # an all-padded sample contributes zero instead of dividing by zero.
            per_sample = (token_loss * valid).sum(dim=1) / valid_count.clamp_min(1.0)
        weight = timestep_weight.to(
            device=per_sample.device,
            dtype=per_sample.dtype,
        )
        if weight.ndim == 0:
            weight = weight.expand_as(per_sample)
        if weight.shape != per_sample.shape:
            raise ValueError(
                "Action scheduler training weight must be scalar or [B], got "
                f"{tuple(weight.shape)}."
            )
        raw = per_sample.mean()
        weighted = (per_sample * weight).mean()
        return raw, weighted, per_sample

    def _compute_momap_velocity_loss(
        self,
        prediction: torch.Tensor,
        target_velocity: torch.Tensor,
        *,
        inputs: Mapping[str, Any],
        timestep_weight: torch.Tensor,
    ):
        """Version hook for World loss reduction.

        Historical methods keep the original pixel-valid XYZ reduction.  The
        K4 heterogeneous wrapper overrides this hook to apply component-level
        All-view-XYZ supervision without changing old checkpoints.
        """

        return self.momap_objective.velocity_loss(
            prediction,
            target_velocity,
            valid_mask=inputs["flow_valid"],
            timestep_weight=timestep_weight,
        )

    def compute_world_loss(
        self,
        sample: Mapping[str, Any],
        *,
        momap_noise: Optional[torch.Tensor] = None,
        momap_timestep: Optional[torch.Tensor] = None,
        tiled: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute world-only FM loss without touching the Action branch."""

        if tiled:
            raise ValueError("PixelDiT MoMap training does not support tiling.")
        inputs = self._prepare_training_sample(sample, include_action=False)
        if getattr(self, "_capture_training_shape_evidence", False):
            self._last_training_shape_evidence = {
                "packed_target_flow": [
                    int(value) for value in inputs["target_flow_meters"].shape
                ],
                "packed_flow_valid": [
                    int(value) for value in inputs["flow_valid"].shape
                ],
            }
        world_condition = inputs["world_condition"]
        source_geometry_scale_meters = inputs["source_geometry_scale_meters"]
        target_flow_normalized = self._normalize_target_flow_for_training(
            inputs["target_flow_meters"],
            sample_scale_meters=source_geometry_scale_meters,
            inputs=inputs,
        ).to(dtype=self.torch_dtype)
        momap_noise = self._coerce_noise(
            momap_noise,
            reference=target_flow_normalized,
            name="momap_noise",
        )
        resolved_momap_t = self._coerce_timestep(
            momap_timestep,
            scheduler=self.train_world_scheduler,
            batch_size=int(world_condition.shape[0]),
            dtype=torch.float32,
            name="momap_timestep",
        )
        momap_batch = self.momap_objective.prepare(
            target_flow_normalized,
            noise=momap_noise,
            timestep=resolved_momap_t,
        )
        pixel_clean_t = self._pixel_clean_timestep(
            momap_batch.timestep,
            scheduler=self.train_world_scheduler,
        )
        predicted_momap_velocity = self.dit.forward_world_only(
            world_condition,
            momap_batch.noised_flow,
            pixel_clean_t,
            inputs["world_context"],
            inputs["world_context_mask"],
            source_support_mask=inputs["source_support_mask"],
        )
        momap_loss = self._compute_momap_velocity_loss(
            predicted_momap_velocity,
            momap_batch.target_velocity,
            inputs=inputs,
            timestep_weight=momap_batch.timestep_weight,
        )
        total = self.loss_lambda_momap * momap_loss.weighted
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError("World-only MoMap loss is NaN or Inf.")

        valid_ratios = momap_loss.valid_ratio_by_horizon.detach().float()
        metrics: dict[str, float] = {
            "loss/momap_raw": float(momap_loss.raw.detach().item()),
            "loss/momap_weighted": float(momap_loss.weighted.detach().item()),
            "loss/total": float(total.detach().item()),
            "valid_ratio/by_horizon": float(valid_ratios.mean().item()),
            "fm_timestep/momap": float(
                momap_batch.timestep.detach().float().mean().item()
            ),
        }
        metrics.update(
            {
                f"valid_ratio/h{index}": float(value.item())
                for index, value in enumerate(valid_ratios)
            }
        )
        if source_geometry_scale_meters is not None:
            scale_for_metrics = source_geometry_scale_meters.detach().float()
            source_valid_count = (
                inputs["source_support_mask"]
                .detach()
                .flatten(1)
                .sum(dim=1)
                .float()
            )
            metrics.update(
                {
                    "geometry_scale/source_xyz_p99_m_local_batch_mean": float(
                        scale_for_metrics.mean().item()
                    ),
                    "geometry_scale/source_xyz_p99_m_local_batch_min": float(
                        scale_for_metrics.amin().item()
                    ),
                    "geometry_scale/source_xyz_p99_m_local_batch_max": float(
                        scale_for_metrics.amax().item()
                    ),
                    "geometry_scale/source_valid_points_local_batch_mean": float(
                        source_valid_count.mean().item()
                    ),
                    "geometry_scale/source_valid_points_local_batch_min": float(
                        source_valid_count.amin().item()
                    ),
                    "geometry_scale/source_valid_points_local_batch_max": float(
                        source_valid_count.amax().item()
                    ),
                }
            )
        return total, metrics

    def compute_joint_loss(
        self,
        sample: Mapping[str, Any],
        *,
        momap_noise: Optional[torch.Tensor] = None,
        action_noise: Optional[torch.Tensor] = None,
        momap_timestep: Optional[torch.Tensor] = None,
        action_timestep: Optional[torch.Tensor] = None,
        return_audit: bool = False,
        tiled: bool = False,
    ) -> (
        tuple[torch.Tensor, dict[str, float]]
        | tuple[torch.Tensor, dict[str, float], dict[str, Any]]
    ):
        """Compute joint FM loss, or strict Action loss in action_only stage."""

        if tiled:
            raise ValueError("PixelDiT MoMap training does not support tiling.")
        action_only = self.training_stage == "action_only"
        if return_audit and action_only:
            raise ValueError(
                "return_audit is unavailable in action_only because the World "
                "prediction head and MoMap loss are intentionally skipped."
            )
        inputs = self._prepare_training_sample(sample)
        if getattr(self, "_capture_training_shape_evidence", False):
            self._last_training_shape_evidence = {
                "packed_target_flow": [
                    int(value) for value in inputs["target_flow_meters"].shape
                ],
                "packed_flow_valid": [
                    int(value) for value in inputs["flow_valid"].shape
                ],
            }
        world_condition = inputs["world_condition"]
        source_geometry_scale_meters = inputs["source_geometry_scale_meters"]
        target_flow_meters = inputs["target_flow_meters"]
        action_raw = inputs["action_raw"]
        action_normalized = inputs["action_normalized"]
        action = action_normalized
        batch_size = int(world_condition.shape[0])

        target_flow_normalized = self._normalize_target_flow_for_training(
            target_flow_meters,
            sample_scale_meters=source_geometry_scale_meters,
            inputs=inputs,
        ).to(dtype=self.torch_dtype)
        momap_noise = self._coerce_noise(
            momap_noise,
            reference=target_flow_normalized,
            name="momap_noise",
        )
        action_noise = self._coerce_noise(
            action_noise,
            reference=action,
            name="action_noise",
        )
        resolved_momap_t, resolved_action_t = self._resolve_training_timesteps(
            batch_size=batch_size,
            dtype=torch.float32,
            momap_timestep=momap_timestep,
            action_timestep=action_timestep,
        )
        momap_batch = self.momap_objective.prepare(
            target_flow_normalized,
            noise=momap_noise,
            timestep=resolved_momap_t,
        )
        noised_action = self.train_action_scheduler.add_noise(
            action,
            action_noise,
            resolved_action_t,
        )
        target_action_velocity = self.train_action_scheduler.training_target(
            action,
            action_noise,
            resolved_action_t,
        )
        action_weight = self.train_action_scheduler.training_weight(
            resolved_action_t
        ).to(device=self.device, dtype=torch.float32)

        pixel_clean_t = self._pixel_clean_timestep(
            momap_batch.timestep,
            scheduler=self.train_world_scheduler,
        )
        action_model_t = (
            self._action_noise_fraction_timestep(
                resolved_action_t,
                scheduler=self.train_action_scheduler,
            )
            if isinstance(self.action_expert, PixelDiTAlignedActionDiT)
            else resolved_action_t
        )
        outputs = self.dit(
            world_condition=world_condition,
            noised_flow=momap_batch.noised_flow,
            t_world=pixel_clean_t,
            noised_action=noised_action,
            t_action=action_model_t,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            world_context=inputs["world_context"],
            world_context_mask=inputs["world_context_mask"],
            source_support_mask=inputs["source_support_mask"],
            freeze_world=action_only,
            return_world_velocity=not action_only,
        )
        predicted_momap_velocity = outputs["world_velocity"]
        predicted_action_velocity = outputs["action_velocity"]
        momap_loss = None
        if not action_only:
            if not isinstance(predicted_momap_velocity, torch.Tensor):
                raise TypeError("Joint training requires a World velocity tensor.")
            momap_loss = self._compute_momap_velocity_loss(
                predicted_momap_velocity,
                momap_batch.target_velocity,
                inputs=inputs,
                timestep_weight=momap_batch.timestep_weight,
            )
        elif predicted_momap_velocity is not None:
            raise RuntimeError(
                "action_only must skip the World prediction head entirely."
            )
        action_loss_raw, action_weighted, action_per_sample = self._action_losses(
            predicted_action_velocity,
            target_action_velocity,
            inputs["action_is_pad"],
            action_weight,
        )
        action_feature_std = torch.as_tensor(
            self.action_normalization_contract.std,
            device=predicted_action_velocity.device,
            dtype=torch.float32,
        )
        (
            action_denormalized_raw,
            action_denormalized_weighted,
            _,
        ) = self._action_losses(
            predicted_action_velocity,
            target_action_velocity,
            inputs["action_is_pad"],
            action_weight,
            feature_scale=action_feature_std,
        )
        total = self.loss_lambda_action * action_weighted
        if momap_loss is not None:
            total = total + self.loss_lambda_momap * momap_loss.weighted
        if not bool(torch.isfinite(total).item()):
            objective = "Action-only" if action_only else "Joint MoMap/action"
            raise FloatingPointError(f"{objective} loss is NaN or Inf.")

        metrics: dict[str, float] = {
            "loss/action_raw": float(action_loss_raw.detach().item()),
            "loss/action_weighted": float(action_weighted.detach().item()),
            "loss/action_denormalized_raw": float(
                action_denormalized_raw.detach().item()
            ),
            "loss/action_denormalized_weighted": float(
                action_denormalized_weighted.detach().item()
            ),
            "loss/total": float(total.detach().item()),
            "fm_timestep/momap": float(
                momap_batch.timestep.detach().float().mean().item()
            ),
            "fm_timestep/action": float(
                resolved_action_t.detach().float().mean().item()
            ),
        }
        if self.method_contract == FORMAL_METHOD_CONTRACT_V9:
            correction_scale = outputs.get("world_logit_correction_scale")
            if not isinstance(correction_scale, (int, float)):
                raise TypeError(
                    "V9 aligned Action output must expose a scalar "
                    "world_logit_correction_scale."
                )
            metrics["native_kv_count_correction_scale"] = float(correction_scale)
        if momap_loss is not None:
            valid_ratios = momap_loss.valid_ratio_by_horizon.detach().float()
            metrics.update(
                {
                    "loss/momap_raw": float(momap_loss.raw.detach().item()),
                    "loss/momap_weighted": float(
                        momap_loss.weighted.detach().item()
                    ),
                    "valid_ratio/by_horizon": float(valid_ratios.mean().item()),
                    **{
                        f"valid_ratio/h{index}": float(value.item())
                        for index, value in enumerate(valid_ratios)
                    },
                }
            )
        if source_geometry_scale_meters is not None:
            scale_for_metrics = source_geometry_scale_meters.detach().float()
            source_valid_count = (
                inputs["source_support_mask"].detach().flatten(1).sum(dim=1).float()
            )
            metrics.update(
                {
                    "geometry_scale/source_xyz_p99_m_local_batch_mean": float(
                        scale_for_metrics.mean().item()
                    ),
                    "geometry_scale/source_xyz_p99_m_local_batch_min": float(
                        scale_for_metrics.amin().item()
                    ),
                    "geometry_scale/source_xyz_p99_m_local_batch_max": float(
                        scale_for_metrics.amax().item()
                    ),
                    "geometry_scale/source_valid_points_local_batch_mean": float(
                        source_valid_count.mean().item()
                    ),
                    "geometry_scale/source_valid_points_local_batch_min": float(
                        source_valid_count.amin().item()
                    ),
                    "geometry_scale/source_valid_points_local_batch_max": float(
                        source_valid_count.amax().item()
                    ),
                }
            )
        if not return_audit:
            return total, metrics

        assert momap_loss is not None
        assert isinstance(predicted_momap_velocity, torch.Tensor)
        valid_ratios = momap_loss.valid_ratio_by_horizon.detach().float()
        momap_sigma = (
            momap_batch.timestep / float(self.train_world_scheduler.num_train_timesteps)
        ).view(batch_size, 1, 1, 1)
        action_sigma = (
            resolved_action_t / float(self.train_action_scheduler.num_train_timesteps)
        ).view(batch_size, 1, 1)
        momap_clean_estimate_normalized = (
            momap_batch.noised_flow
            - momap_sigma.to(
                dtype=momap_batch.noised_flow.dtype,
                device=momap_batch.noised_flow.device,
            )
            * predicted_momap_velocity
        )
        momap_clean_estimate_meters = self._denormalize_geometry_offset(
            momap_clean_estimate_normalized.to(dtype=torch.float32),
            sample_scale_meters=source_geometry_scale_meters,
        )
        action_clean_estimate = (
            noised_action
            - action_sigma.to(
                dtype=noised_action.dtype,
                device=noised_action.device,
            )
            * predicted_action_velocity
        )
        action_clean_estimate_normalized = action_clean_estimate
        action_clean_estimate_raw = self.action_normalization_contract.denormalize(
            action_clean_estimate_normalized.to(dtype=torch.float32)
        )
        audit = {
            "momap_clean_meters": target_flow_meters,
            "momap_clean_normalized": target_flow_normalized,
            "momap_noise": momap_batch.noise,
            "momap_noised": momap_batch.noised_flow,
            "momap_target_velocity": momap_batch.target_velocity,
            "momap_pred_velocity": predicted_momap_velocity,
            "momap_timestep": momap_batch.timestep,
            "momap_timestep_weight": momap_batch.timestep_weight,
            "momap_clean_estimate_normalized": (momap_clean_estimate_normalized),
            "momap_clean_estimate_meters": momap_clean_estimate_meters,
            "flow_valid": inputs["flow_valid"],
            "momap_lateness_seconds": inputs["momap_lateness_seconds"],
            "valid_ratio_by_horizon": valid_ratios,
            "valid_horizon_ratio": (momap_loss.valid_horizon_ratio.detach().float()),
            "action_clean": action_raw,
            "action_clean_raw": action_raw,
            "action_clean_normalized": action_normalized,
            "action_noise": action_noise,
            "action_noised": noised_action,
            "action_target_velocity": target_action_velocity,
            "action_pred_velocity": predicted_action_velocity,
            "action_timestep": resolved_action_t,
            "action_timestep_weight": action_weight,
            "action_loss_per_sample": action_per_sample,
            "action_clean_estimate": action_clean_estimate_raw,
            "action_clean_estimate_raw": action_clean_estimate_raw,
            "action_clean_estimate_normalized": (action_clean_estimate_normalized),
        }
        for component_mask_key in ("flow_component_valid", "flow_loss_valid"):
            if component_mask_key in inputs:
                audit[component_mask_key] = inputs[component_mask_key]
        if source_geometry_scale_meters is not None:
            audit["source_geometry_scale_meters"] = source_geometry_scale_meters
        return total, metrics, audit

    def training_loss(
        self,
        sample: Mapping[str, Any],
        tiled: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.training_stage == "world_only":
            return self.compute_world_loss(sample, tiled=tiled)
        if self.training_stage not in {"action_only", "joint"}:
            raise RuntimeError(
                f"Unsupported runtime training_stage {self.training_stage!r}."
            )
        result = self.compute_joint_loss(sample, tiled=tiled)
        assert len(result) == 2
        return result

    def _prepare_observation(
        self,
        observation: Optional[Mapping[str, Any]],
        *,
        clean_image: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
        proprio_mode: Optional[str],
        proprio_normalization: Optional[str],
        source_support_mask: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        if source_support_mask is not None:
            raise ValueError(
                "Formal wrapper rejects externally supplied source_support_mask "
                "until a source-observation-only producer and provenance contract "
                "are implemented."
            )
        direct_values = {
            "clean_image": clean_image,
            "context": context,
            "context_mask": context_mask,
            "proprio": proprio,
            "proprio_mode": proprio_mode,
            "proprio_normalization": proprio_normalization,
        }
        supplied_direct = [
            key for key, value in direct_values.items() if value is not None
        ]
        if observation is None:
            if supplied_direct:
                raise ValueError(
                    "V2 inference requires a structured `observation` containing "
                    "RGB, segmentation, and canonical XYZ; flat direct condition "
                    f"arguments are not sufficient: {supplied_direct}."
                )
            raise ValueError(
                "V2 inference requires a structured target-free `observation`."
            )
        if not isinstance(observation, Mapping):
            raise TypeError("`observation` must be a mapping.")
        if supplied_direct:
            raise ValueError(
                "`observation` is mutually exclusive with direct "
                f"condition arguments: {supplied_direct}."
            )
        forbidden = sorted(set(observation) & self._INFERENCE_FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError(
                "Target-free inference rejects target/future/private "
                f"fields: {forbidden}."
            )
        raw_source_rgb = observation.get("source_rgb")
        is_raw_multiview = (
            isinstance(raw_source_rgb, torch.Tensor)
            and raw_source_rgb.ndim == 5
        )
        if is_raw_multiview:
            self._strict_keyset(
                observation,
                required=self._RAW_OBSERVATION_REQUIRED_KEYS,
                optional=self._RAW_OBSERVATION_OPTIONAL_KEYS,
                name="Raw target-free inference observation",
            )
            values = self.batch_adapter.adapt_observation(observation)
        else:
            values = dict(observation)
        self._strict_keyset(
            values,
            required=self._OBSERVATION_REQUIRED_KEYS,
            optional=self._OBSERVATION_OPTIONAL_KEYS,
            name="Adapted inference observation",
        )

        (
            world_condition,
            prepared_support_mask,
            source_geometry_scale_meters,
        ) = self._prepare_world_condition(values)
        prepared_context = self._required_tensor(values, "context")
        batch_size = int(world_condition.shape[0])
        raw_context_mask = values.get("context_mask")
        if raw_context_mask is not None and not isinstance(
            raw_context_mask,
            torch.Tensor,
        ):
            raise TypeError("`context_mask` must be a torch.Tensor.")
        raw_proprio = values.get("proprio")
        raw_proprio_mode = values.get("proprio_mode")
        raw_proprio_normalization = values.get("proprio_normalization")
        proprio_normalized: Optional[torch.Tensor] = None
        if self.proprio_dim is None:
            if raw_proprio is not None:
                raise ValueError(
                    "Inference provides proprio but wrapper has `proprio_dim=None`."
                )
            if raw_proprio_mode is not None or raw_proprio_normalization is not None:
                raise ValueError("Proprio metadata is invalid without proprio.")
        else:
            if not isinstance(raw_proprio, torch.Tensor):
                raise TypeError("`proprio` must be a torch.Tensor.")
            if not isinstance(raw_proprio_mode, str) or not isinstance(
                raw_proprio_normalization,
                str,
            ):
                raise TypeError(
                    "Target-free proprio requires string `proprio_mode` and "
                    "`proprio_normalization` metadata."
                )
            assert self.proprio_normalization_contract is not None
            if raw_proprio_mode != self.proprio_normalization_contract.semantic_kind:
                raise ValueError("Inference proprio_mode does not match the contract.")
            if (
                raw_proprio_normalization
                != self.proprio_normalization_contract.source_normalization
            ):
                raise ValueError(
                    "Inference proprio_normalization does not match the raw contract."
                )
            allowed_shapes = {
                (batch_size, self.proprio_dim),
                (batch_size, 1, self.proprio_dim),
            }
            if tuple(raw_proprio.shape) not in allowed_shapes:
                raise ValueError(
                    "Inference proprio must be the current observation [B,14] or "
                    f"[B,1,14], got {tuple(raw_proprio.shape)}."
                )
            self._require_float_finite(raw_proprio, name="proprio_raw")
            proprio_fp32 = raw_proprio.to(
                device=self.device,
                dtype=torch.float32,
            )
            proprio_normalized = (
                self.proprio_normalization_contract.normalize(proprio_fp32).to(
                    dtype=self.torch_dtype
                )
            )
        (
            action_context,
            action_context_mask,
            world_context,
            world_context_mask,
        ) = self._prepare_context(
            prepared_context,
            raw_context_mask,
            proprio_normalized,
            batch_size=batch_size,
        )
        return {
            "world_condition": world_condition,
            "source_geometry_scale_meters": source_geometry_scale_meters,
            "context": action_context,
            "context_mask": action_context_mask,
            "world_context": world_context,
            "world_context_mask": world_context_mask,
            "source_support_mask": prepared_support_mask,
        }

    def _inference_noise(
        self,
        value: Optional[torch.Tensor],
        *,
        shape: tuple[int, ...],
        name: str,
        seed: Optional[int],
        seed_offset: int,
    ) -> torch.Tensor:
        if value is not None:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"`{name}` must be a torch.Tensor.")
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"`{name}` must have shape {shape}, got " f"{tuple(value.shape)}."
                )
            self._require_float_finite(value, name=name)
            return value.to(device=self.device, dtype=torch.float32)
        if seed is None:
            return torch.randn(
                shape,
                device=self.device,
                dtype=torch.float32,
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + seed_offset)
        return torch.randn(
            shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=self.device)

    @torch.no_grad()
    def infer_joint(
        self,
        observation: Optional[Mapping[str, Any]] = None,
        *,
        clean_image: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        proprio_mode: Optional[str] = None,
        proprio_normalization: Optional[str] = None,
        source_support_mask: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        num_inference_steps: int = 20,
        momap_noise: Optional[torch.Tensor] = None,
        action_noise: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Joint Euler sampling from independent Gaussian MoMap/action states."""

        if action_horizon is None:
            action_horizon = self.action_horizon
        else:
            action_horizon = self._positive_int(
                action_horizon,
                name="action_horizon",
            )
            if action_horizon != self.action_horizon:
                raise ValueError(
                    "Inference action_horizon must equal the checkpointed "
                    f"contract {self.action_horizon}, got {action_horizon}."
                )
        num_inference_steps = self._positive_int(
            num_inference_steps,
            name="num_inference_steps",
        )
        if seed is not None:
            if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
                raise ValueError(
                    f"`seed` must be a non-negative integer, got {seed!r}."
                )
            seed = int(seed)
        conditions = self._prepare_observation(
            observation,
            clean_image=clean_image,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            proprio_mode=proprio_mode,
            proprio_normalization=proprio_normalization,
            source_support_mask=source_support_mask,
        )
        world_condition = conditions["world_condition"]
        batch_size, _, height, width = world_condition.shape
        momap_state = self._inference_noise(
            momap_noise,
            shape=(
                batch_size,
                self.kflow_contract.flow_channels,
                height,
                width,
            ),
            name="momap_noise",
            seed=seed,
            seed_offset=0,
        )
        action_state = self._inference_noise(
            action_noise,
            shape=(
                batch_size,
                action_horizon,
                int(self.action_expert.action_dim),
            ),
            name="action_noise",
            seed=seed,
            seed_offset=1,
        )
        world_timesteps, world_deltas = (
            self.infer_world_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=torch.float32,
            )
        )
        action_timesteps, action_deltas = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=torch.float32,
            )
        )
        schedules = (
            world_timesteps,
            world_deltas,
            action_timesteps,
            action_deltas,
        )
        if any(
            schedule.ndim != 1 or schedule.shape[0] != num_inference_steps
            for schedule in schedules
        ):
            raise RuntimeError(
                "World/action inference schedules must have the same requested "
                "lockstep length."
            )

        was_training = self.training
        self.eval()
        try:
            for step_index in range(num_inference_steps):
                raw_world_t = world_timesteps[step_index].expand(batch_size)
                raw_action_t = action_timesteps[step_index].expand(batch_size)
                pixel_clean_t = self._pixel_clean_timestep(
                    raw_world_t,
                    scheduler=self.infer_world_scheduler,
                )
                action_model_t = (
                    self._action_noise_fraction_timestep(
                        raw_action_t,
                        scheduler=self.infer_action_scheduler,
                    )
                    if isinstance(self.action_expert, PixelDiTAlignedActionDiT)
                    else raw_action_t
                )
                output = self.dit(
                    world_condition=world_condition,
                    noised_flow=momap_state.to(
                        dtype=self.torch_dtype,
                    ),
                    t_world=pixel_clean_t,
                    noised_action=action_state.to(
                        dtype=self.torch_dtype,
                    ),
                    t_action=action_model_t,
                    context=conditions["context"],
                    context_mask=conditions["context_mask"],
                    source_support_mask=conditions["source_support_mask"],
                    world_context=conditions["world_context"],
                    world_context_mask=conditions["world_context_mask"],
                )
                momap_state = self.infer_world_scheduler.step(
                    output["world_velocity"].to(
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    world_deltas[step_index],
                    momap_state,
                )
                action_state = self.infer_action_scheduler.step(
                    output["action_velocity"].to(
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    action_deltas[step_index],
                    action_state,
                )
                if not bool(
                    torch.isfinite(momap_state).all().item()
                    and torch.isfinite(action_state).all().item()
                ):
                    raise FloatingPointError(
                        "Joint FP32 Euler state became NaN or Inf."
                    )
        finally:
            self.train(was_training)

        source_geometry_scale_meters = conditions[
            "source_geometry_scale_meters"
        ]
        packed_meters = self._denormalize_geometry_offset(
            momap_state.to(dtype=torch.float32),
            sample_scale_meters=source_geometry_scale_meters,
        )
        action_raw = self.action_normalization_contract.denormalize(
            action_state.to(dtype=torch.float32)
        )
        structured_meters = unpack_kflow(
            packed_meters,
            k=self.kflow_contract.k,
        )
        result = {
            "momap_normalized": momap_state,
            "momap_meters": packed_meters,
            "momap_structured_meters": structured_meters,
            "action_normalized": action_state,
            "action_raw": action_raw,
            "action": action_raw,
        }
        if source_geometry_scale_meters is not None:
            result["source_geometry_scale_meters"] = source_geometry_scale_meters
        return result

    @torch.no_grad()
    def sample_joint(
        self,
        observation: Optional[Mapping[str, Any]] = None,
        *,
        clean_image: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        proprio_mode: Optional[str] = None,
        proprio_normalization: Optional[str] = None,
        source_support_mask: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        num_inference_steps: int = 20,
        momap_noise: Optional[torch.Tensor] = None,
        action_noise: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        return self.infer_joint(
            observation,
            clean_image=clean_image,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            proprio_mode=proprio_mode,
            proprio_normalization=proprio_normalization,
            source_support_mask=source_support_mask,
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            momap_noise=momap_noise,
            action_noise=action_noise,
            seed=seed,
        )

    def _initialization_report(self) -> dict[str, Any]:
        action_report = getattr(
            self.action_expert,
            "pretrained_load_report",
            None,
        )
        if action_report is not None and not isinstance(action_report, Mapping):
            raise TypeError(
                "ActionDiT `pretrained_load_report` must be a mapping or None."
            )
        pixeldit_report = getattr(self, "pixeldit_load_report", None)
        if pixeldit_report is not None and not isinstance(
            pixeldit_report,
            Mapping,
        ):
            raise TypeError(
                "PixelDiT `pixeldit_load_report` must be a mapping or None."
            )
        normalization_sources = getattr(
            self,
            "normalization_contract_sources",
            None,
        )
        if normalization_sources is not None and not isinstance(
            normalization_sources,
            Mapping,
        ):
            raise TypeError(
                "`normalization_contract_sources` must be a mapping or None."
            )
        method_contract = getattr(self, "method_contract", None)
        if method_contract is not None and not isinstance(method_contract, Mapping):
            raise TypeError("`method_contract` must be a mapping or None.")
        return {
            "model_paths": copy.deepcopy(getattr(self, "model_paths", None)),
            "pixeldit_load_report": copy.deepcopy(pixeldit_report),
            "actiondit_pretrained_load_report": copy.deepcopy(action_report),
            "normalization_contract_sources": copy.deepcopy(
                normalization_sources
            ),
            "method_contract": copy.deepcopy(method_contract),
        }

    def _optimizer_checkpoint_policy(self) -> dict[str, Any]:
        action_report = getattr(
            self.action_expert,
            "pretrained_load_report",
            None,
        )
        source_policy = (
            action_report.get("optimizer_state_policy")
            if isinstance(action_report, Mapping)
            else None
        )
        return {
            "format": "fastwam_pixeldit_joint_optimizer_v2",
            "checkpoint_optimizer_scope": "joint_v2_post_initialization",
            "action_pretrained_optimizer_state_policy": (
                "not_applicable" if source_policy is None else source_policy
            ),
            "pretrained_optimizer_state_consumed": False,
        }

    @staticmethod
    def _validate_module_state_dict(
        module: nn.Module,
        state: Mapping[str, Any],
        *,
        name: str,
    ) -> None:
        from mwam.utils.checkpoint_integrity import (
            validate_floating_state_dict,
        )

        validate_floating_state_dict(state, name=f"checkpoint {name}")
        expected = module.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        if missing or unexpected:
            raise ValueError(
                f"Checkpoint `{name}` key mismatch: missing={missing}, "
                f"unexpected={unexpected}."
            )
        non_tensors = sorted(
            key for key, value in state.items() if not isinstance(value, torch.Tensor)
        )
        if non_tensors:
            raise TypeError(
                f"Checkpoint `{name}` entries must be tensors; "
                f"invalid keys={non_tensors}."
            )
        shape_mismatches = {
            key: (tuple(state[key].shape), tuple(expected[key].shape))
            for key in expected
            if state[key].shape != expected[key].shape
        }
        if shape_mismatches:
            raise ValueError(
                f"Checkpoint `{name}` shape mismatch before loading: "
                f"{shape_mismatches}."
            )

    @staticmethod
    def _preflight_optimizer_state(
        optimizer: torch.optim.Optimizer,
        state: Mapping[str, Any],
    ) -> None:
        backup = copy.deepcopy(optimizer.state_dict())
        error: Optional[Exception] = None
        try:
            optimizer.load_state_dict(copy.deepcopy(dict(state)))
        except Exception as exc:  # pragma: no cover - optimizer-specific text
            error = exc
        finally:
            optimizer.load_state_dict(backup)
        if error is not None:
            raise ValueError(
                "Checkpoint optimizer state is incompatible with the current "
                f"optimizer: {error}."
            ) from error

    @staticmethod
    def _normalization_math_contract(value: Mapping[str, Any]) -> dict[str, Any]:
        """Keep normalization semantics and values, not artifact identity."""

        if not isinstance(value, Mapping):
            raise TypeError("Checkpoint normalization contract must be a mapping.")
        result = copy.deepcopy(dict(value))
        for key in tuple(result):
            if key in {
                "digest",
                "statistics_id",
                "split_name",
            } or key.endswith("_sha256"):
                result.pop(key, None)
        return result

    @classmethod
    def _resume_training_contract(
        cls,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return the model/math contract used for resume compatibility."""

        if not isinstance(value, Mapping):
            raise TypeError("Checkpoint `training_contract` must be a mapping.")
        result = copy.deepcopy(dict(value))
        # These legacy fields describe where initialization came from. They do
        # not affect the checkpoint parameter or optimizer structures.
        result.pop("initialization", None)
        result.pop("optimizer_policy", None)
        for key in (
            "normalization",
            "action_normalization",
            "proprio_normalization",
        ):
            raw = result.get(key)
            if raw is not None:
                result[key] = cls._normalization_math_contract(raw)
        return result

    def save_checkpoint(
        self,
        path: str | Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: Optional[int] = None,
    ) -> None:
        if step is not None and (
            isinstance(step, bool) or int(step) != step or int(step) < 0
        ):
            raise ValueError(
                f"`step` must be a non-negative integer or None, got {step!r}."
            )
        payload: dict[str, Any] = {
            "format": self.CHECKPOINT_FORMAT,
            "training_contract": self._checkpoint_contract(),
            "model_state": self.state_dict(),
            "step": None if step is None else int(step),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        _atomic_torch_save(payload, path)

    def load_checkpoint(
        self,
        path: str | Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict[str, Any]:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise TypeError("Checkpoint payload must be a dictionary.")
        required = {"format", "training_contract", "model_state", "step"}
        # Initialization/provenance metadata from older checkpoints is accepted
        # for compatibility but is never a resume gate.
        allowed = required | {"optimizer", "initialization", "optimizer_policy"}
        missing = sorted(required - set(payload))
        unexpected = sorted(set(payload) - allowed)
        if missing or unexpected:
            raise ValueError(
                "Checkpoint top-level key mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if payload["format"] != self.CHECKPOINT_FORMAT:
            raise ValueError(f"Checkpoint is not {self.CHECKPOINT_FORMAT!r}: {path}.")
        loaded_contract = self._resume_training_contract(
            payload["training_contract"]
        )
        if loaded_contract != self._checkpoint_contract():
            raise ValueError(
                "Checkpoint training contract does not match the current "
                "wrapper's model/math contract."
            )
        if payload["step"] is not None and (
            isinstance(payload["step"], bool)
            or not isinstance(payload["step"], int)
            or payload["step"] < 0
        ):
            raise ValueError("Checkpoint `step` must be non-negative or None.")
        if not isinstance(payload["model_state"], Mapping):
            raise TypeError("Checkpoint `model_state` must be a mapping.")
        if optimizer is not None:
            if not isinstance(payload.get("optimizer"), Mapping):
                raise ValueError("Checkpoint is missing a valid optimizer state.")

        self._validate_module_state_dict(
            self,
            payload["model_state"],
            name="model_state",
        )
        if optimizer is not None:
            self._preflight_optimizer_state(
                optimizer,
                payload["optimizer"],
            )

        self.load_state_dict(payload["model_state"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    @staticmethod
    def _contract_difference_paths(
        source: Any,
        target: Any,
        *,
        atomic_paths: frozenset[str],
        path: tuple[str, ...] = (),
    ) -> set[str]:
        label = ".".join(path)
        if label in atomic_paths:
            return set() if source == target else {label}
        if isinstance(source, Mapping) and isinstance(target, Mapping):
            differences: set[str] = set()
            for key in sorted(set(source) | set(target)):
                child_path = path + (str(key),)
                if key not in source or key not in target:
                    differences.add(".".join(child_path))
                    continue
                differences.update(
                    FastWAMPixelDiT._contract_difference_paths(
                        source[key],
                        target[key],
                        atomic_paths=atomic_paths,
                        path=child_path,
                    )
                )
            return differences
        if source != target:
            return {label or "<root>"}
        return set()

    @staticmethod
    def _require_transition_tensor_compatibility(
        source: Mapping[str, Any],
        target: Mapping[str, Any],
        *,
        name: str,
    ) -> None:
        non_tensors = sorted(
            key for key, value in source.items() if not isinstance(value, torch.Tensor)
        )
        if non_tensors:
            raise TypeError(
                f"Checkpoint `{name}` entries must be tensors; invalid keys="
                f"{non_tensors}."
            )
        missing = sorted(set(target) - set(source))
        unexpected = sorted(set(source) - set(target))
        if missing or unexpected:
            raise ValueError(
                f"Checkpoint `{name}` key mismatch: missing={missing}, "
                f"unexpected={unexpected}."
            )
        mismatches = {
            key: {
                "source_shape": list(source[key].shape),
                "target_shape": list(target[key].shape),
                "source_dtype": str(source[key].dtype),
                "target_dtype": str(target[key].dtype),
            }
            for key in target
            if (
                source[key].shape != target[key].shape
                or source[key].dtype != target[key].dtype
            )
        }
        if mismatches:
            raise ValueError(
                f"Checkpoint `{name}` tensor schema mismatch: {mismatches}."
            )

    def load_checkpoint_for_fresh_stage_transition(
        self,
        path: str | Path,
        *,
        transition_id: Any,
        source_topology: Any,
        target_topology: Any,
        state_dict_policy: Any,
    ) -> dict[str, Any]:
        """Import weights through one exact, allowlisted fresh-phase transition."""

        transition_specs: dict[str, dict[str, Any]] = {
            "v8_action14_random_from_frozen_s08119_v1": {
                "source_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "target_topology": PixelDiTAlignedActionMoT.TOPOLOGY_VERSION,
                "state_dict_policy": ACTION14_RANDOM_INITIALIZATION,
                "source_method": FORMAL_METHOD_CONTRACT_V6,
                "target_method": FORMAL_METHOD_CONTRACT_V8,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION
                ),
                "source_step": 8119,
                "allowed_contract_differences": frozenset(
                    {
                        "action",
                        "bridge",
                        "kflow.architecture_version",
                        "method_contract",
                    }
                ),
                "atomic_paths": frozenset(
                    {"action", "bridge", "method_contract"}
                ),
            },
            "v8_action14_world_transfer_from_frozen_s08119_v1": {
                "source_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "target_topology": PixelDiTAlignedActionMoT.TOPOLOGY_VERSION,
                "state_dict_policy": ACTION14_WORLD_TRANSFER_INITIALIZATION,
                "source_method": FORMAL_METHOD_CONTRACT_V6,
                "target_method": FORMAL_METHOD_CONTRACT_V8,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION
                ),
                "source_step": 8119,
                "allowed_contract_differences": frozenset(
                    {
                        "action",
                        "bridge",
                        "kflow.architecture_version",
                        "method_contract",
                    }
                ),
                "atomic_paths": frozenset(
                    {"action", "bridge", "method_contract"}
                ),
            },
            "v8_action14_worldtransfer_s08119_fresh32_joint_staticbias_v1": {
                "source_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "target_topology": PixelDiTAlignedActionMoT.TOPOLOGY_VERSION,
                "state_dict_policy": ACTION14_WORLD_TRANSFER_INITIALIZATION,
                "source_method": FORMAL_METHOD_CONTRACT_V6,
                "target_method": FORMAL_METHOD_CONTRACT_V8,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION
                ),
                "source_step": 8119,
                "allowed_contract_differences": frozenset(
                    {
                        "action",
                        "bridge",
                        "kflow.architecture_version",
                        "method_contract",
                    }
                ),
                "atomic_paths": frozenset(
                    {"action", "bridge", "method_contract"}
                ),
            },
            "v8_action14_worldtransfer_s09502_actiononly_to_v9_joint_nobias_anneal500_16rank_v1": {
                "source_topology": PixelDiTAlignedActionMoT.TOPOLOGY_VERSION,
                "target_topology": PixelDiTAlignedActionMoT.TOPOLOGY_VERSION,
                "state_dict_policy": "exact_schema",
                "source_method": FORMAL_METHOD_CONTRACT_V8,
                "target_method": FORMAL_METHOD_CONTRACT_V9,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION
                ),
                "source_step": 9502,
                "allowed_contract_differences": frozenset(
                    {
                        "action.native_kv_fusion",
                        "bridge.bridge_semantics_version",
                        "bridge.bridge_source_schedule",
                        "method_contract",
                        "native_kv_fusion_transition",
                    }
                ),
                "atomic_paths": frozenset(
                    {
                        "action.native_kv_fusion",
                        "bridge.bridge_semantics_version",
                        "bridge.bridge_source_schedule",
                        "method_contract",
                        "native_kv_fusion_transition",
                    }
                ),
            },
            "v9_action14_worldtransfer_s08119_fresh_step0_joint_anneal500_16rank_v1": {
                "source_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "target_topology": PixelDiTAlignedActionMoT.TOPOLOGY_VERSION,
                "state_dict_policy": ACTION14_WORLD_TRANSFER_INITIALIZATION,
                "source_method": FORMAL_METHOD_CONTRACT_V6,
                "target_method": FORMAL_METHOD_CONTRACT_V9,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION
                ),
                "source_step": 8119,
                "allowed_contract_differences": frozenset(
                    {
                        "action",
                        "bridge",
                        "kflow.architecture_version",
                        "method_contract",
                        "native_kv_fusion_transition",
                    }
                ),
                "atomic_paths": frozenset(
                    {"action", "bridge", "method_contract"}
                ),
            },
            "v7_h8s4_a30_allnative_highnoise405505_uniform_to_initial10_v1": {
                "source_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "target_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "state_dict_policy": "exact_schema",
                "source_method": FORMAL_METHOD_CONTRACT_V6,
                "target_method": FORMAL_METHOD_CONTRACT_V6,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "source_step": 20000,
                "allowed_contract_differences": frozenset(),
                "atomic_paths": frozenset(),
            },
            "v7_h8s4_a30_allnative_highnoise405505_uniform_to_initial10_32rank_v1": {
                "source_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "target_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "state_dict_policy": "exact_schema",
                "source_method": FORMAL_METHOD_CONTRACT_V6,
                "target_method": FORMAL_METHOD_CONTRACT_V6,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "source_step": 20000,
                "allowed_contract_differences": frozenset(),
                "atomic_paths": frozenset(),
            },
            "v7_h8s4_a30_allnative_weighted_to_highnoise405505_uniform_v1": {
                "source_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "target_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "state_dict_policy": "exact_schema",
                "source_method": FORMAL_METHOD_CONTRACT_V6,
                "target_method": FORMAL_METHOD_CONTRACT_V6,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "source_step": 20000,
                "allowed_contract_differences": frozenset(
                    {
                        "loss.scheduler_weighting",
                        "schedulers.train_action",
                        "schedulers.train_world",
                    }
                ),
                "atomic_paths": frozenset(
                    {
                        "schedulers.train_action",
                        "schedulers.train_world",
                    }
                ),
            },
            "v7_h8s4_a30_pitcompact_weighted_to_highnoise405505_uniform_v1": {
                "source_topology": (
                    PixelDiTActionMoT.ACTION30_PIT_ONLY_TOPOLOGY_VERSION
                ),
                "target_topology": (
                    PixelDiTActionMoT.ACTION30_PIT_ONLY_TOPOLOGY_VERSION
                ),
                "state_dict_policy": "exact_schema",
                "source_method": FORMAL_METHOD_CONTRACT_V3,
                "target_method": FORMAL_METHOD_CONTRACT_V3,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION
                ),
                "source_step": 20000,
                "allowed_contract_differences": frozenset(
                    {
                        "loss.scheduler_weighting",
                        "schedulers.train_action",
                        "schedulers.train_world",
                    }
                ),
                "atomic_paths": frozenset(
                    {
                        "schedulers.train_action",
                        "schedulers.train_world",
                    }
                ),
            },
            "v7_h8s4_nores_l24_a30_allnative_weighted_to_highnoise405505_uniform_v1": {
                "source_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "target_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "state_dict_policy": "exact_schema",
                "source_method": FORMAL_METHOD_CONTRACT_V6,
                "target_method": FORMAL_METHOD_CONTRACT_V6,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "source_step": 20000,
                "allowed_contract_differences": frozenset(
                    {
                        "loss.scheduler_weighting",
                        "schedulers.train_action",
                        "schedulers.train_world",
                    }
                ),
                "atomic_paths": frozenset(
                    {
                        "schedulers.train_action",
                        "schedulers.train_world",
                    }
                ),
            },
            "v7_h8s4_nores_l24_a30_allnative_highnoise405505_uniform_to_initial10_8to16_v1": {
                "source_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "target_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "state_dict_policy": "exact_schema",
                "source_method": FORMAL_METHOD_CONTRACT_V6,
                "target_method": FORMAL_METHOD_CONTRACT_V6,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "source_step": 20000,
                "allowed_contract_differences": frozenset(),
                "atomic_paths": frozenset(),
            },
            "v7_nores_l24_initial10_to_failure_replay_groupedlr_from_step31970_v1": {
                "source_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "target_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "state_dict_policy": "exact_schema",
                "source_method": FORMAL_METHOD_CONTRACT_V6,
                "target_method": FORMAL_METHOD_CONTRACT_V6,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "source_step": 31970,
                "allowed_contract_differences": frozenset(),
                "atomic_paths": frozenset(),
            },
            "v7_nores_l24_lr5e3_actiononly_from_step8119_v1": {
                "source_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "target_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "state_dict_policy": "exact_schema",
                "source_method": FORMAL_METHOD_CONTRACT_V6,
                "target_method": FORMAL_METHOD_CONTRACT_V6,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "source_step": 8119,
                "allowed_contract_differences": frozenset(),
                "atomic_paths": frozenset(),
            },
            "v7_nores_l24_lr5e3_joint_from_step6395_v1": {
                "source_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "target_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "state_dict_policy": "exact_schema",
                "source_method": FORMAL_METHOD_CONTRACT_V6,
                "target_method": FORMAL_METHOD_CONTRACT_V6,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "source_step": 6395,
                "allowed_contract_differences": frozenset(),
                "atomic_paths": frozenset(),
            },
            "v7_world_v2_to_pit_only_action30_v1": {
                "source_topology": PixelDiTActionMoT.ACTION30_TOPOLOGY_VERSION,
                "target_topology": (
                    PixelDiTActionMoT.ACTION30_PIT_ONLY_TOPOLOGY_VERSION
                ),
                "state_dict_policy": "exact_schema",
                "source_method": FORMAL_METHOD_CONTRACT_V2,
                "target_method": FORMAL_METHOD_CONTRACT_V3,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION
                ),
                "allowed_contract_differences": frozenset(
                    {
                        "bridge.bridge_semantics_version",
                        "bridge.bridge_source_schedule",
                        "bridge.topology_version",
                        "kflow.architecture_version",
                        "method_contract",
                    }
                ),
                "atomic_paths": frozenset(
                    {"bridge.bridge_source_schedule", "method_contract"}
                ),
            },
            "v7_world_v2_to_global_only_action26_v1": {
                "source_topology": PixelDiTActionMoT.ACTION30_TOPOLOGY_VERSION,
                "target_topology": PixelDiTActionMoT.LEGACY_TOPOLOGY_VERSION,
                "state_dict_policy": "drop_action_tail_v1",
                "source_method": FORMAL_METHOD_CONTRACT_V2,
                "target_method": FORMAL_METHOD_CONTRACT_V4,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_ARCHITECTURE_VERSION
                ),
                "allowed_contract_differences": frozenset(
                    {
                        "action.norm_eps_by_layer",
                        "action.num_layers",
                        "bridge.adapter_count",
                        "bridge.adapters",
                        "bridge.bridge_semantics_version",
                        "bridge.bridge_source_schedule",
                        "bridge.layer_map",
                        "bridge.topology_version",
                        "kflow.architecture_version",
                        "method_contract",
                    }
                ),
                "atomic_paths": frozenset(
                    {
                        "action.norm_eps_by_layer",
                        "bridge.adapters",
                        "bridge.bridge_source_schedule",
                        "bridge.layer_map",
                        "method_contract",
                    }
                ),
            },
            "v7_world_v2_to_global_native_kv_mot_action26_v1": {
                "source_topology": PixelDiTActionMoT.ACTION30_TOPOLOGY_VERSION,
                "target_topology": (
                    PixelDiTActionMoT.ACTION26_GLOBAL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "state_dict_policy": (
                    "drop_action_tail_replace_bridges_with_global_native_kv_v1"
                ),
                "source_method": FORMAL_METHOD_CONTRACT_V2,
                "target_method": FORMAL_METHOD_CONTRACT_V7,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "source_step": 10000,
                "allowed_contract_differences": frozenset(
                    {
                        "action.norm_eps_by_layer",
                        "action.num_layers",
                        "bridge.adapter_count",
                        "bridge.adapters",
                        "bridge.bridge_semantics_version",
                        "bridge.bridge_source_schedule",
                        "bridge.layer_map",
                        "bridge.native_kv_adapter_count",
                        "bridge.native_kv_adapters",
                        "bridge.topology_version",
                        "kflow.architecture_version",
                        "method_contract",
                    }
                ),
                "atomic_paths": frozenset(
                    {
                        "action.norm_eps_by_layer",
                        "bridge.adapters",
                        "bridge.bridge_source_schedule",
                        "bridge.layer_map",
                        "bridge.native_kv_adapters",
                        "method_contract",
                    }
                ),
            },
            "v7_world_v2_to_all_native_kv_mot_action30_v1": {
                "source_topology": PixelDiTActionMoT.ACTION30_TOPOLOGY_VERSION,
                "target_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "state_dict_policy": (
                    "replace_all_bridges_with_world_native_kv_v1"
                ),
                "source_method": FORMAL_METHOD_CONTRACT_V2,
                "target_method": FORMAL_METHOD_CONTRACT_V6,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "source_step": 10000,
                "allowed_contract_differences": frozenset(
                    {
                        "bridge.adapter_count",
                        "bridge.adapters",
                        "bridge.bridge_semantics_version",
                        "bridge.bridge_source_schedule",
                        "bridge.native_kv_adapter_count",
                        "bridge.native_kv_adapters",
                        "bridge.topology_version",
                        "kflow.architecture_version",
                        "method_contract",
                    }
                ),
                "atomic_paths": frozenset(
                    {
                        "bridge.adapters",
                        "bridge.bridge_source_schedule",
                        "bridge.native_kv_adapters",
                        "method_contract",
                    }
                ),
            },
            "v7_nores_world_v1_to_all_native_kv_mot_action30_v1": {
                "source_topology": PixelDiTActionMoT.ACTION30_TOPOLOGY_VERSION,
                "target_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "state_dict_policy": (
                    "replace_all_bridges_with_world_native_kv_v1"
                ),
                "source_method": FORMAL_METHOD_CONTRACT_V2,
                "target_method": FORMAL_METHOD_CONTRACT_V6,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "source_step": 10000,
                "allowed_contract_differences": frozenset(
                    {
                        "bridge.adapter_count",
                        "bridge.adapters",
                        "bridge.bridge_semantics_version",
                        "bridge.bridge_source_schedule",
                        "bridge.native_kv_adapter_count",
                        "bridge.native_kv_adapters",
                        "bridge.topology_version",
                        "kflow.architecture_version",
                        "method_contract",
                    }
                ),
                "atomic_paths": frozenset(
                    {
                        "bridge.adapters",
                        "bridge.bridge_source_schedule",
                        "bridge.native_kv_adapters",
                        "method_contract",
                    }
                ),
            },
            "v7_h32s1_world_v1_to_all_native_kv_mot_action30_v1": {
                "source_topology": PixelDiTActionMoT.ACTION30_TOPOLOGY_VERSION,
                "target_topology": (
                    PixelDiTActionMoT.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
                ),
                "state_dict_policy": (
                    "replace_all_bridges_with_world_native_kv_v1"
                ),
                "source_method": FORMAL_METHOD_CONTRACT_V2,
                "target_method": FORMAL_METHOD_CONTRACT_V6,
                "source_architecture": (
                    FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION
                ),
                "target_architecture": (
                    FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
                ),
                "source_step": 10000,
                "allowed_contract_differences": frozenset(
                    {
                        "bridge.adapter_count",
                        "bridge.adapters",
                        "bridge.bridge_semantics_version",
                        "bridge.bridge_source_schedule",
                        "bridge.native_kv_adapter_count",
                        "bridge.native_kv_adapters",
                        "bridge.topology_version",
                        "kflow.architecture_version",
                        "method_contract",
                    }
                ),
                "atomic_paths": frozenset(
                    {
                        "bridge.adapters",
                        "bridge.bridge_source_schedule",
                        "bridge.native_kv_adapters",
                        "method_contract",
                    }
                ),
            },
        }
        if not isinstance(transition_id, str) or transition_id not in transition_specs:
            raise ValueError(
                "Unsupported fresh-stage topology transition: "
                f"{transition_id!r}."
            )
        spec = transition_specs[transition_id]
        expected_inputs = {
            "source_topology": spec["source_topology"],
            "target_topology": spec["target_topology"],
            "state_dict_policy": spec["state_dict_policy"],
        }
        actual_inputs = {
            "source_topology": source_topology,
            "target_topology": target_topology,
            "state_dict_policy": state_dict_policy,
        }
        if actual_inputs != expected_inputs:
            raise ValueError(
                "Fresh-stage transition policy differs from its allowlisted "
                f"identity: expected={expected_inputs}, got={actual_inputs}."
            )
        if self.dit.topology_version != spec["target_topology"]:
            raise ValueError(
                "Fresh-stage transition target topology differs from the model"
            )

        action14_transition = spec["state_dict_policy"] in {
            ACTION14_RANDOM_INITIALIZATION,
            ACTION14_WORLD_TRANSFER_INITIALIZATION,
        }
        source_checkpoint_identity = None
        if action14_transition:
            distributed = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            )
            rank = torch.distributed.get_rank() if distributed else 0
            identity_objects: list[Any] = [
                checkpoint_file_identity(path) if rank == 0 else None
            ]
            if distributed:
                torch.distributed.broadcast_object_list(
                    identity_objects,
                    src=0,
                )
            source_checkpoint_identity = identity_objects[0]
            if not isinstance(source_checkpoint_identity, Mapping):
                raise TypeError(
                    "Rank-zero checkpoint identity broadcast did not produce "
                    "a mapping."
                )

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise TypeError("Checkpoint payload must be a dictionary.")
        required = {"format", "training_contract", "model_state", "step"}
        allowed = required | {"optimizer", "initialization", "optimizer_policy"}
        missing_top = sorted(required - set(payload))
        unexpected_top = sorted(set(payload) - allowed)
        if missing_top or unexpected_top:
            raise ValueError(
                "Checkpoint top-level key mismatch for transition: "
                f"missing={missing_top}, unexpected={unexpected_top}."
            )
        if payload["format"] != self.CHECKPOINT_FORMAT:
            raise ValueError(
                f"Checkpoint is not {self.CHECKPOINT_FORMAT!r}: {path}."
            )
        if not isinstance(payload["model_state"], Mapping):
            raise TypeError("Checkpoint `model_state` must be a mapping.")
        expected_source_step = spec.get("source_step")
        if (
            expected_source_step is not None
            and payload["step"] != expected_source_step
        ):
            raise ValueError(
                "Fresh-stage source checkpoint step mismatch: expected "
                f"{expected_source_step}, got {payload['step']}."
            )

        source_contract = self._resume_training_contract(
            payload["training_contract"]
        )
        target_contract = self._resume_training_contract(
            self._checkpoint_contract()
        )
        if source_contract.get("method_contract") != spec["source_method"]:
            raise ValueError("Fresh-stage source method contract is not allowlisted")
        if target_contract.get("method_contract") != spec["target_method"]:
            raise ValueError("Fresh-stage target method contract is not allowlisted")
        if (
            source_contract.get("kflow", {}).get("architecture_version")
            != spec["source_architecture"]
            or target_contract.get("kflow", {}).get("architecture_version")
            != spec["target_architecture"]
        ):
            raise ValueError(
                "Fresh-stage KFlow architecture transition is not allowlisted"
            )

        if transition_id in {
            "v7_h8s4_a30_allnative_weighted_to_highnoise405505_uniform_v1",
            "v7_h8s4_a30_pitcompact_weighted_to_highnoise405505_uniform_v1",
            "v7_h8s4_nores_l24_a30_allnative_weighted_to_highnoise405505_uniform_v1",
        }:
            expected_source_train_scheduler = {
                "class": "WanContinuousFlowMatchScheduler",
                "num_train_timesteps": 1000,
                "shift": 5.0,
                "eps": 1.0e-10,
            }
            expected_target_train_scheduler = {
                "class": (
                    "WanHighNoise405505UniformWeightFlowMatchScheduler"
                ),
                "num_train_timesteps": 1000,
                "shift": 5.0,
                "eps": 1.0e-10,
                "training_policy": {
                    "schema_version": (
                        "wan_high_noise_405505_uniform_weight_v1"
                    ),
                    "continuous_density": (
                        "base_shifted_uniform_conditioned_per_interval_v1"
                    ),
                    "sigma_boundary": 0.95,
                    "low_interval": "[0,0.95)",
                    "high_interval": "[0.95,1)",
                    "endpoint_sigma": 1.0,
                    "low_probability": 0.4,
                    "high_probability": 0.55,
                    "endpoint_probability": 0.05,
                    "timestep_weighting": "uniform_one_v1",
                },
            }
            source_schedulers = source_contract.get("schedulers")
            target_schedulers = target_contract.get("schedulers")
            source_loss = source_contract.get("loss")
            target_loss = target_contract.get("loss")
            if not all(
                isinstance(value, Mapping)
                for value in (
                    source_schedulers,
                    target_schedulers,
                    source_loss,
                    target_loss,
                )
            ):
                raise ValueError(
                    "H8-S4 fresh-objective scheduler/loss contract is invalid"
                )
            if any(
                source_schedulers.get(name)
                != expected_source_train_scheduler
                for name in ("train_world", "train_action")
            ):
                raise ValueError(
                    "H8-S4 fresh-objective source training scheduler contract "
                    "is not exact"
                )
            if any(
                target_schedulers.get(name)
                != expected_target_train_scheduler
                for name in ("train_world", "train_action")
            ):
                raise ValueError(
                    "H8-S4 fresh-objective target training scheduler contract "
                    "is not exact"
                )
            if (
                source_loss.get("scheduler_weighting")
                != "per_sample_then_batch_mean"
                or target_loss.get("scheduler_weighting")
                != "uniform_one_per_sample_then_batch_mean_v1"
            ):
                raise ValueError(
                    "H8-S4 fresh-objective scheduler weighting contract is "
                    "not exact"
                )

        if transition_id == "v7_world_v2_to_global_only_action26_v1":
            source_action = source_contract.get("action")
            target_action = target_contract.get("action")
            source_bridge = source_contract.get("bridge")
            target_bridge = target_contract.get("bridge")
            if not all(
                isinstance(value, Mapping)
                for value in (
                    source_action,
                    target_action,
                    source_bridge,
                    target_bridge,
                )
            ):
                raise ValueError("Fresh-stage source/target action contract is invalid")
            if (
                source_action.get("num_layers") != 30
                or target_action.get("num_layers") != 26
                or source_action.get("norm_eps_by_layer", [])[:26]
                != target_action.get("norm_eps_by_layer")
                or len(source_action.get("norm_eps_by_layer", [])) != 30
            ):
                raise ValueError("Action30-to-26 layer contract is not an exact prefix")
            if (
                source_bridge.get("adapter_count") != 30
                or target_bridge.get("adapter_count") != 26
                or source_bridge.get("adapters", [])[:26]
                != target_bridge.get("adapters")
                or len(source_bridge.get("adapters", [])) != 30
                or source_bridge.get("bridge_source_schedule", [])[:26]
                != target_bridge.get("bridge_source_schedule")
                or source_bridge.get("layer_map", [])[:26]
                != target_bridge.get("layer_map")
            ):
                raise ValueError("Action30-to-26 bridge contract is not an exact prefix")

        if (
            transition_id
            == "v7_world_v2_to_global_native_kv_mot_action26_v1"
        ):
            source_action = source_contract.get("action")
            target_action = target_contract.get("action")
            source_bridge = source_contract.get("bridge")
            target_bridge = target_contract.get("bridge")
            if not all(
                isinstance(value, Mapping)
                for value in (
                    source_action,
                    target_action,
                    source_bridge,
                    target_bridge,
                )
            ):
                raise ValueError(
                    "Global-native A26 transition action/bridge contract is invalid"
                )
            source_norms = source_action.get("norm_eps_by_layer", [])
            target_norms = target_action.get("norm_eps_by_layer", [])
            expected_source_schedule = [
                *[["global", index] for index in range(26)],
                *[
                    ["final_global_concat_pit_compact", index]
                    for index in range(4)
                ],
            ]
            expected_target_schedule = [
                ["global_native_kv_mixed_with_action_self_kv", index]
                for index in range(26)
            ]
            if (
                source_action.get("num_layers") != 30
                or target_action.get("num_layers") != 26
                or len(source_norms) != 30
                or source_norms[:26] != target_norms
            ):
                raise ValueError(
                    "Global-native A26 transition requires an exact Action30-to-26 contract"
                )
            if (
                source_bridge.get("adapter_count") != 30
                or len(source_bridge.get("adapters", [])) != 30
                or "native_kv_adapter_count" in source_bridge
                or target_bridge.get("adapter_count") != 0
                or target_bridge.get("adapters") != []
                or target_bridge.get("native_kv_adapter_count") != 26
                or len(target_bridge.get("native_kv_adapters", [])) != 26
                or source_bridge.get("bridge_source_schedule")
                != expected_source_schedule
                or target_bridge.get("bridge_source_schedule")
                != expected_target_schedule
                or source_bridge.get("layer_map", [])[:26]
                != target_bridge.get("layer_map")
                or len(source_bridge.get("layer_map", [])) != 30
            ):
                raise ValueError(
                    "Global-native A26 transition bridge contract is not exact"
                )

        differences = self._contract_difference_paths(
            source_contract,
            target_contract,
            atomic_paths=spec["atomic_paths"],
        )
        if differences != spec["allowed_contract_differences"]:
            raise ValueError(
                "Fresh-stage training-contract differences exceed or miss the "
                "exact allowlist: expected="
                f"{sorted(spec['allowed_contract_differences'])}, got="
                f"{sorted(differences)}."
            )

        source_state = dict(payload["model_state"])
        dropped_keys: list[str] = []
        target_initialized_keys: list[str] = []
        imported_keys: list[str] = []
        action_initialization_receipt = None
        target_state = self.state_dict()
        if action14_transition:
            if not isinstance(self.dit, PixelDiTAlignedActionMoT):
                raise TypeError(
                    "Action14 transition requires PixelDiTAlignedActionMoT."
                )
            frozen_prefixes = (
                "dit.world_expert.",
                "dit.world_text_projection.",
            )
            source_frozen = {
                key: value
                for key, value in source_state.items()
                if key.startswith(frozen_prefixes)
            }
            target_frozen = {
                key: value
                for key, value in target_state.items()
                if key.startswith(frozen_prefixes)
            }
            self._require_transition_tensor_compatibility(
                source_frozen,
                target_frozen,
                name="v8_action14_frozen_world_state",
            )
            imported_keys = sorted(source_frozen)
            initialized_state = dict(target_state)
            initialized_state.update(source_frozen)
            self.load_state_dict(initialized_state, strict=True)
            assert source_checkpoint_identity is not None
            distributed = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            )
            rank = torch.distributed.get_rank() if distributed else 0
            action_initialization_receipt = initialize_pixeldit_aligned_action(
                self.action_expert,
                source_payload=payload,
                source_identity=source_checkpoint_identity,
                mode=spec["state_dict_policy"],
                audit_content=rank == 0,
            )
            receipt_keys = action_initialization_receipt.get("keys", [])
            if not isinstance(receipt_keys, list):
                raise TypeError("Action initialization receipt keys must be a list.")
            target_initialized_keys = sorted(
                "dit.action_expert." + str(item["target_key"])
                for item in receipt_keys
                if isinstance(item, Mapping) and item.get("source_key") is not None
            )
            source_state = dict(self.state_dict())
        elif spec["state_dict_policy"] == "drop_action_tail_v1":
            allowed_prefixes = tuple(
                prefix
                for index in range(26, 30)
                for prefix in (
                    f"dit.action_expert.blocks.{index}.",
                    f"dit.world_to_action_adapters.{index}.",
                )
            )
            dropped_keys = sorted(
                key for key in source_state if key.startswith(allowed_prefixes)
            )
            if len(dropped_keys) != 144:
                raise ValueError(
                    "Action30-to-26 transition must drop exactly 144 tail tensors, "
                    f"got {len(dropped_keys)}."
                )
            source_state = {
                key: value
                for key, value in source_state.items()
                if key not in set(dropped_keys)
            }
        elif (
            spec["state_dict_policy"]
            == "drop_action_tail_replace_bridges_with_global_native_kv_v1"
        ):
            legacy_bridge_suffixes = {
                "world_norm.weight",
                "world_norm.bias",
                "action_norm.weight",
                "action_norm.bias",
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "out_proj.weight",
                "residual_gate",
            }
            action_block_suffixes = {
                "modulation",
                "self_attn.q.weight",
                "self_attn.q.bias",
                "self_attn.k.weight",
                "self_attn.k.bias",
                "self_attn.v.weight",
                "self_attn.v.bias",
                "self_attn.o.weight",
                "self_attn.o.bias",
                "self_attn.norm_q.weight",
                "self_attn.norm_k.weight",
                "cross_attn.q.weight",
                "cross_attn.q.bias",
                "cross_attn.k.weight",
                "cross_attn.k.bias",
                "cross_attn.v.weight",
                "cross_attn.v.bias",
                "cross_attn.o.weight",
                "cross_attn.o.bias",
                "cross_attn.norm_q.weight",
                "cross_attn.norm_k.weight",
                "norm3.weight",
                "norm3.bias",
                "ffn.0.weight",
                "ffn.0.bias",
                "ffn.2.weight",
                "ffn.2.bias",
            }
            expected_legacy_bridge_keys = {
                f"dit.world_to_action_adapters.{index}.{suffix}"
                for index in range(30)
                for suffix in legacy_bridge_suffixes
            }
            expected_action_tail_keys = {
                f"dit.action_expert.blocks.{index}.{suffix}"
                for index in range(26, 30)
                for suffix in action_block_suffixes
            }
            expected_dropped_keys = (
                expected_legacy_bridge_keys | expected_action_tail_keys
            )
            dropped_keys = sorted(
                key
                for key in source_state
                if key.startswith("dit.world_to_action_adapters.")
                or key.startswith(
                    tuple(
                        f"dit.action_expert.blocks.{index}."
                        for index in range(26, 30)
                    )
                )
            )
            if set(dropped_keys) != expected_dropped_keys:
                raise ValueError(
                    "V2-to-V7 A26 source drop set mismatch: missing="
                    f"{sorted(expected_dropped_keys - set(dropped_keys))}, "
                    "unexpected="
                    f"{sorted(set(dropped_keys) - expected_dropped_keys)}."
                )
            source_shared_state = {
                key: value
                for key, value in source_state.items()
                if key not in expected_dropped_keys
            }
            expected_target_initialized_keys = {
                f"dit.world_native_kv_adapters.{index}.{projection}.weight"
                for index in range(26)
                for projection in ("k_proj", "v_proj")
            }
            target_initialized_keys = sorted(
                key
                for key in target_state
                if key.startswith("dit.world_native_kv_adapters.")
            )
            if set(target_initialized_keys) != expected_target_initialized_keys:
                raise ValueError(
                    "V7 A26 target native-K/V tensor set mismatch: missing="
                    f"{sorted(expected_target_initialized_keys - set(target_initialized_keys))}, "
                    "unexpected="
                    f"{sorted(set(target_initialized_keys) - expected_target_initialized_keys)}."
                )
            target_shared_state = {
                key: value
                for key, value in target_state.items()
                if key not in expected_target_initialized_keys
            }
            self._require_transition_tensor_compatibility(
                source_shared_state,
                target_shared_state,
                name="fresh_stage_a30_prefix_shared_model_state",
            )
            imported_keys = sorted(source_shared_state)
            source_state = dict(target_state)
            source_state.update(source_shared_state)
        elif (
            spec["state_dict_policy"]
            == "replace_all_bridges_with_world_native_kv_v1"
        ):
            legacy_bridge_suffixes = {
                "world_norm.weight",
                "world_norm.bias",
                "action_norm.weight",
                "action_norm.bias",
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "out_proj.weight",
                "residual_gate",
            }
            expected_dropped_keys = {
                f"dit.world_to_action_adapters.{index}.{suffix}"
                for index in range(30)
                for suffix in legacy_bridge_suffixes
            }
            dropped_keys = sorted(
                key
                for key in source_state
                if key.startswith("dit.world_to_action_adapters.")
            )
            if set(dropped_keys) != expected_dropped_keys:
                raise ValueError(
                    "V2-to-V6 transition source bridge tensor set mismatch: "
                    f"missing={sorted(expected_dropped_keys - set(dropped_keys))}, "
                    f"unexpected={sorted(set(dropped_keys) - expected_dropped_keys)}."
                )
            source_shared_state = {
                key: value
                for key, value in source_state.items()
                if key not in expected_dropped_keys
            }
            expected_target_initialized_keys = {
                f"dit.world_native_kv_adapters.{index}.{projection}.weight"
                for index in range(30)
                for projection in ("k_proj", "v_proj")
            }
            target_initialized_keys = sorted(
                key
                for key in target_state
                if key.startswith("dit.world_native_kv_adapters.")
            )
            if set(target_initialized_keys) != expected_target_initialized_keys:
                raise ValueError(
                    "V2-to-V6 target native-K/V tensor set mismatch: "
                    "missing="
                    f"{sorted(expected_target_initialized_keys - set(target_initialized_keys))}, "
                    "unexpected="
                    f"{sorted(set(target_initialized_keys) - expected_target_initialized_keys)}."
                )
            target_shared_state = {
                key: value
                for key, value in target_state.items()
                if key not in expected_target_initialized_keys
            }
            self._require_transition_tensor_compatibility(
                source_shared_state,
                target_shared_state,
                name="fresh_stage_shared_model_state",
            )
            source_state = dict(target_state)
            source_state.update(source_shared_state)

        target_state = self.state_dict()
        self._require_transition_tensor_compatibility(
            source_state,
            target_state,
            name="fresh_stage_model_state",
        )
        self._validate_module_state_dict(
            self,
            source_state,
            name="fresh_stage_model_state",
        )
        self.load_state_dict(source_state, strict=True)
        report = {
            "transition_id": transition_id,
            "source_step": payload["step"],
            "source_topology": spec["source_topology"],
            "target_topology": spec["target_topology"],
            "source_architecture_version": spec["source_architecture"],
            "target_architecture_version": spec["target_architecture"],
            "state_dict_policy": spec["state_dict_policy"],
            "source_tensor_count": len(payload["model_state"]),
            "target_tensor_count": len(target_state),
            "dropped_tensor_count": len(dropped_keys),
            "dropped_tensor_keys": dropped_keys,
            "target_initialized_tensor_count": len(target_initialized_keys),
            "target_initialized_tensor_keys": target_initialized_keys,
            "imported_tensor_count": len(imported_keys),
            "imported_tensor_keys": imported_keys,
            "contract_difference_paths": sorted(differences),
        }
        if action_initialization_receipt is not None:
            report["action_initialization"] = dict(
                action_initialization_receipt
            )
        return report

    def _checkpoint_contract(self) -> dict[str, Any]:
        world = self.world_expert
        action = self.action_expert
        adapters = self.dit.world_to_action_adapters
        native_adapters = self.dit.world_native_kv_adapters
        tail_native_adapters = self.dit.pit_native_kv_adapters
        scheduler_contracts = {
            "train_world": self._scheduler_contract(self.train_world_scheduler),
            "infer_world": self._scheduler_contract(self.infer_world_scheduler),
            "train_action": self._scheduler_contract(self.train_action_scheduler),
            "infer_action": self._scheduler_contract(self.infer_action_scheduler),
        }
        scheduler_weighting = self._scheduler_weighting_contract(
            scheduler_contracts["train_world"],
            scheduler_contracts["train_action"],
        )
        world_fields = (
            "in_channels",
            "clean_channels",
            "rgb_channels",
            "flow_channels",
            "out_channels",
            "k",
            "hidden_size",
            "num_groups",
            "patch_depth",
            "pixel_depth",
            "patch_size",
            "pixel_hidden_size",
            "num_classes",
            "use_pixel_abs_pos",
            "pit_adaln_post_modulation",
            "use_gradient_checkpointing",
        )
        if bool(world.require_full_rank_output):
            world_fields += ("require_full_rank_output",)
        if (
            world.momap_output_head_type
            in {
                MOMAP_LOCAL_GLOBAL_FUSION_OUTPUT_HEAD,
                MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD,
                "hybrid_allxyz_local_global_fusion_v1",
            }
        ):
            world_fields += (
                "momap_output_head_type",
                "momap_output_hidden_size",
                "momap_output_head_initialization",
            )
        if (
            world.momap_output_head_type
            == "hybrid_allxyz_local_global_fusion_v1"
        ):
            world_fields += (
                "STREAM_SEMANTICS_VERSION",
                "ATTENTION_SEMANTICS_VERSION",
                "CARRIER_SEMANTICS_VERSION",
            )
        if (
            world.momap_output_head_type
            == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
        ):
            world_fields += ("raw_flow_noise_floor",)
        adapter_specs = [
            {
                "world_hidden_size": int(adapter.world_hidden_size),
                "action_hidden_size": int(adapter.action_hidden_size),
                "mot_dim": int(adapter.mot_dim),
                "mot_num_heads": int(adapter.mot_num_heads),
                "head_dim": int(adapter.head_dim),
                "norm_eps": float(adapter.norm_eps),
                "residual_gate_shape": list(adapter.residual_gate.shape),
            }
            for adapter in adapters
        ]
        native_adapter_specs = [
            {
                "world_attn_dim": int(adapter.world_attn_dim),
                "action_attn_dim": int(adapter.action_attn_dim),
                "k_proj_shape": list(adapter.k_proj.weight.shape),
                "v_proj_shape": list(adapter.v_proj.weight.shape),
            }
            for adapter in native_adapters
        ]
        tail_native_adapter_specs = [
            {
                "world_attn_dim": int(adapter.world_attn_dim),
                "action_attn_dim": int(adapter.action_attn_dim),
                "k_proj_shape": list(adapter.k_proj.weight.shape),
                "v_proj_shape": list(adapter.v_proj.weight.shape),
            }
            for adapter in tail_native_adapters
        ]
        if isinstance(action, PixelDiTTemporalActionDiT):
            action_contract = {
                "architecture": "pixeldit_temporal32_action_v1",
                "hidden_dim": int(action.hidden_dim),
                "action_dim": int(action.action_dim),
                "action_horizon": int(action.action_horizon),
                "text_dim": int(action.text_dim),
                "num_heads": int(action.num_heads),
                "attn_head_dim": int(action.attn_head_dim),
                "num_layers": len(action.blocks),
                "mlp_ratio": float(action.mlp_ratio),
                "action_encoder_weight_shape": list(
                    action.action_encoder.weight.shape
                ),
                "action_decoder_weight_shape": list(
                    action.action_decoder.weight.shape
                ),
                "context_projection_weight_shape": list(
                    action.context_projection.weight.shape
                ),
                "temporal_rope_shape": list(action.temporal_rope.shape),
                "temporal_rope_persistent": False,
                "token_semantics": action.TOKEN_SEMANTICS_VERSION,
                "position_encoding": action.POSITION_ENCODING_VERSION,
                "native_kv_fusion": action.native_kv_fusion_version,
                "action_qk_rope": "temporal_1d",
                "norm_eps_by_layer": [
                    float(block.norm1.variance_epsilon)
                    for block in action.blocks
                ],
                "use_gradient_checkpointing": bool(
                    action.use_gradient_checkpointing
                ),
            }
        elif isinstance(action, PixelDiTAlignedActionDiT):
            action_contract = {
                "architecture": "pixeldit_aligned_joint14_action_v1",
                "hidden_dim": int(action.hidden_dim),
                "action_dim": int(action.action_dim),
                "action_horizon": int(action.action_horizon),
                "text_dim": int(action.text_dim),
                "num_heads": int(action.num_heads),
                "attn_head_dim": int(action.attn_head_dim),
                "num_layers": len(action.blocks),
                "mlp_ratio": float(action.mlp_ratio),
                "action_encoder_weight_shape": list(
                    action.action_encoder.weight.shape
                ),
                "action_decoder_weight_shape": list(
                    action.action_decoder.weight.shape
                ),
                "context_projection_weight_shape": list(
                    action.context_projection.weight.shape
                ),
                "fixed_position_shape": list(action.joint_pos_embed.shape),
                "fixed_position_persistent": False,
                "token_semantics": action.TOKEN_SEMANTICS_VERSION,
                "position_encoding": action.POSITION_ENCODING_VERSION,
                "native_kv_fusion": action.native_kv_fusion_version,
                "action_qk_rope": "none",
                "norm_eps_by_layer": [
                    float(block.norm1.variance_epsilon)
                    for block in action.blocks
                ],
                "use_gradient_checkpointing": bool(
                    action.use_gradient_checkpointing
                ),
            }
        else:
            action_contract = {
                "hidden_dim": int(action.hidden_dim),
                "action_dim": int(action.action_dim),
                "ffn_dim": int(action.ffn_dim),
                "text_dim": int(action.text_dim),
                "freq_dim": int(action.freq_dim),
                "num_heads": int(action.num_heads),
                "attn_head_dim": int(action.attn_head_dim),
                "num_layers": len(action.blocks),
                "norm_eps_by_layer": [
                    float(block.norm1.eps) for block in action.blocks
                ],
                "use_gradient_checkpointing": bool(
                    action.use_gradient_checkpointing
                ),
            }
        native_kv_fusion_transition = None
        if self.method_contract == FORMAL_METHOD_CONTRACT_V9:
            transition_provider = getattr(
                self.dit,
                "native_kv_fusion_transition_contract",
                None,
            )
            if not callable(transition_provider):
                raise TypeError(
                    "V9 aligned Action wrapper must expose its native-K/V "
                    "transition contract."
                )
            native_kv_fusion_transition = transition_provider(require_bound=True)
        return {
            "format": self.CHECKPOINT_FORMAT,
            "generation_mode": self.GENERATION_MODE,
            "method_contract": self._resolve_method_contract(
                self.method_contract
            ),
            **(
                {
                    "native_kv_fusion_transition": native_kv_fusion_transition,
                }
                if native_kv_fusion_transition is not None
                else {}
            ),
            "numeric_precision_policy": dict(
                FORMAL_NUMERIC_PRECISION_POLICY_V1
            ),
            "kflow": self.kflow_contract.as_dict(),
            "normalization": self._normalization_math_contract(
                self.normalization_contract.as_dict()
            ),
            **(
                {
                    "flow_statistics_provenance": copy.deepcopy(
                        self.flow_statistics_provenance_contract
                    )
                }
                if self.flow_statistics_provenance_contract is not None
                else {}
            ),
            "action_horizon": self.action_horizon,
            "action_normalization": self._normalization_math_contract(
                self.action_normalization_contract.as_dict()
            ),
            "proprio_normalization": (
                None
                if self.proprio_normalization_contract is None
                else self._normalization_math_contract(
                    self.proprio_normalization_contract.as_dict()
                )
            ),
            "action_proprio_semantics": self.batch_adapter.semantic_contract(),
            "batch_adapter": type(self.batch_adapter).__name__,
            "text_dim": self.text_dim,
            "proprio_dim": self.proprio_dim,
            "world": {name: getattr(world, name) for name in world_fields},
            "action": action_contract,
            "bridge": {
                "topology_version": self.dit.topology_version,
                "interaction_mode": self.dit.interaction_mode,
                "bridge_semantics_version": self.dit.bridge_semantics_version,
                "bridge_source_schedule": [
                    [str(source_kind), int(source_index)]
                    for source_kind, source_index in self.dit.bridge_source_schedule
                ],
                "layer_map": list(self.dit.layer_map),
                "adapter_count": len(adapters),
                "mot_dim": int(self.dit.mot_dim),
                "mot_num_heads": int(self.dit.mot_num_heads),
                "use_gradient_checkpointing": bool(self.dit.use_gradient_checkpointing),
                "adapters": adapter_specs,
                **(
                    {
                        "native_kv_adapter_count": len(native_adapters),
                        "native_kv_adapters": native_adapter_specs,
                    }
                    if native_adapters
                    else {}
                ),
                **(
                    {
                        "tail_native_kv_adapter_count": len(tail_native_adapters),
                        "tail_native_kv_adapters": tail_native_adapter_specs,
                    }
                    if tail_native_adapters
                    else {}
                ),
            },
            "schedulers": scheduler_contracts,
            "timestep_coupling": self.timestep_coupling,
            "loss": {
                "lambda_momap": self.loss_lambda_momap,
                "lambda_action": self.loss_lambda_action,
                **(
                    {
                        "flow_supervision_scope": self.flow_supervision_scope,
                    }
                    if self.flow_supervision_scope
                    != FLOW_SUPERVISION_ALL_MOSAIC
                    else {}
                ),
                "momap_reduction": "xyz_mean_then_valid_pixel_then_horizon_then_sample",
                "action_reduction": "action_dim_mean_then_nonpad_token_then_sample",
                "scheduler_weighting": scheduler_weighting,
            },
        }

    def configure_controlled_native_kv_count_correction_anneal(
        self,
        *,
        source_global_step: int,
        transition_steps: int,
        transition_curve: str,
        transition_id: str,
        source_native_kv_fusion: str,
        target_native_kv_fusion: str,
    ) -> dict[str, Any]:
        """Bind the V9 schedule before strict full-state resume validation."""

        configure = getattr(
            self.dit,
            "configure_controlled_native_kv_count_correction_anneal",
            None,
        )
        if not callable(configure):
            raise TypeError(
                "Current model does not support controlled native-K/V "
                "count-correction annealing."
            )
        return configure(
            source_global_step=source_global_step,
            transition_steps=transition_steps,
            transition_curve=transition_curve,
            transition_id=transition_id,
            source_native_kv_fusion=source_native_kv_fusion,
            target_native_kv_fusion=target_native_kv_fusion,
        )

    def set_attention_runtime_step(self, global_step: Optional[int]) -> None:
        """Inject the pre-forward successful optimizer count into V9 attention."""

        setter = getattr(self.dit, "set_attention_runtime_step", None)
        if not callable(setter):
            if global_step is not None:
                raise TypeError(
                    "Current model does not support an attention runtime step."
                )
            return
        setter(global_step)

    def resume_math_contract(self) -> dict[str, Any]:
        """Return the finite math semantics that full-state resume must bind."""

        return copy.deepcopy(self._checkpoint_contract())

    def forward(
        self,
        sample: Mapping[str, Any],
        tiled: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return self.training_loss(sample, tiled=tiled)


__all__ = [
    "FLOW_SUPERVISION_ALL_MOSAIC",
    "FLOW_SUPERVISION_HEAD_TILE_ONLY",
    "FLOW_SUPERVISION_SCOPES",
    "FORMAL_NUMERIC_PRECISION_POLICY_V1",
    "FastWAMPixelDiT",
    "apply_flow_supervision_scope",
]
