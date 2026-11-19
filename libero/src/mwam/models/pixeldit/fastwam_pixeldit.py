from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn as nn


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
from .kflow import ROBOTWIN_FASTWAM_HEAD_SIZE, ROBOTWIN_FASTWAM_MOSAIC_SIZE, KFlowContract, MoMapBatchAdapter
from .method_contract import FORMAL_METHOD_CONTRACT_V1, resolve_formal_method_contract
from .momap_flow import MoMapFlowMatchingObjective
from .normalization import FixedFeatureNormalizationContract
from .pixeldit_action_mot import PixelDiTActionMoT


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
            "future_horizon_valid",
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
        horizon_valid = sample.get("future_horizon_valid")
        if horizon_valid is None:
            horizon_valid = torch.ones_like(frame_offsets_long, dtype=torch.bool)
        elif (not isinstance(horizon_valid, torch.Tensor) or horizon_valid.dtype != torch.bool
              or horizon_valid.shape != frame_offsets.shape):
            raise ValueError("`future_horizon_valid` must be bool [B,K].")
        horizon_valid = horizon_valid.to(frame_offsets.device)
        if bool((frame_offsets_long[horizon_valid] <= 0).any().item()):
            raise ValueError("Valid `frame_offsets` must be positive.")
        for offsets_row, valid_row in zip(frame_offsets_long, horizon_valid):
            present_offsets = offsets_row[valid_row]
            if bool((present_offsets[1:] <= present_offsets[:-1]).any().item()):
                raise ValueError("Valid `frame_offsets` must be increasing.")
        if bool((frame_offsets_long[~horizon_valid] != -1).any().item()):
            raise ValueError("Missing future horizons require frame_offsets=-1.")

        dt_seconds = self._required_tensor(sample, "momap_dt_seconds")
        if dt_seconds.shape != (batch_size, self.kflow_contract.k):
            raise ValueError(
                "`momap_dt_seconds` must be [B,K], got " f"{tuple(dt_seconds.shape)}."
            )
        self._require_float_finite(dt_seconds, name="momap_dt_seconds")
        horizon_valid = horizon_valid.to(dt_seconds.device)
        if bool((dt_seconds[horizon_valid] <= 0).any().item()):
            raise ValueError("Valid `momap_dt_seconds` must be strictly positive.")
        if bool((dt_seconds[~horizon_valid] != 0).any().item()):
            raise ValueError("Missing future horizons require momap_dt_seconds=0.")
        expected = torch.tensor(
            expected_horizons,
            device=dt_seconds.device,
            dtype=dt_seconds.dtype,
        ).unsqueeze(0)
        if bool(((dt_seconds + 1.0e-7 < expected) & horizon_valid).any().item()):
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
        expected_lateness = torch.where(horizon_valid, dt_seconds - expected, 0.)
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
        horizon_valid = adapted.get("future_horizon_valid")
        if horizon_valid is not None:
            missing = ~horizon_valid.to(flow_valid.device)
            if bool((flow_valid & missing[:, :, None, None, None]).any().item()):
                raise ValueError("Missing future horizons cannot contain valid flow supervision.")
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
            if horizon_valid is not None and bool((
                    value & ~horizon_valid.to(value.device)[:, :, None, None, None]).any().item()):
                raise ValueError(f"Missing future horizons cannot contain valid {key}.")
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
        if horizon_valid is not None:
            prepared["future_horizon_valid"] = horizon_valid.to(device=self.device)
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
        if bool(action_is_pad.any().item()) and not getattr(self,"ALLOW_ACTION_PADDING",False):
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


    def forward(
        self,
        sample: Mapping[str, Any],
        tiled: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return self.training_loss(sample, tiled=tiled)
