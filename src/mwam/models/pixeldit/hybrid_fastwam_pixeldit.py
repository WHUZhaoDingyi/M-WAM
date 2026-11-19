"""Training wrapper for the versioned K4 All-view-XYZ method."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .fastwam_pixeldit import FastWAMPixelDiT
from .hybrid_normalization import HybridHeadWristNormalizationContract
from .kflow import (
    KFLOW_TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV,
    KFLOW_UNIT_HEAD_METER_WRIST_TILE_PIXEL,
)
from .method_contract import (
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2,
)
from .momap_flow import MoMapVelocityLoss
from .pixeldit_aligned_action import (
    PixelDiTAlignedActionDiT,
    PixelDiTTemporalActionDiT,
)


HYBRID_METHOD_CONTRACTS = (
    FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2,
)


def _masked_region_mean(
    squared_error: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[B,K]`` mean and nonempty flags for one component/spatial region."""

    if squared_error.shape != valid.shape or squared_error.ndim != 5:
        raise ValueError("Hybrid squared error and valid mask must match [B,K,C,H,W].")
    valid_float = valid.to(dtype=squared_error.dtype)
    count = valid_float.sum(dim=(2, 3, 4))
    nonempty = count > 0
    total = torch.where(valid, squared_error, 0.0).sum(dim=(2, 3, 4))
    return total / count.clamp_min(1.0), nonempty


class HybridFastWAMPixelDiT(FastWAMPixelDiT):
    """Keep the historical public solver while enforcing heterogeneous loss math."""

    CHECKPOINT_FORMAT = "fastwam_pixeldit_k4_hybrid_cw_v2"
    GENERATION_MODE = "all_xyz_pooled_zscore_to_action_flow_matching_v2"

    def _validate_contract_alignment(self) -> None:
        super()._validate_contract_alignment()
        if self.method_contract not in HYBRID_METHOD_CONTRACTS:
            raise ValueError("Hybrid wrapper requires the ACW or AC K4 method contract.")
        contract = self.kflow_contract
        if tuple(contract.future_offsets) != (8, 16, 24, 32) or contract.k != 4:
            raise ValueError("Hybrid wrapper requires K4 offsets [8,16,24,32].")
        if (
            contract.target_representation
            != KFLOW_TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV
            or contract.unit != KFLOW_UNIT_HEAD_METER_WRIST_TILE_PIXEL
            or contract.flow_channels != 12
        ):
            raise ValueError("Hybrid wrapper requires the packed All-view-dXYZ carrier.")
        if not isinstance(
            self.normalization_contract,
            HybridHeadWristNormalizationContract,
        ):
            raise TypeError("Hybrid wrapper requires HybridHeadWristNormalizationContract.")
        if self.normalization_contract.future_offsets != contract.future_offsets:
            raise ValueError("Hybrid normalization/KFlow offsets disagree.")

        world = self.world_expert
        expected_world = {
            "clean_channels": 9,
            "k": 4,
            "flow_channels": 12,
            "out_channels": 12,
            "hidden_size": 1152,
            "pixel_hidden_size": 16,
            "patch_depth": 26,
            "pixel_depth": 4,
            "patch_size": 16,
            "num_groups": 16,
        }
        actual_world = {name: int(getattr(world, name, -1)) for name in expected_world}
        if actual_world != expected_world:
            raise ValueError(
                f"Hybrid World architecture drift: {actual_world} != {expected_world}."
            )
        action = self.action_expert
        if not isinstance(action, PixelDiTAlignedActionDiT):
            raise TypeError("Hybrid Action must use a PixelDiT-aligned Action DiT.")
        expected_action = {
            "action_horizon": 32,
            "action_dim": 14,
            "hidden_size": 1152,
            "num_heads": 16,
            "num_layers": 30,
            "text_dim": self.text_dim,
        }
        actual_action = {name: int(getattr(action, name, -1)) for name in expected_action}
        if actual_action != expected_action:
            raise ValueError(
                f"Hybrid Action architecture drift: {actual_action} != {expected_action}."
            )
        if isinstance(action, PixelDiTTemporalActionDiT):
            expected_temporal = {
                "encoder_shape": (1152, 14),
                "decoder_shape": (14, 1152),
                "rope_shape": (32, 36),
                "token_semantics": "time32_per_step_action14_tokens_v1",
                "position_encoding": "temporal_1d_rope_index_0_31_v1",
            }
            actual_temporal = {
                "encoder_shape": tuple(action.action_encoder.weight.shape),
                "decoder_shape": tuple(action.action_decoder.weight.shape),
                "rope_shape": tuple(action.temporal_rope.shape),
                "token_semantics": action.TOKEN_SEMANTICS_VERSION,
                "position_encoding": action.POSITION_ENCODING_VERSION,
            }
            if actual_temporal != expected_temporal:
                raise ValueError(
                    "Hybrid temporal32 Action architecture drift: "
                    f"{actual_temporal} != {expected_temporal}."
                )
            if self.method_contract != FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2:
                raise ValueError("Temporal32 Hybrid Action requires its dedicated method contract.")
        elif tuple(action.action_encoder.weight.shape) != (1152, 32):
            raise ValueError("Hybrid joint14 Action must use fourteen 32-step trajectory tokens.")

        expected_topology = self.method_contract["topology"]
        if getattr(self.dit, "topology_version", None) != expected_topology:
            raise ValueError("Hybrid MoT topology and method contract disagree.")
        expected_scope = (
            "c_and_w"
            if self.method_contract
            in (
                FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW,
                FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2,
            )
            else "c_only"
        )
        observed_scope = getattr(self.dit, "action_world_visibility", None)
        if observed_scope != expected_scope:
            raise ValueError(
                f"Hybrid Action visibility must be {expected_scope!r}, got "
                f"{observed_scope!r}."
            )
        if int(getattr(self.dit, "world_depth", -1)) != 26:
            raise ValueError("Hybrid MoT requires 26 Global layers.")
        if int(getattr(self.dit, "pixel_depth", -1)) != 4:
            raise ValueError("Hybrid MoT requires four PiT layers.")
        if int(getattr(self.dit, "action_depth", -1)) != 30:
            raise ValueError("Hybrid MoT requires 30 Action layers.")
        for name in (
            "world_to_action_adapters",
            "world_native_kv_adapters",
            "pit_native_kv_adapters",
        ):
            modules = getattr(self.dit, name, None)
            if modules is None or len(modules) != 0:
                raise ValueError(f"Hybrid direct native K/V requires empty {name}.")

    def _prepare_training_sample(
        self,
        sample: Mapping[str, Any],
        *,
        include_action: bool = True,
    ) -> dict[str, Any]:
        prepared = super()._prepare_training_sample(
            sample,
            include_action=include_action,
        )
        if "flow_loss_valid" not in prepared:
            raise KeyError(
                "Hybrid training requires component-level `flow_loss_valid`; "
                "pixel-only flow_valid cannot disable Wrist Z or the native-200px gate."
            )
        return prepared

    def _compute_momap_velocity_loss(
        self,
        prediction: torch.Tensor,
        target_velocity: torch.Tensor,
        *,
        inputs: Mapping[str, Any],
        timestep_weight: torch.Tensor,
    ) -> MoMapVelocityLoss:
        self.momap_objective._validate_packed_flow(prediction, name="prediction")
        self.momap_objective._validate_packed_flow(
            target_velocity,
            name="target_velocity",
        )
        if prediction.shape != target_velocity.shape:
            raise ValueError("Hybrid prediction and velocity target shapes differ.")
        batch_size, _, height, width = prediction.shape
        if (height, width) != (256 + 128, 320):
            raise ValueError("Hybrid loss requires the fixed 384x320 mosaic.")
        valid = inputs["flow_loss_valid"]
        expected_valid = (batch_size, 4, 3, height, width)
        if not isinstance(valid, torch.Tensor) or tuple(valid.shape) != expected_valid:
            raise ValueError(f"flow_loss_valid must be bool {expected_valid}.")
        if valid.dtype != torch.bool:
            raise TypeError("flow_loss_valid must use torch.bool.")
        valid = valid.to(device=prediction.device)
        error = (
            prediction.float().reshape(batch_size, 4, 3, height, width)
            - target_velocity.float().reshape(batch_size, 4, 3, height, width)
        ).square()

        head_value, head_nonempty = _masked_region_mean(
            error[:, :, :, :256, :],
            valid[:, :, :, :256, :],
        )
        left_value, left_nonempty = _masked_region_mean(
            error[:, :, :, 256:, :160],
            valid[:, :, :, 256:, :160],
        )
        right_value, right_nonempty = _masked_region_mean(
            error[:, :, :, 256:, 160:],
            valid[:, :, :, 256:, 160:],
        )
        if not bool(head_nonempty.any(dim=1).all().item()):
            failed = (~head_nonempty.any(dim=1)).nonzero().flatten().tolist()
            raise ValueError(f"Every sample requires Head XYZ supervision; missing={failed}.")

        wrist_count = left_nonempty.to(error.dtype) + right_nonempty.to(error.dtype)
        wrist_value = (
            left_value * left_nonempty.to(error.dtype)
            + right_value * right_nonempty.to(error.dtype)
        ) / wrist_count.clamp_min(1.0)
        wrist_nonempty = wrist_count > 0
        per_horizon = torch.where(
            wrist_nonempty,
            0.5 * head_value + 0.5 * wrist_value,
            head_value,
        )
        horizon_nonempty = head_nonempty
        horizon_float = horizon_nonempty.to(error.dtype)
        per_sample = (per_horizon * horizon_float).sum(dim=1) / horizon_float.sum(
            dim=1
        ).clamp_min(1.0)

        weight = torch.as_tensor(
            timestep_weight,
            device=prediction.device,
            dtype=per_sample.dtype,
        )
        if weight.ndim == 0:
            weight = weight.expand(batch_size)
        if tuple(weight.shape) != (batch_size,) or not bool(torch.isfinite(weight).all()):
            raise ValueError("Hybrid timestep_weight must be finite [B].")
        if bool((weight < 0).any().item()):
            raise ValueError("Hybrid timestep_weight must be non-negative.")

        self._last_hybrid_world_loss_metrics = {
            "loss/momap_head_3d_raw": float(
                (
                    (head_value * head_nonempty.to(error.dtype)).sum()
                    / head_nonempty.sum().clamp_min(1)
                ).detach().item()
            ),
            "loss/momap_wrist_3d_raw": float(
                (
                    (wrist_value * wrist_nonempty.to(error.dtype)).sum()
                    / wrist_nonempty.sum().clamp_min(1)
                ).detach().item()
            ),
            "valid_ratio/head_3d": float(valid[:, :, :, :256].float().mean().item()),
            "valid_ratio/wrist_3d": float(valid[:, :, :, 256:].float().mean().item()),
        }
        valid_ratio_by_horizon = valid.float().mean(dim=(0, 2, 3, 4))
        valid_horizon_ratio = horizon_nonempty.float().mean(dim=0)
        return MoMapVelocityLoss(
            raw=per_sample.mean(),
            weighted=(per_sample * weight).mean(),
            per_sample=per_sample,
            valid_ratio_by_horizon=valid_ratio_by_horizon,
            valid_horizon_ratio=valid_horizon_ratio,
        )


    def compute_world_loss(self, *args: Any, **kwargs: Any):
        result = super().compute_world_loss(*args, **kwargs)
        result[1].update(getattr(self, "_last_hybrid_world_loss_metrics", {}))
        return result

    def compute_joint_loss(self, *args: Any, **kwargs: Any):
        result = super().compute_joint_loss(*args, **kwargs)
        result[1].update(getattr(self, "_last_hybrid_world_loss_metrics", {}))
        return result


__all__ = ["HybridFastWAMPixelDiT"]
