"""Training wrapper for the versioned K4 All-view-XYZ method."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .fastwam_pixeldit import FastWAMPixelDiT
from .method_contract import (
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2,
)
from .momap_flow import MoMapVelocityLoss


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
        allow_empty_views = getattr(self, "ALLOW_EMPTY_FLOW_VIEWS", False)
        if not allow_empty_views and not bool(head_nonempty.any(dim=1).all().item()):
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
        if allow_empty_views:
            # Fully occluded source views can have no eligible future tracks.
            # Keep equal head/wrist-group weights when both exist; otherwise
            # train only the available group. Never fabricate valid flow labels.
            group_count = head_nonempty.to(error.dtype) + wrist_nonempty.to(error.dtype)
            per_horizon = (
                head_value * head_nonempty.to(error.dtype)
                + wrist_value * wrist_nonempty.to(error.dtype)
            ) / group_count.clamp_min(1.0)
            horizon_nonempty = head_nonempty | wrist_nonempty
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
        if allow_empty_views:
            self._last_hybrid_world_loss_metrics.update({
                "valid_ratio/samples_without_head": float((~head_nonempty.any(dim=1)).float().mean().item()),
                "valid_ratio/samples_without_flow": float((~horizon_nonempty.any(dim=1)).float().mean().item()),
            })
        valid_ratio_by_horizon = valid.float().mean(dim=(0, 2, 3, 4))
        valid_horizon_ratio = horizon_nonempty.float().mean(dim=0)
        return MoMapVelocityLoss(
            raw=per_sample.mean(),
            weighted=(per_sample * weight).mean(),
            per_sample=per_sample,
            valid_ratio_by_horizon=valid_ratio_by_horizon,
            valid_horizon_ratio=valid_horizon_ratio,
        )

    @staticmethod
    def _append_hybrid_metrics(result):
        return result

    def compute_world_loss(self, *args: Any, **kwargs: Any):
        result = super().compute_world_loss(*args, **kwargs)
        result[1].update(getattr(self, "_last_hybrid_world_loss_metrics", {}))
        return result

    def compute_joint_loss(self, *args: Any, **kwargs: Any):
        result = super().compute_joint_loss(*args, **kwargs)
        result[1].update(getattr(self, "_last_hybrid_world_loss_metrics", {}))
        return result
