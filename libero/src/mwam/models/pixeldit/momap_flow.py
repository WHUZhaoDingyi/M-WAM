"""Official FastWAM flow-matching objective for packed K-horizon MoMap.

This module deliberately delegates the interpolation direction, target
velocity, timestep distribution, weighting, and Euler solver convention to
``WanContinuousFlowMatchScheduler``.  It only adds the K-flow tensor contract
and mask-aware per-horizon reduction required by MoMap supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from .kflow import unpack_kflow


@dataclass(frozen=True)
class MoMapFlowMatchingBatch:
    """One official FastWAM noising step for packed MoMap."""

    noised_flow: torch.Tensor
    target_velocity: torch.Tensor
    timestep: torch.Tensor
    noise: torch.Tensor
    timestep_weight: torch.Tensor


@dataclass(frozen=True)
class MoMapVelocityLoss:
    """Raw and scheduler-weighted MoMap loss plus audit statistics."""

    raw: torch.Tensor
    weighted: torch.Tensor
    per_sample: torch.Tensor
    valid_ratio_by_horizon: torch.Tensor
    valid_horizon_ratio: torch.Tensor


class MoMapFlowMatchingObjective:
    """K-flow adapter around FastWAM's canonical continuous FM scheduler."""

    def __init__(
        self,
        scheduler: WanContinuousFlowMatchScheduler,
        *,
        k: int,
    ) -> None:
        if not isinstance(scheduler, WanContinuousFlowMatchScheduler):
            raise TypeError(
                "`scheduler` must be WanContinuousFlowMatchScheduler, "
                f"got {type(scheduler).__name__}."
            )
        if isinstance(k, bool) or int(k) != k or int(k) <= 0:
            raise ValueError(f"`k` must be a positive integer, got {k!r}.")
        self.scheduler = scheduler
        self.k = int(k)

    @property
    def flow_channels(self) -> int:
        return 3 * self.k

    def _validate_packed_flow(
        self,
        flow: torch.Tensor,
        *,
        name: str,
    ) -> None:
        if not isinstance(flow, torch.Tensor):
            raise TypeError(f"`{name}` must be a torch.Tensor.")
        if flow.ndim != 4 or flow.shape[1] != self.flow_channels:
            raise ValueError(
                f"`{name}` must be [B,{self.flow_channels},H,W] for K={self.k}, "
                f"got {tuple(flow.shape)}."
            )
        if not flow.is_floating_point():
            raise TypeError(f"`{name}` must be floating point, got {flow.dtype}.")
        if not bool(torch.isfinite(flow).all().item()):
            raise ValueError(f"`{name}` contains NaN or Inf.")

    def prepare(
        self,
        clean_flow: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> MoMapFlowMatchingBatch:
        """Noise clean MoMap without consulting any target-derived valid mask."""

        self._validate_packed_flow(clean_flow, name="clean_flow")
        batch_size = int(clean_flow.shape[0])

        if noise is None:
            noise = torch.randn_like(clean_flow)
        else:
            self._validate_packed_flow(noise, name="noise")
            if noise.shape != clean_flow.shape:
                raise ValueError(
                    "`noise` shape must exactly match clean_flow, got "
                    f"{tuple(noise.shape)} vs {tuple(clean_flow.shape)}."
                )
            noise = noise.to(device=clean_flow.device, dtype=clean_flow.dtype)

        if timestep is None:
            timestep = self.scheduler.sample_training_t(
                batch_size=batch_size,
                device=clean_flow.device,
                dtype=torch.float32,
            )
        else:
            if not isinstance(timestep, torch.Tensor):
                raise TypeError("`timestep` must be a torch.Tensor when provided.")
            if timestep.ndim != 1 or timestep.shape[0] != batch_size:
                raise ValueError(
                    f"`timestep` must be [B]={batch_size}, got {tuple(timestep.shape)}."
                )
            timestep = timestep.to(
                device=clean_flow.device,
                dtype=torch.float32,
            )
        if not bool(torch.isfinite(timestep).all().item()):
            raise ValueError("`timestep` contains NaN or Inf.")
        max_t = float(self.scheduler.num_train_timesteps)
        if bool(((timestep < 0) | (timestep > max_t)).any().item()):
            raise ValueError(
                f"`timestep` must lie in [0,{max_t}], got {timestep.tolist()}."
            )

        noised = self.scheduler.add_noise(clean_flow, noise, timestep)
        target = self.scheduler.training_target(clean_flow, noise, timestep)
        weight = self.scheduler.training_weight(timestep).to(
            device=clean_flow.device,
            dtype=torch.float32,
        )
        if weight.ndim == 0:
            weight = weight.expand(batch_size)
        if weight.shape != (batch_size,):
            raise RuntimeError(
                "FastWAM scheduler returned an unexpected training weight shape: "
                f"{tuple(weight.shape)}."
            )
        return MoMapFlowMatchingBatch(
            noised_flow=noised,
            target_velocity=target,
            timestep=timestep,
            noise=noise,
            timestep_weight=weight,
        )

    def velocity_loss(
        self,
        prediction: torch.Tensor,
        target_velocity: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor],
        timestep_weight: torch.Tensor,
    ) -> MoMapVelocityLoss:
        """Reduce MSE per horizon, then per sample, then apply scheduler weight."""

        self._validate_packed_flow(prediction, name="prediction")
        self._validate_packed_flow(target_velocity, name="target_velocity")
        if prediction.shape != target_velocity.shape:
            raise ValueError(
                "Prediction/target velocity shapes must match, got "
                f"{tuple(prediction.shape)} vs {tuple(target_velocity.shape)}."
            )
        batch_size, _, height, width = prediction.shape

        if valid_mask is None:
            valid = torch.ones(
                (batch_size, self.k, height, width),
                device=prediction.device,
                dtype=torch.bool,
            )
        else:
            if not isinstance(valid_mask, torch.Tensor):
                raise TypeError("`valid_mask` must be a torch.Tensor when provided.")
            expected = (batch_size, self.k, 1, height, width)
            if tuple(valid_mask.shape) != expected:
                raise ValueError(
                    f"`valid_mask` must be [B,K,1,H,W]={expected}, "
                    f"got {tuple(valid_mask.shape)}."
                )
            if valid_mask.dtype != torch.bool:
                raise TypeError(
                    f"`valid_mask` must use torch.bool, got {valid_mask.dtype}."
                )
            valid = valid_mask[:, :, 0].to(device=prediction.device)

        pred = unpack_kflow(prediction.float(), k=self.k)
        target = unpack_kflow(target_velocity.float(), k=self.k)
        squared_xyz = (pred - target).square().mean(dim=2)
        valid_f = valid.to(dtype=squared_xyz.dtype)
        valid_count = valid_f.sum(dim=(-2, -1))
        horizon_is_valid = valid_count > 0
        if not bool(horizon_is_valid.any(dim=1).all().item()):
            invalid_samples = (
                ~horizon_is_valid.any(dim=1)
            ).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "Every sample must contain at least one supervised MoMap horizon; "
                f"all-invalid sample indices={invalid_samples}."
            )

        per_horizon = (
            torch.where(valid, squared_xyz, torch.zeros_like(squared_xyz)).sum(
                dim=(-2, -1)
            )
            / valid_count.clamp_min(1.0)
        )
        horizon_f = horizon_is_valid.to(dtype=per_horizon.dtype)
        per_sample = (per_horizon * horizon_f).sum(dim=1) / horizon_f.sum(
            dim=1
        ).clamp_min(1.0)

        if not isinstance(timestep_weight, torch.Tensor):
            raise TypeError("`timestep_weight` must be a torch.Tensor.")
        weight = timestep_weight.to(
            device=per_sample.device,
            dtype=per_sample.dtype,
        )
        if weight.ndim == 0:
            weight = weight.expand(batch_size)
        if weight.shape != (batch_size,):
            raise ValueError(
                f"`timestep_weight` must be [B]={batch_size}, got {tuple(weight.shape)}."
            )
        if not bool(torch.isfinite(weight).all().item()) or bool(
            (weight < 0).any().item()
        ):
            raise ValueError("`timestep_weight` must be finite and non-negative.")

        raw = per_sample.mean()
        weighted = (per_sample * weight).mean()
        valid_ratio_by_horizon = valid_f.mean(dim=(0, 2, 3))
        valid_horizon_ratio = horizon_f.mean(dim=0)
        return MoMapVelocityLoss(
            raw=raw,
            weighted=weighted,
            per_sample=per_sample,
            valid_ratio_by_horizon=valid_ratio_by_horizon,
            valid_horizon_ratio=valid_horizon_ratio,
        )


