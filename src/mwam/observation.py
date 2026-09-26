"""Target-free, already assembled RoboTwin observation tensors.

This boundary deliberately does not require a simulator or a training Dataset.
Depth-to-XYZ, segmentation and text embedding happen upstream.
"""
from pathlib import Path

import numpy as np
import torch


TENSOR_KEYS = frozenset({
    "source_rgb", "source_seg_condition", "source_xyz_ref",
    "source_condition_valid", "context", "context_mask", "proprio",
})
METADATA = {"proprio_mode": "last_command_drive_target", "proprio_normalization": "none"}


def validate_observation(observation):
    if set(observation) != TENSOR_KEYS | METADATA.keys():
        raise ValueError("Observation must contain only current-frame tensors and proprio metadata")
    for key, value in METADATA.items():
        if observation[key] != value:
            raise ValueError(f"Incorrect {key}")
    for key in TENSOR_KEYS:
        value = observation[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a Tensor")
        if key in {"context_mask", "source_condition_valid"}:
            if value.dtype != torch.bool:
                raise ValueError(f"{key} must have boolean dtype")
        elif not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{key} must contain finite floating point values")
    rgb = observation["source_rgb"]
    if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, 384, 320) or rgb.shape[0] < 1:
        raise ValueError("Expected mosaic source_rgb [B,3,384,320]")
    batch = rgb.shape[0]
    for key in ("source_seg_condition", "source_xyz_ref"):
        if observation[key].shape != rgb.shape:
            raise ValueError(f"{key} must match source_rgb shape")
    for key in ("source_rgb", "source_seg_condition"):
        if observation[key].min() < -1 or observation[key].max() > 1:
            raise ValueError(f"{key} must be in [-1,1]")
    if observation["source_condition_valid"].shape != (batch, 1, 384, 320):
        raise ValueError("Expected source_condition_valid [B,1,384,320]")
    context = observation["context"]
    if context.ndim != 3 or context.shape[0] != batch or context.shape[2] != 4096:
        raise ValueError("Expected context [B,L,4096]")
    if observation["context_mask"].shape != context.shape[:2]:
        raise ValueError("context_mask must be [B,L]")
    if not observation["context_mask"].any(dim=1).all():
        raise ValueError("Every sample needs at least one valid text token")
    if observation["proprio"].shape not in {(batch, 14), (batch, 1, 14)}:
        raise ValueError("Expected current proprio [B,14] or [B,1,14]")
    return observation


def load_observation(path: str | Path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != TENSOR_KEYS:
            raise ValueError("NPZ must contain exactly the seven current-observation tensors; no GT")
        result = {key: torch.from_numpy(data[key].copy()) for key in TENSOR_KEYS}
    result.update(METADATA)
    return validate_observation(result)


def synthetic_observation(seed=0):
    """Interface smoke input ONLY. Not a real observation or a quality benchmark."""
    rng = torch.Generator().manual_seed(seed)
    result = {
        "source_rgb": torch.rand((1, 3, 384, 320), generator=rng) * 2 - 1,
        "source_seg_condition": torch.zeros(1, 3, 384, 320),
        "source_xyz_ref": torch.rand((1, 3, 384, 320), generator=rng) - 0.5,
        "source_condition_valid": torch.ones(1, 1, 384, 320, dtype=torch.bool),
        "context": torch.randn((1, 16, 4096), generator=rng),
        "context_mask": torch.ones(1, 16, dtype=torch.bool),
        "proprio": torch.zeros(1, 1, 14),
        **METADATA,
    }
    return validate_observation(result)
