"""Versioned train-split normalization for Action and Proprio features."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch


FIXED_FEATURE_DIM = 14
FIXED_FEATURE_NORMALIZATION_SCHEMA_VERSION = "fastwam_fixed_feature_zscore_v1"
FIXED_FEATURE_NORMALIZATION_MODE = "fixed_feature_zscore"


def _finite_tuple(values: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(
            f"`{name}` must be a list or tuple of built-in int/float values."
        )
    if not values:
        raise ValueError(f"`{name}` must not be empty.")
    for index, value in enumerate(values):
        if type(value) not in (int, float):
            raise TypeError(
                f"`{name}[{index}]` must be a built-in int or float, got "
                f"{type(value).__name__}."
            )
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"`{name}` contains NaN or Inf.")
    return result


@dataclass(frozen=True)
class FixedFeatureNormalizationContract:
    """Versioned fixed train-split z-score statistics for one 14-D feature.

    Action and proprioception intentionally use separate instances. The
    semantic/source fields bind statistics to the raw Dataset meaning, while
    ``digest`` makes every value independently reproducible in checkpoints.
    """

    feature_kind: str
    semantic_kind: str
    source_normalization: str
    statistics_id: str
    mean: tuple[float, ...]
    std: tuple[float, ...]
    schema_version: str = FIXED_FEATURE_NORMALIZATION_SCHEMA_VERSION
    mode: str = FIXED_FEATURE_NORMALIZATION_MODE
    digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != FIXED_FEATURE_NORMALIZATION_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported fixed-feature normalization schema_version "
                f"{self.schema_version!r}."
            )
        if self.mode != FIXED_FEATURE_NORMALIZATION_MODE:
            raise ValueError(
                f"Unsupported fixed-feature normalization mode {self.mode!r}."
            )
        if self.feature_kind not in {"action", "proprio"}:
            raise ValueError(
                "`feature_kind` must be 'action' or 'proprio', got "
                f"{self.feature_kind!r}."
            )
        if not str(self.semantic_kind).strip():
            raise ValueError("`semantic_kind` must be non-empty.")
        if self.source_normalization != "none":
            raise ValueError(
                "Fixed feature statistics require raw source_normalization="
                f"'none', got {self.source_normalization!r}; per-sample "
                "normalization is forbidden."
            )
        if not str(self.statistics_id).strip():
            raise ValueError("`statistics_id` must be non-empty.")
        object.__setattr__(self, "mean", _finite_tuple(self.mean, name="mean"))
        object.__setattr__(self, "std", _finite_tuple(self.std, name="std"))
        if len(self.mean) != FIXED_FEATURE_DIM or len(self.std) != FIXED_FEATURE_DIM:
            raise ValueError(
                "Fixed action/proprio normalization must contain exactly "
                f"{FIXED_FEATURE_DIM} features, got mean={len(self.mean)}, "
                f"std={len(self.std)}."
            )
        if any(value <= 0.0 for value in self.std):
            raise ValueError("Every fixed feature normalization std must be positive.")

        computed = self.recompute_digest()
        if self.digest is not None and self.digest != computed:
            raise ValueError(
                "Fixed feature normalization digest mismatch: supplied "
                f"{self.digest!r}, recomputed {computed!r}."
            )
        object.__setattr__(self, "digest", computed)

    @property
    def feature_dim(self) -> int:
        return FIXED_FEATURE_DIM

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "feature_dim": self.feature_dim,
            "feature_kind": self.feature_kind,
            "semantic_kind": self.semantic_kind,
            "source_normalization": self.source_normalization,
            "statistics_id": self.statistics_id,
            "mean": list(self.mean),
            "std": list(self.std),
        }

    def recompute_digest(self) -> str:
        encoded = json.dumps(
            self._digest_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._digest_payload(),
            "digest": self.digest,
        }

    def _stats_for(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{self.feature_kind} normalization input must be a torch.Tensor."
            )
        if value.ndim < 1 or value.shape[-1] != self.feature_dim:
            raise ValueError(
                f"{self.feature_kind} must end in feature dim {self.feature_dim}, "
                f"got {tuple(value.shape)}."
            )
        if not value.is_floating_point():
            raise TypeError(
                f"{self.feature_kind} must be floating point, got {value.dtype}."
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{self.feature_kind} contains NaN or Inf.")
        broadcast_shape = (1,) * (value.ndim - 1) + (self.feature_dim,)
        mean = torch.tensor(
            self.mean,
            device=value.device,
            dtype=value.dtype,
        ).view(broadcast_shape)
        std = torch.tensor(
            self.std,
            device=value.device,
            dtype=value.dtype,
        ).view(broadcast_shape)
        return mean, std

    def normalize(self, value_raw: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats_for(value_raw)
        return (value_raw - mean) / std

    def denormalize(self, value_normalized: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats_for(value_normalized)
        return value_normalized * std + mean


