"""All-camera XYZ normalization in canonical head@source_t0 metres.
Head and pooled wrists each have three mean/std pairs across four horizons.
Source observation XYZ keeps the original affine statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch

from .conditioning import CANONICAL_REFERENCE_FRAME
from .kflow import KFLOW_CHANNEL_ORDER, KFLOW_UNIT_HEAD_METER_WRIST_TILE_PIXEL


HYBRID_HEAD_WRIST_NORMALIZATION_SCHEMA_VERSION = (
    "robotwin_momap_k4_all_xyz_normalization_v1"
)
HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_SCHEMA_VERSION = (
    "robotwin_momap_k4_all_xyz_normalization_v2"
)
HYBRID_HEAD_WRIST_NORMALIZATION_MODE = (
    "fixed_source_xyz_affine_head_dxyz_scale_wrist_dxyz_meter_scale_v1"
)
HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_MODE = (
    "fixed_source_xyz_affine_head_dxyz_pooled_zscore_"
    "wrist_dxyz_pooled_meter_zscore_v1"
)


def _finite_tuple(
    values: tuple[float, ...] | list[float],
    *,
    name: str,
    length: int,
    positive: bool = False,
) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or len(values) != length:
        raise ValueError(f"`{name}` must contain exactly {length} values.")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"`{name}` contains NaN or Inf.")
    if positive and any(value <= 0.0 for value in result):
        raise ValueError(f"Every `{name}` value must be positive.")
    return result


@dataclass(frozen=True)
class HybridHeadWristNormalizationContract:
    """Normalize source XYZ and all-camera dXYZ with separate pooled statistics."""

    statistics_id: str
    future_offsets: tuple[int, ...]
    source_mean_xyz_meters: tuple[float, ...]
    source_std_xyz_meters: tuple[float, ...]
    # v1 only: K-major Head and Wrist XYZ [4,3] RMS scales.
    head_scale_channels_meters: tuple[float, ...] = ()
    wrist_scale_channels_meters: tuple[float, ...] = ()
    # v2 only: one pooled z-score statistic per physical component.
    head_flow_mean_xyz_meters: tuple[float, ...] = ()
    head_flow_std_xyz_meters: tuple[float, ...] = ()
    wrist_flow_mean_xyz_meters: tuple[float, ...] = ()
    wrist_flow_std_xyz_meters: tuple[float, ...] = ()
    head_height: int = 256
    full_height: int = 384
    full_width: int = 320
    reference_frame: str = CANONICAL_REFERENCE_FRAME
    source_unit: str = KFLOW_UNIT_HEAD_METER_WRIST_TILE_PIXEL
    channel_order: str = KFLOW_CHANNEL_ORDER
    schema_version: str = HYBRID_HEAD_WRIST_NORMALIZATION_SCHEMA_VERSION
    mode: str = HYBRID_HEAD_WRIST_NORMALIZATION_MODE
    digest: str | None = None

    def __post_init__(self) -> None:
        is_legacy_per_offset = (
            self.schema_version == HYBRID_HEAD_WRIST_NORMALIZATION_SCHEMA_VERSION
            and self.mode == HYBRID_HEAD_WRIST_NORMALIZATION_MODE
        )
        is_pooled_zscore = (
            self.schema_version == HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_SCHEMA_VERSION
            and self.mode == HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_MODE
        )
        if not is_legacy_per_offset and not is_pooled_zscore:
            raise ValueError(
                "Unsupported hybrid normalization schema/mode pair "
                f"{self.schema_version!r}/{self.mode!r}."
            )
        if not str(self.statistics_id).strip():
            raise ValueError("`statistics_id` must be non-empty.")
        offsets = tuple(int(value) for value in self.future_offsets)
        if offsets != (8, 16, 24, 32):
            raise ValueError(
                "Hybrid K4 normalization requires future_offsets=(8,16,24,32), "
                f"got {offsets}."
            )
        object.__setattr__(self, "future_offsets", offsets)
        object.__setattr__(
            self,
            "source_mean_xyz_meters",
            _finite_tuple(
                self.source_mean_xyz_meters,
                name="source_mean_xyz_meters",
                length=3,
            ),
        )
        object.__setattr__(
            self,
            "source_std_xyz_meters",
            _finite_tuple(
                self.source_std_xyz_meters,
                name="source_std_xyz_meters",
                length=3,
                positive=True,
            ),
        )
        if is_legacy_per_offset:
            object.__setattr__(
                self,
                "head_scale_channels_meters",
                _finite_tuple(
                    self.head_scale_channels_meters,
                    name="head_scale_channels_meters",
                    length=12,
                    positive=True,
                ),
            )
            object.__setattr__(
                self,
                "wrist_scale_channels_meters",
                _finite_tuple(
                    self.wrist_scale_channels_meters,
                    name="wrist_scale_channels_meters",
                    length=12,
                    positive=True,
                ),
            )
            for name in (
                "head_flow_mean_xyz_meters",
                "head_flow_std_xyz_meters",
                "wrist_flow_mean_xyz_meters",
                "wrist_flow_std_xyz_meters",
            ):
                object.__setattr__(
                    self,
                    name,
                    _finite_tuple(getattr(self, name), name=name, length=0),
                )
        else:
            for name in (
                "head_scale_channels_meters",
                "wrist_scale_channels_meters",
            ):
                object.__setattr__(
                    self,
                    name,
                    _finite_tuple(getattr(self, name), name=name, length=0),
                )
            object.__setattr__(
                self,
                "head_flow_mean_xyz_meters",
                _finite_tuple(
                    self.head_flow_mean_xyz_meters,
                    name="head_flow_mean_xyz_meters",
                    length=3,
                ),
            )
            object.__setattr__(
                self,
                "head_flow_std_xyz_meters",
                _finite_tuple(
                    self.head_flow_std_xyz_meters,
                    name="head_flow_std_xyz_meters",
                    length=3,
                    positive=True,
                ),
            )
            object.__setattr__(
                self,
                "wrist_flow_mean_xyz_meters",
                _finite_tuple(
                    self.wrist_flow_mean_xyz_meters,
                    name="wrist_flow_mean_xyz_meters",
                    length=3,
                ),
            )
            object.__setattr__(
                self,
                "wrist_flow_std_xyz_meters",
                _finite_tuple(
                    self.wrist_flow_std_xyz_meters,
                    name="wrist_flow_std_xyz_meters",
                    length=3,
                    positive=True,
                ),
            )
        if (self.head_height, self.full_height, self.full_width) != (256, 384, 320):
            raise ValueError(
                "Hybrid normalization requires the fixed FastWAM 384x320 mosaic "
                "with a 256-row Head tile."
            )
        if self.reference_frame != CANONICAL_REFERENCE_FRAME:
            raise ValueError(
                f"reference_frame must be {CANONICAL_REFERENCE_FRAME!r}."
            )
        if self.source_unit != KFLOW_UNIT_HEAD_METER_WRIST_TILE_PIXEL:
            raise ValueError("All-camera source_unit must bind canonical metric XYZ.")
        if self.channel_order != KFLOW_CHANNEL_ORDER:
            raise ValueError(f"Unsupported channel_order {self.channel_order!r}.")
        computed = self.recompute_digest()
        if self.digest is not None and self.digest != computed:
            raise ValueError(
                f"Hybrid normalization digest mismatch: {self.digest!r} != {computed!r}."
            )
        object.__setattr__(self, "digest", computed)

    @property
    def k(self) -> int:
        return 4

    @property
    def flow_channels(self) -> int:
        return 12

    @property
    def uses_pooled_flow_zscore(self) -> bool:
        return (
            self.schema_version == HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_SCHEMA_VERSION
        )

    @property
    def uses_pooled_flow_scales(self) -> bool:
        """Compatibility alias for the earlier unshipped pooled branch."""

        return self.uses_pooled_flow_zscore

    def _digest_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "statistics_id": self.statistics_id,
            "future_offsets": list(self.future_offsets),
            "source_mean_xyz_meters": list(self.source_mean_xyz_meters),
            "source_std_xyz_meters": list(self.source_std_xyz_meters),
            "head_height": self.head_height,
            "full_height": self.full_height,
            "full_width": self.full_width,
            "reference_frame": self.reference_frame,
            "source_unit": self.source_unit,
            "channel_order": self.channel_order,
            "head_flow_unit": "meter",
            "wrist_flow_unit": "meter",
        }
        if self.uses_pooled_flow_zscore:
            payload.update(
                {
                    "head_flow_mean_xyz_meters": list(
                        self.head_flow_mean_xyz_meters
                    ),
                    "head_flow_std_xyz_meters": list(
                        self.head_flow_std_xyz_meters
                    ),
                    "wrist_flow_mean_xyz_meters": list(
                        self.wrist_flow_mean_xyz_meters
                    ),
                    "wrist_flow_std_xyz_meters": list(
                        self.wrist_flow_std_xyz_meters
                    ),
                    "centering": (
                        "source_xyz_mean_and_flow_pooled_mean_std_zscore_v1"
                    ),
                }
            )
        else:
            # Preserve v1's exact serialized digest payload.
            payload.update(
                {
                    "head_scale_channels_meters": list(
                        self.head_scale_channels_meters
                    ),
                    "wrist_scale_channels_meters": list(
                        self.wrist_scale_channels_meters
                    ),
                    "centering": (
                        "source_xyz_mean_only_flow_zero_preserving_scale_only"
                    ),
                }
            )
        return payload

    def recompute_digest(self) -> str:
        payload = json.dumps(
            self._digest_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**self._digest_payload(), "digest": self.digest}

    @staticmethod
    def _require_spatial(value: torch.Tensor, *, channels: int, name: str) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"`{name}` must be a torch.Tensor.")
        if value.ndim != 4 or value.shape[1] != channels:
            raise ValueError(
                f"`{name}` must be [B,{channels},H,W], got {tuple(value.shape)}."
            )
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"`{name}` must be finite floating point.")

    def normalize_xyz(
        self,
        xyz_meters: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._require_spatial(xyz_meters, channels=3, name="xyz_meters")
        mean = torch.as_tensor(
            self.source_mean_xyz_meters,
            device=xyz_meters.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        std = torch.as_tensor(
            self.source_std_xyz_meters,
            device=xyz_meters.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        result = (xyz_meters.float() - mean) / std
        if valid is not None:
            if valid.dtype != torch.bool or valid.shape not in {
                xyz_meters.shape,
                (xyz_meters.shape[0], 1, xyz_meters.shape[2], xyz_meters.shape[3]),
            }:
                raise ValueError("`valid` must be bool [B,1,H,W] or [B,3,H,W].")
            result = torch.where(valid.to(device=result.device), result, 0.0)
        return result

    def denormalize_xyz(self, xyz_normalized: torch.Tensor) -> torch.Tensor:
        self._require_spatial(xyz_normalized, channels=3, name="xyz_normalized")
        mean = torch.as_tensor(
            self.source_mean_xyz_meters,
            device=xyz_normalized.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        std = torch.as_tensor(
            self.source_std_xyz_meters,
            device=xyz_normalized.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        return xyz_normalized.float() * std + mean

    def _flow_mean_and_std(
        self,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_spatial(value, channels=12, name="packed_hybrid_flow")
        if tuple(value.shape[-2:]) != (self.full_height, self.full_width):
            raise ValueError(
                "Hybrid flow must use the fixed 384x320 mosaic, got "
                f"{tuple(value.shape[-2:])}."
            )
        batch = int(value.shape[0])
        mean = torch.zeros(
            (1, 4, 3, self.full_height, 1),
            device=value.device,
            dtype=torch.float32,
        )
        std = torch.ones(
            (1, 4, 3, self.full_height, 1),
            device=value.device,
            dtype=torch.float32,
        )
        if self.uses_pooled_flow_zscore:
            head_mean = torch.as_tensor(
                self.head_flow_mean_xyz_meters,
                device=value.device,
                dtype=torch.float32,
            ).view(1, 1, 3, 1, 1).expand(1, 4, 3, 1, 1)
            head_std = torch.as_tensor(
                self.head_flow_std_xyz_meters,
                device=value.device,
                dtype=torch.float32,
            ).view(1, 1, 3, 1, 1).expand(1, 4, 3, 1, 1)
            wrist_mean_uv = torch.as_tensor(
                self.wrist_flow_mean_xyz_meters,
                device=value.device,
                dtype=torch.float32,
            ).view(1, 1, 3, 1, 1).expand(1, 4, 3, 1, 1)
            wrist_std_uv = torch.as_tensor(
                self.wrist_flow_std_xyz_meters,
                device=value.device,
                dtype=torch.float32,
            ).view(1, 1, 3, 1, 1).expand(1, 4, 3, 1, 1)
        else:
            head_mean = torch.zeros(
                (1, 4, 3, 1, 1), device=value.device, dtype=torch.float32
            )
            head_std = torch.as_tensor(
                self.head_scale_channels_meters,
                device=value.device,
                dtype=torch.float32,
            ).view(1, 4, 3, 1, 1)
            wrist_mean_uv = torch.zeros(
                (1, 4, 3, 1, 1), device=value.device, dtype=torch.float32
            )
            wrist_std_uv = torch.as_tensor(
                self.wrist_scale_channels_meters,
                device=value.device,
                dtype=torch.float32,
            ).view(1, 4, 3, 1, 1)
        mean[:, :, :, : self.head_height] = head_mean
        std[:, :, :, : self.head_height] = head_std
        mean[:, :, :, self.head_height :] = wrist_mean_uv
        std[:, :, :, self.head_height :] = wrist_std_uv
        # All three displacement axes are normalized and supervised.
        return (
            mean.expand(batch, -1, -1, -1, self.full_width).reshape_as(value),
            std.expand(batch, -1, -1, -1, self.full_width).reshape_as(value),
        )

    def _flow_scale(self, value: torch.Tensor) -> torch.Tensor:
        """Compatibility helper for callers that only need the denominator."""

        _, std = self._flow_mean_and_std(value)
        return std

    def normalize_offset(self, offset: torch.Tensor) -> torch.Tensor:
        mean, std = self._flow_mean_and_std(offset)
        return (offset.float() - mean) / std

    def denormalize_offset(self, normalized: torch.Tensor) -> torch.Tensor:
        mean, std = self._flow_mean_and_std(normalized)
        return normalized.float() * std + mean


__all__ = [
    "HYBRID_HEAD_WRIST_NORMALIZATION_MODE",
    "HYBRID_HEAD_WRIST_NORMALIZATION_SCHEMA_VERSION",
    "HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_MODE",
    "HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_SCHEMA_VERSION",
    "HybridHeadWristNormalizationContract",
]
