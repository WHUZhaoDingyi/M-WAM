"""Explicit world-conditioning contracts for the PixelDiT MoMap expert.

The production contract follows MoMaps' source-observation conditioning:

* normalized RGB (3 channels);
* deterministic instance-colour segmentation (3 channels);
* canonical XYZ in one fixed source-time reference frame (3 channels);
* packed noisy future offsets (``3*K`` channels) are appended by PixelDiT.

Raw depth is deliberately absent.  It is an upstream oracle measurement used
to construct canonical XYZ and may be retained for auditing, but it is not a
second, redundant model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WORLD_CONDITIONING_V2 = "rgb_seg_canonical_xyz_v2"
CANONICAL_REFERENCE_FRAME = "head_camera@source_t0"


@dataclass(frozen=True)
class WorldConditioningContract:
    """Checkpointable channel semantics for the PixelDiT global input."""

    mode: str = WORLD_CONDITIONING_V2
    reference_frame: str = CANONICAL_REFERENCE_FRAME
    rgb_channels: int = 3
    segmentation_channels: int = 3
    canonical_xyz_channels: int = 3
    raw_depth_channels: int = 0
    segmentation_encoding: str = "deterministic_instance_rgb_signed_with_zero_background"
    condition_valid_semantics: str = "source_depth_and_geometry_valid"

    def __post_init__(self) -> None:
        if self.mode != WORLD_CONDITIONING_V2:
            raise ValueError(
                "PixelDiT only supports the production RGB + segmentation + "
                f"canonical-XYZ conditioning mode {WORLD_CONDITIONING_V2!r}, "
                f"got {self.mode!r}."
            )
        if self.raw_depth_channels != 0:
            raise ValueError(
                "Raw depth must not be a direct PixelDiT channel; derive canonical "
                "XYZ upstream instead."
            )
        expected = {
            "reference_frame": CANONICAL_REFERENCE_FRAME,
            "rgb_channels": 3,
            "segmentation_channels": 3,
            "canonical_xyz_channels": 3,
            "raw_depth_channels": 0,
        }
        actual = {name: getattr(self, name) for name in expected}
        if actual != expected:
            raise ValueError(
                f"Conditioning mode {self.mode!r} has inconsistent channel/frame "
                f"semantics: actual={actual}, expected={expected}."
            )
        if not self.segmentation_encoding:
            raise ValueError("`segmentation_encoding` must be non-empty.")
        if not self.condition_valid_semantics:
            raise ValueError("`condition_valid_semantics` must be non-empty.")

    @property
    def condition_channels(self) -> int:
        return (
            self.rgb_channels
            + self.segmentation_channels
            + self.canonical_xyz_channels
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reference_frame": self.reference_frame,
            "rgb_channels": self.rgb_channels,
            "segmentation_channels": self.segmentation_channels,
            "canonical_xyz_channels": self.canonical_xyz_channels,
            "raw_depth_channels": self.raw_depth_channels,
            "condition_channels": self.condition_channels,
            "segmentation_encoding": self.segmentation_encoding,
            "condition_valid_semantics": self.condition_valid_semantics,
        }


__all__ = [
    "CANONICAL_REFERENCE_FRAME",
    "WORLD_CONDITIONING_V2",
    "WorldConditioningContract",
]
