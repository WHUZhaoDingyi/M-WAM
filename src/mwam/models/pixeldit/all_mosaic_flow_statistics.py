"""Formal pooled-horizon XYZ moments for the full FastWAM Flow mosaic."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .feature_statistics_artifact import (
    EXPLICIT_EPISODE_STATISTICS_QUALIFICATION,
    FORMAL_STATISTICS_QUALIFICATION,
)
from .geometry_normalization import (
    HYBRID_FLOW_POPULATION_ALL_VALID,
    HybridSourceP99FlowXYZMeanStdGeometryNormalizationContract,
)
from .head_flow_statistics import (
    FORMAL_TRAIN_SPLIT_STRATIFIED_CALIBRATION_QUALIFICATION,
    HEAD_TILE_FLOW_TENSOR_FIELDS,
    BoundHeadTileFlowXYZMeanStdStatisticsArtifact,
    HeadTileFlowXYZMeanStdStatisticsAccumulator,
    _plain_json,
    _sha256_hex,
)
from .kflow import ROBOTWIN_FASTWAM_MOSAIC_SIZE, ROBOTWIN_FASTWAM_VIEW_MODE


ALL_MOSAIC_FLOW_XYZ_STATISTICS_ARTIFACT_SCHEMA_VERSION = (
    "robotwin_momap_all_mosaic_flow_xyz_mean_std_statistics_v1"
)
ALL_MOSAIC_FLOW_XYZ_STATISTICS_REDUCTION = (
    "valid_only_population_xyz_mean_std_ddof_0_pooled_horizons"
)
ALL_MOSAIC_FLOW_XYZ_STATISTICS_POPULATION = (
    "fastwam_source_point_mosaic_384x320_all_valid_kflow_values"
)
ALL_MOSAIC_FLOW_XYZ_CALIBRATION_MODEL_VISIBLE_POPULATION = (
    "head/left/right source-point FastWAM mosaic 384x320; "
    "valid K1-K8 XYZ pooled"
)
ALL_MOSAIC_FLOW_TENSOR_FIELDS = HEAD_TILE_FLOW_TENSOR_FIELDS


class AllMosaicFlowXYZMeanStdStatisticsAccumulator(
    HeadTileFlowXYZMeanStdStatisticsAccumulator
):
    """Mergeable CPU-float64 XYZ moments over every valid mosaic Flow value."""

    def update(
        self,
        *,
        flow_meters: torch.Tensor,
        flow_valid: torch.Tensor,
    ) -> None:
        expected_spatial = ROBOTWIN_FASTWAM_MOSAIC_SIZE
        if (
            not isinstance(flow_meters, torch.Tensor)
            or flow_meters.ndim != 5
            or flow_meters.shape[1:3] != (self.k, 3)
            or tuple(flow_meters.shape[-2:]) != expected_spatial
            or not flow_meters.is_floating_point()
        ):
            raise ValueError(
                "flow_meters must be floating [B,K,3,384,320] with "
                f"K={self.k}, got {getattr(flow_meters, 'shape', None)}."
            )
        expected_valid = (
            flow_meters.shape[0],
            self.k,
            1,
            *expected_spatial,
        )
        if (
            not isinstance(flow_valid, torch.Tensor)
            or tuple(flow_valid.shape) != expected_valid
            or flow_valid.dtype != torch.bool
        ):
            raise ValueError(
                f"flow_valid must be bool {expected_valid}, got "
                f"{getattr(flow_valid, 'shape', None)}."
            )

        flow = flow_meters.detach().to(device="cpu", dtype=torch.float64)
        valid = flow_valid.detach().to(device="cpu").expand(-1, -1, 3, -1, -1)
        if bool(valid.any().item()) and not bool(torch.isfinite(flow[valid]).all()):
            raise ValueError("Valid all-mosaic Flow contains NaN or Inf.")
        safe = torch.where(valid, flow, torch.zeros_like(flow))
        self._count += valid.sum(dim=(0, 1, 3, 4), dtype=torch.int64)
        self._sum += safe.sum(dim=(0, 1, 3, 4))
        self._sum_square += safe.square().sum(dim=(0, 1, 3, 4))
        self._minimum = torch.minimum(
            self._minimum,
            torch.where(valid, flow, torch.full_like(flow, math.inf)).amin(
                dim=(0, 1, 3, 4)
            ),
        )
        self._maximum = torch.maximum(
            self._maximum,
            torch.where(valid, flow, torch.full_like(flow, -math.inf)).amax(
                dim=(0, 1, 3, 4)
            ),
        )
        self._masked_nonfinite_count += (
            (~valid) & (~torch.isfinite(flow))
        ).sum(dim=(0, 1, 3, 4), dtype=torch.int64)
        self._batch_count += 1

    def receipt(self) -> dict[str, Any]:
        minimum = [
            None if math.isinf(float(value)) else float(value)
            for value in self._minimum.tolist()
        ]
        maximum = [
            None if math.isinf(float(value)) else float(value)
            for value in self._maximum.tolist()
        ]
        result: dict[str, Any] = {
            "future_offsets": list(self.future_offsets),
            "batch_count": self._batch_count,
            "valid_count_xyz": [int(value) for value in self._count.tolist()],
            "observed_min_xyz_meters": minimum,
            "observed_max_xyz_meters": maximum,
            "observed_max_abs_xyz_meters": [
                None
                if lower is None or upper is None
                else max(abs(lower), abs(upper))
                for lower, upper in zip(minimum, maximum, strict=True)
            ],
            "masked_nonfinite_count_xyz": [
                int(value) for value in self._masked_nonfinite_count.tolist()
            ],
            "accumulator_dtype": "float64",
            "reduction": ALL_MOSAIC_FLOW_XYZ_STATISTICS_REDUCTION,
            "population": ALL_MOSAIC_FLOW_XYZ_STATISTICS_POPULATION,
            "view_mode": ROBOTWIN_FASTWAM_VIEW_MODE,
            "sampling": "source_point",
            "mosaic_size": list(ROBOTWIN_FASTWAM_MOSAIC_SIZE),
            "camera_order": ["head_camera", "left_camera", "right_camera"],
            "axis_order": "xyz",
        }
        if bool((self._count > 0).all().item()):
            mean, std = self._moments()
            result["mean_xyz_meters"] = [float(value) for value in mean.tolist()]
            result["std_xyz_meters"] = [float(value) for value in std.tolist()]
        else:
            result["mean_xyz_meters"] = None
            result["std_xyz_meters"] = None
        return result

    def finalize(
        self,
        *,
        statistics_id: str,
        split_manifest_sha256: str,
        dataset_receipt_sha256: str,
        qualification: str,
        scale_floor: float,
    ) -> "BoundAllMosaicFlowXYZMeanStdStatisticsArtifact":
        if not str(statistics_id).strip():
            raise ValueError("statistics_id must be non-empty.")
        if not math.isfinite(float(scale_floor)) or float(scale_floor) <= 0.0:
            raise ValueError("scale_floor must be finite and positive.")
        mean, std = self._moments()
        if bool((std <= float(scale_floor)).any().item()):
            raise ValueError(
                "Observed all-mosaic Flow XYZ std is not greater than "
                f"scale_floor={float(scale_floor):g}."
            )
        contract = HybridSourceP99FlowXYZMeanStdGeometryNormalizationContract(
            future_offsets=self.future_offsets,
            statistics_id=str(statistics_id),
            flow_mean_xyz_meters=tuple(float(value) for value in mean.tolist()),
            flow_std_xyz_meters=tuple(float(value) for value in std.tolist()),
            flow_std_floor_meters=float(scale_floor),
            flow_population=HYBRID_FLOW_POPULATION_ALL_VALID,
        )
        return BoundAllMosaicFlowXYZMeanStdStatisticsArtifact(
            normalization_contract=contract,
            split_name="train",
            split_manifest_sha256=split_manifest_sha256,
            dataset_receipt_sha256=dataset_receipt_sha256,
            qualification=qualification,
            accumulator_receipt=self.receipt(),
            scale_floor=float(scale_floor),
        )


@dataclass(frozen=True)
class BoundAllMosaicFlowXYZMeanStdStatisticsArtifact(
    BoundHeadTileFlowXYZMeanStdStatisticsArtifact
):
    """Hash-bound formal artifact for all valid values in the three-view mosaic."""

    schema_version: str = ALL_MOSAIC_FLOW_XYZ_STATISTICS_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(
            self.normalization_contract,
            HybridSourceP99FlowXYZMeanStdGeometryNormalizationContract,
        ):
            raise TypeError("Unexpected all-mosaic Flow normalization contract.")
        if self.schema_version != ALL_MOSAIC_FLOW_XYZ_STATISTICS_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Unsupported all-mosaic Flow statistics schema.")
        if self.split_name != "train":
            raise ValueError("All-mosaic Flow statistics must bind split_name=train.")
        object.__setattr__(
            self,
            "split_manifest_sha256",
            _sha256_hex(self.split_manifest_sha256, name="split_manifest_sha256"),
        )
        object.__setattr__(
            self,
            "dataset_receipt_sha256",
            _sha256_hex(self.dataset_receipt_sha256, name="dataset_receipt_sha256"),
        )
        if self.qualification not in {
            FORMAL_STATISTICS_QUALIFICATION,
            EXPLICIT_EPISODE_STATISTICS_QUALIFICATION,
            FORMAL_TRAIN_SPLIT_STRATIFIED_CALIBRATION_QUALIFICATION,
        }:
            raise ValueError("Unsupported all-mosaic Flow statistics qualification.")
        if self.normalization_contract.flow_population != HYBRID_FLOW_POPULATION_ALL_VALID:
            raise ValueError(
                "All-mosaic Flow artifact requires the all-valid population contract."
            )
        if not math.isfinite(float(self.scale_floor)) or float(self.scale_floor) <= 0:
            raise ValueError("scale_floor must be finite and positive.")
        receipt = _plain_json(self.accumulator_receipt, name="accumulator_receipt")
        if not isinstance(receipt, dict):
            raise TypeError("accumulator_receipt must be a mapping.")
        self._validate_receipt(receipt)
        object.__setattr__(self, "accumulator_receipt", receipt)
        computed = self.recompute_digest()
        if self.artifact_digest is not None and self.artifact_digest != computed:
            raise ValueError("All-mosaic Flow statistics artifact digest mismatch.")
        object.__setattr__(self, "artifact_digest", computed)

    def _validate_receipt(self, receipt: Mapping[str, Any]) -> None:
        expected = {
            "future_offsets": list(self.normalization_contract.future_offsets),
            "accumulator_dtype": "float64",
            "reduction": ALL_MOSAIC_FLOW_XYZ_STATISTICS_REDUCTION,
            "population": ALL_MOSAIC_FLOW_XYZ_STATISTICS_POPULATION,
            "view_mode": ROBOTWIN_FASTWAM_VIEW_MODE,
            "sampling": "source_point",
            "mosaic_size": list(ROBOTWIN_FASTWAM_MOSAIC_SIZE),
            "camera_order": ["head_camera", "left_camera", "right_camera"],
            "axis_order": "xyz",
            "mean_xyz_meters": list(
                self.normalization_contract.flow_mean_xyz_meters
            ),
            "std_xyz_meters": list(
                self.normalization_contract.flow_std_xyz_meters
            ),
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(f"All-mosaic Flow receipt {key} mismatch.")
        batch_count = receipt.get("batch_count")
        counts = receipt.get("valid_count_xyz")
        if (
            isinstance(batch_count, bool)
            or not isinstance(batch_count, int)
            or batch_count <= 0
            or not isinstance(counts, list)
            or len(counts) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in counts
            )
        ):
            raise ValueError("All-mosaic Flow accumulator counts are invalid.")

    @classmethod
    def load_json(
        cls,
        path: str | os.PathLike[str],
        *,
        require_formal: bool = False,
        allow_formal_calibration: bool = False,
        expected_split_manifest_sha256: str | None = None,
        expected_dataset_receipt_sha256: str | None = None,
    ) -> "BoundAllMosaicFlowXYZMeanStdStatisticsArtifact":
        artifact = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        accepted_formal_qualifications = {FORMAL_STATISTICS_QUALIFICATION}
        if allow_formal_calibration:
            accepted_formal_qualifications.add(
                FORMAL_TRAIN_SPLIT_STRATIFIED_CALIBRATION_QUALIFICATION
            )
        if require_formal and artifact.qualification not in accepted_formal_qualifications:
            raise ValueError("All-mosaic Flow statistics are not Formal-qualified.")
        for name, actual, expected in (
            (
                "split_manifest_sha256",
                artifact.split_manifest_sha256,
                expected_split_manifest_sha256,
            ),
            (
                "dataset_receipt_sha256",
                artifact.dataset_receipt_sha256,
                expected_dataset_receipt_sha256,
            ),
        ):
            if expected is not None and actual != _sha256_hex(expected, name=name):
                raise ValueError(f"All-mosaic Flow statistics {name} mismatch.")
        return artifact


def load_formal_all_mosaic_flow_xyz_mean_std_contract(
    path: str | os.PathLike[str],
    *,
    expected_split_manifest_sha256: str,
    expected_dataset_receipt_sha256: str,
) -> HybridSourceP99FlowXYZMeanStdGeometryNormalizationContract:
    """Strict factory-facing loader for one Formal all-mosaic Flow asset."""

    artifact = BoundAllMosaicFlowXYZMeanStdStatisticsArtifact.load_json(
        path,
        require_formal=True,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
    )
    return artifact.normalization_contract


__all__ = [
    "ALL_MOSAIC_FLOW_TENSOR_FIELDS",
    "ALL_MOSAIC_FLOW_XYZ_CALIBRATION_MODEL_VISIBLE_POPULATION",
    "ALL_MOSAIC_FLOW_XYZ_STATISTICS_ARTIFACT_SCHEMA_VERSION",
    "ALL_MOSAIC_FLOW_XYZ_STATISTICS_POPULATION",
    "ALL_MOSAIC_FLOW_XYZ_STATISTICS_REDUCTION",
    "AllMosaicFlowXYZMeanStdStatisticsAccumulator",
    "BoundAllMosaicFlowXYZMeanStdStatisticsArtifact",
    "load_formal_all_mosaic_flow_xyz_mean_std_contract",
]
