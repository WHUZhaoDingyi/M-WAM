"""Persistent train-split normalization for canonical XYZ and MoMap offsets."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .conditioning import CANONICAL_REFERENCE_FRAME
from .kflow import KFLOW_CHANNEL_ORDER


GEOMETRY_NORMALIZATION_SCHEMA_VERSION = "robotwin_momap_geometry_normalization_v1"
GEOMETRY_NORMALIZATION_MODE = "fixed_train_xyz_affine_offset_scale_only"
PER_SAMPLE_P99_GEOMETRY_NORMALIZATION_SCHEMA_VERSION = (
    "robotwin_momap_per_sample_geometry_normalization_v1"
)
PER_SAMPLE_P99_GEOMETRY_NORMALIZATION_MODE = (
    "per_sample_source_xyz_l2_p99_shared_scale"
)
HYBRID_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION = (
    "robotwin_momap_hybrid_source_p99_flow_xyz_mean_std_v1"
)
HYBRID_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_MODE = (
    "per_sample_source_xyz_l2_p99_fixed_flow_xyz_mean_std_shared_horizons"
)
SOURCE_XYZ_MEAN_STD_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION = (
    "robotwin_momap_source_xyz_mean_std_flow_xyz_mean_std_v1"
)
SOURCE_XYZ_MEAN_STD_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_MODE = (
    "fixed_source_xyz_mean_std_fixed_flow_xyz_mean_std_shared_horizons"
)
SOURCE_XYZ_MEAN_STD_POPULATION = "formal_train_split_valid_source_xyz"
SOURCE_XYZ_MEAN_STD_STATISTICS_REDUCTION = "population_mean_std_ddof_0"
FIXED_GLOBAL_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION = (
    "robotwin_momap_fixed_global_source_p99_flow_xyz_mean_std_v1"
)
FIXED_GLOBAL_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_MODE = (
    "fixed_global_source_p99_fixed_flow_xyz_mean_std_shared_horizons"
)
FIXED_GLOBAL_SOURCE_SCALE_POOLED_L2_P99_SCALAR = "pooled_l2_p99_scalar"
FIXED_GLOBAL_SOURCE_SCALE_POOLED_AXIS_ABS_P99_XYZ = "pooled_axis_abs_p99_xyz"
FIXED_GLOBAL_SOURCE_SCALE_KINDS = frozenset(
    {
        FIXED_GLOBAL_SOURCE_SCALE_POOLED_L2_P99_SCALAR,
        FIXED_GLOBAL_SOURCE_SCALE_POOLED_AXIS_ABS_P99_XYZ,
    }
)
FIXED_GLOBAL_SOURCE_POPULATION = "model_visible_mosaic_source_valid_points"
FIXED_GLOBAL_SOURCE_STATISTICS_REDUCTION = (
    "deterministic_train_window_calibration_nearest_rank"
)
HYBRID_FLOW_POPULATION_ALL_VALID = "all_flow_valid_values_pooled_horizons"
HYBRID_FLOW_POPULATION_FASTWAM_HEAD_TILE = (
    "fastwam_head_tile_flow_valid_values_pooled_horizons"
)
FASTWAM_HEAD_TILE_FLOW_VALID_VALUES_POOLED_HORIZONS = (
    HYBRID_FLOW_POPULATION_FASTWAM_HEAD_TILE
)


def _finite_tuple(values: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"`{name}` must be a list or tuple.")
    if not values:
        raise ValueError(f"`{name}` must not be empty.")
    if any(type(value) not in (int, float) for value in values):
        raise TypeError(f"`{name}` must contain only built-in int/float values.")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"`{name}` contains NaN or Inf.")
    return result


def _sha256_hex(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"`{name}` must be a lowercase SHA256 hex string.")
    normalized = value.removeprefix("sha256:")
    if (
        len(normalized) != 64
        or normalized.lower() != normalized
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"`{name}` must contain exactly 64 lowercase hex digits.")
    return normalized


@dataclass(frozen=True)
class GeometryNormalizationContract:
    """Immutable geometry statistics tied to the formal train split.

    Canonical XYZ is centered and scaled with one global XYZ transform.
    Offsets are scale-only, so physical zero remains exact zero after
    normalization.  Per-view, per-episode, and per-sample statistics are not
    representable by this schema.
    """

    statistics_id: str
    future_offsets: tuple[int, ...]
    xyz_center: tuple[float, ...]
    xyz_scale: tuple[float, ...]
    offset_scale: tuple[float, ...]
    # Legacy offline-audit metadata is optional and is never required by the
    # model factory, checkpoint contract, or resume path.
    split_name: str | None = None
    split_manifest_sha256: str | None = None
    dataset_receipt_sha256: str | None = None
    reference_frame: str = CANONICAL_REFERENCE_FRAME
    source_unit: str = "meter"
    channel_order: str = KFLOW_CHANNEL_ORDER
    schema_version: str = GEOMETRY_NORMALIZATION_SCHEMA_VERSION
    mode: str = GEOMETRY_NORMALIZATION_MODE
    reduction: str = "valid_only_population_xyz_std_and_offset_rms"
    digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != GEOMETRY_NORMALIZATION_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported geometry normalization schema_version "
                f"{self.schema_version!r}."
            )
        if self.mode != GEOMETRY_NORMALIZATION_MODE:
            raise ValueError(
                f"Unsupported geometry normalization mode {self.mode!r}."
            )
        if not str(self.statistics_id).strip():
            raise ValueError("`statistics_id` must be non-empty.")
        if self.split_name is not None and self.split_name != "train":
            raise ValueError(
                "Geometry statistics split_name must be 'train' when supplied; "
                f"got {self.split_name!r}."
            )
        if self.split_manifest_sha256 is not None:
            object.__setattr__(
                self,
                "split_manifest_sha256",
                _sha256_hex(
                    self.split_manifest_sha256,
                    name="split_manifest_sha256",
                ),
            )
        if self.dataset_receipt_sha256 is not None:
            object.__setattr__(
                self,
                "dataset_receipt_sha256",
                _sha256_hex(
                    self.dataset_receipt_sha256,
                    name="dataset_receipt_sha256",
                ),
            )
        if (
            not isinstance(self.future_offsets, (list, tuple))
            or not self.future_offsets
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.future_offsets
            )
        ):
            raise ValueError("`future_offsets` must be non-empty integer values.")
        offsets = tuple(int(value) for value in self.future_offsets)
        if any(value <= 0 for value in offsets) or any(
            left >= right for left, right in zip(offsets, offsets[1:])
        ):
            raise ValueError(
                "`future_offsets` must be positive and strictly increasing."
            )
        object.__setattr__(self, "future_offsets", offsets)
        object.__setattr__(
            self,
            "xyz_center",
            _finite_tuple(self.xyz_center, name="xyz_center"),
        )
        object.__setattr__(
            self,
            "xyz_scale",
            _finite_tuple(self.xyz_scale, name="xyz_scale"),
        )
        object.__setattr__(
            self,
            "offset_scale",
            _finite_tuple(self.offset_scale, name="offset_scale"),
        )
        if len(self.xyz_center) != 3 or len(self.xyz_scale) != 3:
            raise ValueError("Canonical XYZ center/scale must each contain XYZ3.")
        if len(self.offset_scale) != 3 * len(self.future_offsets):
            raise ValueError(
                "Offset scale must contain K*XYZ channels: expected "
                f"{3 * len(self.future_offsets)}, got {len(self.offset_scale)}."
            )
        if any(value <= 0.0 for value in self.xyz_scale):
            raise ValueError("Every canonical XYZ scale must be positive.")
        if any(value <= 0.0 for value in self.offset_scale):
            raise ValueError("Every offset scale must be positive.")
        if self.reference_frame != CANONICAL_REFERENCE_FRAME:
            raise ValueError(
                "Geometry normalization requires the common reference "
                f"{CANONICAL_REFERENCE_FRAME!r}, got {self.reference_frame!r}."
            )
        if self.source_unit != "meter":
            raise ValueError("Geometry normalization source_unit must be 'meter'.")
        if self.channel_order != KFLOW_CHANNEL_ORDER:
            raise ValueError(
                f"Unsupported channel order {self.channel_order!r}; expected "
                f"{KFLOW_CHANNEL_ORDER!r}."
            )
        if self.reduction != "valid_only_population_xyz_std_and_offset_rms":
            raise ValueError(
                f"Unsupported geometry statistics reduction {self.reduction!r}."
            )
        computed_digest = self.recompute_digest()
        if self.digest is not None and self.digest != computed_digest:
            raise ValueError(
                "Geometry normalization digest mismatch: supplied "
                f"{self.digest!r}, recomputed {computed_digest!r}."
            )
        object.__setattr__(self, "digest", computed_digest)

    @property
    def k(self) -> int:
        return len(self.future_offsets)

    @property
    def flow_channels(self) -> int:
        return 3 * self.k

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "statistics_id": self.statistics_id,
            "split_name": self.split_name,
            "split_manifest_sha256": self.split_manifest_sha256,
            "dataset_receipt_sha256": self.dataset_receipt_sha256,
            "reference_frame": self.reference_frame,
            "source_unit": self.source_unit,
            "channel_order": self.channel_order,
            "future_offsets": list(self.future_offsets),
            "k": self.k,
            "xyz_center": list(self.xyz_center),
            "xyz_scale": list(self.xyz_scale),
            "offset_scale": list(self.offset_scale),
            "offset_center": [0.0] * self.flow_channels,
            "reduction": self.reduction,
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
        return {**self._digest_payload(), "digest": self.digest}

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "GeometryNormalizationContract":
        if not isinstance(payload, Mapping):
            raise TypeError("Geometry normalization payload must be a mapping.")
        required = {
            "schema_version",
            "mode",
            "statistics_id",
            "split_name",
            "split_manifest_sha256",
            "dataset_receipt_sha256",
            "reference_frame",
            "source_unit",
            "channel_order",
            "future_offsets",
            "xyz_center",
            "xyz_scale",
            "offset_scale",
            "reduction",
            "digest",
        }
        optional = {"k", "offset_center"}
        missing = sorted(required - set(payload))
        unexpected = sorted(set(payload) - required - optional)
        if missing or unexpected:
            raise ValueError(
                "Geometry normalization artifact key mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if "k" in payload and payload["k"] != len(payload["future_offsets"]):
            raise ValueError("Geometry normalization artifact `k` is inconsistent.")
        offset_center = payload.get(
            "offset_center",
            [0.0] * len(payload["offset_scale"]),
        )
        if (
            not isinstance(offset_center, (list, tuple))
            or len(offset_center) != len(payload["offset_scale"])
            or any(
                type(value) not in (int, float) or float(value) != 0.0
                for value in offset_center
            )
        ):
            raise ValueError(
                "Geometry normalization offset_center must be explicit all-zero."
            )
        return cls(**{key: payload[key] for key in required})

    @classmethod
    def load_json(
        cls,
        path: str | os.PathLike[str],
    ) -> "GeometryNormalizationContract":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def dump_json(self, path: str | os.PathLike[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            json.dumps(
                self.as_dict(),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    @staticmethod
    def _validate_spatial(
        value: torch.Tensor,
        *,
        channels: int,
        name: str,
    ) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"`{name}` must be a torch.Tensor.")
        if value.ndim != 4 or value.shape[1] != channels:
            raise ValueError(
                f"`{name}` must be [B,{channels},H,W], got {tuple(value.shape)}."
            )
        if not value.is_floating_point():
            raise TypeError(f"`{name}` must be floating point, got {value.dtype}.")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"`{name}` contains NaN or Inf.")

    @staticmethod
    def _validate_valid(
        valid: torch.Tensor | None,
        *,
        value: torch.Tensor,
    ) -> torch.Tensor | None:
        if valid is None:
            return None
        if not isinstance(valid, torch.Tensor):
            raise TypeError("Geometry `valid` must be a torch.Tensor.")
        expected = (value.shape[0], 1, value.shape[2], value.shape[3])
        if tuple(valid.shape) != expected:
            raise ValueError(
                f"Geometry `valid` must be [B,1,H,W]={expected}, "
                f"got {tuple(valid.shape)}."
            )
        if valid.dtype != torch.bool:
            raise TypeError("Geometry `valid` must use torch.bool.")
        return valid.to(device=value.device)

    def normalize_xyz(
        self,
        xyz_meters: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_spatial(xyz_meters, channels=3, name="xyz_meters")
        valid = self._validate_valid(valid, value=xyz_meters)
        center = torch.tensor(
            self.xyz_center,
            device=xyz_meters.device,
            dtype=xyz_meters.dtype,
        ).view(1, 3, 1, 1)
        scale = torch.tensor(
            self.xyz_scale,
            device=xyz_meters.device,
            dtype=xyz_meters.dtype,
        ).view(1, 3, 1, 1)
        normalized = (xyz_meters - center) / scale
        if valid is not None:
            normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
        return normalized

    def denormalize_xyz(self, xyz_normalized: torch.Tensor) -> torch.Tensor:
        self._validate_spatial(xyz_normalized, channels=3, name="xyz_normalized")
        center = torch.tensor(
            self.xyz_center,
            device=xyz_normalized.device,
            dtype=xyz_normalized.dtype,
        ).view(1, 3, 1, 1)
        scale = torch.tensor(
            self.xyz_scale,
            device=xyz_normalized.device,
            dtype=xyz_normalized.dtype,
        ).view(1, 3, 1, 1)
        return xyz_normalized * scale + center

    def normalize_offset(self, offset_meters: torch.Tensor) -> torch.Tensor:
        self._validate_spatial(
            offset_meters,
            channels=self.flow_channels,
            name="offset_meters",
        )
        scale = torch.tensor(
            self.offset_scale,
            device=offset_meters.device,
            dtype=offset_meters.dtype,
        ).view(1, -1, 1, 1)
        return offset_meters / scale

    def denormalize_offset(self, offset_normalized: torch.Tensor) -> torch.Tensor:
        self._validate_spatial(
            offset_normalized,
            channels=self.flow_channels,
            name="offset_normalized",
        )
        scale = torch.tensor(
            self.offset_scale,
            device=offset_normalized.device,
            dtype=offset_normalized.dtype,
        ).view(1, -1, 1, 1)
        return offset_normalized * scale


@dataclass(frozen=True)
class PerSampleP99GeometryNormalizationContract:
    """Source-only per-sample scale shared by canonical XYZ and KFlow.

    The scale is the nearest-rank P99 of valid canonical XYZ vector lengths in
    the exact source mosaic consumed by the model. It is derived independently
    for every sample from current observations only; no future target or
    dataset-level statistics are used.
    """

    future_offsets: tuple[int, ...]
    quantile_numerator: int = 99
    quantile_denominator: int = 100
    minimum_valid_points: int = 1
    scale_floor_meters: float = 1.0e-6
    reference_frame: str = CANONICAL_REFERENCE_FRAME
    source_unit: str = "meter"
    channel_order: str = KFLOW_CHANNEL_ORDER
    schema_version: str = PER_SAMPLE_P99_GEOMETRY_NORMALIZATION_SCHEMA_VERSION
    mode: str = PER_SAMPLE_P99_GEOMETRY_NORMALIZATION_MODE
    vector_norm: str = "l2"
    quantile_definition: str = "nearest_rank"
    population: str = "model_visible_mosaic_source_valid_points"
    centering: str = "none_head_camera_origin"
    clip_policy: str = "none"
    degenerate_policy: str = "error"
    digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != PER_SAMPLE_P99_GEOMETRY_NORMALIZATION_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported per-sample geometry normalization schema_version "
                f"{self.schema_version!r}."
            )
        if self.mode != PER_SAMPLE_P99_GEOMETRY_NORMALIZATION_MODE:
            raise ValueError(
                f"Unsupported per-sample geometry normalization mode {self.mode!r}."
            )
        if (
            not isinstance(self.future_offsets, (list, tuple))
            or not self.future_offsets
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.future_offsets
            )
        ):
            raise ValueError("`future_offsets` must be non-empty integer values.")
        offsets = tuple(int(value) for value in self.future_offsets)
        if any(value <= 0 for value in offsets) or any(
            left >= right for left, right in zip(offsets, offsets[1:])
        ):
            raise ValueError(
                "`future_offsets` must be positive and strictly increasing."
            )
        object.__setattr__(self, "future_offsets", offsets)
        integer_fields = (
            self.quantile_numerator,
            self.quantile_denominator,
            self.minimum_valid_points,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_fields
        ):
            raise TypeError(
                "quantile numerator/denominator and minimum_valid_points must be integers."
            )
        if not 0 < self.quantile_numerator <= self.quantile_denominator:
            raise ValueError("The empirical quantile must lie in (0,1].")
        if self.minimum_valid_points <= 0:
            raise ValueError("`minimum_valid_points` must be positive.")
        if (
            type(self.scale_floor_meters) not in (int, float)
            or not math.isfinite(float(self.scale_floor_meters))
            or float(self.scale_floor_meters) <= 0.0
        ):
            raise ValueError("`scale_floor_meters` must be finite and positive.")
        object.__setattr__(
            self,
            "scale_floor_meters",
            float(self.scale_floor_meters),
        )
        expected_strings = {
            "reference_frame": CANONICAL_REFERENCE_FRAME,
            "source_unit": "meter",
            "channel_order": KFLOW_CHANNEL_ORDER,
            "vector_norm": "l2",
            "quantile_definition": "nearest_rank",
            "population": "model_visible_mosaic_source_valid_points",
            "centering": "none_head_camera_origin",
            "clip_policy": "none",
            "degenerate_policy": "error",
        }
        for name, expected in expected_strings.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"Per-sample geometry normalization {name} must be "
                    f"{expected!r}."
                )
        computed_digest = self.recompute_digest()
        if self.digest is not None and self.digest != computed_digest:
            raise ValueError(
                "Per-sample geometry normalization digest mismatch: supplied "
                f"{self.digest!r}, recomputed {computed_digest!r}."
            )
        object.__setattr__(self, "digest", computed_digest)

    @property
    def k(self) -> int:
        return len(self.future_offsets)

    @property
    def flow_channels(self) -> int:
        return 3 * self.k

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "future_offsets": list(self.future_offsets),
            "k": self.k,
            "reference_frame": self.reference_frame,
            "source_unit": self.source_unit,
            "channel_order": self.channel_order,
            "quantile_numerator": self.quantile_numerator,
            "quantile_denominator": self.quantile_denominator,
            "vector_norm": self.vector_norm,
            "quantile_definition": self.quantile_definition,
            "population": self.population,
            "centering": self.centering,
            "minimum_valid_points": self.minimum_valid_points,
            "scale_floor_meters": self.scale_floor_meters,
            "clip_policy": self.clip_policy,
            "degenerate_policy": self.degenerate_policy,
            "target_scale": "same_source_xyz_p99_scalar_all_horizons_xyz",
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
        return {**self._digest_payload(), "digest": self.digest}

    def derive_scale(
        self,
        xyz_meters: torch.Tensor,
        *,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return one FP32 ``[B,1,1,1]`` source-derived P99 scale per sample."""

        GeometryNormalizationContract._validate_spatial(
            xyz_meters,
            channels=3,
            name="xyz_meters",
        )
        prepared_valid = GeometryNormalizationContract._validate_valid(
            valid,
            value=xyz_meters,
        )
        assert prepared_valid is not None
        xyz_fp32 = xyz_meters.to(dtype=torch.float32)
        mask = prepared_valid[:, 0].reshape(xyz_fp32.shape[0], -1)
        valid_count = mask.sum(dim=1, dtype=torch.long)
        too_sparse = valid_count < self.minimum_valid_points
        if bool(too_sparse.any().item()):
            indices = torch.nonzero(too_sparse, as_tuple=False).flatten().tolist()
            raise ValueError(
                "Per-sample P99 geometry normalization requires at least "
                f"{self.minimum_valid_points} valid source points; bad samples={indices}."
            )

        radius = torch.linalg.vector_norm(xyz_fp32, ord=2, dim=1).reshape(
            xyz_fp32.shape[0],
            -1,
        )
        ordered = radius.masked_fill(~mask, torch.inf).sort(dim=1).values
        rank_one_based = torch.div(
            self.quantile_numerator * valid_count + self.quantile_denominator - 1,
            self.quantile_denominator,
            rounding_mode="floor",
        )
        rank_zero_based = rank_one_based - 1
        scale = ordered.gather(1, rank_zero_based.unsqueeze(1)).reshape(-1, 1, 1, 1)
        invalid_scale = (~torch.isfinite(scale)) | (scale <= self.scale_floor_meters)
        if bool(invalid_scale.any().item()):
            indices = torch.nonzero(
                invalid_scale.reshape(-1),
                as_tuple=False,
            ).flatten().tolist()
            raise ValueError(
                "Per-sample source XYZ P99 scale is non-finite or not greater "
                f"than {self.scale_floor_meters:g} meter; bad samples={indices}."
            )
        return scale

    def _validate_sample_scale(
        self,
        sample_scale_meters: torch.Tensor,
        *,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(sample_scale_meters, torch.Tensor):
            raise TypeError("`sample_scale_meters` must be a torch.Tensor.")
        expected = (value.shape[0], 1, 1, 1)
        if tuple(sample_scale_meters.shape) != expected:
            raise ValueError(
                f"`sample_scale_meters` must be {expected}, got "
                f"{tuple(sample_scale_meters.shape)}."
            )
        scale = sample_scale_meters.to(device=value.device, dtype=torch.float32)
        if bool(
            ((~torch.isfinite(scale)) | (scale <= self.scale_floor_meters)).any().item()
        ):
            raise ValueError("`sample_scale_meters` contains an invalid scale.")
        return scale

    def normalize_xyz(
        self,
        xyz_meters: torch.Tensor,
        *,
        valid: torch.Tensor,
        sample_scale_meters: torch.Tensor | None = None,
    ) -> torch.Tensor:
        GeometryNormalizationContract._validate_spatial(
            xyz_meters,
            channels=3,
            name="xyz_meters",
        )
        prepared_valid = GeometryNormalizationContract._validate_valid(
            valid,
            value=xyz_meters,
        )
        assert prepared_valid is not None
        xyz_fp32 = xyz_meters.to(dtype=torch.float32)
        if sample_scale_meters is None:
            sample_scale_meters = self.derive_scale(
                xyz_fp32,
                valid=prepared_valid,
            )
        scale = self._validate_sample_scale(
            sample_scale_meters,
            value=xyz_fp32,
        )
        normalized = xyz_fp32 / scale
        return torch.where(
            prepared_valid,
            normalized,
            torch.zeros_like(normalized),
        )

    def normalize_offset(
        self,
        offset_meters: torch.Tensor,
        *,
        sample_scale_meters: torch.Tensor,
    ) -> torch.Tensor:
        GeometryNormalizationContract._validate_spatial(
            offset_meters,
            channels=self.flow_channels,
            name="offset_meters",
        )
        offset_fp32 = offset_meters.to(dtype=torch.float32)
        scale = self._validate_sample_scale(
            sample_scale_meters,
            value=offset_fp32,
        )
        return offset_fp32 / scale

    def denormalize_offset(
        self,
        offset_normalized: torch.Tensor,
        *,
        sample_scale_meters: torch.Tensor,
    ) -> torch.Tensor:
        GeometryNormalizationContract._validate_spatial(
            offset_normalized,
            channels=self.flow_channels,
            name="offset_normalized",
        )
        offset_fp32 = offset_normalized.to(dtype=torch.float32)
        scale = self._validate_sample_scale(
            sample_scale_meters,
            value=offset_fp32,
        )
        return offset_fp32 * scale


@dataclass(frozen=True)
class HybridSourceP99FlowXYZMeanStdGeometryNormalizationContract(
    PerSampleP99GeometryNormalizationContract
):
    """Keep source-only P99 geometry scaling but z-score KFlow by XYZ axis.

    The source observation remains normalized exactly like
    :class:`PerSampleP99GeometryNormalizationContract`. KFlow offsets instead
    use three frozen-population means and population standard deviations shared
    by every future horizon. Values are not clipped. Consequently, physical
    zero generally maps to ``-mean / std`` rather than normalized zero.
    """

    statistics_id: str = ""
    flow_mean_xyz_meters: tuple[float, ...] = ()
    flow_std_xyz_meters: tuple[float, ...] = ()
    flow_std_floor_meters: float = 1.0e-6
    flow_axis_order: str = "xyz"
    flow_population: str = HYBRID_FLOW_POPULATION_ALL_VALID
    flow_statistics_reduction: str = "population_mean_std_ddof_0"
    flow_centering: str = "fixed_mean_xyz"
    flow_clip_policy: str = "none"
    schema_version: str = (
        HYBRID_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION
    )
    mode: str = HYBRID_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_MODE

    def __post_init__(self) -> None:
        supplied_digest = self.digest
        source_contract = PerSampleP99GeometryNormalizationContract(
            future_offsets=self.future_offsets,
            quantile_numerator=self.quantile_numerator,
            quantile_denominator=self.quantile_denominator,
            minimum_valid_points=self.minimum_valid_points,
            scale_floor_meters=self.scale_floor_meters,
            reference_frame=self.reference_frame,
            source_unit=self.source_unit,
            channel_order=self.channel_order,
            vector_norm=self.vector_norm,
            quantile_definition=self.quantile_definition,
            population=self.population,
            centering=self.centering,
            clip_policy=self.clip_policy,
            degenerate_policy=self.degenerate_policy,
        )
        source_fields = (
            "future_offsets",
            "quantile_numerator",
            "quantile_denominator",
            "minimum_valid_points",
            "scale_floor_meters",
            "reference_frame",
            "source_unit",
            "channel_order",
            "vector_norm",
            "quantile_definition",
            "population",
            "centering",
            "clip_policy",
            "degenerate_policy",
        )
        for name in source_fields:
            object.__setattr__(self, name, getattr(source_contract, name))

        if (
            self.schema_version
            != HYBRID_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported hybrid geometry normalization schema_version "
                f"{self.schema_version!r}."
            )
        if (
            self.mode
            != HYBRID_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_MODE
        ):
            raise ValueError(
                f"Unsupported hybrid geometry normalization mode {self.mode!r}."
            )
        statistics_id = str(self.statistics_id).strip()
        if not statistics_id:
            raise ValueError("`statistics_id` must be non-empty.")
        object.__setattr__(self, "statistics_id", statistics_id)

        means = _finite_tuple(
            self.flow_mean_xyz_meters,
            name="flow_mean_xyz_meters",
        )
        if len(means) != 3:
            raise ValueError("`flow_mean_xyz_meters` must contain XYZ3 means.")
        stds = _finite_tuple(
            self.flow_std_xyz_meters,
            name="flow_std_xyz_meters",
        )
        if len(stds) != 3:
            raise ValueError("`flow_std_xyz_meters` must contain XYZ3 stds.")
        if (
            type(self.flow_std_floor_meters) not in (int, float)
            or not math.isfinite(float(self.flow_std_floor_meters))
            or float(self.flow_std_floor_meters) <= 0.0
        ):
            raise ValueError("`flow_std_floor_meters` must be finite and positive.")
        flow_floor = float(self.flow_std_floor_meters)
        if any(value <= flow_floor for value in stds):
            raise ValueError(
                "Every Flow XYZ std must be greater than "
                f"flow_std_floor_meters={flow_floor:g}."
            )
        object.__setattr__(self, "flow_mean_xyz_meters", means)
        object.__setattr__(self, "flow_std_xyz_meters", stds)
        object.__setattr__(self, "flow_std_floor_meters", flow_floor)

        expected_flow_strings = {
            "flow_axis_order": "xyz",
            "flow_statistics_reduction": "population_mean_std_ddof_0",
            "flow_centering": "fixed_mean_xyz",
            "flow_clip_policy": "none",
        }
        for name, expected in expected_flow_strings.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"Hybrid Flow normalization {name} must be {expected!r}."
                )
        if self.flow_population not in {
            HYBRID_FLOW_POPULATION_ALL_VALID,
            HYBRID_FLOW_POPULATION_FASTWAM_HEAD_TILE,
        }:
            raise ValueError(
                "Hybrid Flow normalization flow_population must explicitly "
                "identify either all valid Flow or the FastWAM head tile."
            )

        computed_digest = self.recompute_digest()
        if supplied_digest is not None and supplied_digest != computed_digest:
            raise ValueError(
                "Hybrid geometry normalization digest mismatch: supplied "
                f"{supplied_digest!r}, recomputed {computed_digest!r}."
            )
        object.__setattr__(self, "digest", computed_digest)

    @property
    def flow_mean_channels_meters(self) -> tuple[float, ...]:
        return self.flow_mean_xyz_meters * self.k

    @property
    def flow_std_channels_meters(self) -> tuple[float, ...]:
        return self.flow_std_xyz_meters * self.k

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "statistics_id": self.statistics_id,
            "future_offsets": list(self.future_offsets),
            "k": self.k,
            "reference_frame": self.reference_frame,
            "source_unit": self.source_unit,
            "channel_order": self.channel_order,
            "quantile_numerator": self.quantile_numerator,
            "quantile_denominator": self.quantile_denominator,
            "vector_norm": self.vector_norm,
            "quantile_definition": self.quantile_definition,
            "population": self.population,
            "centering": self.centering,
            "minimum_valid_points": self.minimum_valid_points,
            "scale_floor_meters": self.scale_floor_meters,
            "clip_policy": self.clip_policy,
            "degenerate_policy": self.degenerate_policy,
            "source_scale": "per_sample_source_xyz_l2_p99_scalar",
            "flow_mean_xyz_meters": list(self.flow_mean_xyz_meters),
            "flow_mean_channels_meters": list(self.flow_mean_channels_meters),
            "flow_std_xyz_meters": list(self.flow_std_xyz_meters),
            "flow_std_channels_meters": list(self.flow_std_channels_meters),
            "flow_population": self.flow_population,
            "flow_statistics_reduction": self.flow_statistics_reduction,
            "flow_axis_order": self.flow_axis_order,
            "flow_centering": self.flow_centering,
            "flow_std_floor_meters": self.flow_std_floor_meters,
            "flow_clip_policy": self.flow_clip_policy,
            "target_scale": "fixed_flow_xyz_mean_std_zscore_shared_all_horizons",
        }

    def normalize_offset(
        self,
        offset_meters: torch.Tensor,
        *,
        sample_scale_meters: torch.Tensor,
    ) -> torch.Tensor:
        GeometryNormalizationContract._validate_spatial(
            offset_meters,
            channels=self.flow_channels,
            name="offset_meters",
        )
        offset_fp32 = offset_meters.to(dtype=torch.float32)
        self._validate_sample_scale(sample_scale_meters, value=offset_fp32)
        mean = torch.tensor(
            self.flow_mean_channels_meters,
            device=offset_fp32.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        std = torch.tensor(
            self.flow_std_channels_meters,
            device=offset_fp32.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        return (offset_fp32 - mean) / std

    def denormalize_offset(
        self,
        offset_normalized: torch.Tensor,
        *,
        sample_scale_meters: torch.Tensor,
    ) -> torch.Tensor:
        GeometryNormalizationContract._validate_spatial(
            offset_normalized,
            channels=self.flow_channels,
            name="offset_normalized",
        )
        offset_fp32 = offset_normalized.to(dtype=torch.float32)
        self._validate_sample_scale(sample_scale_meters, value=offset_fp32)
        mean = torch.tensor(
            self.flow_mean_channels_meters,
            device=offset_fp32.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        std = torch.tensor(
            self.flow_std_channels_meters,
            device=offset_fp32.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        return offset_fp32 * std + mean


@dataclass(frozen=True)
class SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract:
    """Use distinct frozen XYZ z-scores for source positions and Flow offsets.

    Source XYZ contains absolute source-t0 coordinates. KFlow contains future
    displacement vectors relative to source-t0. They share units, axes and a
    train split, but are different random variables and therefore retain
    separate means and population standard deviations.
    """

    statistics_id: str
    source_statistics_id: str
    flow_statistics_id: str
    split_manifest_sha256: str
    dataset_receipt_sha256: str
    future_offsets: tuple[int, ...]
    source_mean_xyz_meters: tuple[float, ...]
    source_std_xyz_meters: tuple[float, ...]
    flow_mean_xyz_meters: tuple[float, ...]
    flow_std_xyz_meters: tuple[float, ...]
    split_name: str = "train"
    source_std_floor_meters: float = 1.0e-6
    flow_std_floor_meters: float = 1.0e-6
    reference_frame: str = CANONICAL_REFERENCE_FRAME
    source_unit: str = "meter"
    channel_order: str = KFLOW_CHANNEL_ORDER
    source_axis_order: str = "xyz"
    source_population: str = SOURCE_XYZ_MEAN_STD_POPULATION
    source_statistics_reduction: str = SOURCE_XYZ_MEAN_STD_STATISTICS_REDUCTION
    source_centering: str = "fixed_mean_xyz"
    source_clip_policy: str = "none"
    flow_axis_order: str = "xyz"
    flow_population: str = HYBRID_FLOW_POPULATION_ALL_VALID
    flow_statistics_reduction: str = "population_mean_std_ddof_0"
    flow_centering: str = "fixed_mean_xyz"
    flow_clip_policy: str = "none"
    schema_version: str = (
        SOURCE_XYZ_MEAN_STD_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION
    )
    mode: str = SOURCE_XYZ_MEAN_STD_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_MODE
    digest: str | None = None

    def __post_init__(self) -> None:
        supplied_digest = self.digest
        if (
            self.schema_version
            != SOURCE_XYZ_MEAN_STD_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported source/Flow mean-std geometry schema_version "
                f"{self.schema_version!r}."
            )
        if (
            self.mode
            != SOURCE_XYZ_MEAN_STD_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_MODE
        ):
            raise ValueError(
                "Unsupported source/Flow mean-std geometry mode "
                f"{self.mode!r}."
            )
        for name in ("statistics_id", "source_statistics_id", "flow_statistics_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"`{name}` must be non-empty.")
            object.__setattr__(self, name, value)
        if self.split_name != "train":
            raise ValueError("Source/Flow mean-std statistics must use split_name='train'.")
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
        if (
            not isinstance(self.future_offsets, (list, tuple))
            or not self.future_offsets
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.future_offsets
            )
        ):
            raise ValueError("`future_offsets` must be non-empty integer values.")
        offsets = tuple(int(value) for value in self.future_offsets)
        if any(value <= 0 for value in offsets) or any(
            left >= right for left, right in zip(offsets, offsets[1:])
        ):
            raise ValueError(
                "`future_offsets` must be positive and strictly increasing."
            )
        object.__setattr__(self, "future_offsets", offsets)

        for prefix in ("source", "flow"):
            means_name = f"{prefix}_mean_xyz_meters"
            stds_name = f"{prefix}_std_xyz_meters"
            floor_name = f"{prefix}_std_floor_meters"
            means = _finite_tuple(getattr(self, means_name), name=means_name)
            stds = _finite_tuple(getattr(self, stds_name), name=stds_name)
            floor_value = getattr(self, floor_name)
            if len(means) != 3 or len(stds) != 3:
                raise ValueError(
                    f"`{means_name}` and `{stds_name}` must each contain XYZ3."
                )
            if (
                type(floor_value) not in (int, float)
                or not math.isfinite(float(floor_value))
                or float(floor_value) <= 0.0
            ):
                raise ValueError(f"`{floor_name}` must be finite and positive.")
            floor = float(floor_value)
            if any(value <= floor for value in stds):
                raise ValueError(
                    f"Every {prefix} XYZ std must exceed {floor_name}={floor:g}."
                )
            object.__setattr__(self, means_name, means)
            object.__setattr__(self, stds_name, stds)
            object.__setattr__(self, floor_name, floor)

        expected_strings = {
            "reference_frame": CANONICAL_REFERENCE_FRAME,
            "source_unit": "meter",
            "channel_order": KFLOW_CHANNEL_ORDER,
            "source_axis_order": "xyz",
            "source_population": SOURCE_XYZ_MEAN_STD_POPULATION,
            "source_statistics_reduction": SOURCE_XYZ_MEAN_STD_STATISTICS_REDUCTION,
            "source_centering": "fixed_mean_xyz",
            "source_clip_policy": "none",
            "flow_axis_order": "xyz",
            "flow_statistics_reduction": "population_mean_std_ddof_0",
            "flow_centering": "fixed_mean_xyz",
            "flow_clip_policy": "none",
        }
        for name, expected in expected_strings.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected!r}.")
        if self.flow_population not in {
            HYBRID_FLOW_POPULATION_ALL_VALID,
            HYBRID_FLOW_POPULATION_FASTWAM_HEAD_TILE,
        }:
            raise ValueError(
                "Flow population must identify all-valid mosaic or head-tile values."
            )

        computed_digest = self.recompute_digest()
        if supplied_digest is not None and supplied_digest != computed_digest:
            raise ValueError(
                "Source/Flow mean-std geometry digest mismatch: supplied "
                f"{supplied_digest!r}, recomputed {computed_digest!r}."
            )
        object.__setattr__(self, "digest", computed_digest)

    @property
    def k(self) -> int:
        return len(self.future_offsets)

    @property
    def flow_channels(self) -> int:
        return 3 * self.k

    @property
    def flow_mean_channels_meters(self) -> tuple[float, ...]:
        return self.flow_mean_xyz_meters * self.k

    @property
    def flow_std_channels_meters(self) -> tuple[float, ...]:
        return self.flow_std_xyz_meters * self.k

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "statistics_id": self.statistics_id,
            "source_statistics_id": self.source_statistics_id,
            "flow_statistics_id": self.flow_statistics_id,
            "split_name": self.split_name,
            "split_manifest_sha256": self.split_manifest_sha256,
            "dataset_receipt_sha256": self.dataset_receipt_sha256,
            "future_offsets": list(self.future_offsets),
            "k": self.k,
            "reference_frame": self.reference_frame,
            "source_unit": self.source_unit,
            "channel_order": self.channel_order,
            "source_mean_xyz_meters": list(self.source_mean_xyz_meters),
            "source_std_xyz_meters": list(self.source_std_xyz_meters),
            "source_axis_order": self.source_axis_order,
            "source_population": self.source_population,
            "source_statistics_reduction": self.source_statistics_reduction,
            "source_centering": self.source_centering,
            "source_std_floor_meters": self.source_std_floor_meters,
            "source_clip_policy": self.source_clip_policy,
            "flow_mean_xyz_meters": list(self.flow_mean_xyz_meters),
            "flow_mean_channels_meters": list(self.flow_mean_channels_meters),
            "flow_std_xyz_meters": list(self.flow_std_xyz_meters),
            "flow_std_channels_meters": list(self.flow_std_channels_meters),
            "flow_axis_order": self.flow_axis_order,
            "flow_population": self.flow_population,
            "flow_statistics_reduction": self.flow_statistics_reduction,
            "flow_centering": self.flow_centering,
            "flow_std_floor_meters": self.flow_std_floor_meters,
            "flow_clip_policy": self.flow_clip_policy,
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
        return {**self._digest_payload(), "digest": self.digest}

    def normalize_xyz(
        self,
        xyz_meters: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        GeometryNormalizationContract._validate_spatial(
            xyz_meters,
            channels=3,
            name="xyz_meters",
        )
        prepared_valid = GeometryNormalizationContract._validate_valid(
            valid,
            value=xyz_meters,
        )
        xyz_fp32 = xyz_meters.to(dtype=torch.float32)
        mean = torch.tensor(
            self.source_mean_xyz_meters,
            device=xyz_fp32.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.source_std_xyz_meters,
            device=xyz_fp32.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        normalized = (xyz_fp32 - mean) / std
        if prepared_valid is not None:
            normalized = torch.where(
                prepared_valid,
                normalized,
                torch.zeros_like(normalized),
            )
        return normalized

    def denormalize_xyz(self, xyz_normalized: torch.Tensor) -> torch.Tensor:
        GeometryNormalizationContract._validate_spatial(
            xyz_normalized,
            channels=3,
            name="xyz_normalized",
        )
        xyz_fp32 = xyz_normalized.to(dtype=torch.float32)
        mean = torch.tensor(
            self.source_mean_xyz_meters,
            device=xyz_fp32.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.source_std_xyz_meters,
            device=xyz_fp32.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        return xyz_fp32 * std + mean

    def normalize_offset(self, offset_meters: torch.Tensor) -> torch.Tensor:
        GeometryNormalizationContract._validate_spatial(
            offset_meters,
            channels=self.flow_channels,
            name="offset_meters",
        )
        offset_fp32 = offset_meters.to(dtype=torch.float32)
        mean = torch.tensor(
            self.flow_mean_channels_meters,
            device=offset_fp32.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        std = torch.tensor(
            self.flow_std_channels_meters,
            device=offset_fp32.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        return (offset_fp32 - mean) / std

    def denormalize_offset(self, offset_normalized: torch.Tensor) -> torch.Tensor:
        GeometryNormalizationContract._validate_spatial(
            offset_normalized,
            channels=self.flow_channels,
            name="offset_normalized",
        )
        offset_fp32 = offset_normalized.to(dtype=torch.float32)
        mean = torch.tensor(
            self.flow_mean_channels_meters,
            device=offset_fp32.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        std = torch.tensor(
            self.flow_std_channels_meters,
            device=offset_fp32.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        return offset_fp32 * std + mean


@dataclass(frozen=True)
class FixedGlobalSourceP99FlowXYZMeanStdGeometryNormalizationContract:
    """Use frozen train-calibration P99 scales for source XYZ and Flow z-score.

    Unlike :class:`HybridSourceP99FlowXYZMeanStdGeometryNormalizationContract`,
    this contract never derives a scale from the current sample.  Variant B
    broadcasts one pooled L2 P99 scalar across XYZ; variant C stores one
    pooled absolute-coordinate P99 for each axis.  Both variants retain the
    same fixed Flow XYZ mean/std transform shared across future horizons.
    """

    statistics_id: str
    future_offsets: tuple[int, ...]
    source_scale_kind: str
    source_scale_xyz_meters: tuple[float, ...]
    flow_mean_xyz_meters: tuple[float, ...]
    flow_std_xyz_meters: tuple[float, ...]
    source_calibration_manifest_sha256: str
    source_calibration_window_count: int
    source_quantile_numerator: int = 99
    source_quantile_denominator: int = 100
    source_quantile_definition: str = "nearest_rank"
    source_scale_floor_meters: float = 1.0e-6
    flow_std_floor_meters: float = 1.0e-6
    reference_frame: str = CANONICAL_REFERENCE_FRAME
    source_unit: str = "meter"
    channel_order: str = KFLOW_CHANNEL_ORDER
    source_population: str = FIXED_GLOBAL_SOURCE_POPULATION
    source_statistics_reduction: str = FIXED_GLOBAL_SOURCE_STATISTICS_REDUCTION
    source_centering: str = "none_head_camera_origin"
    source_clip_policy: str = "none"
    flow_axis_order: str = "xyz"
    flow_population: str = HYBRID_FLOW_POPULATION_ALL_VALID
    flow_statistics_reduction: str = "population_mean_std_ddof_0"
    flow_centering: str = "fixed_mean_xyz"
    flow_clip_policy: str = "none"
    schema_version: str = (
        FIXED_GLOBAL_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION
    )
    mode: str = (
        FIXED_GLOBAL_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_MODE
    )
    digest: str | None = None

    def __post_init__(self) -> None:
        supplied_digest = self.digest
        if (
            self.schema_version
            != FIXED_GLOBAL_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported fixed-global geometry normalization schema_version "
                f"{self.schema_version!r}."
            )
        if (
            self.mode
            != FIXED_GLOBAL_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_MODE
        ):
            raise ValueError(
                "Unsupported fixed-global geometry normalization mode "
                f"{self.mode!r}."
            )
        statistics_id = str(self.statistics_id).strip()
        if not statistics_id:
            raise ValueError("`statistics_id` must be non-empty.")
        object.__setattr__(self, "statistics_id", statistics_id)

        if (
            not isinstance(self.future_offsets, (list, tuple))
            or not self.future_offsets
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.future_offsets
            )
        ):
            raise ValueError("`future_offsets` must be non-empty integer values.")
        offsets = tuple(int(value) for value in self.future_offsets)
        if any(value <= 0 for value in offsets) or any(
            left >= right for left, right in zip(offsets, offsets[1:])
        ):
            raise ValueError(
                "`future_offsets` must be positive and strictly increasing."
            )
        object.__setattr__(self, "future_offsets", offsets)

        if self.source_scale_kind not in FIXED_GLOBAL_SOURCE_SCALE_KINDS:
            raise ValueError(
                "`source_scale_kind` must be one of "
                f"{sorted(FIXED_GLOBAL_SOURCE_SCALE_KINDS)}, got "
                f"{self.source_scale_kind!r}."
            )
        source_scale = _finite_tuple(
            self.source_scale_xyz_meters,
            name="source_scale_xyz_meters",
        )
        if len(source_scale) != 3:
            raise ValueError("`source_scale_xyz_meters` must contain XYZ3 scales.")
        if (
            type(self.source_scale_floor_meters) not in (int, float)
            or not math.isfinite(float(self.source_scale_floor_meters))
            or float(self.source_scale_floor_meters) <= 0.0
        ):
            raise ValueError(
                "`source_scale_floor_meters` must be finite and positive."
            )
        source_floor = float(self.source_scale_floor_meters)
        if any(value <= source_floor for value in source_scale):
            raise ValueError(
                "Every fixed-global source XYZ scale must be greater than "
                f"source_scale_floor_meters={source_floor:g}."
            )
        if (
            self.source_scale_kind
            == FIXED_GLOBAL_SOURCE_SCALE_POOLED_L2_P99_SCALAR
            and not source_scale[0] == source_scale[1] == source_scale[2]
        ):
            raise ValueError(
                "`pooled_l2_p99_scalar` requires one identical scale "
                "broadcast across XYZ."
            )
        object.__setattr__(self, "source_scale_xyz_meters", source_scale)
        object.__setattr__(self, "source_scale_floor_meters", source_floor)

        integer_fields = (
            self.source_quantile_numerator,
            self.source_quantile_denominator,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_fields
        ):
            raise TypeError("Source quantile numerator/denominator must be integers.")
        if not 0 < self.source_quantile_numerator <= self.source_quantile_denominator:
            raise ValueError("The source empirical quantile must lie in (0,1].")
        if (
            isinstance(self.source_calibration_window_count, bool)
            or not isinstance(self.source_calibration_window_count, int)
            or self.source_calibration_window_count <= 0
        ):
            raise ValueError(
                "`source_calibration_window_count` must be a positive integer."
            )
        object.__setattr__(
            self,
            "source_calibration_manifest_sha256",
            _sha256_hex(
                self.source_calibration_manifest_sha256,
                name="source_calibration_manifest_sha256",
            ),
        )

        means = _finite_tuple(
            self.flow_mean_xyz_meters,
            name="flow_mean_xyz_meters",
        )
        stds = _finite_tuple(
            self.flow_std_xyz_meters,
            name="flow_std_xyz_meters",
        )
        if len(means) != 3:
            raise ValueError("`flow_mean_xyz_meters` must contain XYZ3 means.")
        if len(stds) != 3:
            raise ValueError("`flow_std_xyz_meters` must contain XYZ3 stds.")
        if (
            type(self.flow_std_floor_meters) not in (int, float)
            or not math.isfinite(float(self.flow_std_floor_meters))
            or float(self.flow_std_floor_meters) <= 0.0
        ):
            raise ValueError("`flow_std_floor_meters` must be finite and positive.")
        flow_floor = float(self.flow_std_floor_meters)
        if any(value <= flow_floor for value in stds):
            raise ValueError(
                "Every Flow XYZ std must be greater than "
                f"flow_std_floor_meters={flow_floor:g}."
            )
        object.__setattr__(self, "flow_mean_xyz_meters", means)
        object.__setattr__(self, "flow_std_xyz_meters", stds)
        object.__setattr__(self, "flow_std_floor_meters", flow_floor)

        expected_strings = {
            "reference_frame": CANONICAL_REFERENCE_FRAME,
            "source_unit": "meter",
            "channel_order": KFLOW_CHANNEL_ORDER,
            "source_population": FIXED_GLOBAL_SOURCE_POPULATION,
            "source_statistics_reduction": FIXED_GLOBAL_SOURCE_STATISTICS_REDUCTION,
            "source_quantile_definition": "nearest_rank",
            "source_centering": "none_head_camera_origin",
            "source_clip_policy": "none",
            "flow_axis_order": "xyz",
            "flow_statistics_reduction": "population_mean_std_ddof_0",
            "flow_centering": "fixed_mean_xyz",
            "flow_clip_policy": "none",
        }
        for name, expected in expected_strings.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"Fixed-global geometry normalization {name} must be "
                    f"{expected!r}."
                )
        if self.flow_population not in {
            HYBRID_FLOW_POPULATION_ALL_VALID,
            HYBRID_FLOW_POPULATION_FASTWAM_HEAD_TILE,
        }:
            raise ValueError(
                "Fixed-global Flow normalization flow_population must explicitly "
                "identify either all valid Flow or the FastWAM head tile."
            )

        computed_digest = self.recompute_digest()
        if supplied_digest is not None and supplied_digest != computed_digest:
            raise ValueError(
                "Fixed-global geometry normalization digest mismatch: supplied "
                f"{supplied_digest!r}, recomputed {computed_digest!r}."
            )
        object.__setattr__(self, "digest", computed_digest)

    @property
    def k(self) -> int:
        return len(self.future_offsets)

    @property
    def flow_channels(self) -> int:
        return 3 * self.k

    @property
    def flow_mean_channels_meters(self) -> tuple[float, ...]:
        return self.flow_mean_xyz_meters * self.k

    @property
    def flow_std_channels_meters(self) -> tuple[float, ...]:
        return self.flow_std_xyz_meters * self.k

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "statistics_id": self.statistics_id,
            "future_offsets": list(self.future_offsets),
            "k": self.k,
            "reference_frame": self.reference_frame,
            "source_unit": self.source_unit,
            "channel_order": self.channel_order,
            "source_scale_kind": self.source_scale_kind,
            "source_scale_xyz_meters": list(self.source_scale_xyz_meters),
            "source_calibration_manifest_sha256": (
                self.source_calibration_manifest_sha256
            ),
            "source_calibration_window_count": self.source_calibration_window_count,
            "source_quantile_numerator": self.source_quantile_numerator,
            "source_quantile_denominator": self.source_quantile_denominator,
            "source_quantile_definition": self.source_quantile_definition,
            "source_population": self.source_population,
            "source_statistics_reduction": self.source_statistics_reduction,
            "source_centering": self.source_centering,
            "source_scale_floor_meters": self.source_scale_floor_meters,
            "source_clip_policy": self.source_clip_policy,
            "flow_mean_xyz_meters": list(self.flow_mean_xyz_meters),
            "flow_mean_channels_meters": list(self.flow_mean_channels_meters),
            "flow_std_xyz_meters": list(self.flow_std_xyz_meters),
            "flow_std_channels_meters": list(self.flow_std_channels_meters),
            "flow_population": self.flow_population,
            "flow_statistics_reduction": self.flow_statistics_reduction,
            "flow_axis_order": self.flow_axis_order,
            "flow_centering": self.flow_centering,
            "flow_std_floor_meters": self.flow_std_floor_meters,
            "flow_clip_policy": self.flow_clip_policy,
            "target_scale": "fixed_flow_xyz_mean_std_zscore_shared_all_horizons",
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
        return {**self._digest_payload(), "digest": self.digest}

    def normalize_xyz(
        self,
        xyz_meters: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        GeometryNormalizationContract._validate_spatial(
            xyz_meters,
            channels=3,
            name="xyz_meters",
        )
        prepared_valid = GeometryNormalizationContract._validate_valid(
            valid,
            value=xyz_meters,
        )
        xyz_fp32 = xyz_meters.to(dtype=torch.float32)
        scale = torch.tensor(
            self.source_scale_xyz_meters,
            device=xyz_fp32.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        normalized = xyz_fp32 / scale
        if prepared_valid is not None:
            normalized = torch.where(
                prepared_valid,
                normalized,
                torch.zeros_like(normalized),
            )
        return normalized

    def normalize_offset(self, offset_meters: torch.Tensor) -> torch.Tensor:
        GeometryNormalizationContract._validate_spatial(
            offset_meters,
            channels=self.flow_channels,
            name="offset_meters",
        )
        offset_fp32 = offset_meters.to(dtype=torch.float32)
        mean = torch.tensor(
            self.flow_mean_channels_meters,
            device=offset_fp32.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        std = torch.tensor(
            self.flow_std_channels_meters,
            device=offset_fp32.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        return (offset_fp32 - mean) / std

    def denormalize_offset(self, offset_normalized: torch.Tensor) -> torch.Tensor:
        GeometryNormalizationContract._validate_spatial(
            offset_normalized,
            channels=self.flow_channels,
            name="offset_normalized",
        )
        offset_fp32 = offset_normalized.to(dtype=torch.float32)
        mean = torch.tensor(
            self.flow_mean_channels_meters,
            device=offset_fp32.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        std = torch.tensor(
            self.flow_std_channels_meters,
            device=offset_fp32.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        return offset_fp32 * std + mean


GeometryNormalization = (
    GeometryNormalizationContract
    | PerSampleP99GeometryNormalizationContract
    | HybridSourceP99FlowXYZMeanStdGeometryNormalizationContract
    | SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract
    | FixedGlobalSourceP99FlowXYZMeanStdGeometryNormalizationContract
)


