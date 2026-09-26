"""Versioned file binding for Hybrid K4 heterogeneous normalization.

The Hybrid K4 model keeps one auditable connection between a completed Flow
statistics scan and its numerical normalization contract.  v1 binds the
historical per-offset RMS scales; v2 binds the approved pooled mean/std z-score
across offsets ``[8,16,24,32]`` without rescanning any training window.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .hybrid_normalization import (
    HYBRID_HEAD_WRIST_NORMALIZATION_SCHEMA_VERSION,
    HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_MODE,
    HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_SCHEMA_VERSION,
    HybridHeadWristNormalizationContract,
)


HYBRID_K4_NORMALIZATION_ASSET_SCHEMA_VERSION = (
    "robotwin_momap_k4_all_xyz_normalization_asset_v1"
)
HYBRID_K4_FLOW_STATISTICS_SCHEMA_VERSION = (
    "robotwin_momap_k4_all_xyz_statistics_v1"
)
HYBRID_K4_FUTURE_OFFSETS = (8, 16, 24, 32)
HYBRID_K4_CALIBRATION_SELECTED_WINDOW_COUNT = 4096
FLOW_NORMALIZATION_PER_OFFSET_RMS = "per_offset_rms"
FLOW_NORMALIZATION_POOLED_ZSCORE = "pooled_zscore"
FLOW_NORMALIZATION_MODES = frozenset(
    {FLOW_NORMALIZATION_PER_OFFSET_RMS, FLOW_NORMALIZATION_POOLED_ZSCORE}
)


def canonical_sha256(value: Any) -> str:
    """Return the deterministic digest used by every Hybrid K4 JSON asset."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _regular_json(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular non-symlink JSON file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object: {resolved}")
    return resolved, value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _numeric_matrix(
    value: Any,
    *,
    rows: int,
    columns: int,
    label: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{label} must be {rows}x{columns}.")
    result: list[tuple[float, ...]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError(f"{label} must be {rows}x{columns}.")
        parsed: list[float] = []
        for item in row:
            if not isinstance(item, (int, float)) or isinstance(item, bool):
                raise ValueError(f"{label} contains a non-numeric value.")
            scalar = float(item)
            if not math.isfinite(scalar):
                raise ValueError(f"{label} contains NaN or Inf.")
            if positive and scalar <= 0.0:
                raise ValueError(f"{label} requires strictly positive values.")
            if nonnegative and scalar < 0.0:
                raise ValueError(f"{label} requires non-negative values.")
            parsed.append(scalar)
        result.append(tuple(parsed))
    return tuple(result)


def _flatten(matrix: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(item for row in matrix for item in row)


def _pooled_moments(
    summary: Mapping[str, Any],
    *,
    rows: int,
    columns: int,
    label: str,
) -> dict[str, tuple[float, ...]]:
    """Pool exact reported per-offset moments by their valid sample counts."""

    counts = _numeric_matrix(
        summary.get("count"),
        rows=rows,
        columns=columns,
        label=f"{label}.count",
        positive=True,
    )
    means = _numeric_matrix(
        summary.get("mean"),
        rows=rows,
        columns=columns,
        label=f"{label}.mean",
    )
    rms = _numeric_matrix(
        summary.get("rms"),
        rows=rows,
        columns=columns,
        label=f"{label}.rms",
        positive=True,
    )
    pooled_count: list[float] = []
    pooled_mean: list[float] = []
    pooled_std: list[float] = []
    pooled_rms: list[float] = []
    for column in range(columns):
        count = math.fsum(counts[row][column] for row in range(rows))
        if not math.isfinite(count) or count <= 0.0:
            raise ValueError(f"{label} has no valid values for component {column}.")
        mean = math.fsum(
            counts[row][column] * means[row][column] for row in range(rows)
        ) / count
        squared_mean = math.fsum(
            counts[row][column] * rms[row][column] ** 2 for row in range(rows)
        ) / count
        variance = squared_mean - mean**2
        # Floating point reconstruction of the published summaries can produce
        # a tiny negative epsilon; a materially negative value is invalid.
        tolerance = 1e-12 * max(1.0, squared_mean)
        if variance < -tolerance:
            raise ValueError(f"{label} produced negative pooled variance.")
        variance = max(variance, 0.0)
        std = math.sqrt(variance)
        if not math.isfinite(std) or std <= 0.0:
            raise ValueError(f"{label} pooled std must be positive.")
        pooled_count.append(count)
        pooled_mean.append(mean)
        pooled_std.append(std)
        pooled_rms.append(math.sqrt(squared_mean))
    return {
        "count": tuple(pooled_count),
        "mean": tuple(pooled_mean),
        "std_ddof0": tuple(pooled_std),
        "rms": tuple(pooled_rms),
    }


def _validate_statistics_payload(
    value: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    if value.get("schema_version") != HYBRID_K4_FLOW_STATISTICS_SCHEMA_VERSION:
        raise ValueError(f"{label} schema_version is not Hybrid K4 statistics v1.")
    if value.get("status") != "PASS" or value.get("passed") is not True:
        raise ValueError(f"{label} is not a PASS statistics artifact.")
    claimed_digest = value.get("artifact_digest")
    payload = dict(value)
    payload.pop("artifact_digest", None)
    if claimed_digest != _statistics_digest(payload):
        raise ValueError(f"{label} artifact_digest mismatch.")
    contract = _mapping(value.get("contract"), label=f"{label}.contract")
    if tuple(contract.get("future_offsets", ())) != HYBRID_K4_FUTURE_OFFSETS:
        raise ValueError(f"{label} has unexpected future_offsets.")
    selected = _mapping(contract.get("selected"), label=f"{label}.contract.selected")
    if contract.get("selection_mode") != "stratified" or int(
        selected.get("count", 0)
    ) != HYBRID_K4_CALIBRATION_SELECTED_WINDOW_COUNT:
        raise ValueError(
            f"{label} must be the fixed stratified "
            f"{HYBRID_K4_CALIBRATION_SELECTED_WINDOW_COUNT}-window calibration."
        )
    if value.get("selected_population_exact_cover") is not True:
        raise ValueError(f"{label} lacks selected-population exact cover.")
    _mapping(value.get("summary"), label=f"{label}.summary")
    return contract


def _statistics_digest(payload: Mapping[str, Any]) -> str:
    """Compatibility digest for the statistics tool (same bytes, no prefix)."""

    return canonical_sha256(payload).removeprefix("sha256:")


def _source_xyz_from_geometry(
    value: Mapping[str, Any],
    *,
    label: str,
) -> tuple[tuple[float, ...], tuple[float, ...], str]:
    source = value.get("normalization_contract", value)
    source = _mapping(source, label=label)
    mean = source.get("source_mean_xyz_meters", source.get("xyz_center"))
    std = source.get("source_std_xyz_meters", source.get("xyz_scale"))
    statistics_id = source.get("statistics_id")
    if not isinstance(statistics_id, str) or not statistics_id:
        raise ValueError(f"{label} lacks statistics_id.")

    def vector(raw: Any, *, vector_label: str) -> tuple[float, ...]:
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError(f"{vector_label} must contain exactly XYZ3 values.")
        result: list[float] = []
        for item in raw:
            if not isinstance(item, (int, float)) or isinstance(item, bool):
                raise ValueError(f"{vector_label} contains a non-numeric value.")
            scalar = float(item)
            if not math.isfinite(scalar):
                raise ValueError(f"{vector_label} contains NaN or Inf.")
            result.append(scalar)
        return tuple(result)

    mean_values = vector(mean, vector_label=f"{label}.source_mean_xyz_meters")
    std_values = vector(std, vector_label=f"{label}.source_std_xyz_meters")
    if any(item <= 0.0 for item in std_values):
        raise ValueError(f"{label}.source_std_xyz_meters must be positive.")
    return mean_values, std_values, statistics_id


def build_hybrid_k4_normalization_asset(
    *,
    statistics_path: str | Path,
    source_geometry_path: str | Path,
    flow_normalization: str = FLOW_NORMALIZATION_PER_OFFSET_RMS,
) -> dict[str, Any]:
    """Bind one PASS scan to a frozen Hybrid K4 normalization asset.

    ``per_offset_rms`` preserves the historical zero-preserving v1 contract.
    ``pooled_zscore`` creates the approved v2 contract: one Head XYZ and one
    Wrist UV mean/std set shared by offsets ``[8,16,24,32]``.
    """

    if flow_normalization not in FLOW_NORMALIZATION_MODES:
        raise ValueError(
            "flow_normalization must be one of "
            f"{sorted(FLOW_NORMALIZATION_MODES)!r}, got {flow_normalization!r}."
        )
    statistics_file, statistics = _regular_json(
        statistics_path, label="Hybrid K4 flow statistics"
    )
    scan_contract = _validate_statistics_payload(
        statistics, label="Hybrid K4 flow statistics"
    )
    source_file, source = _regular_json(
        source_geometry_path, label="Hybrid K4 source XYZ normalization"
    )
    source_mean, source_std, source_statistics_id = _source_xyz_from_geometry(
        source, label="Hybrid K4 source XYZ normalization"
    )
    summary = _mapping(statistics.get("summary"), label="Hybrid K4 statistics summary")
    head = _mapping(summary.get("head_dxyz_meters"), label="Hybrid Head summary")
    wrist = _mapping(
        summary.get("shared_wrist_dxyz_meters"), label="Hybrid shared Wrist summary"
    )
    scan_statistics_id = scan_contract.get("statistics_id")
    if not isinstance(scan_statistics_id, str) or not scan_statistics_id:
        raise ValueError("Hybrid K4 statistics contract lacks statistics_id.")

    if flow_normalization == FLOW_NORMALIZATION_PER_OFFSET_RMS:
        head_scale = _flatten(
            _numeric_matrix(
                head.get("rms"),
                rows=4,
                columns=3,
                label="Hybrid Head RMS",
                positive=True,
            )
        )
        wrist_scale = _flatten(
            _numeric_matrix(
                wrist.get("rms"),
                rows=4,
                columns=2,
                label="Hybrid Wrist RMS",
                positive=True,
            )
        )
        normalization = HybridHeadWristNormalizationContract(
            statistics_id=f"{scan_statistics_id}__hybrid_k4_rms_binding_v1",
            future_offsets=HYBRID_K4_FUTURE_OFFSETS,
            source_mean_xyz_meters=source_mean,
            source_std_xyz_meters=source_std,
            head_scale_channels_meters=head_scale,
            wrist_scale_channels_meters=wrist_scale,
        )
        derivation: dict[str, Any] = {
            "flow_normalization": FLOW_NORMALIZATION_PER_OFFSET_RMS,
            "flow_centering": "none_zero_preserving",
            "head_dxyz_scale": "statistics.summary.head_dxyz_meters.rms",
            "wrist_duv_scale": (
                "statistics.summary.shared_wrist_dxyz_meters.rms"
            ),
        }
    else:
        head_moments = _pooled_moments(
            head,
            rows=4,
            columns=3,
            label="Hybrid Head pooled moments",
        )
        wrist_moments = _pooled_moments(
            wrist,
            rows=4,
            columns=2,
            label="Hybrid Wrist pooled moments",
        )
        normalization = HybridHeadWristNormalizationContract(
            statistics_id=(
                f"{scan_statistics_id}__hybrid_k4_pooled_zscore_binding_v1"
            ),
            future_offsets=HYBRID_K4_FUTURE_OFFSETS,
            source_mean_xyz_meters=source_mean,
            source_std_xyz_meters=source_std,
            head_flow_mean_xyz_meters=head_moments["mean"],
            head_flow_std_xyz_meters=head_moments["std_ddof0"],
            wrist_flow_mean_xyz_meters=wrist_moments["mean"],
            wrist_flow_std_xyz_meters=wrist_moments["std_ddof0"],
            schema_version=HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_SCHEMA_VERSION,
            mode=HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_MODE,
        )
        derivation = {
            "flow_normalization": FLOW_NORMALIZATION_POOLED_ZSCORE,
            "flow_centering": "pooled_mean_std_zscore_across_offsets_v1",
            "head_dxyz_mean_std": (
                "count_weighted_statistics.summary.head_dxyz_meters."
                "count_mean_rms_pooled_offsets_8_16_24_32"
            ),
            "wrist_duv_mean_std": (
                "count_weighted_statistics.summary.shared_wrist_dxyz_meters."
                "count_mean_rms_pooled_offsets_8_16_24_32"
            ),
            "pooled_flow_moments": {
                "head_dxyz_meters": {
                    key: list(values) for key, values in head_moments.items()
                },
                "shared_wrist_dxyz_meters": {
                    key: list(values) for key, values in wrist_moments.items()
                },
            },
        }

    result: dict[str, Any] = {
        "schema_version": HYBRID_K4_NORMALIZATION_ASSET_SCHEMA_VERSION,
        "status": "PASS",
        "passed": True,
        "normalization_contract": normalization.as_dict(),
        "scale_derivation": {
            **derivation,
            "statistics_selection_mode": scan_contract["selection_mode"],
            "statistics_selected_window_count": scan_contract["selected"]["count"],
        },
        "statistics_asset": {
            "path": str(statistics_file),
            "sha256": sha256_file(statistics_file),
            "artifact_digest": statistics["artifact_digest"],
            "statistics_id": scan_statistics_id,
        },
        "source_geometry_asset": {
            "path": str(source_file),
            "sha256": sha256_file(source_file),
            "statistics_id": source_statistics_id,
        },
    }
    result["binding_digest"] = canonical_sha256(result)
    return result


def _expected_derived_fields(
    schema_version: str,
) -> dict[str, str]:
    common = {
        "head_flow_unit": "meter",
        "wrist_flow_unit": "model_tile_pixel",
    }
    if schema_version == HYBRID_HEAD_WRIST_NORMALIZATION_SCHEMA_VERSION:
        return {
            **common,
            "centering": "source_xyz_mean_only_flow_zero_preserving_scale_only",
        }
    if schema_version == HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_SCHEMA_VERSION:
        return {
            **common,
            "centering": "source_xyz_mean_and_flow_pooled_mean_std_zscore_v1",
        }
    raise ValueError("Hybrid K4 normalization contract schema_version is unsupported.")


def _expected_statistics_id(
    scan_statistics_id: str,
    schema_version: str,
) -> str:
    if schema_version == HYBRID_HEAD_WRIST_NORMALIZATION_SCHEMA_VERSION:
        return f"{scan_statistics_id}__hybrid_k4_rms_binding_v1"
    if schema_version == HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_SCHEMA_VERSION:
        return f"{scan_statistics_id}__hybrid_k4_pooled_zscore_binding_v1"
    raise ValueError("Hybrid K4 normalization contract schema_version is unsupported.")


def load_bound_hybrid_k4_normalization_asset(
    path: str | Path,
) -> tuple[HybridHeadWristNormalizationContract, dict[str, Any]]:
    """Load and verify one immutable statistics-to-model binding asset."""

    asset_path, asset = _regular_json(path, label="Hybrid K4 normalization asset")
    if asset.get("schema_version") != HYBRID_K4_NORMALIZATION_ASSET_SCHEMA_VERSION:
        raise ValueError("Hybrid K4 normalization asset schema_version mismatch.")
    if asset.get("status") != "PASS" or asset.get("passed") is not True:
        raise ValueError("Hybrid K4 normalization asset is not PASS.")
    claimed_binding = asset.get("binding_digest")
    payload = dict(asset)
    payload.pop("binding_digest", None)
    if claimed_binding != canonical_sha256(payload):
        raise ValueError("Hybrid K4 normalization binding_digest mismatch.")
    contract_raw = _mapping(
        asset.get("normalization_contract"), label="Hybrid K4 normalization_contract"
    )
    constructor_raw = dict(contract_raw)
    derived = _expected_derived_fields(str(constructor_raw.get("schema_version", "")))
    for key, expected in derived.items():
        if constructor_raw.pop(key, None) != expected:
            raise ValueError(f"Hybrid K4 normalization derived field {key} drifted.")
    unexpected = sorted(
        set(constructor_raw) - set(HybridHeadWristNormalizationContract.__dataclass_fields__)
    )
    if unexpected:
        raise ValueError(
            "Hybrid K4 normalization contract has unexpected fields: "
            + repr(unexpected)
        )
    contract = HybridHeadWristNormalizationContract(**constructor_raw)

    statistics_binding = _mapping(
        asset.get("statistics_asset"), label="Hybrid K4 statistics_asset"
    )
    source_binding = _mapping(
        asset.get("source_geometry_asset"), label="Hybrid K4 source_geometry_asset"
    )
    statistics_path, statistics = _regular_json(
        statistics_binding.get("path"), label="bound Hybrid K4 flow statistics"
    )
    if statistics_binding.get("sha256") != sha256_file(statistics_path):
        raise ValueError("Bound Hybrid K4 flow statistics SHA256 mismatch.")
    statistics_contract = _validate_statistics_payload(
        statistics, label="bound Hybrid K4 flow statistics"
    )
    if statistics_binding.get("artifact_digest") != statistics.get("artifact_digest"):
        raise ValueError("Bound Hybrid K4 flow statistics artifact digest mismatch.")
    if statistics_binding.get("statistics_id") != statistics_contract.get("statistics_id"):
        raise ValueError("Bound Hybrid K4 flow statistics ID mismatch.")
    expected_statistics_id = _expected_statistics_id(
        str(statistics_contract["statistics_id"]),
        contract.schema_version,
    )
    if contract.statistics_id != expected_statistics_id:
        raise ValueError("Hybrid K4 normalization contract/statistics ID mismatch.")

    source_path, source = _regular_json(
        source_binding.get("path"), label="bound Hybrid K4 source XYZ normalization"
    )
    if source_binding.get("sha256") != sha256_file(source_path):
        raise ValueError("Bound Hybrid K4 source XYZ SHA256 mismatch.")
    source_mean, source_std, source_statistics_id = _source_xyz_from_geometry(
        source, label="bound Hybrid K4 source XYZ normalization"
    )
    if source_binding.get("statistics_id") != source_statistics_id:
        raise ValueError("Bound Hybrid K4 source XYZ statistics ID mismatch.")
    if (
        contract.source_mean_xyz_meters != source_mean
        or contract.source_std_xyz_meters != source_std
    ):
        raise ValueError("Hybrid K4 source XYZ values differ from the bound asset.")

    source = {
        "kind": "bound_hybrid_k4_statistics_artifact",
        "path": str(asset_path),
        "sha256": sha256_file(asset_path),
        "binding_digest": claimed_binding,
        "schema_version": asset["schema_version"],
        "statistics_id": contract.statistics_id,
        "statistics_asset": dict(statistics_binding),
        "source_geometry_asset": dict(source_binding),
    }
    return contract, source


__all__ = [
    "FLOW_NORMALIZATION_MODES",
    "FLOW_NORMALIZATION_PER_OFFSET_RMS",
    "FLOW_NORMALIZATION_POOLED_ZSCORE",
    "HYBRID_K4_NORMALIZATION_ASSET_SCHEMA_VERSION",
    "build_hybrid_k4_normalization_asset",
    "canonical_sha256",
    "load_bound_hybrid_k4_normalization_asset",
    "sha256_file",
]
