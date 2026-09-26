"""Lightweight loader for frozen 14-D Action/Proprio statistics.

Runtime model construction needs only this authenticated JSON wrapper and the
small normalization contract.  Formal split scans, HDF5 providers, coverage,
and geometry statistics intentionally stay outside this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .normalization import FIXED_FEATURE_DIM, FixedFeatureNormalizationContract


FEATURE_ARTIFACT_SCHEMA_VERSION = "fastwam_feature_normalization_artifact_v1"
BOUND_FEATURE_STATISTICS_SCHEMA_VERSION = (
    "robotwin_momap_bound_feature_statistics_v2"
)
FORMAL_STATISTICS_QUALIFICATION = "formal_train_split_statistics"
EXPLICIT_EPISODE_STATISTICS_QUALIFICATION = (
    "explicit_episode_paths_unqualified_statistics"
)
FEATURE_STATISTICS_REDUCTION = "valid_only_population_zscore"
ACCUMULATOR_DTYPE = "float64"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _plain_json_value(value: Any, *, name: str) -> Any:
    try:
        return json.loads(_canonical_json_bytes(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"`{name}` must contain only finite JSON values.") from exc


def _sha256_hex(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"`{name}` must be a lowercase SHA256 hex string.")
    normalized = value.removeprefix("sha256:")
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"`{name}` must contain exactly 64 lowercase hex digits.")
    return normalized


def _digest_payload(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    return f"sha256:{digest}"


def _finite_vector(value: Any, *, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values.")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name}[{index}] must be numeric.")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite.")
        result.append(number)
    return tuple(result)


def _nonnegative_count_vector(
    value: Any,
    *,
    length: int,
    name: str,
    positive: bool = False,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values.")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name}[{index}] must be an integer.")
        if item < 0 or (positive and item == 0):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name}[{index}] must be {qualifier}.")
        result.append(item)
    return tuple(result)


def _validate_range_audit(
    receipt: Mapping[str, Any],
    *,
    minimum_key: str,
    maximum_key: str,
    maximum_absolute_key: str,
    length: int,
) -> None:
    minimum = _finite_vector(
        receipt.get(minimum_key), length=length, name=minimum_key
    )
    maximum = _finite_vector(
        receipt.get(maximum_key), length=length, name=maximum_key
    )
    maximum_absolute = _finite_vector(
        receipt.get(maximum_absolute_key),
        length=length,
        name=maximum_absolute_key,
    )
    for index, (lower, upper, observed_absolute) in enumerate(
        zip(minimum, maximum, maximum_absolute, strict=True)
    ):
        expected_absolute = max(abs(lower), abs(upper))
        if lower > upper:
            raise ValueError(
                f"{minimum_key}/{maximum_key}[{index}] is inverted."
            )
        if observed_absolute < 0.0 or observed_absolute != expected_absolute:
            raise ValueError(
                f"{maximum_absolute_key}[{index}] is inconsistent with range."
            )


def validate_feature_accumulator_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_feature_kind: str,
) -> None:
    if receipt.get("feature_kind") != expected_feature_kind:
        raise ValueError("Accumulator feature_kind does not match artifact.")
    if receipt.get("feature_dim") != FIXED_FEATURE_DIM:
        raise ValueError("Accumulator feature_dim must be 14.")
    batch_count = receipt.get("batch_count")
    if (
        isinstance(batch_count, bool)
        or not isinstance(batch_count, int)
        or batch_count <= 0
    ):
        raise ValueError("Accumulator batch_count must be positive.")
    _nonnegative_count_vector(
        receipt.get("valid_count"),
        length=FIXED_FEATURE_DIM,
        name="valid_count",
        positive=True,
    )
    _validate_range_audit(
        receipt,
        minimum_key="observed_min",
        maximum_key="observed_max",
        maximum_absolute_key="observed_max_abs",
        length=FIXED_FEATURE_DIM,
    )
    _nonnegative_count_vector(
        receipt.get("masked_nonfinite_count"),
        length=FIXED_FEATURE_DIM,
        name="masked_nonfinite_count",
    )
    if (
        receipt.get("accumulator_dtype") != ACCUMULATOR_DTYPE
        or receipt.get("reduction") != FEATURE_STATISTICS_REDUCTION
    ):
        raise ValueError("Accumulator receipt reduction contract mismatch.")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class BoundFeatureStatisticsArtifact:
    """Authenticated wrapper around one runtime 14-D normalization contract."""

    normalization_contract: FixedFeatureNormalizationContract
    split_name: str
    split_manifest_sha256: str
    dataset_receipt_sha256: str
    feature_semantics: Mapping[str, Any]
    accumulator_receipt: Mapping[str, Any]
    qualification: str
    scale_floor: float
    schema_version: str = FEATURE_ARTIFACT_SCHEMA_VERSION
    formal_schema_version: str = BOUND_FEATURE_STATISTICS_SCHEMA_VERSION
    artifact_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.normalization_contract, FixedFeatureNormalizationContract
        ):
            raise TypeError(
                "`normalization_contract` must be FixedFeatureNormalizationContract."
            )
        if self.schema_version != FEATURE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Unsupported feature artifact wrapper schema.")
        if self.formal_schema_version != BOUND_FEATURE_STATISTICS_SCHEMA_VERSION:
            raise ValueError("Unsupported bound feature statistics schema.")
        if self.split_name != "train":
            raise ValueError("Feature statistics must bind split_name='train'.")
        object.__setattr__(
            self,
            "split_manifest_sha256",
            _sha256_hex(
                self.split_manifest_sha256, name="split_manifest_sha256"
            ),
        )
        object.__setattr__(
            self,
            "dataset_receipt_sha256",
            _sha256_hex(
                self.dataset_receipt_sha256, name="dataset_receipt_sha256"
            ),
        )
        if self.qualification not in {
            FORMAL_STATISTICS_QUALIFICATION,
            EXPLICIT_EPISODE_STATISTICS_QUALIFICATION,
        }:
            raise ValueError(
                f"Unsupported statistics qualification {self.qualification!r}."
            )
        if (
            not math.isfinite(float(self.scale_floor))
            or float(self.scale_floor) <= 0
        ):
            raise ValueError("`scale_floor` must be finite and positive.")

        semantics = _plain_json_value(
            self.feature_semantics, name="feature_semantics"
        )
        if not isinstance(semantics, dict):
            raise TypeError("`feature_semantics` must be a mapping.")
        contract = self.normalization_contract
        expected_semantics = {
            "feature_kind": contract.feature_kind,
            "feature_dim": contract.feature_dim,
            "semantic_kind": contract.semantic_kind,
            "source_normalization": contract.source_normalization,
        }
        for key, expected in expected_semantics.items():
            if semantics.get(key) != expected:
                raise ValueError(
                    f"feature_semantics.{key}={semantics.get(key)!r}, "
                    f"expected {expected!r}."
                )
        validity = semantics.get("validity")
        if not isinstance(validity, str) or not validity.strip():
            raise ValueError("feature_semantics.validity must be non-empty.")
        object.__setattr__(self, "feature_semantics", semantics)

        receipt = _plain_json_value(
            self.accumulator_receipt, name="accumulator_receipt"
        )
        if not isinstance(receipt, dict):
            raise TypeError("`accumulator_receipt` must be a mapping.")
        validate_feature_accumulator_receipt(
            receipt, expected_feature_kind=contract.feature_kind
        )
        object.__setattr__(self, "accumulator_receipt", receipt)

        computed = self.recompute_digest()
        if self.artifact_digest is not None and self.artifact_digest != computed:
            raise ValueError(
                "Bound feature statistics digest mismatch: supplied "
                f"{self.artifact_digest!r}, recomputed {computed!r}."
            )
        object.__setattr__(self, "artifact_digest", computed)

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "formal_schema_version": self.formal_schema_version,
            "qualification": self.qualification,
            "split_name": self.split_name,
            "split_manifest_sha256": self.split_manifest_sha256,
            "dataset_receipt_sha256": self.dataset_receipt_sha256,
            "feature_semantics": dict(self.feature_semantics),
            "accumulator_receipt": dict(self.accumulator_receipt),
            "scale_floor": float(self.scale_floor),
            "normalization_contract": self.normalization_contract.as_dict(),
        }

    def recompute_digest(self) -> str:
        return _digest_payload(self._digest_payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._digest_payload(), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "BoundFeatureStatisticsArtifact":
        if not isinstance(payload, Mapping):
            raise TypeError("Bound feature statistics payload must be a mapping.")
        required = {
            "schema_version",
            "formal_schema_version",
            "qualification",
            "split_name",
            "split_manifest_sha256",
            "dataset_receipt_sha256",
            "feature_semantics",
            "accumulator_receipt",
            "scale_floor",
            "normalization_contract",
            "artifact_digest",
        }
        missing = sorted(required - set(payload))
        unexpected = sorted(set(payload) - required)
        if missing or unexpected:
            raise ValueError(
                "Bound feature artifact key mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        raw_contract = payload["normalization_contract"]
        if not isinstance(raw_contract, Mapping):
            raise TypeError("normalization_contract must be a mapping.")
        contract_kwargs = dict(raw_contract)
        feature_dim = contract_kwargs.pop("feature_dim", FIXED_FEATURE_DIM)
        if feature_dim != FIXED_FEATURE_DIM:
            raise ValueError("normalization_contract feature_dim must be 14.")
        contract = FixedFeatureNormalizationContract(**contract_kwargs)
        return cls(
            normalization_contract=contract,
            split_name=str(payload["split_name"]),
            split_manifest_sha256=str(payload["split_manifest_sha256"]),
            dataset_receipt_sha256=str(payload["dataset_receipt_sha256"]),
            feature_semantics=payload["feature_semantics"],
            accumulator_receipt=payload["accumulator_receipt"],
            qualification=str(payload["qualification"]),
            scale_floor=float(payload["scale_floor"]),
            schema_version=str(payload["schema_version"]),
            formal_schema_version=str(payload["formal_schema_version"]),
            artifact_digest=str(payload["artifact_digest"]),
        )

    @classmethod
    def load_json(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_feature_kind: str | None = None,
        expected_split_manifest_sha256: str | None = None,
        expected_dataset_receipt_sha256: str | None = None,
        require_formal: bool = False,
    ) -> "BoundFeatureStatisticsArtifact":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact = cls.from_dict(payload)
        if (
            expected_feature_kind is not None
            and artifact.normalization_contract.feature_kind
            != expected_feature_kind
        ):
            raise ValueError("Bound feature statistics feature_kind mismatch.")
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
                raise ValueError(f"Bound feature statistics {name} mismatch.")
        if (
            require_formal
            and artifact.qualification != FORMAL_STATISTICS_QUALIFICATION
        ):
            raise ValueError("Feature statistics are not Formal-qualified.")
        return artifact

    def dump_json(self, path: str | os.PathLike[str]) -> None:
        _atomic_write_json(Path(path), self.as_dict())


__all__ = [
    "ACCUMULATOR_DTYPE",
    "BOUND_FEATURE_STATISTICS_SCHEMA_VERSION",
    "EXPLICIT_EPISODE_STATISTICS_QUALIFICATION",
    "FEATURE_ARTIFACT_SCHEMA_VERSION",
    "FEATURE_STATISTICS_REDUCTION",
    "FORMAL_STATISTICS_QUALIFICATION",
    "BoundFeatureStatisticsArtifact",
    "validate_feature_accumulator_receipt",
]
