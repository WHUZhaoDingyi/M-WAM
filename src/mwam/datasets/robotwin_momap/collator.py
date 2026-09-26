"""Strict fixed-shape collator for MoMap training samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .schema import (
    FINAL_PERMITTED_KEYS,
    FINAL_REQUIRED_KEYS,
    METADATA_KEY,
    OPTIONAL_TENSOR_SAMPLE_KEYS,
    TENSOR_SAMPLE_KEYS,
    validate_training_sample,
)


class MoMapCollator:
    """Validate and stack samples without changing their semantics.

    Padding, normalization, target construction, camera concatenation, and
    temporal resampling do not belong in this collator.  validate_samples
    can be disabled only when the producer has already run the full
    validate_training_sample contract (the production MoMap Dataset does
    this in __getitem__); shape, dtype, key, and cross-sample contract
    checks remain active in either mode.
    """

    def __init__(self, *, validate_samples: bool = True) -> None:
        if not isinstance(validate_samples, bool):
            raise TypeError("validate_samples must be a boolean")
        self.validate_samples = validate_samples

    _CONTRACT_FIELDS = (
        "coordinate_frame",
        "action_sampling_mode",
        "action_rate_hz",
        "action_contract_schema_version",
        "action_label_kind",
        "action_is_high_level_policy_action",
        "action_normalization",
        "proprio_mode",
        "proprio_normalization",
        "momap_normalization",
        "source_condition_schema_version",
        "source_condition_reference_camera",
        "source_condition_grid_size",
        "source_seg_encoding",
        "source_seg_color_seed",
        "source_depth_semantics",
    )

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, object]:
        if not samples:
            raise ValueError("cannot collate an empty MoMap batch")

        if self.validate_samples:
            validated = [validate_training_sample(sample) for sample in samples]
        else:
            validated = []
            for sample in samples:
                if not isinstance(sample, Mapping):
                    raise TypeError(
                        "fast MoMap collator expects mapping samples"
                    )
                actual_keys = set(sample)
                if (
                    not FINAL_REQUIRED_KEYS <= actual_keys
                    or not actual_keys <= FINAL_PERMITTED_KEYS
                ):
                    raise ValueError(
                        "fast MoMap collator received an invalid sample key set"
                    )
                if not isinstance(sample[METADATA_KEY], Mapping):
                    raise TypeError(
                        "fast MoMap collator metadata must be a mapping"
                    )
                for key in (
                    *TENSOR_SAMPLE_KEYS,
                    *(key for key in OPTIONAL_TENSOR_SAMPLE_KEYS if key in sample),
                ):
                    if not isinstance(sample[key], torch.Tensor):
                        raise TypeError(
                            f"fast MoMap collator field {key!r} must be a tensor"
                        )
                validated.append(sample)
        reference = validated[0]
        reference_keys = set(reference)
        tensor_keys = (
            *TENSOR_SAMPLE_KEYS,
            *(key for key in OPTIONAL_TENSOR_SAMPLE_KEYS if key in reference),
        )
        reference_metadata = reference[METADATA_KEY]
        reference_cameras = tuple(reference_metadata["camera_names"])
        reference_contract = {
            field: reference_metadata[field] for field in self._CONTRACT_FIELDS
        }
        reference_shapes = {
            key: tuple(reference[key].shape) for key in tensor_keys
        }
        reference_dtypes = {key: reference[key].dtype for key in tensor_keys}

        for sample_index, sample in enumerate(validated[1:], start=1):
            if set(sample) != reference_keys:
                raise ValueError(
                    "optional sample fields mismatch within batch: "
                    f"sample 0={sorted(reference_keys)}, "
                    f"sample {sample_index}={sorted(sample)}"
                )
            metadata = sample[METADATA_KEY]
            camera_names = tuple(metadata["camera_names"])
            if camera_names != reference_cameras:
                raise ValueError(
                    "camera_names mismatch within batch: "
                    f"sample 0={reference_cameras}, sample {sample_index}={camera_names}"
                )
            for field, expected in reference_contract.items():
                actual = metadata[field]
                if actual != expected:
                    raise ValueError(
                        f"{field} mismatch within batch: "
                        f"sample 0={expected!r}, "
                        f"sample {sample_index}={actual!r}"
                    )
            for key, expected_shape in reference_shapes.items():
                actual_shape = tuple(sample[key].shape)
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"{key} shape mismatch within batch: "
                        f"sample 0={expected_shape}, "
                        f"sample {sample_index}={actual_shape}"
                    )
                actual_dtype = sample[key].dtype
                expected_dtype = reference_dtypes[key]
                if actual_dtype != expected_dtype:
                    raise TypeError(
                        f"{key} dtype mismatch within batch: "
                        f"sample 0={expected_dtype}, "
                        f"sample {sample_index}={actual_dtype}"
                    )

        batch: dict[str, object] = {
            key: torch.stack([sample[key] for sample in validated], dim=0)
            for key in tensor_keys
        }
        batch[METADATA_KEY] = [dict(sample[METADATA_KEY]) for sample in validated]
        if set(batch) != reference_keys:
            raise RuntimeError("collator produced an unexpected batch schema")
        return batch
