import hashlib
import logging
import json
import inspect
import os
import random
from collections.abc import Mapping
from math import ceil
from numbers import Real
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)

from .evaluation.contracts import LEGACY_VIDEO_PROMPT_EVALUATOR
from .training import checkpoint_state
from .training.adaptive_replay import (
    initialize_adaptive_replay_state,
    is_same_mode_adaptive_replay_resume,
    validate_adaptive_replay_training_config,
)
from .training.resume_contract import (
    RESUME_CONTRACT_SCHEMA_VERSION,
    normalize_saved_resume_contract,
    operational_sampler_resume_state,
)
from .training.runtime_topology import (
    RUNTIME_EXECUTION_SCHEMA_VERSION,
    RUNTIME_TOPOLOGY_SCHEMA_VERSION,
    all_gather_rank_objects,
    discover_runtime_execution,
    discover_runtime_topology,
    validated_runtime_topology,
)
from .training.sampler_loader import (
    ALL_WINDOWS_PER_EPOCH,
    FAILURE_REPLAY_80PCT_INITIAL10,
    FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL,
)
from .training.strategies import TrainerStrategies
from .utils.deepspeed_zero2_graph_task_cache import (
    configure_deepspeed_zero2_graph_task_cache,
)
from .utils.fs import ensure_dir
from .utils.jsonl_metrics import TrainingMetricsJsonlWriter
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import (
    INITIAL_WINDOW_10PCT_REMAINING_90PCT,
    LEGACY_UNPADDED_TAIL_POLICY,
    ONE_RANDOM_WINDOW_PER_EPISODE_PER_EPOCH,
)
from .utils.state_directory_manifest import (
    CHECKPOINT_RANK_STATE_SCHEMA_VERSION,
    audit_state_directory,
)
from .utils.video_io import save_mp4

logger = get_logger(__name__)


OPTIMIZER_UPDATE_COUNTERS_SCHEMA_VERSION = "fastwam_optimizer_update_counters_v1"
TRAINING_STAGES = frozenset({"action_only", "joint", "world_only"})
OPTIMIZER_PARAMETER_GROUP_NAMES = frozenset(
    {"action_adapters", "action_backbone", "action_io", "world"}
)
OPTIMIZER_PARAMETER_GROUP_ORDER = (
    "action_backbone",
    "action_adapters",
    "action_io",
    "world",
)
LOADED_RESUME_SOURCE_SCHEMA_VERSION = (
    checkpoint_state.LOADED_RESUME_SOURCE_SCHEMA_VERSION
)
CONTROLLED_SAMPLER_MIGRATION_SCHEMA_VERSION = (
    "fastwam_controlled_sampler_migration_v1"
)
CONTROLLED_SOURCE_INPUT_MIGRATION_SCHEMA_VERSION = (
    "fastwam_controlled_source_input_migration_v1"
)
CONTROLLED_SOURCE_INPUT_REPLACEMENT_MIGRATION_SCHEMA_VERSION = (
    "fastwam_controlled_source_input_migration_v2"
)
CONTROLLED_MODEL_MATH_MIGRATION_SCHEMA_VERSION = (
    "fastwam_controlled_model_math_migration_v1"
)
CONTROLLED_MODEL_MATH_MIGRATION_TRANSITION_ID = (
    "v8_native_kv_log_count_corrected_to_fastwam_direct_concat_cosine500_v1"
)
CONTROLLED_MODEL_MATH_SOURCE_NATIVE_KV_FUSION = (
    "world_then_action_token_concat_log_count_corrected_v1"
)
CONTROLLED_MODEL_MATH_TARGET_NATIVE_KV_FUSION = (
    "world_then_action_token_concat_log_count_cosine_anneal_to_none_v1"
)
CONTROLLED_MODEL_MATH_SOURCE_BRIDGE_SEMANTICS = (
    "world_global26_then_pit4_native_kv_direct_concat_action_self_kv_"
    "log14_over_valid_world_token_count_v1"
)
CONTROLLED_MODEL_MATH_TARGET_BRIDGE_SEMANTICS = (
    "world_global26_then_pit4_native_kv_direct_concat_action_self_kv_"
    "log14_over_valid_world_token_count_cosine_anneal_to_none_v1"
)
CONTROLLED_MODEL_MATH_SOURCE_METHOD_FIELDS = {
    "schema_version": "fastwam_momap_v2_method_contract_v8",
    "bridge": (
        "global_native_kv_0_25_then_pit_native_kv_0_3_concat_"
        "action_self_kv_equal_width_v8"
    ),
    "world_key_logit_correction": (
        "per_sample_log_14_over_valid_world_key_count_v1"
    ),
}
CONTROLLED_MODEL_MATH_TARGET_METHOD_FIELDS = {
    "schema_version": "fastwam_momap_v2_method_contract_v9",
    "bridge": (
        "global_native_kv_0_25_then_pit_native_kv_0_3_concat_"
        "action_self_kv_equal_width_v9"
    ),
    "world_key_logit_correction": (
        "per_sample_log_14_over_valid_world_key_count_cosine_anneal_to_none_"
        "500_optimizer_steps_v1"
    ),
}
CONTROLLED_MODEL_MATH_TRANSITION_SCHEMA_VERSION = (
    "fastwam_native_kv_count_correction_transition_v1"
)
CONTROLLED_MODEL_MATH_TRANSITION_STEPS = 500
CONTROLLED_MODEL_MATH_TRANSITION_CURVE = "cosine"
_CONTROLLED_MODEL_MATH_CURSOR_POLICY = (
    "preserve_epoch_batch_optimizer_boundary"
)
CONTROLLED_SCHEDULER_FLOOR_MIGRATION_SCHEMA_VERSION = (
    "fastwam_controlled_scheduler_floor_migration_v1"
)
_CONTROLLED_SCHEDULER_FLOOR_SOURCE = 1.0e-5
_CONTROLLED_SCHEDULER_FLOOR_TARGET = 1.0e-6
_CONTROLLED_SCHEDULER_FLOOR_CURSOR_POLICY = (
    "preserve_optimizer_scheduler_rng_sampler_cursor"
)


def _canonical_mapping_sha256(value: Mapping[str, object]) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validated_lowercase_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be lowercase SHA256")
    return value


def _validated_warmup_ratio(value: object) -> float:
    """Return one finite scheduler warmup fraction without changing defaults."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("warmup_ratio must be a real number, not bool")
    ratio = float(value)
    if not np.isfinite(ratio) or ratio < 0.0 or ratio >= 1.0:
        raise ValueError("warmup_ratio must be finite and lie in [0, 1)")
    return ratio


def _validated_cosine_min_learning_rate(
    value: object,
    *,
    learning_rate: float,
) -> float | None:
    """Validate an optional absolute cosine learning-rate floor."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            "cosine_min_learning_rate must be a real number, not bool"
        )
    minimum = float(value)
    if not np.isfinite(minimum) or minimum < 0.0:
        raise ValueError(
            "cosine_min_learning_rate must be finite and non-negative"
        )
    if minimum > learning_rate:
        raise ValueError(
            "cosine_min_learning_rate must not exceed learning_rate"
        )
    return minimum


def _validated_debug_optimizer_audit_metrics(
    value: object,
) -> dict[str, float]:
    """Canonicalize optional debug-only post-optimizer scalar evidence."""

    if not isinstance(value, Mapping):
        raise TypeError("debug optimizer audit metrics must be a mapping")
    result: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(
                "debug optimizer audit metric keys must be non-empty strings"
            )
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError(
                f"debug optimizer audit metric {key!r} must be a real scalar"
            )
        scalar = float(raw)
        if not np.isfinite(scalar):
            raise FloatingPointError(
                f"debug optimizer audit metric {key!r} is non-finite"
            )
        result[key] = scalar
    return dict(sorted(result.items()))


def _validated_optimizer_parameter_groups(
    value: object,
    *,
    default_cosine_min_learning_rate: float | None,
) -> dict[str, dict[str, float]] | None:
    """Validate optional named optimizer groups without changing legacy runs."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("optimizer_parameter_groups must be a mapping or null")
    if not value:
        raise ValueError("optimizer_parameter_groups must not be empty")

    unknown_groups = sorted(set(value).difference(OPTIMIZER_PARAMETER_GROUP_NAMES))
    if unknown_groups:
        raise ValueError(
            "optimizer_parameter_groups contains unsupported groups: "
            f"{unknown_groups}; expected a subset of "
            f"{sorted(OPTIMIZER_PARAMETER_GROUP_NAMES)}"
        )

    validated: dict[str, dict[str, float]] = {}
    allowed_fields = {
        "learning_rate",
        "cosine_min_learning_rate",
        "warmup_start_learning_rate",
    }
    for group_name in sorted(value):
        raw_group = value[group_name]
        if not isinstance(raw_group, Mapping):
            raise TypeError(
                f"optimizer_parameter_groups.{group_name} must be a mapping"
            )
        unknown_fields = sorted(set(raw_group).difference(allowed_fields))
        if unknown_fields:
            raise ValueError(
                f"optimizer_parameter_groups.{group_name} contains unsupported "
                f"fields: {unknown_fields}"
            )
        if "learning_rate" not in raw_group:
            raise ValueError(
                f"optimizer_parameter_groups.{group_name}.learning_rate is required"
            )
        raw_peak = raw_group["learning_rate"]
        if isinstance(raw_peak, bool) or not isinstance(raw_peak, Real):
            raise TypeError(
                f"optimizer_parameter_groups.{group_name}.learning_rate must be "
                "a real number, not bool"
            )
        peak = float(raw_peak)
        if not np.isfinite(peak) or peak <= 0.0:
            raise ValueError(
                f"optimizer_parameter_groups.{group_name}.learning_rate must be "
                "finite and positive"
            )

        raw_floor = raw_group.get(
            "cosine_min_learning_rate",
            default_cosine_min_learning_rate,
        )
        floor = peak * 0.01 if raw_floor is None else raw_floor
        if isinstance(floor, bool) or not isinstance(floor, Real):
            raise TypeError(
                "optimizer_parameter_groups."
                f"{group_name}.cosine_min_learning_rate must be a real number, "
                "not bool"
            )
        floor = float(floor)
        if not np.isfinite(floor) or floor < 0.0 or floor > peak:
            raise ValueError(
                "optimizer_parameter_groups."
                f"{group_name}.cosine_min_learning_rate must be finite and lie "
                "in [0, learning_rate]"
            )

        raw_start = raw_group.get("warmup_start_learning_rate", floor)
        if isinstance(raw_start, bool) or not isinstance(raw_start, Real):
            raise TypeError(
                "optimizer_parameter_groups."
                f"{group_name}.warmup_start_learning_rate must be a real number, "
                "not bool"
            )
        start = float(raw_start)
        if not np.isfinite(start) or start < 0.0 or start > peak:
            raise ValueError(
                "optimizer_parameter_groups."
                f"{group_name}.warmup_start_learning_rate must be finite and lie "
                "in [0, learning_rate]"
            )
        validated[group_name] = {
            "learning_rate": peak,
            "cosine_min_learning_rate": floor,
            "warmup_start_learning_rate": start,
        }
    return validated


def _validated_controlled_sampler_migration(
    value: object,
) -> dict[str, object] | None:
    """Accept only the explicitly approved I5 sampler transition.

    Full-state resume normally requires an identical sampler stream.  This
    narrow opt-in is the sole exception: it switches the frozen I5 replay
    stream from the historical task-merged/initial-window implementation to
    the versioned task×variant/no-initial-window implementation while every
    model, optimizer, scheduler, topology, and data-math contract stays
    identical.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("controlled_sampler_migration must be a mapping or null")
    required = {
        "schema_version",
        "enabled",
        "source_state_dir",
        "source_global_step",
        "source_world_size",
        "source_sampling_mode",
        "target_sampling_mode",
        "source_failure_manifest_sha256",
        "target_failure_manifest_sha256",
        "cursor_policy",
    }
    if set(value) != required:
        raise ValueError(
            "controlled_sampler_migration fields drifted: "
            f"missing={sorted(required - set(value))}, "
            f"unexpected={sorted(set(value) - required)}"
        )
    if value["schema_version"] != CONTROLLED_SAMPLER_MIGRATION_SCHEMA_VERSION:
        raise ValueError("controlled_sampler_migration schema is unsupported")
    if value["enabled"] is not True:
        raise ValueError("controlled_sampler_migration.enabled must be true")
    source_state_dir = value["source_state_dir"]
    if not isinstance(source_state_dir, str) or not source_state_dir.strip():
        raise ValueError(
            "controlled_sampler_migration.source_state_dir must be non-empty text"
        )
    source_path = Path(source_state_dir).expanduser()
    if not source_path.is_absolute():
        raise ValueError(
            "controlled_sampler_migration.source_state_dir must be absolute"
        )
    integer_fields = ("source_global_step", "source_world_size")
    normalized: dict[str, object] = {
        "schema_version": CONTROLLED_SAMPLER_MIGRATION_SCHEMA_VERSION,
        "enabled": True,
        "source_state_dir": str(source_path),
    }
    for field in integer_fields:
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(
                f"controlled_sampler_migration.{field} must be positive int"
            )
        normalized[field] = int(raw)
    if value["source_sampling_mode"] != FAILURE_REPLAY_80PCT_INITIAL10:
        raise ValueError(
            "controlled_sampler_migration source mode must be "
            "failure_replay_80pct_initial10"
        )
    if (
        value["target_sampling_mode"]
        != FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL
    ):
        raise ValueError(
            "controlled_sampler_migration target mode must be "
            "failure_replay_80pct_task_variant_noinitial"
        )
    normalized["source_sampling_mode"] = FAILURE_REPLAY_80PCT_INITIAL10
    normalized["target_sampling_mode"] = (
        FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL
    )
    for field in (
        "source_failure_manifest_sha256",
        "target_failure_manifest_sha256",
    ):
        raw = value[field]
        if (
            not isinstance(raw, str)
            or len(raw) != 64
            or any(character not in "0123456789abcdef" for character in raw)
        ):
            raise ValueError(
                f"controlled_sampler_migration.{field} must be lowercase SHA256"
            )
        normalized[field] = raw
    if value["cursor_policy"] != "preserve_epoch_batch_optimizer_boundary":
        raise ValueError(
            "controlled_sampler_migration cursor_policy must preserve the "
            "saved epoch/batch optimizer boundary"
        )
    normalized["cursor_policy"] = "preserve_epoch_batch_optimizer_boundary"
    return normalized


def _validated_controlled_source_input_migration(
    value: object,
) -> dict[str, object] | None:
    """Validate a full-state transition that changes only source inputs."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(
            "controlled_source_input_migration must be a mapping or null"
        )
    schema_version = value.get("schema_version")
    common = {
        "schema_version",
        "enabled",
        "source_state_dir",
        "source_global_step",
        "cursor_policy",
    }
    if schema_version == CONTROLLED_SOURCE_INPUT_MIGRATION_SCHEMA_VERSION:
        required = common | {
            "target_seg_policy_content_digest",
            "target_depth_noise_schema_version",
        }
    elif (
        schema_version
        == CONTROLLED_SOURCE_INPUT_REPLACEMENT_MIGRATION_SCHEMA_VERSION
    ):
        required = common | {
            "source_seg_policy_content_digest",
            "target_seg_policy_content_digest",
            "source_depth_noise_schema_version",
            "target_depth_noise_schema_version",
            "source_augmentation_contract_sha256",
            "target_augmentation_contract_sha256",
        }
    else:
        raise ValueError(
            "controlled_source_input_migration schema is unsupported"
        )
    if set(value) != required:
        raise ValueError(
            "controlled_source_input_migration fields drifted: "
            f"missing={sorted(required - set(value))}, "
            f"unexpected={sorted(set(value) - required)}"
        )
    if value["enabled"] is not True:
        raise ValueError(
            "controlled_source_input_migration.enabled must be true"
        )
    source_state_dir = value["source_state_dir"]
    if not isinstance(source_state_dir, str) or not source_state_dir.strip():
        raise ValueError(
            "controlled_source_input_migration.source_state_dir must be "
            "non-empty text"
        )
    source_path = Path(source_state_dir).expanduser()
    if not source_path.is_absolute():
        raise ValueError(
            "controlled_source_input_migration.source_state_dir must be absolute"
        )
    source_global_step = value["source_global_step"]
    if (
        isinstance(source_global_step, bool)
        or not isinstance(source_global_step, int)
        or source_global_step <= 0
    ):
        raise ValueError(
            "controlled_source_input_migration.source_global_step must be "
            "positive int"
        )
    normalized_specific: dict[str, object] = {}
    seg_digest_fields = ["target_seg_policy_content_digest"]
    depth_schema_fields = ["target_depth_noise_schema_version"]
    if (
        schema_version
        == CONTROLLED_SOURCE_INPUT_REPLACEMENT_MIGRATION_SCHEMA_VERSION
    ):
        seg_digest_fields.insert(0, "source_seg_policy_content_digest")
        depth_schema_fields.insert(0, "source_depth_noise_schema_version")
        for field in (
            "source_augmentation_contract_sha256",
            "target_augmentation_contract_sha256",
        ):
            normalized_specific[field] = _validated_lowercase_sha256(
                value[field],
                field=f"controlled_source_input_migration.{field}",
            )
    for field in seg_digest_fields:
        normalized_specific[field] = _validated_lowercase_sha256(
            value[field],
            field=f"controlled_source_input_migration.{field}",
        )
    for field in depth_schema_fields:
        depth_schema = value[field]
        if not isinstance(depth_schema, str) or not depth_schema.strip():
            raise ValueError(
                f"controlled_source_input_migration.{field} must be "
                "non-empty text"
            )
        normalized_specific[field] = depth_schema
    if value["cursor_policy"] != "preserve_epoch_batch_optimizer_boundary":
        raise ValueError(
            "controlled_source_input_migration cursor_policy must preserve "
            "the saved epoch/batch optimizer boundary"
        )
    return {
        "schema_version": schema_version,
        "enabled": True,
        "source_state_dir": str(source_path),
        "source_global_step": int(source_global_step),
        **normalized_specific,
        "cursor_policy": "preserve_epoch_batch_optimizer_boundary",
    }


def _validated_controlled_scheduler_floor_migration(
    value: object,
) -> dict[str, object] | None:
    """Validate the one exact 1e-5 to 1e-6 full-state LR migration."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(
            "controlled_scheduler_floor_migration must be a mapping or null"
        )
    required = {
        "schema_version",
        "enabled",
        "source_state_dir",
        "source_global_step",
        "source_cosine_min_learning_rate",
        "target_cosine_min_learning_rate",
        "source_warmup_start_learning_rate",
        "target_warmup_start_learning_rate",
        "cursor_policy",
    }
    if set(value) != required:
        raise ValueError(
            "controlled_scheduler_floor_migration fields drifted: "
            f"missing={sorted(required - set(value))}, "
            f"unexpected={sorted(set(value) - required)}"
        )
    if (
        value["schema_version"]
        != CONTROLLED_SCHEDULER_FLOOR_MIGRATION_SCHEMA_VERSION
        or value["enabled"] is not True
    ):
        raise ValueError(
            "controlled_scheduler_floor_migration schema/enabled is invalid"
        )
    source_path = Path(str(value["source_state_dir"])).expanduser()
    if not source_path.is_absolute():
        raise ValueError(
            "controlled_scheduler_floor_migration.source_state_dir must be absolute"
        )
    source_step = value["source_global_step"]
    if (
        isinstance(source_step, bool)
        or not isinstance(source_step, int)
        or source_step <= 0
    ):
        raise ValueError(
            "controlled_scheduler_floor_migration.source_global_step must be positive int"
        )
    expected_values = {
        "source_cosine_min_learning_rate": _CONTROLLED_SCHEDULER_FLOOR_SOURCE,
        "target_cosine_min_learning_rate": _CONTROLLED_SCHEDULER_FLOOR_TARGET,
        "source_warmup_start_learning_rate": _CONTROLLED_SCHEDULER_FLOOR_SOURCE,
        "target_warmup_start_learning_rate": _CONTROLLED_SCHEDULER_FLOOR_TARGET,
    }
    for field, expected in expected_values.items():
        raw = value[field]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, Real)
            or not np.isfinite(float(raw))
            or float(raw) != expected
        ):
            raise ValueError(
                f"controlled_scheduler_floor_migration.{field} must equal {expected}"
            )
    if value["cursor_policy"] != _CONTROLLED_SCHEDULER_FLOOR_CURSOR_POLICY:
        raise ValueError(
            "controlled_scheduler_floor_migration cursor policy is unsupported"
        )
    return {
        **dict(value),
        "source_state_dir": str(source_path),
        "source_global_step": int(source_step),
        **expected_values,
    }


def _validated_controlled_model_math_migration(
    value: object,
) -> dict[str, object] | None:
    """Validate the one approved V8 count-correction transition.

    This is deliberately not a generic model-math override. The source step is
    omitted from configuration on purpose: it is derived from the audited
    source-state metadata and trainer state at load time, then becomes the
    deterministic origin of the 500-step cosine schedule.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(
            "controlled_model_math_migration must be a mapping or null"
        )
    required = {
        "schema_version",
        "enabled",
        "transition_id",
        "source_state_dir",
        "source_native_kv_fusion",
        "target_native_kv_fusion",
        "transition_steps",
        "transition_curve",
        "cursor_policy",
    }
    if set(value) != required:
        raise ValueError(
            "controlled_model_math_migration fields drifted: "
            f"missing={sorted(required - set(value))}, "
            f"unexpected={sorted(set(value) - required)}"
        )
    if value["schema_version"] != CONTROLLED_MODEL_MATH_MIGRATION_SCHEMA_VERSION:
        raise ValueError("controlled_model_math_migration schema is unsupported")
    if value["enabled"] is not True:
        raise ValueError("controlled_model_math_migration.enabled must be true")
    if value["transition_id"] != CONTROLLED_MODEL_MATH_MIGRATION_TRANSITION_ID:
        raise ValueError(
            "controlled_model_math_migration transition_id is unsupported"
        )
    source_state_dir = value["source_state_dir"]
    if not isinstance(source_state_dir, str) or not source_state_dir.strip():
        raise ValueError(
            "controlled_model_math_migration.source_state_dir must be "
            "non-empty text"
        )
    source_path = Path(source_state_dir).expanduser()
    if not source_path.is_absolute():
        raise ValueError(
            "controlled_model_math_migration.source_state_dir must be absolute"
        )
    if (
        value["source_native_kv_fusion"]
        != CONTROLLED_MODEL_MATH_SOURCE_NATIVE_KV_FUSION
    ):
        raise ValueError(
            "controlled_model_math_migration source_native_kv_fusion must be "
            "the V8 log-count-corrected fusion"
        )
    if (
        value["target_native_kv_fusion"]
        != CONTROLLED_MODEL_MATH_TARGET_NATIVE_KV_FUSION
    ):
        raise ValueError(
            "controlled_model_math_migration target_native_kv_fusion must be "
            "the V8 cosine-anneal-to-none fusion"
        )
    if value["transition_steps"] != CONTROLLED_MODEL_MATH_TRANSITION_STEPS:
        raise ValueError(
            "controlled_model_math_migration.transition_steps must be exactly "
            f"{CONTROLLED_MODEL_MATH_TRANSITION_STEPS}"
        )
    if value["transition_curve"] != CONTROLLED_MODEL_MATH_TRANSITION_CURVE:
        raise ValueError(
            "controlled_model_math_migration.transition_curve must be cosine"
        )
    if value["cursor_policy"] != _CONTROLLED_MODEL_MATH_CURSOR_POLICY:
        raise ValueError(
            "controlled_model_math_migration.cursor_policy must preserve the "
            "saved epoch/batch optimizer boundary"
        )
    return {
        "schema_version": CONTROLLED_MODEL_MATH_MIGRATION_SCHEMA_VERSION,
        "enabled": True,
        "transition_id": CONTROLLED_MODEL_MATH_MIGRATION_TRANSITION_ID,
        "source_state_dir": str(source_path),
        "source_native_kv_fusion": (
            CONTROLLED_MODEL_MATH_SOURCE_NATIVE_KV_FUSION
        ),
        "target_native_kv_fusion": (
            CONTROLLED_MODEL_MATH_TARGET_NATIVE_KV_FUSION
        ),
        "transition_steps": CONTROLLED_MODEL_MATH_TRANSITION_STEPS,
        "transition_curve": CONTROLLED_MODEL_MATH_TRANSITION_CURVE,
        "cursor_policy": _CONTROLLED_MODEL_MATH_CURSOR_POLICY,
    }


class Wan22Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        *,
        cfg: DictConfig,
        strategies: TrainerStrategies | None = None,
    ):
        self.strategies = strategies or TrainerStrategies()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.adam_betas = (0.9, 0.95)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        prefetch_factor = cfg.get("prefetch_factor", None)
        if prefetch_factor is not None:
            if isinstance(prefetch_factor, bool) or not isinstance(
                prefetch_factor,
                int,
            ):
                raise ValueError("prefetch_factor must be a positive integer or null")
            if prefetch_factor <= 0:
                raise ValueError("prefetch_factor must be a positive integer or null")
            if self.num_workers <= 0:
                raise ValueError(
                    "prefetch_factor requires num_workers to be positive"
                )
        self.prefetch_factor = prefetch_factor
        dataloader_non_blocking = cfg.get("dataloader_non_blocking", False)
        if not isinstance(dataloader_non_blocking, bool):
            raise ValueError("dataloader_non_blocking must be a boolean")
        self.dataloader_non_blocking = dataloader_non_blocking
        training_timing_enabled = cfg.get("training_timing_enabled", False)
        if not isinstance(training_timing_enabled, bool):
            raise ValueError("training_timing_enabled must be a boolean")
        self.training_timing_enabled = training_timing_enabled
        persistent_workers = cfg.get("persistent_workers", False)
        if not isinstance(persistent_workers, bool):
            raise ValueError("persistent_workers must be a boolean")
        self.persistent_workers = persistent_workers
        if self.persistent_workers and self.num_workers <= 0:
            raise ValueError(
                "persistent_workers=true requires num_workers to be positive"
            )
        self.num_epochs = int(cfg.num_epochs)
        self.lr_scheduler_type = str(cfg.lr_scheduler_type).strip().lower()
        self.cosine_min_learning_rate = (
            _validated_cosine_min_learning_rate(
                cfg.get("cosine_min_learning_rate", None),
                learning_rate=self.learning_rate,
            )
        )
        self.optimizer_parameter_groups = (
            _validated_optimizer_parameter_groups(
                cfg.get("optimizer_parameter_groups", None),
                default_cosine_min_learning_rate=(
                    self.cosine_min_learning_rate
                ),
            )
        )
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        stop_after_step = cfg.get("stop_after_step", None)
        self.stop_after_step = (
            int(stop_after_step) if stop_after_step is not None else None
        )
        checkpoint_on_stop_after_step = cfg.get(
            "checkpoint_on_stop_after_step",
            True,
        )
        if not isinstance(checkpoint_on_stop_after_step, bool):
            raise ValueError("checkpoint_on_stop_after_step must be a boolean")
        self.checkpoint_on_stop_after_step = checkpoint_on_stop_after_step
        runtime_checkpoint_control_enabled = cfg.get(
            "runtime_checkpoint_control_enabled",
            True,
        )
        if not isinstance(runtime_checkpoint_control_enabled, bool):
            raise ValueError(
                "runtime_checkpoint_control_enabled must be a boolean"
            )
        self.runtime_checkpoint_control_enabled = (
            runtime_checkpoint_control_enabled
        )
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self._assert_in_trainer_evaluation_contract()
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.warmup_ratio = _validated_warmup_ratio(
            cfg.get("warmup_ratio", 0.05)
        )
        raw_cosine_progress_start_step = cfg.get(
            "cosine_progress_start_step",
            None,
        )
        if raw_cosine_progress_start_step is None:
            self.cosine_progress_start_step: int | None = None
        elif (
            isinstance(raw_cosine_progress_start_step, bool)
            or not isinstance(raw_cosine_progress_start_step, int)
            or raw_cosine_progress_start_step < 0
        ):
            raise ValueError(
                "cosine_progress_start_step must be a non-negative integer "
                "or null"
            )
        else:
            self.cosine_progress_start_step = int(
                raw_cosine_progress_start_step
            )
        self.seed = int(cfg.seed)
        raw_training_stage = cfg.get("training_stage", "joint")
        if not isinstance(raw_training_stage, str):
            raise TypeError("training_stage must be a string")
        self.training_stage = raw_training_stage.strip().lower()
        if self.training_stage not in TRAINING_STAGES:
            raise ValueError(
                "training_stage must be one of "
                f"{sorted(TRAINING_STAGES)}, got {raw_training_stage!r}"
            )
        self.train_sampling_mode = str(
            cfg.get("train_sampling_mode", ALL_WINDOWS_PER_EPOCH)
        ).strip()
        if self.train_sampling_mode not in {
            ALL_WINDOWS_PER_EPOCH,
            FAILURE_REPLAY_80PCT_INITIAL10,
            FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL,
            INITIAL_WINDOW_10PCT_REMAINING_90PCT,
            ONE_RANDOM_WINDOW_PER_EPISODE_PER_EPOCH,
        }:
            raise ValueError(
                "Unsupported train_sampling_mode: "
                f"{self.train_sampling_mode!r}. Expected one of: "
                f"[{ALL_WINDOWS_PER_EPOCH!r}, "
                f"{FAILURE_REPLAY_80PCT_INITIAL10!r}, "
                f"{FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL!r}, "
                f"{INITIAL_WINDOW_10PCT_REMAINING_90PCT!r}, "
                f"{ONE_RANDOM_WINDOW_PER_EPISODE_PER_EPOCH!r}]."
            )
        strict_resumable_sampler = cfg.get(
            "strict_resumable_sampler",
            False,
        )
        if not isinstance(strict_resumable_sampler, bool):
            raise ValueError("strict_resumable_sampler must be a boolean")
        self.strict_resumable_sampler = strict_resumable_sampler
        if (
            self.train_sampling_mode
            in {
                FAILURE_REPLAY_80PCT_INITIAL10,
                FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL,
                INITIAL_WINDOW_10PCT_REMAINING_90PCT,
                ONE_RANDOM_WINDOW_PER_EPISODE_PER_EPOCH,
            }
            and not self.strict_resumable_sampler
        ):
            raise ValueError(
                "episode-aware sampling requires "
                "strict_resumable_sampler=true"
            )

        self.resume = cfg.resume
        self.adaptive_replay_config = (
            validate_adaptive_replay_training_config(
                cfg.get("adaptive_replay", None),
                sampling_mode=self.train_sampling_mode,
                repository_root=Path(__file__).resolve().parents[2],
            )
        )
        if (
            self.adaptive_replay_config is not None
            and self.adaptive_replay_config.checkpoint_interval
            != self.save_every
        ):
            raise ValueError(
                "adaptive replay checkpoint_interval must equal save_every"
            )
        raw_weights_only_warmstart = cfg.get(
            "weights_only_warmstart",
            None,
        )
        if raw_weights_only_warmstart is None:
            self.weights_only_warmstart = None
        elif not isinstance(raw_weights_only_warmstart, Mapping):
            raise TypeError("weights_only_warmstart must be a mapping or null")
        else:
            self.weights_only_warmstart = dict(raw_weights_only_warmstart)
        raw_fresh_stage_transition = cfg.get(
            "fresh_stage_weights_transition",
            None,
        )
        if raw_fresh_stage_transition is None:
            self.fresh_stage_weights_transition = None
        elif not isinstance(raw_fresh_stage_transition, Mapping):
            raise TypeError(
                "fresh_stage_weights_transition must be a mapping or null"
            )
        else:
            self.fresh_stage_weights_transition = dict(
                raw_fresh_stage_transition
            )
        self.controlled_sampler_migration = (
            _validated_controlled_sampler_migration(
                cfg.get("controlled_sampler_migration", None)
            )
        )
        if is_same_mode_adaptive_replay_resume(self):
            self.controlled_sampler_migration = None
        if (
            self.adaptive_replay_config is not None
            and self.controlled_sampler_migration is not None
            and self.controlled_sampler_migration["source_global_step"]
            != self.adaptive_replay_config.initial_manifest_source_step
        ):
            raise ValueError(
                "adaptive initial manifest step must equal controlled sampler "
                "migration source step"
            )
        self.controlled_source_input_migration = (
            _validated_controlled_source_input_migration(
                cfg.get("controlled_source_input_migration", None)
            )
        )
        self.controlled_scheduler_floor_migration = (
            _validated_controlled_scheduler_floor_migration(
                cfg.get("controlled_scheduler_floor_migration", None)
            )
        )
        self.controlled_model_math_migration = (
            _validated_controlled_model_math_migration(
                cfg.get("controlled_model_math_migration", None)
            )
        )
        if (
            self.weights_only_warmstart is not None
            and self.fresh_stage_weights_transition is not None
        ):
            raise ValueError(
                "weights_only_warmstart and fresh_stage_weights_transition "
                "are mutually exclusive"
            )
        if self.controlled_sampler_migration is not None:
            if (
                self.weights_only_warmstart is not None
                or self.fresh_stage_weights_transition is not None
            ):
                raise ValueError(
                    "controlled_sampler_migration requires a full-state "
                    "resume and is mutually exclusive with weights-only and "
                    "fresh-stage transitions"
                )
            if (
                self.train_sampling_mode
                != FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL
            ):
                raise ValueError(
                    "controlled_sampler_migration requires "
                    "failure_replay_80pct_task_variant_noinitial"
                )
            if not isinstance(self.resume, str) or not self.resume.strip():
                raise ValueError(
                    "controlled_sampler_migration requires a configured "
                    "full-state resume directory"
                )
        if self.controlled_source_input_migration is not None:
            if (
                self.weights_only_warmstart is not None
                or self.fresh_stage_weights_transition is not None
            ):
                raise ValueError(
                    "controlled_source_input_migration requires a full-state "
                    "resume and is mutually exclusive with weights-only and "
                    "fresh-stage transitions"
                )
            if not isinstance(self.resume, str) or not self.resume.strip():
                raise ValueError(
                    "controlled_source_input_migration requires a configured "
                    "full-state resume directory"
                )
        if self.controlled_scheduler_floor_migration is not None:
            if (
                self.weights_only_warmstart is not None
                or self.fresh_stage_weights_transition is not None
                or self.controlled_sampler_migration is not None
                or self.controlled_source_input_migration is not None
                or self.controlled_model_math_migration is not None
            ):
                raise ValueError(
                    "controlled_scheduler_floor_migration must be the sole "
                    "full-state resume-contract exception"
                )
            if not isinstance(self.resume, str) or not self.resume.strip():
                raise ValueError(
                    "controlled_scheduler_floor_migration requires a configured "
                    "full-state resume directory"
                )
        if self.controlled_model_math_migration is not None:
            if (
                self.weights_only_warmstart is not None
                or self.fresh_stage_weights_transition is not None
            ):
                raise ValueError(
                    "controlled_model_math_migration requires a full-state "
                    "resume and is mutually exclusive with weights-only and "
                    "fresh-stage transitions"
                )
            if not isinstance(self.resume, str) or not self.resume.strip():
                raise ValueError(
                    "controlled_model_math_migration requires a configured "
                    "full-state resume directory"
                )
            if (
                self.controlled_sampler_migration is not None
                or self.controlled_source_input_migration is not None
                or self.controlled_scheduler_floor_migration is not None
            ):
                raise ValueError(
                    "controlled_model_math_migration is intentionally the sole "
                    "resume-contract exception and cannot be combined with "
                    "sampler or source-input migration"
                )
        if (
            self.controlled_sampler_migration is not None
            and self.controlled_source_input_migration is not None
        ):
            sampler_migration = self.controlled_sampler_migration
            source_migration = self.controlled_source_input_migration
            for field in (
                "source_state_dir",
                "source_global_step",
                "cursor_policy",
            ):
                if sampler_migration.get(field) != source_migration.get(field):
                    raise ValueError(
                        "combined controlled migrations must share the same "
                        f"{field}"
                    )
        allow_budget_extension = cfg.get(
            "allow_constant_lr_max_steps_extension",
            False,
        )
        if not isinstance(allow_budget_extension, bool):
            raise ValueError(
                "allow_constant_lr_max_steps_extension must be a boolean"
            )
        self.allow_constant_lr_max_steps_extension = allow_budget_extension
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        raw_graph_cache_enabled = cfg.get(
            "deepspeed_zero2_graph_task_cache_enabled",
            False,
        )
        if not isinstance(raw_graph_cache_enabled, bool):
            raise ValueError(
                "deepspeed_zero2_graph_task_cache_enabled must be boolean"
            )
        self.deepspeed_zero2_graph_task_cache_enabled = raw_graph_cache_enabled

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
            dataloader_config=DataLoaderConfiguration(
                non_blocking=self.dataloader_non_blocking,
            ),
        )
        deepspeed_plugin = getattr(
            self.accelerator.state,
            "deepspeed_plugin",
            None,
        )
        deepspeed_config = getattr(
            deepspeed_plugin,
            "deepspeed_config",
            None,
        )
        zero_stage = None
        if isinstance(deepspeed_config, dict):
            zero_config = deepspeed_config.get("zero_optimization", {})
            if isinstance(zero_config, dict):
                zero_stage = zero_config.get("stage")
        self.deepspeed_zero2_graph_task_cache_report = (
            configure_deepspeed_zero2_graph_task_cache(
                enabled=self.deepspeed_zero2_graph_task_cache_enabled,
                distributed_type=self.accelerator.distributed_type,
                zero_stage=zero_stage,
            )
        )
        if self.deepspeed_zero2_graph_task_cache_report.enabled:
            logger.warning(
                "Enabled guarded DeepSpeed graph-task used-parameter cache: %s",
                self.deepspeed_zero2_graph_task_cache_report,
            )
        self._runtime_topology = self._discover_runtime_topology()
        self.runtime_execution = self._discover_runtime_execution()

        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            zero_stage if zero_stage is not None else "not-applicable",
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info(
            "Runtime topology: world_size=%d node_count=%d ranks_per_node=%s",
            self._runtime_topology["world_size"],
            self._runtime_topology["node_count"],
            self._runtime_topology["ranks_per_node"],
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        logger.info(
            "DataLoader runtime: workers=%d persistent=%s prefetch_factor=%s "
            "pin_memory=%s accelerate_non_blocking_h2d=%s timing=%s",
            self.num_workers,
            self.persistent_workers,
            self.prefetch_factor,
            torch.cuda.is_available(),
            self.dataloader_non_blocking,
            self.training_timing_enabled,
        )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Establish the loss graph and freeze policy before optimizer/ZeRO state
        # is constructed.  Stage 1 owns only the PixelDiT world path.
        stage_setter = getattr(self.model, "set_training_stage", None)
        if callable(stage_setter):
            stage_setter(self.training_stage)
        elif self.training_stage != "joint":
            raise TypeError(
                "non-joint training requires model.set_training_stage(...)"
            )
        self._apply_dit_only_train_mode(
            self.model,
            training_stage=self.training_stage,
        )
        trainable_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_params:
            raise RuntimeError(
                f"training_stage={self.training_stage!r} selected no parameters"
            )
        optimizer_params: list[torch.nn.Parameter] | list[dict]
        if self.optimizer_parameter_groups is None:
            optimizer_params = trainable_params
        else:
            optimizer_params = self._build_optimizer_parameter_groups(
                self.model,
                self.optimizer_parameter_groups,
            )
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.adam_betas,
        )

        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        self.adaptive_replay_state = initialize_adaptive_replay_state(self)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        from mwam.training.sample_progress_reset import validate_reset
        self.sample_progress_reset = validate_reset(
            self.cfg.get("sample_progress_reset", None), max_steps=total_train_steps,
            batch_size=self.batch_size, accumulation=self.gradient_accumulation_steps,
        )
        if self.sample_progress_reset is not None:
            if (self.lr_scheduler_type != "cosine" or self.warmup_ratio != 0.0
                    or self.optimizer_parameter_groups is None
                    or self.accelerator.num_processes != 32):
                raise ValueError("reset stage requires named cosine scheduler, warmup_ratio=0, world32")
        if self.cosine_progress_start_step is not None:
            if self.lr_scheduler_type != "cosine":
                raise ValueError(
                    "cosine_progress_start_step requires lr_scheduler_type=cosine"
                )
            if self.warmup_ratio != 0.0:
                raise ValueError(
                    "cosine_progress_start_step requires warmup_ratio=0"
                )
            if self.cosine_progress_start_step >= total_train_steps:
                raise ValueError(
                    "cosine_progress_start_step must be below max_steps"
                )
        if self.stop_after_step is not None and (
            self.stop_after_step <= 0
            or self.stop_after_step > self.max_steps
        ):
            raise ValueError(
                "stop_after_step must be positive and no greater than the "
                "production max_steps contract: "
                f"{self.stop_after_step} vs {self.max_steps}"
            )
        warmup_steps = int(total_train_steps * self.warmup_ratio)
        if self.optimizer_parameter_groups is None:
            self.scheduler = self._build_scheduler(
                scheduler_type=self.lr_scheduler_type,
                total_train_steps=total_train_steps,
                warmup_steps=warmup_steps,
            )
        else:
            self.scheduler = self._build_parameter_group_scheduler(
                scheduler_type=self.lr_scheduler_type,
                total_train_steps=total_train_steps,
                warmup_steps=warmup_steps,
            )
        self.global_step = 0
        self.optimizer_updates_attempted = 0
        self.optimizer_updates_successful = 0
        self.optimizer_updates_skipped = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self._last_checkpoint_step: int | None = None
        self.loaded_resume_source: dict | None = None
        self.weights_only_warmstart_source: dict | None = None
        self.fresh_stage_weights_transition_source: dict | None = None

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.training_metrics_writer = None
        if self.accelerator.is_main_process:
            self.training_metrics_writer = TrainingMetricsJsonlWriter(
                os.path.join(self.output_dir, "training_metrics.jsonl")
            )

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self._assert_prepared_deepspeed_contract()
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)


    def _resolved_strategies(self) -> TrainerStrategies:
        strategies = getattr(self, "strategies", None)
        if strategies is None:
            strategies = TrainerStrategies()
            self.strategies = strategies
        return strategies

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        return self._resolved_strategies().sampler_loader(
            self,
            dataset,
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _all_gather_rank_objects(self, value):
        return all_gather_rank_objects(self, value)

    def _discover_runtime_topology(self) -> dict:
        return discover_runtime_topology(self)

    def _discover_runtime_execution(self) -> dict:
        return discover_runtime_execution(self)

    def _assert_prepared_deepspeed_contract(self) -> None:
        """Validate the effective DeepSpeed engine for the selected profile."""

        from .training.deepspeed_contract import (
            assert_prepared_deepspeed_contract,
            is_momap_production_accelerate_config,
        )

        accelerate_config = os.environ.get(
            "FASTWAM_ACCELERATE_CONFIG_FILE"
        )
        return assert_prepared_deepspeed_contract(
            self,
            require_momap_production_zero2=(
                is_momap_production_accelerate_config(accelerate_config)
            ),
        )

    def _assert_in_trainer_evaluation_contract(self) -> None:
        """Reject external-evaluator datasets before the legacy loop runs."""

        if self.eval_every <= 0:
            return
        if self.val_dataset is None:
            raise ValueError(
                "eval_every>0 requires a validation dataset for the legacy "
                "in-trainer evaluator"
            )
        contract = getattr(
            self.val_dataset,
            "in_trainer_evaluation_contract",
            LEGACY_VIDEO_PROMPT_EVALUATOR,
        )
        if contract != LEGACY_VIDEO_PROMPT_EVALUATOR:
            raise ValueError(
                "eval_every>0 selected Wan22Trainer.evaluate(), but the "
                f"validation dataset declares {contract!r}. MoMap validation "
                "must use the standalone checkpoint evaluator; keep "
                "eval_every=0 for this training run."
            )

    @staticmethod
    def _validated_runtime_topology(topology: object) -> dict:
        return validated_runtime_topology(topology)

    def _set_train_data_epoch(self, epoch: int) -> None:
        absolute_epoch = int(epoch)
        if absolute_epoch < 0:
            raise ValueError("training epoch must be non-negative")
        dataset_setter = getattr(
            getattr(self, "train_dataset", None),
            "set_epoch",
            None,
        )
        if callable(dataset_setter):
            dataset_setter(absolute_epoch)
        loader_setter = getattr(self.train_loader, "set_epoch", None)
        if callable(loader_setter):
            loader_setter(absolute_epoch)
            return
        sampler_setter = getattr(self.train_sampler, "set_epoch", None)
        if callable(sampler_setter):
            sampler_setter(absolute_epoch)

    @staticmethod
    def _validated_optimizer_update_counters(
        counters: object,
        *,
        global_step: int,
        require_zero_skipped: bool,
    ) -> dict:
        if not isinstance(counters, dict) or set(counters) != {
            "schema_version",
            "attempted",
            "successful",
            "skipped",
        }:
            raise ValueError("optimizer update counters have an invalid structure")
        if (
            counters.get("schema_version")
            != OPTIMIZER_UPDATE_COUNTERS_SCHEMA_VERSION
        ):
            raise ValueError("optimizer update counters have an unsupported schema")
        values: dict[str, int] = {}
        for key in ("attempted", "successful", "skipped"):
            value = counters.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"optimizer update counter `{key}` must be a non-negative integer"
                )
            values[key] = int(value)
        if values["successful"] != int(global_step):
            raise ValueError(
                "successful optimizer update count must equal global_step"
            )
        if values["attempted"] != values["successful"] + values["skipped"]:
            raise ValueError(
                "attempted optimizer updates must equal successful + skipped"
            )
        if require_zero_skipped and values["skipped"] != 0:
            raise ValueError(
                "refusing to resume after a skipped optimizer update"
            )
        return {
            "schema_version": OPTIMIZER_UPDATE_COUNTERS_SCHEMA_VERSION,
            **values,
        }

    def _optimizer_update_counters(self) -> dict:
        return self._validated_optimizer_update_counters(
            {
                "schema_version": OPTIMIZER_UPDATE_COUNTERS_SCHEMA_VERSION,
                "attempted": self.optimizer_updates_attempted,
                "successful": self.optimizer_updates_successful,
                "skipped": self.optimizer_updates_skipped,
            },
            global_step=self.global_step,
            require_zero_skipped=False,
        )

    def _validate_training_cursor(
        self,
        *,
        global_step: int,
        epoch: int,
        batch_in_epoch: int,
    ) -> None:
        sampler = self.train_sampler.binding()
        microbatches = sampler.get("global_batches_per_epoch")
        sampler_accumulation = sampler.get("gradient_accumulation_steps")
        tail_policy = sampler.get("tail_policy")
        if (
            isinstance(microbatches, bool)
            or not isinstance(microbatches, int)
            or microbatches <= 0
        ):
            raise ValueError("sampler global_batches_per_epoch is invalid")
        if sampler_accumulation != self.gradient_accumulation_steps:
            raise ValueError("sampler gradient accumulation contract drifted")
        accumulation = self.gradient_accumulation_steps
        if batch_in_epoch > microbatches:
            raise ValueError("training cursor exceeds the sampler epoch")
        if tail_policy == LEGACY_UNPADDED_TAIL_POLICY:
            if (
                batch_in_epoch % accumulation != 0
                and batch_in_epoch != microbatches
            ):
                raise ValueError(
                    "legacy training cursor is not an optimizer-update boundary"
                )
            updates_per_epoch = int(
                sampler.get("optimizer_updates_per_epoch", 0)
            )
            if updates_per_epoch <= 0:
                raise ValueError(
                    "legacy sampler optimizer_updates_per_epoch is invalid"
                )
            updates_at_cursor = ceil(batch_in_epoch / accumulation)
        else:
            if microbatches % accumulation != 0:
                raise ValueError(
                    "strict sampler epoch contains a partial accumulation tail"
                )
            if batch_in_epoch % accumulation != 0:
                raise ValueError(
                    "training cursor is not an optimizer-update boundary"
                )
            updates_per_epoch = microbatches // accumulation
            updates_at_cursor = batch_in_epoch // accumulation
        expected_step = epoch * updates_per_epoch + updates_at_cursor
        from .training.topology_transition import contract as topology_contract, cursor_offset
        transition = topology_contract(getattr(self, 'cfg', {}))
        if transition is not None:
            expected_step -= cursor_offset(transition)
        if global_step != expected_step:
            raise ValueError(
                "training cursor arithmetic drifted: "
                f"global_step={global_step}, expected={expected_step}, "
                f"epoch={epoch}, batch_in_epoch={batch_in_epoch}"
            )

    def _is_valid_optimizer_boundary_window(
        self,
        window_microbatch_count: int,
    ) -> bool:
        """Accept full windows plus Accelerate's original final partial tail."""

        if window_microbatch_count == self.gradient_accumulation_steps:
            return True
        return bool(
            self.train_sampler.binding().get("tail_policy")
            == LEGACY_UNPADDED_TAIL_POLICY
            and self.batch_in_epoch
            == self.train_sampler.global_batches_per_epoch
            and 0
            < window_microbatch_count
            < self.gradient_accumulation_steps
        )

    def _model_math_resume_contract(self) -> dict | None:
        """Return finite model math semantics for full-state resume."""

        model = self.model
        unwrap_model = getattr(self.accelerator, "unwrap_model", None)
        if callable(unwrap_model):
            model = unwrap_model(model)
        contract_provider = getattr(model, "resume_math_contract", None)
        if not callable(contract_provider):
            return None
        raw = contract_provider()
        if not isinstance(raw, Mapping):
            raise TypeError("model resume_math_contract() must return a mapping")
        try:
            normalized = json.loads(
                json.dumps(
                    dict(raw),
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "model resume math contract must contain finite JSON values"
            ) from exc
        if not isinstance(normalized, dict):
            raise AssertionError(
                "model resume math contract normalization returned a non-object"
            )
        return normalized

    def _data_math_resume_contract(self) -> dict | None:
        """Return finite dataset math semantics for full-state resume."""

        dataset = getattr(self, "train_dataset", None)
        contract_provider = getattr(dataset, "resume_math_contract", None)
        if not callable(contract_provider):
            return None
        raw = contract_provider()
        if not isinstance(raw, Mapping):
            raise TypeError("dataset resume_math_contract() must return a mapping")
        try:
            normalized = json.loads(
                json.dumps(
                    dict(raw),
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "dataset resume math contract must contain finite JSON values"
            ) from exc
        if not isinstance(normalized, dict):
            raise AssertionError(
                "dataset resume math contract normalization returned a non-object"
            )
        return normalized

    def _resume_contract(self) -> dict:
        sampler_state = operational_sampler_resume_state(
            self.train_sampler.binding()
        )
        topology = self._validated_runtime_topology(self._runtime_topology)
        if sampler_state.get("world_size") != topology["world_size"]:
            raise ValueError(
                "sampler world_size differs from the measured runtime topology"
            )
        training_contract = {
            "gradient_accumulation_steps": int(
                self.gradient_accumulation_steps
            ),
            "max_steps": int(self.max_steps),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "adam_betas": list(self.adam_betas),
            "max_grad_norm": self.max_grad_norm,
            "mixed_precision": self.mixed_precision,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "deepspeed": self.effective_deepspeed_contract,
        }
        cosine_min_learning_rate = getattr(
            self,
            "cosine_min_learning_rate",
            None,
        )
        if self.cfg.get("dataloader_rng_isolated", False):
            training_contract["dataloader_rng_isolated"] = True
        if cosine_min_learning_rate is not None:
            training_contract["cosine_min_learning_rate"] = (
                cosine_min_learning_rate
            )
        cosine_progress_start_step = getattr(
            self,
            "cosine_progress_start_step",
            None,
        )
        if cosine_progress_start_step is not None:
            training_contract["cosine_progress_start_step"] = int(
                cosine_progress_start_step
            )
        optimizer_parameter_groups = getattr(
            self,
            "optimizer_parameter_groups",
            None,
        )
        if optimizer_parameter_groups is not None:
            training_contract["optimizer_parameter_groups"] = (
                optimizer_parameter_groups
            )
        reset = getattr(self, "sample_progress_reset", None)
        if reset is not None:
            training_contract["sample_progress_reset"] = reset
        from .training.feedback_continuation import contract as feedback_contract
        feedback = feedback_contract(self.cfg)
        if feedback is not None:
            training_contract["feedback_continuation"] = feedback
        from .training.libero_feedback import contract as libero_feedback_contract
        libero_feedback = libero_feedback_contract(self.cfg)
        if libero_feedback is not None:
            training_contract['libero_failure_replay'] = libero_feedback
        from .training.topology_transition import contract as topology_contract
        transition = topology_contract(self.cfg)
        if transition is not None:
            training_contract['topology_transition'] = transition
        from .training.real_budget_extension import contract as real_extension_contract
        real_extension = real_extension_contract(self.cfg)
        if real_extension is not None:
            training_contract['real_budget_extension'] = real_extension
        from .training.libero_transfer_extension import contract as transfer_extension_contract
        transfer_extension = transfer_extension_contract(self.cfg)
        if transfer_extension is not None:
            training_contract['libero_transfer_extension'] = transfer_extension
        from .training.plus_50k_gradient import contract as plus_50k_contract
        plus_50k = plus_50k_contract(self.cfg)
        if plus_50k is not None:
            training_contract['plus_50k_gradient'] = plus_50k
        from .training.libero_50k_extension import contract as libero_50k_contract
        libero_50k = libero_50k_contract(self.cfg)
        if libero_50k is not None:
            training_contract['libero_50k_extension'] = libero_50k
        from .training.scene_replay import contract as scene_replay_contract
        scene_replay = scene_replay_contract(self.cfg)
        if scene_replay is not None:
            training_contract['scene_replay'] = scene_replay
        from .training.task_replay import contract as task_replay_contract
        task_replay = task_replay_contract(self.cfg)
        if task_replay is not None:
            training_contract['task_failure_replay'] = task_replay
        training_stage = getattr(self, "training_stage", "joint")
        if training_stage != "joint":
            training_contract["training_stage"] = training_stage
        return {
            "schema_version": RESUME_CONTRACT_SCHEMA_VERSION,
            "sampler": sampler_state,
            "runtime_topology": topology,
            "model_math": self._model_math_resume_contract(),
            "data_math": self._data_math_resume_contract(),
            "training": training_contract,
        }

    @staticmethod
    def _controlled_model_math_transition_contract(
        *,
        source_global_step: int,
    ) -> dict[str, object]:
        if (
            isinstance(source_global_step, bool)
            or not isinstance(source_global_step, int)
            or source_global_step < 0
        ):
            raise ValueError(
                "controlled model-math source_global_step must be a "
                "non-negative integer"
            )
        return {
            "schema_version": CONTROLLED_MODEL_MATH_TRANSITION_SCHEMA_VERSION,
            "transition_id": CONTROLLED_MODEL_MATH_MIGRATION_TRANSITION_ID,
            "source_global_step": int(source_global_step),
            "transition_steps": CONTROLLED_MODEL_MATH_TRANSITION_STEPS,
            "transition_curve": CONTROLLED_MODEL_MATH_TRANSITION_CURVE,
            "source_native_kv_fusion": (
                CONTROLLED_MODEL_MATH_SOURCE_NATIVE_KV_FUSION
            ),
            "target_native_kv_fusion": (
                CONTROLLED_MODEL_MATH_TARGET_NATIVE_KV_FUSION
            ),
        }

    @staticmethod
    def _controlled_model_math_bridge_schedule(
        *,
        global_kind: str,
        pit_kind: str,
    ) -> list[list[object]]:
        return [
            *[[global_kind, index] for index in range(26)],
            *[[pit_kind, index] for index in range(4)],
        ]

    @staticmethod
    def _canonical_json_object(
        value: object,
        *,
        name: str,
    ) -> dict[str, object] | None:
        if not isinstance(value, Mapping):
            return None
        try:
            copied = json.loads(
                json.dumps(
                    dict(value),
                    ensure_ascii=True,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError):
            return None
        if not isinstance(copied, dict):
            return None
        return copied

    def _derived_controlled_model_math_source_binding(
        self,
        *,
        source_state_dir: Path | None,
        source_global_step: int | None,
    ) -> dict[str, object] | None:
        """Read the configured source state and derive its immutable step.

        The YAML deliberately contains no source step. Every process derives it
        from the same collectively audited checkpoint metadata/trainer state,
        preventing a stale hand-entered step from shifting the anneal window.
        """

        migration = getattr(self, "controlled_model_math_migration", None)
        if not isinstance(migration, Mapping):
            return None
        configured_path = Path(
            str(migration.get("source_state_dir", ""))
        ).expanduser()
        if configured_path.is_symlink():
            raise ValueError(
                "controlled_model_math_migration source_state_dir must not be "
                "a symlink"
            )
        try:
            configured_path = configured_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                "controlled_model_math_migration source_state_dir is not an "
                "existing directory"
            ) from exc
        if not configured_path.is_dir():
            raise ValueError(
                "controlled_model_math_migration source_state_dir is not a "
                "directory"
            )

        source_audit = self._collective_state_directory_audit(configured_path)
        source_resume_state = source_audit.get("resume_state")
        source_trainer_state = source_audit.get("trainer_state")
        if not isinstance(source_resume_state, Mapping) or not isinstance(
            source_trainer_state,
            Mapping,
        ):
            raise ValueError(
                "controlled_model_math_migration source state lacks audited "
                "resume metadata"
            )
        metadata_step = source_resume_state.get("global_step")
        trainer_step = source_trainer_state.get("global_step")
        for name, value in (
            ("checkpoint metadata global_step", metadata_step),
            ("trainer_state global_step", trainer_step),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    "controlled_model_math_migration source "
                    f"{name} must be a non-negative integer"
                )
        if metadata_step != trainer_step:
            raise ValueError(
                "controlled_model_math_migration source checkpoint metadata "
                "and trainer_state global_step differ"
            )

        source_contract = source_trainer_state.get("resume_contract")
        source_model_math = (
            source_contract.get("model_math")
            if isinstance(source_contract, Mapping)
            else None
        )
        source_action = (
            source_model_math.get("action")
            if isinstance(source_model_math, Mapping)
            else None
        )
        if (
            not isinstance(source_action, Mapping)
            or source_action.get("native_kv_fusion")
            != migration.get("source_native_kv_fusion")
        ):
            raise ValueError(
                "controlled_model_math_migration source state does not carry "
                "the declared V8 log-count-corrected model math"
            )

        if source_state_dir is not None:
            try:
                actual_path = source_state_dir.resolve(strict=True)
            except OSError as exc:
                raise ValueError(
                    "resume source state directory cannot be resolved"
                ) from exc
            if actual_path == configured_path:
                if (
                    isinstance(source_global_step, bool)
                    or not isinstance(source_global_step, int)
                    or source_global_step < 0
                ):
                    raise ValueError(
                        "resume source global_step must be a non-negative "
                        "integer"
                    )
                if source_global_step != metadata_step:
                    raise ValueError(
                        "configured migration source step differs from audited "
                        "checkpoint metadata"
                    )
        return {
            "source_state_dir": str(configured_path),
            "source_global_step": int(metadata_step),
        }

    def _bind_controlled_model_math_migration(
        self,
        *,
        source_state_dir: Path | None,
        source_global_step: int | None,
    ) -> None:
        """Bind one model runtime to the metadata-derived anneal origin."""

        migration = getattr(self, "controlled_model_math_migration", None)
        if not isinstance(migration, Mapping):
            return
        binding = self._derived_controlled_model_math_source_binding(
            source_state_dir=source_state_dir,
            source_global_step=source_global_step,
        )
        if binding is None:
            raise AssertionError("controlled model-math binding unexpectedly missing")
        existing = getattr(
            self,
            "_controlled_model_math_migration_binding",
            None,
        )
        if existing is not None and existing != binding:
            raise ValueError(
                "controlled_model_math_migration cannot rebind a live process "
                "to a different source state"
            )
        unwrap_model = getattr(self.accelerator, "unwrap_model", None)
        model = unwrap_model(self.model) if callable(unwrap_model) else self.model
        configure = getattr(
            model,
            "configure_controlled_native_kv_count_correction_anneal",
            None,
        )
        if not callable(configure):
            raise TypeError(
                "controlled_model_math_migration requires model method "
                "configure_controlled_native_kv_count_correction_anneal()"
            )
        configure(
            source_global_step=int(binding["source_global_step"]),
            transition_steps=int(migration["transition_steps"]),
            transition_curve=str(migration["transition_curve"]),
            transition_id=str(migration["transition_id"]),
            source_native_kv_fusion=str(migration["source_native_kv_fusion"]),
            target_native_kv_fusion=str(migration["target_native_kv_fusion"]),
        )
        self._controlled_model_math_migration_binding = dict(binding)

    def _set_controlled_model_math_migration_runtime_step(
        self,
        *,
        global_step: int,
    ) -> None:
        """Set the schedule point after a full-state load without touching ZeRO."""

        migration = getattr(self, "controlled_model_math_migration", None)
        if not isinstance(migration, Mapping):
            return
        if (
            isinstance(global_step, bool)
            or not isinstance(global_step, int)
            or global_step < 0
        ):
            raise ValueError(
                "controlled model-math runtime global_step must be a "
                "non-negative integer"
            )
        binding = getattr(
            self,
            "_controlled_model_math_migration_binding",
            None,
        )
        if not isinstance(binding, Mapping):
            raise RuntimeError(
                "controlled_model_math_migration was not bound before setting "
                "its runtime step"
            )
        unwrap_model = getattr(self.accelerator, "unwrap_model", None)
        model = unwrap_model(self.model) if callable(unwrap_model) else self.model
        set_step = getattr(
            model,
            "set_attention_runtime_step",
            None,
        )
        if not callable(set_step):
            raise TypeError(
                "controlled_model_math_migration requires model method "
                "set_attention_runtime_step()"
            )
        set_step(global_step=int(global_step))

    def _collective_set_controlled_model_math_migration_runtime_step(
        self,
        *,
        global_step: int,
    ) -> None:
        """Apply the post-load schedule point consistently on every rank."""

        if not isinstance(
            getattr(self, "controlled_model_math_migration", None),
            Mapping,
        ):
            return
        local_error: Exception | None = None
        try:
            self._set_controlled_model_math_migration_runtime_step(
                global_step=global_step,
            )
        except Exception as exc:
            local_error = exc
        local_status = (
            None
            if local_error is None
            else {
                "error_type": type(local_error).__name__,
                "message": str(local_error),
            }
        )
        gathered_status = self._all_gather_rank_objects(local_status)
        if any(status is not None for status in gathered_status):
            if len(gathered_status) == 1 and local_error is not None:
                raise local_error
            raise RuntimeError(
                "controlled model-math runtime-step binding failed on one or "
                f"more ranks: {json.dumps(gathered_status, sort_keys=True)}"
            )

    def _allows_controlled_model_math_migration(
        self,
        saved: Mapping[str, object],
        current: Mapping[str, object],
        *,
        source_state_dir: Path | None,
        source_global_step: int | None,
    ) -> bool:
        """Allow only the declared V8 count-correction -> anneal math diff."""

        migration = getattr(self, "controlled_model_math_migration", None)
        binding = getattr(
            self,
            "_controlled_model_math_migration_binding",
            None,
        )
        if not isinstance(migration, Mapping) or not isinstance(binding, Mapping):
            return False
        if source_state_dir is None or source_global_step is None:
            return False
        try:
            expected_source = Path(
                str(binding.get("source_state_dir", ""))
            ).resolve(strict=True)
            actual_source = source_state_dir.resolve(strict=True)
        except OSError:
            return False
        if expected_source != actual_source:
            return False
        if binding.get("source_global_step") != source_global_step:
            return False

        saved_model = self._canonical_json_object(
            saved.get("model_math"),
            name="saved model_math",
        )
        candidate_model = self._canonical_json_object(
            current.get("model_math"),
            name="current model_math",
        )
        if saved_model is None or candidate_model is None:
            return False
        saved_action = saved_model.get("action")
        candidate_action = candidate_model.get("action")
        saved_bridge = saved_model.get("bridge")
        candidate_bridge = candidate_model.get("bridge")
        saved_method = saved_model.get("method_contract")
        candidate_method = candidate_model.get("method_contract")
        if not all(
            isinstance(value, dict)
            for value in (
                saved_action,
                candidate_action,
                saved_bridge,
                candidate_bridge,
                saved_method,
                candidate_method,
            )
        ):
            return False
        if (
            saved_action.get("native_kv_fusion")
            != migration.get("source_native_kv_fusion")
            or candidate_action.get("native_kv_fusion")
            != migration.get("target_native_kv_fusion")
            or candidate_bridge.get("bridge_semantics_version")
            != CONTROLLED_MODEL_MATH_TARGET_BRIDGE_SEMANTICS
        ):
            return False
        expected_transition = self._controlled_model_math_transition_contract(
            source_global_step=int(source_global_step),
        )
        if "native_kv_fusion_transition" in saved_model:
            return False
        transition = candidate_model.pop("native_kv_fusion_transition", None)
        if transition != expected_transition:
            return False
        if any(
            saved_method.get(field) != expected
            for field, expected in CONTROLLED_MODEL_MATH_SOURCE_METHOD_FIELDS.items()
        ):
            return False
        if any(
            candidate_method.get(field) != expected
            for field, expected in CONTROLLED_MODEL_MATH_TARGET_METHOD_FIELDS.items()
        ):
            return False
        expected_target_schedule = self._controlled_model_math_bridge_schedule(
            global_kind="global_native_kv_direct_count_correction_anneal_to_none",
            pit_kind="pit_native_kv_direct_count_correction_anneal_to_none",
        )
        if candidate_bridge.get("bridge_source_schedule") != expected_target_schedule:
            return False

        candidate_action["native_kv_fusion"] = str(
            migration["source_native_kv_fusion"]
        )
        candidate_bridge["bridge_semantics_version"] = (
            CONTROLLED_MODEL_MATH_SOURCE_BRIDGE_SEMANTICS
        )
        candidate_bridge["bridge_source_schedule"] = (
            self._controlled_model_math_bridge_schedule(
                global_kind="global_native_kv_direct_count_corrected",
                pit_kind="pit_native_kv_direct_count_corrected",
            )
        )
        candidate_method.update(CONTROLLED_MODEL_MATH_SOURCE_METHOD_FIELDS)
        if candidate_model != saved_model:
            return False
        unchanged = dict(saved)
        unchanged["model_math"] = current.get("model_math")
        return unchanged == dict(current)

    def _allows_constant_lr_max_steps_extension(
        self,
        saved: Mapping[str, object],
        current: Mapping[str, object],
    ) -> bool:
        """Allow only an explicit constant-LR extension of the step budget."""

        if not bool(
            getattr(self, "allow_constant_lr_max_steps_extension", False)
        ):
            return False
        saved_training_raw = saved.get("training")
        current_training_raw = current.get("training")
        if not isinstance(saved_training_raw, Mapping) or not isinstance(
            current_training_raw,
            Mapping,
        ):
            return False
        saved_training = dict(saved_training_raw)
        current_training = dict(current_training_raw)
        saved_max_steps = saved_training.get("max_steps")
        current_max_steps = current_training.get("max_steps")
        if (
            isinstance(saved_max_steps, bool)
            or not isinstance(saved_max_steps, int)
            or isinstance(current_max_steps, bool)
            or not isinstance(current_max_steps, int)
            or current_max_steps <= saved_max_steps
        ):
            return False
        if (
            saved_training.get("lr_scheduler_type") != "constant"
            or current_training.get("lr_scheduler_type") != "constant"
            or saved_training.get("warmup_ratio") != 0.0
            or current_training.get("warmup_ratio") != 0.0
        ):
            return False
        saved_training["max_steps"] = current_max_steps
        normalized_saved = dict(saved)
        normalized_saved["training"] = saved_training
        return normalized_saved == dict(current)

    def _allows_controlled_sampler_migration(
        self,
        saved: Mapping[str, object],
        current: Mapping[str, object],
        *,
        source_state_dir: Path | None,
        source_global_step: int | None,
    ) -> bool:
        """Allow exactly one declared old-replay -> task×variant transition."""

        migration = getattr(self, "controlled_sampler_migration", None)
        if not isinstance(migration, Mapping):
            return False
        if source_state_dir is None or source_global_step is None:
            return False
        expected_source = Path(
            str(migration.get("source_state_dir", ""))
        ).expanduser().resolve(strict=False)
        if expected_source != source_state_dir.resolve(strict=False):
            return False
        if migration.get("source_global_step") != source_global_step:
            return False

        saved_sampler = saved.get("sampler")
        current_sampler = current.get("sampler")
        if not isinstance(saved_sampler, Mapping) or not isinstance(
            current_sampler,
            Mapping,
        ):
            return False
        if (
            saved_sampler.get("sampling_mode")
            != migration.get("source_sampling_mode")
            or current_sampler.get("sampling_mode")
            != migration.get("target_sampling_mode")
            or saved_sampler.get("failure_manifest_sha256")
            != migration.get("source_failure_manifest_sha256")
            or current_sampler.get("failure_manifest_sha256")
            != migration.get("target_failure_manifest_sha256")
        ):
            return False
        expected_world_size = migration.get("source_world_size")
        if (
            saved_sampler.get("world_size") != expected_world_size
            or current_sampler.get("world_size") != expected_world_size
        ):
            return False

        sampler_invariants = (
            "dataset_split_name",
            "dataset_length",
            "episode_count",
            "episode_ranges",
            "seed",
            "batch_size_per_rank",
            "world_size",
            "gradient_accumulation_steps",
            "global_microbatch_size",
            "effective_global_batch_size",
            "stream_samples_per_epoch",
            "selected_samples_per_epoch",
            "padded_samples_per_epoch",
            "global_batches_per_epoch",
            "optimizer_updates_per_epoch",
            "tail_policy",
        )
        if any(
            saved_sampler.get(field) != current_sampler.get(field)
            for field in sampler_invariants
        ):
            return False

        unchanged = dict(saved)
        unchanged["sampler"] = dict(current_sampler)
        if unchanged != dict(current):
            return False
        return True

    def _allows_controlled_source_input_migration(
        self,
        saved: Mapping[str, object],
        current: Mapping[str, object],
        *,
        source_state_dir: Path | None,
        source_global_step: int | None,
    ) -> bool:
        """Allow only one exactly declared Seg/Depth source-input change."""

        migration = getattr(self, "controlled_source_input_migration", None)
        if not isinstance(migration, Mapping):
            return False
        if source_state_dir is None or source_global_step is None:
            return False
        expected_source = Path(
            str(migration.get("source_state_dir", ""))
        ).expanduser().resolve(strict=False)
        if expected_source != source_state_dir.resolve(strict=False):
            return False
        if migration.get("source_global_step") != source_global_step:
            return False
        saved_data = saved.get("data_math")
        current_data = current.get("data_math")
        if not isinstance(saved_data, Mapping) or not isinstance(
            current_data, Mapping
        ):
            return False
        saved_data_copy = json.loads(json.dumps(saved_data, sort_keys=True))
        current_data_copy = json.loads(json.dumps(current_data, sort_keys=True))
        saved_source = saved_data_copy.get("source")
        current_source = current_data_copy.get("source")
        if not isinstance(saved_source, dict) or not isinstance(
            current_source, dict
        ):
            return False
        migration_schema = migration.get("schema_version")
        if migration_schema == CONTROLLED_SOURCE_INPUT_MIGRATION_SCHEMA_VERSION:
            if "source_input_augmentation" in saved_source:
                return False
            augmentation = current_source.pop(
                "source_input_augmentation", None
            )
            if not isinstance(augmentation, Mapping):
                return False
            segmentation = augmentation.get("segmentation")
            depth = augmentation.get("depth")
            if not isinstance(segmentation, Mapping) or not isinstance(
                depth, Mapping
            ):
                return False
            if (
                segmentation.get("policy_content_digest")
                != migration.get("target_seg_policy_content_digest")
                or depth.get("schema_version")
                != migration.get("target_depth_noise_schema_version")
            ):
                return False
        elif (
            migration_schema
            == CONTROLLED_SOURCE_INPUT_REPLACEMENT_MIGRATION_SCHEMA_VERSION
        ):
            saved_augmentation = saved_source.pop(
                "source_input_augmentation", None
            )
            current_augmentation = current_source.pop(
                "source_input_augmentation", None
            )
            if not isinstance(saved_augmentation, Mapping) or not isinstance(
                current_augmentation, Mapping
            ):
                return False
            saved_segmentation = saved_augmentation.get("segmentation")
            current_segmentation = current_augmentation.get("segmentation")
            saved_depth = saved_augmentation.get("depth")
            current_depth = current_augmentation.get("depth")
            if not all(
                isinstance(item, Mapping)
                for item in (
                    saved_segmentation,
                    current_segmentation,
                    saved_depth,
                    current_depth,
                )
            ):
                return False
            if (
                saved_segmentation.get("policy_content_digest")
                != migration.get("source_seg_policy_content_digest")
                or current_segmentation.get("policy_content_digest")
                != migration.get("target_seg_policy_content_digest")
                or saved_depth.get("schema_version")
                != migration.get("source_depth_noise_schema_version")
                or current_depth.get("schema_version")
                != migration.get("target_depth_noise_schema_version")
                or _canonical_mapping_sha256(saved_augmentation)
                != migration.get("source_augmentation_contract_sha256")
                or _canonical_mapping_sha256(current_augmentation)
                != migration.get("target_augmentation_contract_sha256")
            ):
                return False
        else:
            return False
        current_data_copy["source"] = current_source
        if current_data_copy != saved_data_copy:
            return False
        unchanged = dict(saved)
        unchanged["data_math"] = dict(current_data)
        return unchanged == dict(current)

    def _allows_controlled_scheduler_floor_migration(
        self,
        saved: Mapping[str, object],
        current: Mapping[str, object],
        *,
        source_state_dir: Path | None,
        source_global_step: int | None,
    ) -> bool:
        """Allow only the declared 1e-5 to 1e-6 named-group LR change."""

        migration = getattr(
            self,
            "controlled_scheduler_floor_migration",
            None,
        )
        if not isinstance(migration, Mapping):
            return False
        if source_state_dir is None or source_global_step is None:
            return False
        expected_source = Path(
            str(migration.get("source_state_dir", ""))
        ).expanduser().resolve(strict=False)
        if expected_source != source_state_dir.resolve(strict=False):
            return False
        if migration.get("source_global_step") != source_global_step:
            return False

        saved_training = saved.get("training")
        current_training = current.get("training")
        if not isinstance(saved_training, Mapping) or not isinstance(
            current_training, Mapping
        ):
            return False
        saved_copy = json.loads(json.dumps(saved_training, sort_keys=True))
        current_copy = json.loads(json.dumps(current_training, sort_keys=True))
        if (
            saved_copy.get("lr_scheduler_type") != "cosine"
            or current_copy.get("lr_scheduler_type") != "cosine"
            or saved_copy.get("cosine_min_learning_rate")
            != migration["source_cosine_min_learning_rate"]
            or current_copy.get("cosine_min_learning_rate")
            != migration["target_cosine_min_learning_rate"]
        ):
            return False

        saved_groups = saved_copy.get("optimizer_parameter_groups")
        current_groups = current_copy.get("optimizer_parameter_groups")
        expected_names = {"world", "action_backbone", "action_io"}
        if (
            not isinstance(saved_groups, dict)
            or not isinstance(current_groups, dict)
            or set(saved_groups) != expected_names
            or set(current_groups) != expected_names
        ):
            return False
        for name in expected_names:
            source_group = saved_groups.get(name)
            target_group = current_groups.get(name)
            if not isinstance(source_group, dict) or not isinstance(
                target_group, dict
            ):
                return False
            if (
                source_group.get("cosine_min_learning_rate")
                != migration["source_cosine_min_learning_rate"]
                or target_group.get("cosine_min_learning_rate")
                != migration["target_cosine_min_learning_rate"]
                or source_group.get("warmup_start_learning_rate")
                != migration["source_warmup_start_learning_rate"]
                or target_group.get("warmup_start_learning_rate")
                != migration["target_warmup_start_learning_rate"]
            ):
                return False
            source_group["cosine_min_learning_rate"] = (
                migration["target_cosine_min_learning_rate"]
            )
            source_group["warmup_start_learning_rate"] = (
                migration["target_warmup_start_learning_rate"]
            )
        saved_copy["cosine_min_learning_rate"] = (
            migration["target_cosine_min_learning_rate"]
        )
        if saved_copy != current_copy:
            return False
        unchanged = dict(saved)
        unchanged["training"] = dict(current_training)
        return unchanged == dict(current)

    def _allows_controlled_sampler_and_source_input_migration(
        self,
        saved: Mapping[str, object],
        current: Mapping[str, object],
        *,
        source_state_dir: Path | None,
        source_global_step: int | None,
    ) -> bool:
        """Allow exactly the two declared migrations from one source state."""

        sampler_migration = getattr(
            self,
            "controlled_sampler_migration",
            None,
        )
        source_migration = getattr(
            self,
            "controlled_source_input_migration",
            None,
        )
        if not isinstance(sampler_migration, Mapping) or not isinstance(
            source_migration,
            Mapping,
        ):
            return False
        for field in (
            "source_state_dir",
            "source_global_step",
            "cursor_policy",
        ):
            if sampler_migration.get(field) != source_migration.get(field):
                return False

        sampler_only_current = dict(current)
        sampler_only_current["data_math"] = saved.get("data_math")
        if not self._allows_controlled_sampler_migration(
            saved,
            sampler_only_current,
            source_state_dir=source_state_dir,
            source_global_step=source_global_step,
        ):
            return False

        source_only_current = dict(current)
        source_only_current["sampler"] = saved.get("sampler")
        return self._allows_controlled_source_input_migration(
            saved,
            source_only_current,
            source_state_dir=source_state_dir,
            source_global_step=source_global_step,
        )

    def _validate_resume_contract(
        self,
        saved: object,
        *,
        source_state_dir: Path | None = None,
        source_global_step: int | None = None,
    ) -> None:
        self._bind_controlled_model_math_migration(
            source_state_dir=source_state_dir,
            source_global_step=source_global_step,
        )
        current = self._resume_contract()
        try:
            normalized_saved = normalize_saved_resume_contract(
                saved,
                current_sampler_binding=self.train_sampler.binding(),
                current_model_math=current["model_math"],
                current_data_math=current["data_math"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "trainer_state.json lacks a compatible operational resume state"
            ) from exc
        if normalized_saved == current:
            return
        from .training.task_replay import contract as task_contract, permits_migration as permits_task_replay
        if permits_task_replay(normalized_saved,current,task_contract(self.cfg),source_state_dir,source_global_step):
            logger.warning('Authorized 70/30 task replay migration; Adam/LR/RNG/cursor/budget unchanged')
            return
        from .training.real_budget_extension import allows as allows_real_extension
        if allows_real_extension(normalized_saved,current,self.cfg,source_state_dir,source_global_step):
            logger.warning('Authorized real-data 2000 -> 50000; Adam/RNG/sampler preserved')
            return
        from .training.libero_transfer_extension import allows as allows_transfer_extension
        if allows_transfer_extension(normalized_saved,current,self.cfg,source_state_dir,source_global_step):
            logger.warning('Authorized Plus-initialized LIBERO extension 1500 -> 50000; Adam/RNG/replay/cursor preserved')
            return
        from .training.libero_50k_extension import allows as allows_libero_50k
        if allows_libero_50k(normalized_saved,current,self.cfg,source_state_dir,source_global_step):
            logger.warning('Authorized LIBERO current-stage 50K extension; preserve Adam/RNG/replay/cursor')
            return
        from .training.plus_50k_gradient import allows as allows_plus_50k
        if allows_plus_50k(normalized_saved,current,self.cfg,source_state_dir,source_global_step):
            logger.warning('Authorized PLUS total50K, Action->World detach until20K then 1000-step ramp; Adam/RNG/cursor preserved')
            return
        from .training.plus_lr_correction import allows as allows_plus_lr_correction
        if allows_plus_lr_correction(normalized_saved,current,self.cfg,source_state_dir,source_global_step):
            logger.warning('Authorized PLUS peak LR 1e-6 -> 1e-4, floor1e-6; preserve Adam/RNG/sampler/step and 30K budget')
            return
        from .training.scene_replay import contract as scene_contract, permits_migration as permits_scene_replay
        if permits_scene_replay(normalized_saved,current,scene_contract(self.cfg),source_state_dir,source_global_step):
            logger.warning('Authorized test-scene expert replay: 50/50, preserved Adam/RNG/cursor, new LR cycle')
            return
        from .training.libero_feedback import contract as libero_feedback_contract, permits_migration as permits_libero_replay
        if permits_libero_replay(normalized_saved,current,libero_feedback_contract(self.cfg),source_state_dir,source_global_step):
            logger.warning('Authorized LIBERO 80/20 task replay: preserve model, Adam, LR schedule, RNG, step and base sampler cursor')
            return
        from .training.feedback_continuation import contract as feedback_contract, permits_migration
        if permits_migration(normalized_saved, current, feedback_contract(self.cfg),
                             source_state_dir, source_global_step):
            logger.warning("Authorized feedback continuation: preserve Adam/model/RNG/cursor; new LR cycle")
            return
        from mwam.training.gt_background_migration import allows_gt_background_migration
        gt_migration = getattr(self, "cfg", {}).get("controlled_gt_background_migration", None)
        if gt_migration is not None:
            if not allows_gt_background_migration(
                normalized_saved, current, gt_migration,
                source_state_dir=source_state_dir, source_global_step=source_global_step,
            ):
                raise ValueError("GT background migration does not match the exact source and supervision-only contract")
            logger.warning("GT_BACKGROUND_MIGRATION: 200px -> GT entity background exclusion at step=%s; "
                           "model inputs, Adam, scheduler, RNG and sampler preserved.", source_global_step)
            return
        from mwam.training.sample_progress_reset import allows_step_budget_extension
        if allows_step_budget_extension(
            normalized_saved, current, getattr(self, "cfg", {}).get("controlled_step_budget_extension", None),
            source_state_dir=source_state_dir, source_global_step=source_global_step,
        ):
            logger.warning("Resuming source-bound cumulative30000 extension at step=%s; "
                           "Adam, scheduler cursor, RNG and sampler remain restored.", source_global_step)
            return
        if self._allows_constant_lr_max_steps_extension(
            normalized_saved,
            current,
        ):
            logger.warning(
                "Extending an explicitly authorized constant-LR resume budget "
                "from max_steps=%d to max_steps=%d; optimizer, scheduler, RNG, "
                "sampler, model, data, and topology state remain bound.",
                int(normalized_saved["training"]["max_steps"]),
                int(current["training"]["max_steps"]),
            )
            return
        if isinstance(
            getattr(
                self,
                "controlled_scheduler_floor_migration",
                None,
            ),
            Mapping,
        ):
            if self._allows_controlled_scheduler_floor_migration(
                normalized_saved,
                current,
                source_state_dir=source_state_dir,
                source_global_step=source_global_step,
            ):
                logger.warning(
                    "Applying explicitly configured scheduler-floor migration "
                    "at step=%d: cosine floor and warmup-start contract "
                    "1e-5 -> 1e-6. Model, optimizer state, scheduler cursor, "
                    "RNG, sampler and global step remain preserved.",
                    source_global_step,
                )
                return
            raise ValueError(
                "controlled_scheduler_floor_migration does not match the "
                "declared source state and exact resume contract"
            )
        if isinstance(
            getattr(self, "controlled_model_math_migration", None),
            Mapping,
        ):
            if self._allows_controlled_model_math_migration(
                normalized_saved,
                current,
                source_state_dir=source_state_dir,
                source_global_step=source_global_step,
            ):
                logger.warning(
                    "Applying explicitly configured V8 model-math migration "
                    "at metadata-derived source step=%d: log-count correction "
                    "will cosine-anneal to zero over %d optimizer steps. "
                    "Model parameters/schema, optimizer, scheduler, RNG, "
                    "sampler, topology, global_step and saved epoch/batch "
                    "optimizer boundary remain unchanged.",
                    source_global_step,
                    CONTROLLED_MODEL_MATH_TRANSITION_STEPS,
                )
                return
            raise ValueError(
                "controlled model-math migration does not match the declared "
                "source state and exact V8 resume contract"
            )
        combined_migration_configured = bool(
            isinstance(
                getattr(self, "controlled_sampler_migration", None),
                Mapping,
            )
            and isinstance(
                getattr(self, "controlled_source_input_migration", None),
                Mapping,
            )
        )
        if combined_migration_configured:
            if self._allows_controlled_sampler_and_source_input_migration(
                normalized_saved,
                current,
                source_state_dir=source_state_dir,
                source_global_step=source_global_step,
            ):
                logger.warning(
                    "Applying explicitly configured sampler and source-input "
                    "migrations from the same state at step=%d. Model, "
                    "optimizer, scheduler, global_step and the saved "
                    "epoch/batch optimizer boundary remain unchanged.",
                    source_global_step,
                )
                return
            raise ValueError(
                "combined controlled sampler/source-input migration does not "
                "match the declared source state and exact resume contract"
            )
        if self._allows_controlled_sampler_migration(
            normalized_saved,
            current,
            source_state_dir=source_state_dir,
            source_global_step=source_global_step,
        ):
            logger.warning(
                "Applying explicitly configured sampler migration at step=%d: "
                "%s -> %s. Model/optimizer/scheduler/global_step and the "
                "saved epoch/batch optimizer boundary remain unchanged.",
                source_global_step,
                normalized_saved["sampler"]["sampling_mode"],
                current["sampler"]["sampling_mode"],
            )
            return
        if self._allows_controlled_source_input_migration(
            normalized_saved,
            current,
            source_state_dir=source_state_dir,
            source_global_step=source_global_step,
        ):
            logger.warning(
                "Applying explicitly configured source-input migration at "
                "step=%d. Model/optimizer/scheduler/sampler/global_step and "
                "the saved epoch/batch optimizer boundary remain unchanged.",
                source_global_step,
            )
            return
        raise ValueError(
            "resume operational state mismatch; refusing to change sampler "
            "stream, topology, batch, optimizer, or scheduler semantics: "
            f"saved={json.dumps(normalized_saved, sort_keys=True)} "
            f"current={json.dumps(current, sort_keys=True)}"
        )

    def _apply_controlled_scheduler_floor_migration_after_resume(self) -> None:
        """Realign the restored optimizer LR to the new LambdaLR curve."""

        migration = getattr(
            self,
            "controlled_scheduler_floor_migration",
            None,
        )
        if not isinstance(migration, Mapping):
            return
        source_step = int(migration["source_global_step"])
        if self.global_step != source_step:
            raise ValueError(
                "scheduler-floor migration runtime step differs from source step"
            )
        loaded_source = getattr(self, "loaded_resume_source", None)
        if not isinstance(loaded_source, Mapping):
            raise RuntimeError(
                "scheduler-floor migration lacks verified loaded resume source"
            )
        expected_source = Path(
            str(migration["source_state_dir"])
        ).expanduser().resolve(strict=False)
        actual_source = Path(
            str(loaded_source.get("source_path", ""))
        ).expanduser().resolve(strict=False)
        if (
            actual_source != expected_source
            or loaded_source.get("source_global_step") != source_step
        ):
            raise ValueError(
                "scheduler-floor migration loaded source identity drifted"
            )

        scheduler = getattr(self.scheduler, "scheduler", self.scheduler)
        if not isinstance(scheduler, LambdaLR):
            raise TypeError(
                "scheduler-floor migration requires the named-group LambdaLR"
            )
        if scheduler.last_epoch != source_step:
            raise ValueError(
                "scheduler-floor migration scheduler cursor drifted"
            )
        param_groups = getattr(self.optimizer, "param_groups", None)
        if not isinstance(param_groups, list):
            raise TypeError(
                "scheduler-floor migration optimizer lacks parameter groups"
            )
        if not (
            len(param_groups)
            == len(scheduler.base_lrs)
            == len(scheduler.lr_lambdas)
        ):
            raise ValueError(
                "scheduler-floor migration LR group count drifted"
            )
        new_lrs = [
            float(base_lr) * float(lr_lambda(source_step))
            for base_lr, lr_lambda in zip(
                scheduler.base_lrs,
                scheduler.lr_lambdas,
                strict=True,
            )
        ]
        if any(not np.isfinite(value) or value <= 0 for value in new_lrs):
            raise ValueError(
                "scheduler-floor migration produced invalid learning rates"
            )
        old_lrs = [float(group["lr"]) for group in param_groups]
        for group, learning_rate in zip(param_groups, new_lrs, strict=True):
            group["lr"] = learning_rate
        scheduler._last_lr = list(new_lrs)
        self.applied_scheduler_floor_migration = {
            "schema_version": (
                CONTROLLED_SCHEDULER_FLOOR_MIGRATION_SCHEMA_VERSION
            ),
            "source_global_step": source_step,
            "old_learning_rates": old_lrs,
            "new_learning_rates": list(new_lrs),
        }
        self.accelerator.wait_for_everyone()
        logger.warning(
            "Realigned restored named-group LR at step=%d for the controlled "
            "scheduler-floor migration: old=%s new=%s.",
            source_step,
            old_lrs,
            new_lrs,
        )

    def _is_deepspeed_runtime(self) -> bool:
        runtime_execution = getattr(self, "runtime_execution", None)
        if runtime_execution is None:
            return False
        if not isinstance(runtime_execution, dict):
            raise RuntimeError("runtime execution evidence must be a mapping")
        ranks = runtime_execution.get("ranks")
        if (
            not isinstance(ranks, list)
            or len(ranks) != int(self.accelerator.num_processes)
            or any(not isinstance(record, dict) for record in ranks)
        ):
            raise RuntimeError("runtime execution evidence has an invalid rank set")
        runtime_types = {
            str(record.get("distributed_type", "")).upper()
            for record in ranks
        }
        if runtime_types == {"DEEPSPEED"}:
            return True
        if "DEEPSPEED" in runtime_types:
            raise RuntimeError("DeepSpeed runtime type differs across ranks")
        return False

    def _local_checkpoint_cursor_evidence(
        self,
        *,
        expected_step: int,
    ) -> dict:
        scheduler_state = self.scheduler.state_dict()
        if not isinstance(scheduler_state, dict):
            raise RuntimeError("scheduler state_dict must return a mapping")
        scheduler_last_epoch = scheduler_state.get("last_epoch")
        if (
            isinstance(scheduler_last_epoch, bool)
            or not isinstance(scheduler_last_epoch, int)
            or scheduler_last_epoch != expected_step
        ):
            raise RuntimeError(
                "scheduler cursor disagrees with successful optimizer updates: "
                f"{scheduler_last_epoch!r} != {expected_step}"
            )

        deepspeed_global_steps = None
        if self._is_deepspeed_runtime():
            deepspeed_global_steps = getattr(self.model, "global_steps", None)
            if (
                isinstance(deepspeed_global_steps, bool)
                or not isinstance(deepspeed_global_steps, int)
                or deepspeed_global_steps != expected_step
            ):
                raise RuntimeError(
                    "DeepSpeed global_steps disagrees with the trainer cursor: "
                    f"{deepspeed_global_steps!r} != {expected_step}"
                )
        return {
            "schema_version": CHECKPOINT_RANK_STATE_SCHEMA_VERSION,
            "rank": int(self.accelerator.process_index),
            "world_size": int(self.accelerator.num_processes),
            "global_step": expected_step,
            "scheduler_last_epoch": scheduler_last_epoch,
            "deepspeed_global_steps": deepspeed_global_steps,
        }

    def _collective_checkpoint_cursor_evidence(
        self,
        *,
        expected_step: int,
        phase: str,
    ) -> list[dict]:
        local_record: dict | None = None
        local_error: Exception | None = None
        try:
            local_record = self._local_checkpoint_cursor_evidence(
                expected_step=expected_step
            )
        except Exception as exc:
            local_error = exc
        local_status = (
            None
            if local_error is None
            else {
                "error_type": type(local_error).__name__,
                "message": str(local_error),
            }
        )
        gathered_status = self._all_gather_rank_objects(local_status)
        if any(status is not None for status in gathered_status):
            if len(gathered_status) == 1 and local_error is not None:
                raise local_error
            raise RuntimeError(
                f"{phase} checkpoint cursor validation failed on one or more "
                f"ranks: {json.dumps(gathered_status, sort_keys=True)}"
            )
        gathered = self._all_gather_rank_objects(local_record)
        if any(not isinstance(record, dict) for record in gathered):
            raise RuntimeError(
                f"{phase} checkpoint cursor evidence is missing on one or more ranks"
            )
        normalized = [dict(record) for record in gathered]
        baseline = {
            key: value
            for key, value in normalized[0].items()
            if key != "rank"
        }
        for rank, record in enumerate(normalized):
            if record.get("rank") != rank:
                raise RuntimeError(
                    f"{phase} checkpoint cursor evidence is not rank ordered"
                )
            comparable = {
                key: value for key, value in record.items() if key != "rank"
            }
            if comparable != baseline:
                raise RuntimeError(
                    f"{phase} checkpoint cursor differs across distributed ranks"
                )
        return normalized

    @staticmethod
    def _normalized_rng_tensor(value: object, *, key: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint RNG field {key!r} must be a tensor")
        normalized = value.detach().cpu()
        if normalized.dtype != torch.uint8 or normalized.ndim != 1:
            raise ValueError(
                f"checkpoint RNG field {key!r} must be a one-dimensional "
                "torch.uint8 tensor"
            )
        return normalized

    @staticmethod
    def _numpy_rng_states_equal(
        expected: object,
        observed: tuple,
    ) -> bool:
        if (
            not isinstance(expected, tuple)
            or len(expected) != 5
            or not isinstance(expected[1], np.ndarray)
            or not isinstance(observed, tuple)
            or len(observed) != 5
        ):
            return False
        expected_array = expected[1]
        observed_array = observed[1]
        if (
            not isinstance(observed_array, np.ndarray)
            or expected_array.dtype != observed_array.dtype
            or expected_array.shape != observed_array.shape
            or not np.array_equal(expected_array, observed_array)
        ):
            return False

        def _same_scalar(left: object, right: object) -> bool:
            if isinstance(left, float) and isinstance(right, float):
                if np.isnan(left) and np.isnan(right):
                    return True
            return bool(left == right)

        return (
            expected[0] == observed[0]
            and _same_scalar(expected[2], observed[2])
            and _same_scalar(expected[3], observed[3])
            and _same_scalar(expected[4], observed[4])
        )

    def _verify_local_loaded_rng_state(self, state_dir: Path) -> None:
        rank = int(self.accelerator.process_index)
        rng_path = state_dir / f"random_states_{rank}.pkl"
        try:
            saved = torch.load(
                rng_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"strict post-load RNG read failed for rank {rank}: {exc}"
            ) from exc
        if not isinstance(saved, Mapping):
            raise TypeError(
                f"strict post-load RNG state for rank {rank} must be a mapping"
            )
        required_keys = {
            "random_state",
            "numpy_random_seed",
            "torch_manual_seed",
        }
        missing = sorted(required_keys.difference(saved))
        if missing:
            raise ValueError(
                f"strict post-load RNG state for rank {rank} lacks keys: {missing}"
            )
        if saved["random_state"] != random.getstate():
            raise RuntimeError(
                f"strict post-load Python RNG state mismatch on rank {rank}"
            )
        if not self._numpy_rng_states_equal(
            saved["numpy_random_seed"],
            np.random.get_state(),
        ):
            raise RuntimeError(
                f"strict post-load NumPy RNG state mismatch on rank {rank}"
            )
        saved_torch = self._normalized_rng_tensor(
            saved["torch_manual_seed"],
            key="torch_manual_seed",
        )
        if not torch.equal(saved_torch, torch.get_rng_state().cpu()):
            raise RuntimeError(
                f"strict post-load torch CPU RNG state mismatch on rank {rank}"
            )

        if torch.cuda.is_available():
            cuda_key = "torch_cuda_manual_seed"
            if cuda_key not in saved:
                raise ValueError(
                    f"strict post-load RNG state for rank {rank} lacks "
                    f"required CUDA key {cuda_key!r}"
                )
            saved_cuda = saved[cuda_key]
            if not isinstance(saved_cuda, (list, tuple)):
                raise TypeError(
                    f"checkpoint RNG field {cuda_key!r} must be a sequence"
                )
            observed_cuda = torch.cuda.get_rng_state_all()
            if len(saved_cuda) != len(observed_cuda):
                raise RuntimeError(
                    "strict post-load CUDA RNG device-count mismatch on rank "
                    f"{rank}: {len(saved_cuda)} != {len(observed_cuda)}"
                )
            for device_index, (expected, observed) in enumerate(
                zip(saved_cuda, observed_cuda, strict=True)
            ):
                expected_tensor = self._normalized_rng_tensor(
                    expected,
                    key=f"{cuda_key}[{device_index}]",
                )
                if not torch.equal(expected_tensor, observed.detach().cpu()):
                    raise RuntimeError(
                        "strict post-load CUDA RNG state mismatch on rank "
                        f"{rank} device {device_index}"
                    )

    def _collective_verify_loaded_rng_state(self, state_dir: Path) -> None:
        local_error: Exception | None = None
        try:
            self._verify_local_loaded_rng_state(state_dir)
        except Exception as exc:
            local_error = exc
        local_status = (
            None
            if local_error is None
            else {
                "error_type": type(local_error).__name__,
                "message": str(local_error),
            }
        )
        gathered_status = self._all_gather_rank_objects(local_status)
        if any(status is not None for status in gathered_status):
            if len(gathered_status) == 1 and local_error is not None:
                raise local_error
            raise RuntimeError(
                "strict post-load RNG verification failed on one or more ranks: "
                f"{json.dumps(gathered_status, sort_keys=True)}"
            )

    def _collective_load_accelerator_state(self, state_dir: Path) -> None:
        local_error: Exception | None = None
        try:
            self.accelerator.load_state(input_dir=str(state_dir))
        except Exception as exc:
            local_error = exc
        local_status = (
            None
            if local_error is None
            else {
                "error_type": type(local_error).__name__,
                "message": str(local_error),
            }
        )
        gathered_status = self._all_gather_rank_objects(local_status)
        if any(status is not None for status in gathered_status):
            if len(gathered_status) == 1 and local_error is not None:
                raise local_error
            raise RuntimeError(
                "Accelerate/DeepSpeed state load failed on one or more ranks: "
                f"{json.dumps(gathered_status, sort_keys=True)}"
            )

    def _collective_state_directory_audit(self, state_dir: Path) -> dict:
        scanner_rank = 0
        local_result: dict | None = None
        local_error: Exception | None = None
        if int(self.accelerator.process_index) == scanner_rank:
            try:
                local_result = audit_state_directory(state_dir)
            except Exception as exc:
                local_error = exc
        local_status = (
            {
                "ok": False,
                "scanner": True,
                "error_type": type(local_error).__name__,
                "message": str(local_error),
            }
            if local_error is not None
            else {
                "ok": True,
                "scanner": local_result is not None,
            }
        )
        gathered_status = self._all_gather_rank_objects(local_status)
        errors = [
            {"rank": rank, **status}
            for rank, status in enumerate(gathered_status)
            if not status.get("ok")
        ]
        if errors:
            if len(gathered_status) == 1 and local_error is not None:
                raise local_error
            raise RuntimeError(
                "checkpoint state audit failed on one or more ranks: "
                f"{json.dumps(errors, sort_keys=True)}"
            )
        if not gathered_status[scanner_rank].get("scanner") or any(
            status.get("scanner")
            for rank, status in enumerate(gathered_status)
            if rank != scanner_rank
        ):
            raise RuntimeError(
                "checkpoint audit scanner assignment is inconsistent"
            )
        gathered_results = self._all_gather_rank_objects(local_result)
        result = gathered_results[scanner_rank]
        if not isinstance(result, dict):
            raise AssertionError(
                "checkpoint audit scanner produced no result"
            )
        if any(
            candidate is not None
            for rank, candidate in enumerate(gathered_results)
            if rank != scanner_rank
        ):
            raise RuntimeError(
                "non-scanner rank produced a checkpoint audit"
            )
        return result

    def _collective_pending_state_path(
        self,
        *,
        final_path: Path,
    ) -> Path:
        rank = int(self.accelerator.process_index)
        local_name = (
            f".{final_path.name}.pending.{os.getpid()}.{time.time_ns()}"
            if rank == 0
            else None
        )
        gathered_names = self._all_gather_rank_objects(local_name)
        pending_name = gathered_names[0]
        if not isinstance(pending_name, str) or not pending_name:
            raise RuntimeError("rank zero did not assign a pending checkpoint path")
        if any(name is not None for name in gathered_names[1:]):
            raise RuntimeError("non-zero rank assigned a pending checkpoint path")
        pending_path = final_path.parent / pending_name

        local_error: Exception | None = None
        if rank == 0:
            try:
                if final_path.exists() or final_path.is_symlink():
                    raise FileExistsError(
                        f"final checkpoint state already exists: {final_path}"
                    )
                pending_path.mkdir(mode=0o700, parents=False, exist_ok=False)
            except Exception as exc:
                local_error = exc
        local_status = (
            None
            if local_error is None
            else {
                "error_type": type(local_error).__name__,
                "message": str(local_error),
            }
        )
        gathered_status = self._all_gather_rank_objects(local_status)
        if any(status is not None for status in gathered_status):
            if len(gathered_status) == 1 and local_error is not None:
                raise local_error
            raise RuntimeError(
                "pending checkpoint creation failed: "
                f"{json.dumps(gathered_status, sort_keys=True)}"
            )
        return pending_path

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        opt_steps_per_epoch = max(
            int(self.train_sampler.optimizer_updates_per_epoch), 1
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    @staticmethod
    def _build_optimizer_parameter_groups(
        model,
        group_specs: Mapping[str, Mapping[str, float]],
    ) -> list[dict]:
        """Assign every trainable PixelDiT parameter to one explicit owner."""

        dit = getattr(model, "dit", None)
        world_expert = getattr(dit, "world_expert", None)
        world_text_projection = getattr(dit, "world_text_projection", None)
        action_expert = getattr(dit, "action_expert", None)
        if (
            dit is None
            or world_expert is None
            or world_text_projection is None
            or action_expert is None
        ):
            raise TypeError(
                "optimizer_parameter_groups requires a PixelDiT model with "
                "dit.world_expert, dit.world_text_projection, and "
                "dit.action_expert"
            )

        def _parameter_ids(*modules) -> set[int]:
            result: set[int] = set()
            for module in modules:
                if module is not None:
                    result.update(id(parameter) for parameter in module.parameters())
            return result

        world_ids = _parameter_ids(world_expert, world_text_projection)
        adapter_modules = tuple(
            getattr(dit, name, None)
            for name in (
                "world_to_action_adapters",
                "world_native_kv_adapters",
                "pit_native_kv_adapters",
            )
        )
        adapter_ids = _parameter_ids(*adapter_modules)
        action_io_modules = tuple(
            getattr(action_expert, name, None)
            for name in (
                "action_encoder",
                "action_decoder",
                "context_projection",
                "t_embedder",
                "text_embedding",
                "time_embedding",
                "time_projection",
                "head",
            )
        ) + (getattr(model, "proprio_encoder", None),)
        action_io_ids = _parameter_ids(*action_io_modules)
        action_expert_ids = _parameter_ids(action_expert)
        action_backbone_ids = action_expert_ids.difference(action_io_ids)

        owner_ids = {
            "world": world_ids,
            "action_backbone": action_backbone_ids,
            "action_adapters": adapter_ids,
            "action_io": action_io_ids,
        }
        all_owner_ids: set[int] = set()
        for group_name in OPTIMIZER_PARAMETER_GROUP_ORDER:
            overlap = all_owner_ids.intersection(owner_ids[group_name])
            if overlap:
                raise RuntimeError(
                    "optimizer parameter ownership overlaps for group "
                    f"{group_name!r}"
                )
            all_owner_ids.update(owner_ids[group_name])

        parameters_by_group: dict[str, list[torch.nn.Parameter]] = {
            name: [] for name in OPTIMIZER_PARAMETER_GROUP_ORDER
        }
        unowned: list[str] = []
        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            owners = [
                group_name
                for group_name, parameter_ids in owner_ids.items()
                if id(parameter) in parameter_ids
            ]
            if len(owners) != 1:
                unowned.append(parameter_name)
                continue
            parameters_by_group[owners[0]].append(parameter)
        if unowned:
            raise RuntimeError(
                "trainable parameters do not have exactly one optimizer owner: "
                f"{unowned[:20]}"
            )

        active_groups = {
            group_name
            for group_name, parameters in parameters_by_group.items()
            if parameters
        }
        configured_groups = set(group_specs)
        if configured_groups != active_groups:
            missing = sorted(active_groups.difference(configured_groups))
            inactive = sorted(configured_groups.difference(active_groups))
            raise ValueError(
                "optimizer_parameter_groups must exactly match active parameter "
                f"owners; missing={missing}, inactive={inactive}"
            )

        optimizer_groups: list[dict] = []
        for group_name in OPTIMIZER_PARAMETER_GROUP_ORDER:
            parameters = parameters_by_group[group_name]
            if not parameters:
                continue
            spec = group_specs[group_name]
            optimizer_groups.append(
                {
                    "name": group_name,
                    "params": parameters,
                    "lr": float(spec["learning_rate"]),
                }
            )
            logger.info(
                "Optimizer parameter group: name=%s tensors=%d parameters=%d "
                "peak_lr=%.8g floor_lr=%.8g warmup_start_lr=%.8g",
                group_name,
                len(parameters),
                sum(parameter.numel() for parameter in parameters),
                float(spec["learning_rate"]),
                float(spec["cosine_min_learning_rate"]),
                float(spec["warmup_start_learning_rate"]),
            )
        return optimizer_groups

    def _build_parameter_group_scheduler(
        self,
        scheduler_type,
        total_train_steps: int,
        warmup_steps: int = 0,
    ):
        """Warm each named group from its absolute start, then decay to its floor."""

        scheduler_type = str(scheduler_type).strip().lower()
        if scheduler_type not in {"constant", "cosine"}:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)
        remaining_steps = max(total_train_steps - warmup_steps, 1)
        group_specs = self.optimizer_parameter_groups
        if group_specs is None:
            raise RuntimeError("named scheduler requires optimizer_parameter_groups")
        cosine_progress_start_step = getattr(
            self,
            "cosine_progress_start_step",
            None,
        )

        lr_lambdas = []
        for parameter_group in self.optimizer.param_groups:
            group_name = parameter_group.get("name")
            if group_name not in group_specs:
                raise RuntimeError(
                    f"optimizer group {group_name!r} lacks a scheduler contract"
                )
            spec = group_specs[group_name]
            peak = float(spec["learning_rate"])
            floor = float(spec["cosine_min_learning_rate"])
            start = float(spec["warmup_start_learning_rate"])

            def _multiplier(
                step: int,
                *,
                peak: float = peak,
                floor: float = floor,
                start: float = start,
            ) -> float:
                step = max(int(step), 0)
                if warmup_steps > 0 and step <= warmup_steps:
                    alpha = step / warmup_steps
                    return (start + alpha * (peak - start)) / peak
                if scheduler_type == "constant":
                    return 1.0
                if cosine_progress_start_step is None:
                    progress = min(
                        max((step - warmup_steps) / remaining_steps, 0.0),
                        1.0,
                    )
                else:
                    continuation_steps = (
                        total_train_steps - cosine_progress_start_step
                    )
                    progress = min(
                        max(
                            (step - cosine_progress_start_step)
                            / continuation_steps,
                            0.0,
                        ),
                        1.0,
                    )
                cosine = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
                return (floor + (peak - floor) * cosine) / peak

            lr_lambdas.append(_multiplier)
        reset = getattr(self, "sample_progress_reset", None)
        if reset is not None:
            from mwam.training.sample_progress_reset import reset_learning_rate
            lr_lambdas = []
            for parameter_group in self.optimizer.param_groups:
                spec = group_specs[parameter_group["name"]]
                def reset_multiplier(step, spec=spec):
                    return reset_learning_rate(step, reset,
                        peak=float(spec["learning_rate"]),
                        floor=float(spec["cosine_min_learning_rate"]),
                        start=float(spec["warmup_start_learning_rate"])) / float(spec["learning_rate"])
                lr_lambdas.append(reset_multiplier)
        from .training.feedback_continuation import contract as feedback_contract, lr_at
        feedback = feedback_contract(self.cfg)
        if feedback is not None:
            lr_lambdas = [lambda step, spec=feedback: lr_at(step, spec) / 1e-4
                          for _ in self.optimizer.param_groups]
        from .training.scene_replay import contract as scene_contract, lr_at as scene_lr_at
        scene = scene_contract(self.cfg)
        if scene is not None:
            lr_lambdas = [lambda step, spec=scene: scene_lr_at(step,spec)/1e-4 for _ in self.optimizer.param_groups]
        from .training.libero_transfer_extension import contract as transfer_contract, lr_at as transfer_lr_at
        extension = transfer_contract(self.cfg)
        if extension is not None:
            lr_lambdas = [lambda step, spec=extension: transfer_lr_at(step,spec)/spec['peak_lr']
                          for _ in self.optimizer.param_groups]
        from .training.real_budget_extension import contract as real_contract, lr_at as real_lr_at
        real_extension = real_contract(self.cfg)
        if real_extension is not None:
            lr_lambdas = [lambda step, spec=real_extension: real_lr_at(step,spec)/spec['peak_lr']
                          for _ in self.optimizer.param_groups]
        return LambdaLR(self.optimizer, lr_lambda=lr_lambdas)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            cosine_min_learning_rate = getattr(
                self,
                "cosine_min_learning_rate",
                None,
            )
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=(
                    self.learning_rate * 0.01
                    if cosine_min_learning_rate is None
                    else cosine_min_learning_rate
                ),
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _collect_cuda_memory_metrics(
        self,
        device: torch.device,
    ) -> dict[str, float]:
        return self._resolved_strategies().telemetry.collect_cuda_memory_metrics(
            self,
            device,
        )

    def _estimate_eta(self):
        return self._resolved_strategies().telemetry.estimate_eta(self)

    def _resume_or_load_checkpoint(self):
        return self._resolved_strategies().checkpoint.resume_or_load(self)

    def _set_dit_only_train_mode(self):
        training_stage = getattr(self, "training_stage", "joint")
        logger.info(
            "Applying training-stage freeze policy: %s.",
            training_stage,
        )
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(
            model,
            training_stage=training_stage,
        )

    @staticmethod
    def _apply_dit_only_train_mode(model, *, training_stage="joint"):
        if training_stage not in TRAINING_STAGES:
            raise ValueError(
                "training_stage must be one of "
                f"{sorted(TRAINING_STAGES)}, got {training_stage!r}"
            )
        model.eval()
        model.requires_grad_(False)
        if training_stage == "world_only":
            dit = getattr(model, "dit", None)
            world_expert = getattr(dit, "world_expert", None)
            world_text_projection = getattr(
                dit,
                "world_text_projection",
                None,
            )
            if world_expert is None or world_text_projection is None:
                raise TypeError(
                    "world_only training requires dit.world_expert and "
                    "dit.world_text_projection"
                )
            world_expert.train()
            world_expert.requires_grad_(True)
            world_text_projection.train()
            world_text_projection.requires_grad_(True)
            return
        if training_stage == "action_only":
            dit = getattr(model, "dit", None)
            action_expert = getattr(dit, "action_expert", None)
            if dit is None or action_expert is None:
                raise TypeError(
                    "action_only training requires dit.action_expert"
                )
            # Keep the joint container in training mode so Action gradient
            # checkpointing remains active, then put the frozen World owner
            # back in eval mode explicitly.
            dit.train()
            dit.requires_grad_(False)
            world_expert = getattr(dit, "world_expert", None)
            world_text_projection = getattr(
                dit,
                "world_text_projection",
                None,
            )
            if world_expert is None or world_text_projection is None:
                raise TypeError(
                    "action_only training requires dit.world_expert and "
                    "dit.world_text_projection"
                )
            world_expert.eval()
            world_text_projection.eval()
            action_modules = [
                action_expert,
                getattr(dit, "world_to_action_adapters", None),
                getattr(dit, "world_native_kv_adapters", None),
                getattr(dit, "pit_native_kv_adapters", None),
                getattr(model, "proprio_encoder", None),
            ]
            for module in action_modules:
                if module is None:
                    continue
                module.train()
                module.requires_grad_(True)
            return
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_training = any(module.training for module in model.modules())
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        return self._resolved_strategies().checkpoint.save_weights(
            self,
            step_tag=step_tag,
        )

    def _save_trainer_state(
        self,
        state_path: str,
        *,
        rank_state: list[dict],
    ):
        return self._resolved_strategies().checkpoint.save_trainer_state(
            self,
            state_path,
            rank_state=rank_state,
        )

    def save_checkpoint(self):
        return self._resolved_strategies().checkpoint.save(self)

    def load_training_state(self, state_dir: str):
        return self._resolved_strategies().checkpoint.load(self, state_dir)

    def _collect_debug_optimizer_audit_metrics(
        self,
        *,
        device: torch.device,
    ) -> dict[str, float]:
        """Collect an opt-in model audit after a successful optimizer step."""

        unwrap_model = getattr(self.accelerator, "unwrap_model", None)
        model = unwrap_model(self.model) if callable(unwrap_model) else self.model
        provider = getattr(model, "debug_optimizer_audit_metrics", None)
        if not callable(provider):
            return {}
        local = _validated_debug_optimizer_audit_metrics(provider())
        keys = tuple(local)
        gathered_keys = self._all_gather_rank_objects(keys)
        if any(candidate != keys for candidate in gathered_keys):
            raise RuntimeError(
                "debug optimizer audit metric keys differ across ranks"
            )
        result: dict[str, float] = {}
        for key in keys:
            gathered = self.accelerator.gather(
                torch.tensor(
                    [local[key]],
                    device=device,
                    dtype=torch.float64,
                )
            ).reshape(-1)
            if gathered.numel() != int(self.accelerator.num_processes):
                raise RuntimeError(
                    "debug optimizer audit collective did not return one "
                    "value per rank"
                )
            if not bool(torch.isfinite(gathered).all().item()):
                raise FloatingPointError(
                    f"debug optimizer audit metric {key!r} became non-finite"
                )
            result[key] = float(gathered.mean().item())
        return result

    def train(self):
        return self._resolved_strategies().loop(self)
