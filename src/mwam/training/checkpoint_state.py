"""Checkpoint save and exact-resume responsibilities shared by trainers."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

import torch

from ..utils.checkpoint_integrity import tensor_state_dict_schema
from ..utils.logging_config import get_logger
from ..utils.pytorch_utils import set_global_seed
from ..utils.state_directory_manifest import (
    audit_state_directory,
    publish_checkpoint,
    validate_checkpoint_rank_state,
    write_checkpoint_metadata,
)
from .adaptive_replay import (
    adaptive_replay_state_snapshot,
    restore_adaptive_replay_state,
)

logger = get_logger(__name__)


TRAINER_CHECKPOINT_STATE_SCHEMA_VERSION = "fastwam_trainer_checkpoint_state_v2"
LOADED_RESUME_SOURCE_SCHEMA_VERSION = "fastwam_loaded_resume_source_v2"
WEIGHTS_ONLY_WARMSTART_SCHEMA_VERSION = (
    "fastwam_weights_only_topology_warmstart_v1"
)
WEIGHTS_ONLY_WARMSTART_MODE = "weights_only_topology_transition"
FRESH_STAGE_WEIGHTS_TRANSITION_SCHEMA_VERSION = (
    "fastwam_fresh_stage_weights_transition_v1"
)
FRESH_STAGE_WEIGHTS_TRANSITION_MODE = "fresh_stage_step0"
PRESERVE_CURSOR_FRESH_OPTIMIZER_SCHEDULER_MODE = (
    "preserve_cursor_fresh_optimizer_scheduler_v1"
)
H8S4_A30_HIGHNOISE_FRESH_OBJECTIVE_TRANSITION_ID = (
    "v7_h8s4_a30_allnative_weighted_to_highnoise405505_uniform_v1"
)
H8S4_A30_HIGHNOISE_INITIAL10_FRESH_OBJECTIVE_TRANSITION_ID = (
    "v7_h8s4_a30_allnative_highnoise405505_uniform_to_initial10_v1"
)
H8S4_A30_HIGHNOISE_INITIAL10_32RANK_FRESH_OBJECTIVE_TRANSITION_ID = (
    "v7_h8s4_a30_allnative_highnoise405505_uniform_to_initial10_32rank_v1"
)
H8S4_A30_PITCOMPACT_HIGHNOISE_FRESH_OBJECTIVE_TRANSITION_ID = (
    "v7_h8s4_a30_pitcompact_weighted_to_highnoise405505_uniform_v1"
)
H8S4_NORES_L24_A30_HIGHNOISE_FRESH_OBJECTIVE_TRANSITION_ID = (
    "v7_h8s4_nores_l24_a30_allnative_weighted_to_highnoise405505_uniform_v1"
)
H8S4_NORES_L24_A30_HIGHNOISE_INITIAL10_16RANK_FRESH_OBJECTIVE_TRANSITION_ID = (
    "v7_h8s4_nores_l24_a30_allnative_highnoise405505_uniform_to_initial10_8to16_v1"
)
H8S4_NORES_L24_FAILURE_REPLAY_GROUPED_LR_FROM_STEP31970_TRANSITION_ID = (
    "v7_nores_l24_initial10_to_failure_replay_groupedlr_from_step31970_v1"
)
H8S4_NORES_L24_LR5E3_ACTIONONLY_FROM_STEP8119_TRANSITION_ID = (
    "v7_nores_l24_lr5e3_actiononly_from_step8119_v1"
)
H8S4_NORES_L24_LR5E3_JOINT_FROM_STEP6395_TRANSITION_ID = (
    "v7_nores_l24_lr5e3_joint_from_step6395_v1"
)
V8_ACTION14_RANDOM_FROM_FROZEN_STEP8119_TRANSITION_ID = (
    "v8_action14_random_from_frozen_s08119_v1"
)
V8_ACTION14_WORLD_TRANSFER_FROM_FROZEN_STEP8119_TRANSITION_ID = (
    "v8_action14_world_transfer_from_frozen_s08119_v1"
)
V8_ACTION14_WORLDTRANSFER_STEP8119_FRESH32_JOINT_STATICBIAS_TRANSITION_ID = (
    "v8_action14_worldtransfer_s08119_fresh32_joint_staticbias_v1"
)
V8_ACTION14_WORLDTRANSFER_STEP9502_ACTIONONLY_TO_V9_JOINT_ANNEAL500_TRANSITION_ID = (
    "v8_action14_worldtransfer_s09502_actiononly_to_v9_joint_"
    "nobias_anneal500_16rank_v1"
)
V9_ACTION14_WORLDTRANSFER_STEP8119_FRESH_STEP0_JOINT_ANNEAL500_TRANSITION_ID = (
    "v9_action14_worldtransfer_s08119_fresh_step0_joint_anneal500_16rank_v1"
)
_FRESH_OBJECTIVE_TRANSITION_REQUIREMENTS = {
    H8S4_A30_HIGHNOISE_FRESH_OBJECTIVE_TRANSITION_ID: {
        "source_global_step": 20000,
        "source_world_size": 16,
        "target_world_size": 16,
        "rng_seed": 43,
    },
    H8S4_A30_HIGHNOISE_INITIAL10_FRESH_OBJECTIVE_TRANSITION_ID: {
        "source_global_step": 20000,
        "source_world_size": 16,
        "target_world_size": 16,
        "rng_seed": 43,
    },
    H8S4_A30_HIGHNOISE_INITIAL10_32RANK_FRESH_OBJECTIVE_TRANSITION_ID: {
        "source_global_step": 20000,
        "source_world_size": 16,
        "target_world_size": 32,
        "rng_seed": 43,
    },
    H8S4_A30_PITCOMPACT_HIGHNOISE_FRESH_OBJECTIVE_TRANSITION_ID: {
        "source_global_step": 20000,
        "source_world_size": 16,
        "target_world_size": 8,
        "rng_seed": 43,
    },
    H8S4_NORES_L24_A30_HIGHNOISE_FRESH_OBJECTIVE_TRANSITION_ID: {
        "source_global_step": 20000,
        "source_world_size": 8,
        "target_world_size": 8,
        "rng_seed": 43,
    },
    H8S4_NORES_L24_A30_HIGHNOISE_INITIAL10_16RANK_FRESH_OBJECTIVE_TRANSITION_ID: {
        "source_global_step": 20000,
        "source_world_size": 8,
        "target_world_size": 16,
        "rng_seed": 43,
    },
    H8S4_NORES_L24_FAILURE_REPLAY_GROUPED_LR_FROM_STEP31970_TRANSITION_ID: {
        "source_global_step": 31970,
        "source_world_size": 16,
        "target_world_size": 16,
        "rng_seed": 43,
        "target_training_stages": frozenset({"action_only", "joint"}),
    },
    H8S4_NORES_L24_LR5E3_ACTIONONLY_FROM_STEP8119_TRANSITION_ID: {
        "source_global_step": 8119,
        "source_world_size": 16,
        "target_world_size": 16,
        "rng_seed": 43,
        "source_training_stage": "action_only",
        "target_training_stages": frozenset({"action_only"}),
    },
    H8S4_NORES_L24_LR5E3_JOINT_FROM_STEP6395_TRANSITION_ID: {
        "source_global_step": 6395,
        "source_world_size": 16,
        "target_world_size": 16,
        "rng_seed": 43,
        "source_training_stage": "joint",
        "target_training_stages": frozenset({"joint"}),
    },
    V8_ACTION14_RANDOM_FROM_FROZEN_STEP8119_TRANSITION_ID: {
        "source_global_step": 8119,
        "source_world_size": 16,
        "target_world_size": 16,
        "rng_seed": 43,
        "source_training_stage": "action_only",
        "target_training_stages": frozenset({"action_only"}),
    },
    V8_ACTION14_WORLD_TRANSFER_FROM_FROZEN_STEP8119_TRANSITION_ID: {
        "source_global_step": 8119,
        "source_world_size": 16,
        "target_world_size": 16,
        "rng_seed": 43,
        "source_training_stage": "action_only",
        "target_training_stages": frozenset({"action_only"}),
    },
    V8_ACTION14_WORLDTRANSFER_STEP8119_FRESH32_JOINT_STATICBIAS_TRANSITION_ID: {
        "source_global_step": 8119,
        "source_world_size": 16,
        "target_world_size": 32,
        "rng_seed": 43,
        "source_training_stage": "action_only",
        "target_training_stages": frozenset({"joint"}),
    },
    V8_ACTION14_WORLDTRANSFER_STEP9502_ACTIONONLY_TO_V9_JOINT_ANNEAL500_TRANSITION_ID: {
        "source_global_step": 9502,
        "source_world_size": 16,
        "target_world_size": 16,
        "rng_seed": 43,
        "source_training_stage": "action_only",
        "target_training_stages": frozenset({"joint"}),
        "mode": PRESERVE_CURSOR_FRESH_OPTIMIZER_SCHEDULER_MODE,
    },
    V9_ACTION14_WORLDTRANSFER_STEP8119_FRESH_STEP0_JOINT_ANNEAL500_TRANSITION_ID: {
        "source_global_step": 8119,
        "source_world_size": 16,
        "target_world_size": 16,
        "rng_seed": 43,
        "source_training_stage": "action_only",
        "target_training_stages": frozenset({"joint"}),
        "native_kv_anneal_start_global_step": 0,
    },
}


def _refresh_deepspeed_fp32_master_params_after_weights_only_load(
    trainer: Any,
) -> None:
    """Copy newly loaded low-precision weights into ZeRO FP32 masters."""

    if not trainer._is_deepspeed_runtime():
        return
    optimizer = getattr(trainer.model, "optimizer", None)
    refresh_fp32_params = getattr(optimizer, "refresh_fp32_params", None)
    if not callable(refresh_fp32_params):
        raise RuntimeError(
            "DeepSpeed weights-only load requires "
            "engine.optimizer.refresh_fp32_params(); refusing to continue "
            "with stale FP32 master parameters"
        )
    refresh_fp32_params()
    logger.info(
        "Refreshed DeepSpeed FP32 master parameters from the newly loaded "
        "low-precision model weights."
    )


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _write_warmstart_lineage(trainer: Any, lineage: dict[str, Any]) -> None:
    if not trainer.accelerator.is_main_process:
        return
    destination = Path(trainer.output_dir) / "weights_only_warmstart_lineage.json"
    pending = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"weights-only warm-start lineage already exists: {destination}"
        )
    try:
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(
                lineage,
                stream,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, destination)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass


def _write_fresh_stage_transition_lineage(
    trainer: Any,
    lineage: dict[str, Any],
) -> None:
    if not trainer.accelerator.is_main_process:
        return
    destination = (
        Path(trainer.output_dir)
        / "fresh_stage_weights_transition_lineage.json"
    )
    pending = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "fresh-stage transition lineage already exists: "
            f"{destination}"
        )
    try:
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(
                lineage,
                stream,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, destination)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass



def _map_preserved_composite_sampler_cursor(
    trainer: Any,
    *,
    source_trainer_state: Mapping[str, Any],
    source_resume_contract: Mapping[str, Any],
    source_step: int,
    source_world_size: int,
    target_world_size: int,
) -> dict[str, int]:
    """Map the exact all-window cursor while intentionally resetting Adam."""

    source_sampler = source_resume_contract.get("sampler")
    if not isinstance(source_sampler, Mapping):
        raise ValueError("composite transition source lacks sampler binding")
    target_sampler = trainer.train_sampler.binding()
    if not isinstance(target_sampler, Mapping):
        raise TypeError("composite transition target sampler binding is invalid")
    compared_fields = (
        "dataset_length",
        "seed",
        "effective_global_batch_size",
        "padded_samples_per_epoch",
        "optimizer_updates_per_epoch",
        "tail_policy",
    )
    for field in compared_fields:
        if source_sampler.get(field) != target_sampler.get(field):
            raise ValueError(
                "composite transition sampler field "
                f"{field!r} differs from the V8 source"
            )
    if source_sampler.get("world_size") != source_world_size:
        raise ValueError("composite transition source sampler world size drifted")
    if target_sampler.get("world_size") != target_world_size:
        raise ValueError("composite transition target sampler world size drifted")

    source_epoch = _nonnegative_int(
        source_trainer_state.get("epoch"),
        name="composite transition source epoch",
    )
    source_batch = _nonnegative_int(
        source_trainer_state.get("batch_in_epoch"),
        name="composite transition source batch_in_epoch",
    )
    source_microbatch = _positive_int(
        source_sampler.get("global_microbatch_size"),
        name="composite transition source global_microbatch_size",
    )
    source_batches_per_epoch = _positive_int(
        source_sampler.get("global_batches_per_epoch"),
        name="composite transition source global_batches_per_epoch",
    )
    source_accumulation = _positive_int(
        source_sampler.get("gradient_accumulation_steps"),
        name="composite transition source gradient_accumulation_steps",
    )
    source_updates_per_epoch = _positive_int(
        source_sampler.get("optimizer_updates_per_epoch"),
        name="composite transition source optimizer_updates_per_epoch",
    )
    if source_batch > source_batches_per_epoch:
        raise ValueError("composite transition source cursor exceeds its epoch")
    if source_batch % source_accumulation != 0:
        raise ValueError(
            "composite transition source cursor is not an update boundary"
        )
    source_cursor_step = (
        source_epoch * source_updates_per_epoch
        + source_batch // source_accumulation
    )
    if source_cursor_step != source_step:
        raise ValueError(
            "composite transition source sampler cursor disagrees with step"
        )

    samples_per_epoch = _positive_int(
        source_sampler.get("padded_samples_per_epoch"),
        name="composite transition source padded_samples_per_epoch",
    )
    consumed_samples = (
        source_epoch * samples_per_epoch
        + source_batch * source_microbatch
    )
    target_microbatch = _positive_int(
        target_sampler.get("global_microbatch_size"),
        name="composite transition target global_microbatch_size",
    )
    target_accumulation = _positive_int(
        target_sampler.get("gradient_accumulation_steps"),
        name="composite transition target gradient_accumulation_steps",
    )
    target_updates_per_epoch = _positive_int(
        target_sampler.get("optimizer_updates_per_epoch"),
        name="composite transition target optimizer_updates_per_epoch",
    )
    target_epoch, target_sample_offset = divmod(
        consumed_samples,
        samples_per_epoch,
    )
    target_batch, remainder = divmod(
        target_sample_offset,
        target_microbatch,
    )
    if remainder != 0 or target_batch % target_accumulation != 0:
        raise ValueError(
            "composite transition source cursor cannot map to target update "
            "boundary"
        )
    target_cursor_step = (
        target_epoch * target_updates_per_epoch
        + target_batch // target_accumulation
    )
    if target_cursor_step != source_step:
        raise ValueError(
            "composite transition mapped sampler cursor changes global_step"
        )

    trainer.global_step = source_step
    trainer.optimizer_updates_attempted = source_step
    trainer.optimizer_updates_successful = source_step
    trainer.optimizer_updates_skipped = 0
    trainer.epoch = int(target_epoch)
    trainer.batch_in_epoch = int(target_batch)
    from .feedback_continuation import restore as restore_feedback_continuation
    restore_feedback_continuation(trainer, payload)
    trainer.train_sampler.set_epoch_offset(0)
    trainer._set_train_data_epoch(trainer.epoch)
    trainer.train_sampler.set_resume_batch_offset(trainer.batch_in_epoch)
    if trainer._is_deepspeed_runtime():
        if not hasattr(trainer.model, "global_steps") or not hasattr(
            trainer.model,
            "global_samples",
        ):
            raise RuntimeError("DeepSpeed engine lacks global cursor fields")
        trainer.model.global_steps = source_step
        trainer.model.global_samples = consumed_samples

    return {
        "consumed_samples": int(consumed_samples),
        "target_epoch": int(target_epoch),
        "target_batch_in_epoch": int(target_batch),
    }


def _initialize_composite_absolute_cosine_scheduler(
    trainer: Any,
    *,
    source_step: int,
) -> dict[str, object]:
    """Start a new optimizer at an audited absolute cosine position."""

    if getattr(trainer, "lr_scheduler_type", None) != "cosine":
        raise ValueError("composite transition requires the cosine scheduler")
    if float(getattr(trainer, "warmup_ratio", -1.0)) != 0.0:
        raise ValueError("composite transition requires warmup_ratio=0")
    if getattr(trainer, "cosine_progress_start_step", None) != source_step:
        raise ValueError(
            "composite transition cosine_progress_start_step must equal "
            "the audited source step"
        )
    if int(getattr(trainer, "max_steps", 0)) <= source_step:
        raise ValueError(
            "composite transition max_steps must exceed the source step"
        )
    group_specs = getattr(trainer, "optimizer_parameter_groups", None)
    if not isinstance(group_specs, Mapping) or set(group_specs) != {
        "action_backbone",
        "action_io",
        "world",
    }:
        raise ValueError(
            "composite transition requires exactly world/action_backbone/"
            "action_io optimizer groups"
        )
    expected_lrs = {
        name: float(spec["learning_rate"]) for name, spec in group_specs.items()
    }
    if len({round(value, 18) for value in expected_lrs.values()}) != 1:
        raise ValueError(
            "composite transition requires World and Action groups to share "
            "one learning rate"
        )

    scheduler = getattr(trainer.scheduler, "scheduler", trainer.scheduler)
    step = getattr(scheduler, "step", None)
    if not callable(step):
        raise TypeError("composite transition scheduler lacks step()")
    step(source_step)
    param_groups = getattr(trainer.optimizer, "param_groups", None)
    if not isinstance(param_groups, list):
        raise TypeError("composite transition optimizer lacks param_groups")
    actual_lrs = {
        group.get("name"): float(group["lr"])
        for group in param_groups
        if isinstance(group.get("name"), str)
    }
    if set(actual_lrs) != set(expected_lrs):
        raise ValueError(
            "composite transition scheduler optimizer groups drifted"
        )
    for name, expected in expected_lrs.items():
        actual = actual_lrs[name]
        if abs(actual - expected) > max(1.0e-14, abs(expected) * 1.0e-12):
            raise ValueError(
                "composite transition scheduler did not begin at the requested "
                f"learning rate for {name!r}: {actual} vs {expected}"
            )
    scheduler_state = scheduler.state_dict()
    if scheduler_state.get("last_epoch") != source_step:
        raise ValueError(
            "composite transition scheduler did not bind absolute source step"
        )
    return {
        "policy": "fresh_absolute_cosine_from_preserved_global_step_v1",
        "absolute_start_step": source_step,
        "absolute_end_step": int(trainer.max_steps),
        "start_learning_rate": next(iter(expected_lrs.values())),
        "floor_learning_rate": float(
            next(iter(group_specs.values()))["cosine_min_learning_rate"]
        ),
    }

def _apply_fresh_stage_weights_transition(
    trainer: Any,
    *,
    resume_path: Path,
    config: Mapping[str, Any],
) -> None:
    expected_keys = {
        "enabled",
        "mode",
        "rng_seed",
        "source_global_step",
        "source_state_dir",
        "source_topology",
        "source_world_size",
        "state_dict_policy",
        "target_topology",
        "target_world_size",
        "transition_id",
    }
    if set(config) != expected_keys:
        raise ValueError(
            "fresh_stage_weights_transition keys differ from the supported "
            "contract"
        )
    if config.get("enabled") is not True:
        raise ValueError("fresh_stage_weights_transition.enabled must be true")
    transition_mode = config.get("mode")
    if transition_mode not in {
        FRESH_STAGE_WEIGHTS_TRANSITION_MODE,
        PRESERVE_CURSOR_FRESH_OPTIMIZER_SCHEDULER_MODE,
    }:
        raise ValueError("fresh_stage_weights_transition.mode is unsupported")
    transition_id = config.get("transition_id")
    fresh_objective_requirements = (
        _FRESH_OBJECTIVE_TRANSITION_REQUIREMENTS.get(transition_id)
    )
    expected_mode = (
        fresh_objective_requirements.get(
            "mode",
            FRESH_STAGE_WEIGHTS_TRANSITION_MODE,
        )
        if fresh_objective_requirements is not None
        else FRESH_STAGE_WEIGHTS_TRANSITION_MODE
    )
    if transition_mode != expected_mode:
        raise ValueError(
            "fresh_stage_weights_transition.mode disagrees with the "
            "allowlisted transition identity"
        )
    preserve_cursor = (
        transition_mode == PRESERVE_CURSOR_FRESH_OPTIMIZER_SCHEDULER_MODE
    )
    native_kv_anneal_start_global_step = None
    if fresh_objective_requirements is not None:
        configured_anneal_start = fresh_objective_requirements.get(
            "native_kv_anneal_start_global_step"
        )
        if configured_anneal_start is not None:
            native_kv_anneal_start_global_step = _nonnegative_int(
                configured_anneal_start,
                name="fresh-stage V9 native-K/V anneal start step",
            )
    allowed_target_training_stages = (
        fresh_objective_requirements.get(
            "target_training_stages",
            frozenset({"joint"}),
        )
        if fresh_objective_requirements is not None
        else frozenset({"joint"})
    )
    target_training_stage = getattr(trainer, "training_stage", None)
    if target_training_stage not in allowed_target_training_stages:
        if allowed_target_training_stages == frozenset({"joint"}):
            raise ValueError(
                "fresh-stage topology transition is allowed only for joint "
                "training"
            )
        raise ValueError(
            "fresh-stage topology transition target training_stage must be one "
            f"of {sorted(allowed_target_training_stages)}"
        )
    if getattr(trainer, "weights_only_warmstart", None) is not None:
        raise ValueError(
            "fresh-stage transition and weights_only_warmstart are mutually exclusive"
        )

    is_fresh_objective_transition = fresh_objective_requirements is not None
    source_step = _positive_int(
        config.get("source_global_step"),
        name="fresh_stage_weights_transition.source_global_step",
    )
    if (
        fresh_objective_requirements is not None
        and source_step
        != fresh_objective_requirements["source_global_step"]
    ):
        raise ValueError(
            "H8-S4 high-noise fresh-objective transition requires "
            "its exact source_global_step"
        )
    source_world_size = _positive_int(
        config.get("source_world_size"),
        name="fresh_stage_weights_transition.source_world_size",
    )
    target_world_size = _positive_int(
        config.get("target_world_size"),
        name="fresh_stage_weights_transition.target_world_size",
    )
    rng_seed = _positive_int(
        config.get("rng_seed"),
        name="fresh_stage_weights_transition.rng_seed",
    )
    if fresh_objective_requirements is not None and (
        source_world_size
        != fresh_objective_requirements["source_world_size"]
        or target_world_size
        != fresh_objective_requirements["target_world_size"]
        or rng_seed != fresh_objective_requirements["rng_seed"]
    ):
        raise ValueError(
            "H8-S4 high-noise fresh-objective transition requires "
            "its exact source/target world sizes and RNG seed"
        )
    if rng_seed >= 2**32:
        raise ValueError("fresh_stage_weights_transition.rng_seed exceeds uint32")
    if int(trainer.accelerator.num_processes) != target_world_size:
        raise ValueError(
            "fresh-stage transition target_world_size differs from runtime"
        )

    source_state_value = config.get("source_state_dir")
    if not isinstance(source_state_value, str) or not source_state_value:
        raise ValueError(
            "fresh_stage_weights_transition.source_state_dir must be a path"
        )
    requested_source_state = Path(source_state_value)
    if not requested_source_state.is_absolute():
        raise ValueError(
            "fresh_stage_weights_transition.source_state_dir must be absolute"
        )
    if requested_source_state.is_symlink():
        raise ValueError("fresh-stage transition source state may not be a symlink")
    source_state = requested_source_state.resolve(strict=True)
    if not source_state.is_dir():
        raise ValueError("fresh-stage transition source state must be a directory")
    if is_fresh_objective_transition:
        if not resume_path.is_absolute():
            raise ValueError(
                "H8-S4 high-noise fresh-objective weights path must be absolute"
            )
        if resume_path.is_symlink() or not resume_path.is_file():
            raise ValueError(
                "H8-S4 high-noise fresh-objective weights path must be a real file"
            )
        expected_weights_path = (
            source_state.parent.parent / "weights" / resume_path.name
        )
        if resume_path.resolve(strict=True) != expected_weights_path.resolve(
            strict=False
        ):
            raise ValueError(
                "H8-S4 high-noise source weights and state must share one "
                "checkpoint root"
            )
    step_tag = f"step_{source_step:06d}"
    if source_state.name != step_tag or resume_path.stem != step_tag:
        raise ValueError(
            "fresh-stage transition weights/state paths disagree with source step"
        )

    source_audit = audit_state_directory(source_state)
    source_trainer_state = source_audit["trainer_state"]
    source_checkpoint_meta = source_audit["checkpoint_meta"]
    if source_trainer_state.get("global_step") != source_step:
        raise ValueError("fresh-stage source trainer step differs from config")
    if source_checkpoint_meta.get("global_step") != source_step:
        raise ValueError("fresh-stage source checkpoint step differs from config")
    if source_checkpoint_meta.get("world_size") != source_world_size:
        raise ValueError("fresh-stage source world size differs from config")
    source_resume_contract = source_trainer_state.get("resume_contract")
    if not isinstance(source_resume_contract, Mapping):
        raise ValueError("fresh-stage source lacks resume_contract")
    source_training = source_resume_contract.get("training")
    if not isinstance(source_training, Mapping):
        raise ValueError("fresh-stage source lacks training contract")
    source_training_stage = source_training.get("training_stage", "joint")
    expected_source_training_stage = (
        fresh_objective_requirements.get("source_training_stage", "joint")
        if is_fresh_objective_transition
        else "world_only"
    )
    if source_training_stage != expected_source_training_stage:
        raise ValueError(
            "fresh-stage source must be a "
            f"{expected_source_training_stage} checkpoint"
        )

    if any(
        int(getattr(trainer, name, 0)) != 0
        for name in (
            "global_step",
            "optimizer_updates_attempted",
            "optimizer_updates_successful",
            "optimizer_updates_skipped",
            "epoch",
            "batch_in_epoch",
        )
    ):
        raise RuntimeError(
            "fresh-stage transition must run before any target-stage progress"
        )

    model = trainer.accelerator.unwrap_model(trainer.model)
    loader = getattr(
        model,
        "load_checkpoint_for_fresh_stage_transition",
        None,
    )
    if not callable(loader):
        raise TypeError(
            "model must implement load_checkpoint_for_fresh_stage_transition"
        )
    anneal_contract = None
    if preserve_cursor or native_kv_anneal_start_global_step is not None:
        configure_anneal = getattr(
            model,
            "configure_controlled_native_kv_count_correction_anneal",
            None,
        )
        if not callable(configure_anneal):
            raise TypeError(
                "fresh-stage transition target model lacks V9 anneal binding"
            )
        anneal_start_step = (
            source_step
            if preserve_cursor
            else native_kv_anneal_start_global_step
        )
        if anneal_start_step is None:
            raise AssertionError("V9 anneal start step was not resolved")
        anneal_value = configure_anneal(
            source_global_step=anneal_start_step,
            transition_steps=500,
            transition_curve="cosine",
            transition_id=transition_id,
            source_native_kv_fusion=(
                "world_then_action_token_concat_log_count_corrected_v1"
            ),
            target_native_kv_fusion=(
                "world_then_action_token_concat_log_count_cosine_anneal_to_none_v1"
            ),
        )
        if not isinstance(anneal_value, Mapping):
            raise TypeError(
                "fresh-stage V9 anneal binding must return a mapping"
            )
        anneal_contract = anneal_value

    transition_report = loader(
        str(resume_path),
        transition_id=transition_id,
        source_topology=config.get("source_topology"),
        target_topology=config.get("target_topology"),
        state_dict_policy=config.get("state_dict_policy"),
    )
    if not isinstance(transition_report, Mapping):
        raise TypeError("fresh-stage model transition report must be a mapping")
    if transition_report.get("source_step") != source_step:
        raise ValueError(
            "fresh-stage weights payload step differs from source state/config"
        )
    _refresh_deepspeed_fp32_master_params_after_weights_only_load(trainer)
    cursor_metadata: dict[str, int] | None = None
    scheduler_metadata: dict[str, object] | None = None
    if preserve_cursor:
        cursor_metadata = _map_preserved_composite_sampler_cursor(
            trainer,
            source_trainer_state=source_trainer_state,
            source_resume_contract=source_resume_contract,
            source_step=source_step,
            source_world_size=source_world_size,
            target_world_size=target_world_size,
        )
        scheduler_metadata = _initialize_composite_absolute_cosine_scheduler(
            trainer,
            source_step=source_step,
        )
        # Cursor evidence validates scheduler.last_epoch against successful
        # updates. It must run only after the fresh scheduler is explicitly
        # positioned at the preserved V8 global step.
        trainer._collective_checkpoint_cursor_evidence(
            expected_step=source_step,
            phase="composite-v8-v9-cursor-map-after-scheduler-position",
        )



    set_global_seed(rng_seed)
    lineage = {
        "schema_version": FRESH_STAGE_WEIGHTS_TRANSITION_SCHEMA_VERSION,
        "mode": transition_mode,
        "transition_id": transition_id,
        "source_weights_path": str(resume_path.resolve(strict=True)),
        "source_state_path": str(source_state),
        "source_global_step": source_step,
        "source_world_size": source_world_size,
        "target_world_size": target_world_size,
        "source_training_stage": source_training_stage,
        "target_training_stage": target_training_stage,
        "optimizer_state": "fresh",
        "scheduler_state": (
            "fresh_absolute_cosine_at_source_step"
            if preserve_cursor
            else "fresh"
        ),
        "trainer_step": source_step if preserve_cursor else 0,
        "sampler_cursor": (
            "mapped_preserved_epoch_batch_optimizer_boundary"
            if preserve_cursor
            else "fresh_epoch0_batch0"
        ),
        "rng_state": "fresh_reseeded",
        "rng_seed": rng_seed,
        "model_transition": dict(transition_report),
    }
    if preserve_cursor:
        if cursor_metadata is None or scheduler_metadata is None:
            raise AssertionError(
                "composite transition failed to record cursor/scheduler metadata"
            )
        lineage["cursor_mapping"] = cursor_metadata
        lineage["scheduler_position"] = scheduler_metadata
        lineage["bias_anneal"] = {
            "scale_at_absolute_step_9502": 1.0,
            "scale_at_absolute_step_10002": 0.0,
            "transition_optimizer_steps": 500,
        }
    elif native_kv_anneal_start_global_step is not None:
        if anneal_contract is None:
            raise AssertionError("fresh step-0 V9 anneal binding was not recorded")
        lineage["bias_anneal"] = {
            "target_start_global_step": native_kv_anneal_start_global_step,
            "scale_at_target_step_000000": 1.0,
            "scale_at_target_step_000250": 0.5,
            "scale_at_target_step_000500": 0.0,
            "transition_optimizer_steps": 500,
            "contract": dict(anneal_contract),
        }
    trainer.fresh_stage_weights_transition_source = lineage
    _write_fresh_stage_transition_lineage(trainer, lineage)
    trainer.accelerator.wait_for_everyone()
    if preserve_cursor:
        logger.warning(
            "Applied composite V8->V9 joint transition %s at preserved "
            "step=%d: complete model weights loaded; Adam is fresh; sampler "
            "cursor is mapped; absolute cosine starts at this step; and the "
            "legacy count correction reaches zero after 500 successful updates.",
            transition_id,
            source_step,
        )
        return
    if native_kv_anneal_start_global_step is not None:
        logger.warning(
            "Applied fresh step-0 V9 joint transition %s from source step=%d: "
            "World and allowlisted Action weights imported; Adam/scheduler/"
            "step/sampler are fresh; count correction is 1 at step 0, "
            "0.5 at step 250, and 0 at step 500.",
            transition_id,
            source_step,
        )
        return
    if is_fresh_objective_transition:
        logger.warning(
            "Applied explicit joint-to-%s fresh-objective weights transition "
            "%s from step=%d; Adam/scheduler/step/sampler are fresh and RNG "
            "seed=%d.",
            target_training_stage,
            transition_id,
            source_step,
            rng_seed,
        )
    else:
        logger.warning(
            "Applied explicit fresh Stage2 weights transition %s from World "
            "step=%d; Adam/scheduler/step/sampler are fresh and RNG seed=%d.",
            transition_id,
            source_step,
            rng_seed,
        )


def _apply_weights_only_warmstart(
    trainer: Any,
    *,
    resume_path: Path,
    config: Mapping[str, Any],
) -> None:
    expected_keys = {
        "enabled",
        "resume_mode",
        "rng_seed",
        "source_global_step",
        "source_state_dir",
        "source_world_size",
        "target_world_size",
    }
    if set(config) != expected_keys:
        raise ValueError(
            "weights_only_warmstart keys differ from the supported contract"
        )
    if config.get("enabled") is not True:
        raise ValueError("weights_only_warmstart.enabled must be true")
    if config.get("resume_mode") != WEIGHTS_ONLY_WARMSTART_MODE:
        raise ValueError(
            "weights_only_warmstart.resume_mode is unsupported"
        )

    source_step = _positive_int(
        config.get("source_global_step"),
        name="weights_only_warmstart.source_global_step",
    )
    source_world_size = _positive_int(
        config.get("source_world_size"),
        name="weights_only_warmstart.source_world_size",
    )
    target_world_size = _positive_int(
        config.get("target_world_size"),
        name="weights_only_warmstart.target_world_size",
    )
    rng_seed = _positive_int(
        config.get("rng_seed"),
        name="weights_only_warmstart.rng_seed",
    )
    if rng_seed >= 2**32:
        raise ValueError("weights_only_warmstart.rng_seed exceeds uint32")
    active_world_size = int(trainer.accelerator.num_processes)
    if active_world_size != target_world_size:
        raise ValueError(
            "weights-only warm-start target_world_size differs from runtime"
        )

    source_state_value = config.get("source_state_dir")
    if not isinstance(source_state_value, str) or not source_state_value:
        raise ValueError(
            "weights_only_warmstart.source_state_dir must be a path"
        )
    requested_source_state = Path(source_state_value)
    if not requested_source_state.is_absolute():
        raise ValueError(
            "weights_only_warmstart.source_state_dir must be absolute"
        )
    if requested_source_state.is_symlink():
        raise ValueError("weights-only warm-start source may not be a symlink")
    source_state = requested_source_state.resolve(strict=True)
    if not source_state.is_dir():
        raise ValueError("weights-only warm-start source must be a directory")
    step_tag = f"step_{source_step:06d}"
    if source_state.name != step_tag or resume_path.stem != step_tag:
        raise ValueError(
            "weights and state paths do not match source_global_step"
        )

    trainer_state = _read_json_object(
        source_state / "trainer_state.json",
        name="source trainer_state",
    )
    checkpoint_meta = _read_json_object(
        source_state / "checkpoint_meta.json",
        name="source checkpoint metadata",
    )
    if trainer_state.get("global_step") != source_step:
        raise ValueError("source trainer_state global_step differs from config")
    if checkpoint_meta.get("global_step") != source_step:
        raise ValueError("source checkpoint metadata step differs from config")
    if checkpoint_meta.get("world_size") != source_world_size:
        raise ValueError("source checkpoint world_size differs from config")

    resume_contract = trainer_state.get("resume_contract")
    if not isinstance(resume_contract, dict):
        raise ValueError("source trainer_state lacks resume_contract")
    source_sampler = resume_contract.get("sampler")
    if not isinstance(source_sampler, dict):
        raise ValueError("source resume contract lacks sampler binding")
    target_sampler = trainer.train_sampler.binding()
    if not isinstance(target_sampler, dict):
        raise ValueError("active sampler binding must be a mapping")
    compared_fields = (
        "dataset_length",
        "seed",
        "effective_global_batch_size",
        "padded_samples_per_epoch",
        "optimizer_updates_per_epoch",
        "tail_policy",
    )
    for field in compared_fields:
        if source_sampler.get(field) != target_sampler.get(field):
            raise ValueError(
                f"weights-only warm-start sampler field {field!r} drifted"
            )
    if source_sampler.get("world_size") != source_world_size:
        raise ValueError("source sampler world_size differs from config")
    if target_sampler.get("world_size") != target_world_size:
        raise ValueError("target sampler world_size differs from runtime")

    source_epoch = _nonnegative_int(
        trainer_state.get("epoch"),
        name="source trainer_state epoch",
    )
    source_batch = _nonnegative_int(
        trainer_state.get("batch_in_epoch"),
        name="source trainer_state batch_in_epoch",
    )
    source_microbatch = _positive_int(
        source_sampler.get("global_microbatch_size"),
        name="source sampler global_microbatch_size",
    )
    source_batches_per_epoch = _positive_int(
        source_sampler.get("global_batches_per_epoch"),
        name="source sampler global_batches_per_epoch",
    )
    source_accumulation = _positive_int(
        source_sampler.get("gradient_accumulation_steps"),
        name="source sampler gradient_accumulation_steps",
    )
    source_updates_per_epoch = _positive_int(
        source_sampler.get("optimizer_updates_per_epoch"),
        name="source sampler optimizer_updates_per_epoch",
    )
    if source_batch > source_batches_per_epoch:
        raise ValueError("source sampler cursor exceeds its epoch")
    if source_batch % source_accumulation != 0:
        raise ValueError("source sampler cursor is not an update boundary")
    source_cursor_step = (
        source_epoch * source_updates_per_epoch
        + source_batch // source_accumulation
    )
    if source_cursor_step != source_step:
        raise ValueError("source sampler cursor disagrees with global_step")

    samples_per_epoch = _positive_int(
        source_sampler.get("padded_samples_per_epoch"),
        name="source sampler padded_samples_per_epoch",
    )
    consumed_samples = (
        source_epoch * samples_per_epoch
        + source_batch * source_microbatch
    )
    target_microbatch = _positive_int(
        target_sampler.get("global_microbatch_size"),
        name="target sampler global_microbatch_size",
    )
    target_accumulation = _positive_int(
        target_sampler.get("gradient_accumulation_steps"),
        name="target sampler gradient_accumulation_steps",
    )
    target_updates_per_epoch = _positive_int(
        target_sampler.get("optimizer_updates_per_epoch"),
        name="target sampler optimizer_updates_per_epoch",
    )
    target_epoch, target_sample_offset = divmod(
        consumed_samples,
        samples_per_epoch,
    )
    target_batch, remainder = divmod(
        target_sample_offset,
        target_microbatch,
    )
    if remainder != 0 or target_batch % target_accumulation != 0:
        raise ValueError(
            "source sample cursor cannot map to a target update boundary"
        )
    target_cursor_step = (
        target_epoch * target_updates_per_epoch
        + target_batch // target_accumulation
    )
    if target_cursor_step != source_step:
        raise ValueError("mapped target sampler cursor changes global_step")

    scheduler_path = source_state / "scheduler.bin"
    scheduler_state = torch.load(
        scheduler_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(scheduler_state, dict):
        raise ValueError("source scheduler state must be a mapping")
    if scheduler_state.get("last_epoch") != source_step:
        raise ValueError("source scheduler cursor disagrees with global_step")
    trainer.scheduler.load_state_dict(scheduler_state)
    last_lr = scheduler_state.get("_last_lr")
    param_groups = getattr(trainer.optimizer, "param_groups", None)
    if (
        not isinstance(last_lr, list)
        or not isinstance(param_groups, list)
        or len(last_lr) != len(param_groups)
    ):
        raise ValueError("source scheduler learning-rate groups drifted")
    for group, learning_rate in zip(param_groups, last_lr, strict=True):
        group["lr"] = float(learning_rate)

    trainer.global_step = source_step
    trainer.optimizer_updates_attempted = source_step
    trainer.optimizer_updates_successful = source_step
    trainer.optimizer_updates_skipped = 0
    trainer.epoch = int(target_epoch)
    trainer.batch_in_epoch = int(target_batch)
    trainer.train_sampler.set_epoch_offset(0)
    trainer._set_train_data_epoch(trainer.epoch)
    trainer.train_sampler.set_resume_batch_offset(trainer.batch_in_epoch)
    if trainer._is_deepspeed_runtime():
        if not hasattr(trainer.model, "global_steps") or not hasattr(
            trainer.model,
            "global_samples",
        ):
            raise RuntimeError("DeepSpeed engine lacks global cursor fields")
        trainer.model.global_steps = source_step
        trainer.model.global_samples = consumed_samples

    trainer._collective_checkpoint_cursor_evidence(
        expected_step=source_step,
        phase="weights-only-warmstart",
    )
    lineage = {
        "schema_version": WEIGHTS_ONLY_WARMSTART_SCHEMA_VERSION,
        "resume_mode": WEIGHTS_ONLY_WARMSTART_MODE,
        "source_weights_path": str(resume_path.resolve(strict=True)),
        "source_state_path": str(source_state),
        "source_global_step": source_step,
        "source_world_size": source_world_size,
        "target_world_size": target_world_size,
        "consumed_samples": consumed_samples,
        "target_epoch": int(target_epoch),
        "target_batch_in_epoch": int(target_batch),
        "optimizer_state": "fresh",
        "scheduler_state": "restored",
        "sampler_state": "sample_cursor_mapped",
        "rng_state": "reseeded",
        "rng_seed": rng_seed,
    }
    trainer.weights_only_warmstart_source = lineage
    _write_warmstart_lineage(trainer, lineage)
    set_global_seed(rng_seed)
    trainer.accelerator.wait_for_everyone()
    logger.warning(
        "Applied weights-only topology warm-start at cumulative step=%d: "
        "world_size %d->%d, fresh Adam, scheduler restored, sampler "
        "cursor mapped to epoch=%d batch=%d, RNG reseeded=%d.",
        source_step,
        source_world_size,
        target_world_size,
        target_epoch,
        target_batch,
        rng_seed,
    )


def _model_state_schema(trainer: Any) -> dict[str, Any]:
    unwrap = getattr(trainer.accelerator, "unwrap_model", None)
    model = unwrap(trainer.model) if callable(unwrap) else trainer.model
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("checkpoint model must expose state_dict()")
    return tensor_state_dict_schema(
        state_dict(),
        name="checkpoint model state_dict",
    )


def _runtime_scaler_required(trainer: Any) -> bool:
    return getattr(trainer.accelerator, "scaler", None) is not None



def _bind_v8_step9502_v9_joint_anneal_for_full_state_resume(
    trainer: Any,
    *,
    config: Mapping[str, Any],
) -> None:
    """Bind the one V8-to-V9 anneal before an exact target-state resume."""

    expected_keys = {
        "enabled",
        "mode",
        "rng_seed",
        "source_global_step",
        "source_state_dir",
        "source_topology",
        "source_world_size",
        "state_dict_policy",
        "target_topology",
        "target_world_size",
        "transition_id",
    }
    if set(config) != expected_keys:
        raise ValueError(
            "V9 full-state resume fresh-stage transition keys differ from the "
            "allowlisted composite contract"
        )
    if config.get("enabled") is not True:
        raise ValueError("V9 full-state resume transition must be enabled")
    if config.get("transition_id") != (
        V8_ACTION14_WORLDTRANSFER_STEP9502_ACTIONONLY_TO_V9_JOINT_ANNEAL500_TRANSITION_ID
    ):
        raise ValueError(
            "fresh_stage_weights_transition must be null for this full-state "
            "resume"
        )
    if config.get("mode") != PRESERVE_CURSOR_FRESH_OPTIMIZER_SCHEDULER_MODE:
        raise ValueError("V9 full-state resume transition mode is invalid")

    requirements = _FRESH_OBJECTIVE_TRANSITION_REQUIREMENTS[
        V8_ACTION14_WORLDTRANSFER_STEP9502_ACTIONONLY_TO_V9_JOINT_ANNEAL500_TRANSITION_ID
    ]
    for field in (
        "source_global_step",
        "source_world_size",
        "target_world_size",
        "rng_seed",
    ):
        if config.get(field) != requirements[field]:
            raise ValueError(
                "V9 full-state resume transition does not match its exact "
                f"allowlisted {field}"
            )
    expected_topology = "world_g26_p4_action30_linear14_aligned_native_kv_v8"
    if (
        config.get("source_topology") != expected_topology
        or config.get("target_topology") != expected_topology
        or config.get("state_dict_policy") != "exact_schema"
    ):
        raise ValueError("V9 full-state resume topology contract is invalid")
    if getattr(trainer, "training_stage", None) != "joint":
        raise ValueError("V9 full-state resume is allowed only for joint training")
    if int(trainer.accelerator.num_processes) != requirements["target_world_size"]:
        raise ValueError("V9 full-state resume world size differs from contract")

    source_state_value = config.get("source_state_dir")
    if not isinstance(source_state_value, str) or not source_state_value:
        raise ValueError("V9 full-state resume requires the audited V8 source state")
    source_state = Path(source_state_value)
    if not source_state.is_absolute() or source_state.is_symlink():
        raise ValueError("V9 full-state resume source state path is invalid")
    source_state = source_state.resolve(strict=True)
    if not source_state.is_dir() or source_state.name != "step_009502":
        raise ValueError("V9 full-state resume source state must be V8 step_009502")

    unwrap = getattr(trainer.accelerator, "unwrap_model", None)
    model = unwrap(trainer.model) if callable(unwrap) else trainer.model
    configure = getattr(
        model,
        "configure_controlled_native_kv_count_correction_anneal",
        None,
    )
    if not callable(configure):
        raise TypeError("V9 full-state resume target lacks anneal binding")
    anneal_contract = configure(
        source_global_step=9502,
        transition_steps=500,
        transition_curve="cosine",
        transition_id=(
            V8_ACTION14_WORLDTRANSFER_STEP9502_ACTIONONLY_TO_V9_JOINT_ANNEAL500_TRANSITION_ID
        ),
        source_native_kv_fusion=(
            "world_then_action_token_concat_log_count_corrected_v1"
        ),
        target_native_kv_fusion=(
            "world_then_action_token_concat_log_count_cosine_anneal_to_none_v1"
        ),
    )
    if not isinstance(anneal_contract, Mapping):
        raise TypeError("V9 full-state resume anneal binding must return a mapping")
    expected_binding = {
        "transition_id": (
            V8_ACTION14_WORLDTRANSFER_STEP9502_ACTIONONLY_TO_V9_JOINT_ANNEAL500_TRANSITION_ID
        ),
        "source_global_step": 9502,
        "transition_steps": 500,
        "transition_curve": "cosine",
        "source_native_kv_fusion": (
            "world_then_action_token_concat_log_count_corrected_v1"
        ),
        "target_native_kv_fusion": (
            "world_then_action_token_concat_log_count_cosine_anneal_to_none_v1"
        ),
    }
    if any(anneal_contract.get(key) != value for key, value in expected_binding.items()):
        raise ValueError("V9 full-state resume anneal binding drifted")


def _set_v9_joint_anneal_runtime_step_after_full_state_resume(
    trainer: Any,
) -> None:
    """Make the first post-resume forward use the restored optimizer step."""

    unwrap = getattr(trainer.accelerator, "unwrap_model", None)
    model = unwrap(trainer.model) if callable(unwrap) else trainer.model
    set_runtime_step = getattr(model, "set_attention_runtime_step", None)
    if not callable(set_runtime_step):
        raise TypeError("V9 full-state resume target lacks attention-step binding")
    set_runtime_step(global_step=int(trainer.global_step))

def resume_or_load_checkpoint(trainer: Any) -> None:
    """Restore a full state directory or load a weights-only checkpoint."""

    resume = trainer.resume
    if not resume:
        return
    resume_path = Path(str(resume))
    if resume_path.is_dir():
        if getattr(trainer, "weights_only_warmstart", None) is not None:
            raise ValueError(
                "weights_only_warmstart must be null for full-state resume"
            )
        fresh_transition = getattr(
            trainer,
            "fresh_stage_weights_transition",
            None,
        )
        if fresh_transition is not None:
            if not isinstance(fresh_transition, Mapping):
                raise TypeError(
                    "fresh_stage_weights_transition must be a mapping or null"
                )
            _bind_v8_step9502_v9_joint_anneal_for_full_state_resume(
                trainer,
                config=fresh_transition,
            )
            logger.info(
                "Resuming exact V9 joint state with the bound V8 step_9502 "
                "count-correction anneal: %s",
                resume,
            )
            trainer.load_training_state(str(resume_path))
            _set_v9_joint_anneal_runtime_step_after_full_state_resume(trainer)
            return
        logger.info("Resuming full training state from directory: %s", resume)
        trainer.load_training_state(str(resume_path))
        apply_scheduler_floor_migration = getattr(
            trainer,
            "_apply_controlled_scheduler_floor_migration_after_resume",
            None,
        )
        if callable(apply_scheduler_floor_migration):
            apply_scheduler_floor_migration()
        from .plus_lr_correction import apply_after_resume as apply_plus_lr_correction
        apply_plus_lr_correction(trainer)
        from .plus_50k_gradient import apply_after_resume as apply_plus_50k
        apply_plus_50k(trainer)
        from .libero_50k_extension import apply_after_resume as apply_libero_50k
        apply_libero_50k(trainer)
        from .real_budget_extension import apply_after_resume as apply_real_extension
        apply_real_extension(trainer)
        from .libero_transfer_extension import apply_after_resume as apply_transfer_extension
        apply_transfer_extension(trainer)
        return
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
    logger.info("Loading weight checkpoint only: %s", resume)
    fresh_transition = getattr(
        trainer,
        "fresh_stage_weights_transition",
        None,
    )
    if fresh_transition is not None:
        _apply_fresh_stage_weights_transition(
            trainer,
            resume_path=resume_path,
            config=fresh_transition,
        )
        return
    trainer.accelerator.unwrap_model(trainer.model).load_checkpoint(
        str(resume_path),
        optimizer=None,
    )
    _refresh_deepspeed_fp32_master_params_after_weights_only_load(trainer)
    warmstart = getattr(trainer, "weights_only_warmstart", None)
    if warmstart is not None:
        _apply_weights_only_warmstart(
            trainer,
            resume_path=resume_path,
            config=warmstart,
        )
        return
    logger.warning(
        "Loaded .pt weights only; optimizer/scheduler/step were not restored "
        "under ZeRO2."
    )


def _save_weights_checkpoint(trainer: Any, *, step_tag: str) -> str:
    model = trainer.accelerator.unwrap_model(trainer.model)
    checkpoint_path = os.path.join(trainer.weights_dir, f"{step_tag}.pt")
    model.save_checkpoint(
        checkpoint_path,
        optimizer=None,
        step=trainer.global_step,
    )
    return checkpoint_path


def _save_trainer_state(
    trainer: Any,
    *,
    state_path: str,
    rank_state: list[dict[str, Any]],
) -> None:
    global_step = int(trainer.global_step)
    world_size = int(trainer.accelerator.num_processes)
    deepspeed_required = trainer._is_deepspeed_runtime()
    trainer._validate_training_cursor(
        global_step=global_step,
        epoch=int(trainer.epoch),
        batch_in_epoch=int(trainer.batch_in_epoch),
    )
    normalized_rank_state = validate_checkpoint_rank_state(
        rank_state,
        global_step=global_step,
        world_size=world_size,
        deepspeed_required=deepspeed_required,
    )
    state_file = Path(state_path) / "trainer_state.json"
    payload = {
        "checkpoint_state_schema_version": (
            TRAINER_CHECKPOINT_STATE_SCHEMA_VERSION
        ),
        "global_step": global_step,
        "optimizer_updates": trainer._optimizer_update_counters(),
        "epoch": int(trainer.epoch),
        "batch_in_epoch": int(trainer.batch_in_epoch),
        "resume_contract": trainer._resume_contract(),
        "model_state_schema": _model_state_schema(trainer),
        "rank_state": normalized_rank_state,
        "loaded_resume_source": trainer.loaded_resume_source,
    }
    task_state = getattr(trainer.train_sampler, 'task_replay_state', None)
    if task_state is not None:
        payload['task_replay_state'] = task_state
    feedback_state = getattr(trainer.train_sampler, "feedback_state", None)
    scene_state = getattr(trainer.train_sampler, 'scene_replay_state', None)
    if scene_state is not None:
        payload['scene_replay_state'] = scene_state
    if feedback_state is not None:
        from .libero_feedback import STATE_SCHEMA as LIBERO_REPLAY_STATE_SCHEMA
        key = 'libero_failure_replay_state' if feedback_state.get('schema_version')==LIBERO_REPLAY_STATE_SCHEMA else 'feedback_continuation_state'
        payload[key] = feedback_state
    runtime_execution = getattr(trainer, "runtime_execution", None)
    if runtime_execution is not None:
        payload["runtime_execution"] = runtime_execution
    adaptive_replay_state = adaptive_replay_state_snapshot(trainer)
    if adaptive_replay_state is not None:
        payload["adaptive_replay_state"] = adaptive_replay_state
    pending = state_file.with_name(f".{state_file.name}.tmp.{os.getpid()}")
    try:
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, state_file)
        directory_descriptor = os.open(
            state_file.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass


def save_checkpoint(trainer: Any) -> dict[str, str | None]:
    """Save weights plus an atomically published resume directory."""

    step = int(trainer.global_step)
    step_tag = f"step_{step:06d}"
    final_path = Path(trainer.state_dir) / step_tag
    if trainer._last_checkpoint_step == step:
        return {
            "weights_path": (
                os.path.join(trainer.weights_dir, f"{step_tag}.pt")
                if trainer.accelerator.is_main_process
                else None
            ),
            "state_path": str(final_path),
        }

    trainer._validate_training_cursor(
        global_step=step,
        epoch=int(trainer.epoch),
        batch_in_epoch=int(trainer.batch_in_epoch),
    )
    trainer._optimizer_update_counters()
    trainer._collective_checkpoint_cursor_evidence(
        expected_step=step,
        phase="pre-save",
    )

    def _raise_phase_errors(
        local_error: Exception | None,
        *,
        phase: str,
    ) -> None:
        local_status = (
            None
            if local_error is None
            else {
                "error_type": type(local_error).__name__,
                "message": str(local_error),
            }
        )
        statuses = trainer._all_gather_rank_objects(local_status)
        if any(status is not None for status in statuses):
            if len(statuses) == 1 and local_error is not None:
                raise local_error
            raise RuntimeError(
                f"{phase} failed on one or more ranks: "
                f"{json.dumps(statuses, sort_keys=True)}"
            )

    checkpoint_path = None
    local_error: Exception | None = None
    if trainer.accelerator.is_main_process:
        try:
            checkpoint_path = trainer._save_weights_checkpoint(step_tag=step_tag)
        except Exception as exc:
            local_error = exc
    _raise_phase_errors(local_error, phase="weights checkpoint save")

    pending_path = trainer._collective_pending_state_path(
        final_path=final_path
    )
    local_error = None
    try:
        trainer.accelerator.save_state(output_dir=str(pending_path))
    except Exception as exc:
        local_error = exc
    _raise_phase_errors(local_error, phase="Accelerate/DeepSpeed state save")

    rank_state = trainer._collective_checkpoint_cursor_evidence(
        expected_step=step,
        phase="post-save",
    )
    rank = int(trainer.accelerator.process_index)
    local_error = None
    try:
        rng_path = pending_path / f"random_states_{rank}.pkl"
        if (
            rng_path.is_symlink()
            or not rng_path.is_file()
            or rng_path.stat().st_size <= 0
        ):
            raise FileNotFoundError(
                f"Accelerate did not publish rank RNG state: {rng_path}"
            )
    except Exception as exc:
        local_error = exc
    _raise_phase_errors(local_error, phase="per-rank RNG state validation")

    local_error = None
    if rank == 0:
        try:
            trainer._save_trainer_state(
                str(pending_path),
                rank_state=rank_state,
            )
            write_checkpoint_metadata(
                pending_path,
                global_step=step,
                world_size=int(trainer.accelerator.num_processes),
                deepspeed_required=trainer._is_deepspeed_runtime(),
                scaler_required=_runtime_scaler_required(trainer),
            )
            publish_checkpoint(pending_path, final_path)
        except Exception as exc:
            local_error = exc
    _raise_phase_errors(local_error, phase="atomic checkpoint publication")

    published = trainer._collective_state_directory_audit(final_path)
    published_state = published["resume_state"]
    if (
        published_state.get("global_step") != step
        or published_state.get("world_size")
        != int(trainer.accelerator.num_processes)
    ):
        raise RuntimeError(
            "published checkpoint resume metadata disagrees with runtime"
        )
    trainer._last_checkpoint_step = step
    return {
        "weights_path": checkpoint_path,
        "state_path": str(final_path),
    }


def load_training_state(trainer: Any, state_dir: str) -> None:
    """Load and verify a distributed training-state directory."""

    requested_path = Path(state_dir)
    if requested_path.is_symlink():
        raise ValueError("resume state path must not be a symlink")
    resolved_path = requested_path.resolve(strict=True)
    before_audit = trainer._collective_state_directory_audit(resolved_path)
    resume_state = before_audit["resume_state"]
    world_size = int(trainer.accelerator.num_processes)
    deepspeed_required = trainer._is_deepspeed_runtime()
    scaler_required = _runtime_scaler_required(trainer)
    if resume_state.get("world_size") != world_size:
        raise ValueError(
            "checkpoint world_size differs from the active distributed runtime"
        )
    if resume_state.get("deepspeed_required") != deepspeed_required:
        raise ValueError(
            "checkpoint DeepSpeed runtime requirement differs from the active "
            "runtime"
        )
    if resume_state.get("scaler_required") != scaler_required:
        raise ValueError(
            "checkpoint AMP scaler requirement differs from the active runtime"
        )

    payload = before_audit["trainer_state"]
    local_error: Exception | None = None
    candidate_counters: dict | None = None
    candidate_rank_state: list[dict[str, Any]] | None = None
    try:
        global_step = payload.get("global_step")
        epoch = payload.get("epoch")
        batch_in_epoch = payload.get("batch_in_epoch")
        for field_name, value in (
            ("global_step", global_step),
            ("epoch", epoch),
            ("batch_in_epoch", batch_in_epoch),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"trainer_state {field_name!r} must be a non-negative "
                    "integer"
                )
        if resume_state.get("global_step") != global_step:
            raise ValueError(
                "checkpoint resume metadata global_step differs from trainer_state"
            )
        trainer._validate_resume_contract(
            payload.get("resume_contract"),
            source_state_dir=resolved_path,
            source_global_step=global_step,
        )
        trainer._validate_training_cursor(
            global_step=global_step,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
        )
        candidate_counters = trainer._validated_optimizer_update_counters(
            payload.get("optimizer_updates"),
            global_step=global_step,
            require_zero_skipped=True,
        )

        checkpoint_schema = payload.get(
            "checkpoint_state_schema_version"
        )
        if checkpoint_schema is None:
            # Historical checkpoints rely on Accelerate/DeepSpeed's strict
            # state_dict load for name/shape/dtype compatibility. Their old
            # governance files and rank-local cursor files are irrelevant.
            candidate_rank_state = None
        elif checkpoint_schema != TRAINER_CHECKPOINT_STATE_SCHEMA_VERSION:
            raise ValueError(
                "trainer_state checkpoint schema is unsupported"
            )
        else:
            saved_model_schema = payload.get("model_state_schema")
            current_model_schema = _model_state_schema(trainer)
            if saved_model_schema != current_model_schema:
                raise ValueError(
                    "checkpoint model state_dict names/shapes/dtypes differ "
                    "from the active model"
                )
            candidate_rank_state = validate_checkpoint_rank_state(
                payload.get("rank_state"),
                global_step=global_step,
                world_size=world_size,
                deepspeed_required=deepspeed_required,
            )
    except Exception as exc:
        local_error = exc
    validation_status = (
        None
        if local_error is None
        else {
            "error_type": type(local_error).__name__,
            "message": str(local_error),
        }
    )
    gathered_validation = trainer._all_gather_rank_objects(validation_status)
    if any(status is not None for status in gathered_validation):
        if len(gathered_validation) == 1 and local_error is not None:
            raise local_error
        raise RuntimeError(
            "resume trainer-state validation failed on one or more ranks: "
            f"{json.dumps(gathered_validation, sort_keys=True)}"
        )
    if candidate_counters is None:
        raise AssertionError("validated trainer counters unexpectedly missing")

    trainer._collective_load_accelerator_state(resolved_path)
    trainer._collective_verify_loaded_rng_state(resolved_path)
    loaded_rank_state = trainer._collective_checkpoint_cursor_evidence(
        expected_step=int(payload["global_step"]),
        phase="post-load",
    )
    if (
        candidate_rank_state is not None
        and loaded_rank_state != candidate_rank_state
    ):
        raise RuntimeError(
            "restored scheduler/DeepSpeed rank state differs from trainer_state"
        )
    try:
        after_audit = trainer._collective_state_directory_audit(resolved_path)
    except Exception as exc:
        raise RuntimeError(
            "resume state directory changed while Accelerator/DeepSpeed "
            f"loaded it: {exc}"
        ) from exc
    if after_audit != before_audit:
        raise RuntimeError(
            "resume state directory changed while Accelerator/DeepSpeed "
            "loaded it"
        )

    trainer.global_step = int(payload["global_step"])
    trainer.optimizer_updates_attempted = candidate_counters["attempted"]
    trainer.optimizer_updates_successful = candidate_counters["successful"]
    trainer.optimizer_updates_skipped = candidate_counters["skipped"]
    trainer.epoch = int(payload["epoch"])
    trainer.batch_in_epoch = int(payload["batch_in_epoch"])
    trainer.loaded_resume_source = {
        "schema_version": LOADED_RESUME_SOURCE_SCHEMA_VERSION,
        "source_path": str(resolved_path),
        "source_global_step": int(payload["global_step"]),
    }
    trainer._collective_set_controlled_model_math_migration_runtime_step(
        global_step=trainer.global_step,
    )
    restore_adaptive_replay_state(
        trainer,
        payload.get("adaptive_replay_state"),
    )
    from .libero_feedback import restore as restore_libero_feedback
    restore_libero_feedback(trainer,payload)
    from .scene_replay import restore as restore_scene_replay
    restore_scene_replay(trainer,payload)
    from .task_replay import restore as restore_task_replay
    restore_task_replay(trainer,payload)
    trainer.train_sampler.set_epoch_offset(0)
    trainer._set_train_data_epoch(trainer.epoch)
    trainer.train_sampler.set_resume_batch_offset(trainer.batch_in_epoch)
    logger.info(
        "Restored dataloader progress: epoch=%d batch_in_epoch=%d "
        "sample_offset=%d",
        trainer.epoch,
        trainer.batch_in_epoch,
        trainer.batch_in_epoch
        * trainer.batch_size
        * trainer.accelerator.num_processes,
    )
    trainer.accelerator.wait_for_everyone()
    logger.info(
        "Loaded verified Accelerate training state from %s at step=%d "
        "format=%s",
        resolved_path,
        trainer.global_step,
        resume_state["format"],
    )
    configured_state_dir = getattr(trainer, "state_dir", None)
    if configured_state_dir is not None:
        same_step_destination = (
            Path(configured_state_dir)
            / f"step_{trainer.global_step:06d}"
        ).resolve(strict=False)
        if same_step_destination == resolved_path:
            trainer._last_checkpoint_step = trainer.global_step
