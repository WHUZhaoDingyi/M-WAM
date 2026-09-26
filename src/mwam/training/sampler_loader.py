"""Injectable training sampler and DataLoader construction."""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..utils.logging_config import get_logger
from ..utils.samplers import (
    FAILURE_REPLAY_80PCT_INITIAL10,
    FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL,
    INITIAL_WINDOW_10PCT_REMAINING_90PCT,
    ONE_RANDOM_WINDOW_PER_EPISODE_PER_EPOCH,
    FailureReplayInitialWindowSampler,
    FailureReplayTaskVariantNoInitialSampler,
    InitialWindowPreferredSampler,
    ResumableEpisodeWindowSampler,
    ResumableEpochSampler,
    StrictResumableEpochSampler,
)
from .adaptive_replay import manifest_for_adaptive_resume
from .resume_contract import operational_sampler_resume_state


logger = get_logger(__name__)
ALL_WINDOWS_PER_EPOCH = "all_windows_per_epoch"


def _config_get(value: object, key: str) -> object | None:
    if value is None:
        return None
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, None)
    return getattr(value, key, None)


def _failure_replay_inputs(trainer: Any) -> tuple[object, str]:
    manifest_source = getattr(trainer, "failure_replay_manifest", None)
    cfg = getattr(trainer, "cfg", None)
    if manifest_source is None:
        manifest_source = _config_get(cfg, "failure_replay_manifest")
    if isinstance(manifest_source, Mapping):
        manifest: object = dict(manifest_source)
    elif isinstance(manifest_source, (str, Path)) and str(
        manifest_source
    ).strip():
        manifest_path = Path(manifest_source).expanduser()
        if not manifest_path.is_file():
            raise ValueError(
                "failure_replay_manifest is not a regular file: "
                f"{manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"failed to read failure replay manifest: {manifest_path}"
            ) from exc
    else:
        raise ValueError(
            "failure_replay_manifest must be a JSON path or mapping"
        )

    split_name = getattr(trainer, "failure_replay_dataset_split", None)
    train_cfg = _config_get(_config_get(cfg, "data"), "train")
    if split_name is None:
        split_name = _config_get(train_cfg, "split_name")
    if split_name != "train":
        raise ValueError(
            "failure replay DataLoader requires data.train.split_name=train"
        )
    runtime_cache = getattr(
        trainer,
        "failure_replay_runtime_index_cache_dir",
        None,
    )
    if runtime_cache is None:
        runtime_cache = _config_get(train_cfg, "runtime_index_cache_dir")
    if not isinstance(runtime_cache, (str, Path)) or not str(
        runtime_cache
    ).strip():
        raise ValueError(
            "failure replay is restricted to a configured train runtime-index "
            "cache"
        )
    return manifest, str(split_name)


def build_train_loader(
    trainer: Any,
    dataset: Any,
    *,
    worker_init_fn: Any = None,
) -> DataLoader:
    """Build the default resumable sampler and DataLoader."""

    sampling_mode = getattr(
        trainer,
        "train_sampling_mode",
        ALL_WINDOWS_PER_EPOCH,
    )
    if sampling_mode == ALL_WINDOWS_PER_EPOCH:
        sampler_type = (
            StrictResumableEpochSampler
            if getattr(trainer, "strict_resumable_sampler", False)
            else ResumableEpochSampler
        )
    elif sampling_mode == ONE_RANDOM_WINDOW_PER_EPISODE_PER_EPOCH:
        sampler_type = ResumableEpisodeWindowSampler
    elif sampling_mode == INITIAL_WINDOW_10PCT_REMAINING_90PCT:
        sampler_type = InitialWindowPreferredSampler
    elif sampling_mode == FAILURE_REPLAY_80PCT_INITIAL10:
        sampler_type = FailureReplayInitialWindowSampler
    elif sampling_mode == FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL:
        sampler_type = FailureReplayTaskVariantNoInitialSampler
    else:
        raise ValueError(
            "Unsupported train_sampling_mode while building DataLoader: "
            f"{sampling_mode!r}"
        )
    sampler_kwargs = dict(
        dataset=dataset,
        seed=trainer.seed,
        batch_size=trainer.batch_size,
        num_processes=trainer.accelerator.num_processes,
        gradient_accumulation_steps=trainer.gradient_accumulation_steps,
    )
    if sampling_mode in {
        FAILURE_REPLAY_80PCT_INITIAL10,
        FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL,
    }:
        manifest, split_name = _failure_replay_inputs(trainer)
        if sampling_mode == FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL:
            manifest = manifest_for_adaptive_resume(trainer, manifest)
        sampler_kwargs.update(
            failure_manifest=manifest,
            dataset_split_name=split_name,
        )
    from .feedback_continuation import contract as feedback_contract, FeedbackSampler
    if feedback_contract(trainer.cfg) is not None:
        if sampling_mode != ALL_WINDOWS_PER_EPOCH:
            raise ValueError("Feedback continuation requires normal all-window base")
        sampler_type = FeedbackSampler
    from .libero_feedback import contract as libero_feedback_contract, LiberoFeedbackSampler
    if libero_feedback_contract(trainer.cfg) is not None:
        if sampling_mode != ALL_WINDOWS_PER_EPOCH or not trainer.strict_resumable_sampler:
            raise ValueError('LIBERO replay requires the original strict all-window sampler')
        sampler_type = LiberoFeedbackSampler
    from .scene_replay import contract as scene_contract, SceneReplaySampler, SceneReplayDataset
    scene_spec = scene_contract(trainer.cfg)
    if scene_spec is not None:
        if sampling_mode != ALL_WINDOWS_PER_EPOCH or not trainer.strict_resumable_sampler:
            raise ValueError('Scene replay requires a strict all-window base sampler')
        sampler_type = SceneReplaySampler
        sampler_kwargs['scene_spec'] = scene_spec
    from .task_replay import contract as task_contract
    task_spec = task_contract(trainer.cfg)
    if task_spec is not None and task_spec['method'] in ('plus', 'libero'):
        if sampling_mode != ALL_WINDOWS_PER_EPOCH or not trainer.strict_resumable_sampler:
            raise ValueError('Task replay requires a strict all-window base sampler')
        sampler_type = LiberoFeedbackSampler
    trainer.train_sampler = sampler_type(**sampler_kwargs)
    from .task_replay import initialize as initialize_task_replay
    initialize_task_replay(trainer)
    from .plus_task_replay import SCHEMA as PLUS_REPLAY_SCHEMA, SCHEMA20 as PLUS_REPLAY_SCHEMA20
    replay_spec = libero_feedback_contract(trainer.cfg)
    if replay_spec is not None and replay_spec.get('schema_version') in (PLUS_REPLAY_SCHEMA, PLUS_REPLAY_SCHEMA20):
        trainer.train_sampler.plus_task_replay = True
    if scene_spec is not None:
        from omegaconf import OmegaConf
        dataset = SceneReplayDataset(dataset, OmegaConf.to_container(trainer.cfg.data.train, resolve=True),
                                     replay_ratio=scene_spec['replay_ratio'])
    from .libero_feedback import initialize_stage_sampler
    if not trainer.cfg.get("task_failure_replay_config"):
        initialize_stage_sampler(trainer.train_sampler, libero_feedback_contract(trainer.cfg))
    logger.info(
        "Training sampler: mode=%s type=%s binding=%s",
        sampling_mode,
        sampler_type.__name__,
        json.dumps(
            operational_sampler_resume_state(
                trainer.train_sampler.binding()
            ),
            sort_keys=True,
        ),
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": trainer.batch_size,
        "shuffle": False,
        "sampler": trainer.train_sampler,
        "num_workers": trainer.num_workers,
        "persistent_workers": getattr(
            trainer,
            "persistent_workers",
            False,
        ),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
        "collate_fn": getattr(dataset, "collate_fn", None),
    }
    prefetch_factor = getattr(trainer, "prefetch_factor", None)
    if trainer.cfg.get("dataloader_rng_isolated", False):
        # Recreating a DataLoader iterator after resume must not consume the
        # model's CPU RNG (used by dropout/noise). Native source augmentation
        # is keyed by sample/epoch, independently of this worker-base seed.
        loader_kwargs["generator"] = torch.Generator().manual_seed(trainer.seed)
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    requires_fork_cow = bool(
        getattr(dataset, "requires_fork_cow_workers", False)
    )
    if requires_fork_cow and trainer.num_workers > 0:
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError(
                "Dataset requires fork-COW workers, but this platform "
                "does not provide the 'fork' multiprocessing context"
            )
        loader_kwargs["multiprocessing_context"] = "fork"
    return DataLoader(**loader_kwargs)


__all__ = [
    "ALL_WINDOWS_PER_EPOCH",
    "FAILURE_REPLAY_80PCT_INITIAL10",
    "FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL",
    "INITIAL_WINDOW_10PCT_REMAINING_90PCT",
    "build_train_loader",
]
