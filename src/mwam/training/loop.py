"""Default injectable Wan22 training loop."""

from __future__ import annotations

import time

import numpy as np
import torch

from ..utils.deepspeed_zero2_graph_task_cache import (
    reset_deepspeed_zero2_graph_task_cache,
)
from ..utils.logging_config import get_logger
from .adaptive_replay import AdaptiveReplayActivationRuntime
from .runtime_checkpoint_control import (
    RuntimeCheckpointControl,
    process_runtime_checkpoint_boundary,
)

logger = get_logger(__name__)


def _collect_optimizer_learning_rate_metrics(optimizer) -> dict[str, float]:
    """Return the legacy LR plus auditable values for named parameter groups."""

    param_groups = optimizer.param_groups
    if not param_groups:
        raise RuntimeError("optimizer must contain at least one parameter group")

    metrics = {"train/lr": float(param_groups[0]["lr"])}
    named_groups: set[str] = set()
    for index, param_group in enumerate(param_groups):
        if "name" not in param_group:
            continue
        group_name = param_group["name"]
        if (
            not isinstance(group_name, str)
            or not group_name
            or group_name != group_name.strip()
            or "/" in group_name
        ):
            raise ValueError(
                "optimizer parameter group name must be a non-empty, "
                f"trimmed path component; group index={index} name={group_name!r}"
            )
        if group_name in named_groups:
            raise ValueError(
                f"duplicate optimizer parameter group name: {group_name!r}"
            )
        named_groups.add(group_name)
        metrics[f"train/learning_rate/{group_name}"] = float(param_group["lr"])
    return metrics


def run_training_loop(trainer):
    """Run the default optimizer-update loop for one trainer instance."""

    self = trainer
    self._set_dit_only_train_mode()

    if self.max_steps is None:
        raise ValueError(
            "`max_steps` must be set before entering the while-step training loop."
        )

    runtime_checkpoint_control = RuntimeCheckpointControl.from_trainer(self)
    runtime_checkpoint_last_error: str | None = None
    if (
        runtime_checkpoint_control.initialization_error is not None
        and self.accelerator.is_main_process
    ):
        runtime_checkpoint_last_error = (
            runtime_checkpoint_control.initialization_error
        )
        logger.warning(
            "Runtime checkpoint control is unavailable for this session: %s",
            runtime_checkpoint_last_error,
        )

    stop_after_step = getattr(self, "stop_after_step", None)
    if stop_after_step is not None and (
        isinstance(stop_after_step, bool)
        or not isinstance(stop_after_step, int)
        or stop_after_step <= 0
        or stop_after_step > self.max_steps
    ):
        raise ValueError(
            "stop_after_step must be a positive integer no greater than "
            "the production max_steps contract"
        )
    run_target_step = (
        self.max_steps if stop_after_step is None else stop_after_step
    )
    checkpoint_run_target = (
        stop_after_step is None
        or bool(getattr(self, "checkpoint_on_stop_after_step", True))
    )
    if self.global_step > run_target_step:
        raise ValueError(
            "resume global_step is already beyond stop_after_step: "
            f"{self.global_step} > {run_target_step}"
        )
    self.run_target_step = run_target_step
    self._optimizer_update_counters()
    logger.info(
        "Starting training with production max_steps=%d run_target_step=%d.",
        self.max_steps,
        run_target_step,
    )
    if self.global_step == run_target_step:
        if checkpoint_run_target:
            ckpt_info = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info(
                    "[done] requested run target already complete at step=%d "
                    "weights=%s state=%s",
                    self.global_step,
                    ckpt_info["weights_path"],
                    ckpt_info["state_path"],
                )
        elif self.accelerator.is_main_process:
            logger.info(
                "[done] throughput-only stop_after_step already complete "
                "at step=%d; checkpoint intentionally disabled",
                self.global_step,
            )
        runtime_checkpoint_control.close(
            status="run-target-already-complete",
            global_step=self.global_step,
        )
        return

    adaptive_replay_runtime = AdaptiveReplayActivationRuntime.from_trainer(self)
    self._set_train_data_epoch(self.epoch)
    data_iter = iter(self.train_loader)
    self.run_start_step = self.global_step
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    self.run_start_time = time.perf_counter()
    self.run_microbatches_consumed = 0
    window_loss_sum = 0.0
    window_metric_sums: dict[str, float] = {}
    window_metric_keys: tuple[str, ...] | None = None
    window_microbatch_count = 0
    timing_enabled = bool(getattr(self, "training_timing_enabled", False))
    timing_uses_cuda_events = (
        timing_enabled and self.accelerator.device.type == "cuda"
    )
    timing_interval_seconds = {
        "batch_wait_and_h2d": 0.0,
        "model_forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
    }
    timing_interval_updates = 0
    timing_interval_wall_started = (
        time.perf_counter() if timing_enabled else None
    )
    timing_forward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    timing_backward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    while self.global_step < run_target_step:
        batch_wait_started = time.perf_counter() if timing_enabled else None
        try:
            sample = next(data_iter)
            if batch_wait_started is not None:
                # Accelerate's prepared DataLoader performs recursive H2D
                # before yielding, so this is intentionally wait + H2D.
                timing_interval_seconds["batch_wait_and_h2d"] += (
                    time.perf_counter() - batch_wait_started
                )
            self.run_microbatches_consumed += 1
            self.batch_in_epoch += 1
        except StopIteration:
            if window_microbatch_count != 0:
                raise RuntimeError(
                    "DataLoader ended inside a gradient-accumulation window"
                )
            self.epoch += 1
            self.batch_in_epoch = 0
            self.train_sampler.clear_resume_batch_offset()
            self._set_train_data_epoch(self.epoch)
            data_iter = iter(self.train_loader)
            continue

        with self.accelerator.accumulate(self.model):
            # V9's controlled native-K/V migration needs the number of
            # successful optimizer updates completed before this forward.
            # V8 and unrelated models do not expose this optional hook, so
            # their execution path is unchanged.
            unwrap_model = getattr(self.accelerator, "unwrap_model", None)
            unwrapped_model = (
                unwrap_model(self.model) if callable(unwrap_model) else self.model
            )
            set_attention_runtime_step = getattr(
                unwrapped_model,
                "set_attention_runtime_step",
                None,
            )
            if callable(set_attention_runtime_step):
                set_attention_runtime_step(global_step=int(self.global_step))
            forward_wall_started = None
            forward_event_start = None
            forward_event_end = None
            if timing_enabled:
                if timing_uses_cuda_events:
                    forward_event_start = torch.cuda.Event(enable_timing=True)
                    forward_event_end = torch.cuda.Event(enable_timing=True)
                    forward_event_start.record()
                else:
                    forward_wall_started = time.perf_counter()
            with self.accelerator.autocast():
                loss, loss_dict = self.model(sample)
            if timing_enabled:
                if timing_uses_cuda_events:
                    assert forward_event_start is not None
                    assert forward_event_end is not None
                    forward_event_end.record()
                    timing_forward_events.append(
                        (forward_event_start, forward_event_end)
                    )
                else:
                    assert forward_wall_started is not None
                    timing_interval_seconds["model_forward"] += (
                        time.perf_counter() - forward_wall_started
                    )
            local_finite = bool(torch.isfinite(loss.detach()).all().item())
            gathered_finite = self.accelerator.gather(
                torch.tensor(
                    [int(local_finite)],
                    device=loss.device,
                    dtype=torch.int64,
                )
            ).reshape(-1)
            if gathered_finite.numel() != int(self.accelerator.num_processes):
                raise RuntimeError(
                    "loss-finite collective did not return one flag per rank"
                )
            if bool(torch.any(gathered_finite == 0).item()):
                bad_ranks = torch.nonzero(
                    gathered_finite == 0, as_tuple=False
                ).reshape(-1).cpu().tolist()
                raise FloatingPointError(
                    "Non-finite training loss on distributed rank(s) "
                    f"{bad_ranks} before optimizer step {self.global_step}."
                )
            current_metric_keys = tuple(sorted(loss_dict))
            gathered_metric_keys = self._all_gather_rank_objects(
                current_metric_keys
            )
            if any(keys != current_metric_keys for keys in gathered_metric_keys):
                raise RuntimeError(
                    "loss metric keys differ across distributed ranks"
                )
            if window_metric_keys is None:
                window_metric_keys = current_metric_keys
            elif current_metric_keys != window_metric_keys:
                raise RuntimeError(
                    "loss metric keys changed inside an accumulation window"
                )
            window_loss_sum += float(loss.detach().float().item())
            local_metrics_finite = True
            for key in current_metric_keys:
                value = loss_dict[key]
                metric_value = float(
                    value.detach().float().item()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                if not np.isfinite(metric_value):
                    local_metrics_finite = False
                window_metric_sums[key] = (
                    window_metric_sums.get(key, 0.0) + metric_value
                )
            gathered_metrics_finite = self.accelerator.gather(
                torch.tensor(
                    [int(local_metrics_finite)],
                    device=loss.device,
                    dtype=torch.int64,
                )
            ).reshape(-1)
            if bool(torch.any(gathered_metrics_finite == 0).item()):
                bad_metric_ranks = torch.nonzero(
                    gathered_metrics_finite == 0, as_tuple=False
                ).reshape(-1).cpu().tolist()
                raise FloatingPointError(
                    "Non-finite detached loss metrics on distributed rank(s) "
                    f"{bad_metric_ranks}"
                )
            window_microbatch_count += 1
            backward_wall_started = None
            backward_event_start = None
            backward_event_end = None
            if timing_enabled:
                if timing_uses_cuda_events:
                    backward_event_start = torch.cuda.Event(enable_timing=True)
                    backward_event_end = torch.cuda.Event(enable_timing=True)
                    backward_event_start.record()
                else:
                    backward_wall_started = time.perf_counter()
            if bool(
                getattr(
                    self,
                    "deepspeed_zero2_graph_task_cache_enabled",
                    False,
                )
            ):
                reset_deepspeed_zero2_graph_task_cache()
            self.accelerator.backward(loss)
            if timing_enabled:
                if timing_uses_cuda_events:
                    assert backward_event_start is not None
                    assert backward_event_end is not None
                    backward_event_end.record()
                    timing_backward_events.append(
                        (backward_event_start, backward_event_end)
                    )
                else:
                    assert backward_wall_started is not None
                    timing_interval_seconds["backward"] += (
                        time.perf_counter() - backward_wall_started
                    )

            if self.accelerator.sync_gradients:
                if not self._is_valid_optimizer_boundary_window(
                    window_microbatch_count
                ):
                    raise RuntimeError(
                        "optimizer boundary has an incomplete accumulation window: "
                        f"{window_microbatch_count} != "
                        f"{self.gradient_accumulation_steps}"
                    )
                optimizer_wall_started = None
                optimizer_event_start = None
                optimizer_event_end = None
                if timing_enabled:
                    if timing_uses_cuda_events:
                        optimizer_event_start = torch.cuda.Event(
                            enable_timing=True
                        )
                        optimizer_event_end = torch.cuda.Event(
                            enable_timing=True
                        )
                        optimizer_event_start.record()
                    else:
                        optimizer_wall_started = time.perf_counter()
                deepspeed_runtime = self._is_deepspeed_runtime()
                grad_norm = None
                if not deepspeed_runtime:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                    if grad_norm is None:
                        raise FloatingPointError(
                            "No reliable global gradient norm was available "
                            "before the optimizer update"
                        )
                self.optimizer.step()
                if deepspeed_runtime:
                    grad_norm_getter = getattr(
                        self.model,
                        "get_global_grad_norm",
                        None,
                    )
                    if not callable(grad_norm_getter):
                        raise RuntimeError(
                            "DeepSpeed engine lacks get_global_grad_norm"
                        )
                    grad_norm = grad_norm_getter()
                if timing_enabled:
                    if timing_uses_cuda_events:
                        assert optimizer_event_start is not None
                        assert optimizer_event_end is not None
                        optimizer_event_end.record()
                        # One synchronization per optimizer update is only
                        # enabled for canary timing. It makes all phase
                        # events queryable without synchronizing each
                        # individual forward or backward.
                        optimizer_event_end.synchronize()
                        timing_interval_seconds["model_forward"] += sum(
                            start.elapsed_time(end) / 1000.0
                            for start, end in timing_forward_events
                        )
                        timing_interval_seconds["backward"] += sum(
                            start.elapsed_time(end) / 1000.0
                            for start, end in timing_backward_events
                        )
                        timing_interval_seconds["optimizer"] += (
                            optimizer_event_start.elapsed_time(
                                optimizer_event_end
                            )
                            / 1000.0
                        )
                        timing_forward_events.clear()
                        timing_backward_events.clear()
                    else:
                        assert optimizer_wall_started is not None
                        timing_interval_seconds["optimizer"] += (
                            time.perf_counter() - optimizer_wall_started
                        )
                    timing_interval_updates += 1
                if grad_norm is None:
                    raise FloatingPointError(
                        "No reliable global gradient norm was available "
                        "after the optimizer update"
                    )
                try:
                    grad_norm_value = float(
                        grad_norm.detach().float().item()
                        if isinstance(grad_norm, torch.Tensor)
                        else grad_norm
                    )
                except (TypeError, ValueError) as exc:
                    raise FloatingPointError(
                        "Global gradient norm is not numeric"
                    ) from exc
                local_skipped = bool(
                    self.accelerator.optimizer_step_was_skipped
                )
                gathered_skipped = self.accelerator.gather(
                    torch.tensor(
                        [int(local_skipped)],
                        device=loss.device,
                        dtype=torch.int64,
                    )
                ).reshape(-1)
                if gathered_skipped.numel() != int(
                    self.accelerator.num_processes
                ):
                    raise RuntimeError(
                        "optimizer skip collective did not return one flag per rank"
                    )
                self.optimizer_updates_attempted += 1
                if bool(torch.any(gathered_skipped != 0).item()):
                    self.optimizer_updates_skipped += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    skipped_ranks = torch.nonzero(
                        gathered_skipped != 0,
                        as_tuple=False,
                    ).reshape(-1).cpu().tolist()
                    raise FloatingPointError(
                        "Optimizer update was skipped on distributed rank(s) "
                        f"{skipped_ranks}; failing closed without advancing "
                        "global_step or consuming further samples."
                    )
                if not np.isfinite(grad_norm_value) or grad_norm_value < 0.0:
                    raise FloatingPointError(
                        "Invalid gradient norm for a successful optimizer "
                        f"update at step {self.global_step}."
                    )
                debug_optimizer_metrics = (
                    self._collect_debug_optimizer_audit_metrics(
                        device=loss.device,
                    )
                )
                self.optimizer_updates_successful += 1
                self.global_step = self.optimizer_updates_successful
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                loss_stats = torch.tensor(
                    [[window_loss_sum, float(window_microbatch_count)]],
                    device=loss.device,
                    dtype=torch.float64,
                )
                gathered_loss_stats = self.accelerator.gather(
                    loss_stats
                ).reshape(-1, 2)
                global_loss = float(
                    gathered_loss_stats[:, 0].sum().item()
                    / gathered_loss_stats[:, 1].sum().item()
                )
                global_loss_metrics = {}
                for key in window_metric_keys or ():
                    metric_stats = torch.tensor(
                        [[
                            window_metric_sums[key],
                            float(window_microbatch_count),
                        ]],
                        device=loss.device,
                        dtype=torch.float64,
                    )
                    gathered_metric_stats = self.accelerator.gather(
                        metric_stats
                    ).reshape(-1, 2)
                    global_loss_metrics[key] = float(
                        gathered_metric_stats[:, 0].sum().item()
                        / gathered_metric_stats[:, 1].sum().item()
                    )
                overlapping_debug_keys = set(global_loss_metrics).intersection(
                    debug_optimizer_metrics
                )
                if overlapping_debug_keys:
                    raise RuntimeError(
                        "debug optimizer metrics collide with loss metrics: "
                        f"{sorted(overlapping_debug_keys)}"
                    )
                global_loss_metrics.update(debug_optimizer_metrics)
                window_loss_sum = 0.0
                window_metric_sums.clear()
                window_metric_keys = None
                window_microbatch_count = 0
                grad_norm_tensor = torch.tensor(
                    grad_norm_value,
                    device=loss.device,
                    dtype=torch.float32,
                )
                global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                learning_rate_metrics = (
                    _collect_optimizer_learning_rate_metrics(self.optimizer)
                )
                current_lr = learning_rate_metrics["train/lr"]
                should_log = (
                    self.log_every > 0
                    and self.global_step % self.log_every == 0
                )
                cuda_memory_metrics = (
                    self._collect_cuda_memory_metrics(loss.device)
                    if should_log
                    else {}
                )
                timing_metrics: dict[str, float] = {}
                if should_log and timing_enabled:
                    assert timing_interval_wall_started is not None
                    interval_wall_seconds = (
                        time.perf_counter() - timing_interval_wall_started
                    )
                    local_timing = torch.tensor(
                        [[
                            timing_interval_seconds["batch_wait_and_h2d"],
                            timing_interval_seconds["model_forward"],
                            timing_interval_seconds["backward"],
                            timing_interval_seconds["optimizer"],
                            interval_wall_seconds,
                            float(timing_interval_updates),
                        ]],
                        device=loss.device,
                        dtype=torch.float64,
                    )
                    gathered_timing = self.accelerator.gather(
                        local_timing
                    ).reshape(-1, 6)
                    if gathered_timing.shape[0] != int(
                        self.accelerator.num_processes
                    ):
                        raise RuntimeError(
                            "training timing collective did not return one "
                            "row per rank"
                        )
                    gathered_update_counts = gathered_timing[:, 5]
                    if (
                        timing_interval_updates <= 0
                        or not bool(
                            torch.all(
                                gathered_update_counts
                                == gathered_update_counts[0]
                            ).item()
                        )
                    ):
                        raise RuntimeError(
                            "training timing update counts differ across ranks"
                        )
                    update_count = float(gathered_update_counts[0].item())
                    rank_max_phase_seconds = gathered_timing[:, :4].max(
                        dim=0
                    ).values / update_count
                    rank_max_wall_seconds = float(
                        gathered_timing[:, 4].max().item() / update_count
                    )
                    residual_seconds = float(
                        torch.clamp(
                            gathered_timing[:, 4]
                            - gathered_timing[:, :4].sum(dim=1),
                            min=0.0,
                        ).max().item()
                        / update_count
                    )
                    phase_sum = float(rank_max_phase_seconds.sum().item())
                    wait_seconds = float(rank_max_phase_seconds[0].item())
                    timing_metrics = {
                        "performance/timing_batch_wait_h2d_s_per_update": (
                            wait_seconds
                        ),
                        "performance/timing_model_forward_s_per_update": float(
                            rank_max_phase_seconds[1].item()
                        ),
                        "performance/timing_backward_s_per_update": float(
                            rank_max_phase_seconds[2].item()
                        ),
                        "performance/timing_optimizer_s_per_update": float(
                            rank_max_phase_seconds[3].item()
                        ),
                        "performance/timing_wall_s_per_update": (
                            rank_max_wall_seconds
                        ),
                        "performance/timing_residual_s_per_update": (
                            residual_seconds
                        ),
                        "performance/timing_rank_max_phase_sum_s_per_update": (
                            phase_sum
                        ),
                        "performance/timing_batch_wait_h2d_fraction_of_wall": (
                            wait_seconds
                            / max(rank_max_wall_seconds, 1.0e-12)
                        ),
                        "performance/timing_updates": update_count,
                    }
                    for key in timing_interval_seconds:
                        timing_interval_seconds[key] = 0.0
                    timing_interval_updates = 0
                    timing_interval_wall_started = time.perf_counter()

                if should_log and self.accelerator.is_main_process:
                    eta_str, steps_per_sec, samples_per_sec = self._estimate_eta()
                    description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                        self.epoch,
                        self.global_step,
                        self.max_steps,
                        global_loss,
                    )
                    if global_loss_metrics:
                        detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                        description += detail_str + " "
                    description += "lr=%.2e speed=%.2f update/s, %.2f samples/s eta=%s" % (
                        current_lr,
                        steps_per_sec,
                        samples_per_sec,
                        eta_str,
                    )
                    if timing_metrics:
                        description += (
                            " timing(wall=%.3fs wait+h2d=%.3fs fwd=%.3fs "
                            "bwd=%.3fs opt=%.3fs residual=%.3fs "
                            "wait_wall=%.1f%%)"
                            % (
                                timing_metrics[
                                    "performance/timing_wall_s_per_update"
                                ],
                                timing_metrics[
                                    "performance/timing_batch_wait_h2d_s_per_update"
                                ],
                                timing_metrics[
                                    "performance/timing_model_forward_s_per_update"
                                ],
                                timing_metrics[
                                    "performance/timing_backward_s_per_update"
                                ],
                                timing_metrics[
                                    "performance/timing_optimizer_s_per_update"
                                ],
                                timing_metrics[
                                    "performance/timing_residual_s_per_update"
                                ],
                                100.0
                                * timing_metrics[
                                    "performance/timing_batch_wait_h2d_fraction_of_wall"
                                ],
                            )
                        )
                    logger.info(description)

                    wandb_payload = {
                        "train/loss": global_loss,
                        "train/grad_norm": global_grad_norm,
                        "performance/optimizer_updates_per_sec": steps_per_sec,
                        # Backward-compatible alias; a "step" here is an
                        # optimizer update, not a DataLoader micro-step.
                        "performance/steps_per_sec": steps_per_sec,
                        "performance/samples_per_sec": samples_per_sec,
                    }
                    wandb_payload.update(learning_rate_metrics)
                    wandb_payload.update(timing_metrics)
                    wandb_payload.update(cuda_memory_metrics)
                    for key, value in global_loss_metrics.items():
                        wandb_payload[f"train/{key}"] = value
                    self._wandb_log(wandb_payload)
                    if self.training_metrics_writer is not None:
                        try:
                            self.training_metrics_writer.append(
                                step=self.global_step,
                                epoch=self.epoch,
                                max_steps=self.max_steps,
                                optimizer_updates_attempted=(
                                    self.optimizer_updates_attempted
                                ),
                                optimizer_updates_successful=(
                                    self.optimizer_updates_successful
                                ),
                                optimizer_updates_skipped=(
                                    self.optimizer_updates_skipped
                                ),
                                metrics=wandb_payload,
                            )
                        except (OSError, TypeError, ValueError) as exc:
                            logger.warning(
                                "Failed to append live training metrics to %s: %s",
                                self.training_metrics_writer.path,
                                exc,
                            )

                runtime_checkpoint_result = (
                    process_runtime_checkpoint_boundary(
                        self,
                        runtime_checkpoint_control,
                        device=loss.device,
                    )
                )
                if (
                    runtime_checkpoint_result.poll_error is not None
                    and runtime_checkpoint_result.poll_error
                    != runtime_checkpoint_last_error
                    and self.accelerator.is_main_process
                ):
                    runtime_checkpoint_last_error = (
                        runtime_checkpoint_result.poll_error
                    )
                    logger.warning(
                        "Runtime checkpoint request polling failed: %s",
                        runtime_checkpoint_last_error,
                    )
                if (
                    runtime_checkpoint_result.action is not None
                    and self.accelerator.is_main_process
                ):
                    checkpoint_info = runtime_checkpoint_result.checkpoint_info
                    assert checkpoint_info is not None
                    logger.info(
                        "[runtime-ckpt] request=%s action=%s step=%d "
                        "weights=%s state=%s",
                        runtime_checkpoint_result.request_id,
                        runtime_checkpoint_result.action,
                        self.global_step,
                        checkpoint_info.get("weights_path"),
                        checkpoint_info.get("state_path"),
                    )
                if runtime_checkpoint_result.stop_requested:
                    runtime_checkpoint_control.close(
                        status="checkpoint-and-stop-completed",
                        global_step=self.global_step,
                    )
                    return

                if (
                    self.eval_every > 0
                    and self.val_dataset is not None
                    and self.global_step % self.eval_every == 0
                ):
                    metrics = self.evaluate()
                    self.accelerator.wait_for_everyone()
                    if metrics is not None and self.accelerator.is_main_process:
                        description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                            self.global_step,
                            metrics["val_loss"],
                            metrics["psnr_rd"],
                            metrics["ssim_rd"],
                        )
                        if "action_l2" in metrics:
                            description += " action_l2=%.4f" % metrics["action_l2"]
                        if "action_l1" in metrics:
                            description += " action_l1=%.4f" % metrics["action_l1"]
                        logger.info(description)
                        eval_payload = {
                            "eval/val_loss": float(metrics["val_loss"]),
                            "eval/psnr_rg": float(metrics["psnr_rg"]),
                            "eval/ssim_rg": float(metrics["ssim_rg"]),
                            "eval/psnr_rd": float(metrics["psnr_rd"]),
                            "eval/ssim_rd": float(metrics["ssim_rd"]),
                            "eval/psnr_dg": float(metrics["psnr_dg"]),
                            "eval/ssim_dg": float(metrics["ssim_dg"]),
                        }
                        if "action_l2" in metrics:
                            eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                        if "action_l1" in metrics:
                            eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                        self._wandb_log(eval_payload)

                from .feedback_continuation import checkpoint_due, boundary as feedback_boundary
                from .libero_feedback import boundary as libero_feedback_boundary
                from .task_replay import boundary as task_boundary, checkpoint_due as task_checkpoint_due
                if task_boundary(self) or feedback_boundary(self) or libero_feedback_boundary(self):
                    close_iterator = getattr(data_iter, "close", None)
                    if callable(close_iterator):
                        close_iterator()
                    self._set_train_data_epoch(self.epoch)
                    self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                    data_iter = iter(self.train_loader)
                    logger.info("Feedback task replay activated at step=%d", self.global_step)

                from .scene_replay import checkpoint_due as scene_checkpoint_due, boundary as scene_boundary, contract as scene_contract
                save_due = task_checkpoint_due(self) if self.cfg.get('task_failure_replay_config') else (scene_checkpoint_due(self) if scene_contract(self.cfg) is not None else checkpoint_due(self))
                if save_due:
                    ckpt_info = self.save_checkpoint()
                    if self.accelerator.is_main_process:
                        logger.info(
                            "[ckpt] step=%d weights=%s state=%s",
                            self.global_step,
                            ckpt_info["weights_path"],
                            ckpt_info["state_path"],
                        )

                if scene_boundary(self):
                    close_iterator = getattr(data_iter, 'close', None)
                    if callable(close_iterator):
                        close_iterator()
                    self._set_train_data_epoch(self.epoch)
                    self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                    data_iter = iter(self.train_loader)

                if adaptive_replay_runtime is not None:
                    adaptive_result = adaptive_replay_runtime.boundary(
                        self,
                        step=self.global_step,
                    )
                    if adaptive_result.reset_data_iterator:
                        close_iterator = getattr(data_iter, "close", None)
                        if callable(close_iterator):
                            close_iterator()
                        self._set_train_data_epoch(self.epoch)
                        self.train_sampler.set_resume_batch_offset(
                            self.batch_in_epoch
                        )
                        data_iter = iter(self.train_loader)
                        logger.info(
                            "Adaptive replay rebuilt the DataLoader iterator "
                            "at epoch=%d batch_in_epoch=%d request=%s",
                            self.epoch,
                            self.batch_in_epoch,
                            adaptive_result.request_id,
                        )

                if self.global_step >= run_target_step:
                    reason = (
                        "production max_steps"
                        if stop_after_step is None
                        else "stop_after_step"
                    )
                    if checkpoint_run_target:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] %s reached step=%d weights=%s state=%s",
                                reason,
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                    elif self.accelerator.is_main_process:
                        logger.info(
                            "[done] %s reached step=%d; checkpoint "
                            "intentionally disabled for throughput timing",
                            reason,
                            self.global_step,
                        )
                    runtime_checkpoint_control.close(
                        status="run-target-completed",
                        global_step=self.global_step,
                    )
                    return

    ckpt_info = self.save_checkpoint()
    if self.accelerator.is_main_process:
        logger.info(
            "[done] training finished step=%d weights=%s state=%s",
            self.global_step,
            ckpt_info["weights_path"],
            ckpt_info["state_path"],
        )
    runtime_checkpoint_control.close(
        status="training-completed",
        global_step=self.global_step,
    )


__all__ = ["run_training_loop"]
