"""Injectable training telemetry with no model or checkpoint ownership."""

from __future__ import annotations

import time
from typing import Any

import torch


class TrainingTelemetry:
    """Default low-frequency allocator and throughput telemetry."""

    def collect_cuda_memory_metrics(
        self,
        trainer: Any,
        device: torch.device,
    ) -> dict[str, float]:
        if device.type != "cuda" or not torch.cuda.is_available():
            return {}
        local = torch.tensor(
            [[
                torch.cuda.memory_allocated(device),
                torch.cuda.memory_reserved(device),
                torch.cuda.max_memory_allocated(device),
                torch.cuda.max_memory_reserved(device),
                torch.cuda.get_device_properties(device).total_memory,
            ]],
            device=device,
            dtype=torch.int64,
        )
        gathered = trainer.accelerator.gather(local).reshape(-1, 5)
        if gathered.shape[0] != int(trainer.accelerator.num_processes):
            raise RuntimeError(
                "CUDA memory collective did not return one row per rank"
            )
        names = (
            "allocated_bytes",
            "reserved_bytes",
            "peak_allocated_bytes_since_session_start",
            "peak_reserved_bytes_since_session_start",
            "device_total_bytes",
        )
        metrics: dict[str, float] = {}
        for column, name in enumerate(names):
            values = gathered[:, column]
            metrics[
                f"performance/pytorch_cuda_{name}_rank_min"
            ] = float(values.min().item())
            metrics[
                f"performance/pytorch_cuda_{name}_rank_max"
            ] = float(values.max().item())
        return metrics

    def estimate_eta(self, trainer: Any) -> tuple[str, float, float]:
        elapsed = max(time.perf_counter() - trainer.run_start_time, 1e-6)
        done_steps = max(
            trainer.global_step - trainer.run_start_step,
            1,
        )
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(
            trainer.max_steps - trainer.global_step,
            0,
        )
        eta_seconds = int(
            remaining_steps / max(steps_per_sec, 1e-9)
        )
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        microbatches = max(
            int(getattr(trainer, "run_microbatches_consumed", 0)),
            0,
        )
        global_batch_size = (
            trainer.batch_size
            * int(trainer.accelerator.num_processes)
        )
        samples_per_sec = (
            microbatches * global_batch_size / elapsed
        )
        return (
            f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}",
            steps_per_sec,
            samples_per_sec,
        )


__all__ = ["TrainingTelemetry"]
