"""Injectable policies for the generic Wan22 trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import checkpoint_state
from .loop import run_training_loop
from .sampler_loader import build_train_loader
from .telemetry import TrainingTelemetry


class CheckpointAdapter:
    """Default checkpoint policy backed by receipt-free state helpers."""

    def resume_or_load(self, trainer: Any) -> None:
        checkpoint_state.resume_or_load_checkpoint(trainer)

    def save_weights(self, trainer: Any, *, step_tag: str) -> str:
        return checkpoint_state._save_weights_checkpoint(
            trainer,
            step_tag=step_tag,
        )

    def save_trainer_state(
        self,
        trainer: Any,
        state_path: str,
        *,
        rank_state: list[dict],
    ) -> None:
        checkpoint_state._save_trainer_state(
            trainer,
            state_path=state_path,
            rank_state=rank_state,
        )

    def save(self, trainer: Any) -> dict[str, str | None]:
        return checkpoint_state.save_checkpoint(trainer)

    def load(self, trainer: Any, state_dir: str) -> None:
        checkpoint_state.load_training_state(trainer, state_dir)


@dataclass(frozen=True)
class TrainerStrategies:
    """Default strategy bundle; callers may inject any compatible member."""

    sampler_loader: Callable[..., Any] = build_train_loader
    checkpoint: Any = field(default_factory=CheckpointAdapter)
    telemetry: Any = field(default_factory=TrainingTelemetry)
    loop: Callable[[Any], Any] = run_training_loop


__all__ = [
    "CheckpointAdapter",
    "TrainerStrategies",
]
