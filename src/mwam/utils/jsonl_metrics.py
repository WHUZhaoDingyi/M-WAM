from __future__ import annotations

import fcntl
import json
import math
import os
import stat
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path


TRAINING_METRICS_SCHEMA_VERSION = "fastwam_training_metrics_v1"


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_metrics(metrics: Mapping[str, Real]) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")

    validated: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metric names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"metric {key!r} must be a real number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"metric {key!r} must be finite")
        validated[key] = number
    return validated


class TrainingMetricsJsonlWriter:
    """Append validated, process-session-scoped training samples to JSONL."""

    def __init__(self, path: str | os.PathLike[str], *, session_id: str | None = None):
        self.path = Path(path)
        self.session_id = str(uuid.UUID(session_id)) if session_id else str(uuid.uuid4())

    def append(
        self,
        *,
        step: int,
        epoch: int,
        max_steps: int,
        optimizer_updates_attempted: int,
        optimizer_updates_successful: int,
        optimizer_updates_skipped: int,
        metrics: Mapping[str, Real],
    ) -> dict:
        step = _non_negative_int("step", step)
        epoch = _non_negative_int("epoch", epoch)
        max_steps = _non_negative_int("max_steps", max_steps)
        attempted = _non_negative_int(
            "optimizer_updates_attempted", optimizer_updates_attempted
        )
        successful = _non_negative_int(
            "optimizer_updates_successful", optimizer_updates_successful
        )
        skipped = _non_negative_int(
            "optimizer_updates_skipped", optimizer_updates_skipped
        )
        if max_steps == 0:
            raise ValueError("max_steps must be positive")
        if step > max_steps:
            raise ValueError("step cannot exceed max_steps")
        if attempted != successful + skipped:
            raise ValueError(
                "attempted optimizer updates must equal successful plus skipped updates"
            )

        record = {
            "schema_version": TRAINING_METRICS_SCHEMA_VERSION,
            "session_id": self.session_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "step": step,
            "epoch": epoch,
            "max_steps": max_steps,
            "optimizer_updates": {
                "attempted": attempted,
                "successful": successful,
                "skipped": skipped,
            },
            "metrics": _finite_metrics(metrics),
        }
        line = (
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.path, flags, 0o600)
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise OSError(f"metrics path is not a regular file: {self.path}")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._truncate_incomplete_tail(fd)
            view = memoryview(line)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("failed to append training metrics")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        return record

    @staticmethod
    def _truncate_incomplete_tail(fd: int) -> None:
        end = os.lseek(fd, 0, os.SEEK_END)
        if end == 0 or os.pread(fd, 1, end - 1) == b"\n":
            return

        search_end = end
        truncate_at = 0
        while search_end > 0:
            search_start = max(0, search_end - 64 * 1024)
            chunk = os.pread(fd, search_end - search_start, search_start)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                truncate_at = search_start + newline + 1
                break
            search_end = search_start
        os.ftruncate(fd, truncate_at)
        os.fsync(fd)
