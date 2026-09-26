"""Offline temporal selection and explicit observation/action alignment.

Saved RoboTwin frames are a non-decreasing observation timeline: more than one
frame may legitimately share the same simulator time. These helpers never
assume a fixed saved-frame period and never reinterpret low-level drive targets
as high-level policy actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


LOW_LEVEL_CONTROL_STREAM = "control_stream_low_level_drive_target"
LEGACY_SAVED_FRAME_ACTION = "saved_frame_action_target"


def _nondecreasing_times(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("timestamps_sec must be a non-empty 1D array")
    if not np.isfinite(result).all():
        raise ValueError("timestamps_sec contains NaN/Inf")
    if np.any(np.diff(result) < 0):
        raise ValueError("timestamps_sec must be non-decreasing")
    return result


def deduplicate_saved_frames_by_time(
    timestamps_sec: Sequence[float],
    *,
    keep: str = "last",
) -> np.ndarray:
    """Return indices with exactly duplicate timestamps collapsed.

    Exact equality is deliberate: close but distinct physics times must not be
    merged by an arbitrary tolerance. Keeping the last capture is useful when
    several captures share one simulator state.
    """

    times = _nondecreasing_times(timestamps_sec)
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'")
    if times.size == 1:
        return np.asarray([0], dtype=np.int64)
    if keep == "first":
        mask = np.r_[True, times[1:] != times[:-1]]
    else:
        mask = np.r_[times[:-1] != times[1:], True]
    return np.flatnonzero(mask).astype(np.int64, copy=False)


@dataclass(frozen=True)
class PhysicalTimeTargets:
    source_index: int
    target_indices: np.ndarray
    requested_offsets_sec: np.ndarray
    actual_dt_sec: np.ndarray
    lateness_sec: np.ndarray


def select_future_frames_by_physical_time(
    timestamps_sec: Sequence[float],
    *,
    source_index: int,
    offsets_sec: Sequence[float],
) -> PhysicalTimeTargets:
    """Select the first saved frame at/after each requested physical horizon."""

    times = _nondecreasing_times(timestamps_sec)
    source = int(source_index)
    if not 0 <= source < times.size:
        raise IndexError(f"source_index={source} outside [0, {times.size})")
    offsets = np.asarray(offsets_sec, dtype=np.float64)
    if (
        offsets.ndim != 1
        or offsets.size == 0
        or not np.isfinite(offsets).all()
        or np.any(offsets <= 0)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("offsets_sec must be finite, positive and increasing")
    requested_times = times[source] + offsets
    targets = np.searchsorted(times, requested_times, side="left").astype(np.int64)
    if np.any(targets >= times.size):
        missing = offsets[targets >= times.size].tolist()
        raise IndexError(f"physical-time targets exceed episode: {missing}")
    if np.any(targets <= source):
        raise RuntimeError("selected target does not occur after source frame")
    actual = times[targets] - times[source]
    return PhysicalTimeTargets(
        source_index=source,
        target_indices=targets,
        requested_offsets_sec=offsets,
        actual_dt_sec=actual,
        lateness_sec=actual - offsets,
    )


@dataclass(frozen=True)
class LowLevelActionChunk:
    values: np.ndarray
    is_pad: np.ndarray
    control_stream_indices: np.ndarray
    timestamp_after_sec: np.ndarray
    label_kind: str = "low_level_joint_drive_target"
    high_level_policy_action: bool = False


def read_future_low_level_action_chunk(
    reader: Any,
    *,
    source_frame: int,
    horizon: int,
) -> LowLevelActionChunk:
    """Read commands not yet applied at the source observation.

    The exclusive stop pointer for a saved frame is the first future command.
    Padding is allowed only beyond the recorded episode stream. Missing stream
    data is a hard error and never falls back to capture-time drive_target or
    the past saved-frame action_target.
    """

    if horizon <= 0:
        raise ValueError("action horizon must be positive")
    if not reader.has_control_stream:
        raise ValueError(
            "complete control_stream is required for future low-level labels"
        )
    _, first = reader.saved_frame_control_span(int(source_frame))
    count = int(reader.control_stream_count)
    if first >= count:
        raise ValueError(
            f"source frame {source_frame} has no future control command"
        )
    stop = min(first + int(horizon), count)
    fields = reader.read_control_stream_fields(
        first,
        stop,
        fields=("drive_target", "stream_index", "timestamp_after_sec"),
    )
    action = np.asarray(fields["drive_target"], dtype=np.float32)
    if action.ndim != 2 or not np.isfinite(action).all():
        raise ValueError("control_stream drive target is not finite [C,A]")
    values = np.zeros((horizon, action.shape[1]), dtype=np.float32)
    values[: len(action)] = action
    is_pad = np.ones(horizon, dtype=bool)
    is_pad[: len(action)] = False
    indices = np.full(horizon, -1, dtype=np.int64)
    indices[: len(action)] = np.asarray(fields["stream_index"], dtype=np.int64)
    timestamps = np.full(horizon, np.nan, dtype=np.float64)
    timestamps[: len(action)] = np.asarray(
        fields["timestamp_after_sec"], dtype=np.float64
    )
    return LowLevelActionChunk(
        values=values,
        is_pad=is_pad,
        control_stream_indices=indices,
        timestamp_after_sec=timestamps,
    )
