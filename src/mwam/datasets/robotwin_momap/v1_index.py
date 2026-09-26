"""Deterministic training-window index for canonical RoboTwin raw-v1 episodes.

The index is deliberately manifest driven.  It never globs a data root and it
never pads an episode tail.  A source frame is present only when every MoMap
target frame and every requested future low-level action exists.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

import numpy as np

from mwam.robotwin_momap.temporal_alignment import (
    select_future_frames_by_physical_time,
)

from .schema import (
    FASTWAM_CLIP_FRAME_15HZ_OFFSET_BASIS,
    KNOWN_FUTURE_OFFSET_BASES,
    LEGACY_SAVED_FRAME_OFFSET_BASIS,
    SampleRef,
)


PHYSICAL_CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
FASTWAM_CLIP_FPS_HZ = 15.0
FASTWAM_K1_LOGICAL_FUTURE_OFFSETS = (4,)
FASTWAM_K1_FUTURE_HORIZONS_SECONDS = tuple(
    offset / FASTWAM_CLIP_FPS_HZ
    for offset in FASTWAM_K1_LOGICAL_FUTURE_OFFSETS
)
FASTWAM_K4_HYBRID_LOGICAL_FUTURE_OFFSETS = (8, 16, 24, 32)
FASTWAM_K4_HYBRID_FUTURE_HORIZONS_SECONDS = tuple(
    offset / FASTWAM_CLIP_FPS_HZ
    for offset in FASTWAM_K4_HYBRID_LOGICAL_FUTURE_OFFSETS
)
FASTWAM_LOGICAL_FUTURE_OFFSETS = tuple(range(4, 33, 4))
FASTWAM_FUTURE_HORIZONS_SECONDS = tuple(
    offset / FASTWAM_CLIP_FPS_HZ for offset in FASTWAM_LOGICAL_FUTURE_OFFSETS
)
FASTWAM_H32S1_LOGICAL_FUTURE_OFFSETS = tuple(range(1, 33))
FASTWAM_H32S1_FUTURE_HORIZONS_SECONDS = tuple(
    offset / FASTWAM_CLIP_FPS_HZ
    for offset in FASTWAM_H32S1_LOGICAL_FUTURE_OFFSETS
)
_FASTWAM_SUPPORTED_FUTURE_CONTRACTS = {
    FASTWAM_K1_LOGICAL_FUTURE_OFFSETS: FASTWAM_K1_FUTURE_HORIZONS_SECONDS,
    FASTWAM_K4_HYBRID_LOGICAL_FUTURE_OFFSETS: (
        FASTWAM_K4_HYBRID_FUTURE_HORIZONS_SECONDS
    ),
    FASTWAM_LOGICAL_FUTURE_OFFSETS: FASTWAM_FUTURE_HORIZONS_SECONDS,
    FASTWAM_H32S1_LOGICAL_FUTURE_OFFSETS: FASTWAM_H32S1_FUTURE_HORIZONS_SECONDS,
}
RAW_V1_SCHEMA = "robotwin_momap_raw_episode_v1"
ACTION_CONTRACT_SCHEMA = "robotwin_momap_observation_action_contract_v0"
ACTION_LABEL_KIND = "low_level_joint_drive_target"
ACTION_FUTURE_LABEL = (
    "for observation s, the first future low-level command is "
    "control_stream.drive_target[saved_frame_applied_stop[s]]"
)
KNOWN_RANK_SUMMARY_SCHEMAS = frozenset(
    {
        "robotwin_momap_v1_candidate_rank_summary_v3",
    }
)
KNOWN_GLOBAL_RECEIPT_SCHEMAS = frozenset(
    {
        "robotwin_momap_v1_candidate_global_result_v3",
    }
)
_RECORD_SET_PREFIX = b"robotwin-momap-v1-candidate-record-set-v1"
V1_INDEX_AUDIT_SCHEMA_VERSION = "robotwin_momap_v1_index_build_audit_v1"
V1_RUNTIME_INDEX_BINDING_SCHEMA_VERSION = (
    "robotwin_momap_v1_runtime_index_binding_v1"
)
V1_RUNTIME_INDEX_COMPOSITION_ALGORITHM = (
    "ordered_compact_episode_windows_and_exact_audit_v1"
)
V1_INDEX_AUDIT_KEYS = (
    "episode_count",
    "candidate_source_frames",
    "excluded_physical_horizon_unavailable",
    "excluded_non_increasing_target_indices",
    "excluded_action_horizon_unavailable",
    "excluded_last_command_without_history",
    "included_windows",
)

ActionMode = Literal["baseline_rate_resampled@15Hz", "dense250Hz"]


class ActionHorizonUnavailable(IndexError):
    """A valid source has insufficient recorded future control for the horizon."""


class V1ReaderLike(Protocol):
    """The small part of ``RawEpisodeReader`` needed while indexing."""

    @property
    def num_frames(self) -> int: ...

    @property
    def camera_names(self) -> tuple[str, ...]: ...

    @property
    def schema_version(self) -> str: ...

    @property
    def metadata(self) -> dict[str, Any]: ...

    @property
    def has_control_stream(self) -> bool: ...

    @property
    def control_stream_count(self) -> int: ...

    @property
    def control_stream_contract(self) -> dict[str, Any]: ...

    def saved_frame_control_span(self, index: int) -> tuple[int, int]: ...

    def read_control_stream_fields(
        self,
        start: int = 0,
        stop: int | None = None,
        *,
        fields: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]: ...

    def read_frame_fields(self, index: int, **kwargs: Any) -> dict[str, Any]: ...

    def close(self) -> None: ...


ReaderFactory = Callable[[str | Path], V1ReaderLike]
IndexProgressCallback = Callable[[int, int, int], None]


def _default_reader_factory(path: str | Path) -> V1ReaderLike:
    try:
        from mwam.robotwin_momap.raw_episode_hdf5 import RawEpisodeReader
    except ImportError as exc:  # pragma: no cover - depends on optional migration.
        raise RuntimeError(
            "canonical V1 loading requires mwam.robotwin_momap."
            "raw_episode_hdf5.RawEpisodeReader"
        ) from exc
    return RawEpisodeReader(path, require_complete=True)


@dataclass(frozen=True)
class ActionSamplingConfig:
    """Future low-level action sampling, independent of MoMap frame offsets."""

    mode: ActionMode = "baseline_rate_resampled@15Hz"
    horizon: int = 32
    baseline_rate_hz: float = 15.0
    timestamp_tolerance_sec: float = 1.0e-9

    def __post_init__(self) -> None:
        aliases = {
            "baseline_rate_resampled": "baseline_rate_resampled@15Hz",
            "dense_control_stream@250Hz": "dense250Hz",
        }
        normalized = aliases.get(str(self.mode), str(self.mode))
        if normalized not in {"baseline_rate_resampled@15Hz", "dense250Hz"}:
            raise ValueError(f"unsupported action sampling mode: {self.mode!r}")
        object.__setattr__(self, "mode", normalized)
        if self.horizon <= 0:
            raise ValueError("action horizon must be positive")
        if not np.isfinite(self.baseline_rate_hz) or self.baseline_rate_hz <= 0:
            raise ValueError("baseline_rate_hz must be finite and positive")
        if (
            not np.isfinite(self.timestamp_tolerance_sec)
            or self.timestamp_tolerance_sec < 0
        ):
            raise ValueError("timestamp_tolerance_sec must be finite and non-negative")
        if self.mode == "baseline_rate_resampled@15Hz" and not np.isclose(
            self.baseline_rate_hz, 15.0
        ):
            raise ValueError(
                "baseline_rate_resampled@15Hz requires baseline_rate_hz=15"
            )


@dataclass(frozen=True)
class ActionSelection:
    """Resolved control-stream rows and their physical query times."""

    source_control_start: int
    indices: tuple[int, ...]
    query_times_sec: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.source_control_start < 0:
            raise ValueError("source_control_start must be non-negative")
        if not self.indices:
            raise ValueError("action selection must not be empty")
        if self.indices[0] != self.source_control_start:
            raise ValueError(
                "first future action must be saved_frame_applied_stop[source]"
            )
        if any(right <= left for left, right in zip(self.indices, self.indices[1:])):
            raise ValueError("action control-stream indices must be strictly increasing")
        if len(self.query_times_sec) != len(self.indices):
            raise ValueError("action index/query-time length mismatch")
        if not np.isfinite(np.asarray(self.query_times_sec, dtype=np.float64)).all():
            raise ValueError("action query times contain NaN/Inf")


@dataclass(frozen=True)
class V1EpisodeRecord:
    episode_path: Path
    episode_id: str | None = None
    task_name: str | None = None
    task_config: str | None = None
    variant: str | None = None
    source_manifest: Path | None = None
    expected_hdf5_sha256: str | None = None
    candidate_report: Path | None = None
    candidate_report_sha256: str | None = None
    expected_frame_count: int | None = None
    scope_id: str | None = None
    candidate_ordinal: int | None = None
    seed: int | None = None
    expected_episode_number: int | None = None
    canonical_trajectory_sha256: str | None = None
    planning_scene_sha256: str | None = None
    replay_scene_sha256: str | None = None
    saved_frame_timeline_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_path", self.episode_path.expanduser().resolve())
        if self.source_manifest is not None:
            object.__setattr__(
                self, "source_manifest", self.source_manifest.expanduser().resolve()
            )
        if self.candidate_report is not None:
            object.__setattr__(
                self, "candidate_report", self.candidate_report.expanduser().resolve()
            )


@dataclass(frozen=True)
class V1IndexConfig:
    future_offsets: tuple[int, ...] | None = None
    future_horizons_seconds: tuple[float, ...] | None = None
    future_offset_basis: str | None = None
    camera_names: tuple[str, ...] = PHYSICAL_CAMERA_NAMES
    action: ActionSamplingConfig = field(default_factory=ActionSamplingConfig)
    proprio_mode: str = "last_command"
    coordinate_frame: str = "head_camera@source_t0"
    require_raw_v1: bool = True
    verify_manifest_hashes: bool = True
    verify_hdf5_hashes: bool = False

    def __post_init__(self) -> None:
        if self.future_offsets is None:
            raise ValueError(
                "future_offsets must be explicitly provided; V1 has no implicit K"
            )
        if not isinstance(self.future_offsets, tuple) or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in self.future_offsets
        ):
            raise TypeError("future_offsets must be a tuple of integers")
        offsets = self.future_offsets
        cameras = tuple(str(value) for value in self.camera_names)
        if not offsets or any(value <= 0 for value in offsets):
            raise ValueError("future_offsets must contain positive frame offsets")
        if tuple(sorted(offsets)) != offsets or len(set(offsets)) != len(offsets):
            raise ValueError("future_offsets must be strictly increasing")
        if self.future_offset_basis is None:
            raise ValueError(
                "future_offset_basis must be explicitly provided with future_offsets"
            )
        if not isinstance(self.future_offset_basis, str):
            raise TypeError("future_offset_basis must be a string")
        if self.future_offset_basis not in KNOWN_FUTURE_OFFSET_BASES:
            raise ValueError(
                "future_offset_basis must be one of "
                f"{sorted(KNOWN_FUTURE_OFFSET_BASES)}, got "
                f"{self.future_offset_basis!r}"
            )
        if self.future_horizons_seconds is None:
            horizons: tuple[float, ...] | None = None
        else:
            if not isinstance(self.future_horizons_seconds, tuple) or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in self.future_horizons_seconds
            ):
                raise TypeError(
                    "future_horizons_seconds must be a tuple of numbers or None"
                )
            horizons = tuple(float(value) for value in self.future_horizons_seconds)
            if len(horizons) != len(offsets):
                raise ValueError(
                    "future_horizons_seconds and future_offsets must have the same K"
                )
            if not np.isfinite(np.asarray(horizons, dtype=np.float64)).all() or any(
                value <= 0 for value in horizons
            ):
                raise ValueError(
                    "future_horizons_seconds must contain finite positive values"
                )
            if any(right <= left for left, right in zip(horizons, horizons[1:])):
                raise ValueError("future_horizons_seconds must be strictly increasing")
        if self.future_offset_basis == LEGACY_SAVED_FRAME_OFFSET_BASIS:
            if horizons is not None:
                raise ValueError(
                    "legacy_saved_frame offsets must not define physical horizons"
                )
        else:
            expected_horizons = _FASTWAM_SUPPORTED_FUTURE_CONTRACTS.get(offsets)
            if expected_horizons is None:
                raise ValueError(
                    "fastwam_clip_frame@15hz requires supported logical offsets "
                    f"{tuple(_FASTWAM_SUPPORTED_FUTURE_CONTRACTS)!r}"
                )
            if horizons is None:
                raise ValueError(
                    "fastwam_clip_frame@15hz requires future_horizons_seconds"
                )
            if not np.allclose(
                horizons,
                expected_horizons,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "fastwam_clip_frame@15hz horizons must equal each "
                    "logical_offset / 15"
                )
        if cameras != PHYSICAL_CAMERA_NAMES:
            raise ValueError(
                "V1 camera order must be "
                f"{PHYSICAL_CAMERA_NAMES!r}, got {cameras!r}"
            )
        proprio_aliases = {
            "measured_qpos_real": "qpos_real",
            "last_command_drive_target": "last_command",
        }
        proprio = proprio_aliases.get(self.proprio_mode, self.proprio_mode)
        if proprio not in {"qpos_real", "last_command"}:
            raise ValueError(f"unsupported proprio_mode: {self.proprio_mode!r}")
        if not self.coordinate_frame.strip():
            raise ValueError("coordinate_frame must not be empty")
        object.__setattr__(self, "future_offsets", offsets)
        object.__setattr__(self, "future_horizons_seconds", horizons)
        object.__setattr__(self, "camera_names", cameras)
        object.__setattr__(self, "proprio_mode", proprio)


@dataclass(frozen=True)
class V1SampleIndex:
    episodes: tuple[V1EpisodeRecord, ...]
    sample_refs: Sequence[SampleRef]
    config: V1IndexConfig
    index_sha256: str | None = None
    source_manifest_bindings: tuple[tuple[str, str, str | None], ...] = ()
    index_audit: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("V1SampleIndex requires at least one episode")
        if len(self.sample_refs) <= 0:
            raise ValueError("no legal V1 source frames remain after boundary checks")
        if not isinstance(self.sample_refs[0], SampleRef):
            raise TypeError("V1SampleIndex must resolve SampleRef values")
        if self.index_sha256 is not None and not _valid_sha256(self.index_sha256):
            raise ValueError("index_sha256 must be lowercase SHA256 hex")
        names = [name for name, _ in self.index_audit]
        if len(names) != len(set(names)) or any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for name, value in self.index_audit
        ):
            raise ValueError("index_audit must contain unique non-negative counters")

    def __len__(self) -> int:
        return len(self.sample_refs)

    @property
    def compact_index_nbytes(self) -> int | None:
        value = getattr(self.sample_refs, "compact_nbytes", None)
        return None if value is None else int(value)


    def runtime_index_binding(self) -> dict[str, Any]:
        """Return composable exact evidence for this compact runtime index."""

        if not isinstance(self.sample_refs, CompactSampleRefSequence):
            raise TypeError(
                "runtime index evidence requires CompactSampleRefSequence"
            )
        return _runtime_index_binding_from_parts(
            index_config_sha256=_sha256_json(
                _index_config_payload(self.config),
                prefix=b"robotwin-momap-v1-index-config-v1",
            ),
            episode_indices=tuple(
                _compact_episode_index_binding(plan)
                for plan in self.sample_refs.plans
            ),
            index_audit=dict(self.index_audit),
        )


@dataclass(frozen=True)
class _CompactEpisodeWindows:
    """Contiguous NumPy storage for one episode's legal window addresses."""

    episode: V1EpisodeRecord
    episode_ordinal: int
    episode_id: str
    source_indices: np.ndarray
    target_indices: np.ndarray
    action_indices: np.ndarray
    action_query_time_base_sec: np.ndarray
    dense_action_query_times_sec: np.ndarray | None = None
    index_audit: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.episode_ordinal, bool)
            or not isinstance(self.episode_ordinal, int)
            or self.episode_ordinal < 0
        ):
            raise ValueError("compact episode_ordinal must be non-negative")
        count = int(self.source_indices.shape[0])
        if count <= 0 or self.source_indices.ndim != 1:
            raise ValueError("compact episode must contain legal source indices")
        if self.target_indices.ndim != 2 or self.target_indices.shape[0] != count:
            raise ValueError("compact target_indices shape mismatch")
        if np.any(self.target_indices <= self.source_indices[:, None]):
            raise ValueError("compact target_indices must be after source_indices")
        if self.target_indices.shape[1] > 1 and np.any(
            self.target_indices[:, 1:] <= self.target_indices[:, :-1]
        ):
            raise ValueError("compact target_indices must be strictly increasing")
        if self.action_indices.ndim != 2 or self.action_indices.shape[0] != count:
            raise ValueError("compact action_indices shape mismatch")
        if self.action_query_time_base_sec.shape != (count,):
            raise ValueError("compact action query base shape mismatch")
        if self.dense_action_query_times_sec is not None and (
            self.dense_action_query_times_sec.shape != self.action_indices.shape
        ):
            raise ValueError("compact dense action query-time shape mismatch")
        audit = _validated_index_audit(
            {
                "schema_version": V1_INDEX_AUDIT_SCHEMA_VERSION,
                "available": True,
                "counts": dict(self.index_audit),
            }
        )
        if (
            audit["counts"]["episode_count"] != 1
            or audit["counts"]["included_windows"] != count
        ):
            raise ValueError("compact episode index audit/count mismatch")
        for value in (
            self.source_indices,
            self.target_indices,
            self.action_indices,
            self.action_query_time_base_sec,
            self.dense_action_query_times_sec,
        ):
            if value is not None and value.flags.writeable:
                raise ValueError("compact index arrays must be read-only")

    @property
    def count(self) -> int:
        return int(self.source_indices.shape[0])

    @property
    def nbytes(self) -> int:
        return sum(
            int(value.nbytes)
            for value in (
                self.source_indices,
                self.target_indices,
                self.action_indices,
                self.action_query_time_base_sec,
                self.dense_action_query_times_sec,
            )
            if value is not None
        )


class CompactSampleRefSequence(Sequence[SampleRef]):
    """Prefix-sum address table that materializes only the requested ref."""

    __slots__ = ("_plans", "_prefix", "_config", "_compact_nbytes")

    def __init__(
        self,
        plans: Sequence[_CompactEpisodeWindows],
        *,
        config: V1IndexConfig,
    ) -> None:
        self._plans = tuple(plans)
        if not self._plans:
            raise ValueError("CompactSampleRefSequence requires legal windows")
        prefix = [0]
        for plan in self._plans:
            prefix.append(prefix[-1] + plan.count)
        self._prefix = tuple(prefix)
        self._config = config
        self._compact_nbytes = sum(plan.nbytes for plan in self._plans)

    @property
    def compact_nbytes(self) -> int:
        return self._compact_nbytes

    @property
    def plans(self) -> tuple[_CompactEpisodeWindows, ...]:
        return self._plans

    def episode_index_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return ordered half-open global sample ranges, one per episode."""

        return tuple(
            (self._prefix[position], self._prefix[position + 1])
            for position in range(len(self._plans))
        )

    def episode_index_ranges_binding(self) -> dict[str, Any]:
        """Return a compact identity for the ordered episode range partition."""

        schema_version = "robotwin_momap_episode_index_ranges_binding_v1"
        digest_algorithm = (
            "ordered_episode_id_ordinal_half_open_global_range_sha256_v1"
        )
        entries = [
            {
                "episode_id": plan.episode_id,
                "episode_ordinal": int(plan.episode_ordinal),
                "start": self._prefix[position],
                "stop": self._prefix[position + 1],
            }
            for position, plan in enumerate(self._plans)
        ]
        return {
            "schema_version": schema_version,
            "digest_algorithm": digest_algorithm,
            "episode_count": len(entries),
            "sample_count": len(self),
            "ranges_sha256": _sha256_json(
                {
                    "schema_version": schema_version,
                    "digest_algorithm": digest_algorithm,
                    "entries": entries,
                },
                prefix=b"robotwin-momap-episode-index-ranges-v1",
            ),
        }

    def __len__(self) -> int:
        return self._prefix[-1]

    def __getitem__(self, index: int | slice) -> SampleRef | tuple[SampleRef, ...]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("sample index must be an integer or slice")
        position = index + len(self) if index < 0 else index
        if position < 0 or position >= len(self):
            raise IndexError(index)
        plan_index = bisect_right(self._prefix, position) - 1
        plan = self._plans[plan_index]
        local_index = position - self._prefix[plan_index]
        action_indices = tuple(
            int(value) for value in plan.action_indices[local_index]
        )
        if self._config.action.mode == "baseline_rate_resampled@15Hz":
            base = float(plan.action_query_time_base_sec[local_index])
            query_times = base + np.arange(
                self._config.action.horizon,
                dtype=np.float64,
            ) / float(self._config.action.baseline_rate_hz)
        else:
            if plan.dense_action_query_times_sec is None:
                raise RuntimeError("dense compact index lacks action query times")
            query_times = plan.dense_action_query_times_sec[local_index]
        return SampleRef(
            episode_id=plan.episode_id,
            episode_path=plan.episode.episode_path,
            source_index=int(plan.source_indices[local_index]),
            target_indices=tuple(
                int(value) for value in plan.target_indices[local_index]
            ),
            future_offsets=self._config.future_offsets,
            future_horizons_seconds=self._config.future_horizons_seconds,
            future_offset_basis=self._config.future_offset_basis,
            camera_names=self._config.camera_names,
            action_start=action_indices[0],
            action_stop=action_indices[-1] + 1,
            coordinate_frame=self._config.coordinate_frame,
            action_indices=action_indices,
            action_query_times_sec=tuple(float(value) for value in query_times),
            action_sampling_mode=self._config.action.mode,
            action_rate_hz=(
                self._config.action.baseline_rate_hz
                if self._config.action.mode == "baseline_rate_resampled@15Hz"
                else None
            ),
            action_contract_schema_version=ACTION_CONTRACT_SCHEMA,
            action_label_kind=ACTION_LABEL_KIND,
            action_is_high_level_policy_action=False,
            action_normalization="none",
            proprio_mode=(
                "measured_qpos_real"
                if self._config.proprio_mode == "qpos_real"
                else "last_command_drive_target"
            ),
            proprio_normalization="none",
            momap_normalization="none",
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any, *, prefix: bytes = b"") -> str:
    digest = hashlib.sha256()
    if prefix:
        digest.update(prefix)
        digest.update(b"\0")
    digest.update(_canonical_json(value).encode("utf-8"))
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _readonly_unsigned_array(values: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.size <= 0 or raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain non-negative integers")
    if bool(np.any(raw < 0)):
        raise ValueError(f"{name} contains a negative address")
    maximum = int(raw.max())
    for dtype in (np.uint16, np.uint32, np.uint64):
        if maximum <= np.iinfo(dtype).max:
            compact = np.ascontiguousarray(raw, dtype=dtype)
            compact.setflags(write=False)
            return compact
    raise ValueError(f"{name} exceeds uint64 address range")


def _readonly_float64_array(values: Any, *, name: str) -> np.ndarray:
    compact = np.ascontiguousarray(values, dtype=np.float64)
    if compact.size <= 0 or not bool(np.isfinite(compact).all()):
        raise ValueError(f"{name} must contain finite values")
    compact.setflags(write=False)
    return compact


def _source_manifest_bindings(
    episodes: Sequence[V1EpisodeRecord],
) -> tuple[tuple[str, str, str | None], ...]:
    bindings: list[tuple[str, str, str | None]] = []
    paths = sorted(
        {
            episode.source_manifest
            for episode in episodes
            if episode.source_manifest is not None
        },
        key=str,
    )
    for path in paths:
        assert path is not None
        payload = json.loads(path.read_text(encoding="utf-8"))
        semantic = None
        if isinstance(payload, Mapping):
            for key in (
                "manifest_sha256",
                "receipt_sha256",
                "record_set_sha256",
            ):
                if _valid_sha256(payload.get(key)):
                    semantic = str(payload[key])
                    break
        bindings.append((str(path), _sha256_file(path), semantic))
    return tuple(bindings)


def _index_config_payload(config: V1IndexConfig) -> dict[str, Any]:
    return {
        "future_offsets": list(config.future_offsets),
        "future_horizons_seconds": (
            None
            if config.future_horizons_seconds is None
            else list(config.future_horizons_seconds)
        ),
        "future_offset_basis": config.future_offset_basis,
        "camera_names": list(config.camera_names),
        "action": {
            "mode": config.action.mode,
            "horizon": config.action.horizon,
            "baseline_rate_hz": config.action.baseline_rate_hz,
            "timestamp_tolerance_sec": config.action.timestamp_tolerance_sec,
        },
        "proprio_mode": config.proprio_mode,
        "coordinate_frame": config.coordinate_frame,
    }


def _compact_episode_index_binding(
    plan: _CompactEpisodeWindows,
) -> dict[str, Any]:
    digest = hashlib.sha256(
        b"robotwin-momap-v1-compact-episode-index-v1\0"
    )
    audit = _validated_index_audit(
        {
            "schema_version": V1_INDEX_AUDIT_SCHEMA_VERSION,
            "available": True,
            "counts": dict(plan.index_audit),
        }
    )
    identity = {
        "episode_ordinal": plan.episode_ordinal,
        "episode_path": str(plan.episode.episode_path),
        "episode_id": plan.episode_id,
        "expected_hdf5_sha256": plan.episode.expected_hdf5_sha256,
        "legal_sample_count": plan.count,
        "index_audit": audit,
        "index_audit_sha256": _sha256_json(
            audit,
            prefix=b"robotwin-momap-v1-index-audit-v1",
        ),
    }
    digest.update(_canonical_json(identity).encode("utf-8"))
    for name, array in (
        ("source_indices", plan.source_indices),
        ("target_indices", plan.target_indices),
        ("action_indices", plan.action_indices),
        ("action_query_time_base_sec", plan.action_query_time_base_sec),
        ("dense_action_query_times_sec", plan.dense_action_query_times_sec),
    ):
        digest.update(name.encode("ascii"))
        if array is None:
            digest.update(b"none\0")
            continue
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return {
        **identity,
        "episode_index_sha256": digest.hexdigest(),
    }


def _validated_index_audit(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "available",
        "counts",
    }:
        raise ValueError("runtime index_audit fields drifted")
    if (
        value.get("schema_version") != V1_INDEX_AUDIT_SCHEMA_VERSION
        or value.get("available") is not True
    ):
        raise ValueError("runtime index_audit is not strict available v1")
    counts = value.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != set(V1_INDEX_AUDIT_KEYS):
        raise ValueError("runtime index_audit counter fields drifted")
    result_counts: dict[str, int] = {}
    for name in V1_INDEX_AUDIT_KEYS:
        raw = counts.get(name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"runtime index_audit {name} is invalid")
        result_counts[name] = raw
    excluded = sum(
        value
        for name, value in result_counts.items()
        if name.startswith("excluded_")
    )
    if (
        result_counts["candidate_source_frames"]
        != result_counts["included_windows"] + excluded
    ):
        raise ValueError("runtime index_audit does not exactly cover candidates")
    return {
        "schema_version": V1_INDEX_AUDIT_SCHEMA_VERSION,
        "available": True,
        "counts": result_counts,
    }


def _runtime_index_binding_from_parts(
    *,
    index_config_sha256: str,
    episode_indices: Sequence[Mapping[str, Any]],
    index_audit: Mapping[str, int],
) -> dict[str, Any]:
    if not _valid_sha256(index_config_sha256):
        raise ValueError("runtime index_config_sha256 is invalid")
    entries = [dict(value) for value in episode_indices]
    expected_entry_keys = {
        "episode_ordinal",
        "episode_path",
        "episode_id",
        "expected_hdf5_sha256",
        "legal_sample_count",
        "index_audit",
        "index_audit_sha256",
        "episode_index_sha256",
    }
    if not entries:
        raise ValueError("runtime index requires episode evidence")
    for entry in entries:
        if set(entry) != expected_entry_keys:
            raise ValueError("runtime episode-index evidence fields drifted")
        ordinal = entry.get("episode_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("runtime episode ordinal is invalid")
        path = entry.get("episode_path")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("runtime episode-index path must be absolute")
        count = entry.get("legal_sample_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("runtime episode legal_sample_count is invalid")
        expected_hdf5 = entry.get("expected_hdf5_sha256")
        if expected_hdf5 is not None and not _valid_sha256(expected_hdf5):
            raise ValueError("runtime episode expected HDF5 SHA256 is invalid")
        if not _valid_sha256(entry.get("episode_index_sha256")):
            raise ValueError("runtime episode index SHA256 is invalid")
        entry_audit = _validated_index_audit(entry.get("index_audit"))
        entry_audit_sha = _sha256_json(
            entry_audit,
            prefix=b"robotwin-momap-v1-index-audit-v1",
        )
        if (
            entry.get("index_audit_sha256") != entry_audit_sha
            or entry_audit["counts"]["episode_count"] != 1
            or entry_audit["counts"]["included_windows"] != count
        ):
            raise ValueError("runtime episode index audit/counts differ")
        entry["index_audit"] = entry_audit
    entries.sort(key=lambda value: value["episode_ordinal"])
    ordinals = [entry["episode_ordinal"] for entry in entries]
    expected_ordinals = list(range(ordinals[0], ordinals[0] + len(entries)))
    if ordinals != expected_ordinals:
        raise ValueError("runtime episode ordinals are not a contiguous range")
    paths = [entry["episode_path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("runtime index contains duplicate episode paths")

    audit = _validated_index_audit(
        {
            "schema_version": V1_INDEX_AUDIT_SCHEMA_VERSION,
            "available": True,
            "counts": dict(index_audit),
        }
    )
    legal_sample_count = sum(
        int(entry["legal_sample_count"]) for entry in entries
    )
    composed_counts = {name: 0 for name in V1_INDEX_AUDIT_KEYS}
    for entry in entries:
        for name in V1_INDEX_AUDIT_KEYS:
            composed_counts[name] += entry["index_audit"]["counts"][name]
    if (
        audit["counts"]["episode_count"] != len(entries)
        or audit["counts"]["included_windows"] != legal_sample_count
        or audit["counts"] != composed_counts
    ):
        raise ValueError("runtime index audit differs from episode evidence")
    audit_sha256 = _sha256_json(
        audit,
        prefix=b"robotwin-momap-v1-index-audit-v1",
    )
    identity = {
        "schema_version": V1_RUNTIME_INDEX_BINDING_SCHEMA_VERSION,
        "composition_algorithm": V1_RUNTIME_INDEX_COMPOSITION_ALGORITHM,
        "index_config_sha256": index_config_sha256,
        "legal_sample_count": legal_sample_count,
        "episode_count": len(entries),
        "episode_ordinal_start": ordinals[0],
        "episode_ordinal_end": ordinals[-1] + 1,
        "index_audit": audit,
        "index_audit_sha256": audit_sha256,
        "episode_indices": entries,
    }
    identity["index_sha256"] = _sha256_json(
        identity,
        prefix=b"robotwin-momap-v1-runtime-index-binding-v1",
    )
    return identity


def validate_v1_runtime_index_binding(value: Any) -> dict[str, Any]:
    """Validate and canonically rebuild one runtime index identity."""

    if not isinstance(value, Mapping):
        raise TypeError("runtime index binding must be a mapping")
    required = {
        "schema_version",
        "composition_algorithm",
        "index_config_sha256",
        "index_sha256",
        "legal_sample_count",
        "episode_count",
        "episode_ordinal_start",
        "episode_ordinal_end",
        "index_audit",
        "index_audit_sha256",
        "episode_indices",
    }
    if set(value) != required:
        raise ValueError("runtime index binding fields drifted")
    if (
        value.get("schema_version")
        != V1_RUNTIME_INDEX_BINDING_SCHEMA_VERSION
        or value.get("composition_algorithm")
        != V1_RUNTIME_INDEX_COMPOSITION_ALGORITHM
    ):
        raise ValueError("runtime index binding schema/algorithm drifted")
    audit = _validated_index_audit(value.get("index_audit"))
    rebuilt = _runtime_index_binding_from_parts(
        index_config_sha256=str(value.get("index_config_sha256")),
        episode_indices=value.get("episode_indices", ()),
        index_audit=audit["counts"],
    )
    if dict(value) != rebuilt:
        raise ValueError("runtime index binding digest/counts differ")
    return rebuilt


def merge_v1_runtime_index_bindings(
    values: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compose exact disjoint shard-local bindings into the global index."""

    if not values:
        raise ValueError("runtime index merge requires at least one binding")
    bindings = [validate_v1_runtime_index_binding(value) for value in values]
    bindings.sort(key=lambda value: value["episode_ordinal_start"])
    cursor = bindings[0]["episode_ordinal_start"]
    for binding in bindings:
        if binding["episode_ordinal_start"] != cursor:
            raise ValueError(
                "runtime index shard ordinal ranges overlap or contain a gap"
            )
        cursor = binding["episode_ordinal_end"]
    config_sha = bindings[0]["index_config_sha256"]
    if any(value["index_config_sha256"] != config_sha for value in bindings):
        raise ValueError("runtime index configs differ across shards")
    entries: list[dict[str, Any]] = []
    counts = {name: 0 for name in V1_INDEX_AUDIT_KEYS}
    for binding in bindings:
        entries.extend(binding["episode_indices"])
        for name in V1_INDEX_AUDIT_KEYS:
            counts[name] += binding["index_audit"]["counts"][name]
    return _runtime_index_binding_from_parts(
        index_config_sha256=config_sha,
        episode_indices=entries,
        index_audit=counts,
    )


def select_v1_runtime_index_binding(
    value: Mapping[str, Any],
    *,
    episode_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Select an exact episode subset from a qualified global identity."""

    binding = validate_v1_runtime_index_binding(value)
    requested = [str(Path(path).expanduser().resolve()) for path in episode_paths]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("runtime index subset paths must be unique and non-empty")
    by_path = {
        entry["episode_path"]: entry for entry in binding["episode_indices"]
    }
    missing = sorted(set(requested) - set(by_path))
    if missing:
        raise ValueError(
            f"runtime index subset contains unknown episode paths: {missing[:4]}"
        )
    entries = [by_path[path] for path in requested]
    entries.sort(key=lambda entry: entry["episode_ordinal"])
    counts = {name: 0 for name in V1_INDEX_AUDIT_KEYS}
    for entry in entries:
        for name in V1_INDEX_AUDIT_KEYS:
            counts[name] += entry["index_audit"]["counts"][name]
    return _runtime_index_binding_from_parts(
        index_config_sha256=binding["index_config_sha256"],
        episode_indices=entries,
        index_audit=counts,
    )


def project_v1_sample_index_future_contract(
    index: V1SampleIndex,
    *,
    future_offsets: Sequence[int],
    future_horizons_seconds: Sequence[float],
    compute_index_sha256: bool = True,
) -> V1SampleIndex:
    """Project a frozen compact index onto an explicit subset of future targets.

    Source windows, episode order, Action labels, Proprio semantics and audit
    counts remain exactly those of the input index. Only target columns are
    selected. This is intentionally different from rebuilding a natural
    shorter-horizon index, which would admit additional episode-tail source
    windows and make an otherwise controlled K ablation use a different
    training population.
    """

    if not isinstance(index, V1SampleIndex):
        raise TypeError("index must be a V1SampleIndex")
    if not isinstance(index.sample_refs, CompactSampleRefSequence):
        raise TypeError("future-contract projection requires a compact V1 index")
    offsets = tuple(int(value) for value in future_offsets)
    horizons = tuple(float(value) for value in future_horizons_seconds)
    source_offsets = tuple(index.config.future_offsets)
    source_horizons = index.config.future_horizons_seconds
    if source_horizons is None:
        raise ValueError("source index lacks physical future horizons")
    try:
        columns = tuple(source_offsets.index(value) for value in offsets)
    except ValueError as exc:
        raise ValueError(
            "projected future offsets must be an ordered subset of source offsets"
        ) from exc
    if tuple(sorted(columns)) != columns or len(set(columns)) != len(columns):
        raise ValueError(
            "projected future offsets must preserve source offset order"
        )
    expected_horizons = tuple(float(source_horizons[column]) for column in columns)
    if len(horizons) != len(expected_horizons) or not np.allclose(
        horizons,
        expected_horizons,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "projected future horizons differ from the source index contract"
        )
    projected_config = replace(
        index.config,
        future_offsets=offsets,
        future_horizons_seconds=horizons,
    )
    if projected_config == index.config:
        return index

    if not isinstance(compute_index_sha256, bool):
        raise TypeError("compute_index_sha256 must be a bool")

    projected_plans: list[_CompactEpisodeWindows] = []
    arithmetic_step = columns[1] - columns[0] if len(columns) > 1 else 1
    use_strided_view = arithmetic_step > 0 and all(
        right - left == arithmetic_step
        for left, right in zip(columns[:-1], columns[1:], strict=True)
    )
    for plan in index.sample_refs.plans:
        if use_strided_view:
            stop = columns[-1] + 1
            target_indices = plan.target_indices[
                :, columns[0] : stop : arithmetic_step
            ]
        else:
            target_indices = np.ascontiguousarray(
                np.take(plan.target_indices, columns, axis=1)
            )
        target_indices.setflags(write=False)
        projected_plans.append(
            replace(plan, target_indices=target_indices)
        )
    sequence = CompactSampleRefSequence(
        projected_plans,
        config=projected_config,
    )
    if len(sequence) != len(index):
        raise RuntimeError("future-contract projection changed sample count")
    return V1SampleIndex(
        episodes=index.episodes,
        sample_refs=sequence,
        config=projected_config,
        index_sha256=(
            _compact_sequence_sha256(
                sequence,
                index_audit=index.index_audit,
            )
            if compute_index_sha256
            else None
        ),
        source_manifest_bindings=index.source_manifest_bindings,
        index_audit=index.index_audit,
    )


def slice_v1_sample_index_by_episode_ordinals(
    index: V1SampleIndex,
    *,
    start: int,
    end: int,
) -> V1SampleIndex:
    """Return a zero-copy compact episode slice from a cached V1 index.

    The NumPy/mmap pools remain shared with ``index``.  ``index_sha256`` is
    intentionally left unset so slicing itself does not reread the payload;
    callers that need evidence can call ``runtime_index_binding()`` once for
    the complete selected shard.
    """

    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise ValueError("episode ordinal slice must satisfy 0 <= start < end")
    if not isinstance(index, V1SampleIndex):
        raise TypeError("index must be a V1SampleIndex")
    if not isinstance(index.sample_refs, CompactSampleRefSequence):
        raise TypeError("episode ordinal slicing requires a compact V1 index")
    source_plans = index.sample_refs.plans
    first_ordinal = source_plans[0].episode_ordinal
    local_start = start - first_ordinal
    local_end = end - first_ordinal
    if local_start < 0 or local_end > len(source_plans):
        raise ValueError(
            "requested episode ordinal range is outside the cached index"
        )
    plans = source_plans[local_start:local_end]
    observed_ordinals = tuple(plan.episode_ordinal for plan in plans)
    expected_ordinals = tuple(range(start, end))
    if observed_ordinals != expected_ordinals:
        raise ValueError(
            "cached V1 index does not exactly cover the requested episode "
            f"ordinal range [{start}, {end})"
        )
    counts = {name: 0 for name in V1_INDEX_AUDIT_KEYS}
    for plan in plans:
        audit = dict(plan.index_audit)
        if set(audit) != set(V1_INDEX_AUDIT_KEYS):
            raise ValueError("cached episode index audit fields drifted")
        for name in V1_INDEX_AUDIT_KEYS:
            counts[name] += int(audit[name])
    sequence = CompactSampleRefSequence(plans, config=index.config)
    if counts["episode_count"] != len(plans):
        raise ValueError("cached episode count differs from its index audit")
    if counts["included_windows"] != len(sequence):
        raise ValueError("cached sample count differs from its index audit")
    return V1SampleIndex(
        episodes=tuple(plan.episode for plan in plans),
        sample_refs=sequence,
        config=index.config,
        index_sha256=None,
        source_manifest_bindings=(),
        index_audit=tuple(sorted(counts.items())),
    )


def _compact_sequence_sha256(
    sequence: CompactSampleRefSequence,
    *,
    index_audit: tuple[tuple[str, int], ...],
) -> str:
    return _runtime_index_binding_from_parts(
        index_config_sha256=_sha256_json(
            _index_config_payload(sequence._config),
            prefix=b"robotwin-momap-v1-index-config-v1",
        ),
        episode_indices=tuple(
            _compact_episode_index_binding(plan) for plan in sequence.plans
        ),
        index_audit=dict(index_audit),
    )["index_sha256"]


def _required_count(value: Mapping[str, Any], name: str, *, label: str) -> int:
    count = value.get(name)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{label} {name} must be a non-negative integer")
    return count


def _require_candidate_only(value: Mapping[str, Any], *, label: str) -> None:
    if (
        value.get("candidate_only") is not True
        or value.get("production_authorized") is not False
        or value.get("formal_index_eligible") is not False
    ):
        raise ValueError(f"{label} is not a candidate-only, non-production artifact")


def _validate_candidate_report(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    episode_path: Path,
    episode_sha256: str,
    episode_id: str,
    frame_count: int,
    seed: int,
    canonical_trajectory_sha256: str,
    planning_scene_sha256: str,
    replay_scene_sha256: str,
) -> None:
    if report.get("schema_version") != "robotwin_momap_canonical_rollout_v1_candidate_report_v1":
        raise ValueError(f"candidate report schema mismatch: {report_path}")
    if report.get("status") != "PASS":
        raise ValueError(f"candidate report is not PASS: {report_path}")
    _require_candidate_only(report, label=f"candidate report {report_path}")
    if report.get("raw_episode_schema_version") != RAW_V1_SCHEMA:
        raise ValueError(f"candidate report raw episode schema mismatch: {report_path}")
    if tuple(report.get("camera_names", ())) != PHYSICAL_CAMERA_NAMES:
        raise ValueError(f"candidate report camera order mismatch: {report_path}")
    if report.get("frame_count") != frame_count:
        raise ValueError(f"candidate report frame_count mismatch: {report_path}")

    identity = report.get("canonical_rollout_identity")
    if not isinstance(identity, Mapping) or (
        identity.get("schema_version")
        != "robotwin_momap_canonical_rollout_identity_v2"
        or identity.get("passed") is not True
        or identity.get("rollout_id") != episode_id
        or identity.get("seed") != seed
        or identity.get("planning_scene_sha256") != planning_scene_sha256
        or identity.get("replay_scene_sha256") != replay_scene_sha256
    ):
        raise ValueError(f"candidate report rollout identity mismatch: {report_path}")
    artifacts = report.get("artifacts")
    episode_artifact = (
        artifacts.get("episode_hdf5") if isinstance(artifacts, Mapping) else None
    )
    if not isinstance(episode_artifact, Mapping):
        raise ValueError(f"candidate report lacks episode_hdf5 artifact: {report_path}")
    artifact_path = Path(str(episode_artifact.get("path", ""))).expanduser().resolve()
    if artifact_path != episode_path:
        raise ValueError(f"candidate report HDF5 path mismatch: {report_path}")
    if episode_artifact.get("sha256") != episode_sha256:
        raise ValueError(f"candidate report HDF5 SHA mismatch: {report_path}")
    trajectory_artifact = artifacts.get("canonical_trajectory")
    if not isinstance(trajectory_artifact, Mapping) or (
        trajectory_artifact.get("sha256") != canonical_trajectory_sha256
    ):
        raise ValueError(
            f"candidate report canonical trajectory mismatch: {report_path}"
        )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root must be an object: {path}")
    return value


def _rank_summary_records(
    path: Path,
    *,
    verify_manifest_hashes: bool,
) -> list[V1EpisodeRecord]:
    summary = _load_json_object(path, "V1 rank summary")
    schema = str(summary.get("schema_version", ""))
    if schema not in KNOWN_RANK_SUMMARY_SCHEMAS:
        raise ValueError(f"unsupported V1 rank summary schema: {schema!r}")
    if summary.get("passed") is not True or summary.get("status") != "PASS":
        raise ValueError(f"V1 rank summary is not PASS: {path}")
    _require_candidate_only(summary, label=f"V1 rank summary {path}")
    raw_records = summary.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"V1 rank summary has no records: {path}")
    expected_count = _required_count(
        summary, "expected_candidate_count", label=f"V1 rank summary {path}"
    )
    verified_count = _required_count(
        summary, "verified_candidate_count", label=f"V1 rank summary {path}"
    )
    if expected_count != verified_count or verified_count != len(raw_records):
        raise ValueError(f"V1 rank summary candidate count mismatch: {path}")

    result: list[V1EpisodeRecord] = []
    slots: set[tuple[str, int]] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"rank summary record {index} is not an object")
        hdf5 = str(raw.get("hdf5", "")).strip()
        hdf5_sha256 = str(raw.get("hdf5_sha256", ""))
        candidate_report = str(raw.get("candidate_report", "")).strip()
        candidate_report_sha256 = str(raw.get("candidate_report_sha256", ""))
        task_name = str(raw.get("task_name", "")).strip()
        task_config = str(raw.get("task_config", "")).strip()
        variant = str(raw.get("variant", "")).strip()
        scope_id = str(raw.get("scope_id", "")).strip()
        record_episode_id = str(raw.get("canonical_rollout_id", "")).strip()
        trajectory_sha256 = str(raw.get("canonical_trajectory_sha256", ""))
        planning_scene_sha256 = str(raw.get("planning_scene_sha256", ""))
        replay_scene_sha256 = str(raw.get("replay_scene_sha256", ""))
        frame_count = raw.get("frame_count")
        candidate_ordinal = raw.get("candidate_ordinal")
        seed = raw.get("seed")
        episode_number = raw.get("episode_id")
        if (
            not hdf5
            or not candidate_report
            or not task_name
            or not task_config
            or not variant
            or not scope_id
            or not record_episode_id
        ):
            raise ValueError(
                f"rank summary record {index} lacks required identity fields"
            )
        if not all(
            _valid_sha256(value)
            for value in (
                hdf5_sha256,
                candidate_report_sha256,
                trajectory_sha256,
                planning_scene_sha256,
                replay_scene_sha256,
            )
        ):
            raise ValueError(f"rank summary record {index} has an invalid SHA-256")
        for field_name, field_value in (
            ("frame_count", frame_count),
            ("candidate_ordinal", candidate_ordinal),
        ):
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value <= 0
            ):
                raise ValueError(
                    f"rank summary record {index} {field_name} must be positive int"
                )
        for field_name, field_value in (
            ("seed", seed),
            ("episode_id", episode_number),
        ):
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value < 0
            ):
                raise ValueError(
                    f"rank summary record {index} {field_name} must be non-negative int"
                )
        expected_variant = {
            "demo_clean": "clean",
            "demo_randomized": "randomized",
        }.get(task_config)
        if expected_variant != variant or scope_id != f"{task_name}:{variant}":
            raise ValueError(f"rank summary record {index} task/config/scope mismatch")
        if (
            raw.get("candidate_only") is not True
            or raw.get("production_authorized") is not False
            or raw.get("formal_index_eligible") is not False
        ):
            raise ValueError(f"rank summary record {index} is not candidate-only")
        slot = (scope_id, int(candidate_ordinal))
        if slot in slots:
            raise ValueError(f"duplicate candidate slot in rank summary: {slot}")
        slots.add(slot)
        timeline_sha256 = raw.get("saved_frame_timeline_identity_sha256")
        if schema.endswith("_v4") and not _valid_sha256(timeline_sha256):
            raise ValueError(
                f"rank summary v4 record {index} lacks timeline identity SHA"
            )

        path_value = Path(hdf5).expanduser().resolve()
        if not path_value.is_file():
            raise FileNotFoundError(f"V1 episode does not exist: {path_value}")
        report_path = Path(candidate_report).expanduser().resolve()
        if not report_path.is_file():
            raise FileNotFoundError(f"V1 candidate report does not exist: {report_path}")
        if verify_manifest_hashes and _sha256_file(report_path) != candidate_report_sha256:
            raise ValueError(f"V1 candidate report SHA mismatch: {report_path}")
        report = _load_json_object(report_path, "V1 candidate report")
        _validate_candidate_report(
            report,
            report_path=report_path,
            episode_path=path_value,
            episode_sha256=hdf5_sha256,
            episode_id=record_episode_id,
            frame_count=int(frame_count),
            seed=int(seed),
            canonical_trajectory_sha256=trajectory_sha256,
            planning_scene_sha256=planning_scene_sha256,
            replay_scene_sha256=replay_scene_sha256,
        )
        result.append(
            V1EpisodeRecord(
                episode_path=path_value,
                episode_id=record_episode_id,
                task_name=task_name,
                task_config=task_config,
                variant=variant,
                source_manifest=path,
                expected_hdf5_sha256=hdf5_sha256,
                candidate_report=report_path,
                candidate_report_sha256=candidate_report_sha256,
                expected_frame_count=int(frame_count),
                scope_id=scope_id,
                candidate_ordinal=int(candidate_ordinal),
                seed=int(seed),
                expected_episode_number=int(episode_number),
                canonical_trajectory_sha256=trajectory_sha256,
                planning_scene_sha256=planning_scene_sha256,
                replay_scene_sha256=replay_scene_sha256,
                saved_frame_timeline_identity_sha256=(
                    str(timeline_sha256)
                    if timeline_sha256 is not None
                    else None
                ),
            )
        )
    return result


def _record_set_sha256(
    records: Sequence[V1EpisodeRecord],
    *,
    include_timeline: bool,
) -> str:
    rows: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (str(item.scope_id), int(item.candidate_ordinal or 0)),
    ):
        row = {
            "scope_id": record.scope_id,
            "candidate_ordinal": record.candidate_ordinal,
            "seed": record.seed,
            "hdf5_sha256": record.expected_hdf5_sha256,
            "candidate_report_sha256": record.candidate_report_sha256,
        }
        if include_timeline:
            row["saved_frame_timeline_identity_sha256"] = (
                record.saved_frame_timeline_identity_sha256
            )
        rows.append(row)
    return _sha256_json(rows, prefix=_RECORD_SET_PREFIX)


def load_v1_episode_records(
    *,
    global_receipt: str | Path | None = None,
    rank_summaries: Sequence[str | Path] | None = None,
    episode_paths: Sequence[str | Path] | None = None,
    verify_manifest_hashes: bool = True,
    verify_hdf5_hashes: bool = False,
) -> tuple[V1EpisodeRecord, ...]:
    """Load one explicit episode set; data-root globbing is intentionally absent."""

    supplied = sum(
        value is not None
        for value in (global_receipt, rank_summaries, episode_paths)
    )
    if supplied != 1:
        raise ValueError(
            "provide exactly one of global_receipt, rank_summaries, episode_paths"
        )

    records: list[V1EpisodeRecord] = []
    if global_receipt is not None:
        receipt_path = Path(global_receipt).expanduser().resolve()
        receipt = _load_json_object(receipt_path, "V1 global receipt")
        schema = str(receipt.get("schema_version", ""))
        if schema not in KNOWN_GLOBAL_RECEIPT_SCHEMAS:
            raise ValueError(f"unsupported V1 global receipt schema: {schema!r}")
        if receipt.get("passed") is not True or receipt.get("status") != "PASS":
            raise ValueError(f"V1 global receipt is not PASS: {receipt_path}")
        _require_candidate_only(receipt, label=f"V1 global receipt {receipt_path}")
        paths = receipt.get("rank_summary_paths")
        hashes = receipt.get("rank_summary_sha256")
        if not isinstance(paths, list) or not paths:
            raise ValueError("V1 global receipt has no rank_summary_paths")
        if not isinstance(hashes, list) or len(hashes) != len(paths):
            raise ValueError("V1 global receipt rank summary/hash length mismatch")
        if len({str(path) for path in paths}) != len(paths):
            raise ValueError("V1 global receipt has duplicate rank summary paths")
        for raw_path, expected_hash in zip(paths, hashes):
            if not _valid_sha256(expected_hash):
                raise ValueError("V1 global receipt has an invalid rank summary SHA")
            summary_path = Path(str(raw_path)).expanduser().resolve()
            if not summary_path.is_file():
                raise FileNotFoundError(f"missing V1 rank summary: {summary_path}")
            if verify_manifest_hashes and _sha256_file(summary_path) != expected_hash:
                raise ValueError(f"V1 rank summary SHA mismatch: {summary_path}")
            records.extend(
                _rank_summary_records(
                    summary_path,
                    verify_manifest_hashes=verify_manifest_hashes,
                )
            )

        expected_count = _required_count(
            receipt, "expected_candidate_count", label=f"V1 global receipt {receipt_path}"
        )
        verified_count = _required_count(
            receipt, "verified_candidate_count", label=f"V1 global receipt {receipt_path}"
        )
        if expected_count != verified_count or verified_count != len(records):
            raise ValueError("V1 global receipt candidate count mismatch")
        slots = [(str(record.scope_id), int(record.candidate_ordinal or 0)) for record in records]
        if len(set(slots)) != len(slots):
            raise ValueError("V1 global receipt contains duplicate candidate slots")
        claimed_record_set = receipt.get("record_set_sha256")
        if not _valid_sha256(claimed_record_set):
            raise ValueError("V1 global receipt has an invalid record_set_sha256")
        include_timeline = schema.endswith("_v4")
        if include_timeline and any(
            not _valid_sha256(record.saved_frame_timeline_identity_sha256)
            for record in records
        ):
            raise ValueError("V1 global v4 records lack timeline identity SHA")
        actual_record_set = _record_set_sha256(
            records,
            include_timeline=include_timeline,
        )
        if actual_record_set != claimed_record_set:
            raise ValueError("V1 global receipt record_set_sha256 mismatch")
    elif rank_summaries is not None:
        if not rank_summaries:
            raise ValueError("rank_summaries must not be empty")
        for raw_path in rank_summaries:
            records.extend(
                _rank_summary_records(
                    Path(raw_path).expanduser().resolve(),
                    verify_manifest_hashes=verify_manifest_hashes,
                )
            )
    else:
        assert episode_paths is not None
        if not episode_paths:
            raise ValueError("episode_paths must not be empty")
        for raw_path in episode_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"V1 episode does not exist: {path}")
            records.append(V1EpisodeRecord(episode_path=path))

    if verify_hdf5_hashes:
        for record in records:
            if record.expected_hdf5_sha256 is None:
                raise ValueError(
                    "verify_hdf5_hashes requires receipt-backed episode records"
                )
            if _sha256_file(record.episode_path) != record.expected_hdf5_sha256:
                raise ValueError(f"V1 episode HDF5 SHA mismatch: {record.episode_path}")

    receipt_slots = [
        (record.scope_id, record.candidate_ordinal)
        for record in records
        if record.source_manifest is not None
    ]
    if len(receipt_slots) != len(set(receipt_slots)):
        raise ValueError("V1 manifests contain duplicate candidate slots")

    unique: dict[Path, V1EpisodeRecord] = {}
    for record in records:
        if record.episode_path in unique:
            raise ValueError(f"duplicate V1 episode: {record.episode_path}")
        unique[record.episode_path] = record
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.task_name or "",
                item.task_config or "",
                item.episode_id or "",
                str(item.episode_path),
            ),
        )
    )


def _source_timestamp(reader: V1ReaderLike, source_index: int) -> float:
    frame = reader.read_frame_fields(
        source_index,
        camera_names=(),
        observation_fields=(),
        private_geometry_fields=(),
        timeline_fields=("timestamp_sec",),
        robot_state_fields=(),
        entity_fields=(),
    )
    try:
        timestamp = float(frame["timeline"]["timestamp_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"source frame {source_index} has no valid timeline.timestamp_sec"
        ) from exc
    if not np.isfinite(timestamp):
        raise ValueError(f"source frame {source_index} timestamp is NaN/Inf")
    return timestamp


def resolve_action_selection(
    reader: V1ReaderLike,
    source_index: int,
    config: ActionSamplingConfig,
    *,
    control_timestamps_before_sec: np.ndarray | None = None,
    control_timestamps_validated: bool = False,
    source_timestamp_sec: float | None = None,
    source_control_span: tuple[int, int] | None = None,
) -> ActionSelection:
    """Resolve exact future control rows for one source observation.

    The first row is always the saved frame's exclusive applied-control
    endpoint.  Baseline-rate mode then samples target times
    ``observation_time + j / 15`` and takes the latest future command whose
    pre-step timestamp is at or before each target time.  This is the command
    active at that instant; selecting the next row would leak one control step.
    """

    if source_control_span is None:
        applied_start, future_start = reader.saved_frame_control_span(source_index)
    else:
        if len(source_control_span) != 2:
            raise ValueError("source_control_span must contain start and stop")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in source_control_span
        ):
            raise TypeError("source_control_span values must be integers")
        applied_start, future_start = (
            int(source_control_span[0]),
            int(source_control_span[1]),
        )
    count = int(reader.control_stream_count)
    if (
        applied_start < 0
        or future_start < applied_start
        or future_start > count
    ):
        raise ValueError(
            f"source {source_index} has corrupt control span "
            f"[{applied_start}:{future_start}] for count {count}"
        )
    if future_start == count:
        raise ActionHorizonUnavailable(
            f"source {source_index} has no future control at endpoint {future_start}"
        )

    source_time = (
        _source_timestamp(reader, source_index)
        if source_timestamp_sec is None
        else float(source_timestamp_sec)
    )
    if not np.isfinite(source_time):
        raise ValueError(f"source frame {source_index} timestamp is NaN/Inf")

    if config.mode == "dense250Hz":
        stop = future_start + config.horizon
        if stop > count:
            raise ActionHorizonUnavailable(
                f"dense action [{future_start}:{stop}] exceeds control count {count}"
            )
        indices = tuple(range(future_start, stop))
        if control_timestamps_before_sec is None:
            fields = reader.read_control_stream_fields(
                future_start,
                stop,
                fields=("timestamp_before_sec",),
            )
            timestamps = np.asarray(fields["timestamp_before_sec"], dtype=np.float64)
        else:
            full_timestamps = np.asarray(
                control_timestamps_before_sec, dtype=np.float64
            )
            if full_timestamps.shape != (count,):
                raise ValueError(
                    f"control timestamp shape={full_timestamps.shape}, "
                    f"expected={(count,)}"
                )
            if not control_timestamps_validated and (
                not np.isfinite(full_timestamps).all()
                or (
                    full_timestamps.size > 1
                    and np.any(np.diff(full_timestamps) <= 0)
                )
            ):
                raise ValueError(
                    "control timestamps must be finite and strictly increasing"
                )
            timestamps = full_timestamps[future_start:stop]
        if timestamps.shape != (config.horizon,) or (
            not np.isfinite(timestamps).all()
            or (timestamps.size > 1 and np.any(np.diff(timestamps) <= 0))
        ):
            raise ValueError("dense action timestamps are invalid")
        if not np.isclose(
            timestamps[0],
            source_time,
            rtol=0.0,
            atol=config.timestamp_tolerance_sec,
        ):
            raise ValueError(
                "saved-frame control endpoint timestamp does not match observation"
            )
        return ActionSelection(future_start, indices, tuple(float(x) for x in timestamps))

    if control_timestamps_before_sec is None:
        fields = reader.read_control_stream_fields(
            fields=("timestamp_before_sec",)
        )
        timestamps = np.asarray(fields["timestamp_before_sec"], dtype=np.float64)
    else:
        timestamps = np.asarray(control_timestamps_before_sec, dtype=np.float64)
    if timestamps.shape != (count,):
        raise ValueError(
            f"control timestamp shape={timestamps.shape}, expected={(count,)}"
        )
    if not control_timestamps_validated and (
        not np.isfinite(timestamps).all()
        or (timestamps.size > 1 and np.any(np.diff(timestamps) <= 0))
    ):
        raise ValueError("control timestamps must be finite and strictly increasing")

    if not np.isclose(
        timestamps[future_start],
        source_time,
        rtol=0.0,
        atol=config.timestamp_tolerance_sec,
    ):
        raise ValueError(
            "saved-frame control endpoint timestamp does not match observation"
        )
    query = source_time + np.arange(config.horizon, dtype=np.float64) / float(
        config.baseline_rate_hz
    )
    tolerance = float(config.timestamp_tolerance_sec)
    if query[-1] > timestamps[-1] + tolerance:
        raise ActionHorizonUnavailable(
            f"15Hz action horizon from source {source_index} exceeds control stream"
        )
    # Selection itself is strictly causal.  Timestamp tolerance is only for
    # validating the source boundary and horizon availability; adding it to a
    # query would select a command that occurs just after that query.
    selected = np.searchsorted(
        timestamps,
        query,
        side="right",
    ).astype(np.int64) - 1
    selected = np.maximum(selected, future_start)
    # The first post-observation command is part of the contract, even if the
    # two clocks differ by a sub-nanosecond serialization roundoff.
    selected[0] = future_start
    if np.any(selected < 0) or np.any(selected >= count):
        raise ActionHorizonUnavailable(
            f"15Hz action horizon from source {source_index} exceeds control stream"
        )
    if selected.size > 1 and np.any(np.diff(selected) <= 0):
        raise ValueError(
            "15Hz action queries map to duplicate/out-of-order control rows"
        )
    if selected.size > 1 and np.any(timestamps[selected[1:]] > query[1:]):
        raise ValueError(
            "15Hz action selection violated latest-command-at-or-before causality"
        )
    return ActionSelection(
        source_control_start=future_start,
        indices=tuple(int(value) for value in selected),
        query_times_sec=tuple(float(value) for value in query),
    )


def _resolved_episode_identity(
    record: V1EpisodeRecord, metadata: Mapping[str, Any]
) -> str:
    value = str(metadata.get("rollout_id", "")).strip() or record.episode_id
    if value is None or not str(value).strip():
        value = record.episode_path.stem
    return str(value)


def _validate_record_metadata(
    record: V1EpisodeRecord,
    metadata: Mapping[str, Any],
    *,
    frame_count: int,
) -> None:
    for record_value, metadata_key in (
        (record.task_name, "task_name"),
        (record.task_config, "task_config"),
    ):
        if record_value is not None and str(metadata.get(metadata_key, "")) != record_value:
            raise ValueError(
                f"rank summary/HDF5 {metadata_key} mismatch for {record.episode_path}"
            )
    if record.episode_id is not None and metadata.get("rollout_id") != record.episode_id:
        raise ValueError(f"rank summary/HDF5 rollout_id mismatch for {record.episode_path}")
    if record.seed is not None and metadata.get("seed") != record.seed:
        raise ValueError(f"rank summary/HDF5 seed mismatch for {record.episode_path}")
    if (
        record.expected_episode_number is not None
        and metadata.get("canonical_episode_id") != record.expected_episode_number
    ):
        raise ValueError(
            f"rank summary/HDF5 canonical_episode_id mismatch for {record.episode_path}"
        )
    if (
        record.canonical_trajectory_sha256 is not None
        and metadata.get("trajectory_artifact_sha256")
        != record.canonical_trajectory_sha256
    ):
        raise ValueError(
            f"rank summary/HDF5 trajectory SHA mismatch for {record.episode_path}"
        )
    identity = metadata.get("canonical_rollout_identity")
    if record.source_manifest is not None and not isinstance(identity, Mapping):
        raise ValueError(
            f"receipt-backed HDF5 lacks canonical identity: {record.episode_path}"
        )
    if isinstance(identity, Mapping) and record.source_manifest is not None:
        if (
            identity.get("schema_version")
            != "robotwin_momap_canonical_rollout_identity_v2"
            or identity.get("passed") is not True
            or identity.get("rollout_id") != record.episode_id
            or identity.get("seed") != record.seed
            or identity.get("planning_scene_sha256")
            != record.planning_scene_sha256
            or identity.get("replay_scene_sha256") != record.replay_scene_sha256
        ):
            raise ValueError(
                f"rank summary/HDF5 canonical identity mismatch for "
                f"{record.episode_path}"
            )
    if record.expected_frame_count is not None and (
        frame_count != record.expected_frame_count
        or metadata.get("captured_frames") != record.expected_frame_count
        or metadata.get("expected_num_frames") != record.expected_frame_count
    ):
        raise ValueError(f"rank summary/HDF5 frame_count mismatch for {record.episode_path}")
    if record.source_manifest is not None and (
        metadata.get("candidate_only") is not True
        or metadata.get("production_authorized") is not False
        or metadata.get("formal_index_eligible") is not False
    ):
        raise ValueError(f"receipt-backed HDF5 is not candidate-only: {record.episode_path}")


def _validate_control_stream_contract(reader: V1ReaderLike) -> None:
    contract = reader.control_stream_contract
    if not isinstance(contract, Mapping):
        raise ValueError("control_stream contract must be a mapping")
    if contract.get("schema_version") != ACTION_CONTRACT_SCHEMA:
        raise ValueError("control_stream action contract schema mismatch")
    if contract.get("command_kind") != ACTION_LABEL_KIND:
        raise ValueError("control_stream command_kind is not low-level drive target")
    if contract.get("high_level_policy_action") is not False:
        raise ValueError("control_stream must not claim high-level policy actions")
    if contract.get("future_label") != ACTION_FUTURE_LABEL:
        raise ValueError("control_stream future-label boundary is not explicit")


def _validate_episode_control_alignment(
    reader: V1ReaderLike,
    *,
    frame_count: int,
    timestamp_before_sec: np.ndarray,
    timestamp_after_sec: np.ndarray,
    tolerance_sec: float,
    frame_timestamps_sec: np.ndarray | None = None,
    saved_frame_applied_start: np.ndarray | None = None,
    saved_frame_applied_stop: np.ndarray | None = None,
) -> np.ndarray:
    count = int(reader.control_stream_count)
    before = np.asarray(timestamp_before_sec, dtype=np.float64)
    after = np.asarray(timestamp_after_sec, dtype=np.float64)
    if before.shape != (count,) or after.shape != (count,):
        raise ValueError("control timestamp count differs from control stream")
    if (
        not np.isfinite(before).all()
        or not np.isfinite(after).all()
        or np.any(after <= before)
        or (count > 1 and np.any(np.diff(before) <= 0))
        or (count > 1 and np.any(np.diff(after) <= 0))
    ):
        raise ValueError("control timestamps must be finite and strictly increasing")
    if count > 1 and not np.allclose(
        after[:-1], before[1:], rtol=0.0, atol=tolerance_sec
    ):
        raise ValueError("adjacent control timestamps are not contiguous")

    batch_values = (
        frame_timestamps_sec,
        saved_frame_applied_start,
        saved_frame_applied_stop,
    )
    if any(value is not None for value in batch_values) and not all(
        value is not None for value in batch_values
    ):
        raise ValueError(
            "batched timeline/control spans must be provided together"
        )
    if frame_timestamps_sec is None:
        frame_timestamps = np.empty((frame_count,), dtype=np.float64)
        applied_starts = np.empty((frame_count,), dtype=np.int64)
        applied_stops = np.empty((frame_count,), dtype=np.int64)
        for frame_index in range(frame_count):
            applied_start, applied_stop = reader.saved_frame_control_span(
                frame_index
            )
            applied_starts[frame_index] = applied_start
            applied_stops[frame_index] = applied_stop
            frame_timestamps[frame_index] = _source_timestamp(
                reader,
                frame_index,
            )
    else:
        raw_starts = np.asarray(saved_frame_applied_start)
        raw_stops = np.asarray(saved_frame_applied_stop)
        if raw_starts.dtype.kind not in {"i", "u"} or raw_stops.dtype.kind not in {
            "i",
            "u",
        }:
            raise ValueError("saved-frame control spans must contain integers")
        applied_starts = np.asarray(raw_starts, dtype=np.int64)
        applied_stops = np.asarray(raw_stops, dtype=np.int64)
        frame_timestamps = np.asarray(frame_timestamps_sec, dtype=np.float64)
        if (
            applied_starts.shape != (frame_count,)
            or applied_stops.shape != (frame_count,)
            or frame_timestamps.shape != (frame_count,)
            or not np.isfinite(frame_timestamps).all()
        ):
            raise ValueError("batched timeline/control span shapes are invalid")

    previous_stop = 0
    for frame_index in range(frame_count):
        applied_start = int(applied_starts[frame_index])
        applied_stop = int(applied_stops[frame_index])
        if (
            applied_start != previous_stop
            or applied_start < 0
            or applied_stop < applied_start
            or applied_stop > count
        ):
            raise ValueError(
                f"saved-frame control spans are corrupt at frame {frame_index}: "
                f"[{applied_start}:{applied_stop}], previous_stop={previous_stop}"
            )
        timestamp = float(frame_timestamps[frame_index])
        if applied_stop > 0 and not np.isclose(
            after[applied_stop - 1],
            timestamp,
            rtol=0.0,
            atol=tolerance_sec,
        ):
            raise ValueError(
                f"saved frame {frame_index} timestamp disagrees with applied control"
            )
        if applied_stop < count and not np.isclose(
            before[applied_stop],
            timestamp,
            rtol=0.0,
            atol=tolerance_sec,
        ):
            raise ValueError(
                f"saved frame {frame_index} timestamp disagrees with future control"
            )
        previous_stop = applied_stop
    return frame_timestamps


def build_v1_sample_index(
    *,
    config: V1IndexConfig | None = None,
    episode_paths: Sequence[str | Path] | None = None,
    reader_factory: ReaderFactory | None = None,
) -> V1SampleIndex:
    """Build a deterministic index directly from real episode HDF5 files."""

    if config is None:
        raise ValueError(
            "`config` is required; build_v1_sample_index has no implicit K or "
            "future-offset basis"
        )
    if episode_paths is None or isinstance(episode_paths, (str, Path)):
        raise ValueError("episode_paths must be a non-empty sequence of files")
    records: dict[Path, V1EpisodeRecord] = {}
    for raw_path in episode_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"V1 episode does not exist: {path}")
        if path in records:
            raise ValueError(f"duplicate V1 episode: {path}")
        records[path] = V1EpisodeRecord(episode_path=path)
    if not records:
        raise ValueError("episode_paths must not be empty")
    resolved_config = config
    episodes = tuple(records.values())
    episodes = tuple(sorted(episodes, key=lambda value: str(value.episode_path)))
    return _build_v1_sample_index_from_episode_records(
        config=resolved_config,
        episodes=episodes,
        reader_factory=reader_factory,
    )


def _build_v1_sample_index_from_episode_records(
    *,
    config: V1IndexConfig,
    episodes: Sequence[V1EpisodeRecord],
    reader_factory: ReaderFactory | None = None,
    episode_ordinals: Sequence[int] | None = None,
    progress_callback: IndexProgressCallback | None = None,
    progress_every_episodes: int = 100,
    include_source_manifest_bindings: bool = True,
) -> V1SampleIndex:
    """Build a compact, lazily materialized address index.

    Candidate callers continue to enter through :func:`build_v1_sample_index`.
    The Formal selector reaches this package-private core only after validating
    its frozen receipt/split chain.  Per-window Python ``SampleRef`` objects are
    intentionally not retained: fixed-width NumPy arrays plus an episode
    prefix sum reduce the full Formal index from multi-GiB object graphs to a
    few hundred MiB of compact addresses per rank.
    """

    resolved_config = config
    if not episodes:
        raise ValueError("episode record selector returned no episodes")
    if episode_ordinals is None:
        resolved_ordinals = tuple(range(len(episodes)))
    else:
        resolved_ordinals = tuple(episode_ordinals)
        if (
            len(resolved_ordinals) != len(episodes)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in resolved_ordinals
            )
            or len(set(resolved_ordinals)) != len(resolved_ordinals)
        ):
            raise ValueError(
                "episode_ordinals must be unique non-negative integers matching episodes"
            )
        if any(
            right != left + 1
            for left, right in zip(resolved_ordinals, resolved_ordinals[1:])
        ):
            raise ValueError("episode_ordinals must be a contiguous ordered range")
    if (
        isinstance(progress_every_episodes, bool)
        or not isinstance(progress_every_episodes, int)
        or progress_every_episodes <= 0
    ):
        raise ValueError("progress_every_episodes must be a positive integer")
    if not isinstance(include_source_manifest_bindings, bool):
        raise TypeError("include_source_manifest_bindings must be bool")
    factory = reader_factory or _default_reader_factory
    plans: list[_CompactEpisodeWindows] = []
    audit = {
        "episode_count": len(episodes),
        "candidate_source_frames": 0,
        "excluded_physical_horizon_unavailable": 0,
        "excluded_non_increasing_target_indices": 0,
        "excluded_action_horizon_unavailable": 0,
        "excluded_last_command_without_history": 0,
        "included_windows": 0,
    }

    total_episodes = len(episodes)
    for progress_ordinal, (episode_ordinal, episode) in enumerate(
        zip(
            resolved_ordinals,
            episodes,
            strict=True,
        )
    ):
        if progress_callback is not None and (
            progress_ordinal == 0
            or progress_ordinal % progress_every_episodes == 0
        ):
            progress_callback(
                progress_ordinal,
                total_episodes,
                int(audit["included_windows"]),
            )
        episode_audit = {name: 0 for name in V1_INDEX_AUDIT_KEYS}
        episode_audit["episode_count"] = 1
        source_indices: list[int] = []
        target_rows: list[tuple[int, ...]] = []
        action_rows: list[tuple[int, ...]] = []
        action_query_bases: list[float] = []
        dense_query_rows: list[tuple[float, ...]] = []
        reader = factory(episode.episode_path)
        try:
            if resolved_config.require_raw_v1 and reader.schema_version != RAW_V1_SCHEMA:
                raise ValueError(
                    f"expected {RAW_V1_SCHEMA}, got {reader.schema_version!r}: "
                    f"{episode.episode_path}"
                )
            available_cameras = set(reader.camera_names)
            missing_cameras = set(resolved_config.camera_names) - available_cameras
            if missing_cameras:
                raise ValueError(
                    f"episode lacks required cameras {sorted(missing_cameras)}: "
                    f"{episode.episode_path}"
                )
            if not reader.has_control_stream:
                raise ValueError(
                    f"episode has no complete control stream: {episode.episode_path}"
                )
            _validate_control_stream_contract(reader)
            metadata = reader.metadata
            frame_count = int(reader.num_frames)
            _validate_record_metadata(
                episode,
                metadata,
                frame_count=frame_count,
            )
            control_fields = reader.read_control_stream_fields(
                fields=("timestamp_before_sec", "timestamp_after_sec")
            )
            alignment_reader = getattr(
                reader,
                "read_index_alignment_fields",
                None,
            )
            alignment_fields: Mapping[str, np.ndarray] | None = None
            if callable(alignment_reader):
                raw_alignment_fields = alignment_reader()
                required_alignment_fields = {
                    "timestamp_sec",
                    "saved_frame_applied_start",
                    "saved_frame_applied_stop",
                }
                if (
                    not isinstance(raw_alignment_fields, Mapping)
                    or set(raw_alignment_fields) != required_alignment_fields
                ):
                    raise ValueError(
                        "read_index_alignment_fields returned invalid fields"
                    )
                alignment_fields = raw_alignment_fields
            control_timestamps = np.asarray(
                control_fields["timestamp_before_sec"], dtype=np.float64
            )
            frame_timestamps = _validate_episode_control_alignment(
                reader,
                frame_count=frame_count,
                timestamp_before_sec=control_timestamps,
                timestamp_after_sec=np.asarray(
                    control_fields["timestamp_after_sec"], dtype=np.float64
                ),
                tolerance_sec=resolved_config.action.timestamp_tolerance_sec,
                frame_timestamps_sec=(
                    None
                    if alignment_fields is None
                    else alignment_fields["timestamp_sec"]
                ),
                saved_frame_applied_start=(
                    None
                    if alignment_fields is None
                    else alignment_fields["saved_frame_applied_start"]
                ),
                saved_frame_applied_stop=(
                    None
                    if alignment_fields is None
                    else alignment_fields["saved_frame_applied_stop"]
                ),
            )
            episode_id = _resolved_episode_identity(episode, metadata)
            uses_physical_time = (
                resolved_config.future_offset_basis
                == FASTWAM_CLIP_FRAME_15HZ_OFFSET_BASIS
            )
            if uses_physical_time:
                source_stop = frame_count
            else:
                max_offset = resolved_config.future_offsets[-1]
                if frame_count <= max_offset:
                    continue
                source_stop = frame_count - max_offset
            audit["candidate_source_frames"] += source_stop
            episode_audit["candidate_source_frames"] += source_stop
            for source_index in range(source_stop):
                if uses_physical_time:
                    assert resolved_config.future_horizons_seconds is not None
                    try:
                        physical_targets = select_future_frames_by_physical_time(
                            frame_timestamps,
                            source_index=source_index,
                            offsets_sec=resolved_config.future_horizons_seconds,
                        )
                    except IndexError:
                        audit["excluded_physical_horizon_unavailable"] += 1
                        episode_audit[
                            "excluded_physical_horizon_unavailable"
                        ] += 1
                        continue
                    targets = tuple(
                        int(index) for index in physical_targets.target_indices
                    )
                else:
                    targets = tuple(
                        source_index + offset
                        for offset in resolved_config.future_offsets
                    )
                if any(
                    right <= left
                    for left, right in zip(targets, targets[1:])
                ):
                    audit["excluded_non_increasing_target_indices"] += 1
                    episode_audit[
                        "excluded_non_increasing_target_indices"
                    ] += 1
                    continue
                try:
                    selection = resolve_action_selection(
                        reader,
                        source_index,
                        resolved_config.action,
                        control_timestamps_before_sec=control_timestamps,
                        control_timestamps_validated=True,
                        source_timestamp_sec=float(frame_timestamps[source_index]),
                        source_control_span=(
                            None
                            if alignment_fields is None
                            else (
                                int(
                                    alignment_fields[
                                        "saved_frame_applied_start"
                                    ][source_index]
                                ),
                                int(
                                    alignment_fields[
                                        "saved_frame_applied_stop"
                                    ][source_index]
                                ),
                            )
                        ),
                    )
                except ActionHorizonUnavailable:
                    audit["excluded_action_horizon_unavailable"] += 1
                    episode_audit[
                        "excluded_action_horizon_unavailable"
                    ] += 1
                    continue
                if (
                    resolved_config.proprio_mode == "last_command"
                    and selection.source_control_start == 0
                ):
                    audit["excluded_last_command_without_history"] += 1
                    episode_audit[
                        "excluded_last_command_without_history"
                    ] += 1
                    continue
                source_indices.append(source_index)
                target_rows.append(targets)
                action_rows.append(selection.indices)
                action_query_bases.append(selection.query_times_sec[0])
                if resolved_config.action.mode == "dense250Hz":
                    dense_query_rows.append(selection.query_times_sec)
                audit["included_windows"] += 1
                episode_audit["included_windows"] += 1

            if source_indices:
                plans.append(
                    _CompactEpisodeWindows(
                        episode=episode,
                        episode_ordinal=episode_ordinal,
                        episode_id=episode_id,
                        source_indices=_readonly_unsigned_array(
                            source_indices,
                            name="source_indices",
                        ),
                        target_indices=_readonly_unsigned_array(
                            target_rows,
                            name="target_indices",
                        ),
                        action_indices=_readonly_unsigned_array(
                            action_rows,
                            name="action_indices",
                        ),
                        action_query_time_base_sec=_readonly_float64_array(
                            action_query_bases,
                            name="action_query_time_base_sec",
                        ),
                        dense_action_query_times_sec=(
                            _readonly_float64_array(
                                dense_query_rows,
                                name="dense_action_query_times_sec",
                            )
                            if dense_query_rows
                            else None
                        ),
                        index_audit=tuple(sorted(episode_audit.items())),
                    )
                )
        finally:
            reader.close()

    if progress_callback is not None:
        progress_callback(
            total_episodes,
            total_episodes,
            int(audit["included_windows"]),
        )
    index_audit = tuple(sorted(audit.items()))
    excluded = sum(
        value for name, value in index_audit if name.startswith("excluded_")
    )
    if audit["candidate_source_frames"] != audit["included_windows"] + excluded:
        raise RuntimeError("V1 index audit counters do not exactly cover candidates")
    if not plans:
        raise ValueError(
            "no legal V1 source frames remain after boundary checks; "
            f"index_audit={dict(index_audit)}"
        )
    refs = CompactSampleRefSequence(plans, config=resolved_config)
    if audit["included_windows"] != len(refs):
        raise RuntimeError("V1 index audit included count differs from compact index")
    return V1SampleIndex(
        episodes=tuple(episodes),
        sample_refs=refs,
        config=resolved_config,
        index_sha256=_compact_sequence_sha256(
            refs,
            index_audit=index_audit,
        ),
        source_manifest_bindings=(
            _source_manifest_bindings(episodes)
            if include_source_manifest_bindings
            else ()
        ),
        index_audit=index_audit,
    )


def _merge_v1_sample_index_shards(
    *,
    config: V1IndexConfig,
    episodes: Sequence[V1EpisodeRecord],
    shards: Sequence[V1SampleIndex],
) -> V1SampleIndex:
    """Merge contiguous independently built shards without changing identity."""

    resolved_episodes = tuple(episodes)
    if not resolved_episodes:
        raise ValueError("cannot merge V1 index shards without episodes")
    if not shards:
        raise ValueError("cannot merge an empty V1 index shard sequence")
    shard_episodes = tuple(
        episode for shard in shards for episode in shard.episodes
    )
    if shard_episodes != resolved_episodes:
        raise ValueError("V1 index shards do not exactly cover episodes in order")

    audit = {name: 0 for name in V1_INDEX_AUDIT_KEYS}
    plans: list[_CompactEpisodeWindows] = []
    for shard in shards:
        if shard.config != config:
            raise ValueError("V1 index shard config differs from merge config")
        shard_audit = dict(shard.index_audit)
        if set(shard_audit) != set(V1_INDEX_AUDIT_KEYS):
            raise ValueError("V1 index shard audit fields drifted")
        for name in V1_INDEX_AUDIT_KEYS:
            value = shard_audit[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V1 index shard audit {name} is invalid")
            audit[name] += value
        if not isinstance(shard.sample_refs, CompactSampleRefSequence):
            raise TypeError("V1 index shard is not compact")
        for plan in shard.sample_refs.plans:
            for array in (
                plan.source_indices,
                plan.target_indices,
                plan.action_indices,
                plan.action_query_time_base_sec,
                plan.dense_action_query_times_sec,
            ):
                if array is not None:
                    array.setflags(write=False)
        plans.extend(shard.sample_refs.plans)

    for plan in plans:
        if (
            plan.episode_ordinal >= len(resolved_episodes)
            or plan.episode != resolved_episodes[plan.episode_ordinal]
        ):
            raise ValueError("V1 index shard plan episode/ordinal drifted")
    if any(
        right.episode_ordinal <= left.episode_ordinal
        for left, right in zip(plans, plans[1:])
    ):
        raise ValueError("V1 index shard plans are not strictly ordered")

    index_audit = tuple(sorted(audit.items()))
    excluded = sum(
        value for name, value in index_audit if name.startswith("excluded_")
    )
    if audit["episode_count"] != len(resolved_episodes):
        raise RuntimeError("V1 index shard episode count differs from selector")
    if audit["candidate_source_frames"] != audit["included_windows"] + excluded:
        raise RuntimeError("V1 merged index audit does not exactly cover candidates")
    if not plans:
        raise ValueError("no legal V1 source frames remain after shard merge")
    refs = CompactSampleRefSequence(plans, config=config)
    if audit["included_windows"] != len(refs):
        raise RuntimeError("V1 merged included count differs from compact index")
    return V1SampleIndex(
        episodes=resolved_episodes,
        sample_refs=refs,
        config=config,
        index_sha256=_compact_sequence_sha256(refs, index_audit=index_audit),
        source_manifest_bindings=_source_manifest_bindings(resolved_episodes),
        index_audit=index_audit,
    )


__all__ = [
    "ActionSamplingConfig",
    "ActionHorizonUnavailable",
    "ActionSelection",
    "CompactSampleRefSequence",
    "FASTWAM_CLIP_FPS_HZ",
    "FASTWAM_K1_FUTURE_HORIZONS_SECONDS",
    "FASTWAM_K1_LOGICAL_FUTURE_OFFSETS",
    "FASTWAM_K4_HYBRID_FUTURE_HORIZONS_SECONDS",
    "FASTWAM_K4_HYBRID_LOGICAL_FUTURE_OFFSETS",
    "FASTWAM_FUTURE_HORIZONS_SECONDS",
    "FASTWAM_LOGICAL_FUTURE_OFFSETS",
    "PHYSICAL_CAMERA_NAMES",
    "RAW_V1_SCHEMA",
    "ReaderFactory",
    "V1_INDEX_AUDIT_KEYS",
    "V1_INDEX_AUDIT_SCHEMA_VERSION",
    "V1EpisodeRecord",
    "V1IndexConfig",
    "V1_RUNTIME_INDEX_BINDING_SCHEMA_VERSION",
    "V1_RUNTIME_INDEX_COMPOSITION_ALGORITHM",
    "V1SampleIndex",
    "build_v1_sample_index",
    "merge_v1_runtime_index_bindings",
    "project_v1_sample_index_future_contract",
    "resolve_action_selection",
    "slice_v1_sample_index_by_episode_ordinals",
    "select_v1_runtime_index_binding",
    "validate_v1_runtime_index_binding",
]
