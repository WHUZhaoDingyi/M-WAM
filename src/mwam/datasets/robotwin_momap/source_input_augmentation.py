"""Deterministic source-only Seg and Depth augmentation for RoboTwin MoMap.

The policies in this module never consume the process-global Python, NumPy,
or Torch RNG streams.  That keeps Action/MoMap timestep and Gaussian-noise
sampling unchanged when source-condition robustness training is enabled.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .source_conditions import (
    ROLE_TRACK_SEGMENTATION_MAX_ROLE_ID,
    validate_role_track_segmentation_ids,
)


SEG_QUALITY_MIX_MANIFEST_SCHEMA = "fastwam.robotwin-seg-quality-mix.v1"
SEG_QUALITY_MIX_MATH_SCHEMA = "fastwam.robotwin-seg-quality-mix-math.v1"
DEPTH_NOISE_MATH_SCHEMA = "fastwam.robotwin-source-depth-noise-math.v1"
DEPTH_EDGE_NOISE_MATH_SCHEMA = "fastwam.robotwin-source-depth-noise-math.v2"
SEG_CACHE_SCHEMA = "fastwam.robotwin-grounded-sam2-reacquire-seg-cache.v3"
HEAD_ROLE_TRACK_CANDIDATE_SCHEMA = (
    "fastwam.robotwin-head-role-track-candidate.v1"
)
HEAD_ROLE_TRACK_CANDIDATE_MATH_SCHEMA = (
    "fastwam.robotwin-head-role-track-candidate-math.v1"
)
HEAD_CAMERA_NAME = "head_camera"


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _canonical_digest(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _unit_interval(*parts: object) -> float:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(1 << 64)


def _seed64(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class SegMixDecision:
    selected_source: str
    configured_p_gt: float
    reason: str
    predicted_by_camera: Mapping[str, np.ndarray] | None


@dataclass(frozen=True)
class HeadRoleTrackCandidate:
    """One fail-closed predicted Seg candidate for the physical head camera.

    This deliberately does not make a GT/Pred mixing decision.  It exposes the
    cache candidate plus enough provenance for a later policy to decide whether
    to use it.  An unavailable candidate has both categorical maps set to
    ``None`` and carries a stable reason in ``provenance``.
    """

    available: bool
    role_id: np.ndarray | None
    track_id: np.ndarray | None
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class _SegPolicyEntry:
    p_gt: float
    cache_path: Path
    quality_bucket: str


class _PIDH5Pool:
    """Worker-local bounded HDF5 pool; inherited handles are discarded."""

    def __init__(self, max_open_files: int) -> None:
        if max_open_files <= 0:
            raise ValueError("max_open_files must be positive")
        self.max_open_files = int(max_open_files)
        self._pid = os.getpid()
        self._handles: OrderedDict[Path, h5py.File] = OrderedDict()

    def _reset_after_fork(self) -> None:
        current = os.getpid()
        if current == self._pid:
            return
        self.close()
        self._pid = current

    def get(self, path: Path) -> h5py.File:
        self._reset_after_fork()
        resolved = path.expanduser().resolve()
        existing = self._handles.pop(resolved, None)
        if existing is not None:
            self._handles[resolved] = existing
            return existing
        handle = h5py.File(resolved, "r", swmr=True)
        self._handles[resolved] = handle
        while len(self._handles) > self.max_open_files:
            _, evicted = self._handles.popitem(last=False)
            evicted.close()
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        return {
            "max_open_files": self.max_open_files,
            "_pid": os.getpid(),
            "_handles": OrderedDict(),
        }


class PooledQualitySegmentationMix:
    """Choose one GT/Pred Seg source for all three cameras of one window."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        seed: int,
        max_open_files: int = 8,
    ) -> None:
        path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("Seg quality mix manifest must be a JSON object")
        if payload.get("schema_version") != SEG_QUALITY_MIX_MANIFEST_SCHEMA:
            raise ValueError("unsupported Seg quality mix manifest schema")
        rows = payload.get("episodes")
        if not isinstance(rows, Mapping) or not rows:
            raise ValueError("Seg quality mix manifest episodes must be non-empty")
        entries: dict[str, _SegPolicyEntry] = {}
        math_rows: list[tuple[str, float, str]] = []
        for episode_id, raw in rows.items():
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError("Seg policy episode_id must be non-empty text")
            if not isinstance(raw, Mapping):
                raise TypeError(f"Seg policy row for {episode_id} must be an object")
            p_gt = raw.get("p_gt")
            if isinstance(p_gt, bool) or not isinstance(p_gt, (int, float)):
                raise TypeError(f"Seg policy p_gt for {episode_id} must be numeric")
            p_gt = float(p_gt)
            if not np.isfinite(p_gt) or not 0.0 <= p_gt <= 1.0:
                raise ValueError(f"Seg policy p_gt for {episode_id} is invalid")
            cache_path = raw.get("cache_path")
            if not isinstance(cache_path, str) or not cache_path.strip():
                raise ValueError(f"Seg policy cache_path for {episode_id} is invalid")
            bucket = raw.get("quality_bucket")
            if not isinstance(bucket, str) or not bucket:
                raise ValueError(
                    f"Seg policy quality_bucket for {episode_id} is invalid"
                )
            entries[episode_id] = _SegPolicyEntry(
                p_gt=p_gt,
                cache_path=Path(cache_path),
                quality_bucket=bucket,
            )
            math_rows.append((episode_id, p_gt, bucket))
        declared_digest = payload.get("policy_content_digest")
        actual_digest = _canonical_digest(sorted(math_rows))
        if declared_digest != actual_digest:
            raise ValueError(
                "Seg quality mix manifest policy_content_digest mismatch"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("Seg quality mix seed must be an integer")
        self.manifest_path = path
        self.seed = int(seed)
        self.policy_content_digest = actual_digest
        self._entries = entries
        self._pool = _PIDH5Pool(max_open_files)


    @staticmethod
    def _validate_cache(
        handle: h5py.File,
        *,
        episode_id: str,
        camera_names: Sequence[str],
    ) -> None:
        if _decode(handle.attrs.get("schema_version", "")) != SEG_CACHE_SCHEMA:
            raise ValueError("predicted Seg cache schema mismatch")
        if _decode(handle.attrs.get("status", "")) != "PASS":
            raise ValueError("predicted Seg cache is not PASS")
        if _decode(handle.attrs.get("episode_id", "")) != episode_id:
            raise ValueError("predicted Seg cache episode_id mismatch")
        if not bool(handle.attrs.get("segmentation_only", False)):
            raise ValueError("predicted Seg cache is not segmentation-only")
        if bool(handle.attrs.get("contains_predicted_depth", True)):
            raise ValueError("predicted Seg cache unexpectedly contains depth")
        actual_cameras = tuple(_decode(value) for value in handle["camera_name"][:])
        if actual_cameras != tuple(camera_names):
            raise ValueError("predicted Seg cache camera order mismatch")
        if "pred/track_id" not in handle or "source_index" not in handle:
            raise ValueError("predicted Seg cache lacks source_index/track_id")

    @staticmethod
    def _read_role_track_row(
        handle: h5py.File,
        *,
        episode_id: str,
        source_index: int,
        camera_names: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read and validate both V3 categorical maps for exactly one frame."""

        PooledQualitySegmentationMix._validate_cache(
            handle,
            episode_id=episode_id,
            camera_names=camera_names,
        )
        if "pred/role_id" not in handle:
            raise ValueError("predicted Seg cache lacks role_id")
        source_indices = np.asarray(handle["source_index"][:], dtype=np.int64)
        if source_indices.ndim != 1 or source_indices.size <= 0:
            raise ValueError("predicted Seg cache source_index must be non-empty [N]")
        if source_indices.size > 1 and not bool(
            np.all(source_indices[1:] > source_indices[:-1])
        ):
            raise ValueError("predicted Seg cache source_index must be strictly sorted")
        expected = (source_indices.size, len(camera_names), 240, 320)
        track_dataset = handle["pred/track_id"]
        role_dataset = handle["pred/role_id"]
        if tuple(track_dataset.shape) != expected or track_dataset.dtype != np.dtype(
            np.uint16
        ):
            raise ValueError(
                "predicted track_id shape/dtype="
                f"{tuple(track_dataset.shape)}/{track_dataset.dtype}, "
                f"expected uint16{expected}"
            )
        if tuple(role_dataset.shape) != expected or role_dataset.dtype != np.dtype(
            np.uint8
        ):
            raise ValueError(
                "predicted role_id shape/dtype="
                f"{tuple(role_dataset.shape)}/{role_dataset.dtype}, "
                f"expected uint8{expected}"
            )
        row = int(np.searchsorted(source_indices, int(source_index)))
        if row >= source_indices.size or int(source_indices[row]) != int(
            source_index
        ):
            raise KeyError(f"source_index={source_index} is absent from cache")
        # Read both datasets before semantic validation.  Do not permit callers
        # to turn a role+track candidate into a track-only candidate silently.
        role = np.asarray(role_dataset[row])
        track = np.asarray(track_dataset[row])
        expected_row = (len(camera_names), 240, 320)
        if role.shape != expected_row or track.shape != expected_row:
            raise ValueError(
                "predicted role_id/track_id row shape mismatch after HDF5 read"
            )
        role, track = validate_role_track_segmentation_ids(role, track)
        return role, track

    @staticmethod
    def _unavailable_head_candidate(
        *,
        reason: str,
        episode_id: str,
        source_index: int,
        entry: _SegPolicyEntry | None,
    ) -> HeadRoleTrackCandidate:
        return HeadRoleTrackCandidate(
            available=False,
            role_id=None,
            track_id=None,
            provenance={
                "schema_version": HEAD_ROLE_TRACK_CANDIDATE_SCHEMA,
                "candidate_source": "predicted_cache_v3",
                "available": False,
                "reason": reason,
                "episode_id": str(episode_id),
                "source_index": int(source_index),
                "camera_name": HEAD_CAMERA_NAME,
                "cache_path": None if entry is None else str(entry.cache_path),
                "quality_bucket": None if entry is None else entry.quality_bucket,
                "configured_p_gt": None if entry is None else entry.p_gt,
            },
        )

    def load_head_role_track_candidate(
        self,
        *,
        episode_id: str,
        source_index: int,
        camera_names: Sequence[str],
    ) -> HeadRoleTrackCandidate:
        """Load one validated Head role+track candidate without sampling p_gt.

        The cache is still validated against the complete configured three-view
        order, but only the head maps are copied into the return value.  Any
        cache problem returns ``available=False`` rather than exposing a
        partially valid role/track pair.
        """

        cameras = tuple(str(camera_name) for camera_name in camera_names)
        if HEAD_CAMERA_NAME not in cameras:
            return self._unavailable_head_candidate(
                reason="missing_head_camera",
                episode_id=episode_id,
                source_index=source_index,
                entry=None,
            )
        entry = self._entries.get(episode_id)
        if entry is None:
            return self._unavailable_head_candidate(
                reason="missing_quality_entry",
                episode_id=episode_id,
                source_index=source_index,
                entry=None,
            )
        try:
            handle = self._pool.get(entry.cache_path)
            role, track = self._read_role_track_row(
                handle,
                episode_id=episode_id,
                source_index=source_index,
                camera_names=cameras,
            )
            head_index = cameras.index(HEAD_CAMERA_NAME)
            head_role = np.array(role[head_index], copy=True)
            head_track = np.array(track[head_index], copy=True)
            validate_role_track_segmentation_ids(head_role, head_track)
        except (OSError, KeyError, TypeError, ValueError):
            return self._unavailable_head_candidate(
                reason="predicted_cache_fallback",
                episode_id=episode_id,
                source_index=source_index,
                entry=entry,
            )
        return HeadRoleTrackCandidate(
            available=True,
            role_id=head_role,
            track_id=head_track,
            provenance={
                "schema_version": HEAD_ROLE_TRACK_CANDIDATE_SCHEMA,
                "candidate_source": "predicted_cache_v3",
                "available": True,
                "reason": "available",
                "episode_id": str(episode_id),
                "source_index": int(source_index),
                "camera_name": HEAD_CAMERA_NAME,
                "cache_path": str(entry.cache_path),
                "cache_schema_version": SEG_CACHE_SCHEMA,
                "quality_bucket": entry.quality_bucket,
                "configured_p_gt": entry.p_gt,
                "role_track_role_max": ROLE_TRACK_SEGMENTATION_MAX_ROLE_ID,
            },
        )

    def decide(
        self,
        *,
        episode_id: str,
        source_index: int,
        camera_names: Sequence[str],
        epoch: int = 0,
    ) -> SegMixDecision:
        entry = self._entries.get(episode_id)
        if entry is None:
            return SegMixDecision("gt", 1.0, "missing_quality_entry", None)
        draw = _unit_interval(
            "seg-source-v1",
            self.seed,
            int(epoch),
            episode_id,
            int(source_index),
        )
        if draw < entry.p_gt:
            return SegMixDecision("gt", entry.p_gt, "quality_probability", None)
        try:
            handle = self._pool.get(entry.cache_path)
            self._validate_cache(
                handle,
                episode_id=episode_id,
                camera_names=camera_names,
            )
            source_indices = np.asarray(handle["source_index"][:], dtype=np.int64)
            row = int(np.searchsorted(source_indices, int(source_index)))
            if row >= source_indices.size or int(source_indices[row]) != int(
                source_index
            ):
                raise KeyError(f"source_index={source_index} is absent from cache")
            track_id = np.asarray(handle["pred/track_id"][row])
            expected = (len(camera_names), 240, 320)
            if track_id.shape != expected or not np.issubdtype(
                track_id.dtype, np.integer
            ):
                raise ValueError(
                    "predicted track_id shape/dtype="
                    f"{track_id.shape}/{track_id.dtype}, "
                    f"expected integer{expected}"
                )
            predicted = {
                str(camera_name): np.array(track_id[index], copy=True)
                for index, camera_name in enumerate(camera_names)
            }
        except (OSError, KeyError, TypeError, ValueError):
            return SegMixDecision("gt", entry.p_gt, "predicted_cache_fallback", None)
        return SegMixDecision("pred", entry.p_gt, "quality_probability", predicted)

    def close(self) -> None:
        self._pool.close()


@dataclass(frozen=True)
class _DepthNoiseLevel:
    name: str
    probability: float
    additive_std_m: float
    relative_std: float
    camera_scale_abs_max: float
    edge_pixel_fraction_min: float
    edge_pixel_fraction_max: float


@dataclass(frozen=True)
class _DepthEdgeCorruption:
    depth_jump_abs_min_m: float
    depth_jump_relative_min: float
    segment_length_min_px: int
    segment_length_max_px: int
    foreground_to_background_probability: float


class SourceDepthNoiseAugmentation:
    """Apply deterministic sensor-like noise only to model-visible GT depth."""

    def __init__(self, config: Mapping[str, Any], *, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("Depth noise seed must be an integer")
        raw_levels = config.get("levels")
        if not isinstance(raw_levels, Sequence) or isinstance(raw_levels, (str, bytes)):
            raise TypeError("depth_noise.levels must be a sequence")
        levels: list[_DepthNoiseLevel] = []
        for raw in raw_levels:
            if not isinstance(raw, Mapping):
                raise TypeError("each depth noise level must be a mapping")
            level = _DepthNoiseLevel(
                name=str(raw.get("name", "")),
                probability=float(raw.get("probability")),
                additive_std_m=float(raw.get("additive_std_m")),
                relative_std=float(raw.get("relative_std")),
                camera_scale_abs_max=float(raw.get("camera_scale_abs_max")),
                edge_pixel_fraction_min=float(
                    raw.get("edge_pixel_fraction_min", 0.0)
                ),
                edge_pixel_fraction_max=float(
                    raw.get("edge_pixel_fraction_max", 0.0)
                ),
            )
            if not level.name:
                raise ValueError("depth noise level name must be non-empty")
            values = (
                level.probability,
                level.additive_std_m,
                level.relative_std,
                level.camera_scale_abs_max,
                level.edge_pixel_fraction_min,
                level.edge_pixel_fraction_max,
            )
            if not all(np.isfinite(value) and value >= 0.0 for value in values):
                raise ValueError(
                    "depth noise level values must be finite and non-negative"
                )
            if (
                level.edge_pixel_fraction_min > 1.0
                or level.edge_pixel_fraction_max > 1.0
                or level.edge_pixel_fraction_min
                > level.edge_pixel_fraction_max
            ):
                raise ValueError(
                    "depth noise edge pixel fractions must satisfy "
                    "0 <= min <= max <= 1"
                )
            levels.append(level)
        if not levels or abs(sum(level.probability for level in levels) - 1.0) > 1e-9:
            raise ValueError("depth noise probabilities must sum to one")
        if len({level.name for level in levels}) != len(levels):
            raise ValueError("depth noise level names must be unique")
        raw_edge = config.get("edge_corruption")
        edge_corruption: _DepthEdgeCorruption | None = None
        if raw_edge is not None:
            if not isinstance(raw_edge, Mapping):
                raise TypeError("depth_noise.edge_corruption must be a mapping")
            required = {
                "enabled",
                "depth_jump_abs_min_m",
                "depth_jump_relative_min",
                "segment_length_min_px",
                "segment_length_max_px",
                "foreground_to_background_probability",
            }
            if set(raw_edge) != required:
                raise ValueError(
                    "depth_noise.edge_corruption fields drifted: "
                    f"missing={sorted(required - set(raw_edge))}, "
                    f"unexpected={sorted(set(raw_edge) - required)}"
                )
            if raw_edge["enabled"] is not True:
                raise ValueError(
                    "depth_noise.edge_corruption.enabled must be true when "
                    "the block is present"
                )
            integer_fields = (
                raw_edge["segment_length_min_px"],
                raw_edge["segment_length_max_px"],
            )
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in integer_fields
            ):
                raise TypeError(
                    "depth edge segment lengths must be integers, not bool"
                )
            edge_corruption = _DepthEdgeCorruption(
                depth_jump_abs_min_m=float(raw_edge["depth_jump_abs_min_m"]),
                depth_jump_relative_min=float(
                    raw_edge["depth_jump_relative_min"]
                ),
                segment_length_min_px=int(raw_edge["segment_length_min_px"]),
                segment_length_max_px=int(raw_edge["segment_length_max_px"]),
                foreground_to_background_probability=float(
                    raw_edge["foreground_to_background_probability"]
                ),
            )
            edge_values = (
                edge_corruption.depth_jump_abs_min_m,
                edge_corruption.depth_jump_relative_min,
                edge_corruption.foreground_to_background_probability,
            )
            if not all(np.isfinite(value) for value in edge_values):
                raise ValueError("depth edge corruption values must be finite")
            if (
                edge_corruption.depth_jump_abs_min_m < 0.0
                or edge_corruption.depth_jump_relative_min < 0.0
                or (
                    edge_corruption.depth_jump_abs_min_m == 0.0
                    and edge_corruption.depth_jump_relative_min == 0.0
                )
            ):
                raise ValueError(
                    "at least one non-negative depth jump threshold must be "
                    "strictly positive"
                )
            if (
                edge_corruption.segment_length_min_px <= 0
                or edge_corruption.segment_length_max_px
                < edge_corruption.segment_length_min_px
            ):
                raise ValueError(
                    "depth edge segment lengths must satisfy 1 <= min <= max"
                )
            if not (
                0.0
                <= edge_corruption.foreground_to_background_probability
                <= 1.0
            ):
                raise ValueError(
                    "depth edge foreground_to_background_probability must "
                    "lie in [0, 1]"
                )
        elif any(level.edge_pixel_fraction_max > 0.0 for level in levels):
            raise ValueError(
                "depth edge pixel fractions require edge_corruption settings"
            )
        self.seed = int(seed)
        self.levels = tuple(levels)
        self.edge_corruption = edge_corruption

    @property
    def math_schema_version(self) -> str:
        return (
            DEPTH_NOISE_MATH_SCHEMA
            if self.edge_corruption is None
            else DEPTH_EDGE_NOISE_MATH_SCHEMA
        )


    def _selected_level(
        self,
        *,
        episode_id: str,
        source_index: int,
        epoch: int,
    ) -> _DepthNoiseLevel:
        draw = _unit_interval(
            "depth-level-v1",
            self.seed,
            int(epoch),
            episode_id,
            int(source_index),
        )
        cumulative = 0.0
        for level in self.levels:
            cumulative += level.probability
            if draw < cumulative:
                return level
        return self.levels[-1]

    @staticmethod
    def _edge_candidate_maps(
        depth: np.ndarray,
        valid: np.ndarray,
        edge: _DepthEdgeCorruption,
    ) -> tuple[np.ndarray, np.ndarray]:
        left = depth[:, :-1]
        right = depth[:, 1:]
        horizontal_threshold = np.maximum(
            edge.depth_jump_abs_min_m,
            edge.depth_jump_relative_min * np.minimum(left, right),
        )
        horizontal = (
            valid[:, :-1]
            & valid[:, 1:]
            & (np.abs(left - right) >= horizontal_threshold)
        )

        upper = depth[:-1, :]
        lower = depth[1:, :]
        vertical_threshold = np.maximum(
            edge.depth_jump_abs_min_m,
            edge.depth_jump_relative_min * np.minimum(upper, lower),
        )
        vertical = (
            valid[:-1, :]
            & valid[1:, :]
            & (np.abs(upper - lower) >= vertical_threshold)
        )
        return horizontal, vertical

    @staticmethod
    def _contiguous_segment(
        candidate: np.ndarray,
        selected: np.ndarray,
        *,
        y: int,
        x: int,
        tangent_axis: int,
        requested_length: int,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        """Select one unbroken run containing ``(y, x)`` along an edge."""

        def available(test_y: int, test_x: int) -> bool:
            return bool(
                0 <= test_y < candidate.shape[0]
                and 0 <= test_x < candidate.shape[1]
                and candidate[test_y, test_x]
                and not selected[test_y, test_x]
            )

        low = y if tangent_axis == 0 else x
        high = low
        while True:
            next_value = low - 1
            next_y, next_x = (
                (next_value, x) if tangent_axis == 0 else (y, next_value)
            )
            if not available(next_y, next_x):
                break
            low = next_value
        while True:
            next_value = high + 1
            next_y, next_x = (
                (next_value, x) if tangent_axis == 0 else (y, next_value)
            )
            if not available(next_y, next_x):
                break
            high = next_value

        length = min(requested_length, high - low + 1)
        seed_value = y if tangent_axis == 0 else x
        start_min = max(low, seed_value - length + 1)
        start_max = min(seed_value, high - length + 1)
        start = int(rng.integers(start_min, start_max + 1))
        if tangent_axis == 0:
            return [(value, x) for value in range(start, start + length)]
        return [(y, value) for value in range(start, start + length)]

    @staticmethod
    def _eligible_edge_pixel_count(
        horizontal: np.ndarray,
        vertical: np.ndarray,
    ) -> int:
        """Count unique pixels that are endpoints of any eligible edge pair."""

        eligible = np.zeros(
            (horizontal.shape[0], vertical.shape[1]),
            dtype=bool,
        )
        horizontal_y, horizontal_x = np.nonzero(horizontal)
        eligible[horizontal_y, horizontal_x] = True
        eligible[horizontal_y, horizontal_x + 1] = True
        vertical_y, vertical_x = np.nonzero(vertical)
        eligible[vertical_y, vertical_x] = True
        eligible[vertical_y + 1, vertical_x] = True
        return int(eligible.sum())

    def _apply_edge_corruption(
        self,
        *,
        raw_depth: np.ndarray,
        gaussian_depth: np.ndarray,
        valid: np.ndarray,
        augmented: np.ndarray,
        level: _DepthNoiseLevel,
        rng: np.random.Generator,
    ) -> None:
        edge = self.edge_corruption
        if edge is None or level.edge_pixel_fraction_max == 0.0:
            return
        if raw_depth.ndim != 2:
            raise ValueError(
                "depth edge corruption requires a two-dimensional depth map"
            )
        horizontal, vertical = self._edge_candidate_maps(raw_depth, valid, edge)
        candidate_count = int(horizontal.sum()) + int(vertical.sum())
        if candidate_count == 0:
            return
        eligible_edge_pixel_count = self._eligible_edge_pixel_count(
            horizontal,
            vertical,
        )
        fraction = float(
            rng.uniform(
                level.edge_pixel_fraction_min,
                level.edge_pixel_fraction_max,
            )
        )
        target_count = min(
            int(np.floor(fraction * eligible_edge_pixel_count)),
            candidate_count,
        )
        if target_count <= 0:
            return

        horizontal_y, horizontal_x = np.nonzero(horizontal)
        vertical_y, vertical_x = np.nonzero(vertical)
        candidates = [
            (0, int(y), int(x))
            for y, x in zip(horizontal_y, horizontal_x, strict=True)
        ] + [
            (1, int(y), int(x))
            for y, x in zip(vertical_y, vertical_x, strict=True)
        ]
        selected_horizontal = np.zeros_like(horizontal, dtype=bool)
        selected_vertical = np.zeros_like(vertical, dtype=bool)
        modified_target = np.zeros_like(valid, dtype=bool)
        modified_count = 0

        for candidate_index in rng.permutation(len(candidates)):
            orientation, y, x = candidates[int(candidate_index)]
            candidate_map = horizontal if orientation == 0 else vertical
            selected_map = (
                selected_horizontal if orientation == 0 else selected_vertical
            )
            if selected_map[y, x]:
                continue
            requested_length = int(
                rng.integers(
                    edge.segment_length_min_px,
                    edge.segment_length_max_px + 1,
                )
            )
            segment = self._contiguous_segment(
                candidate_map,
                selected_map,
                y=y,
                x=x,
                tangent_axis=0 if orientation == 0 else 1,
                requested_length=requested_length,
                rng=rng,
            )
            for pair_y, pair_x in segment:
                selected_map[pair_y, pair_x] = True

            if float(rng.random()) < edge.foreground_to_background_probability:
                foreground_to_background = True
            else:
                foreground_to_background = False

            for pair_y, pair_x in segment:
                if orientation == 0:
                    endpoint0 = (pair_y, pair_x)
                    endpoint1 = (pair_y, pair_x + 1)
                else:
                    endpoint0 = (pair_y, pair_x)
                    endpoint1 = (pair_y + 1, pair_x)
                if raw_depth[endpoint0] <= raw_depth[endpoint1]:
                    foreground, background = endpoint0, endpoint1
                else:
                    foreground, background = endpoint1, endpoint0

                if foreground_to_background:
                    target, source = foreground, background
                else:
                    target, source = background, foreground
                if modified_target[target]:
                    continue
                augmented[target] = gaussian_depth[source]
                modified_target[target] = True
                modified_count += 1
                if modified_count >= target_count:
                    return

    def augment_frame(
        self,
        frame: Mapping[str, Any],
        *,
        episode_id: str,
        source_index: int,
        camera_names: Sequence[str],
        epoch: int = 0,
    ) -> Mapping[str, Any]:
        level = self._selected_level(
            episode_id=episode_id,
            source_index=source_index,
            epoch=epoch,
        )
        if (
            level.additive_std_m == 0.0
            and level.relative_std == 0.0
            and level.camera_scale_abs_max == 0.0
            and level.edge_pixel_fraction_max == 0.0
        ):
            return frame
        private = frame.get("private_geometry")
        if not isinstance(private, Mapping):
            raise ValueError("source frame lacks private_geometry")
        result = dict(frame)
        result_private = dict(private)
        for camera_name in camera_names:
            geometry = private.get(camera_name)
            if not isinstance(geometry, Mapping):
                raise ValueError(f"source frame lacks geometry for {camera_name}")
            depth = np.asarray(geometry.get("depth_m"), dtype=np.float32)
            valid = np.isfinite(depth) & (depth > 0.0)
            rng = np.random.default_rng(
                _seed64(
                    "depth-values-v1",
                    self.seed,
                    int(epoch),
                    episode_id,
                    int(source_index),
                    camera_name,
                )
            )
            scale = rng.uniform(
                -level.camera_scale_abs_max,
                level.camera_scale_abs_max,
            )
            normal = rng.standard_normal(depth.shape, dtype=np.float32)
            augmented = np.array(depth, copy=True)
            noise_std = level.additive_std_m + level.relative_std * depth[valid]
            augmented[valid] = (
                (1.0 + scale) * depth[valid] + noise_std * normal[valid]
            )
            augmented[valid] = np.maximum(augmented[valid], 1.0e-6)
            if self.edge_corruption is not None:
                gaussian_depth = np.array(augmented, copy=True)
                edge_rng = np.random.default_rng(
                    _seed64(
                        "depth-edge-values-v1",
                        self.seed,
                        int(epoch),
                        episode_id,
                        int(source_index),
                        camera_name,
                    )
                )
                self._apply_edge_corruption(
                    raw_depth=depth,
                    gaussian_depth=gaussian_depth,
                    valid=valid,
                    augmented=augmented,
                    level=level,
                    rng=edge_rng,
                )
            result_geometry = dict(geometry)
            result_geometry["depth_m"] = augmented
            result_private[camera_name] = result_geometry
        result["private_geometry"] = result_private
        return result


class SourceInputAugmentation:
    """Configured pair of independent Seg-source and Depth-noise policies."""

    def __init__(
        self,
        *,
        seg_quality_mix_manifest: str | Path | None,
        seg_quality_mix_seed: int,
        depth_noise: Mapping[str, Any] | None,
        depth_noise_seed: int,
        max_open_seg_cache_files: int,
    ) -> None:
        self.segmentation = (
            None
            if seg_quality_mix_manifest is None
            else PooledQualitySegmentationMix(
                seg_quality_mix_manifest,
                seed=seg_quality_mix_seed,
                max_open_files=max_open_seg_cache_files,
            )
        )
        self.depth = (
            None
            if depth_noise is None
            else SourceDepthNoiseAugmentation(depth_noise, seed=depth_noise_seed)
        )

    @property
    def enabled(self) -> bool:
        return self.segmentation is not None or self.depth is not None


    def close(self) -> None:
        if self.segmentation is not None:
            self.segmentation.close()


__all__ = [
    "DEPTH_EDGE_NOISE_MATH_SCHEMA",
    "DEPTH_NOISE_MATH_SCHEMA",
    "HEAD_CAMERA_NAME",
    "HEAD_ROLE_TRACK_CANDIDATE_SCHEMA",
    "HEAD_ROLE_TRACK_CANDIDATE_MATH_SCHEMA",
    "HeadRoleTrackCandidate",
    "PooledQualitySegmentationMix",
    "SEG_QUALITY_MIX_MANIFEST_SCHEMA",
    "SEG_QUALITY_MIX_MATH_SCHEMA",
    "SegMixDecision",
    "SourceDepthNoiseAugmentation",
    "SourceInputAugmentation",
]
