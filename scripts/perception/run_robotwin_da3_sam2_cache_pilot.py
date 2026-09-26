#!/usr/bin/env python3
"""Build a causal predicted-perception cache for the frozen RoboTwin pilot.

Each process owns deterministic logical shards, loads one DA3Metric model, one
GroundingDINO-tiny model, and one SAM2.1 Base+ predictor on its CUDA device, and
writes create-only per-episode HDF5 files.  Only RGB, instruction, and camera
calibration are prediction inputs.  Simulator depth and actor segmentation are
optional post-freeze scoring signals and are never copied into the cache.

The SAM2 adapter is intentionally pinned to the checked-out project commit.  It
feeds a lazy HDF5-backed frame sequence directly into the predictor state; it
does not export temporary JPEGs and it never propagates backwards in time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np



THIS_FILE = Path(__file__).resolve()
FASTWAM_ROOT = THIS_FILE.parents[2]
WORK_ROOT = FASTWAM_ROOT.parent
PERCEPTION_MODEL_ROOT = (
    FASTWAM_ROOT / "artifacts/perception_models/da3_gdino_sam2_pilot_v1"
)
DEFAULT_SAM2_REPO = WORK_ROOT / "perception_third_party/sam2"
DEFAULT_DA3_REPO = WORK_ROOT / "perception_third_party/Depth-Anything-3"
DEFAULT_DA3_MODEL = PERCEPTION_MODEL_ROOT / "depth-anything_DA3METRIC-LARGE"
DEFAULT_GROUNDING_MODEL = (
    PERCEPTION_MODEL_ROOT / "IDEA-Research_grounding-dino-tiny"
)
DEFAULT_SAM2_CHECKPOINT = (
    PERCEPTION_MODEL_ROOT / "sam2_checkpoints/sam2.1_hiera_base_plus.pt"
)
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEFAULT_SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"

MANIFEST_SCHEMA = "fastwam.robotwin-predicted-perception-pilot.v1"
FULL_TRAIN_MANIFEST_SCHEMA = (
    "fastwam.robotwin-predicted-perception-train-split.v1"
)
SUPPORTED_MANIFEST_SCHEMAS = {
    MANIFEST_SCHEMA,
    FULL_TRAIN_MANIFEST_SCHEMA,
}
CACHE_SCHEMA = "fastwam.robotwin-predicted-perception-cache.v1"
SCORE_SCHEMA = "fastwam.robotwin-predicted-perception-score.v1"
SUMMARY_SCHEMA = "fastwam.robotwin-predicted-perception-rank-summary.v1"
try:
    LOGICAL_SHARD_COUNT = int(
        os.environ.get("FASTWAM_PERCEPTION_LOGICAL_SHARD_COUNT", "32")
    )
except ValueError as exc:
    raise RuntimeError(
        "FASTWAM_PERCEPTION_LOGICAL_SHARD_COUNT must be 32 or 64"
    ) from exc
if LOGICAL_SHARD_COUNT not in (32, 64):
    raise RuntimeError(
        "FASTWAM_PERCEPTION_LOGICAL_SHARD_COUNT must be 32 or 64"
    )
CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
HEIGHT = 240
WIDTH = 320

# status_flags[S,3] bit assignments.  Bits describe predicted artifacts only.
FLAG_DEPTH_OK = np.uint16(1 << 0)
FLAG_GROUNDING_SEEDED = np.uint16(1 << 1)
FLAG_SAM2_TRACKED = np.uint16(1 << 2)
FLAG_GROUNDING_MISS = np.uint16(1 << 3)
FLAG_EMPTY_TRACK_MASK = np.uint16(1 << 4)
FLAG_NONFINITE_DEPTH = np.uint16(1 << 5)

ROLE_ORDER = (
    "manipulated_target",
    "destination_or_reference",
    "tool",
    "robot",
)
ROLE_CODE = {role: index + 1 for index, role in enumerate(ROLE_ORDER)}

# Make direct execution and importlib-based CPU tests resolve the adjacent
# project-owned role schema identically.
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

from robotwin_grounding_role_schema import (  # noqa: E402
    SCHEMA_VERSION as ROLE_SCHEMA_VERSION,
    build_grounding_role_prompts,
    validate_role_schema,
)


class CachePilotError(RuntimeError):
    """The frozen pilot contract or runtime inference contract was violated."""


@dataclass(frozen=True)
class Detection:
    camera_index: int
    role: str
    phrase: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    track_id: int


@dataclass
class ModelBundle:
    torch: Any
    device: Any
    da3: Any
    grounding_processor: Any
    grounding_model: Any
    sam2_predictor: Any
    sam2_commit: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve(strict=True)
    encoded = resolved.read_bytes()
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CachePilotError(f"cannot parse manifest {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CachePilotError("manifest root must be an object")
    if payload.get("schema_version") not in SUPPORTED_MANIFEST_SCHEMAS:
        raise CachePilotError(
            f"unsupported manifest schema: {payload.get('schema_version')!r}"
        )
    if payload.get("status") != "PASS":
        raise CachePilotError("manifest is not PASS")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise CachePilotError("manifest episodes must be a non-empty list")
    expected_count = payload.get("aggregate", {}).get("selected_episode_count")
    if expected_count != len(episodes):
        raise CachePilotError(
            f"manifest episode count mismatch: aggregate={expected_count}, "
            f"records={len(episodes)}"
        )
    identities: set[str] = set()
    for index, row in enumerate(episodes):
        if not isinstance(row, Mapping):
            raise CachePilotError(f"episode {index} must be an object")
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise CachePilotError(f"episode {index} has invalid episode_id")
        if episode_id in identities:
            raise CachePilotError(f"duplicate episode_id: {episode_id}")
        identities.add(episode_id)
        source = row.get("source")
        if not isinstance(source, Mapping):
            raise CachePilotError(f"episode {episode_id} lacks source contract")
        causal = source.get("causal_processing_frame_count")
        legal = source.get("legal_source_count")
        start = source.get("legal_source_index_start")
        stop = source.get("legal_source_index_stop_inclusive")
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in (
            causal, legal, start, stop
        )):
            raise CachePilotError(f"episode {episode_id} has invalid source integers")
        if start != 1 or stop - start + 1 != legal or causal != stop + 1:
            raise CachePilotError(
                f"episode {episode_id} source range is not causal frame0..N"
            )
        task_name = row.get("task_name")
        instruction = row.get("instruction")
        try:
            build_grounding_role_prompts(str(task_name), str(instruction))
        except Exception as exc:
            raise CachePilotError(
                f"episode {episode_id} has no valid role prompt: {exc}"
            ) from exc
    return payload, _sha256_bytes(encoded)


def build_balanced_shards(
    episodes: Sequence[Mapping[str, Any]],
    shard_count: int = LOGICAL_SHARD_COUNT,
) -> tuple[list[list[dict[str, Any]]], list[int]]:
    """Greedily balance causal frames into deterministic logical shards."""

    if isinstance(shard_count, bool) or shard_count <= 0:
        raise ValueError("shard_count must be positive")
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    loads = [0 for _ in range(shard_count)]
    ordered = sorted(
        (dict(row) for row in episodes),
        key=lambda row: (
            -int(row["source"]["causal_processing_frame_count"]),
            str(row["episode_id"]),
        ),
    )
    for row in ordered:
        shard_id = min(
            range(shard_count),
            key=lambda index: (loads[index], len(buckets[index]), index),
        )
        buckets[shard_id].append(row)
        loads[shard_id] += int(
            row["source"]["causal_processing_frame_count"]
        )
    for bucket in buckets:
        bucket.sort(key=lambda row: str(row["episode_id"]))
    return buckets, loads


def assigned_shard_ids(rank: int, world_size: int) -> tuple[int, ...]:
    if isinstance(rank, bool) or isinstance(world_size, bool):
        raise ValueError("rank/world_size must be integers")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
    return tuple(
        shard for shard in range(LOGICAL_SHARD_COUNT)
        if shard % world_size == rank
    )


def jobs_for_rank(
    buckets: Sequence[Sequence[Mapping[str, Any]]],
    *,
    rank: int,
    world_size: int,
) -> list[tuple[int, dict[str, Any]]]:
    jobs: list[tuple[int, dict[str, Any]]] = []
    for shard_id in assigned_shard_ids(rank, world_size):
        jobs.extend((shard_id, dict(row)) for row in buckets[shard_id])
    return jobs


def _episode_stem(episode_id: str) -> str:
    return hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:24]


def _episode_output_path(output_dir: Path, episode_id: str) -> Path:
    return output_dir / "episodes" / f"{_episode_stem(episode_id)}.h5"


def _score_output_path(output_dir: Path, episode_id: str) -> Path:
    return output_dir / "scores" / f"{_episode_stem(episode_id)}.json"


def _write_json_atomic(path: Path, value: Any, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(
        f".{path.name}.partial.pid{os.getpid()}.{time.time_ns()}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(partial, flags, 0o644)
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8") + b"\n"
        with os.fdopen(fd, "wb", closefd=True) as stream:
            fd = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and path.exists():
            raise FileExistsError(path)
        os.replace(partial, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if partial.exists():
            partial.unlink()


def _append_progress(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json_bytes(dict(value)) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = (float(item) for item in a)
    bx0, by0, bx1, by1 = (float(item) for item in b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def _deduplicate_detections(
    rows: Sequence[dict[str, Any]],
    *,
    iou_threshold: float,
    max_per_role: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Role-aware score-sorted NMS followed by conservative cross-role NMS."""

    kept_by_role: list[dict[str, Any]] = []
    for role in ROLE_ORDER:
        candidates = sorted(
            (dict(row) for row in rows if row.get("role") == role),
            key=lambda row: (
                -float(row["score"]),
                tuple(float(item) for item in row["box_xyxy"]),
                str(row.get("phrase", "")),
            ),
        )
        selected: list[dict[str, Any]] = []
        limit = max(0, int(max_per_role.get(role, 0)))
        for row in candidates:
            if any(
                _box_iou(row["box_xyxy"], prior["box_xyxy"])
                >= iou_threshold
                for prior in selected
            ):
                continue
            selected.append(row)
            if limit and len(selected) >= limit:
                break
        kept_by_role.extend(selected)

    final: list[dict[str, Any]] = []
    for role in ROLE_ORDER:
        role_rows = sorted(
            (row for row in kept_by_role if row["role"] == role),
            key=lambda row: (
                -float(row["score"]),
                float(row["box_xyxy"][0]),
                float(row["box_xyxy"][1]),
            ),
        )
        for row in role_rows:
            if any(
                _box_iou(row["box_xyxy"], prior["box_xyxy"])
                >= max(0.90, iou_threshold)
                for prior in final
            ):
                continue
            final.append(row)
    return final


def _squeeze_independent_depth(value: Any, expected_batch: int) -> np.ndarray:
    array = np.asarray(value)
    # Axis 0 is the independent-image batch and must survive when B=1.
    while array.ndim > 3:
        singleton = next(
            (axis for axis in range(1, array.ndim) if array.shape[axis] == 1),
            None,
        )
        if singleton is None:
            break
        array = np.squeeze(array, axis=singleton)
    if array.ndim == 2 and expected_batch == 1:
        array = array[None]
    if array.ndim != 3 or array.shape[0] != expected_batch:
        raise CachePilotError(
            f"unexpected independent depth shape {array.shape}; batch={expected_batch}"
        )
    return array.astype(np.float32, copy=False)


def metric_depth_from_canonical(
    raw_depth: np.ndarray,
    camera_k: np.ndarray,
    *,
    original_hw: tuple[int, int],
    processed_hw: tuple[int, int],
) -> np.ndarray:
    """Convert DA3Metric canonical depth with processed-image focal / 300."""

    raw = np.asarray(raw_depth, dtype=np.float32)
    matrix = np.asarray(camera_k, dtype=np.float32)
    if raw.ndim != 2 or matrix.shape != (3, 3):
        raise ValueError(f"invalid depth/K shapes: {raw.shape}, {matrix.shape}")
    original_h, original_w = original_hw
    processed_h, processed_w = processed_hw
    if min(original_h, original_w, processed_h, processed_w) <= 0:
        raise ValueError("image sizes must be positive")
    fx_processed = float(matrix[0, 0]) * processed_w / original_w
    fy_processed = float(matrix[1, 1]) * processed_h / original_h
    focal_processed = 0.5 * (fx_processed + fy_processed)
    if not np.isfinite(focal_processed) or focal_processed <= 0:
        raise ValueError(f"invalid processed focal: {focal_processed}")
    return raw * np.float32(focal_processed / 300.0)


def _resize_float_map(value: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    target_h, target_w = target_hw
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (target_h, target_w):
        return array
    image = Image.fromarray(array, mode="F")
    resized = image.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _frame_value(dataset: Any, frame_index: int) -> np.ndarray:
    if dataset.ndim == 2:
        return np.asarray(dataset[...])
    return np.asarray(dataset[frame_index])


def _normalize_label(value: str) -> str:
    return " ".join(
        value.lower().replace("-", " ").replace("_", " ").split()
    ).strip(" .")


def _roles_for_grounded_phrase(
    label: str, prompt_contract: Mapping[str, Any]
) -> tuple[str, ...]:
    normalized = _normalize_label(label)
    matches: list[str] = []
    phrase_to_roles = prompt_contract.get("phrase_to_roles", {})
    if not isinstance(phrase_to_roles, Mapping):
        return ()
    for phrase, roles in phrase_to_roles.items():
        candidate = _normalize_label(str(phrase))
        if not candidate:
            continue
        if (
            normalized == candidate
            or normalized in candidate
            or candidate in normalized
        ):
            for role in roles:
                if role in ROLE_CODE and role not in matches:
                    matches.append(role)
    return tuple(matches)


def _pure_self_test() -> dict[str, Any]:
    role_result = validate_role_schema()
    synthetic: list[dict[str, Any]] = []
    for index in range(97):
        frames = 5 + ((index * 37) % 113)
        synthetic.append(
            {
                "episode_id": f"episode-{index:04d}",
                "source": {"causal_processing_frame_count": frames},
            }
        )
    buckets_a, loads_a = build_balanced_shards(synthetic)
    buckets_b, loads_b = build_balanced_shards(list(reversed(synthetic)))
    identities_a = [[row["episode_id"] for row in bucket] for bucket in buckets_a]
    identities_b = [[row["episode_id"] for row in bucket] for bucket in buckets_b]
    assert identities_a == identities_b
    assert sorted(item for bucket in identities_a for item in bucket) == sorted(
        row["episode_id"] for row in synthetic
    )
    assert set().union(*(set(assigned_shard_ids(rank, 4)) for rank in range(4))) == set(
        range(LOGICAL_SHARD_COUNT)
    )
    assert sum(len(bucket) for bucket in buckets_a) == len(synthetic)
    raw = np.full((378, 504), 0.5, dtype=np.float32)
    k = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
    meters = metric_depth_from_canonical(
        raw, k, original_hw=(240, 320), processed_hw=(378, 504)
    )
    assert np.allclose(meters, 0.7875, atol=1e-6)
    nms = _deduplicate_detections(
        [
            {"role": "manipulated_target", "score": 0.9, "box_xyxy": [0, 0, 10, 10]},
            {"role": "manipulated_target", "score": 0.8, "box_xyxy": [1, 1, 10, 10]},
            {"role": "robot", "score": 0.7, "box_xyxy": [20, 20, 30, 30]},
        ],
        iou_threshold=0.7,
        max_per_role={role: 4 for role in ROLE_ORDER},
    )
    assert len(nms) == 2
    return {
        "status": "PASS",
        "logical_shard_count": LOGICAL_SHARD_COUNT,
        "synthetic_episode_count": len(synthetic),
        "shard_frame_min": min(loads_a),
        "shard_frame_max": max(loads_a),
        "role_schema": role_result,
        "metric_scaling_test_m": float(meters[0, 0]),
    }

class HDF5FrameSequence:
    """Small-LRU lazy SAM2 frame sequence backed by one episode HDF5."""
    def __init__(self, *, episode_path: Path, camera_name: str, frame_count: int, image_size: int, torch_module: Any, compute_device: Any, cache_size: int = 2) -> None:
        self.episode_path, self.camera_name = episode_path, camera_name
        self.frame_count, self.image_size = int(frame_count), int(image_size)
        self.torch, self.compute_device = torch_module, compute_device
        self.video_height, self.video_width = HEIGHT, WIDTH
        self.cache_size, self._h5 = max(1, int(cache_size)), None
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._mean = self.torch.tensor((0.485,0.456,0.406), dtype=self.torch.float32)[:,None,None]
        self._std = self.torch.tensor((0.229,0.224,0.225), dtype=self.torch.float32)[:,None,None]
    def __len__(self) -> int: return self.frame_count
    def _dataset(self) -> Any:
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.episode_path, "r", swmr=True)
        return self._h5[f"observations/{self.camera_name}/rgb"]
    def __getitem__(self, index: int) -> Any:
        from PIL import Image
        if index < 0: index += self.frame_count
        if index < 0 or index >= self.frame_count: raise IndexError(index)
        cached = self._cache.pop(index, None)
        if cached is not None:
            self._cache[index] = cached
            return cached
        rgb = np.asarray(self._dataset()[index])
        if rgb.shape != (HEIGHT,WIDTH,3) or rgb.dtype != np.uint8:
            raise CachePilotError(f"invalid RGB {self.camera_name}[{index}]: {rgb.shape}/{rgb.dtype}")
        resized = np.asarray(Image.fromarray(rgb, mode="RGB").resize((self.image_size,self.image_size)), dtype=np.float32)
        tensor = self.torch.from_numpy(resized).permute(2,0,1) / 255.0
        tensor = (tensor - self._mean) / self._std
        self._cache[index] = tensor
        while len(self._cache) > self.cache_size: self._cache.popitem(last=False)
        return tensor
    def close(self) -> None:
        self._cache.clear()
        if self._h5 is not None:
            self._h5.close(); self._h5 = None

def _git_head(repo: Path) -> str:
    return subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"], check=True, capture_output=True, text=True).stdout.strip()

def _load_models(args: argparse.Namespace) -> ModelBundle:
    sam2_repo = args.sam2_repo.expanduser().resolve(strict=True)
    da3_repo = args.da3_repo.expanduser().resolve(strict=True)
    sam2_commit = _git_head(sam2_repo)
    if sam2_commit != args.sam2_expected_commit:
        raise CachePilotError(f"SAM2 commit mismatch: expected={args.sam2_expected_commit}, actual={sam2_commit}")
    for required in (args.da3_model,args.grounding_model,args.sam2_checkpoint): required.expanduser().resolve(strict=True)
    for entry in (str(da3_repo / "src"), str(sam2_repo)):
        if entry not in sys.path: sys.path.insert(0,entry)
    import torch
    if not torch.cuda.is_available(): raise CachePilotError("CUDA is required")
    device = torch.device(args.device); torch.cuda.set_device(device)
    from depth_anything_3.api import DepthAnything3
    da3 = DepthAnything3.from_pretrained(str(args.da3_model.resolve()), local_files_only=True).to(device).eval(); da3.device=device
    import transformers.models.grounding_dino.modeling_grounding_dino as mgd
    from transformers import AutoConfig, AutoModelForZeroShotObjectDetection, AutoProcessor
    mgd.is_torch_cuda_available = lambda: False
    gd_path = str(args.grounding_model.resolve())
    gd_config = AutoConfig.from_pretrained(gd_path, local_files_only=True); gd_config.disable_custom_kernels=True
    gd_processor = AutoProcessor.from_pretrained(gd_path, local_files_only=True)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_path, config=gd_config, local_files_only=True, use_safetensors=True).to(device).eval()
    from sam2.build_sam import build_sam2_video_predictor
    sam2 = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint.resolve()), device=str(device), mode="eval")
    return ModelBundle(torch,device,da3,gd_processor,gd_model,sam2,sam2_commit)

def _init_sam2_state_from_hdf5_sequence(predictor: Any, sequence: HDF5FrameSequence, *, torch_module: Any) -> dict[str,Any]:
    """Pinned adapter matching SAM2 commit 2b90b9f init_state fields."""
    device=predictor.device
    state={"images":sequence,"num_frames":len(sequence),"offload_video_to_cpu":True,"offload_state_to_cpu":True,"video_height":HEIGHT,"video_width":WIDTH,"device":device,"storage_device":torch_module.device("cpu"),"point_inputs_per_obj":{},"mask_inputs_per_obj":{},"cached_features":{},"constants":{},"obj_id_to_idx":OrderedDict(),"obj_idx_to_id":OrderedDict(),"obj_ids":[],"output_dict_per_obj":{},"temp_output_dict_per_obj":{},"frames_tracked_per_obj":{}}
    predictor._get_image_feature(state,frame_idx=0,batch_size=1)
    return state

def _validate_episode_hdf5(row: Mapping[str,Any]) -> tuple[Path,int,list[int]]:
    import h5py
    path=Path(str(row["episode_path"])).resolve(strict=True); frames=int(row["expected_frame_count"]); source=row["source"]
    indices=list(range(int(source["legal_source_index_start"]),int(source["legal_source_index_stop_inclusive"])+1))
    with h5py.File(path,"r",swmr=True) as h:
        for camera in CAMERA_NAMES:
            rgb=h[f"observations/{camera}/rgb"]
            if rgb.shape != (frames,HEIGHT,WIDTH,3) or rgb.dtype != np.dtype("uint8"): raise CachePilotError(f"bad RGB contract: {camera} {rgb.shape}/{rgb.dtype}")
            if f"private_geometry/{camera}/K" not in h: raise CachePilotError(f"missing K: {camera}")
    if indices[-1] >= frames: raise CachePilotError("legal source exceeds frame count")
    return path,frames,indices

def _read_frame0_images(path: Path) -> list[np.ndarray]:
    import h5py
    with h5py.File(path,"r",swmr=True) as h: return [np.asarray(h[f"observations/{c}/rgb"][0]) for c in CAMERA_NAMES]

def _ground_frame0(bundle: ModelBundle, *, frame0_images: Sequence[np.ndarray], prompt: Mapping[str,Any], args: argparse.Namespace) -> tuple[list[list[Detection]],dict[str,int]]:
    from PIL import Image
    images=[Image.fromarray(np.asarray(x),mode="RGB") for x in frame0_images]; caption=str(prompt["grounding_dino_text"])
    inputs=bundle.grounding_processor(images=images,text=[caption]*3,padding=True,return_tensors="pt")
    inputs={k:(v.to(bundle.device) if hasattr(v,"to") else v) for k,v in inputs.items()}
    with bundle.torch.inference_mode(): outputs=bundle.grounding_model(**inputs)
    results=bundle.grounding_processor.post_process_grounded_object_detection(outputs,inputs["input_ids"],box_threshold=args.box_threshold,text_threshold=args.text_threshold,target_sizes=[(HEIGHT,WIDTH)]*3)
    limits={r:(args.max_boxes_per_role if r=="robot" else min(args.max_boxes_per_role,max(1,int(prompt["expected_count"].get(r,1))))) for r in ROLE_ORDER}
    result_rows=[]; stats={"raw_boxes":0,"unassigned_boxes":0,"kept_boxes":0}
    for camera_index,result in enumerate(results):
        boxes=result["boxes"].detach().cpu().numpy(); scores=result["scores"].detach().cpu().numpy(); labels=result.get("text_labels",result.get("labels",[])); raw=[]
        for box,score,label in zip(boxes,scores,labels):
            stats["raw_boxes"]+=1; roles=_roles_for_grounded_phrase(str(label),prompt)
            if not roles: stats["unassigned_boxes"]+=1; continue
            for role in roles: raw.append({"role":role,"phrase":str(label),"score":float(score),"box_xyxy":[float(v) for v in box]})
        kept=_deduplicate_detections(raw,iou_threshold=args.dedup_iou_threshold,max_per_role=limits); occurrence={r:0 for r in ROLE_ORDER}; rows=[]
        for item in kept:
            role=str(item["role"]); occurrence[role]+=1; tid=ROLE_CODE[role]*1000+occurrence[role]
            rows.append(Detection(camera_index,role,str(item["phrase"]),float(item["score"]),tuple(float(v) for v in item["box_xyxy"]),tid))
        stats["kept_boxes"]+=len(rows); result_rows.append(rows)
    return result_rows,stats

def _run_da3_batches(bundle: ModelBundle, *, source_h5: Any, legal_indices: Sequence[int], depth_ds: Any, confidence_ds: Any, flags_ds: Any, args: argparse.Namespace) -> dict[str,float|int]:
    items=[(row,cam,src) for row,src in enumerate(legal_indices) for cam in range(3)]; finite_count=value_count=bad_frames=0; total=0.0; minimum=float("inf"); maximum=float("-inf")
    for offset in range(0,len(items),args.da3_batch_size):
        batch=items[offset:offset+args.da3_batch_size]; images=[np.asarray(source_h5[f"observations/{CAMERA_NAMES[cam]}/rgb"][src]) for _,cam,src in batch]
        processed,_,_=bundle.da3.input_processor(images,process_res=args.da3_process_res,process_res_method="upper_bound_resize",num_workers=1,sequential=True,print_progress=False)
        ph,pw=processed.shape[-2:]; independent=processed[:,None].to(bundle.device,non_blocking=True).float()
        with bundle.torch.inference_mode(): raw=bundle.da3.forward(independent, export_feat_layers=[])
        depths=_squeeze_independent_depth(raw["depth"].detach().cpu().numpy(),len(batch)); conf_value=raw.get("depth_conf"); confs=np.ones_like(depths) if conf_value is None else _squeeze_independent_depth(conf_value.detach().cpu().numpy(),len(batch))
        for local,(row,cam,src) in enumerate(batch):
            K=_frame_value(source_h5[f"private_geometry/{CAMERA_NAMES[cam]}/K"],src); meters=_resize_float_map(metric_depth_from_canonical(depths[local],K,original_hw=(HEIGHT,WIDTH),processed_hw=(ph,pw)),(HEIGHT,WIDTH)); conf=_resize_float_map(confs[local],(HEIGHT,WIDTH)); finite=np.isfinite(meters)&(meters>0)
            if not bool(finite.all()): bad_frames+=1; flags_ds[row,cam] |= FLAG_NONFINITE_DEPTH
            safe=np.where(finite,meters,0).astype(np.float32); safe_conf=np.where(finite&np.isfinite(conf),conf,0).astype(np.float32); depth_ds[row,cam]=safe.astype(np.float16); confidence_ds[row,cam]=safe_conf.astype(np.float16); flags_ds[row,cam] |= FLAG_DEPTH_OK
            values=safe[finite]; finite_count+=int(values.size); value_count+=int(meters.size)
            if values.size: total+=float(values.astype(np.float64).sum()); minimum=min(minimum,float(values.min())); maximum=max(maximum,float(values.max()))
        del independent,raw
    return {"pixel_count":value_count,"finite_positive_pixel_count":finite_count,"finite_positive_ratio":finite_count/max(1,value_count),"mean_m":total/max(1,finite_count),"min_m":0.0 if not finite_count else minimum,"max_m":0.0 if not finite_count else maximum,"frames_with_nonfinite_or_nonpositive":bad_frames}

def _track_camera(bundle: ModelBundle, *, episode_path: Path, causal_frame_count: int, legal_indices: Sequence[int], camera_index: int, detections: Sequence[Detection], track_ds: Any, flags_ds: Any, tracks_group: Any, args: argparse.Namespace) -> dict[str,Any]:
    import h5py
    camera=CAMERA_NAMES[camera_index]; group=tracks_group.create_group(camera); count=len(detections)
    group.create_dataset("track_id",data=np.asarray([d.track_id for d in detections],dtype=np.uint16))
    group.create_dataset("role_code",data=np.asarray([ROLE_CODE[d.role] for d in detections],dtype=np.uint8))
    group.create_dataset("seed_score",data=np.asarray([d.score for d in detections],dtype=np.float32))
    group.create_dataset("seed_box_xyxy",data=np.asarray([d.box_xyxy for d in detections],dtype=np.float32).reshape(count,4))
    string_dtype=h5py.string_dtype("utf-8"); group.create_dataset("role",data=np.asarray([d.role for d in detections],dtype=object),dtype=string_dtype); group.create_dataset("grounded_phrase",data=np.asarray([d.phrase for d in detections],dtype=object),dtype=string_dtype)
    score_ds=group.create_dataset("frame_track_score",shape=(len(legal_indices),count),dtype=np.float16,chunks=(min(32,max(1,len(legal_indices))),max(1,count)) if count else None)
    if count==0:
        flags_ds[:,camera_index] |= FLAG_GROUNDING_MISS
        return {"camera":camera,"track_count":0,"tracked_frame_count":0,"empty_frame_count":len(legal_indices)}
    flags_ds[:,camera_index] |= FLAG_GROUNDING_SEEDED
    sequence=HDF5FrameSequence(episode_path=episode_path,camera_name=camera,frame_count=causal_frame_count,image_size=int(bundle.sam2_predictor.image_size),torch_module=bundle.torch,compute_device=bundle.device)
    state=None; seen=set(); empty_frames=0; row_lookup={frame:row for row,frame in enumerate(legal_indices)}
    try:
        state=_init_sam2_state_from_hdf5_sequence(bundle.sam2_predictor,sequence,torch_module=bundle.torch)
        for detection in detections:
            bundle.sam2_predictor.add_new_points_or_box(state,frame_idx=0,obj_id=detection.track_id,box=np.asarray(detection.box_xyxy,dtype=np.float32),clear_old_points=True)
        iterator=bundle.sam2_predictor.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=causal_frame_count,reverse=False)
        for frame_index,obj_ids,mask_logits in iterator:
            if frame_index not in row_lookup: continue
            row=row_lookup[frame_index]; ids=np.asarray([int(x) for x in obj_ids],dtype=np.uint16); logits=mask_logits.detach().float().cpu().numpy()
            if logits.ndim==4 and logits.shape[1]==1: logits=logits[:,0]
            if logits.shape != (len(ids),HEIGHT,WIDTH): raise CachePilotError(f"unexpected SAM2 mask shape {logits.shape} for {camera}")
            best=np.argmax(logits,axis=0); best_logit=np.max(logits,axis=0); valid=best_logit>args.sam2_mask_threshold; track_map=np.zeros((HEIGHT,WIDTH),dtype=np.uint16); track_map[valid]=ids[best[valid]]; track_ds[row,camera_index]=track_map
            if not bool(valid.any()): empty_frames+=1; flags_ds[row,camera_index] |= FLAG_EMPTY_TRACK_MASK
            id_to_col={d.track_id:i for i,d in enumerate(detections)}
            for object_index,track_id in enumerate(ids.tolist()):
                mask=logits[object_index]>args.sam2_mask_threshold
                if bool(mask.any()): score=float((1.0/(1.0+np.exp(-np.clip(logits[object_index][mask],-30,30)))).mean()); score_ds[row,id_to_col[track_id]]=np.float16(score)
            flags_ds[row,camera_index] |= FLAG_SAM2_TRACKED; seen.add(frame_index)
        missing=sorted(set(legal_indices)-seen)
        if missing: raise CachePilotError(f"SAM2 causal propagation omitted {len(missing)} frames for {camera}; first={missing[:5]}")
    finally:
        if state is not None:
            bundle.sam2_predictor.reset_state(state)
        sequence.close()
    return {"camera":camera,"track_count":count,"tracked_frame_count":len(seen),"empty_frame_count":empty_frames}

def _validate_existing_cache(path: Path, *, episode_id: str, manifest_file_sha256: str) -> bool:
    if not path.exists(): return False
    import h5py
    with h5py.File(path,"r",swmr=True) as h:
        valid=h.attrs.get("schema_version")==CACHE_SCHEMA and h.attrs.get("status")=="PASS" and h.attrs.get("episode_id")==episode_id and h.attrs.get("manifest_file_sha256")==manifest_file_sha256
    if not valid: raise CachePilotError(f"existing cache is not valid/identical: {path}")
    return True

def _run_episode(bundle: ModelBundle, *, row: Mapping[str,Any], shard_id: int, output_dir: Path, manifest_path: Path, manifest_file_sha256: str, manifest_semantic_sha256: str, args: argparse.Namespace) -> dict[str,Any]:
    import h5py
    started=time.monotonic(); episode_id=str(row["episode_id"]); source_path,_,legal_indices=_validate_episode_hdf5(row); causal_frames=legal_indices[-1]+1
    prompt=build_grounding_role_prompts(str(row["task_name"]),str(row["instruction"])); detections,ground_stats=_ground_frame0(bundle,frame0_images=_read_frame0_images(source_path),prompt=prompt,args=args)
    final=_episode_output_path(output_dir,episode_id); final.parent.mkdir(parents=True,exist_ok=True); partial=final.with_name(f".{final.name}.partial.rank{args.rank:03d}.pid{os.getpid()}")
    if partial.exists(): raise CachePilotError(f"partial already exists: {partial}")
    camera_summaries=[]
    with h5py.File(partial,"x",libver="latest") as out, h5py.File(source_path,"r",swmr=True) as source:
        out.attrs.update({"schema_version":CACHE_SCHEMA,"status":"WRITING","episode_id":episode_id,"task_name":str(row["task_name"]),"variant":str(row["variant"]),"manifest_path":str(manifest_path),"manifest_file_sha256":manifest_file_sha256,"manifest_sha256":manifest_semantic_sha256,"role_schema_version":ROLE_SCHEMA_VERSION,"sam2_commit":bundle.sam2_commit,"logical_shard_id":shard_id,"rank":args.rank,"created_at_utc":_utc_now(),"gt_depth_or_seg_written":False,"sam2_reverse":False})
        out.create_dataset("source_index",data=np.asarray(legal_indices,dtype=np.uint16)); out.create_dataset("camera_name",data=np.asarray(CAMERA_NAMES,dtype=object),dtype=h5py.string_dtype("utf-8")); out.create_dataset("grounding_prompt_contract_json",data=json.dumps(prompt,ensure_ascii=False,sort_keys=True),dtype=h5py.string_dtype("utf-8"))
        pred=out.create_group("pred"); chunks=(1,1,HEIGHT,WIDTH); depth_ds=pred.create_dataset("depth_m",shape=(len(legal_indices),3,HEIGHT,WIDTH),dtype=np.float16,chunks=chunks,compression="lzf",shuffle=True); conf_ds=pred.create_dataset("depth_confidence",shape=depth_ds.shape,dtype=np.float16,chunks=chunks,compression="lzf",shuffle=True); track_ds=pred.create_dataset("track_id",shape=depth_ds.shape,dtype=np.uint16,chunks=chunks,compression="lzf",shuffle=True); flags_ds=out.create_dataset("status_flags",shape=(len(legal_indices),3),dtype=np.uint16,chunks=(min(128,len(legal_indices)),3)); tracks=out.create_group("tracks")
        depth_summary=_run_da3_batches(bundle,source_h5=source,legal_indices=legal_indices,depth_ds=depth_ds,confidence_ds=conf_ds,flags_ds=flags_ds,args=args)
        for camera_index,camera_detections in enumerate(detections): camera_summaries.append(_track_camera(bundle,episode_path=source_path,causal_frame_count=causal_frames,legal_indices=legal_indices,camera_index=camera_index,detections=camera_detections,track_ds=track_ds,flags_ds=flags_ds,tracks_group=tracks,args=args))
        out.attrs["status"]="PASS"; out.attrs["completed_at_utc"]=_utc_now(); out.attrs["elapsed_seconds"]=time.monotonic()-started; out.flush()
    fd=os.open(partial,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    if final.exists(): raise FileExistsError(final)
    os.rename(partial,final); directory_fd=os.open(final.parent,os.O_RDONLY)
    try: os.fsync(directory_fd)
    finally: os.close(directory_fd)
    return {"episode_id":episode_id,"cache_path":str(final),"cache_size_bytes":final.stat().st_size,"legal_source_count":len(legal_indices),"causal_processing_frame_count":causal_frames,"grounding":ground_stats,"depth":depth_summary,"tracking":camera_summaries,"elapsed_seconds":time.monotonic()-started}

def _parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--da3-repo",type=Path,default=DEFAULT_DA3_REPO)
    parser.add_argument("--da3-model",type=Path,default=DEFAULT_DA3_MODEL)
    parser.add_argument("--grounding-model",type=Path,default=DEFAULT_GROUNDING_MODEL)
    parser.add_argument("--sam2-repo",type=Path,default=DEFAULT_SAM2_REPO)
    parser.add_argument("--sam2-config",default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint",type=Path,default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--sam2-expected-commit",default=DEFAULT_SAM2_COMMIT)
    parser.add_argument("--rank",type=int,default=0)
    parser.add_argument("--world-size",type=int,default=1)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--da3-batch-size",type=int,default=12)
    parser.add_argument("--da3-process-res",type=int,default=504)
    parser.add_argument("--box-threshold",type=float,default=0.25)
    parser.add_argument("--text-threshold",type=float,default=0.20)
    parser.add_argument("--dedup-iou-threshold",type=float,default=0.70)
    parser.add_argument("--sam2-mask-threshold",type=float,default=0.0)
    parser.add_argument("--max-boxes-per-role",type=int,default=4)
    parser.add_argument("--max-episodes",type=int,default=0,help="0 means all assigned episodes")
    parser.add_argument("--fail-fast",action="store_true")
    parser.add_argument("--self-test",action="store_true",help="CPU-only pure-function tests; never loads models")
    return parser

def _validate_args(args: argparse.Namespace) -> None:
    if args.self_test: return
    if args.manifest is None or args.output_dir is None: raise CachePilotError("--manifest and --output-dir are required")
    assigned_shard_ids(args.rank,args.world_size)
    if args.da3_batch_size<=0 or args.da3_process_res<=0 or args.max_boxes_per_role<=0: raise CachePilotError("batch/process-res/max-boxes must be positive")
    if args.max_episodes<0: raise CachePilotError("--max-episodes must be >=0")
    for name in ("box_threshold","text_threshold","dedup_iou_threshold"):
        value=float(getattr(args,name))
        if not 0.0<=value<=1.0: raise CachePilotError(f"--{name.replace('_','-')} must be in [0,1]")

def main() -> int:
    args=_parser().parse_args(); _validate_args(args)
    if args.self_test:
        print(json.dumps(_pure_self_test(),ensure_ascii=False,indent=2,sort_keys=True)); return 0
    manifest_path=args.manifest.expanduser().resolve(strict=True); payload,manifest_file_sha=_load_manifest(manifest_path); buckets,loads=build_balanced_shards(payload["episodes"]); jobs=jobs_for_rank(buckets,rank=args.rank,world_size=args.world_size)
    if args.max_episodes: jobs=jobs[:args.max_episodes]
    output_dir=args.output_dir.expanduser().resolve(); output_dir.mkdir(parents=True,exist_ok=True); progress_path=output_dir/"progress"/f"rank_{args.rank:03d}.jsonl"; summary_path=output_dir/"summaries"/f"rank_{args.rank:03d}.json"
    _append_progress(progress_path,{"timestamp_utc":_utc_now(),"event":"rank_start","rank":args.rank,"world_size":args.world_size,"assigned_shards":list(assigned_shard_ids(args.rank,args.world_size)),"selected_job_count":len(jobs),"manifest_file_sha256":manifest_file_sha})
    pending=[]; skipped=0
    for shard_id,row in jobs:
        path=_episode_output_path(output_dir,str(row["episode_id"]))
        if _validate_existing_cache(path,episode_id=str(row["episode_id"]),manifest_file_sha256=manifest_file_sha):
            skipped+=1; _append_progress(progress_path,{"timestamp_utc":_utc_now(),"event":"episode_skip_complete","rank":args.rank,"logical_shard_id":shard_id,"episode_id":row["episode_id"],"cache_path":str(path)})
        else: pending.append((shard_id,row))
    bundle=None; completed=[]; failures=[]; rank_started=time.monotonic()
    if pending: bundle=_load_models(args)
    for shard_id,row in pending:
        episode_id=str(row["episode_id"]); _append_progress(progress_path,{"timestamp_utc":_utc_now(),"event":"episode_start","rank":args.rank,"logical_shard_id":shard_id,"episode_id":episode_id})
        try:
            result=_run_episode(bundle,row=row,shard_id=shard_id,output_dir=output_dir,manifest_path=manifest_path,manifest_file_sha256=manifest_file_sha,manifest_semantic_sha256=str(payload.get("manifest_sha256","")),args=args); completed.append(result)
            _append_progress(progress_path,{"timestamp_utc":_utc_now(),"event":"episode_complete","rank":args.rank,"logical_shard_id":shard_id,**result})
        except Exception as exc:
            failure={"episode_id":episode_id,"logical_shard_id":shard_id,"error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()}; failures.append(failure); _append_progress(progress_path,{"timestamp_utc":_utc_now(),"event":"episode_failed","rank":args.rank,**failure})
            if args.fail_fast: break
    finite=sum(int(r["depth"]["finite_positive_pixel_count"]) for r in completed); pixels=sum(int(r["depth"]["pixel_count"]) for r in completed); weighted_depth=sum(float(r["depth"]["mean_m"])*int(r["depth"]["finite_positive_pixel_count"]) for r in completed)
    summary={"schema_version":SUMMARY_SCHEMA,"status":"PASS" if not failures else "FAIL","generated_at_utc":_utc_now(),"rank":args.rank,"world_size":args.world_size,"logical_shard_count":LOGICAL_SHARD_COUNT,"assigned_shards":list(assigned_shard_ids(args.rank,args.world_size)),"all_shard_causal_frame_loads":loads,"manifest_path":str(manifest_path),"manifest_file_sha256":manifest_file_sha,"manifest_sha256":payload.get("manifest_sha256"),"selected_job_count":len(jobs),"newly_completed_episode_count":len(completed),"skipped_complete_episode_count":skipped,"failed_episode_count":len(failures),"elapsed_seconds":time.monotonic()-rank_started,"predicted_depth_summary":{"pixel_count":pixels,"finite_positive_pixel_count":finite,"finite_positive_ratio":finite/max(1,pixels),"mean_m":weighted_depth/max(1,finite)},"failures":failures,"cache_contract":{"schema_version":CACHE_SCHEMA,"gt_depth_or_seg_written":False,"sam2_reverse":False,"atomic_publish":"create-only partial then rename"},"model_contract":{"da3_model":str(args.da3_model),"grounding_model":str(args.grounding_model),"sam2_checkpoint":str(args.sam2_checkpoint),"sam2_expected_commit":args.sam2_expected_commit,"device":args.device}}
    _write_json_atomic(summary_path,summary,replace=True); _append_progress(progress_path,{"timestamp_utc":_utc_now(),"event":"rank_complete","rank":args.rank,"status":summary["status"],"summary_path":str(summary_path)}); print(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)); return 0 if not failures else 1

if __name__ == "__main__":
    raise SystemExit(main())
