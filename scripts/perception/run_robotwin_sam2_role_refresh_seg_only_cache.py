#!/usr/bin/env python3
"""Build a create-only RGB/instruction -> GroundingDINO/SAM2 cache.

This V3 path intentionally has no Depth Anything dependency and never writes
predicted or simulator depth.  It reuses the verified role-separated,
forward-only SAM2 tracking implementation from the V2 cache builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


THIS_FILE = Path(__file__).resolve()
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

import run_robotwin_da3_sam2_cache_pilot as v1  # noqa: E402
import run_robotwin_da3_sam2_role_refresh_cache as v2  # noqa: E402
from robotwin_grounding_role_schema import (  # noqa: E402
    SCHEMA_VERSION as ROLE_SCHEMA_VERSION,
    build_grounding_role_prompts,
)


CACHE_SCHEMA = "fastwam.robotwin-grounded-sam2-reacquire-seg-cache.v3"
SUMMARY_SCHEMA = "fastwam.robotwin-grounded-sam2-reacquire-seg-rank-summary.v3"
SELECTION_SCHEMA = "fastwam.perception-episode-selection.v1"
# The segmentation algorithm is unchanged from V2; only the cache payload is
# now segmentation-only.
SEGMENTATION_METHOD = v2.SEGMENTATION_METHOD

CAMERA_NAMES = v2.CAMERA_NAMES
ROLE_ORDER = v2.ROLE_ORDER
ROLE_CODE = v2.ROLE_CODE
HEIGHT = v2.HEIGHT
WIDTH = v2.WIDTH


def _configured_logical_shard_count() -> int:
    """Read the V3-only scheduling width without changing the V1/V2 contract."""

    raw = os.environ.get("FASTWAM_SEG_LOGICAL_SHARD_COUNT", "64")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "FASTWAM_SEG_LOGICAL_SHARD_COUNT must be an integer, "
            f"got {raw!r}"
        ) from exc
    if value not in (64, 128):
        raise RuntimeError(
            "FASTWAM_SEG_LOGICAL_SHARD_COUNT must be 64 or 128, "
            f"got {value}"
        )
    return value


# V1/V2 retain their fixed 64-shard import contract. V3 owns this separate
# scheduling value, so 128 ranks can be used without changing cache contents
# or the historical implementations.
LOGICAL_SHARD_COUNT = _configured_logical_shard_count()

_load_manifest = v2._load_manifest
_episode_output_path = v2._episode_output_path


class CacheError(RuntimeError):
    """The segmentation-only cache contract is invalid."""


def build_balanced_shards(
    episodes: list[Mapping[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[int]]:
    """Balance V3 jobs over the configured 64 or 128 logical shards."""

    return v1.build_balanced_shards(episodes, shard_count=LOGICAL_SHARD_COUNT)


def assigned_shard_ids(rank: int, world_size: int) -> tuple[int, ...]:
    """Return this rank's non-overlapping V3 logical shards."""

    if isinstance(rank, bool) or isinstance(world_size, bool):
        raise CacheError("rank/world_size must be integers")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise CacheError(f"invalid rank/world_size: {rank}/{world_size}")
    return tuple(
        shard for shard in range(LOGICAL_SHARD_COUNT) if shard % world_size == rank
    )


def jobs_for_rank(
    buckets: list[list[dict[str, Any]]], *, rank: int, world_size: int
) -> list[tuple[int, dict[str, Any]]]:
    jobs: list[tuple[int, dict[str, Any]]] = []
    for shard_id in assigned_shard_ids(rank, world_size):
        jobs.extend((shard_id, dict(row)) for row in buckets[shard_id])
    return jobs


@dataclass
class SegmentationModelBundle:
    torch: Any
    device: Any
    grounding_processor: Any
    grounding_model: Any
    sam2_predictor: Any
    sam2_commit: str


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selection_file(
    path: Path | None,
    *,
    manifest_file_sha256: str,
    known_episode_ids: set[str],
) -> tuple[set[str] | None, dict[str, Any] | None]:
    """Load an optional disjoint-work selection without changing manifest identity."""

    if path is None:
        return None, None
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CacheError("selection file must contain a JSON object")
    if payload.get("schema_version") != SELECTION_SCHEMA:
        raise CacheError(f"unsupported selection schema: {payload.get('schema_version')!r}")
    if payload.get("manifest_file_sha256") != manifest_file_sha256:
        raise CacheError("selection file is bound to a different full manifest")
    values = payload.get("episode_ids")
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise CacheError("selection episode_ids must be a JSON string list")
    selected = set(values)
    if len(selected) != len(values):
        raise CacheError("selection episode_ids contains duplicates")
    unknown = sorted(selected - known_episode_ids)
    if unknown:
        raise CacheError(f"selection contains unknown episode IDs: {unknown[:5]}")
    return selected, {
        "path": str(resolved),
        "file_sha256": _sha256_file(resolved),
        "selected_episode_count": len(selected),
    }


def _validate_episode_hdf5(row: Mapping[str, Any]) -> tuple[Path, int, list[int]]:
    """Validate only the RGB fields consumed by segmentation inference."""

    import h5py

    path = Path(str(row["episode_path"])).resolve(strict=True)
    frame_count = int(row["expected_frame_count"])
    source = row["source"]
    legal_indices = list(
        range(
            int(source["legal_source_index_start"]),
            int(source["legal_source_index_stop_inclusive"]) + 1,
        )
    )
    if not legal_indices or legal_indices[-1] >= frame_count:
        raise CacheError("legal source range is empty or exceeds frame count")
    with h5py.File(path, "r", swmr=True) as handle:
        for camera in CAMERA_NAMES:
            rgb = handle[f"observations/{camera}/rgb"]
            if rgb.shape != (frame_count, HEIGHT, WIDTH, 3):
                raise CacheError(f"bad RGB shape for {camera}: {rgb.shape}")
            if rgb.dtype != np.dtype("uint8"):
                raise CacheError(f"bad RGB dtype for {camera}: {rgb.dtype}")
    return path, frame_count, legal_indices


def _load_models(args: argparse.Namespace) -> SegmentationModelBundle:
    """Load GroundingDINO and SAM2 only; Depth Anything is never imported."""

    sam2_repo = args.sam2_repo.expanduser().resolve(strict=True)
    sam2_commit = v1._git_head(sam2_repo)
    if sam2_commit != args.sam2_expected_commit:
        raise CacheError(
            f"SAM2 commit mismatch: expected={args.sam2_expected_commit}, actual={sam2_commit}"
        )
    for required in (args.grounding_model, args.sam2_checkpoint):
        required.expanduser().resolve(strict=True)
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))

    import torch

    if not torch.cuda.is_available():
        raise CacheError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    import transformers.models.grounding_dino.modeling_grounding_dino as mgd
    from transformers import AutoConfig, AutoModelForZeroShotObjectDetection, AutoProcessor

    mgd.is_torch_cuda_available = lambda: False
    grounding_path = str(args.grounding_model.expanduser().resolve())
    config = AutoConfig.from_pretrained(grounding_path, local_files_only=True)
    config.disable_custom_kernels = True
    processor = AutoProcessor.from_pretrained(grounding_path, local_files_only=True)
    grounding = AutoModelForZeroShotObjectDetection.from_pretrained(
        grounding_path,
        config=config,
        local_files_only=True,
        use_safetensors=True,
    ).to(device).eval()

    from sam2.build_sam import build_sam2_video_predictor

    sam2 = build_sam2_video_predictor(
        args.sam2_config,
        str(args.sam2_checkpoint.expanduser().resolve()),
        device=str(device),
        mode="eval",
    )
    return SegmentationModelBundle(
        torch=torch,
        device=device,
        grounding_processor=processor,
        grounding_model=grounding,
        sam2_predictor=sam2,
        sam2_commit=sam2_commit,
    )


def _validate_existing_cache(
    path: Path, *, episode_id: str, manifest_file_sha256: str
) -> bool:
    if not path.exists():
        return False
    import h5py

    with h5py.File(path, "r", swmr=True) as handle:
        valid = (
            _decode(handle.attrs.get("schema_version", "")) == CACHE_SCHEMA
            and _decode(handle.attrs.get("status", "")) == "PASS"
            and _decode(handle.attrs.get("episode_id", "")) == episode_id
            and _decode(handle.attrs.get("manifest_file_sha256", ""))
            == manifest_file_sha256
            and bool(handle.attrs.get("segmentation_only", False))
            and not bool(handle.attrs.get("contains_predicted_depth", True))
            and "pred" in handle
            and set(handle["pred"].keys()) == {"track_id", "role_id"}
        )
    if not valid:
        raise CacheError(f"existing path is not this exact V3 seg-only cache: {path}")
    return True


def _run_episode(
    bundle: SegmentationModelBundle,
    *,
    row: Mapping[str, Any],
    shard_id: int,
    output_dir: Path,
    manifest_path: Path,
    manifest_file_sha256: str,
    manifest_semantic_sha256: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import h5py

    started = time.monotonic()
    episode_id = str(row["episode_id"])
    source_path, _, legal_indices = _validate_episode_hdf5(row)
    causal_frame_count = legal_indices[-1] + 1
    prompt_contract = build_grounding_role_prompts(
        str(row["task_name"]), str(row["instruction"])
    )
    final = _episode_output_path(output_dir, episode_id)
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(
        f".{final.name}.partial.rank{args.rank:03d}.pid{os.getpid()}"
    )
    if partial.exists():
        raise CacheError(f"partial already exists: {partial}")
    camera_summaries: list[dict[str, Any]] = []
    try:
        with h5py.File(partial, "x", libver="latest") as out, h5py.File(
            source_path, "r", swmr=True
        ) as source:
            out.attrs.update(
                {
                    "schema_version": CACHE_SCHEMA,
                    "status": "WRITING",
                    "segmentation_only": True,
                    "contains_predicted_depth": False,
                    "episode_id": episode_id,
                    "task_name": str(row["task_name"]),
                    "variant": str(row["variant"]),
                    "manifest_path": str(manifest_path),
                    "manifest_file_sha256": manifest_file_sha256,
                    "manifest_sha256": manifest_semantic_sha256,
                    "role_schema_version": ROLE_SCHEMA_VERSION,
                    "segmentation_method": SEGMENTATION_METHOD,
                    "role_prompt_mode": "one_grounding_batch_item_per_role",
                    "redetect_interval": args.redetect_interval,
                    "redetect_on_loss": args.redetect_on_loss,
                    "min_track_pixels": args.min_track_pixels,
                    "sam2_commit": bundle.sam2_commit,
                    "sam2_reverse": False,
                    "instance_id_scope": "episode_camera_role_slot",
                    "logical_shard_id": shard_id,
                    "rank": args.rank,
                    "created_at_utc": v1._utc_now(),
                    "gt_depth_or_seg_written": False,
                    "depth_status_flag_bits_reserved": "0,5",
                }
            )
            out.create_dataset(
                "source_index", data=np.asarray(legal_indices, dtype=np.uint16)
            )
            out.create_dataset(
                "camera_name",
                data=np.asarray(CAMERA_NAMES, dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )
            out.create_dataset(
                "grounding_prompt_contract_json",
                data=json.dumps(prompt_contract, ensure_ascii=False, sort_keys=True),
                dtype=h5py.string_dtype("utf-8"),
            )
            pred = out.create_group("pred")
            shape = (len(legal_indices), len(CAMERA_NAMES), HEIGHT, WIDTH)
            chunks = (1, 1, HEIGHT, WIDTH)
            track_ds = pred.create_dataset(
                "track_id", shape=shape, dtype=np.uint16, chunks=chunks,
                compression="lzf", shuffle=True,
            )
            role_ds = pred.create_dataset(
                "role_id", shape=shape, dtype=np.uint8, chunks=chunks,
                compression="lzf", shuffle=True,
            )
            flags_ds = out.create_dataset(
                "status_flags",
                shape=(len(legal_indices), len(CAMERA_NAMES)),
                dtype=np.uint16,
                chunks=(min(128, len(legal_indices)), len(CAMERA_NAMES)),
            )
            tracks = out.create_group("tracks")
            events = out.create_group("redetection_events")
            for camera_index in range(len(CAMERA_NAMES)):
                camera_summaries.append(
                    v2._track_camera(
                        bundle,
                        source_h5=source,
                        episode_path=source_path,
                        causal_frame_count=causal_frame_count,
                        legal_indices=legal_indices,
                        camera_index=camera_index,
                        prompt_contract=prompt_contract,
                        track_ds=track_ds,
                        role_ds=role_ds,
                        flags_ds=flags_ds,
                        tracks_group=tracks,
                        events_group=events,
                        args=args,
                    )
                )
            out.attrs["status"] = "PASS"
            out.attrs["completed_at_utc"] = v1._utc_now()
            out.attrs["elapsed_seconds"] = time.monotonic() - started
            out.flush()
        v2._publish_hdf5_create_only(partial, final)
    finally:
        if partial.exists():
            partial.unlink()
    return {
        "episode_id": episode_id,
        "cache_path": str(final),
        "cache_size_bytes": final.stat().st_size,
        "legal_source_count": len(legal_indices),
        "causal_processing_frame_count": causal_frame_count,
        "tracking": camera_summaries,
        "elapsed_seconds": time.monotonic() - started,
    }


def _pure_self_test() -> dict[str, Any]:
    result = v2._pure_self_test()
    if "depth_anything_3" in sys.modules:
        raise AssertionError("seg-only core imported depth_anything_3")
    return {
        "status": "PASS",
        "schema_version": CACHE_SCHEMA,
        "depth_anything_imported": False,
        "reused_role_refresh_self_test": result,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--grounding-model", type=Path, default=v1.DEFAULT_GROUNDING_MODEL)
    parser.add_argument("--sam2-repo", type=Path, default=v1.DEFAULT_SAM2_REPO)
    parser.add_argument("--sam2-config", default=v1.DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=v1.DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--sam2-expected-commit", default=v1.DEFAULT_SAM2_COMMIT)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--dedup-iou-threshold", type=float, default=0.70)
    parser.add_argument("--sam2-mask-threshold", type=float, default=0.0)
    parser.add_argument("--min-track-pixels", type=int, default=16)
    parser.add_argument("--max-boxes-per-role", type=int, default=4)
    parser.add_argument("--redetect-interval", type=int, default=8)
    parser.add_argument("--reacquire-iou-threshold", type=float, default=0.10)
    parser.add_argument("--reacquire-max-gap", type=int, default=16)
    parser.add_argument(
        "--redetect-on-loss", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    if args.manifest is None or args.output_dir is None:
        raise CacheError("--manifest and --output-dir are required")
    assigned_shard_ids(args.rank, args.world_size)
    for name in ("max_boxes_per_role", "min_track_pixels", "redetect_interval", "reacquire_max_gap"):
        if int(getattr(args, name)) <= 0:
            raise CacheError(f"--{name.replace('_', '-')} must be positive")
    if args.max_episodes < 0:
        raise CacheError("--max-episodes must be non-negative")
    for name in ("box_threshold", "text_threshold", "dedup_iou_threshold", "reacquire_iou_threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise CacheError(f"--{name.replace('_', '-')} must be in [0,1]")


def main() -> int:
    args = _parser().parse_args()
    _validate_args(args)
    if args.self_test:
        print(json.dumps(_pure_self_test(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest, manifest_file_sha = _load_manifest(manifest_path)
    known_ids = {str(row["episode_id"]) for row in manifest["episodes"]}
    selected, selection = load_selection_file(
        args.selection_file,
        manifest_file_sha256=manifest_file_sha,
        known_episode_ids=known_ids,
    )
    # A frozen selection is a scheduling boundary, not merely a post-filter.
    # Rebalance the selected rows before assigning ranks so an independently
    # launched subset uses all local workers evenly. With no selection this is
    # exactly the historical full-manifest scheduling path.
    scheduled_rows = manifest["episodes"]
    if selected is not None:
        scheduled_rows = [
            row for row in scheduled_rows
            if str(row["episode_id"]) in selected
        ]
    buckets, loads = build_balanced_shards(scheduled_rows)
    jobs = jobs_for_rank(buckets, rank=args.rank, world_size=args.world_size)
    if args.max_episodes:
        jobs = jobs[: args.max_episodes]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress" / f"rank_{args.rank:03d}.jsonl"
    summary_path = output_dir / "summaries" / f"rank_{args.rank:03d}.json"
    v1._append_progress(progress_path, {
        "timestamp_utc": v1._utc_now(), "event": "rank_start",
        "rank": args.rank, "world_size": args.world_size,
        "assigned_shards": list(assigned_shard_ids(args.rank, args.world_size)),
        "selected_job_count": len(jobs), "manifest_file_sha256": manifest_file_sha,
        "selection": selection, "segmentation_only": True,
    })

    pending: list[tuple[int, dict[str, Any]]] = []
    skipped = 0
    for shard_id, row in jobs:
        path = _episode_output_path(output_dir, str(row["episode_id"]))
        if _validate_existing_cache(path, episode_id=str(row["episode_id"]), manifest_file_sha256=manifest_file_sha):
            skipped += 1
        else:
            pending.append((shard_id, row))

    bundle = _load_models(args) if pending else None
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    for shard_id, row in pending:
        episode_id = str(row["episode_id"])
        v1._append_progress(progress_path, {
            "timestamp_utc": v1._utc_now(), "event": "episode_start",
            "rank": args.rank, "logical_shard_id": shard_id, "episode_id": episode_id,
        })
        try:
            result = _run_episode(
                bundle, row=row, shard_id=shard_id, output_dir=output_dir,
                manifest_path=manifest_path, manifest_file_sha256=manifest_file_sha,
                manifest_semantic_sha256=str(manifest.get("manifest_sha256", "")), args=args,
            )
            completed.append(result)
            v1._append_progress(progress_path, {
                "timestamp_utc": v1._utc_now(), "event": "episode_complete",
                "rank": args.rank, "logical_shard_id": shard_id, **result,
            })
        except Exception as exc:
            failure = {
                "episode_id": episode_id, "logical_shard_id": shard_id,
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            v1._append_progress(progress_path, {
                "timestamp_utc": v1._utc_now(), "event": "episode_failed",
                "rank": args.rank, **failure,
            })
            if args.fail_fast:
                break

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "generated_at_utc": v1._utc_now(),
        "rank": args.rank, "world_size": args.world_size,
        "logical_shard_count": LOGICAL_SHARD_COUNT,
        "assigned_shards": list(assigned_shard_ids(args.rank, args.world_size)),
        "all_shard_causal_frame_loads": loads,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_file_sha,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "selection": selection,
        "selected_job_count": len(jobs),
        "newly_completed_episode_count": len(completed),
        "skipped_complete_episode_count": skipped,
        "failed_episode_count": len(failures),
        "elapsed_seconds": time.monotonic() - started,
        "failures": failures,
        "cache_contract": {
            "schema_version": CACHE_SCHEMA,
            "segmentation_method": SEGMENTATION_METHOD,
            "segmentation_only": True,
            "contains_predicted_depth": False,
            "gt_depth_or_seg_written": False,
            "sam2_reverse": False,
            "atomic_publish": "create-only hard-link then unlink partial",
        },
        "model_contract": {
            "grounding_model": str(args.grounding_model),
            "sam2_checkpoint": str(args.sam2_checkpoint),
            "sam2_expected_commit": args.sam2_expected_commit,
            "device": args.device,
            "depth_model": None,
        },
    }
    v1._write_json_atomic(summary_path, summary, replace=True)
    v1._append_progress(progress_path, {
        "timestamp_utc": v1._utc_now(), "event": "rank_complete",
        "rank": args.rank, "status": summary["status"], "summary_path": str(summary_path),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
