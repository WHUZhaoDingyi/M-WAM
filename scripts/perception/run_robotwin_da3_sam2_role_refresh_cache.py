#!/usr/bin/env python3
"""Build a create-only DA3 + role-grounded SAM2 causal cache.

This add-only alternative keeps the verified v1 DA3 metric-depth path but
replaces its frame-0-only segmentation logic with the following online loop:

    one GroundingDINO prompt per semantic role
        -> role-local boxes
        -> SAM2 forward-only propagation
        -> redetect every N frames or immediately after track loss

Each GroundingDINO batch item contains exactly one role prompt.  A detector
label is therefore never mapped back to a role by substring matching.  The
current RGB frame and current task instruction are the only grounding inputs;
future RGB, simulator depth/segmentation, and actions are never read as model
inputs.  Existing v1/v2 caches are never modified or overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

import run_robotwin_da3_sam2_cache_pilot as v1  # noqa: E402
from robotwin_grounding_role_schema import (  # noqa: E402
    SCHEMA_VERSION as ROLE_SCHEMA_VERSION,
    build_grounding_role_prompts,
)


CACHE_SCHEMA = "fastwam.robotwin-grounded-sam2-reacquire-cache.v2"
SUMMARY_SCHEMA = "fastwam.robotwin-grounded-sam2-reacquire-rank-summary.v2"
SEGMENTATION_METHOD = "groundingdino-role-separate-sam2-causal-reacquire-v2"

CAMERA_NAMES = v1.CAMERA_NAMES
ROLE_ORDER = v1.ROLE_ORDER
ROLE_CODE = v1.ROLE_CODE
HEIGHT = v1.HEIGHT
WIDTH = v1.WIDTH
LOGICAL_SHARD_COUNT = v1.LOGICAL_SHARD_COUNT

# Public compatibility surface used by the shared launcher.
_load_manifest = v1._load_manifest
build_balanced_shards = v1.build_balanced_shards
jobs_for_rank = v1.jobs_for_rank
_episode_output_path = v1._episode_output_path

# Bit 0 and bit 5 intentionally match the reused v1 DA3 writer.
FLAG_DEPTH_OK = np.uint16(1 << 0)
FLAG_GROUNDING_RAN = np.uint16(1 << 1)
FLAG_SAM2_TRACKED = np.uint16(1 << 2)
FLAG_GROUNDING_MISS = np.uint16(1 << 3)
FLAG_EMPTY_TRACK_MASK = np.uint16(1 << 4)
FLAG_NONFINITE_DEPTH = np.uint16(1 << 5)
FLAG_PERIODIC_REDETECT = np.uint16(1 << 6)
FLAG_LOSS_REDETECT = np.uint16(1 << 7)


class CacheError(RuntimeError):
    """The create-only cache or causal segmentation contract is invalid."""


@dataclass(frozen=True)
class GroundedBox:
    role: str
    phrase: str
    score: float
    box_xyxy: tuple[float, float, float, float]


@dataclass
class TrackRecord:
    track_id: int
    role: str
    first_seen_source_index: int
    last_seen_source_index: int
    detection_count: int = 0
    tracked_frame_count: int = 0
    pixel_count: int = 0
    score_sum: float = 0.0
    score_max: float = 0.0
    last_box_xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class AssignedDetection:
    track_id: int
    role: str
    phrase: str
    score: float
    box_xyxy: tuple[float, float, float, float]


def _role_caption(prompt_contract: Mapping[str, Any], role: str) -> str:
    roles = prompt_contract.get("roles")
    if not isinstance(roles, Mapping):
        raise CacheError("prompt contract lacks role phrases")
    phrases = [str(value).strip() for value in roles.get(role, ()) if str(value).strip()]
    if not phrases:
        return ""
    return " . ".join(phrases) + " ."


def role_prompt_batch(prompt_contract: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return one independent detector caption for each nonempty role."""

    expected = prompt_contract.get("expected_count")
    if not isinstance(expected, Mapping):
        raise CacheError("prompt contract lacks expected role counts")
    result: list[tuple[str, str]] = []
    for role in ROLE_ORDER:
        if int(expected.get(role, 0)) <= 0:
            continue
        caption = _role_caption(prompt_contract, role)
        if not caption:
            raise CacheError(f"role {role!r} has positive count but no prompt")
        result.append((role, caption))
    return result


def _within_role_nms(
    rows: Sequence[GroundedBox], *, iou_threshold: float, limit: int
) -> list[GroundedBox]:
    """Suppress synonyms only within one role; cross-role overlap is valid."""

    selected: list[GroundedBox] = []
    for row in sorted(rows, key=lambda item: (-item.score, item.box_xyxy, item.phrase)):
        if any(
            v1._box_iou(row.box_xyxy, prior.box_xyxy) >= iou_threshold
            for prior in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _mask_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def _associate_role_boxes(
    detections: Sequence[GroundedBox],
    *,
    frame_index: int,
    records: dict[int, TrackRecord],
    next_occurrence: dict[str, int],
    max_tracks_per_role: Mapping[str, int],
    iou_threshold: float,
    max_gap: int,
) -> list[AssignedDetection]:
    """Preserve role-local IDs while keeping every role pool strictly bounded."""

    assignments: list[AssignedDetection] = []
    used_ids: set[int] = set()
    for detection in sorted(
        detections, key=lambda item: (-item.score, item.role, item.box_xyxy)
    ):
        candidates: list[tuple[float, int]] = []
        for track_id, record in records.items():
            if track_id in used_ids or record.role != detection.role:
                continue
            if frame_index - record.last_seen_source_index > max_gap:
                continue
            if record.last_box_xyxy is None:
                continue
            iou = v1._box_iou(detection.box_xyxy, record.last_box_xyxy)
            if iou >= iou_threshold:
                candidates.append((iou, track_id))
        if candidates:
            _, track_id = max(candidates, key=lambda item: (item[0], -item[1]))
            record = records[track_id]
        else:
            role_records = [
                record for record in records.values() if record.role == detection.role
            ]
            role_limit = int(max_tracks_per_role.get(detection.role, 0))
            if role_limit <= 0:
                raise CacheError(f"role {detection.role!r} has no track slots")
            reusable = [
                record for record in role_records if record.track_id not in used_ids
            ]
            if len(role_records) >= role_limit:
                if not reusable:
                    # Grounding is already capped to the same role limit, so this
                    # indicates an internal contract error rather than a reason
                    # to grow the SAM2 object state.
                    raise CacheError(
                        f"role {detection.role!r} exhausted its bounded track pool"
                    )
                record = min(
                    reusable,
                    key=lambda item: (item.last_seen_source_index, item.track_id),
                )
                track_id = record.track_id
            else:
                next_occurrence[detection.role] += 1
                occurrence = next_occurrence[detection.role]
                track_id = ROLE_CODE[detection.role] * 1000 + occurrence
                if track_id >= 2**16:
                    raise CacheError(f"track ID overflow for role {detection.role!r}")
                record = TrackRecord(
                    track_id=track_id,
                    role=detection.role,
                    first_seen_source_index=frame_index,
                    last_seen_source_index=frame_index,
                )
                records[track_id] = record
        used_ids.add(track_id)
        record.last_seen_source_index = frame_index
        record.last_box_xyxy = detection.box_xyxy
        record.detection_count += 1
        record.score_sum += detection.score
        record.score_max = max(record.score_max, detection.score)
        assignments.append(
            AssignedDetection(
                track_id=track_id,
                role=detection.role,
                phrase=detection.phrase,
                score=detection.score,
                box_xyxy=detection.box_xyxy,
            )
        )
    return assignments


def _next_periodic_boundary(frame_index: int, interval: int) -> int:
    return ((int(frame_index) // int(interval)) + 1) * int(interval)


def _ground_camera_frame(
    bundle: Any,
    *,
    rgb: np.ndarray,
    prompt_contract: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[list[GroundedBox], dict[str, int]]:
    """Ground one current frame with independent role batch items."""

    from PIL import Image

    role_prompts = role_prompt_batch(prompt_contract)
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    images = [image for _ in role_prompts]
    texts = [caption for _, caption in role_prompts]
    inputs = bundle.grounding_processor(
        images=images,
        text=texts,
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        key: (value.to(bundle.device) if hasattr(value, "to") else value)
        for key, value in inputs.items()
    }
    with bundle.torch.inference_mode():
        outputs = bundle.grounding_model(**inputs)
    results = bundle.grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[(HEIGHT, WIDTH)] * len(role_prompts),
    )
    expected = prompt_contract["expected_count"]
    kept: list[GroundedBox] = []
    raw_count = 0
    for (role, _), result in zip(role_prompts, results):
        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        labels = result.get("text_labels", result.get("labels", []))
        role_rows: list[GroundedBox] = []
        for box, score, label in zip(boxes, scores, labels):
            raw_count += 1
            role_rows.append(
                GroundedBox(
                    role=role,
                    phrase=str(label),
                    score=float(score),
                    box_xyxy=tuple(float(value) for value in box),
                )
            )
        role_limit = min(
            args.max_boxes_per_role,
            max(1, int(expected.get(role, 1))),
        )
        kept.extend(
            _within_role_nms(
                role_rows,
                iou_threshold=args.dedup_iou_threshold,
                limit=role_limit,
            )
        )
    return kept, {
        "role_prompt_count": len(role_prompts),
        "raw_box_count": raw_count,
        "kept_box_count": len(kept),
    }


def _write_track_table(group: Any, records: Mapping[int, TrackRecord]) -> None:
    import h5py

    ids = sorted(records)
    string_dtype = h5py.string_dtype("utf-8")
    group.create_dataset("track_id", data=np.asarray(ids, dtype=np.uint16))
    group.create_dataset(
        "role_code",
        data=np.asarray([ROLE_CODE[records[value].role] for value in ids], dtype=np.uint8),
    )
    group.create_dataset(
        "role",
        data=np.asarray([records[value].role for value in ids], dtype=object),
        dtype=string_dtype,
    )
    for name, dtype, values in (
        (
            "first_seen_source_index",
            np.uint16,
            [records[value].first_seen_source_index for value in ids],
        ),
        (
            "last_seen_source_index",
            np.uint16,
            [records[value].last_seen_source_index for value in ids],
        ),
        (
            "detection_count",
            np.uint16,
            [records[value].detection_count for value in ids],
        ),
        (
            "tracked_frame_count",
            np.uint32,
            [records[value].tracked_frame_count for value in ids],
        ),
        (
            "pixel_count",
            np.uint64,
            [records[value].pixel_count for value in ids],
        ),
    ):
        group.create_dataset(name, data=np.asarray(values, dtype=dtype))
    group.create_dataset(
        "detector_score_mean",
        data=np.asarray(
            [
                records[value].score_sum / max(1, records[value].detection_count)
                for value in ids
            ],
            dtype=np.float32,
        ),
    )
    group.create_dataset(
        "detector_score_max",
        data=np.asarray([records[value].score_max for value in ids], dtype=np.float32),
    )


def _write_redetection_events(group: Any, events: Sequence[Mapping[str, Any]]) -> None:
    import h5py

    string_dtype = h5py.string_dtype("utf-8")
    group.create_dataset(
        "source_index",
        data=np.asarray([int(row["source_index"]) for row in events], dtype=np.uint16),
    )
    group.create_dataset(
        "reason",
        data=np.asarray([str(row["reason"]) for row in events], dtype=object),
        dtype=string_dtype,
    )
    for name in ("role_prompt_count", "raw_box_count", "kept_box_count"):
        group.create_dataset(
            name,
            data=np.asarray([int(row[name]) for row in events], dtype=np.uint16),
        )


def _track_camera(
    bundle: Any,
    *,
    source_h5: Any,
    episode_path: Path,
    causal_frame_count: int,
    legal_indices: Sequence[int],
    camera_index: int,
    prompt_contract: Mapping[str, Any],
    track_ds: Any,
    role_ds: Any,
    flags_ds: Any,
    tracks_group: Any,
    events_group: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    camera = CAMERA_NAMES[camera_index]
    sequence = v1.HDF5FrameSequence(
        episode_path=episode_path,
        camera_name=camera,
        frame_count=causal_frame_count,
        image_size=int(bundle.sam2_predictor.image_size),
        torch_module=bundle.torch,
        compute_device=bundle.device,
    )
    state = None
    records: dict[int, TrackRecord] = {}
    next_occurrence = {role: 0 for role in ROLE_ORDER}
    expected_count = prompt_contract["expected_count"]
    max_tracks_per_role = {
        role: min(args.max_boxes_per_role, max(1, int(expected_count.get(role, 0))))
        for role in ROLE_ORDER
        if int(expected_count.get(role, 0)) > 0
    }
    active_ids: set[int] = set()
    row_lookup = {frame: row for row, frame in enumerate(legal_indices)}
    seen: set[int] = set()
    events: list[dict[str, Any]] = []
    loss_frame_count = 0
    empty_frame_count = 0
    frame_index = 0
    redetect_reason = "initial"
    rgb_dataset = source_h5[f"observations/{camera}/rgb"]

    try:
        state = v1._init_sam2_state_from_hdf5_sequence(
            bundle.sam2_predictor,
            sequence,
            torch_module=bundle.torch,
        )
        while frame_index < causal_frame_count:
            rgb = np.asarray(rgb_dataset[frame_index], dtype=np.uint8)
            detections, stats = _ground_camera_frame(
                bundle,
                rgb=rgb,
                prompt_contract=prompt_contract,
                args=args,
            )
            assigned = _associate_role_boxes(
                detections,
                frame_index=frame_index,
                records=records,
                next_occurrence=next_occurrence,
                max_tracks_per_role=max_tracks_per_role,
                iou_threshold=args.reacquire_iou_threshold,
                max_gap=args.reacquire_max_gap,
            )
            events.append(
                {
                    "source_index": frame_index,
                    "reason": redetect_reason,
                    **stats,
                }
            )
            row = row_lookup.get(frame_index)
            if row is not None:
                flags_ds[row, camera_index] |= FLAG_GROUNDING_RAN
                if redetect_reason == "periodic":
                    flags_ds[row, camera_index] |= FLAG_PERIODIC_REDETECT
                elif redetect_reason == "loss":
                    flags_ds[row, camera_index] |= FLAG_LOSS_REDETECT
                if not assigned:
                    flags_ds[row, camera_index] |= FLAG_GROUNDING_MISS

            detected_by_role: dict[str, set[int]] = {}
            for detection in assigned:
                detected_by_role.setdefault(detection.role, set()).add(detection.track_id)
                bundle.sam2_predictor.add_new_points_or_box(
                    state,
                    frame_idx=frame_index,
                    obj_id=detection.track_id,
                    box=np.asarray(detection.box_xyxy, dtype=np.float32),
                    clear_old_points=True,
                )
            # Detection refreshes only add/update role slots.  Healthy active
            # tracks that were not detected in this periodic frame stay active;
            # only the winning-mask pixel test below is allowed to retire them.
            for ids in detected_by_role.values():
                active_ids.update(ids)

            next_boundary = min(
                _next_periodic_boundary(frame_index, args.redetect_interval),
                causal_frame_count,
            )
            stop_frame = next_boundary - 1
            if not state["obj_ids"]:
                # No SAM2 object can be propagated yet.  Retry on the next frame;
                # this is the loss/miss fallback and remains strictly causal.
                if row is not None:
                    flags_ds[row, camera_index] |= FLAG_EMPTY_TRACK_MASK
                    seen.add(frame_index)
                    empty_frame_count += 1
                frame_index += 1
                redetect_reason = "loss"
                continue

            loss_at: int | None = None
            iterator = bundle.sam2_predictor.propagate_in_video(
                state,
                start_frame_idx=frame_index,
                max_frame_num_to_track=stop_frame - frame_index,
                reverse=False,
            )
            for current, object_ids, mask_logits in iterator:
                ids = [int(value) for value in object_ids]
                logits = mask_logits.detach().float().cpu().numpy()
                if logits.ndim == 4 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                if logits.shape != (len(ids), HEIGHT, WIDTH):
                    raise CacheError(
                        f"unexpected SAM2 mask shape {logits.shape} for {camera}"
                    )
                index_by_id = {track_id: index for index, track_id in enumerate(ids)}
                usable = [
                    track_id for track_id in sorted(active_ids) if track_id in index_by_id
                ]
                track_map = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
                role_map = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                missing_active = False
                if usable:
                    chosen_logits = np.stack(
                        [logits[index_by_id[track_id]] for track_id in usable], axis=0
                    )
                    best = np.argmax(chosen_logits, axis=0)
                    best_logit = np.max(chosen_logits, axis=0)
                    valid = best_logit > args.sam2_mask_threshold
                    usable_array = np.asarray(usable, dtype=np.uint16)
                    winning_ids = usable_array[best]
                    track_map[valid] = winning_ids[valid]
                    lost_ids: set[int] = set()
                    for track_id in usable:
                        # Judge health after cross-object competition.  A raw SAM2
                        # logit mask can be nonempty while winning zero pixels in
                        # the final segmentation map.
                        object_mask = track_map == track_id
                        winning_pixels = int(object_mask.sum())
                        if winning_pixels < args.min_track_pixels:
                            lost_ids.add(track_id)
                            continue
                        record = records[track_id]
                        record.last_seen_source_index = current
                        record.last_box_xyxy = _mask_box(object_mask)
                        record.tracked_frame_count += 1
                        record.pixel_count += winning_pixels
                        role_map[object_mask] = np.uint8(
                            ROLE_CODE[record.role]
                        )
                    missing_active = bool(lost_ids)
                else:
                    valid = np.zeros((HEIGHT, WIDTH), dtype=bool)
                    lost_ids = set()
                    missing_active = True

                output_row = row_lookup.get(current)
                if output_row is not None:
                    track_ds[output_row, camera_index] = track_map
                    role_ds[output_row, camera_index] = role_map
                    flags_ds[output_row, camera_index] |= FLAG_SAM2_TRACKED
                    seen.add(current)
                    if not bool(valid.any()):
                        flags_ds[output_row, camera_index] |= FLAG_EMPTY_TRACK_MASK
                        empty_frame_count += 1
                if lost_ids:
                    active_ids.difference_update(lost_ids)
                if args.redetect_on_loss and missing_active and current < stop_frame:
                    loss_at = current
                    loss_frame_count += 1
                    break

            if loss_at is not None:
                frame_index = loss_at + 1
                redetect_reason = "loss"
            else:
                frame_index = next_boundary
                redetect_reason = "periodic"
    finally:
        if state is not None:
            bundle.sam2_predictor.reset_state(state)
        sequence.close()

    missing = sorted(set(legal_indices) - seen)
    if missing:
        raise CacheError(
            f"causal tracking omitted {len(missing)} frames for {camera}; "
            f"first={missing[:5]}"
        )
    _write_track_table(tracks_group.create_group(camera), records)
    _write_redetection_events(events_group.create_group(camera), events)
    return {
        "camera": camera,
        "track_count": len(records),
        "redetection_count": len(events),
        "loss_redetection_count": sum(row["reason"] == "loss" for row in events),
        "periodic_redetection_count": sum(
            row["reason"] == "periodic" for row in events
        ),
        "loss_frame_count": loss_frame_count,
        "tracked_frame_count": len(seen),
        "empty_frame_count": empty_frame_count,
    }


def _publish_hdf5_create_only(partial: Path, final: Path) -> None:
    """Atomically publish without ever replacing an existing cache file."""

    fd = os.open(partial, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.link(partial, final)
    directory_fd = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    partial.unlink()


def _validate_existing_cache(
    path: Path, *, episode_id: str, manifest_file_sha256: str
) -> bool:
    if not path.exists():
        return False
    import h5py

    with h5py.File(path, "r", swmr=True) as handle:
        valid = (
            handle.attrs.get("schema_version") == CACHE_SCHEMA
            and handle.attrs.get("status") == "PASS"
            and handle.attrs.get("episode_id") == episode_id
            and handle.attrs.get("manifest_file_sha256") == manifest_file_sha256
        )
    if not valid:
        raise CacheError(f"existing path is not this exact v2 cache: {path}")
    return True


def _run_episode(
    bundle: Any,
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
    source_path, _, legal_indices = v1._validate_episode_hdf5(row)
    causal_frames = legal_indices[-1] + 1
    prompt_contract = build_grounding_role_prompts(
        str(row["task_name"]), str(row["instruction"])
    )
    final = v1._episode_output_path(output_dir, episode_id)
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
            depth_ds = pred.create_dataset(
                "depth_m",
                shape=shape,
                dtype=np.float16,
                chunks=chunks,
                compression="lzf",
                shuffle=True,
            )
            confidence_ds = pred.create_dataset(
                "depth_confidence",
                shape=shape,
                dtype=np.float16,
                chunks=chunks,
                compression="lzf",
                shuffle=True,
            )
            track_ds = pred.create_dataset(
                "track_id",
                shape=shape,
                dtype=np.uint16,
                chunks=chunks,
                compression="lzf",
                shuffle=True,
            )
            role_ds = pred.create_dataset(
                "role_id",
                shape=shape,
                dtype=np.uint8,
                chunks=chunks,
                compression="lzf",
                shuffle=True,
            )
            flags_ds = out.create_dataset(
                "status_flags",
                shape=(len(legal_indices), len(CAMERA_NAMES)),
                dtype=np.uint16,
                chunks=(min(128, len(legal_indices)), len(CAMERA_NAMES)),
            )
            tracks = out.create_group("tracks")
            events = out.create_group("redetection_events")
            depth_summary = v1._run_da3_batches(
                bundle,
                source_h5=source,
                legal_indices=legal_indices,
                depth_ds=depth_ds,
                confidence_ds=confidence_ds,
                flags_ds=flags_ds,
                args=args,
            )
            for camera_index in range(len(CAMERA_NAMES)):
                camera_summaries.append(
                    _track_camera(
                        bundle,
                        source_h5=source,
                        episode_path=source_path,
                        causal_frame_count=causal_frames,
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
        _publish_hdf5_create_only(partial, final)
    finally:
        if partial.exists():
            partial.unlink()
    return {
        "episode_id": episode_id,
        "cache_path": str(final),
        "cache_size_bytes": final.stat().st_size,
        "legal_source_count": len(legal_indices),
        "causal_processing_frame_count": causal_frames,
        "depth": depth_summary,
        "tracking": camera_summaries,
        "elapsed_seconds": time.monotonic() - started,
    }


def _pure_self_test() -> dict[str, Any]:
    prompt = build_grounding_role_prompts(
        "place_object_basket",
        "Place the red toy car into the woven basket.",
    )
    batches = dict(role_prompt_batch(prompt))
    assert "toy car" in batches["manipulated_target"]
    assert "basket" in batches["destination_or_reference"]
    assert "robot" in batches["robot"] or "gripper" in batches["robot"]
    assert "basket" not in batches["manipulated_target"]

    records: dict[int, TrackRecord] = {}
    counters = {role: 0 for role in ROLE_ORDER}
    first = _associate_role_boxes(
        [GroundedBox("manipulated_target", "toy car", 0.9, (10, 10, 30, 30))],
        frame_index=0,
        records=records,
        next_occurrence=counters,
        max_tracks_per_role={"manipulated_target": 1},
        iou_threshold=0.1,
        max_gap=16,
    )
    records[first[0].track_id].last_box_xyxy = (11, 10, 31, 30)
    second = _associate_role_boxes(
        [GroundedBox("manipulated_target", "toy car", 0.8, (12, 10, 32, 30))],
        frame_index=8,
        records=records,
        next_occurrence=counters,
        max_tracks_per_role={"manipulated_target": 1},
        iou_threshold=0.1,
        max_gap=16,
    )
    assert first[0].track_id == second[0].track_id
    # Repeated no-IoU refreshes reuse the fixed role slot instead of adding
    # permanent SAM2 objects every redetection interval.
    for frame in (16, 24, 32, 40):
        refreshed = _associate_role_boxes(
            [
                GroundedBox(
                    "manipulated_target",
                    "toy car",
                    0.7,
                    (100 + frame, 100, 110 + frame, 110),
                )
            ],
            frame_index=frame,
            records=records,
            next_occurrence=counters,
            max_tracks_per_role={"manipulated_target": 1},
            iou_threshold=0.9,
            max_gap=16,
        )
        assert refreshed[0].track_id == first[0].track_id
    assert len(records) == 1
    assert _next_periodic_boundary(0, 8) == 8
    assert _next_periodic_boundary(8, 8) == 16
    assert _mask_box(np.eye(4, dtype=bool)) == (0.0, 0.0, 4.0, 4.0)
    return {
        "status": "PASS",
        "schema_version": CACHE_SCHEMA,
        "segmentation_method": SEGMENTATION_METHOD,
        "role_prompt_count": len(batches),
        "stable_reacquired_track_id": first[0].track_id,
        "redetect_interval_example": 8,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--da3-repo", type=Path, default=v1.DEFAULT_DA3_REPO)
    parser.add_argument("--da3-model", type=Path, default=v1.DEFAULT_DA3_MODEL)
    parser.add_argument(
        "--grounding-model", type=Path, default=v1.DEFAULT_GROUNDING_MODEL
    )
    parser.add_argument("--sam2-repo", type=Path, default=v1.DEFAULT_SAM2_REPO)
    parser.add_argument("--sam2-config", default=v1.DEFAULT_SAM2_CONFIG)
    parser.add_argument(
        "--sam2-checkpoint", type=Path, default=v1.DEFAULT_SAM2_CHECKPOINT
    )
    parser.add_argument(
        "--sam2-expected-commit", default=v1.DEFAULT_SAM2_COMMIT
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--da3-batch-size", type=int, default=12)
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--dedup-iou-threshold", type=float, default=0.70)
    parser.add_argument("--sam2-mask-threshold", type=float, default=0.0)
    parser.add_argument("--min-track-pixels", type=int, default=16)
    parser.add_argument("--max-boxes-per-role", type=int, default=4)
    parser.add_argument("--redetect-interval", type=int, default=8)
    parser.add_argument("--reacquire-iou-threshold", type=float, default=0.10)
    parser.add_argument("--reacquire-max-gap", type=int, default=16)
    parser.add_argument(
        "--redetect-on-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    v1.assigned_shard_ids(args.rank, args.world_size)
    integer_positive = (
        "da3_batch_size",
        "da3_process_res",
        "max_boxes_per_role",
        "min_track_pixels",
        "redetect_interval",
        "reacquire_max_gap",
    )
    for name in integer_positive:
        if int(getattr(args, name)) <= 0:
            raise CacheError(f"--{name.replace('_', '-')} must be positive")
    if args.max_episodes < 0:
        raise CacheError("--max-episodes must be non-negative")
    for name in (
        "box_threshold",
        "text_threshold",
        "dedup_iou_threshold",
        "reacquire_iou_threshold",
    ):
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
    payload, manifest_file_sha = v1._load_manifest(manifest_path)
    buckets, loads = v1.build_balanced_shards(payload["episodes"])
    jobs = v1.jobs_for_rank(buckets, rank=args.rank, world_size=args.world_size)
    if args.max_episodes:
        jobs = jobs[: args.max_episodes]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress" / f"rank_{args.rank:03d}.jsonl"
    summary_path = output_dir / "summaries" / f"rank_{args.rank:03d}.json"
    v1._append_progress(
        progress_path,
        {
            "timestamp_utc": v1._utc_now(),
            "event": "rank_start",
            "rank": args.rank,
            "world_size": args.world_size,
            "assigned_shards": list(v1.assigned_shard_ids(args.rank, args.world_size)),
            "selected_job_count": len(jobs),
            "manifest_file_sha256": manifest_file_sha,
            "segmentation_method": SEGMENTATION_METHOD,
        },
    )
    pending: list[tuple[int, dict[str, Any]]] = []
    skipped = 0
    for shard_id, row in jobs:
        path = v1._episode_output_path(output_dir, str(row["episode_id"]))
        if _validate_existing_cache(
            path,
            episode_id=str(row["episode_id"]),
            manifest_file_sha256=manifest_file_sha,
        ):
            skipped += 1
        else:
            pending.append((shard_id, row))

    bundle = v1._load_models(args) if pending else None
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    rank_started = time.monotonic()
    for shard_id, row in pending:
        episode_id = str(row["episode_id"])
        v1._append_progress(
            progress_path,
            {
                "timestamp_utc": v1._utc_now(),
                "event": "episode_start",
                "rank": args.rank,
                "logical_shard_id": shard_id,
                "episode_id": episode_id,
            },
        )
        try:
            result = _run_episode(
                bundle,
                row=row,
                shard_id=shard_id,
                output_dir=output_dir,
                manifest_path=manifest_path,
                manifest_file_sha256=manifest_file_sha,
                manifest_semantic_sha256=str(payload.get("manifest_sha256", "")),
                args=args,
            )
            completed.append(result)
            v1._append_progress(
                progress_path,
                {
                    "timestamp_utc": v1._utc_now(),
                    "event": "episode_complete",
                    "rank": args.rank,
                    "logical_shard_id": shard_id,
                    **result,
                },
            )
        except Exception as exc:
            failure = {
                "episode_id": episode_id,
                "logical_shard_id": shard_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            v1._append_progress(
                progress_path,
                {
                    "timestamp_utc": v1._utc_now(),
                    "event": "episode_failed",
                    "rank": args.rank,
                    **failure,
                },
            )
            if args.fail_fast:
                break

    finite = sum(
        int(row["depth"]["finite_positive_pixel_count"]) for row in completed
    )
    pixels = sum(int(row["depth"]["pixel_count"]) for row in completed)
    weighted_depth = sum(
        float(row["depth"]["mean_m"])
        * int(row["depth"]["finite_positive_pixel_count"])
        for row in completed
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "generated_at_utc": v1._utc_now(),
        "rank": args.rank,
        "world_size": args.world_size,
        "logical_shard_count": LOGICAL_SHARD_COUNT,
        "assigned_shards": list(v1.assigned_shard_ids(args.rank, args.world_size)),
        "all_shard_causal_frame_loads": loads,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_file_sha,
        "manifest_sha256": payload.get("manifest_sha256"),
        "selected_job_count": len(jobs),
        "newly_completed_episode_count": len(completed),
        "skipped_complete_episode_count": skipped,
        "failed_episode_count": len(failures),
        "elapsed_seconds": time.monotonic() - rank_started,
        "predicted_depth_summary": {
            "pixel_count": pixels,
            "finite_positive_pixel_count": finite,
            "finite_positive_ratio": finite / max(1, pixels),
            "mean_m": weighted_depth / max(1, finite),
        },
        "failures": failures,
        "cache_contract": {
            "schema_version": CACHE_SCHEMA,
            "segmentation_method": SEGMENTATION_METHOD,
            "gt_depth_or_seg_written": False,
            "sam2_reverse": False,
            "redetect_interval": args.redetect_interval,
            "redetect_on_loss": args.redetect_on_loss,
            "min_track_pixels": args.min_track_pixels,
            "instance_id_scope": "episode_camera_role_slot",
            "atomic_publish": "create-only hard-link then unlink partial",
        },
        "model_contract": {
            "da3_model": str(args.da3_model),
            "grounding_model": str(args.grounding_model),
            "sam2_checkpoint": str(args.sam2_checkpoint),
            "sam2_expected_commit": args.sam2_expected_commit,
            "device": args.device,
        },
    }
    v1._write_json_atomic(summary_path, summary, replace=True)
    v1._append_progress(
        progress_path,
        {
            "timestamp_utc": v1._utc_now(),
            "event": "rank_complete",
            "rank": args.rank,
            "status": summary["status"],
            "summary_path": str(summary_path),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
