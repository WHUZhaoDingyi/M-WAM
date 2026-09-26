"""Fail-closed Formal V1 publication and frozen-split contracts.

The raw HDF5 files remain immutable candidate/staging artifacts.  Formal
eligibility is granted only to an independently verified *index* that:

* exactly closes an immutable target universe with episodes plus explicit
  exclusions;
* re-hashes every source receipt, candidate report and HDF5 episode;
* independently validates the raw-v1 metadata and legal K=8/action windows;
* assigns whole episode groups to one deterministic train/validation/test
  split; and
* binds the resulting split manifest to an immutable verification receipt.

No function in this module authorizes a trained model for production.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from multiprocessing import get_context
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

from .v1_index import (
    FASTWAM_FUTURE_HORIZONS_SECONDS,
    FASTWAM_LOGICAL_FUTURE_OFFSETS,
    PHYSICAL_CAMERA_NAMES,
    RAW_V1_SCHEMA,
    ActionSamplingConfig,
    IndexProgressCallback,
    ReaderFactory,
    V1EpisodeRecord,
    V1IndexConfig,
    V1SampleIndex,
    _build_v1_sample_index_from_episode_records,
    _merge_v1_sample_index_shards,
    _default_reader_factory,
    _validate_candidate_report,
    _validate_control_stream_contract,
    _validate_record_metadata,
    merge_v1_runtime_index_bindings,
)
from .route_b_language_offline import RouteBLanguageBundleResolver


SOURCE_INVENTORY_SCHEMA = "robotwin_momap_formal_v1_source_inventory_v1"
DATASET_RECEIPT_SCHEMA = "robotwin_momap_formal_v1_dataset_receipt_v1"
SPLIT_MANIFEST_SCHEMA = "robotwin_momap_formal_v1_split_manifest_v1"
FLEET_MANIFEST_SCHEMA = "robotwin_momap_route_b_v1_fleet_manifest_v2"
NATIVE_TARGET_RECEIPT_SCHEMA = (
    "robotwin_momap_route_b_v1_fleet_target_success_v5"
)
IMPORTED_TARGET_RECEIPT_SCHEMA = (
    "robotwin_momap_route_b_v1_r8_target_imported_success_v1"
)
ALLOWED_SOURCE_RECEIPT_SCHEMAS = frozenset(
    {NATIVE_TARGET_RECEIPT_SCHEMA, IMPORTED_TARGET_RECEIPT_SCHEMA}
)
ALLOWED_EXCLUSION_REASON_CODES = frozenset(
    {
        "semantic_schedule_exhausted",
        "infrastructure_retry_exhausted",
        "explicit_scientific_exclusion",
    }
)
FORMAL_SPLIT_NAMES = ("train", "validation", "test")
FORMAL_COORDINATE_FRAME = "head_camera@source_t0"
FORMAL_FUTURE_OFFSET_BASIS = "fastwam_clip_frame@15hz"
FORMAL_ACTION_MODE = "baseline_rate_resampled@15Hz"
FORMAL_ACTION_HORIZON = 32
FORMAL_PROPRIO_MODE = "last_command"

_INVENTORY_HASH_DOMAIN = b"robotwin-momap-formal-v1-source-inventory-v1"
_EPISODE_SET_HASH_DOMAIN = b"robotwin-momap-formal-v1-episode-set-v1"
_EXCLUSION_SET_HASH_DOMAIN = b"robotwin-momap-formal-v1-exclusion-set-v1"
_RECEIPT_HASH_DOMAIN = b"robotwin-momap-formal-v1-dataset-receipt-v1"
_SPLIT_HASH_DOMAIN = b"robotwin-momap-formal-v1-split-manifest-v1"
_SPLIT_ORDER_HASH_DOMAIN = b"robotwin-momap-formal-v1-split-order-v1"
_FLEET_MANIFEST_HASH_DOMAIN = b"robotwin-momap-route-b-v1-fleet-manifest-v2"

_TaskT = TypeVar("_TaskT")
_ResultT = TypeVar("_ResultT")
ProgressCallback = Callable[[str, int, int], None]


class FormalV1ContractError(ValueError):
    """A Formal V1 artifact failed a fail-closed contract check."""


@dataclass(frozen=True)
class FormalV1Artifacts:
    """Paths and semantic digests emitted by the independent verifier."""

    dataset_receipt_path: Path
    dataset_receipt_sha256: str
    split_manifest_path: Path
    split_manifest_sha256: str


@dataclass(frozen=True)
class _FormalIndexShardTask:
    config: V1IndexConfig
    episodes: tuple[V1EpisodeRecord, ...]
    ordinal_start: int


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FormalV1ContractError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> Any:
    raise FormalV1ContractError(f"non-finite JSON constant is forbidden: {value}")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FormalV1ContractError(f"value is not canonical JSON: {exc}") from exc


def _sha256_json(value: Any, *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(b"\0")
    digest.update(_canonical_json(value).encode("utf-8"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_worker_count(workers: int) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise FormalV1ContractError("workers must be a positive integer")
    return workers


def _run_ordered_tasks(
    tasks: Sequence[_TaskT],
    *,
    worker: Callable[[_TaskT], _ResultT],
    workers: int,
    phase: str,
    progress_callback: ProgressCallback | None,
    process_start_method: str | None = None,
    result_progress_callback: Callable[[_ResultT, int, int], None] | None = None,
) -> list[_ResultT]:
    """Run independent tasks while preserving deterministic input order.

    At most ``2 * workers`` futures are submitted at once.  This bounds parent
    memory for large Formal inventories and keeps fail-closed cancellation
    responsive.  No output artifact is created by this helper.
    """

    resolved_workers = _positive_worker_count(workers)
    total = len(tasks)
    if progress_callback is not None:
        progress_callback(phase, 0, total)
    if resolved_workers == 1:
        serial_results: list[_ResultT] = []
        for completed, task in enumerate(tasks, start=1):
            result = worker(task)
            serial_results.append(result)
            if result_progress_callback is not None:
                result_progress_callback(result, completed, total)
            if progress_callback is not None:
                progress_callback(phase, completed, total)
        return serial_results
    if total == 0:
        return []

    missing = object()
    results: list[Any] = [missing] * total
    pending: dict[Future[_ResultT], int] = {}
    next_index = 0
    completed = 0
    executor_kwargs: dict[str, Any] = {"max_workers": resolved_workers}
    if process_start_method is not None:
        executor_kwargs["mp_context"] = get_context(process_start_method)
    executor = ProcessPoolExecutor(**executor_kwargs)

    def submit_until_full() -> None:
        nonlocal next_index
        maximum_pending = resolved_workers * 2
        while next_index < total and len(pending) < maximum_pending:
            future = executor.submit(worker, tasks[next_index])
            pending[future] = next_index
            next_index += 1

    try:
        submit_until_full()
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: pending[item]):
                result_index = pending.pop(future)
                result = future.result()
                results[result_index] = result
                completed += 1
                if result_progress_callback is not None:
                    result_progress_callback(result, completed, total)
                if progress_callback is not None:
                    progress_callback(phase, completed, total)
            submit_until_full()
    except BaseException:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    if any(result is missing for result in results):
        raise AssertionError("parallel Formal task result is incomplete")
    return list(results)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _strict_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = _regular_file(path, label=label)
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        FormalV1ContractError,
    ) as exc:
        raise FormalV1ContractError(
            f"cannot read strict {label} JSON {resolved}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise FormalV1ContractError(f"{label} root must be one object: {resolved}")
    return value


def _strict_jsonl(path: str | Path, *, label: str) -> list[dict[str, Any]]:
    resolved = _regular_file(path, label=label)
    result: list[dict[str, Any]] = []
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise FormalV1ContractError(f"cannot read {label} {resolved}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_strict_object_pairs,
                parse_constant=_reject_nonfinite_json,
            )
        except (json.JSONDecodeError, FormalV1ContractError) as exc:
            raise FormalV1ContractError(
                f"invalid {label} line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise FormalV1ContractError(
                f"{label} line {line_number} must be one object"
            )
        result.append(value)
    if not result:
        raise FormalV1ContractError(f"{label} must contain at least one record")
    return result


def _regular_file(path: str | Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise FormalV1ContractError(f"{label} path must be absolute: {raw}")
    if raw.is_symlink() or not raw.is_file():
        raise FormalV1ContractError(
            f"{label} must be a regular non-symlink file: {raw}"
        )
    return raw.resolve()


def _safe_relative(root: Path, raw_path: Any, *, label: str) -> Path:
    relative = Path(str(raw_path))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise FormalV1ContractError(f"{label} is not a safe relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise FormalV1ContractError(f"{label} escapes its artifact root") from exc
    return _regular_file(candidate, label=label)


def _required_text(value: Mapping[str, Any], name: str, *, label: str) -> str:
    text = value.get(name)
    if not isinstance(text, str) or not text.strip():
        raise FormalV1ContractError(f"{label}.{name} must be a non-empty string")
    return text.strip()


def _required_int(
    value: Mapping[str, Any],
    name: str,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    result = value.get(name)
    if isinstance(result, bool) or not isinstance(result, int) or result < minimum:
        raise FormalV1ContractError(
            f"{label}.{name} must be an integer >= {minimum}"
        )
    return int(result)


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    label: str,
) -> None:
    observed = set(value)
    missing = sorted(required - observed)
    extra = sorted(observed - required - optional)
    if missing or extra:
        raise FormalV1ContractError(
            f"{label} keys differ: missing={missing} extra={extra}"
        )


def _semantic_digest(
    value: Mapping[str, Any],
    *,
    digest_field: str,
    domain: bytes,
) -> str:
    return _sha256_json(
        {key: item for key, item in value.items() if key != digest_field},
        domain=domain,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create_or_verify_text(path: Path, content: str) -> None:
    """Publish immutable text without ever opening the final path for writes."""

    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.parent.resolve() / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")

    def verify_existing() -> None:
        if path.is_symlink() or not path.is_file():
            raise FormalV1ContractError(
                f"immutable output is not a regular file: {path}"
            )
        if path.read_bytes() != encoded:
            raise FormalV1ContractError(
                f"immutable output already exists with different bytes: {path}"
            )

    if path.exists() or path.is_symlink():
        verify_existing()
        return

    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(16)}"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            verify_existing()
        else:
            _fsync_directory(path.parent)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
            _fsync_directory(path.parent)


def _atomic_create_or_verify(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_create_or_verify_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _validate_universe_manifest(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
) -> tuple[dict[str, Any], tuple[str, ...], Path, str]:
    resolved = _regular_file(path, label="fleet universe manifest")
    file_sha256 = _sha256_file(resolved)
    if expected_file_sha256 is not None and file_sha256 != expected_file_sha256:
        raise FormalV1ContractError("fleet universe manifest file SHA-256 mismatch")
    manifest = _strict_json_object(resolved, label="fleet universe manifest")
    if manifest.get("schema_version") != FLEET_MANIFEST_SCHEMA:
        raise FormalV1ContractError("unsupported fleet universe manifest schema")
    if (
        manifest.get("status") != "READY_FOR_VERSIONED_V1_GENERATION"
        or manifest.get("dataset_role") != "versioned_v1_phase1_staging"
        or manifest.get("formal_index_eligible") is not False
        or manifest.get("formal_publication_authorized") is not False
    ):
        raise FormalV1ContractError("fleet universe manifest policy is not staging-only")
    claimed = manifest.get("manifest_sha256")
    actual = _semantic_digest(
        manifest,
        digest_field="manifest_sha256",
        domain=_FLEET_MANIFEST_HASH_DOMAIN,
    )
    if not _is_sha256(claimed) or claimed != actual:
        raise FormalV1ContractError("fleet universe manifest semantic SHA mismatch")
    partitions = manifest.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        raise FormalV1ContractError("fleet universe manifest has no partitions")
    target_ids: list[str] = []
    for index, partition in enumerate(partitions):
        if not isinstance(partition, Mapping):
            raise FormalV1ContractError(f"fleet partition {index} is not an object")
        raw_ids = partition.get("target_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise FormalV1ContractError(f"fleet partition {index} has no target IDs")
        for raw_id in raw_ids:
            if not isinstance(raw_id, str) or not raw_id.startswith("target_"):
                raise FormalV1ContractError(
                    f"fleet partition {index} has malformed target ID"
                )
            target_ids.append(raw_id)
    if len(target_ids) != len(set(target_ids)):
        raise FormalV1ContractError("fleet universe contains duplicate target IDs")
    target_count = _required_int(
        manifest, "target_count", label="fleet universe manifest", minimum=1
    )
    if len(target_ids) != target_count:
        raise FormalV1ContractError(
            "fleet universe target_count differs from explicit target IDs"
        )
    geometry = manifest.get("geometry_contract")
    if not isinstance(geometry, Mapping) or (
        geometry.get("raw_episode_schema_version") != RAW_V1_SCHEMA
        or geometry.get("geometry_encoding")
        != "depth_float32_plus_explicit_ray_xy_float32_v1"
        or geometry.get("position_reconstruction")
        != "camera_ray_float32_times_depth_float32"
    ):
        raise FormalV1ContractError("fleet universe geometry contract is incompatible")
    geometry_sha256 = manifest.get("geometry_contract_sha256")
    if not _is_sha256(geometry_sha256):
        raise FormalV1ContractError(
            "fleet universe geometry contract SHA-256 is malformed"
        )
    return manifest, tuple(sorted(target_ids)), resolved, file_sha256


def _receipt_artifact_paths(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    universe_root: Path,
    target_id: str,
) -> tuple[Path, str, Path, str, Mapping[str, Any]]:
    schema = receipt.get("schema_version")
    if schema not in ALLOWED_SOURCE_RECEIPT_SCHEMAS:
        raise FormalV1ContractError(
            f"unsupported source target receipt schema: {schema!r}"
        )
    identity = receipt.get("target_identity")
    if not isinstance(identity, Mapping) or identity.get("target_id") != target_id:
        raise FormalV1ContractError("source target receipt identity mismatch")
    if (
        receipt.get("passed") is not True
        or receipt.get("dataset_role") != "versioned_v1_phase1_staging"
        or receipt.get("formal_publication_authorized") is not False
    ):
        raise FormalV1ContractError("source target receipt is not verified staging data")

    if schema == NATIVE_TARGET_RECEIPT_SCHEMA:
        if receipt.get("status") != "SUCCESS":
            raise FormalV1ContractError("native target receipt is not SUCCESS")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise FormalV1ContractError("native target receipt has no artifacts")
        episode_path = _safe_relative(
            universe_root,
            artifacts.get("episode_hdf5"),
            label="native episode HDF5",
        )
        report_path = _safe_relative(
            universe_root,
            artifacts.get("candidate_report"),
            label="native candidate report",
        )
        episode_sha256 = artifacts.get("episode_hdf5_sha256")
        report_sha256 = artifacts.get("candidate_report_sha256")
    else:
        if (
            receipt.get("status") != "IMPORTED_VERIFIED"
            or receipt.get("candidate_only") is not True
            or receipt.get("formal_index_eligible") is not False
            or receipt.get("external_artifact_reference") is not True
            or receipt.get("source_retention_required") is not True
        ):
            raise FormalV1ContractError("imported target receipt policy mismatch")
        source = receipt.get("source_generation")
        closure = receipt.get("source_artifact_closure")
        if not isinstance(source, Mapping) or not isinstance(closure, list):
            raise FormalV1ContractError("imported target receipt lacks source closure")
        source_root_raw = source.get("root")
        source_root = Path(str(source_root_raw)).expanduser()
        if not source_root.is_absolute() or not source_root.is_dir():
            raise FormalV1ContractError("imported source root is unavailable")
        by_name: dict[str, Mapping[str, Any]] = {}
        for raw in closure:
            if not isinstance(raw, Mapping):
                raise FormalV1ContractError("imported source closure is malformed")
            name = _required_text(raw, "name", label="source closure record")
            if name in by_name:
                raise FormalV1ContractError(
                    f"duplicate imported source closure artifact: {name}"
                )
            by_name[name] = raw
        try:
            episode = by_name["episode_hdf5"]
            report = by_name["candidate_report"]
        except KeyError as exc:
            raise FormalV1ContractError(
                "imported source closure lacks HDF5 or candidate report"
            ) from exc
        episode_path = _safe_relative(
            source_root.resolve(),
            episode.get("source_relative_path"),
            label="imported episode HDF5",
        )
        report_path = _safe_relative(
            source_root.resolve(),
            report.get("source_relative_path"),
            label="imported candidate report",
        )
        episode_sha256 = episode.get("sha256")
        report_sha256 = report.get("sha256")

    if not _is_sha256(episode_sha256) or not _is_sha256(report_sha256):
        raise FormalV1ContractError("source target receipt artifact SHA is malformed")
    if receipt.get("frame_count") is None:
        raise FormalV1ContractError("source target receipt lacks frame_count")
    return (
        episode_path,
        str(episode_sha256),
        report_path,
        str(report_sha256),
        identity,
    )


def _validated_saved_frame_timeline_identity_sha256(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    frame_count: int,
) -> str:
    timeline = report.get("saved_frame_timeline_identity")
    if not isinstance(timeline, Mapping):
        raise FormalV1ContractError(
            f"candidate report v2 lacks timeline identity: {report_path}"
        )
    _require_exact_keys(
        timeline,
        required={
            "schema_version",
            "frame_count",
            "control_stream_count",
            "valid_action_frame_count",
            "first_valid_action_frame_index",
            "terminal_sim_step",
            "saved_frame_timeline_sha256",
            "control_stream_contract_sha256",
            "control_stream_sha256",
            "saved_frame_action_span_sha256",
            "identity_sha256",
        },
        label="candidate report v2 timeline identity",
    )
    control_count = _required_int(
        timeline,
        "control_stream_count",
        label="candidate report v2 timeline identity",
        minimum=1,
    )
    valid_action_count = _required_int(
        timeline,
        "valid_action_frame_count",
        label="candidate report v2 timeline identity",
        minimum=1,
    )
    first_valid = _required_int(
        timeline,
        "first_valid_action_frame_index",
        label="candidate report v2 timeline identity",
    )
    terminal_sim_step = _required_int(
        timeline,
        "terminal_sim_step",
        label="candidate report v2 timeline identity",
        minimum=1,
    )
    if (
        timeline.get("schema_version")
        != "robotwin_momap_saved_frame_timeline_identity_v1"
        or timeline.get("frame_count") != frame_count
        or first_valid >= frame_count
        or valid_action_count != frame_count - first_valid
        or terminal_sim_step > control_count
    ):
        raise FormalV1ContractError(
            f"candidate report v2 timeline identity mismatch: {report_path}"
        )
    for name in (
        "saved_frame_timeline_sha256",
        "control_stream_contract_sha256",
        "control_stream_sha256",
        "saved_frame_action_span_sha256",
        "identity_sha256",
    ):
        if not _is_sha256(timeline.get(name)):
            raise FormalV1ContractError(
                f"candidate report v2 timeline {name} is malformed: {report_path}"
            )
    claimed = str(timeline["identity_sha256"])
    actual = _semantic_digest(
        timeline,
        digest_field="identity_sha256",
        domain=b"robotwin-momap-saved-frame-timeline-identity-v1",
    )
    if claimed != actual:
        raise FormalV1ContractError(
            f"candidate report v2 timeline identity SHA mismatch: {report_path}"
        )
    return claimed


def _validate_source_candidate_report(
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
) -> str | None:
    """Validate both historical v1 and current Fleet v2 candidate reports."""

    schema = report.get("schema_version")
    if schema == "robotwin_momap_canonical_rollout_v1_candidate_report_v1":
        _validate_candidate_report(
            report,
            report_path=report_path,
            episode_path=episode_path,
            episode_sha256=episode_sha256,
            episode_id=episode_id,
            frame_count=frame_count,
            seed=seed,
            canonical_trajectory_sha256=canonical_trajectory_sha256,
            planning_scene_sha256=planning_scene_sha256,
            replay_scene_sha256=replay_scene_sha256,
        )
        return None
    if schema != "robotwin_momap_canonical_rollout_v1_candidate_report_v2":
        raise FormalV1ContractError(
            f"unsupported candidate report schema: {report_path}"
        )
    if (
        report.get("status") != "PASS"
        or report.get("candidate_only") is not True
        or report.get("production_authorized") is not False
        or report.get("formal_index_eligible") is not False
        or report.get("publication_stage") != "CANDIDATE_VALIDATION_ONLY"
        or report.get("raw_episode_schema_version") != RAW_V1_SCHEMA
        or report.get("geometry_encoding")
        != "depth_float32_plus_explicit_ray_xy_float32_v1"
        or report.get("alignment_contract") != "same_replay_capture_only"
        or report.get("official_frame_alignment_claimed") is not False
        or tuple(report.get("camera_names", ())) != PHYSICAL_CAMERA_NAMES
        or report.get("frame_count") != frame_count
    ):
        raise FormalV1ContractError(
            f"candidate report v2 policy/shape mismatch: {report_path}"
        )
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
        raise FormalV1ContractError(
            f"candidate report v2 rollout identity mismatch: {report_path}"
        )
    gates = report.get("gates")
    required_true_gates = {
        "action_timeline_aligned",
        "formal_markers_absent",
        "hdf5_complete",
        "identity_artifact_hash_bound",
        "same_replay_capture",
        "saved_frame_timeline_identity_hash_bound",
        "trajectory_hash_bound",
        "v1_all_valid_vectors_checked",
        "v1_physical_position_xyz_absent",
        "v1_three_camera_layout",
    }
    if not isinstance(gates, Mapping) or any(
        gates.get(name) is not True for name in required_true_gates
    ):
        raise FormalV1ContractError(
            f"candidate report v2 required gates did not pass: {report_path}"
        )
    geometry = report.get("geometry_validation")
    if not isinstance(geometry, Mapping) or (
        geometry.get("all_valid_vectors_checked") is not True
        or geometry.get("physical_position_xyz_absent") is not True
        or geometry.get("validated_vector_count")
        != geometry.get("valid_vector_count")
    ):
        raise FormalV1ContractError(
            f"candidate report v2 geometry validation mismatch: {report_path}"
        )
    timeline_sha256 = _validated_saved_frame_timeline_identity_sha256(
        report,
        report_path=report_path,
        frame_count=frame_count,
    )
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FormalV1ContractError(
            f"candidate report v2 has no artifacts: {report_path}"
        )
    hdf5 = artifacts.get("episode_hdf5")
    trajectory = artifacts.get("canonical_trajectory")
    if not isinstance(hdf5, Mapping) or not isinstance(trajectory, Mapping):
        raise FormalV1ContractError(
            f"candidate report v2 lacks required artifacts: {report_path}"
        )
    artifact_path = Path(str(hdf5.get("path", ""))).expanduser().resolve()
    if (
        artifact_path != episode_path
        or hdf5.get("sha256") != episode_sha256
        or trajectory.get("sha256") != canonical_trajectory_sha256
    ):
        raise FormalV1ContractError(
            f"candidate report v2 artifact binding mismatch: {report_path}"
        )
    return timeline_sha256


def _record_to_v1_episode(
    record: Mapping[str, Any],
    *,
    source_manifest: Path,
    candidate_report_payload: Mapping[str, Any] | None = None,
) -> V1EpisodeRecord:
    identity = record.get("identity")
    hdf5 = record.get("hdf5")
    candidate_report = record.get("candidate_report")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(hdf5, Mapping)
        or not isinstance(candidate_report, Mapping)
    ):
        raise FormalV1ContractError("formal episode record is malformed")
    manifest_timeline_sha256 = identity.get(
        "saved_frame_timeline_identity_sha256"
    )
    runtime_timeline_sha256 = manifest_timeline_sha256
    if candidate_report_payload is not None:
        report_timeline_sha256 = _validate_source_candidate_report(
            candidate_report_payload,
            report_path=Path(str(candidate_report["path"])),
            episode_path=Path(str(hdf5["path"])),
            episode_sha256=str(hdf5["sha256"]),
            episode_id=str(identity["canonical_rollout_id"]),
            frame_count=int(record["frame_count"]),
            seed=int(identity["seed"]),
            canonical_trajectory_sha256=str(
                identity["canonical_trajectory_sha256"]
            ),
            planning_scene_sha256=str(identity["planning_scene_sha256"]),
            replay_scene_sha256=str(identity["replay_scene_sha256"]),
        )
        if report_timeline_sha256 is None:
            raise FormalV1ContractError(
                "Formal v2 candidate report lacks timeline identity"
            )
        if (
            manifest_timeline_sha256 is not None
            and manifest_timeline_sha256 != report_timeline_sha256
        ):
            raise FormalV1ContractError(
                "Formal manifest/candidate report timeline identity mismatch"
            )
        runtime_timeline_sha256 = report_timeline_sha256
    return V1EpisodeRecord(
        episode_path=Path(str(hdf5["path"])),
        episode_id=str(identity["canonical_rollout_id"]),
        task_name=str(identity["task_name"]),
        task_config=str(identity["task_config"]),
        variant=str(identity["variant"]),
        source_manifest=source_manifest,
        expected_hdf5_sha256=str(hdf5["sha256"]),
        candidate_report=Path(str(candidate_report["path"])),
        candidate_report_sha256=str(candidate_report["sha256"]),
        expected_frame_count=int(record["frame_count"]),
        scope_id=str(identity["scope_id"]),
        candidate_ordinal=int(identity["global_index"]) + 1,
        seed=int(identity["seed"]),
        expected_episode_number=int(identity["canonical_episode_id"]),
        canonical_trajectory_sha256=str(
            identity["canonical_trajectory_sha256"]
        ),
        planning_scene_sha256=str(identity["planning_scene_sha256"]),
        replay_scene_sha256=str(identity["replay_scene_sha256"]),
        saved_frame_timeline_identity_sha256=(
            str(runtime_timeline_sha256)
            if runtime_timeline_sha256 is not None
            else None
        ),
    )


def _episode_record_from_source_receipt(
    *,
    target_id: str,
    source_receipt_path: Path,
    universe_root: Path,
    geometry_contract_sha256: str,
    reader_factory: ReaderFactory | None = None,
) -> dict[str, Any]:
    source_receipt_path = _regular_file(
        source_receipt_path, label=f"source receipt for {target_id}"
    )
    source_receipt_sha256 = _sha256_file(source_receipt_path)
    receipt = _strict_json_object(source_receipt_path, label="source target receipt")
    if receipt.get("geometry_contract_sha256") != geometry_contract_sha256:
        raise FormalV1ContractError(
            f"source target geometry contract mismatch: {target_id}"
        )
    (
        episode_path,
        expected_episode_sha256,
        candidate_report_path,
        expected_report_sha256,
        target_identity,
    ) = _receipt_artifact_paths(
        receipt,
        receipt_path=source_receipt_path,
        universe_root=universe_root,
        target_id=target_id,
    )
    observed_episode_sha256 = _sha256_file(episode_path)
    if observed_episode_sha256 != expected_episode_sha256:
        raise FormalV1ContractError(f"episode HDF5 SHA mismatch: {target_id}")
    observed_report_sha256 = _sha256_file(candidate_report_path)
    if observed_report_sha256 != expected_report_sha256:
        raise FormalV1ContractError(f"candidate report SHA mismatch: {target_id}")
    report = _strict_json_object(
        candidate_report_path, label="candidate episode report"
    )

    factory = reader_factory or _default_reader_factory
    reader = factory(episode_path)
    try:
        if reader.schema_version != RAW_V1_SCHEMA:
            raise FormalV1ContractError(
                f"raw episode schema mismatch for {target_id}"
            )
        if tuple(reader.camera_names) != PHYSICAL_CAMERA_NAMES:
            raise FormalV1ContractError(
                f"physical camera order mismatch for {target_id}"
            )
        if not reader.has_control_stream:
            raise FormalV1ContractError(
                f"episode has no complete control stream: {target_id}"
            )
        _validate_control_stream_contract(reader)
        metadata = reader.metadata
        if not isinstance(metadata, Mapping):
            raise FormalV1ContractError(
                f"episode metadata is not an object: {target_id}"
            )
        frame_count = int(reader.num_frames)
        task_name = _required_text(
            target_identity, "task_name", label="target identity"
        )
        task_config = _required_text(
            target_identity, "task_config", label="target identity"
        )
        variant = _required_text(
            target_identity, "variant", label="target identity"
        )
        scope_id = _required_text(
            target_identity, "scope_id", label="target identity"
        )
        global_index = _required_int(
            target_identity, "global_index", label="target identity"
        )
        canonical_episode_id = _required_int(
            target_identity, "episode_id", label="target identity"
        )
        seed = _required_int(receipt, "winning_seed", label="source target receipt")
        if (
            scope_id != f"{task_name}:{variant}"
            or metadata.get("task_name") != task_name
            or metadata.get("task_config") != task_config
            or metadata.get("canonical_episode_id") != canonical_episode_id
            or metadata.get("seed") != seed
            or receipt.get("frame_count") != frame_count
        ):
            raise FormalV1ContractError(
                f"target receipt/HDF5 identity mismatch: {target_id}"
            )
        expected_variant = {
            "demo_clean": "clean",
            "demo_randomized": "randomized",
        }.get(task_config)
        if variant != expected_variant:
            raise FormalV1ContractError(
                f"target task_config/variant mismatch: {target_id}"
            )
        canonical_identity = metadata.get("canonical_rollout_identity")
        if not isinstance(canonical_identity, Mapping):
            raise FormalV1ContractError(
                f"HDF5 lacks canonical rollout identity: {target_id}"
            )
        canonical_rollout_id = _required_text(
            metadata, "rollout_id", label="HDF5 metadata"
        )
        trajectory_sha256 = _required_text(
            metadata, "trajectory_artifact_sha256", label="HDF5 metadata"
        )
        planning_scene_sha256 = _required_text(
            canonical_identity,
            "planning_scene_sha256",
            label="canonical rollout identity",
        )
        replay_scene_sha256 = _required_text(
            canonical_identity,
            "replay_scene_sha256",
            label="canonical rollout identity",
        )
        for name, digest in (
            ("canonical trajectory", trajectory_sha256),
            ("planning scene", planning_scene_sha256),
            ("replay scene", replay_scene_sha256),
        ):
            if not _is_sha256(digest):
                raise FormalV1ContractError(
                    f"{name} SHA is malformed for {target_id}"
                )
        report_timeline_sha256 = _validate_source_candidate_report(
            report,
            report_path=candidate_report_path,
            episode_path=episode_path,
            episode_sha256=observed_episode_sha256,
            episode_id=canonical_rollout_id,
            frame_count=frame_count,
            seed=seed,
            canonical_trajectory_sha256=trajectory_sha256,
            planning_scene_sha256=planning_scene_sha256,
            replay_scene_sha256=replay_scene_sha256,
        )
        metadata_timeline_sha256 = metadata.get(
            "saved_frame_timeline_identity_sha256"
        )
        if metadata_timeline_sha256 is not None and not _is_sha256(
            metadata_timeline_sha256
        ):
            raise FormalV1ContractError(
                f"saved-frame timeline SHA is malformed for {target_id}"
            )
        if (
            report_timeline_sha256 is not None
            and metadata_timeline_sha256 is not None
            and report_timeline_sha256 != metadata_timeline_sha256
        ):
            raise FormalV1ContractError(
                f"candidate report/HDF5 timeline identity mismatch: {target_id}"
            )
        timeline_sha256 = report_timeline_sha256 or metadata_timeline_sha256
        identity = {
            "canonical_rollout_id": canonical_rollout_id,
            "task_name": task_name,
            "task_config": task_config,
            "variant": variant,
            "scope_id": scope_id,
            "global_index": global_index,
            "canonical_episode_id": canonical_episode_id,
            "seed": seed,
            "canonical_trajectory_sha256": trajectory_sha256,
            "planning_scene_sha256": planning_scene_sha256,
            "replay_scene_sha256": replay_scene_sha256,
            "saved_frame_timeline_identity_sha256": timeline_sha256,
        }
        provisional = V1EpisodeRecord(
            episode_path=episode_path,
            episode_id=canonical_rollout_id,
            task_name=task_name,
            task_config=task_config,
            variant=variant,
            source_manifest=source_receipt_path,
            expected_hdf5_sha256=observed_episode_sha256,
            candidate_report=candidate_report_path,
            candidate_report_sha256=observed_report_sha256,
            expected_frame_count=frame_count,
            scope_id=scope_id,
            candidate_ordinal=global_index + 1,
            seed=seed,
            expected_episode_number=canonical_episode_id,
            canonical_trajectory_sha256=trajectory_sha256,
            planning_scene_sha256=planning_scene_sha256,
            replay_scene_sha256=replay_scene_sha256,
            saved_frame_timeline_identity_sha256=(
                None if timeline_sha256 is None else str(timeline_sha256)
            ),
        )
        _validate_record_metadata(provisional, metadata, frame_count=frame_count)
    finally:
        reader.close()

    split_group_id = _sha256_json(
        {
            "canonical_rollout_id": identity["canonical_rollout_id"],
            "scope_id": identity["scope_id"],
            "canonical_episode_id": identity["canonical_episode_id"],
            "seed": identity["seed"],
        },
        domain=b"robotwin-momap-formal-v1-episode-group-v1",
    )
    return {
        "target_id": target_id,
        "source_receipt": {
            "path": str(source_receipt_path),
            "sha256": source_receipt_sha256,
            "schema_version": receipt["schema_version"],
        },
        "candidate_report": {
            "path": str(candidate_report_path),
            "sha256": observed_report_sha256,
            "schema_version": report["schema_version"],
        },
        "hdf5": {
            "path": str(episode_path),
            "sha256": observed_episode_sha256,
            "size_bytes": episode_path.stat().st_size,
            "schema_version": RAW_V1_SCHEMA,
        },
        "identity": identity,
        "frame_count": frame_count,
        "geometry_contract_sha256": geometry_contract_sha256,
        "split_group_id": split_group_id,
    }


_BuildEpisodeTask = tuple[str, str, str, str]


def _build_episode_record_default_reader(
    task: _BuildEpisodeTask,
) -> dict[str, Any]:
    """Process-worker entry point for one independently stored episode."""

    target_id, source_receipt, universe_root, geometry_sha256 = task
    return _episode_record_from_source_receipt(
        target_id=target_id,
        source_receipt_path=Path(source_receipt),
        universe_root=Path(universe_root),
        geometry_contract_sha256=geometry_sha256,
    )


def _build_exclusion_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        raw,
        required={"target_id", "reason_code", "decision_id", "evidence"},
        optional={"note"},
        label="exclusion JSONL record",
    )
    target_id = _required_text(raw, "target_id", label="exclusion JSONL record")
    reason_code = _required_text(
        raw, "reason_code", label="exclusion JSONL record"
    )
    decision_id = _required_text(
        raw, "decision_id", label="exclusion JSONL record"
    )
    if reason_code not in ALLOWED_EXCLUSION_REASON_CODES:
        raise FormalV1ContractError(
            f"unsupported exclusion reason_code for {target_id}: {reason_code}"
        )
    raw_evidence = raw.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise FormalV1ContractError(
            f"exclusion {target_id} must have non-empty evidence"
        )
    evidence: list[dict[str, Any]] = []
    for index, value in enumerate(raw_evidence):
        path = _regular_file(
            str(value), label=f"exclusion {target_id} evidence {index}"
        )
        evidence.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    result: dict[str, Any] = {
        "target_id": target_id,
        "reason_code": reason_code,
        "decision_id": decision_id,
        "evidence": sorted(evidence, key=lambda item: item["path"]),
    }
    if "note" in raw:
        note = raw["note"]
        if not isinstance(note, str):
            raise FormalV1ContractError(
                f"exclusion {target_id} note must be a string"
            )
        result["note"] = note
    return result


def _episode_set_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_json(
        sorted((dict(record) for record in records), key=lambda row: row["target_id"]),
        domain=_EPISODE_SET_HASH_DOMAIN,
    )


def _exclusion_set_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_json(
        sorted((dict(record) for record in records), key=lambda row: row["target_id"]),
        domain=_EXCLUSION_SET_HASH_DOMAIN,
    )


def build_formal_source_inventory(
    *,
    universe_manifest_path: str | Path,
    episode_receipts_jsonl: str | Path,
    exclusions_jsonl: str | Path | None,
    output_path: str | Path,
    reader_factory: ReaderFactory | None = None,
    workers: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Build a self-hashed, exact-cover inventory from explicit JSONL inputs.

    Episode JSONL lines have exactly ``{"target_id", "source_receipt"}``.
    Exclusion JSONL lines have ``target_id``, a versioned ``reason_code``, a
    non-empty ``decision_id`` and one or more explicit evidence paths.  No data
    directory is globbed.
    """

    manifest, universe_ids, universe_path, universe_file_sha256 = (
        _validate_universe_manifest(universe_manifest_path)
    )
    requested_workers = _positive_worker_count(workers)
    episode_lines = _strict_jsonl(
        episode_receipts_jsonl, label="episode receipt JSONL"
    )
    geometry_sha256 = str(manifest["geometry_contract_sha256"])
    episode_inputs: list[tuple[str, Path]] = []
    for index, raw in enumerate(episode_lines):
        _require_exact_keys(
            raw,
            required={"target_id", "source_receipt"},
            label=f"episode receipt JSONL record {index}",
        )
        target_id = _required_text(
            raw, "target_id", label=f"episode receipt JSONL record {index}"
        )
        source_receipt = _regular_file(
            _required_text(
                raw,
                "source_receipt",
                label=f"episode receipt JSONL record {index}",
            ),
            label=f"episode source receipt {target_id}",
        )
        episode_inputs.append((target_id, source_receipt))

    exclusions: list[dict[str, Any]] = []
    if exclusions_jsonl is not None:
        exclusions = [
            _build_exclusion_record(raw)
            for raw in _strict_jsonl(
                exclusions_jsonl, label="formal exclusion JSONL"
            )
        ]
    included_ids = [target_id for target_id, _ in episode_inputs]
    excluded_ids = [str(record["target_id"]) for record in exclusions]
    if len(included_ids) != len(set(included_ids)):
        raise FormalV1ContractError(
            "source inventory has duplicate included target IDs"
        )
    if len(excluded_ids) != len(set(excluded_ids)):
        raise FormalV1ContractError(
            "source inventory has duplicate excluded target IDs"
        )
    overlap = sorted(set(included_ids) & set(excluded_ids))
    expected = set(universe_ids)
    observed = set(included_ids) | set(excluded_ids)
    if overlap or observed != expected:
        raise FormalV1ContractError(
            "source inventory does not exactly close its target universe: "
            f"overlap={overlap[:10]} missing={sorted(expected-observed)[:10]} "
            f"extra={sorted(observed-expected)[:10]}"
        )

    # Production readers are process-safe only because every task owns one
    # independent HDF5 file.  Resolve receipt locators up front so duplicate
    # files fail before any large payload is hashed.  ReaderFactory injection
    # intentionally skips this extra parse and remains serial for tests.
    if reader_factory is None:
        preflight_episode_paths: list[str] = []
        if progress_callback is not None:
            progress_callback(
                "source_inventory_receipt_preflight", 0, len(episode_inputs)
            )
        for completed, (target_id, source_receipt) in enumerate(
            episode_inputs, start=1
        ):
            receipt = _strict_json_object(
                source_receipt, label="source target receipt"
            )
            if receipt.get("geometry_contract_sha256") != geometry_sha256:
                raise FormalV1ContractError(
                    f"source target geometry contract mismatch: {target_id}"
                )
            episode_path, _, _, _, _ = _receipt_artifact_paths(
                receipt,
                receipt_path=source_receipt,
                universe_root=universe_path.parent,
                target_id=target_id,
            )
            preflight_episode_paths.append(str(episode_path))
            if progress_callback is not None:
                progress_callback(
                    "source_inventory_receipt_preflight",
                    completed,
                    len(episode_inputs),
                )
        if len(preflight_episode_paths) != len(set(preflight_episode_paths)):
            raise FormalV1ContractError(
                "source inventory contains duplicate HDF5 paths"
            )

    if reader_factory is None:
        tasks: list[_BuildEpisodeTask] = [
            (
                target_id,
                str(source_receipt),
                str(universe_path.parent),
                geometry_sha256,
            )
            for target_id, source_receipt in episode_inputs
        ]
        episodes = _run_ordered_tasks(
            tasks,
            worker=_build_episode_record_default_reader,
            workers=requested_workers,
            phase="source_inventory_episode_verification",
            progress_callback=progress_callback,
        )
    else:
        # Custom reader factories are commonly stateful test doubles and are
        # not assumed to be pickle-safe.  Preserve the injected serial path.
        episodes = _run_ordered_tasks(
            episode_inputs,
            worker=lambda item: _episode_record_from_source_receipt(
                target_id=item[0],
                source_receipt_path=item[1],
                universe_root=universe_path.parent,
                geometry_contract_sha256=geometry_sha256,
                reader_factory=reader_factory,
            ),
            workers=1,
            phase="source_inventory_episode_verification",
            progress_callback=progress_callback,
        )

    episode_paths = [str(record["hdf5"]["path"]) for record in episodes]
    rollout_ids = [
        str(record["identity"]["canonical_rollout_id"]) for record in episodes
    ]
    if len(episode_paths) != len(set(episode_paths)):
        raise FormalV1ContractError("source inventory contains duplicate HDF5 paths")
    if len(rollout_ids) != len(set(rollout_ids)):
        raise FormalV1ContractError(
            "source inventory contains duplicate canonical rollout identities"
        )

    episodes.sort(key=lambda item: item["target_id"])
    exclusions.sort(key=lambda item: item["target_id"])
    inventory: dict[str, Any] = {
        "schema_version": SOURCE_INVENTORY_SCHEMA,
        "status": "READY_FOR_FORMAL_VERIFICATION",
        "candidate_only": True,
        "formal_index_eligible": False,
        "training_authorized": False,
        "production_authorized": False,
        "universe": {
            "path": str(universe_path),
            "file_sha256": universe_file_sha256,
            "semantic_sha256": manifest["manifest_sha256"],
            "schema_version": manifest["schema_version"],
            "target_count": len(universe_ids),
            "target_identity_sha256": manifest["target_identity_sha256"],
            "geometry_contract_sha256": geometry_sha256,
        },
        "expected_target_count": len(universe_ids),
        "included_episode_count": len(episodes),
        "excluded_target_count": len(exclusions),
        "episodes": episodes,
        "exclusions": exclusions,
        "episode_record_set_sha256": _episode_set_sha256(episodes),
        "exclusion_record_set_sha256": _exclusion_set_sha256(exclusions),
    }
    inventory["inventory_sha256"] = _semantic_digest(
        inventory,
        digest_field="inventory_sha256",
        domain=_INVENTORY_HASH_DOMAIN,
    )
    destination = Path(output_path).expanduser()
    if not destination.is_absolute():
        raise FormalV1ContractError("source inventory output path must be absolute")
    _atomic_create_or_verify(destination, inventory)
    return destination.resolve()


def _load_source_inventory(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = _regular_file(path, label="Formal V1 source inventory")
    inventory = _strict_json_object(resolved, label="Formal V1 source inventory")
    if inventory.get("schema_version") != SOURCE_INVENTORY_SCHEMA:
        raise FormalV1ContractError("source inventory schema mismatch")
    if (
        inventory.get("status") != "READY_FOR_FORMAL_VERIFICATION"
        or inventory.get("candidate_only") is not True
        or inventory.get("formal_index_eligible") is not False
        or inventory.get("training_authorized") is not False
        or inventory.get("production_authorized") is not False
    ):
        raise FormalV1ContractError("source inventory policy flags mismatch")
    claimed = inventory.get("inventory_sha256")
    actual = _semantic_digest(
        inventory,
        digest_field="inventory_sha256",
        domain=_INVENTORY_HASH_DOMAIN,
    )
    if not _is_sha256(claimed) or claimed != actual:
        raise FormalV1ContractError("source inventory semantic SHA mismatch")
    universe = inventory.get("universe")
    if not isinstance(universe, Mapping):
        raise FormalV1ContractError("source inventory has no universe binding")
    manifest, universe_ids, _, universe_file_sha = _validate_universe_manifest(
        _required_text(universe, "path", label="source inventory universe"),
        expected_file_sha256=_required_text(
            universe, "file_sha256", label="source inventory universe"
        ),
    )
    if (
        universe.get("semantic_sha256") != manifest["manifest_sha256"]
        or universe.get("schema_version") != manifest["schema_version"]
        or universe.get("target_count") != len(universe_ids)
        or universe.get("geometry_contract_sha256")
        != manifest["geometry_contract_sha256"]
        or universe.get("file_sha256") != universe_file_sha
    ):
        raise FormalV1ContractError("source inventory universe binding mismatch")
    episodes = inventory.get("episodes")
    exclusions = inventory.get("exclusions")
    if not isinstance(episodes, list) or not episodes:
        raise FormalV1ContractError("source inventory has no included episodes")
    if not isinstance(exclusions, list):
        raise FormalV1ContractError("source inventory exclusions must be a list")
    if inventory.get("episode_record_set_sha256") != _episode_set_sha256(episodes):
        raise FormalV1ContractError("source inventory episode set SHA mismatch")
    if inventory.get("exclusion_record_set_sha256") != _exclusion_set_sha256(
        exclusions
    ):
        raise FormalV1ContractError("source inventory exclusion set SHA mismatch")
    included_ids = [str(record.get("target_id")) for record in episodes]
    excluded_ids = [str(record.get("target_id")) for record in exclusions]
    observed = set(included_ids) | set(excluded_ids)
    if (
        len(included_ids) != len(set(included_ids))
        or len(excluded_ids) != len(set(excluded_ids))
        or set(included_ids) & set(excluded_ids)
        or observed != set(universe_ids)
        or inventory.get("expected_target_count") != len(universe_ids)
        or inventory.get("included_episode_count") != len(episodes)
        or inventory.get("excluded_target_count") != len(exclusions)
    ):
        raise FormalV1ContractError("source inventory exact-cover counts mismatch")
    return inventory, resolved


def formal_v1_index_config() -> V1IndexConfig:
    """Return the one versioned K=8/action32 Formal V1 temporal contract."""

    return V1IndexConfig(
        future_offsets=FASTWAM_LOGICAL_FUTURE_OFFSETS,
        future_horizons_seconds=FASTWAM_FUTURE_HORIZONS_SECONDS,
        future_offset_basis=FORMAL_FUTURE_OFFSET_BASIS,
        camera_names=PHYSICAL_CAMERA_NAMES,
        action=ActionSamplingConfig(
            mode=FORMAL_ACTION_MODE,
            horizon=FORMAL_ACTION_HORIZON,
            baseline_rate_hz=15.0,
            timestamp_tolerance_sec=1.0e-9,
        ),
        proprio_mode=FORMAL_PROPRIO_MODE,
        coordinate_frame=FORMAL_COORDINATE_FRAME,
        require_raw_v1=True,
        verify_manifest_hashes=True,
        verify_hdf5_hashes=False,
    )


def _temporal_contract(config: V1IndexConfig) -> dict[str, Any]:
    return {
        "future_offsets": list(config.future_offsets),
        "future_horizons_seconds": list(config.future_horizons_seconds or ()),
        "future_offset_basis": config.future_offset_basis,
        "camera_names": list(config.camera_names),
        "action_mode": config.action.mode,
        "action_horizon": config.action.horizon,
        "action_rate_hz": config.action.baseline_rate_hz,
        "action_timestamp_tolerance_sec": config.action.timestamp_tolerance_sec,
        "proprio_mode": config.proprio_mode,
        "coordinate_frame": config.coordinate_frame,
    }


def _validate_exclusion_record(record: Mapping[str, Any]) -> None:
    target_id = _required_text(record, "target_id", label="exclusion record")
    reason_code = _required_text(record, "reason_code", label="exclusion record")
    _required_text(record, "decision_id", label="exclusion record")
    if reason_code not in ALLOWED_EXCLUSION_REASON_CODES:
        raise FormalV1ContractError(
            f"unsupported exclusion reason for {target_id}: {reason_code}"
        )
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise FormalV1ContractError(f"exclusion {target_id} has no evidence")
    for index, raw in enumerate(evidence):
        if not isinstance(raw, Mapping):
            raise FormalV1ContractError(
                f"exclusion {target_id} evidence {index} is malformed"
            )
        path = _regular_file(
            _required_text(
                raw, "path", label=f"exclusion {target_id} evidence {index}"
            ),
            label=f"exclusion {target_id} evidence {index}",
        )
        if (
            _sha256_file(path) != raw.get("sha256")
            or path.stat().st_size != raw.get("size_bytes")
        ):
            raise FormalV1ContractError(
                f"exclusion evidence drift detected: {target_id}"
            )


def _fraction(value: str | float | Fraction, *, name: str) -> Fraction:
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise FormalV1ContractError(f"invalid {name} split fraction") from exc
    if result < 0 or result > 1:
        raise FormalV1ContractError(f"{name} split fraction must be in [0,1]")
    return result


def _group_allocations(
    group_ids: Sequence[str],
    *,
    split_salt: str,
    fractions: Mapping[str, Fraction],
) -> dict[str, str]:
    if not split_salt:
        raise FormalV1ContractError("split_salt must not be empty")
    unique_groups = sorted(set(group_ids))
    if len(unique_groups) != len(group_ids):
        # Multiple records in one episode group are allowed and remain together.
        pass
    positive = [name for name in FORMAL_SPLIT_NAMES if fractions[name] > 0]
    if len(unique_groups) < len(positive):
        raise FormalV1ContractError(
            "not enough episode groups to populate every non-empty split"
        )
    ordered = sorted(
        unique_groups,
        key=lambda group_id: (
            _sha256_json(
                {"split_salt": split_salt, "split_group_id": group_id},
                domain=_SPLIT_ORDER_HASH_DOMAIN,
            ),
            group_id,
        ),
    )
    total = len(ordered)
    exact = {name: fractions[name] * total for name in FORMAL_SPLIT_NAMES}
    counts = {name: int(exact[name]) for name in FORMAL_SPLIT_NAMES}
    remaining = total - sum(counts.values())
    remainder_order = sorted(
        FORMAL_SPLIT_NAMES,
        key=lambda name: (-(exact[name] - counts[name]), FORMAL_SPLIT_NAMES.index(name)),
    )
    for name in remainder_order[:remaining]:
        counts[name] += 1
    for name in positive:
        if counts[name] == 0:
            donor = max(
                (other for other in FORMAL_SPLIT_NAMES if counts[other] > 1),
                key=lambda other: counts[other],
                default=None,
            )
            if donor is None:
                raise FormalV1ContractError(
                    "cannot populate every requested split without group leakage"
                )
            counts[donor] -= 1
            counts[name] += 1
    allocations: dict[str, str] = {}
    cursor = 0
    for name in FORMAL_SPLIT_NAMES:
        for group_id in ordered[cursor : cursor + counts[name]]:
            allocations[group_id] = name
        cursor += counts[name]
    if cursor != total or len(allocations) != total:
        raise AssertionError("internal split allocation count mismatch")
    return allocations


def _scope_stratified_group_allocations(
    records: Sequence[Mapping[str, Any]],
    *,
    split_salt: str,
    fractions: Mapping[str, Fraction],
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Allocate whole episode groups independently inside every scope.

    A global hash split can leave a low-cardinality task/variant scope absent
    from validation or test even when that scope has enough episode groups.
    Formal V1 therefore treats ``identity.scope_id`` as a stratum while still
    assigning the indivisible ``split_group_id`` with the deterministic hash
    recipe implemented by :func:`_group_allocations`.
    """

    groups_by_scope: dict[str, list[str]] = defaultdict(list)
    scope_by_group: dict[str, str] = {}
    for index, record in enumerate(records):
        identity = record.get("identity")
        if not isinstance(identity, Mapping):
            raise FormalV1ContractError(
                f"verified episode {index} has no identity for split stratification"
            )
        scope_id = _required_text(
            identity, "scope_id", label=f"verified episode {index} identity"
        )
        group_id = _required_text(
            record, "split_group_id", label=f"verified episode {index}"
        )
        previous_scope = scope_by_group.setdefault(group_id, scope_id)
        if previous_scope != scope_id:
            raise FormalV1ContractError(
                "one episode group appears in multiple split strata"
            )
        groups_by_scope[scope_id].append(group_id)
    if not groups_by_scope:
        raise FormalV1ContractError("Formal split has no episode scopes")

    allocations: dict[str, str] = {}
    counts_by_scope: dict[str, dict[str, int]] = {}
    for scope_id in sorted(groups_by_scope):
        scoped_allocations = _group_allocations(
            groups_by_scope[scope_id],
            split_salt=split_salt,
            fractions=fractions,
        )
        overlap = set(allocations) & set(scoped_allocations)
        if overlap:
            raise FormalV1ContractError(
                "episode group is duplicated across split strata"
            )
        allocations.update(scoped_allocations)
        counts = {name: 0 for name in FORMAL_SPLIT_NAMES}
        for split_name in scoped_allocations.values():
            counts[split_name] += 1
        counts_by_scope[scope_id] = counts
    return allocations, counts_by_scope


def _validate_split_fractions(
    train_fraction: str | float | Fraction,
    validation_fraction: str | float | Fraction,
    test_fraction: str | float | Fraction,
) -> dict[str, Fraction]:
    fractions = {
        "train": _fraction(train_fraction, name="train"),
        "validation": _fraction(validation_fraction, name="validation"),
        "test": _fraction(test_fraction, name="test"),
    }
    if sum(fractions.values(), Fraction(0)) != 1:
        raise FormalV1ContractError("split fractions must sum exactly to 1")
    return fractions


def _verify_inventory_episode(
    *,
    index: int,
    raw_record: Mapping[str, Any],
    inventory_path: Path,
    universe_root: Path,
    geometry_sha256: str,
    language_resolver: RouteBLanguageBundleResolver,
    reader_factory: ReaderFactory,
    config: V1IndexConfig,
) -> dict[str, Any]:
    if not isinstance(raw_record, Mapping):
        raise FormalV1ContractError(f"inventory episode {index} is malformed")
    source = raw_record.get("source_receipt")
    if not isinstance(source, Mapping):
        raise FormalV1ContractError(
            f"inventory episode {index} lacks source receipt"
        )
    observed = _episode_record_from_source_receipt(
        target_id=_required_text(
            raw_record, "target_id", label=f"inventory episode {index}"
        ),
        source_receipt_path=Path(
            _required_text(
                source,
                "path",
                label=f"inventory episode {index} source receipt",
            )
        ),
        universe_root=universe_root,
        geometry_contract_sha256=geometry_sha256,
        reader_factory=reader_factory,
    )
    if observed != dict(raw_record):
        raise FormalV1ContractError(
            f"inventory episode drift detected: {raw_record.get('target_id')}"
        )
    v1_record = _record_to_v1_episode(observed, source_manifest=inventory_path)
    language_reader = reader_factory(v1_record.episode_path)
    try:
        resolution = language_resolver.resolve_with_provenance(
            language_reader.metadata
        )
    except Exception as exc:
        raise FormalV1ContractError(
            "Route-B language binding does not cover included episode "
            f"{observed['target_id']}: {exc}"
        ) from exc
    finally:
        language_reader.close()
    index_result = _build_v1_sample_index_from_episode_records(
        config=config,
        episodes=(v1_record,),
        reader_factory=reader_factory,
    )
    if len(index_result) <= 0:
        raise FormalV1ContractError(
            "episode has no legal Formal K=8/action32 windows: "
            f"{observed['target_id']}"
        )
    verified = dict(observed)
    verified["legal_window_count"] = len(index_result)
    verified["language_binding"] = {
        "scope_id": resolution.scope_id,
        "source_kind": resolution.source_kind,
        "binding_sha256": resolution.binding_sha256,
        "instruction_sha256": resolution.instruction_sha256,
    }
    return verified


def _validate_language_inventory_exact_cover(
    language_payload: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> None:
    """Bind a v2 language episode universe to the Formal included universe."""

    if (
        language_payload.get("schema_version")
        != "robotwin_route_b_language_manifest_v2"
    ):
        return
    inventory_episodes = inventory.get("episodes")
    if not isinstance(inventory_episodes, list):
        raise FormalV1ContractError("source inventory has no episode list")
    target_ids: list[str] = []
    for index, episode in enumerate(inventory_episodes):
        if not isinstance(episode, Mapping):
            raise FormalV1ContractError(
                f"source inventory episode {index} is malformed"
            )
        target_ids.append(
            _required_text(
                episode,
                "target_id",
                label=f"source inventory episode {index}",
            )
        )
    expected_target_digest = hashlib.sha256(
        _canonical_json(sorted(target_ids)).encode("utf-8")
    ).hexdigest()
    if (
        language_payload.get("episode_count") != len(inventory_episodes)
        or language_payload.get("target_id_set_sha256")
        != expected_target_digest
    ):
        raise FormalV1ContractError(
            "Route-B language v2 episode universe differs from the Formal "
            "source inventory included universe"
        )


_VerifyEpisodeTask = tuple[int, dict[str, Any], str, str, str, str, str]
_PROCESS_LANGUAGE_RESOLVERS: dict[
    tuple[str, str], RouteBLanguageBundleResolver
] = {}


def _verify_episode_default_reader(
    task: _VerifyEpisodeTask,
) -> dict[str, Any]:
    """Process-worker entry point for independent Formal verification."""

    (
        index,
        raw_record,
        inventory_path,
        universe_root,
        geometry_sha256,
        language_path,
        language_file_sha256,
    ) = task
    resolver_key = (language_path, language_file_sha256)
    resolver = _PROCESS_LANGUAGE_RESOLVERS.get(resolver_key)
    if resolver is None:
        resolver = RouteBLanguageBundleResolver(language_path)
        _PROCESS_LANGUAGE_RESOLVERS[resolver_key] = resolver
    return _verify_inventory_episode(
        index=index,
        raw_record=raw_record,
        inventory_path=Path(inventory_path),
        universe_root=Path(universe_root),
        geometry_sha256=geometry_sha256,
        language_resolver=resolver,
        reader_factory=_default_reader_factory,
        config=formal_v1_index_config(),
    )


def verify_and_freeze_formal_v1(
    *,
    source_inventory_path: str | Path,
    language_manifest_path: str | Path,
    output_dir: str | Path,
    split_salt: str,
    train_fraction: str | float | Fraction = "0.90",
    validation_fraction: str | float | Fraction = "0.05",
    test_fraction: str | float | Fraction = "0.05",
    reader_factory: ReaderFactory | None = None,
    workers: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> FormalV1Artifacts:
    """Independently verify an inventory and freeze a Formal V1 split."""

    inventory, inventory_path = _load_source_inventory(source_inventory_path)
    universe = inventory["universe"]
    universe_path = Path(str(universe["path"])).resolve()
    geometry_sha256 = str(universe["geometry_contract_sha256"])
    requested_workers = _positive_worker_count(workers)
    factory = reader_factory or _default_reader_factory
    config = formal_v1_index_config()
    language_path = _regular_file(
        language_manifest_path, label="Route-B language manifest"
    )
    language_file_sha256 = _sha256_file(language_path)
    language_payload = _strict_json_object(
        language_path, label="Route-B language manifest"
    )
    language_resolver = RouteBLanguageBundleResolver(language_path)
    _validate_language_inventory_exact_cover(language_payload, inventory)
    language_semantic_sha256 = language_payload.get("manifest_sha256")
    if not _is_sha256(language_semantic_sha256):
        raise FormalV1ContractError(
            "Route-B language manifest semantic SHA-256 is malformed"
        )
    language_binding = {
        "path": str(language_path),
        "file_sha256": language_file_sha256,
        "semantic_sha256": str(language_semantic_sha256),
        "schema_version": language_payload.get("schema_version"),
    }

    for raw_exclusion in inventory["exclusions"]:
        if not isinstance(raw_exclusion, Mapping):
            raise FormalV1ContractError("inventory exclusion is malformed")
        _validate_exclusion_record(raw_exclusion)

    episode_inputs: list[tuple[int, Mapping[str, Any]]] = []
    episode_paths: list[str] = []
    for index, raw_record in enumerate(inventory["episodes"]):
        if not isinstance(raw_record, Mapping):
            raise FormalV1ContractError(f"inventory episode {index} is malformed")
        source = raw_record.get("source_receipt")
        if not isinstance(source, Mapping):
            raise FormalV1ContractError(
                f"inventory episode {index} lacks source receipt"
            )
        _required_text(
            raw_record, "target_id", label=f"inventory episode {index}"
        )
        hdf5 = raw_record.get("hdf5")
        if not isinstance(hdf5, Mapping):
            raise FormalV1ContractError(
                f"inventory episode {index} lacks HDF5 binding"
            )
        episode_paths.append(
            _required_text(
                hdf5, "path", label=f"inventory episode {index} HDF5 binding"
            )
        )
        episode_inputs.append((index, raw_record))
    if len(episode_paths) != len(set(episode_paths)):
        raise FormalV1ContractError(
            "source inventory contains duplicate HDF5 paths"
        )

    if reader_factory is None:
        tasks: list[_VerifyEpisodeTask] = [
            (
                index,
                dict(raw_record),
                str(inventory_path),
                str(universe_path.parent),
                geometry_sha256,
                str(language_path),
                language_file_sha256,
            )
            for index, raw_record in episode_inputs
        ]
        verified_records = _run_ordered_tasks(
            tasks,
            worker=_verify_episode_default_reader,
            workers=requested_workers,
            phase="formal_episode_reverification_and_indexing",
            progress_callback=progress_callback,
        )
    else:
        # Preserve stateful/non-pickleable ReaderFactory test injection.
        verified_records = _run_ordered_tasks(
            episode_inputs,
            worker=lambda item: _verify_inventory_episode(
                index=item[0],
                raw_record=item[1],
                inventory_path=inventory_path,
                universe_root=universe_path.parent,
                geometry_sha256=geometry_sha256,
                language_resolver=language_resolver,
                reader_factory=factory,
                config=config,
            ),
            workers=1,
            phase="formal_episode_reverification_and_indexing",
            progress_callback=progress_callback,
        )

    fractions = _validate_split_fractions(
        train_fraction, validation_fraction, test_fraction
    )
    allocations, split_group_counts_by_scope = (
        _scope_stratified_group_allocations(
            verified_records,
            split_salt=split_salt,
            fractions=fractions,
        )
    )
    splits: dict[str, list[dict[str, Any]]] = {
        name: [] for name in FORMAL_SPLIT_NAMES
    }
    seen_groups: dict[str, str] = {}
    for record in verified_records:
        group_id = str(record["split_group_id"])
        split_name = allocations[group_id]
        previous = seen_groups.setdefault(group_id, split_name)
        if previous != split_name:
            raise AssertionError("episode group leaked across splits")
        splits[split_name].append(record)
    for entries in splits.values():
        entries.sort(key=lambda item: item["target_id"])

    verified_record_set_sha256 = _episode_set_sha256(verified_records)
    exclusion_set_sha256 = _exclusion_set_sha256(inventory["exclusions"])
    output = Path(output_dir).expanduser()
    if not output.is_absolute():
        raise FormalV1ContractError("Formal output directory must be absolute")
    output = output.resolve()
    receipt_path = output / "formal_v1_dataset_receipt.json"
    receipt: dict[str, Any] = {
        "schema_version": DATASET_RECEIPT_SCHEMA,
        "status": "PASS",
        "passed": True,
        "dataset_role": "formal_v1_training_dataset_index",
        "formal_index_eligible": True,
        "training_authorized": True,
        "production_authorized": False,
        "source_inventory": {
            "path": str(inventory_path),
            "file_sha256": _sha256_file(inventory_path),
            "semantic_sha256": inventory["inventory_sha256"],
        },
        "language_manifest": language_binding,
        "universe": dict(universe),
        "expected_target_count": inventory["expected_target_count"],
        "verified_episode_count": len(verified_records),
        "excluded_target_count": len(inventory["exclusions"]),
        "verified_legal_window_count": sum(
            int(record["legal_window_count"]) for record in verified_records
        ),
        "verified_episode_record_set_sha256": verified_record_set_sha256,
        "exclusion_record_set_sha256": exclusion_set_sha256,
        "temporal_contract": _temporal_contract(config),
        "verification_policy": {
            "source_receipt_hashes_verified": True,
            "candidate_report_hashes_verified": True,
            "episode_hdf5_hashes_verified": True,
            "raw_hdf5_contract_verified": True,
            "camera_order_verified": list(PHYSICAL_CAMERA_NAMES),
            "control_timeline_verified": True,
            "legal_window_count_verified": True,
            "explicit_exclusions_only": True,
            "directory_globbing_used": False,
            "language_task_info_binding_verified": True,
        },
    }
    receipt["receipt_sha256"] = _semantic_digest(
        receipt,
        digest_field="receipt_sha256",
        domain=_RECEIPT_HASH_DOMAIN,
    )
    _atomic_create_or_verify(receipt_path, receipt)

    split_manifest_path = output / "formal_v1_split_manifest.json"
    split_records: dict[str, Any] = {}
    for name in FORMAL_SPLIT_NAMES:
        entries = splits[name]
        split_records[name] = {
            "episode_count": len(entries),
            "legal_window_count": sum(
                int(record["legal_window_count"]) for record in entries
            ),
            "episode_record_set_sha256": _episode_set_sha256(entries),
            "episodes": entries,
        }
    split_manifest: dict[str, Any] = {
        "schema_version": SPLIT_MANIFEST_SCHEMA,
        "status": "FROZEN",
        "passed": True,
        "dataset_role": "formal_v1_frozen_train_validation_test_split",
        "formal_index_eligible": True,
        "training_authorized": True,
        "production_authorized": False,
        "dataset_receipt": {
            "path": str(receipt_path),
            "file_sha256": _sha256_file(receipt_path),
            "semantic_sha256": receipt["receipt_sha256"],
            "schema_version": DATASET_RECEIPT_SCHEMA,
        },
        "language_manifest": language_binding,
        "verified_episode_record_set_sha256": verified_record_set_sha256,
        "excluded_target_count": len(inventory["exclusions"]),
        "exclusion_record_set_sha256": exclusion_set_sha256,
        "temporal_contract": _temporal_contract(config),
        "split_recipe": {
            "schema_version": (
                "robotwin_momap_episode_group_scope_stratified_hash_split_v2"
            ),
            "split_salt": split_salt,
            "grouping": (
                "canonical_rollout_id+scope_id+canonical_episode_id+seed"
            ),
            "stratification": "identity.scope_id",
            "scope_count": len(split_group_counts_by_scope),
            "split_group_counts_by_scope": split_group_counts_by_scope,
            "every_positive_split_populated_per_scope": True,
            "whole_episode_assignment": True,
            "window_leakage_forbidden": True,
            "fractions": {
                name: str(fractions[name]) for name in FORMAL_SPLIT_NAMES
            },
        },
        "splits": split_records,
    }
    split_manifest["manifest_sha256"] = _semantic_digest(
        split_manifest,
        digest_field="manifest_sha256",
        domain=_SPLIT_HASH_DOMAIN,
    )
    _atomic_create_or_verify(split_manifest_path, split_manifest)
    return FormalV1Artifacts(
        dataset_receipt_path=receipt_path,
        dataset_receipt_sha256=str(receipt["receipt_sha256"]),
        split_manifest_path=split_manifest_path,
        split_manifest_sha256=str(split_manifest["manifest_sha256"]),
    )


def _validate_receipt(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    path = _regular_file(path, label="Formal V1 dataset receipt")
    if _sha256_file(path) != expected.get("file_sha256"):
        raise FormalV1ContractError("Formal dataset receipt file SHA mismatch")
    receipt = _strict_json_object(path, label="Formal V1 dataset receipt")
    if (
        receipt.get("schema_version") != DATASET_RECEIPT_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("passed") is not True
        or receipt.get("formal_index_eligible") is not True
        or receipt.get("training_authorized") is not True
        or receipt.get("production_authorized") is not False
    ):
        raise FormalV1ContractError("Formal dataset receipt policy mismatch")
    claimed = receipt.get("receipt_sha256")
    actual = _semantic_digest(
        receipt,
        digest_field="receipt_sha256",
        domain=_RECEIPT_HASH_DOMAIN,
    )
    if (
        not _is_sha256(claimed)
        or claimed != actual
        or claimed != expected.get("semantic_sha256")
    ):
        raise FormalV1ContractError("Formal dataset receipt semantic SHA mismatch")
    return receipt


def _validate_language_manifest_binding(
    binding: Any,
    *,
    provided_path: str | Path | None = None,
) -> Path:
    if not isinstance(binding, Mapping):
        raise FormalV1ContractError("Formal artifact lacks language manifest binding")
    bound_path = _regular_file(
        _required_text(binding, "path", label="language manifest binding"),
        label="bound Route-B language manifest",
    )
    if provided_path is not None:
        supplied = _regular_file(
            provided_path, label="runtime Route-B language manifest"
        )
        if supplied != bound_path:
            raise FormalV1ContractError(
                "runtime language manifest path differs from frozen Formal binding"
            )
    if _sha256_file(bound_path) != binding.get("file_sha256"):
        raise FormalV1ContractError("bound language manifest file SHA mismatch")
    payload = _strict_json_object(bound_path, label="bound Route-B language manifest")
    # Construction validates its schema, semantic digest, sidecar hashes and
    # duplicate task/config scopes.
    RouteBLanguageBundleResolver(bound_path)
    if (
        payload.get("manifest_sha256") != binding.get("semantic_sha256")
        or payload.get("schema_version") != binding.get("schema_version")
    ):
        raise FormalV1ContractError(
            "bound language manifest semantic identity mismatch"
        )
    return bound_path


def _load_formal_split_manifest(
    formal_split_manifest: str | Path,
    *,
    expected_split_file_sha256: str | None = None,
    expected_split_semantic_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Authenticate the external split anchor before following its bindings."""

    if (expected_split_file_sha256 is None) != (
        expected_split_semantic_sha256 is None
    ):
        raise FormalV1ContractError(
            "expected split file and semantic SHA-256 anchors must be provided "
            "together"
        )
    for name, value in (
        ("expected_split_file_sha256", expected_split_file_sha256),
        ("expected_split_semantic_sha256", expected_split_semantic_sha256),
    ):
        if value is not None and (
            not isinstance(value, str) or not _is_sha256(value)
        ):
            raise FormalV1ContractError(
                f"{name} must be one lowercase 64-character SHA-256 digest"
            )

    manifest_path = _regular_file(
        formal_split_manifest, label="Formal V1 split manifest"
    )
    if (
        expected_split_file_sha256 is not None
        and _sha256_file(manifest_path) != expected_split_file_sha256
    ):
        raise FormalV1ContractError(
            "Formal split manifest file SHA-256 differs from external anchor"
        )
    manifest = _strict_json_object(manifest_path, label="Formal V1 split manifest")
    if (
        manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA
        or manifest.get("status") != "FROZEN"
        or manifest.get("passed") is not True
        or manifest.get("formal_index_eligible") is not True
        or manifest.get("training_authorized") is not True
        or manifest.get("production_authorized") is not False
    ):
        raise FormalV1ContractError("Formal split manifest policy mismatch")
    claimed_manifest_sha = manifest.get("manifest_sha256")
    actual_manifest_sha = _semantic_digest(
        manifest,
        digest_field="manifest_sha256",
        domain=_SPLIT_HASH_DOMAIN,
    )
    if (
        not _is_sha256(claimed_manifest_sha)
        or claimed_manifest_sha != actual_manifest_sha
    ):
        raise FormalV1ContractError("Formal split manifest semantic SHA mismatch")
    if (
        expected_split_semantic_sha256 is not None
        and claimed_manifest_sha != expected_split_semantic_sha256
    ):
        raise FormalV1ContractError(
            "Formal split manifest semantic SHA-256 differs from external anchor"
        )
    return manifest_path, manifest


def load_formal_split_episode_records(
    *,
    formal_split_manifest: str | Path,
    split_name: str,
    expected_split_file_sha256: str | None = None,
    expected_split_semantic_sha256: str | None = None,
    verify_source_artifact_hashes: bool = True,
    verify_hdf5_hashes: bool = False,
) -> tuple[V1EpisodeRecord, ...]:
    """Load exactly one immutable Formal split; no candidate selector is accepted."""

    if split_name not in FORMAL_SPLIT_NAMES:
        raise FormalV1ContractError(
            f"split_name must be one of {FORMAL_SPLIT_NAMES!r}"
        )
    manifest_path, manifest = _load_formal_split_manifest(
        formal_split_manifest,
        expected_split_file_sha256=expected_split_file_sha256,
        expected_split_semantic_sha256=expected_split_semantic_sha256,
    )
    receipt_binding = manifest.get("dataset_receipt")
    if not isinstance(receipt_binding, Mapping):
        raise FormalV1ContractError("Formal split has no dataset receipt binding")
    receipt = _validate_receipt(
        Path(_required_text(receipt_binding, "path", label="dataset receipt binding")),
        receipt_binding,
    )
    if (
        manifest.get("verified_episode_record_set_sha256")
        != receipt.get("verified_episode_record_set_sha256")
        or manifest.get("exclusion_record_set_sha256")
        != receipt.get("exclusion_record_set_sha256")
        or manifest.get("excluded_target_count")
        != receipt.get("excluded_target_count")
        or manifest.get("temporal_contract") != receipt.get("temporal_contract")
        or manifest.get("language_manifest") != receipt.get("language_manifest")
    ):
        raise FormalV1ContractError("Formal split/dataset receipt binding mismatch")
    _validate_language_manifest_binding(manifest.get("language_manifest"))
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(FORMAL_SPLIT_NAMES):
        raise FormalV1ContractError("Formal split set differs from contract")
    all_records: list[Mapping[str, Any]] = []
    group_owner: dict[str, str] = {}
    target_owner: dict[str, str] = {}
    for name in FORMAL_SPLIT_NAMES:
        split = splits.get(name)
        if not isinstance(split, Mapping):
            raise FormalV1ContractError(f"Formal {name} split is malformed")
        episodes = split.get("episodes")
        if not isinstance(episodes, list):
            raise FormalV1ContractError(f"Formal {name} episodes are malformed")
        if split.get("episode_count") != len(episodes):
            raise FormalV1ContractError(f"Formal {name} episode_count mismatch")
        if split.get("episode_record_set_sha256") != _episode_set_sha256(episodes):
            raise FormalV1ContractError(f"Formal {name} episode set SHA mismatch")
        if split.get("legal_window_count") != sum(
            int(record.get("legal_window_count", -1))
            for record in episodes
            if isinstance(record, Mapping)
        ):
            raise FormalV1ContractError(f"Formal {name} window count mismatch")
        for record in episodes:
            if not isinstance(record, Mapping):
                raise FormalV1ContractError(
                    f"Formal {name} contains malformed episode"
                )
            target_id = _required_text(
                record, "target_id", label=f"Formal {name} episode"
            )
            group_id = _required_text(
                record, "split_group_id", label=f"Formal {name} episode"
            )
            if target_id in target_owner or (
                group_id in group_owner and group_owner[group_id] != name
            ):
                raise FormalV1ContractError(
                    "episode or window group leaks across Formal splits"
                )
            target_owner[target_id] = name
            group_owner[group_id] = name
            all_records.append(record)
    if _episode_set_sha256(all_records) != manifest.get(
        "verified_episode_record_set_sha256"
    ):
        raise FormalV1ContractError("Formal global episode set SHA mismatch")
    if len(all_records) != receipt.get("verified_episode_count"):
        raise FormalV1ContractError("Formal receipt episode count mismatch")

    selected = splits[split_name]["episodes"]
    result: list[V1EpisodeRecord] = []
    for record in selected:
        hdf5 = record.get("hdf5")
        source = record.get("source_receipt")
        report = record.get("candidate_report")
        if not all(isinstance(item, Mapping) for item in (hdf5, source, report)):
            raise FormalV1ContractError("Formal episode artifact bindings malformed")
        hdf5_path = _regular_file(
            _required_text(hdf5, "path", label="Formal HDF5 binding"),
            label="Formal HDF5 episode",
        )
        source_path = _regular_file(
            _required_text(source, "path", label="Formal source receipt binding"),
            label="Formal source receipt",
        )
        report_path = _regular_file(
            _required_text(report, "path", label="Formal candidate report binding"),
            label="Formal candidate report",
        )
        if verify_source_artifact_hashes and (
            _sha256_file(source_path) != source.get("sha256")
            or _sha256_file(report_path) != report.get("sha256")
        ):
            raise FormalV1ContractError("Formal source artifact SHA mismatch")
        if verify_hdf5_hashes and _sha256_file(hdf5_path) != hdf5.get("sha256"):
            raise FormalV1ContractError("Formal HDF5 SHA mismatch")
        candidate_report_payload = None
        if (
            report.get("schema_version")
            == "robotwin_momap_canonical_rollout_v1_candidate_report_v2"
        ):
            candidate_report_payload = _strict_json_object(
                report_path,
                label="Formal candidate report v2",
            )
        result.append(
            _record_to_v1_episode(
                record,
                source_manifest=manifest_path,
                candidate_report_payload=candidate_report_payload,
            )
        )
    if not result:
        raise FormalV1ContractError(f"Formal {split_name} split is empty")
    return tuple(result)


def _build_formal_index_shard(task: _FormalIndexShardTask) -> V1SampleIndex:
    """Spawn-safe worker for one contiguous Formal episode range."""

    stop = task.ordinal_start + len(task.episodes)
    return _build_v1_sample_index_from_episode_records(
        config=task.config,
        episodes=task.episodes,
        episode_ordinals=tuple(range(task.ordinal_start, stop)),
        reader_factory=None,
        progress_callback=None,
        include_source_manifest_bindings=False,
    )


def _weighted_contiguous_episode_ranges(
    episodes: Sequence[V1EpisodeRecord],
    *,
    shard_count: int,
) -> tuple[tuple[int, int], ...]:
    """Split frozen episode order into non-empty, approximately equal work."""

    total_episodes = len(episodes)
    if total_episodes <= 0:
        raise ValueError("cannot shard an empty Formal episode sequence")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise TypeError("shard_count must be an integer")
    resolved_count = min(shard_count, total_episodes)
    if resolved_count <= 0:
        raise ValueError("shard_count must be positive")
    weights = tuple(
        (
            episode.expected_frame_count
            if isinstance(episode.expected_frame_count, int)
            and not isinstance(episode.expected_frame_count, bool)
            and episode.expected_frame_count > 0
            else 1
        )
        for episode in episodes
    )
    ranges: list[tuple[int, int]] = []
    start = 0
    remaining_weight = sum(weights)
    for shard_ordinal in range(resolved_count):
        remaining_shards = resolved_count - shard_ordinal
        if remaining_shards == 1:
            stop = total_episodes
        else:
            maximum_stop = total_episodes - (remaining_shards - 1)
            target = remaining_weight / remaining_shards
            stop = start
            accumulated = 0
            while stop < maximum_stop:
                next_weight = weights[stop]
                if (
                    stop > start
                    and accumulated + next_weight > target
                    and abs(accumulated - target)
                    <= abs(accumulated + next_weight - target)
                ):
                    break
                accumulated += next_weight
                stop += 1
                if accumulated >= target:
                    break
        if stop <= start:
            raise AssertionError(
                "Formal weighted sharding produced an empty range"
            )
        ranges.append((start, stop))
        remaining_weight -= sum(weights[start:stop])
        start = stop
    if (
        start != total_episodes
        or ranges[0][0] != 0
        or any(
            left_stop != right_start
            for (_, left_stop), (right_start, _) in zip(
                ranges,
                ranges[1:],
            )
        )
    ):
        raise AssertionError(
            "Formal weighted shards do not exactly cover episodes"
        )
    return tuple(ranges)


def _build_parallel_formal_v1_sample_index(
    *,
    config: V1IndexConfig,
    episodes: Sequence[V1EpisodeRecord],
    workers: int,
    progress_callback: IndexProgressCallback | None,
) -> V1SampleIndex:
    resolved_workers = _positive_worker_count(workers)
    resolved_episodes = tuple(episodes)
    shard_count = min(len(resolved_episodes), resolved_workers * 2)
    ranges = _weighted_contiguous_episode_ranges(
        resolved_episodes,
        shard_count=shard_count,
    )
    tasks = tuple(
        _FormalIndexShardTask(
            config=config,
            episodes=resolved_episodes[start:stop],
            ordinal_start=start,
        )
        for start, stop in ranges
    )

    completed_episodes = 0
    completed_windows = 0
    if progress_callback is not None:
        progress_callback(0, len(resolved_episodes), 0)

    def report_result(
        result: V1SampleIndex,
        _completed_shards: int,
        _total_shards: int,
    ) -> None:
        nonlocal completed_episodes, completed_windows
        completed_episodes += len(result.episodes)
        completed_windows += len(result)
        if progress_callback is not None:
            progress_callback(
                completed_episodes,
                len(resolved_episodes),
                completed_windows,
            )

    shards = _run_ordered_tasks(
        tasks,
        worker=_build_formal_index_shard,
        workers=resolved_workers,
        phase="runtime_index",
        progress_callback=None,
        process_start_method="spawn",
        result_progress_callback=report_result,
    )
    merged = _merge_v1_sample_index_shards(
        config=config,
        episodes=resolved_episodes,
        shards=shards,
    )
    composed = merge_v1_runtime_index_bindings(
        [shard.runtime_index_binding() for shard in shards]
    )
    if composed != merged.runtime_index_binding():
        raise RuntimeError("parallel Formal index merge identity differs")
    if progress_callback is not None:
        progress_callback(
            len(resolved_episodes),
            len(resolved_episodes),
            len(merged),
        )
    return merged


def build_formal_v1_sample_index(
    *,
    formal_split_manifest: str | Path,
    split_name: str,
    language_manifest: str | Path,
    config: V1IndexConfig,
    expected_split_file_sha256: str | None = None,
    expected_split_semantic_sha256: str | None = None,
    verify_source_artifact_hashes: bool = True,
    verify_hdf5_hashes: bool = False,
    reader_factory: ReaderFactory | None = None,
    runtime_index_cache_dir: str | Path | None = None,
    runtime_index_cache_verify_payload_hashes: bool = False,
    progress_callback: IndexProgressCallback | None = None,
    progress_every_episodes: int = 100,
    index_build_workers: int = 1,
) -> V1SampleIndex:
    """Build windows only after the frozen Formal selector verifies."""

    manifest_path, manifest = _load_formal_split_manifest(
        formal_split_manifest,
        expected_split_file_sha256=expected_split_file_sha256,
        expected_split_semantic_sha256=expected_split_semantic_sha256,
    )
    _validate_language_manifest_binding(
        manifest.get("language_manifest"),
        provided_path=language_manifest,
    )
    if _temporal_contract(config) != manifest.get("temporal_contract"):
        raise FormalV1ContractError(
            "runtime temporal contract differs from frozen Formal split"
        )
    if not isinstance(runtime_index_cache_verify_payload_hashes, bool):
        raise TypeError(
            "runtime_index_cache_verify_payload_hashes must be bool"
        )
    resolved_index_workers = _positive_worker_count(index_build_workers)
    if runtime_index_cache_dir is not None:
        if not str(runtime_index_cache_dir).strip():
            raise ValueError(
                "runtime_index_cache_dir must not be empty when provided"
            )
        from .v1_index_cache import _load_v1_runtime_index_cache

        return _load_v1_runtime_index_cache(
            runtime_index_cache_dir,
            config=config,
            formal_split_manifest_path=manifest_path,
            formal_split_file_sha256=_sha256_file(manifest_path),
            formal_split_semantic_sha256=str(manifest["manifest_sha256"]),
            split_name=split_name,
            language_manifest_path=language_manifest,
            require_source_artifact_hashes_verified=(
                verify_source_artifact_hashes
            ),
            require_hdf5_hashes_verified=verify_hdf5_hashes,
            verify_payload_hashes=(
                runtime_index_cache_verify_payload_hashes
            ),
        )
    episodes = load_formal_split_episode_records(
        formal_split_manifest=manifest_path,
        split_name=split_name,
        expected_split_file_sha256=expected_split_file_sha256,
        expected_split_semantic_sha256=expected_split_semantic_sha256,
        verify_source_artifact_hashes=verify_source_artifact_hashes,
        verify_hdf5_hashes=verify_hdf5_hashes,
    )
    if resolved_index_workers > 1:
        if reader_factory is not None:
            raise ValueError(
                "parallel Formal index build requires the default reader_factory"
            )
        return _build_parallel_formal_v1_sample_index(
            config=config,
            episodes=episodes,
            workers=resolved_index_workers,
            progress_callback=progress_callback,
        )
    return _build_v1_sample_index_from_episode_records(
        config=config,
        episodes=episodes,
        reader_factory=reader_factory,
        progress_callback=progress_callback,
        progress_every_episodes=progress_every_episodes,
    )


__all__ = [
    "ALLOWED_EXCLUSION_REASON_CODES",
    "DATASET_RECEIPT_SCHEMA",
    "FORMAL_SPLIT_NAMES",
    "FormalV1Artifacts",
    "FormalV1ContractError",
    "SOURCE_INVENTORY_SCHEMA",
    "SPLIT_MANIFEST_SCHEMA",
    "build_formal_source_inventory",
    "build_formal_v1_sample_index",
    "formal_v1_index_config",
    "load_formal_split_episode_records",
    "verify_and_freeze_formal_v1",
]
