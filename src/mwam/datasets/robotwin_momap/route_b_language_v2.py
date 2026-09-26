"""Full-data Route-B language bindings keyed by resolved task information.

The historical Route-B language manifest v1 contains one binding for each
``(task_name, task_config)`` pair.  That is sufficient for its exact-100
canary source, but not for the full dataset: different successful rollouts of
the same task/config can resolve different object instances or robot arms.

This module defines and publishes manifest v2.  Its lookup identity is:

``(task_name, task_config, SHA256(canonical resolved_info))``.

Publication consumes an explicit JSONL list of successful target receipts.  It
verifies the receipt, candidate-report, canonical-manifest and canonical-
identity chain, derives the instruction from RoboTwin's official instruction
and object-description files, and records every included target in the
self-hashed manifest.  The generated bundle is create-only and does not mutate
the source dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA_VERSION_V2 = "robotwin_route_b_language_manifest_v2"
BINDING_SCHEMA_VERSION_V2 = "robotwin_route_b_task_language_binding_v2"
NATIVE_RECEIPT_SCHEMA = "robotwin_momap_route_b_v1_fleet_target_success_v5"
IMPORTED_RECEIPT_SCHEMA = (
    "robotwin_momap_route_b_v1_r8_target_imported_success_v1"
)
CANDIDATE_REPORT_SCHEMAS = {
    "robotwin_momap_canonical_rollout_v1_candidate_report_v1",
    "robotwin_momap_canonical_rollout_v1_candidate_report_v2",
}
CANONICAL_MANIFEST_SCHEMA = "robotwin_momap_canonical_rollout_manifest_v2"
CANONICAL_IDENTITY_SCHEMA = "robotwin_momap_canonical_rollout_identity_v2"

VARIANT_BY_CONFIG = {
    "demo_clean": "clean",
    "demo_randomized": "randomized",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TEMPLATE_PLACEHOLDER_RE = re.compile(r"{([^}]+)}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return str(value)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {label} JSONL {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid {label} JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"{label} JSONL record {line_number} is not an object"
            )
        records.append(value)
    return records


def _regular_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _safe_relative(root: Path, value: Any, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe {label} relative path")
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its source root") from exc
    return _regular_file(path, label)


def _bound_absolute_file(
    value: Any,
    *,
    allowed_root: Path,
    label: str,
) -> Path:
    raw = Path(str(value)).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = _regular_file(raw, label)
    try:
        path.relative_to(allowed_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes its source root") from exc
    return path


def _file_spec(
    path: Path,
    *,
    expected_sha256: Any,
    expected_size: Any | None = None,
    verify_hash: bool = True,
    label: str,
) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, f"{label}.sha256")
    size = path.stat().st_size
    if expected_size is not None and (
        not isinstance(expected_size, int) or expected_size != size
    ):
        raise ValueError(f"{label} size mismatch")
    if verify_hash and _sha256_file(path) != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return {
        "path": str(path),
        "sha256": expected,
        "size_bytes": size,
        "live_hash_verified": bool(verify_hash),
    }


def _variant(task_config: str) -> str:
    variant = VARIANT_BY_CONFIG.get(task_config)
    if variant is None:
        raise ValueError(f"unsupported Route-B task_config: {task_config!r}")
    return variant


def _scope_id(
    task_name: str,
    task_config: str,
    resolved_task_info_sha256: str,
) -> str:
    return (
        f"{task_name}:{_variant(task_config)}:"
        f"{resolved_task_info_sha256}"
    )


def _task_info_digest(value: Mapping[str, Any]) -> str:
    return _sha256_json(dict(value))


def _source_paths(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    fleet_root: Path,
    verify_hdf5_hash: bool,
) -> dict[str, dict[str, Any]]:
    schema = receipt.get("schema_version")
    if schema not in {NATIVE_RECEIPT_SCHEMA, IMPORTED_RECEIPT_SCHEMA}:
        raise ValueError(f"unsupported successful target receipt schema: {schema!r}")
    if (
        receipt.get("passed") is not True
        or receipt.get("dataset_role") != "versioned_v1_phase1_staging"
        or receipt.get("formal_publication_authorized") is not False
    ):
        raise ValueError("source target receipt is not verified staging data")

    if schema == NATIVE_RECEIPT_SCHEMA:
        if receipt.get("status") != "SUCCESS":
            raise ValueError("native target receipt is not SUCCESS")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("native target receipt has no artifacts")
        source_root = fleet_root
        episode_path = _safe_relative(
            source_root, artifacts.get("episode_hdf5"), "native episode HDF5"
        )
        report_path = _safe_relative(
            source_root,
            artifacts.get("candidate_report"),
            "native candidate report",
        )
        episode_spec = _file_spec(
            episode_path,
            expected_sha256=artifacts.get("episode_hdf5_sha256"),
            verify_hash=verify_hdf5_hash,
            label="native episode HDF5",
        )
        report_spec = _file_spec(
            report_path,
            expected_sha256=artifacts.get("candidate_report_sha256"),
            label="native candidate report",
        )
        closure: dict[str, Mapping[str, Any]] | None = None
    else:
        if (
            receipt.get("status") != "IMPORTED_VERIFIED"
            or receipt.get("candidate_only") is not True
            or receipt.get("formal_index_eligible") is not False
            or receipt.get("external_artifact_reference") is not True
            or receipt.get("source_retention_required") is not True
        ):
            raise ValueError("imported target receipt policy mismatch")
        source = receipt.get("source_generation")
        raw_closure = receipt.get("source_artifact_closure")
        if not isinstance(source, Mapping) or not isinstance(raw_closure, list):
            raise ValueError("imported target receipt lacks source closure")
        source_root = Path(str(source.get("root", ""))).expanduser()
        if not source_root.is_absolute() or not source_root.is_dir():
            raise ValueError("imported source root is unavailable")
        source_root = source_root.resolve()
        closure = {}
        for raw in raw_closure:
            if not isinstance(raw, Mapping):
                raise ValueError("imported source closure is malformed")
            name = str(raw.get("name", "")).strip()
            if not name or name in closure:
                raise ValueError("imported source closure has duplicate/missing name")
            closure[name] = raw
        try:
            episode = closure["episode_hdf5"]
            report = closure["candidate_report"]
        except KeyError as exc:
            raise ValueError(
                "imported source closure lacks HDF5 or candidate report"
            ) from exc
        episode_path = _safe_relative(
            source_root,
            episode.get("source_relative_path"),
            "imported episode HDF5",
        )
        report_path = _safe_relative(
            source_root,
            report.get("source_relative_path"),
            "imported candidate report",
        )
        episode_spec = _file_spec(
            episode_path,
            expected_sha256=episode.get("sha256"),
            expected_size=episode.get("size_bytes"),
            verify_hash=verify_hdf5_hash,
            label="imported episode HDF5",
        )
        report_spec = _file_spec(
            report_path,
            expected_sha256=report.get("sha256"),
            expected_size=report.get("size_bytes"),
            label="imported candidate report",
        )

    report = _load_json(report_path, "candidate report")
    if (
        report.get("schema_version") not in CANDIDATE_REPORT_SCHEMAS
        or report.get("status") != "PASS"
        or report.get("candidate_only") is not True
        or report.get("dataset_mode") != "self_consistent_canonical_rollout"
        or report.get("official_frame_alignment_claimed") is not False
    ):
        raise ValueError("candidate report contract mismatch")
    report_artifacts = report.get("artifacts")
    if not isinstance(report_artifacts, Mapping):
        raise ValueError("candidate report has no artifacts")

    def resolve_small_artifact(name: str, label: str) -> dict[str, Any]:
        raw = report_artifacts.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"candidate report lacks {name}")
        if closure is None:
            path = _bound_absolute_file(
                raw.get("path"), allowed_root=source_root, label=label
            )
            expected_sha256 = raw.get("sha256")
            expected_size = raw.get("size_bytes")
        else:
            closure_record = closure.get(name)
            if not isinstance(closure_record, Mapping):
                raise ValueError(f"imported source closure lacks {name}")
            path = _safe_relative(
                source_root,
                closure_record.get("source_relative_path"),
                f"imported {label}",
            )
            expected_sha256 = closure_record.get("sha256")
            expected_size = closure_record.get("size_bytes")
            if (
                Path(str(raw.get("path", ""))).resolve() != path
                or raw.get("sha256") != expected_sha256
                or raw.get("size_bytes") != expected_size
            ):
                raise ValueError(f"candidate report/import closure {name} mismatch")
        return _file_spec(
            path,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            label=label,
        )

    manifest_spec = resolve_small_artifact("manifest", "canonical manifest")
    identity_spec = resolve_small_artifact(
        "canonical_rollout_identity", "canonical rollout identity"
    )
    report_episode = report_artifacts.get("episode_hdf5")
    if not isinstance(report_episode, Mapping):
        raise ValueError("candidate report lacks episode_hdf5")
    if (
        Path(str(report_episode.get("path", ""))).resolve() != episode_path
        or report_episode.get("sha256") != episode_spec["sha256"]
        or report_episode.get("size_bytes") != episode_spec["size_bytes"]
    ):
        raise ValueError("candidate report/receipt episode HDF5 mismatch")

    return {
        "source_receipt": {
            "path": str(receipt_path),
            "sha256": _sha256_file(receipt_path),
            "size_bytes": receipt_path.stat().st_size,
            "live_hash_verified": True,
        },
        "candidate_report": report_spec,
        "canonical_manifest": manifest_spec,
        "canonical_rollout_identity": identity_spec,
        "episode_hdf5": episode_spec,
    }


def _template_placeholders(value: str) -> tuple[str, ...]:
    return tuple(_TEMPLATE_PLACEHOLDER_RE.findall(value))


def _template_is_compatible(
    template: str,
    resolved_info: Mapping[str, Any],
) -> bool:
    placeholders = set(_template_placeholders(template))
    keys = {str(key).strip("{}") for key in resolved_info}
    arm_keys = {key for key in keys if len(key) == 1 and key.islower()}
    return placeholders == keys or bool(
        arm_keys
        and placeholders | arm_keys == keys
        and not placeholders & arm_keys
    )


def _official_instruction(
    *,
    canonical_manifest: Mapping[str, Any],
    resolved_info: Mapping[str, Any],
) -> dict[str, Any]:
    robotwin = canonical_manifest.get("robotwin")
    if not isinstance(robotwin, Mapping):
        raise ValueError("canonical manifest has no RoboTwin provenance")
    root = Path(str(robotwin.get("project_root", ""))).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("canonical RoboTwin project root is unavailable")
    root = root.resolve()
    description_root = (root / "description").resolve()
    task_name = str(canonical_manifest.get("task_name", "")).strip()
    instruction_path = _regular_file(
        description_root / "task_instruction" / f"{task_name}.json",
        "official task instruction",
    )
    try:
        instruction_path.relative_to(description_root)
    except ValueError as exc:
        raise ValueError("official instruction escapes RoboTwin description root") from exc
    instruction_json = _load_json(instruction_path, "official task instruction")
    seen = instruction_json.get("seen")
    if not isinstance(seen, list) or not seen:
        raise ValueError("official task instruction has no non-empty seen list")
    template_index = -1
    template = ""
    for index, candidate in enumerate(seen):
        if (
            isinstance(candidate, str)
            and candidate == candidate.strip()
            and candidate
            and _template_is_compatible(candidate, resolved_info)
        ):
            template_index = index
            template = candidate
            break
    if template_index < 0:
        raise ValueError(
            "official task instruction has no template compatible with resolved_info"
        )

    normalized_info = {
        str(key).strip("{}"): str(value)
        for key, value in resolved_info.items()
    }
    rendered = template
    placeholder_sources: list[dict[str, Any]] = []
    object_root = (description_root / "objects_description").resolve()
    for placeholder in _template_placeholders(template):
        if placeholder not in normalized_info:
            raise ValueError(f"resolved_info lacks placeholder {placeholder!r}")
        raw_value = normalized_info[placeholder]
        if "/" in raw_value or "\\" in raw_value:
            descriptor = (object_root / f"{raw_value}.json").resolve()
            try:
                descriptor.relative_to(object_root)
            except ValueError as exc:
                raise ValueError(
                    f"object description escapes canonical root: {raw_value!r}"
                ) from exc
            descriptor = _regular_file(
                descriptor, f"official object description for {placeholder}"
            )
            description_json = _load_json(
                descriptor, "official object description"
            )
            descriptions = description_json.get("seen")
            if (
                not isinstance(descriptions, list)
                or not descriptions
                or not isinstance(descriptions[0], str)
                or descriptions[0] != descriptions[0].strip()
                or not descriptions[0]
            ):
                raise ValueError(
                    f"object description has no trimmed non-empty seen[0]: "
                    f"{descriptor}"
                )
            selected = descriptions[0]
            replacement = f"the {selected}"
            source = {
                "placeholder": placeholder,
                "raw_task_info_value": raw_value,
                "source_kind": "official_object_description",
                "path": str(descriptor),
                "sha256": _sha256_file(descriptor),
                "selector": "$.seen[0]",
                "selected_text": selected,
                "selected_text_sha256": _sha256_text(selected),
                "replacement": replacement,
            }
        elif len(placeholder) == 1 and placeholder.islower():
            replacement = f"the {raw_value} arm"
            source = {
                "placeholder": placeholder,
                "raw_task_info_value": raw_value,
                "source_kind": "task_info_arm_value",
                "replacement": replacement,
            }
        else:
            replacement = raw_value
            source = {
                "placeholder": placeholder,
                "raw_task_info_value": raw_value,
                "source_kind": "task_info_literal_value",
                "replacement": replacement,
            }
        rendered = rendered.replace("{" + placeholder + "}", replacement)
        placeholder_sources.append(source)
    if _template_placeholders(rendered):
        raise ValueError("rendered official instruction still has placeholders")
    if not rendered.strip() or rendered != rendered.strip():
        raise ValueError("rendered official instruction is not trimmed non-empty text")

    path_free_sources = [
        {key: value for key, value in source.items() if key != "path"}
        for source in placeholder_sources
    ]
    semantic_provenance = {
        "official_instruction_sha256": _sha256_file(instruction_path),
        "selector": f"$.seen[{template_index}]",
        "template_sha256": _sha256_text(template),
        "placeholder_sources": path_free_sources,
    }
    return {
        "instruction": rendered,
        "instruction_sha256": _sha256_text(rendered),
        "official_instruction": {
            "path": str(instruction_path),
            "sha256": semantic_provenance["official_instruction_sha256"],
            "selector": semantic_provenance["selector"],
            "selected_template": template,
            "selected_template_sha256": semantic_provenance["template_sha256"],
        },
        "placeholder_sources": placeholder_sources,
        "semantic_provenance_sha256": _sha256_json(semantic_provenance),
    }


def _episode_source_record(
    *,
    target_id: str,
    receipt_path: Path,
    fleet_root: Path,
    verify_hdf5_hash: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _load_json(receipt_path, f"target receipt {target_id}")
    identity = receipt.get("target_identity")
    if not isinstance(identity, Mapping) or identity.get("target_id") != target_id:
        raise ValueError(f"source target receipt identity mismatch: {target_id}")
    sources = _source_paths(
        receipt,
        receipt_path=receipt_path,
        fleet_root=fleet_root,
        verify_hdf5_hash=verify_hdf5_hash,
    )
    manifest = _load_json(
        Path(sources["canonical_manifest"]["path"]), "canonical manifest"
    )
    canonical_identity = _load_json(
        Path(sources["canonical_rollout_identity"]["path"]),
        "canonical rollout identity",
    )
    if (
        manifest.get("schema_version") != CANONICAL_MANIFEST_SCHEMA
        or manifest.get("mode") != "self_consistent_canonical_rollout"
        or manifest.get("status") != "complete"
        or manifest.get("task_success") is not True
        or manifest.get("official_frame_alignment_claimed") is not False
    ):
        raise ValueError(f"canonical manifest contract mismatch: {target_id}")
    if (
        canonical_identity.get("schema_version") != CANONICAL_IDENTITY_SCHEMA
        or canonical_identity.get("passed") is not True
        or canonical_identity.get("planning_scene_normalized")
        != canonical_identity.get("replay_scene_normalized")
    ):
        raise ValueError(f"canonical identity contract mismatch: {target_id}")
    if manifest.get("canonical_rollout_identity") != canonical_identity:
        raise ValueError(f"manifest/identity content mismatch: {target_id}")
    planning = manifest.get("planning_task_info")
    replay = manifest.get("replay_task_info")
    if (
        not isinstance(planning, Mapping)
        or planning != replay
        or planning != canonical_identity.get("replay_scene_normalized")
    ):
        raise ValueError(f"manifest task_info/identity mismatch: {target_id}")
    resolved_info = replay.get("info")
    if not isinstance(resolved_info, Mapping):
        raise ValueError(f"canonical identity has no resolved info: {target_id}")
    resolved_info = dict(resolved_info)
    task_name = str(identity.get("task_name", "")).strip()
    task_config = str(identity.get("task_config", "")).strip()
    variant = _variant(task_config)
    episode_id = identity.get("episode_id")
    seed = receipt.get("winning_seed")
    if (
        not task_name
        or manifest.get("task_name") != task_name
        or manifest.get("task_config") != task_config
        or identity.get("variant") != variant
        or manifest.get("canonical_episode_id") != episode_id
        or manifest.get("seed") != seed
        or canonical_identity.get("seed") != seed
        or manifest.get("rollout_id") != canonical_identity.get("rollout_id")
        or manifest.get("output_hdf5") != sources["episode_hdf5"]["path"]
        or receipt.get("frame_count") != manifest.get("captured_frames")
    ):
        raise ValueError(f"receipt/canonical identity mismatch: {target_id}")
    report = _load_json(
        Path(sources["candidate_report"]["path"]), "candidate report"
    )
    compact_identity = report.get("canonical_rollout_identity")
    if not isinstance(compact_identity, Mapping) or any(
        compact_identity.get(key) != canonical_identity.get(key)
        for key in (
            "schema_version",
            "passed",
            "planning_scene_sha256",
            "replay_scene_sha256",
            "rollout_id",
            "seed",
        )
    ):
        raise ValueError(f"candidate report identity mismatch: {target_id}")
    if report.get("frame_count") != receipt.get("frame_count"):
        raise ValueError(f"candidate report frame count mismatch: {target_id}")

    resolved_sha256 = _task_info_digest(resolved_info)
    official = _official_instruction(
        canonical_manifest=manifest,
        resolved_info=resolved_info,
    )
    scope_id = _scope_id(task_name, task_config, resolved_sha256)
    source_record = {
        "target_id": target_id,
        "task_name": task_name,
        "task_config": task_config,
        "variant": variant,
        "canonical_episode_id": int(episode_id),
        "seed": int(seed),
        "canonical_rollout_id": str(canonical_identity["rollout_id"]),
        "resolved_task_info_sha256": resolved_sha256,
        "source_receipt": sources["source_receipt"],
        "candidate_report": sources["candidate_report"],
        "canonical_manifest": sources["canonical_manifest"],
        "canonical_rollout_identity": sources["canonical_rollout_identity"],
        "episode_hdf5": sources["episode_hdf5"],
    }
    binding_source = {
        "scope_id": scope_id,
        "task_name": task_name,
        "task_config": task_config,
        "variant": variant,
        "resolved_info": resolved_info,
        "resolved_task_info_sha256": resolved_sha256,
        "official": official,
        "representative_sources": sources,
        "representative_target_id": target_id,
        "representative_rollout_id": str(canonical_identity["rollout_id"]),
    }
    return source_record, binding_source


def _member_source_set_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_json(
        sorted((dict(record) for record in records), key=lambda row: row["target_id"])
    )


def _binding_value(
    source: Mapping[str, Any],
    *,
    member_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    member_digest = _member_source_set_sha256(member_records)
    official = source["official"]
    value: dict[str, Any] = {
        "schema_version": BINDING_SCHEMA_VERSION_V2,
        "binding_kind": "route_b_resolved_task_info_conditioning",
        "identity": {
            "task_name": source["task_name"],
            "task_config": source["task_config"],
            "variant": source["variant"],
            "scope_id": source["scope_id"],
            "representative_target_id": source["representative_target_id"],
            "representative_rollout_id": source["representative_rollout_id"],
            "official_episode_identity_claimed": False,
            "historical_frame_alignment_claimed": False,
        },
        "task_info": {
            "resolved_info": source["resolved_info"],
            "resolved_info_sha256": source["resolved_task_info_sha256"],
        },
        "language": {
            "instruction": official["instruction"],
            "instruction_sha256": official["instruction_sha256"],
            "scope": "resolved_task_info_conditioning_for_route_b_episodes",
            "selection_policy": (
                "first_compatible_official_seen_template_then_official_"
                "object_seen0_v1"
            ),
            "old_episode_identity_or_frame_alignment": False,
        },
        "official_instruction": official["official_instruction"],
        "placeholder_sources": official["placeholder_sources"],
        "official_semantic_provenance_sha256": official[
            "semantic_provenance_sha256"
        ],
        "representative_sources": source["representative_sources"],
        "membership": {
            "target_count": len(member_records),
            "target_id_set_sha256": _sha256_json(
                sorted(str(record["target_id"]) for record in member_records)
            ),
            "episode_source_set_sha256": member_digest,
        },
    }
    value["binding_sha256"] = _sha256_json(value)
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _safe_binding_filename(
    task_name: str,
    variant: str,
    resolved_sha256: str,
    index: int,
) -> str:
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name).strip("._")
    if not safe_task:
        raise ValueError(f"cannot form binding filename from {task_name!r}")
    return f"{index:05d}_{safe_task}_{variant}_{resolved_sha256[:16]}.json"


@dataclass(frozen=True)
class VerifiedRouteBBindingV2:
    key: tuple[str, str, str]
    scope_id: str
    record: dict[str, Any]
    binding: dict[str, Any]
    path: Path
    file_sha256: str


_EPISODE_KEYS = {
    "target_id",
    "task_name",
    "task_config",
    "variant",
    "canonical_episode_id",
    "seed",
    "canonical_rollout_id",
    "resolved_task_info_sha256",
    "source_receipt",
    "candidate_report",
    "canonical_manifest",
    "canonical_rollout_identity",
    "episode_hdf5",
    "scope_id",
    "binding_sha256",
}

_SOURCE_SPEC_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "live_hash_verified",
}


def _recorded_source_path(
    spec: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    """Validate an immutable source reference without touching live storage."""

    if set(spec) != _SOURCE_SPEC_KEYS:
        raise ValueError(f"{label} source provenance has an invalid shape")
    raw_path = spec.get("path")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or raw_path != raw_path.strip()
    ):
        raise ValueError(f"{label}.path must be a non-empty absolute path")
    path = Path(raw_path)
    if not path.is_absolute() or ".." in path.parts or str(path) != raw_path:
        raise ValueError(f"{label}.path is not a canonical absolute path")
    _require_sha256(spec.get("sha256"), f"{label}.sha256")
    size = spec.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{label}.size_bytes must be a non-negative integer")
    if not isinstance(spec.get("live_hash_verified"), bool):
        raise ValueError(f"{label}.live_hash_verified must be boolean")
    return path


def load_verified_route_b_language_manifest_v2(
    manifest_path: str | Path,
    *,
    verify_bound_source_presence: bool = False,
    verify_bound_source_hashes: bool = False,
    verify_hdf5_hashes: bool = False,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], VerifiedRouteBBindingV2]]:
    """Load v2 and verify its immutable closure.

    The default path verifies the manifest/self-digest, every binding sidecar,
    all recorded source-reference shapes and digests, and exact episode
    membership without issuing live filesystem metadata calls for the five
    provenance files attached to every episode.  This keeps resolver, cache,
    statistics and training startup proportional to the immutable bundle.

    ``verify_bound_source_presence`` explicitly checks that every recorded
    source still exists with its recorded size.  Either hash option implies
    that presence/size check.  ``verify_bound_source_hashes`` hashes the four
    small receipt/report/identity sources; ``verify_hdf5_hashes`` independently
    controls the expensive episode-HDF5 hashes.
    """

    path = _regular_file(
        Path(manifest_path), "Route-B language manifest v2"
    )
    manifest = _load_json(path, "Route-B language manifest v2")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION_V2:
        raise ValueError("Route-B language manifest v2 schema mismatch")
    if manifest.get("binding_schema_version") != BINDING_SCHEMA_VERSION_V2:
        raise ValueError("Route-B language binding v2 schema declaration mismatch")
    payload = dict(manifest)
    claimed_manifest_sha256 = _require_sha256(
        payload.pop("manifest_sha256", None), "manifest_sha256"
    )
    if _sha256_json(payload) != claimed_manifest_sha256:
        raise ValueError("Route-B language manifest v2 digest mismatch")
    raw_records = manifest.get("records")
    raw_episodes = manifest.get("episodes")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Route-B language manifest v2 has no records")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("Route-B language manifest v2 has no episodes")
    if manifest.get("binding_count") != len(raw_records):
        raise ValueError("Route-B language manifest v2 binding_count mismatch")
    if manifest.get("episode_count") != len(raw_episodes):
        raise ValueError("Route-B language manifest v2 episode_count mismatch")

    root = path.parent.resolve()
    bindings: dict[
        tuple[str, str, str], VerifiedRouteBBindingV2
    ] = {}
    by_scope: dict[str, VerifiedRouteBBindingV2] = {}
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"language manifest v2 record {index} is not an object")
        record = dict(raw_record)
        task_name = str(record.get("task_name", "")).strip()
        task_config = str(record.get("task_config", "")).strip()
        variant = _variant(task_config)
        resolved_sha256 = _require_sha256(
            record.get("resolved_task_info_sha256"),
            f"record[{index}].resolved_task_info_sha256",
        )
        scope_id = _scope_id(task_name, task_config, resolved_sha256)
        if (
            not task_name
            or record.get("variant") != variant
            or record.get("scope_id") != scope_id
        ):
            raise ValueError(
                f"language manifest v2 task/config/scope mismatch at record {index}"
            )
        key = (task_name, task_config, resolved_sha256)
        if key in bindings or scope_id in by_scope:
            raise ValueError(f"duplicate Route-B v2 language scope: {scope_id}")
        relative = Path(str(record.get("binding_relpath", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("unsafe Route-B v2 binding relative path")
        binding_path = (root / relative).resolve()
        try:
            binding_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Route-B v2 binding escapes manifest root") from exc
        binding_path = _regular_file(binding_path, "Route-B v2 binding")
        binding_file_sha256 = _sha256_file(binding_path)
        if binding_file_sha256 != record.get("binding_file_sha256"):
            raise ValueError(f"Route-B v2 binding file SHA mismatch: {binding_path}")
        binding = _load_json(binding_path, "Route-B v2 binding")
        if (
            binding.get("schema_version") != BINDING_SCHEMA_VERSION_V2
            or binding.get("binding_kind")
            != "route_b_resolved_task_info_conditioning"
        ):
            raise ValueError("Route-B v2 binding schema/kind mismatch")
        binding_payload = dict(binding)
        claimed_binding_sha256 = _require_sha256(
            binding_payload.pop("binding_sha256", None),
            f"binding[{index}].binding_sha256",
        )
        if _sha256_json(binding_payload) != claimed_binding_sha256:
            raise ValueError("Route-B v2 binding digest mismatch")
        if claimed_binding_sha256 != record.get("binding_sha256"):
            raise ValueError("Route-B v2 manifest/binding digest mismatch")
        identity = binding.get("identity")
        if not isinstance(identity, Mapping) or (
            identity.get("task_name") != task_name
            or identity.get("task_config") != task_config
            or identity.get("variant") != variant
            or identity.get("scope_id") != scope_id
            or identity.get("official_episode_identity_claimed") is not False
            or identity.get("historical_frame_alignment_claimed") is not False
        ):
            raise ValueError("Route-B v2 binding identity mismatch")
        task_info = binding.get("task_info")
        if not isinstance(task_info, Mapping):
            raise ValueError("Route-B v2 binding has no task_info")
        resolved_info = task_info.get("resolved_info")
        if (
            not isinstance(resolved_info, Mapping)
            or _task_info_digest(resolved_info) != resolved_sha256
            or task_info.get("resolved_info_sha256") != resolved_sha256
        ):
            raise ValueError("Route-B v2 resolved task_info digest mismatch")
        language = binding.get("language")
        if not isinstance(language, Mapping):
            raise ValueError("Route-B v2 binding has no language")
        instruction = language.get("instruction")
        instruction_sha256 = (
            _sha256_text(instruction) if isinstance(instruction, str) else ""
        )
        if (
            not isinstance(instruction, str)
            or not instruction
            or instruction != instruction.strip()
            or language.get("instruction_sha256") != instruction_sha256
            or record.get("language_instruction_sha256") != instruction_sha256
            or language.get("scope")
            != "resolved_task_info_conditioning_for_route_b_episodes"
            or language.get("old_episode_identity_or_frame_alignment") is not False
        ):
            raise ValueError("Route-B v2 language instruction contract mismatch")
        membership = binding.get("membership")
        if not isinstance(membership, Mapping) or (
            membership.get("target_count") != record.get("member_target_count")
            or membership.get("target_id_set_sha256")
            != record.get("member_target_id_set_sha256")
            or membership.get("episode_source_set_sha256")
            != record.get("episode_source_set_sha256")
        ):
            raise ValueError("Route-B v2 binding membership mismatch")
        verified = VerifiedRouteBBindingV2(
            key=key,
            scope_id=scope_id,
            record=record,
            binding=binding,
            path=binding_path,
            file_sha256=binding_file_sha256,
        )
        bindings[key] = verified
        by_scope[scope_id] = verified

    episode_rows: list[dict[str, Any]] = []
    seen_target_ids: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = {
        scope_id: [] for scope_id in by_scope
    }
    variant_episode_counts: Counter[str] = Counter()
    for index, raw_episode in enumerate(raw_episodes):
        if not isinstance(raw_episode, Mapping) or set(raw_episode) != _EPISODE_KEYS:
            raise ValueError(
                f"Route-B v2 episode record {index} has an invalid shape"
            )
        episode = dict(raw_episode)
        target_id = str(episode.get("target_id", "")).strip()
        task_name = str(episode.get("task_name", "")).strip()
        task_config = str(episode.get("task_config", "")).strip()
        resolved_sha256 = _require_sha256(
            episode.get("resolved_task_info_sha256"),
            f"episode[{index}].resolved_task_info_sha256",
        )
        scope_id = _scope_id(task_name, task_config, resolved_sha256)
        binding = by_scope.get(scope_id)
        if (
            not target_id
            or target_id in seen_target_ids
            or binding is None
            or episode.get("scope_id") != scope_id
            or episode.get("variant") != _variant(task_config)
            or episode.get("binding_sha256")
            != binding.binding["binding_sha256"]
        ):
            raise ValueError(f"Route-B v2 episode binding mismatch at index {index}")
        seen_target_ids.add(target_id)
        for source_key in (
            "source_receipt",
            "candidate_report",
            "canonical_manifest",
            "canonical_rollout_identity",
            "episode_hdf5",
        ):
            spec = episode.get(source_key)
            if not isinstance(spec, Mapping):
                raise ValueError(f"episode[{index}] has no {source_key} provenance")
            label = f"episode[{index}].{source_key}"
            source_path = _recorded_source_path(spec, label=label)
            should_hash = (
                verify_hdf5_hashes
                if source_key == "episode_hdf5"
                else verify_bound_source_hashes
            )
            should_check_presence = (
                verify_bound_source_presence
                or verify_bound_source_hashes
                or verify_hdf5_hashes
            )
            if should_check_presence:
                live_path = _regular_file(source_path, label)
                if spec["size_bytes"] != live_path.stat().st_size:
                    raise ValueError(f"{label} size mismatch")
                if should_hash and _sha256_file(live_path) != spec["sha256"]:
                    raise ValueError(f"{label} SHA mismatch")
        episode_rows.append(episode)
        grouped[scope_id].append(
            {key: value for key, value in episode.items() if key not in {
                "scope_id",
                "binding_sha256",
            }}
        )
        variant_episode_counts[str(episode["variant"])] += 1

    if manifest.get("episode_binding_set_sha256") != _sha256_json(
        sorted(episode_rows, key=lambda row: row["target_id"])
    ):
        raise ValueError("Route-B v2 episode binding set digest mismatch")
    if manifest.get("target_id_set_sha256") != _sha256_json(
        sorted(seen_target_ids)
    ):
        raise ValueError("Route-B v2 target ID set digest mismatch")
    if manifest.get("task_count") != len(
        {str(row["task_name"]) for row in episode_rows}
    ):
        raise ValueError("Route-B v2 task_count mismatch")
    if manifest.get("variant_episode_counts") != dict(
        sorted(variant_episode_counts.items())
    ):
        raise ValueError("Route-B v2 variant_episode_counts mismatch")
    variant_binding_counts = Counter(
        verified.record["variant"] for verified in bindings.values()
    )
    if manifest.get("variant_binding_counts") != dict(
        sorted(variant_binding_counts.items())
    ):
        raise ValueError("Route-B v2 variant_binding_counts mismatch")
    for scope_id, verified in by_scope.items():
        members = grouped[scope_id]
        membership = verified.binding["membership"]
        if (
            len(members) != membership["target_count"]
            or _sha256_json(sorted(row["target_id"] for row in members))
            != membership["target_id_set_sha256"]
            or _member_source_set_sha256(members)
            != membership["episode_source_set_sha256"]
        ):
            raise ValueError(f"Route-B v2 member closure mismatch: {scope_id}")
    return manifest, bindings


def load_route_b_language_manifest_v2_lightweight(
    manifest_path: str | Path,
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str, str], VerifiedRouteBBindingV2],
]:
    """Load the runtime language lookup without cryptographic qualification.

    This path is intentionally limited to training startup after an offline
    publication has already happened.  It validates the manifest/binding
    schemas, safe paths, task identities, resolved task-info payloads and
    non-empty instructions.  It does not recompute manifest, binding, source
    receipt or episode-HDF5 digests, and it does not walk the episode closure.
    The strict loader above remains the default verifier and publication gate.
    """

    path = _regular_file(
        Path(manifest_path), "Route-B language manifest v2"
    )
    manifest = _load_json(path, "Route-B language manifest v2")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION_V2:
        raise ValueError("Route-B language manifest v2 schema mismatch")
    if manifest.get("binding_schema_version") != BINDING_SCHEMA_VERSION_V2:
        raise ValueError(
            "Route-B language binding v2 schema declaration mismatch"
        )
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Route-B language manifest v2 has no records")

    root = path.parent.resolve()
    bindings: dict[
        tuple[str, str, str], VerifiedRouteBBindingV2
    ] = {}
    seen_scopes: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping):
            raise ValueError(
                f"language manifest v2 record {index} is not an object"
            )
        record = dict(raw_record)
        task_name = str(record.get("task_name", "")).strip()
        task_config = str(record.get("task_config", "")).strip()
        resolved_sha256 = str(
            record.get("resolved_task_info_sha256", "")
        ).strip()
        variant = _variant(task_config)
        scope_id = _scope_id(task_name, task_config, resolved_sha256)
        if (
            not task_name
            or not _SHA256_RE.fullmatch(resolved_sha256)
            or record.get("variant") != variant
            or record.get("scope_id") != scope_id
            or scope_id in seen_scopes
        ):
            raise ValueError(
                f"language manifest v2 task/config/scope mismatch at "
                f"record {index}"
            )
        relative = Path(str(record.get("binding_relpath", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("unsafe Route-B v2 binding relative path")
        binding_path = (root / relative).resolve()
        try:
            binding_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "Route-B v2 binding escapes manifest root"
            ) from exc
        binding_path = _regular_file(binding_path, "Route-B v2 binding")
        binding = _load_json(binding_path, "Route-B v2 binding")
        if (
            binding.get("schema_version") != BINDING_SCHEMA_VERSION_V2
            or binding.get("binding_kind")
            != "route_b_resolved_task_info_conditioning"
        ):
            raise ValueError("Route-B v2 binding schema/kind mismatch")
        identity = binding.get("identity")
        if not isinstance(identity, Mapping) or (
            identity.get("task_name") != task_name
            or identity.get("task_config") != task_config
            or identity.get("variant") != variant
            or identity.get("scope_id") != scope_id
        ):
            raise ValueError("Route-B v2 binding identity mismatch")
        task_info = binding.get("task_info")
        if not isinstance(task_info, Mapping):
            raise ValueError("Route-B v2 binding has no task_info")
        resolved_info = task_info.get("resolved_info")
        if (
            not isinstance(resolved_info, Mapping)
            or task_info.get("resolved_info_sha256") != resolved_sha256
        ):
            raise ValueError("Route-B v2 resolved task_info identity mismatch")
        language = binding.get("language")
        if not isinstance(language, Mapping):
            raise ValueError("Route-B v2 binding has no language")
        instruction = language.get("instruction")
        if (
            not isinstance(instruction, str)
            or not instruction
            or instruction != instruction.strip()
        ):
            raise ValueError("Route-B v2 language instruction is invalid")

        runtime_binding = dict(binding)
        runtime_language = dict(language)
        runtime_language["instruction_sha256"] = _sha256_text(instruction)
        runtime_binding["language"] = runtime_language
        claimed_binding_sha = str(binding.get("binding_sha256", ""))
        if not _SHA256_RE.fullmatch(claimed_binding_sha):
            payload = dict(runtime_binding)
            payload.pop("binding_sha256", None)
            claimed_binding_sha = _sha256_json(payload)
        runtime_binding["binding_sha256"] = claimed_binding_sha
        key = (task_name, task_config, resolved_sha256)
        if key in bindings:
            raise ValueError(f"duplicate Route-B v2 language scope: {scope_id}")
        bindings[key] = VerifiedRouteBBindingV2(
            key=key,
            scope_id=scope_id,
            record=record,
            binding=runtime_binding,
            path=binding_path,
            file_sha256=str(record.get("binding_file_sha256", "")),
        )
        seen_scopes.add(scope_id)

    return manifest, bindings


def publish_route_b_language_bundle_v2(
    *,
    episode_receipts_jsonl: str | Path,
    fleet_root: str | Path,
    output_dir: str | Path,
    expected_episode_count: int = 25_416,
    verify_hdf5_hashes: bool = False,
) -> dict[str, Any]:
    """Publish one v2 binding per distinct resolved task-info identity."""

    receipts_path = _regular_file(
        Path(episode_receipts_jsonl), "successful target receipt JSONL"
    )
    root = Path(fleet_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing fleet root: {root}")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite Route-B language bundle v2: {output}"
        )
    if expected_episode_count <= 0:
        raise ValueError("expected_episode_count must be positive")
    raw_rows = _load_jsonl(receipts_path, "successful target receipt")
    if len(raw_rows) != expected_episode_count:
        raise ValueError(
            f"successful receipt count {len(raw_rows)} != "
            f"expected {expected_episode_count}"
        )
    inputs: list[tuple[str, Path]] = []
    seen_targets: set[str] = set()
    seen_receipts: set[Path] = set()
    for index, raw in enumerate(raw_rows):
        if set(raw) != {"target_id", "source_receipt"}:
            raise ValueError(
                f"receipt JSONL record {index} must contain exactly "
                "target_id and source_receipt"
            )
        target_id = str(raw.get("target_id", "")).strip()
        receipt_path = _regular_file(
            Path(str(raw.get("source_receipt", ""))),
            f"source receipt {target_id}",
        )
        try:
            receipt_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"source receipt is outside fleet root: {receipt_path}"
            ) from exc
        if (
            not target_id
            or target_id in seen_targets
            or receipt_path in seen_receipts
        ):
            raise ValueError("duplicate or empty target/receipt in input JSONL")
        seen_targets.add(target_id)
        seen_receipts.add(receipt_path)
        inputs.append((target_id, receipt_path))
    inputs.sort()

    grouped_records: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = {}
    grouped_sources: dict[tuple[str, str, str], dict[str, Any]] = {}
    for target_id, receipt_path in inputs:
        episode_record, source = _episode_source_record(
            target_id=target_id,
            receipt_path=receipt_path,
            fleet_root=root,
            verify_hdf5_hash=verify_hdf5_hashes,
        )
        key = (
            str(source["task_name"]),
            str(source["task_config"]),
            str(source["resolved_task_info_sha256"]),
        )
        previous = grouped_sources.get(key)
        if previous is None:
            grouped_sources[key] = source
            grouped_records[key] = []
        elif (
            previous["resolved_info"] != source["resolved_info"]
            or previous["official"]["instruction_sha256"]
            != source["official"]["instruction_sha256"]
            or previous["official"]["semantic_provenance_sha256"]
            != source["official"]["semantic_provenance_sha256"]
        ):
            raise ValueError(
                "one resolved task-info identity produced inconsistent official "
                f"language provenance: {source['scope_id']}"
            )
        grouped_records[key].append(episode_record)

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex}")
    partial.mkdir()
    try:
        binding_dir = partial / "bindings"
        binding_dir.mkdir()
        records: list[dict[str, Any]] = []
        episodes: list[dict[str, Any]] = []
        for index, key in enumerate(sorted(grouped_records)):
            task_name, task_config, resolved_sha256 = key
            source = grouped_sources[key]
            members = sorted(
                grouped_records[key], key=lambda row: row["target_id"]
            )
            binding = _binding_value(source, member_records=members)
            variant = _variant(task_config)
            filename = _safe_binding_filename(
                task_name, variant, resolved_sha256, index
            )
            binding_path = binding_dir / filename
            _write_json_exclusive(binding_path, binding)
            binding_relpath = binding_path.relative_to(partial).as_posix()
            membership = binding["membership"]
            records.append(
                {
                    "scope_id": source["scope_id"],
                    "task_name": task_name,
                    "task_config": task_config,
                    "variant": variant,
                    "resolved_task_info_sha256": resolved_sha256,
                    "member_target_count": membership["target_count"],
                    "member_target_id_set_sha256": membership[
                        "target_id_set_sha256"
                    ],
                    "episode_source_set_sha256": membership[
                        "episode_source_set_sha256"
                    ],
                    "language_instruction_sha256": binding["language"][
                        "instruction_sha256"
                    ],
                    "binding_relpath": binding_relpath,
                    "binding_file_sha256": _sha256_file(binding_path),
                    "binding_sha256": binding["binding_sha256"],
                }
            )
            for member in members:
                episodes.append(
                    {
                        **member,
                        "scope_id": source["scope_id"],
                        "binding_sha256": binding["binding_sha256"],
                    }
                )
        records.sort(key=lambda row: row["scope_id"])
        episodes.sort(key=lambda row: row["target_id"])
        variant_episode_counts = Counter(
            str(row["variant"]) for row in episodes
        )
        variant_binding_counts = Counter(
            str(row["variant"]) for row in records
        )
        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION_V2,
            "binding_schema_version": BINDING_SCHEMA_VERSION_V2,
            "source_receipt_inventory": {
                "path": str(receipts_path),
                "file_sha256": _sha256_file(receipts_path),
                "expected_episode_count": expected_episode_count,
                "accepted_receipt_schemas": sorted(
                    {NATIVE_RECEIPT_SCHEMA, IMPORTED_RECEIPT_SCHEMA}
                ),
                "episode_hdf5_hashes_live_verified": verify_hdf5_hashes,
            },
            "episode_count": len(episodes),
            "binding_count": len(records),
            "task_count": len({str(row["task_name"]) for row in episodes}),
            "variant_episode_counts": dict(
                sorted(variant_episode_counts.items())
            ),
            "variant_binding_counts": dict(
                sorted(variant_binding_counts.items())
            ),
            "target_id_set_sha256": _sha256_json(
                sorted(str(row["target_id"]) for row in episodes)
            ),
            "episode_binding_set_sha256": _sha256_json(episodes),
            "records": records,
            "episodes": episodes,
        }
        manifest["manifest_sha256"] = _sha256_json(manifest)
        manifest_path = partial / "route_b_language_manifest.json"
        _write_json_exclusive(manifest_path, manifest)
        load_verified_route_b_language_manifest_v2(manifest_path)
        os.replace(partial, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    final_manifest = output / "route_b_language_manifest.json"
    return {
        "status": "PASS",
        "passed": True,
        "output_dir": str(output),
        "manifest": str(final_manifest),
        "manifest_file_sha256": _sha256_file(final_manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "episode_count": manifest["episode_count"],
        "binding_count": manifest["binding_count"],
        "task_count": manifest["task_count"],
        "unique_instruction_count": len(
            {row["language_instruction_sha256"] for row in records}
        ),
        "verify_hdf5_hashes": verify_hdf5_hashes,
    }


__all__ = [
    "BINDING_SCHEMA_VERSION_V2",
    "MANIFEST_SCHEMA_VERSION_V2",
    "VerifiedRouteBBindingV2",
    "load_route_b_language_manifest_v2_lightweight",
    "load_verified_route_b_language_manifest_v2",
    "publish_route_b_language_bundle_v2",
]
