"""Single-file prompt lookup used by ordinary RoboTwin MoMap training.

The prompt index is an actual model-input lookup. It is intentionally not an
integrity manifest: it has no asset digests, sidecar bindings, or completion
marker. Publication from the historical Route-B bundle is an
explicit offline operation; the runtime resolver only reads this one file.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .v1_source import ResolvedLanguage


PROMPT_INDEX_SCHEMA_VERSION = "fastwam_route_b_prompt_index_v1"
_RECORD_KEYS = frozenset(
    {"task_name", "task_config", "task_info_key", "instruction"}
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def task_info_key(value: Mapping[str, Any]) -> str:
    """Return a cheap lookup key for the small resolved task-info object."""

    return hashlib.sha256(
        _canonical_json(dict(value)).encode("utf-8")
    ).hexdigest()


def prompt_key(prompt: str) -> str:
    """Return the existing content-addressed filename key for one prompt."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read prompt index {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"prompt index root must be an object: {path}")
    return value


class RouteBPromptIndexResolver:
    """Resolve instructions from one lightweight JSON prompt index."""

    def __init__(self, index_path: str | Path) -> None:
        if index_path is None or not str(index_path).strip():
            raise ValueError("prompt_index_file must be provided")
        self.index_path = Path(index_path).expanduser().resolve()
        if not self.index_path.is_file():
            raise FileNotFoundError(
                f"missing prompt index: {self.index_path}"
            )
        payload = _load_json(self.index_path)
        if payload.get("schema_version") != PROMPT_INDEX_SCHEMA_VERSION:
            raise ValueError("prompt index schema mismatch")
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("prompt index must contain at least one record")
        if payload.get("record_count") != len(records):
            raise ValueError("prompt index record_count mismatch")

        resolved: dict[tuple[str, str, str], str] = {}
        for index, raw_record in enumerate(records):
            if not isinstance(raw_record, Mapping) or set(raw_record) != set(
                _RECORD_KEYS
            ):
                raise ValueError(
                    f"prompt index record {index} has invalid fields"
                )
            task_name = str(raw_record["task_name"]).strip()
            task_config = str(raw_record["task_config"]).strip()
            info_key = str(raw_record["task_info_key"]).strip()
            instruction = str(raw_record["instruction"])
            if (
                not task_name
                or not task_config
                or len(info_key) != 64
                or any(ch not in "0123456789abcdef" for ch in info_key)
                or not instruction
                or instruction != instruction.strip()
            ):
                raise ValueError(f"prompt index record {index} is invalid")
            key = (task_name, task_config, info_key)
            if key in resolved:
                raise ValueError(
                    f"duplicate prompt index key at record {index}: {key}"
                )
            resolved[key] = instruction
        self._records = resolved

    def unique_instructions(self) -> tuple[str, ...]:
        """Return deterministic prompt-index coverage for opt-in preload."""

        return tuple(sorted(set(self._records.values())))

    @staticmethod
    def _current_task_info(metadata: Mapping[str, Any]) -> dict[str, Any]:
        identity = metadata.get("canonical_rollout_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(
                "V1 episode metadata has no canonical_rollout_identity"
            )
        replay_scene = identity.get("replay_scene_normalized")
        if not isinstance(replay_scene, Mapping):
            raise ValueError(
                "V1 episode metadata has no replay_scene_normalized"
            )
        info = replay_scene.get("info")
        if not isinstance(info, Mapping):
            raise ValueError("V1 episode metadata has no resolved task info")
        return dict(info)

    def resolve_with_provenance(
        self, episode_metadata: Mapping[str, Any]
    ) -> ResolvedLanguage:
        task_name = str(episode_metadata.get("task_name", "")).strip()
        task_config = str(episode_metadata.get("task_config", "")).strip()
        info_key = task_info_key(self._current_task_info(episode_metadata))
        key = (task_name, task_config, info_key)
        instruction = self._records.get(key)
        if instruction is None:
            identity = episode_metadata.get("canonical_rollout_identity")
            episode_identity = "unknown"
            if isinstance(identity, Mapping):
                episode_identity = str(
                    identity.get("canonical_episode_id")
                    or identity.get("canonical_rollout_id")
                    or identity.get("target_id")
                    or "unknown"
                )
            raise KeyError(
                "missing prompt for "
                f"task_info_key={info_key} task={task_name!r} "
                f"config={task_config!r} episode={episode_identity!r}"
            )
        return ResolvedLanguage(
            instruction=instruction,
            source_kind="route_b_prompt_index",
            scope_id=f"{task_name}:{task_config}:{info_key}",
            binding_sha256=None,
            instruction_sha256=prompt_key(instruction),
        )

    def resolve(self, episode_metadata: Mapping[str, Any]) -> str:
        return self.resolve_with_provenance(episode_metadata).instruction

    def __call__(self, episode_metadata: Mapping[str, Any]) -> str:
        return self.resolve(episode_metadata)


def publish_prompt_index_from_route_b_bundle(
    *,
    language_manifest: str | Path,
    output_file: str | Path,
    verification_mode: str = "strict",
) -> Path:
    """Explicitly publish the small runtime index from an offline bundle.

    This function is never called by a Dataset, launcher, runtime, or Trainer.
    It deliberately delegates legacy bundle verification to the offline loader
    and writes only the fields needed for prompt lookup.
    """

    if verification_mode == "strict":
        from .route_b_language_v2 import (
            load_verified_route_b_language_manifest_v2 as load_bundle,
        )
    elif verification_mode == "lightweight":
        from .route_b_language_v2 import (
            load_route_b_language_manifest_v2_lightweight as load_bundle,
        )
    else:
        raise ValueError(
            "verification_mode must be strict or lightweight"
        )

    # lightweight 只解析语言索引和 binding schema，不重算 receipt/HDF5 SHA。
    _, bindings = load_bundle(language_manifest)
    records = []
    for verified in bindings.values():
        binding = verified.binding
        task_info = binding["task_info"]
        records.append(
            {
                "task_name": verified.key[0],
                "task_config": verified.key[1],
                "task_info_key": verified.key[2],
                "instruction": str(binding["language"]["instruction"]),
            }
        )
    records.sort(
        key=lambda row: (
            row["task_name"], row["task_config"], row["task_info_key"]
        )
    )
    payload = {
        "schema_version": PROMPT_INDEX_SCHEMA_VERSION,
        "record_count": len(records),
        "records": records,
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    destination = Path(output_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite prompt index: {destination}"
        )
    pending = destination.parent / (
        f".{destination.name}.pending.{os.getpid()}.{uuid.uuid4().hex}"
    )
    descriptor = os.open(
        pending,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, destination)
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass
    return destination


__all__ = [
    "PROMPT_INDEX_SCHEMA_VERSION",
    "RouteBPromptIndexResolver",
    "prompt_key",
    "publish_prompt_index_from_route_b_bundle",
    "task_info_key",
]
