"""Legacy Route-B bundle verification for explicit offline tools only.

Ordinary Dataset construction uses :mod:`prompt_index` and must never
import this module. The full-manifest and sidecar SHA checks here are
retained solely for manually invoked asset conversion/audit workflows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .v1_source import ResolvedLanguage

def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root must be an object: {path}")
    return value


class RouteBLanguageBundleResolver:
    """Resolve an instruction from the existing hash-bound Route-B bundle.

    Manifest v1 contains one record per task/config.  Manifest v2 supports the
    full dataset and keys records by task, config and the canonical SHA256 of
    resolved ``task_info``.  Both deliberately avoid claiming that a
    representative language source is frame-aligned with another V1 episode.
    """

    MANIFEST_SCHEMA = "robotwin_route_b_language_manifest_v1"
    BINDING_SCHEMA = "robotwin_route_b_task_language_binding_v1"
    MANIFEST_SCHEMA_V2 = "robotwin_route_b_language_manifest_v2"
    BINDING_SCHEMA_V2 = "robotwin_route_b_task_language_binding_v2"
    VARIANT_BY_CONFIG = {
        "demo_clean": "clean",
        "demo_randomized": "randomized",
    }

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        verification_mode: str = "strict",
    ) -> None:
        if verification_mode not in {"strict", "lightweight"}:
            raise ValueError(
                "verification_mode must be 'strict' or 'lightweight'"
            )
        self.verification_mode = verification_mode
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self._manifest_file_sha256 = _sha256_file(self.manifest_path)
        manifest = _load_json_object(self.manifest_path, "Route-B language manifest")
        self.manifest_schema = str(manifest.get("schema_version", ""))
        self._resolution_cache: dict[
            tuple[str, str, str], ResolvedLanguage
        ] = {}
        self._v2_bindings: dict[tuple[str, str, str], Any] | None = None
        if self.manifest_schema == self.MANIFEST_SCHEMA_V2:
            if self.verification_mode == "strict":
                from .route_b_language_v2 import (
                    load_verified_route_b_language_manifest_v2,
                )

                verified_manifest, self._v2_bindings = (
                    load_verified_route_b_language_manifest_v2(
                        self.manifest_path
                    )
                )
            else:
                from .route_b_language_v2 import (
                    load_route_b_language_manifest_v2_lightweight,
                )

                verified_manifest, self._v2_bindings = (
                    load_route_b_language_manifest_v2_lightweight(
                        self.manifest_path
                    )
                )
            self._manifest_semantic_sha256 = str(
                verified_manifest.get("manifest_sha256")
                or self._manifest_file_sha256
            )
            self._records: dict[tuple[str, str], dict[str, Any]] = {}
            return
        if self.manifest_schema != self.MANIFEST_SCHEMA:
            raise ValueError("Route-B language manifest schema mismatch")
        if self.verification_mode != "strict":
            raise ValueError(
                "lightweight language startup supports only the v2 bundle"
            )
        payload = dict(manifest)
        claimed_digest = payload.pop("manifest_sha256", None)
        if claimed_digest != _sha256_json(payload):
            raise ValueError("Route-B language manifest digest mismatch")
        self._manifest_semantic_sha256 = str(claimed_digest)
        raw_records = manifest.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError("Route-B language manifest has no records")

        records: dict[tuple[str, str], dict[str, Any]] = {}
        for index, raw in enumerate(raw_records):
            if not isinstance(raw, Mapping):
                raise ValueError(f"language manifest record {index} is not an object")
            task_name = str(raw.get("task_name", "")).strip()
            task_config = str(raw.get("task_config", "")).strip()
            variant = str(raw.get("variant", "")).strip()
            expected_variant = self.VARIANT_BY_CONFIG.get(task_config)
            scope_id = f"{task_name}:{variant}"
            if (
                not task_name
                or expected_variant is None
                or variant != expected_variant
                or raw.get("scope_id") != scope_id
            ):
                raise ValueError(
                    f"language manifest task/config/scope mismatch at record {index}"
                )
            key = (task_name, task_config)
            if key in records:
                raise ValueError(f"duplicate language scope: {scope_id}")
            records[key] = dict(raw)
        self._records = records

    def resume_binding(self) -> dict[str, Any]:
        binding = {
            "schema_version": "robotwin_route_b_language_resolver_binding_v1",
            "resolver_adapter": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
            "manifest_path": str(self.manifest_path),
            "manifest_schema_version": self.manifest_schema,
            "manifest_file_sha256": self._manifest_file_sha256,
            "manifest_semantic_sha256": self._manifest_semantic_sha256,
        }
        if self.verification_mode == "lightweight":
            binding["schema_version"] = (
                "robotwin_route_b_language_resolver_binding_v2"
            )
            binding["verification_mode"] = "lightweight"
        return binding

    @staticmethod
    def _current_task_info(metadata: Mapping[str, Any]) -> dict[str, Any]:
        identity = metadata.get("canonical_rollout_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("V1 episode metadata has no canonical_rollout_identity")
        replay_scene = identity.get("replay_scene_normalized")
        if not isinstance(replay_scene, Mapping):
            raise ValueError("V1 episode metadata has no replay_scene_normalized")
        info = replay_scene.get("info")
        if not isinstance(info, Mapping):
            raise ValueError("V1 episode metadata has no resolved task info")
        return dict(info)

    def _binding_path(self, record: Mapping[str, Any]) -> Path:
        relative = Path(str(record.get("binding_relpath", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("unsafe language binding relative path")
        root = self.manifest_path.parent.resolve()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("language binding escapes manifest root") from exc
        if not path.is_file():
            raise FileNotFoundError(f"missing Route-B language binding: {path}")
        if _sha256_file(path) != record.get("binding_file_sha256"):
            raise ValueError(f"Route-B language binding file SHA mismatch: {path}")
        return path

    def resolve_with_provenance(
        self, episode_metadata: Mapping[str, Any]
    ) -> ResolvedLanguage:
        task_name = str(episode_metadata.get("task_name", "")).strip()
        task_config = str(episode_metadata.get("task_config", "")).strip()
        current_task_info = self._current_task_info(episode_metadata)
        current_task_info_sha256 = _sha256_json(current_task_info)
        cache_key = (task_name, task_config, current_task_info_sha256)
        cached = self._resolution_cache.get(cache_key)
        if cached is not None:
            return cached
        if self._v2_bindings is not None:
            verified = self._v2_bindings.get(cache_key)
            if verified is None:
                raise KeyError(
                    "no Route-B language binding for resolved task_info: "
                    f"{task_name}:{task_config}:{current_task_info_sha256}"
                )
            binding = verified.binding
            task_info = binding["task_info"]
            if (
                task_info.get("resolved_info") != current_task_info
                or task_info.get("resolved_info_sha256")
                != current_task_info_sha256
            ):
                raise ValueError(
                    "Route-B v2 language task_info does not match V1 episode"
                )
            language = binding["language"]
            instruction = str(language["instruction"])
            resolution = ResolvedLanguage(
                instruction=instruction,
                source_kind="route_b_resolved_task_info_binding_v2",
                scope_id=verified.scope_id,
                binding_sha256=str(binding["binding_sha256"]),
                instruction_sha256=str(language["instruction_sha256"]),
            )
            self._resolution_cache[cache_key] = resolution
            return resolution
        record = self._records.get((task_name, task_config))
        if record is None:
            raise KeyError(
                f"no Route-B language binding for {task_name}:{task_config}"
            )
        binding = _load_json_object(
            self._binding_path(record), "Route-B language binding"
        )
        if (
            binding.get("schema_version") != self.BINDING_SCHEMA
            or binding.get("binding_kind") != "route_b_task_level_conditioning"
        ):
            raise ValueError("Route-B language binding schema/kind mismatch")
        binding_payload = dict(binding)
        claimed_binding_digest = binding_payload.pop("binding_sha256", None)
        if claimed_binding_digest != _sha256_json(binding_payload):
            raise ValueError("Route-B language binding digest mismatch")
        if claimed_binding_digest != record.get("binding_sha256"):
            raise ValueError("manifest/binding digest mismatch")

        identity = binding.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("Route-B language binding has no identity")
        if (
            identity.get("task_name") != task_name
            or identity.get("task_config") != task_config
            or identity.get("official_episode_identity_claimed") is not False
            or identity.get("historical_frame_alignment_claimed") is not False
        ):
            raise ValueError("Route-B language identity does not match V1 task/config")

        task_info = binding.get("task_info")
        if not isinstance(task_info, Mapping):
            raise ValueError("Route-B language binding has no task_info")
        task_info_payload = dict(task_info)
        task_info_digest = task_info_payload.pop("sha256", None)
        if task_info_digest != _sha256_json(task_info_payload):
            raise ValueError("Route-B language task_info digest mismatch")
        if task_info_digest != record.get("task_info_sha256"):
            raise ValueError("manifest/binding task_info digest mismatch")
        if task_info.get("resolved_info") != current_task_info:
            raise ValueError("Route-B language task_info does not match V1 episode")

        language = binding.get("language")
        if not isinstance(language, Mapping):
            raise ValueError("Route-B language binding has no language record")
        instruction = str(language.get("instruction", "")).strip()
        instruction_digest = _sha256_bytes(instruction.encode("utf-8"))
        if (
            not instruction
            or language.get("instruction_sha256") != instruction_digest
            or record.get("language_instruction_sha256") != instruction_digest
            or language.get("scope")
            != "task_level_conditioning_for_whole_route_b_episode"
            or language.get("old_episode_identity_or_frame_alignment") is not False
        ):
            raise ValueError("Route-B language instruction contract mismatch")
        resolution = ResolvedLanguage(
            instruction=instruction,
            source_kind="route_b_task_level_binding",
            scope_id=str(record["scope_id"]),
            binding_sha256=str(claimed_binding_digest),
            instruction_sha256=instruction_digest,
        )
        self._resolution_cache[cache_key] = resolution
        return resolution

    def resolve(self, episode_metadata: Mapping[str, Any]) -> str:
        return self.resolve_with_provenance(episode_metadata).instruction


__all__ = ["RouteBLanguageBundleResolver"]
