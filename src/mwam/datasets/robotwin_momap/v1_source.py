"""Worker-safe model-visible reader for canonical RoboTwin raw-v1 episodes.

Only source-time RGB, GT depth/seg-derived canonical XYZ, encoded language,
current proprioception and future action leave this source as model
inputs/labels.  Future simulator geometry remains represented by an opaque
locator consumed by the separate MoMap target builder.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch

from mwam.robotwin_momap.se3 import SE3_VALIDATION_SCHEMA_VERSION
from mwam.robotwin_momap.target_service import uniform_cell_center_indices

from .schema import ActionTarget, ModelInputs, PrivateGeometry, SampleRef
from .source_conditions import (
    COMMON_REFERENCE_CAMERA,
    SOURCE_CONDITION_AUGMENTED_SCHEMA_VERSION,
    SOURCE_CONDITION_EDGE_AUGMENTED_SCHEMA_VERSION,
    SOURCE_CONDITION_SCHEMA_VERSION,
    SOURCE_DEPTH_AUGMENTED_SEMANTICS,
    SOURCE_DEPTH_EDGE_AUGMENTED_SEMANTICS,
    SOURCE_DEPTH_SEMANTICS,
    PREDICTED_TRACK_SEGMENTATION_ENCODING,
    SOURCE_SEG_COLOR_SEED,
    SOURCE_SEG_ENCODING,
    SOURCE_SEG_QUALITY_MIX_ENCODING,
    ROLE_TRACK_SEGMENTATION_ENCODING,
    SourceConditionTensors,
    build_source_condition_tensors,
    encode_predicted_track_segmentation,
    encode_role_track_segmentation,
)
from .source_input_augmentation import (
    DEPTH_EDGE_NOISE_MATH_SCHEMA,
    SourceInputAugmentation,
)
from .v1_index import (
    ActionSelection,
    PHYSICAL_CAMERA_NAMES,
    ReaderFactory,
    V1ReaderLike,
    V1SampleIndex,
    _default_reader_factory,
    resolve_action_selection,
)


DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)
_CONTEXT_OUTPUT_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


# The canonical V1 files use physical camera names.  The official FastWAM
# processor, inherited from its LeRobot input contract, uses these historical
# keys.  Keeping the mapping here makes the storage adapter the only place
# that knows both naming schemes.
FASTWAM_CAMERA_KEY_BY_PHYSICAL = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


class LanguageResolver(Protocol):
    def resolve(self, episode_metadata: Mapping[str, Any]) -> str: ...


class ContextEncoder(Protocol):
    def __call__(self, prompt: str) -> Any: ...


@dataclass(frozen=True)
class ResolvedLanguage:
    instruction: str
    source_kind: str
    scope_id: str
    binding_sha256: str | None
    instruction_sha256: str



def _sha256_bytes(value: bytes) -> str:
    """Hash one short prompt/instruction for its content-addressed key."""

    return hashlib.sha256(value).hexdigest()

class _PIDReaderPool:
    """Small worker-local LRU; inherited readers are never reused after fork."""

    def __init__(self, factory: ReaderFactory, max_open_files: int) -> None:
        if max_open_files <= 0:
            raise ValueError("max_open_files must be positive")
        self.factory = factory
        self.max_open_files = int(max_open_files)
        self._pid = os.getpid()
        self._readers: OrderedDict[Path, V1ReaderLike] = OrderedDict()

    def _reset_after_fork(self) -> None:
        current = os.getpid()
        if current == self._pid:
            return
        for reader in self._readers.values():
            try:
                reader.close()
            except Exception:
                pass
        self._readers.clear()
        self._pid = current

    def get(self, path: str | Path) -> V1ReaderLike:
        self._reset_after_fork()
        resolved = Path(path).expanduser().resolve()
        existing = self._readers.pop(resolved, None)
        if existing is not None:
            self._readers[resolved] = existing
            return existing
        reader = self.factory(resolved)
        self._readers[resolved] = reader
        while len(self._readers) > self.max_open_files:
            _, evicted = self._readers.popitem(last=False)
            evicted.close()
        return reader

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_readers"] = OrderedDict()
        state["_pid"] = os.getpid()
        return state


def _resolve_language(
    resolver: LanguageResolver | Callable[[Mapping[str, Any]], str],
    metadata: Mapping[str, Any],
) -> ResolvedLanguage:
    provenance_method = getattr(resolver, "resolve_with_provenance", None)
    if callable(provenance_method):
        resolution = provenance_method(metadata)
        if not isinstance(resolution, ResolvedLanguage):
            raise TypeError(
                "resolve_with_provenance() must return ResolvedLanguage"
            )
        return resolution
    method = getattr(resolver, "resolve", None)
    value = method(metadata) if callable(method) else resolver(metadata)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("language_resolver must return a non-empty instruction")
    instruction = value.strip()
    task_name = str(metadata.get("task_name", "")).strip() or "unknown_task"
    task_config = str(metadata.get("task_config", "")).strip() or "unknown_config"
    return ResolvedLanguage(
        instruction=instruction,
        source_kind="injected_language_resolver",
        scope_id=f"{task_name}:{task_config}",
        binding_sha256=None,
        instruction_sha256=_sha256_bytes(instruction.encode("utf-8")),
    )


def _encoded_context(
    value: Any,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(value, Mapping):
        context = value.get("context")
        mask = value.get("context_mask", value.get("mask"))
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        context, mask = value
    else:
        raise TypeError(
            "context_encoder must return (context, mask) or a matching mapping"
        )
    if output_dtype not in _CONTEXT_OUTPUT_DTYPES.values():
        raise ValueError(
            "encoded context output_dtype must be torch.float32 or "
            "torch.bfloat16"
        )
    context_tensor = torch.as_tensor(context, dtype=output_dtype)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
    if context_tensor.ndim != 2 or mask_tensor.ndim != 1:
        raise ValueError("encoded context/mask must be [L,D] and [L]")
    if context_tensor.shape[0] != mask_tensor.shape[0]:
        raise ValueError("encoded context/mask length mismatch")
    if not bool(torch.isfinite(context_tensor).all().item()):
        raise ValueError("encoded language context contains NaN/Inf")
    context_tensor = context_tensor.clone()
    context_tensor[~mask_tensor] = 0.0
    # Preserve the official FastWAM/Wan behavior: padded embeddings are zero,
    # then the model receives an all-true context mask.
    return context_tensor, torch.ones_like(mask_tensor)


def _adapter_identifier(value: Any) -> str:
    """Return a stable import-style identifier without instance state."""

    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        adapter_type = type(value)
        module = adapter_type.__module__
        qualname = adapter_type.__qualname__
    if (
        not module
        or not qualname
        or module == "__main__"
        or "<locals>" in qualname
        or "<lambda>" in qualname
    ):
        raise ValueError(
            "resume math adapters must have a stable qualified name"
        )
    return f"{module}.{qualname}"


class V1CanonicalSampleSource:
    """Concrete ``CanonicalSampleSource`` backed by raw-v1 HDF5 episodes."""

    def __init__(
        self,
        index: V1SampleIndex,
        *,
        language_resolver: LanguageResolver
        | Callable[[Mapping[str, Any]], str],
        context_encoder: ContextEncoder | None,
        reader_factory: ReaderFactory | None = None,
        proprio_mode: str | None = None,
        defer_text_context: bool = False,
        image_normalization: str = "minus_one_one",
        condition_grid_size: tuple[int, int] = (240, 320),
        reference_camera_name: str = COMMON_REFERENCE_CAMERA,
        prompt_template: str = DEFAULT_PROMPT,
        max_open_files: int = 8,
        max_cached_contexts: int = 32,
        context_output_dtype: str = "float32",
        seg_quality_mix_manifest: str | Path | None = None,
        seg_quality_mix_seed: int = 43017,
        source_head_seg_candidates: bool = False,
        strict_pred_head_only: bool = False,
        depth_noise: Mapping[str, Any] | None = None,
        depth_noise_seed: int = 43018,
        max_open_seg_cache_files: int = 8,
    ) -> None:
        if not isinstance(index, V1SampleIndex):
            raise TypeError("index must be a V1SampleIndex")
        if not isinstance(defer_text_context, bool):
            raise TypeError("defer_text_context must be bool")
        if not isinstance(source_head_seg_candidates, bool):
            raise TypeError("source_head_seg_candidates must be bool")
        if not isinstance(strict_pred_head_only, bool):
            raise TypeError("strict_pred_head_only must be bool")
        if source_head_seg_candidates and strict_pred_head_only:
            raise ValueError(
                "source_head_seg_candidates and strict_pred_head_only are "
                "mutually exclusive"
            )
        if (
            source_head_seg_candidates or strict_pred_head_only
        ) and seg_quality_mix_manifest is None:
            raise ValueError(
                "Head predicted Seg modes require seg_quality_mix_manifest"
            )
        if context_encoder is None and not defer_text_context:
            raise ValueError(
                "context_encoder is required unless text encoding is explicitly "
                "deferred to the official RobotVideoDataset"
            )
        if context_encoder is not None and not callable(context_encoder):
            raise TypeError("context_encoder must be callable")
        if not callable(language_resolver) and not callable(
            getattr(language_resolver, "resolve", None)
        ):
            raise TypeError("language_resolver must be callable or provide resolve()")
        if "{task}" not in prompt_template:
            raise ValueError("prompt_template must contain {task}")
        if (
            isinstance(max_cached_contexts, bool)
            or not isinstance(max_cached_contexts, int)
            or max_cached_contexts <= 0
        ):
            raise ValueError(
                "max_cached_contexts must be a positive integer"
            )
        if not isinstance(context_output_dtype, str):
            raise TypeError("context_output_dtype must be a string")
        if context_output_dtype not in _CONTEXT_OUTPUT_DTYPES:
            raise ValueError(
                "context_output_dtype must be 'float32' or 'bfloat16'"
            )
        aliases = {
            "measured_qpos_real": "qpos_real",
            "last_command_drive_target": "last_command",
        }
        mode = index.config.proprio_mode if proprio_mode is None else proprio_mode
        mode = aliases.get(mode, mode)
        if mode not in {"qpos_real", "last_command"}:
            raise ValueError(f"unsupported proprio_mode: {mode!r}")
        if image_normalization not in {"minus_one_one", "zero_one"}:
            raise ValueError(
                "image_normalization must be 'minus_one_one' or 'zero_one'"
            )
        if (
            not isinstance(condition_grid_size, tuple)
            or len(condition_grid_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in condition_grid_size
            )
        ):
            raise ValueError(
                "condition_grid_size must be a pair of positive integers"
            )
        if reference_camera_name not in PHYSICAL_CAMERA_NAMES:
            raise ValueError(
                "reference_camera_name must be one of "
                f"{PHYSICAL_CAMERA_NAMES!r}"
            )
        if tuple(index.config.camera_names) != PHYSICAL_CAMERA_NAMES:
            raise ValueError("V1 source requires fixed physical camera order")
        expected_frame = f"{reference_camera_name}@source_t0"
        if index.config.coordinate_frame != expected_frame:
            raise ValueError(
                "V1 source conditions require one common reference frame "
                f"{expected_frame!r}"
            )

        self.index = index
        self._sample_refs = index.sample_refs
        self.language_resolver = language_resolver
        self.context_encoder = context_encoder
        self.defer_text_context = defer_text_context
        self.proprio_mode = mode
        self.image_normalization = image_normalization
        self.condition_grid_size = condition_grid_size
        self.reference_camera_name = reference_camera_name
        self.prompt_template = prompt_template
        self.source_head_seg_candidates = source_head_seg_candidates
        self.strict_pred_head_only = strict_pred_head_only
        self._reader_factory = reader_factory or _default_reader_factory
        self._pool = _PIDReaderPool(self._reader_factory, max_open_files)
        self._cache_pid = os.getpid()
        self._max_cached_episode_entries = max_open_files
        self._max_cached_contexts = max_cached_contexts
        self.context_output_dtype = context_output_dtype
        self._context_output_torch_dtype = _CONTEXT_OUTPUT_DTYPES[
            context_output_dtype
        ]
        self.source_input_augmentation = SourceInputAugmentation(
            seg_quality_mix_manifest=seg_quality_mix_manifest,
            seg_quality_mix_seed=seg_quality_mix_seed,
            depth_noise=depth_noise,
            depth_noise_seed=depth_noise_seed,
            max_open_seg_cache_files=max_open_seg_cache_files,
        )
        self._source_input_epoch = 0
        self._metadata_cache: OrderedDict[
            Path, dict[str, Any]
        ] = OrderedDict()
        self._control_timestamp_cache: OrderedDict[
            Path, np.ndarray
        ] = OrderedDict()
        self._language_resolution_cache: OrderedDict[
            Path, ResolvedLanguage
        ] = OrderedDict()
        self._context_cache: OrderedDict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] = OrderedDict()

    @property
    def sample_refs(self) -> Sequence[SampleRef]:
        return self._sample_refs

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic source-augmentation epoch for new workers."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("source augmentation epoch must be non-negative int")
        self._source_input_epoch = int(epoch)

    @property
    def requires_fork_cow_workers(self) -> bool:
        """Whether workers must inherit the optional preloaded text bank."""

        return bool(
            getattr(self.context_encoder, "requires_fork_cow_workers", False)
        )

    def resume_math_contract(self) -> dict[str, Any]:
        """Return only sample-selection and model-input mathematics."""

        config = self.index.config
        action = config.action
        contract = {
            "schema_version": "robotwin_momap_v1_source_math_v1",
            "source_adapter": _adapter_identifier(type(self)),
            "index_config": {
                "future_offsets": list(config.future_offsets),
                "future_horizons_seconds": (
                    None
                    if config.future_horizons_seconds is None
                    else list(config.future_horizons_seconds)
                ),
                "future_offset_basis": config.future_offset_basis,
                "camera_names": list(config.camera_names),
                "action": {
                    "mode": action.mode,
                    "horizon": action.horizon,
                    "baseline_rate_hz": action.baseline_rate_hz,
                    "timestamp_tolerance_sec": (
                        action.timestamp_tolerance_sec
                    ),
                },
                "proprio_mode": config.proprio_mode,
                "coordinate_frame": config.coordinate_frame,
                "require_raw_v1": config.require_raw_v1,
            },
            "tensor_semantics": {
                "image_normalization": self.image_normalization,
                "condition_grid_size": list(self.condition_grid_size),
                "reference_camera_name": self.reference_camera_name,
                "prompt_template": self.prompt_template,
                "context_output_dtype": self.context_output_dtype,
                "defer_text_context": self.defer_text_context,
                "language_resolver_adapter": _adapter_identifier(
                    self.language_resolver
                ),
                "context_encoder_adapter": (
                    None
                    if self.context_encoder is None
                    else _adapter_identifier(self.context_encoder)
                ),
                "reader_factory_adapter": _adapter_identifier(
                    self._reader_factory
                ),
                "source_head_seg_candidates": (
                    self.source_head_seg_candidates
                ),
            },
        }
        if self.strict_pred_head_only:
            contract["tensor_semantics"]["strict_pred_head_only"] = True
        augmentation_contract = self.source_input_augmentation.math_contract()
        if (
            augmentation_contract is not None
            and (
                self.source_head_seg_candidates
                or self.strict_pred_head_only
            )
            and self.source_input_augmentation.segmentation is not None
        ):
            augmentation_contract = dict(augmentation_contract)
            if self.source_head_seg_candidates:
                augmentation_contract["segmentation"] = (
                    self.source_input_augmentation.segmentation
                    .head_candidate_math_contract()
                )
            else:
                segmentation = self.source_input_augmentation.segmentation
                augmentation_contract["segmentation"] = {
                    "schema_version": (
                        "robotwin_momap_pred_head_strict_math_v1"
                    ),
                    "selection_unit": "always_predicted_head_only",
                    "camera_scope": COMMON_REFERENCE_CAMERA,
                    "policy_content_digest": (
                        segmentation.policy_content_digest
                    ),
                    "predicted_seg_fields": [
                        "pred/role_id",
                        "pred/track_id",
                    ],
                    "role_usage": "validation_only_not_model_visible",
                    "encoding": PREDICTED_TRACK_SEGMENTATION_ENCODING,
                    "wrist_segmentation": "zero",
                    "missing_or_invalid_cache_policy": (
                        "raise_no_gt_fallback"
                    ),
                }
        if augmentation_contract is not None:
            contract["source_input_augmentation"] = augmentation_contract
        return contract



    def _reset_caches_after_fork(self) -> None:
        current = os.getpid()
        if current != self._cache_pid:
            self._metadata_cache.clear()
            self._control_timestamp_cache.clear()
            self._language_resolution_cache.clear()
            self._context_cache.clear()
            self._cache_pid = current

    def _reader(self, sample_ref: SampleRef) -> V1ReaderLike:
        self._reset_caches_after_fork()
        return self._pool.get(sample_ref.episode_path)

    @staticmethod
    def _lru_store(
        cache: OrderedDict[Any, Any],
        key: Any,
        value: Any,
        *,
        limit: int,
    ) -> Any:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _metadata(
        self, sample_ref: SampleRef, reader: V1ReaderLike
    ) -> dict[str, Any]:
        path = Path(sample_ref.episode_path).expanduser().resolve()
        try:
            value = self._metadata_cache.pop(path)
        except KeyError:
            value = reader.metadata
        return self._lru_store(
            self._metadata_cache,
            path,
            value,
            limit=self._max_cached_episode_entries,
        )

    def _control_timestamps(
        self, sample_ref: SampleRef, reader: V1ReaderLike
    ) -> np.ndarray:
        path = Path(sample_ref.episode_path).expanduser().resolve()
        try:
            value = self._control_timestamp_cache.pop(path)
        except KeyError:
            value = np.asarray(
                reader.read_control_stream_fields(
                    fields=("timestamp_before_sec",)
                )["timestamp_before_sec"],
                dtype=np.float64,
            )
            expected = (int(reader.control_stream_count),)
            if value.shape != expected:
                raise ValueError(
                    f"control timestamp shape={value.shape}, expected={expected}"
                )
            if not np.isfinite(value).all() or (
                value.size > 1 and np.any(np.diff(value) <= 0)
            ):
                raise ValueError(
                    "control timestamps must be finite and strictly increasing"
                )
            value.setflags(write=False)
        return self._lru_store(
            self._control_timestamp_cache,
            path,
            value,
            limit=self._max_cached_episode_entries,
        )

    def _language_resolution(
        self,
        sample_ref: SampleRef,
        reader: V1ReaderLike,
    ) -> ResolvedLanguage:
        path = Path(sample_ref.episode_path).expanduser().resolve()
        try:
            resolution = self._language_resolution_cache.pop(path)
        except KeyError:
            resolution = _resolve_language(
                self.language_resolver,
                self._metadata(sample_ref, reader),
            )
        return self._lru_store(
            self._language_resolution_cache,
            path,
            resolution,
            limit=self._max_cached_episode_entries,
        )

    def resolve_action_selection(self, sample_ref: SampleRef) -> ActionSelection:
        reader = self._reader(sample_ref)
        selection = resolve_action_selection(
            reader,
            sample_ref.source_index,
            self.index.config.action,
            control_timestamps_before_sec=self._control_timestamps(
                sample_ref, reader
            ),
            control_timestamps_validated=True,
        )
        if selection.indices != sample_ref.action_indices:
            raise ValueError("SampleRef action_indices differ from V1 selection")
        if not np.array_equal(
            np.asarray(selection.query_times_sec, dtype=np.float64),
            np.asarray(sample_ref.action_query_times_sec, dtype=np.float64),
        ):
            raise ValueError("SampleRef action query times differ from V1 selection")
        if sample_ref.action_sampling_mode != self.index.config.action.mode:
            raise ValueError("SampleRef action sampling mode differs from V1 config")
        if (
            sample_ref.action_start != selection.indices[0]
            or sample_ref.action_stop != selection.indices[-1] + 1
        ):
            raise ValueError("SampleRef action range does not enclose selected rows")
        return selection

    def _read_raw_images(
        self, frame: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Return the exact uint8 image mapping consumed by FastWAMProcessor."""

        observations = frame.get("observations")
        if not isinstance(observations, Mapping):
            raise ValueError("V1 frame has no observations mapping")
        images: dict[str, torch.Tensor] = {}
        for camera_name in PHYSICAL_CAMERA_NAMES:
            camera = observations.get(camera_name)
            if not isinstance(camera, Mapping) or "rgb" not in camera:
                raise ValueError(f"V1 frame lacks RGB for {camera_name}")
            rgb = np.asarray(camera["rgb"])
            if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
                raise ValueError(
                    f"{camera_name} RGB must be uint8[240,320,3], "
                    f"got {rgb.shape} {rgb.dtype}"
                )
            key = FASTWAM_CAMERA_KEY_BY_PHYSICAL[camera_name]
            images[key] = (
                torch.from_numpy(np.array(rgb, copy=True))
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
        return images

    def _proprio_from_frame(self, frame: Mapping[str, Any]) -> torch.Tensor:
        robot_state = frame.get("robot_state")
        if not isinstance(robot_state, Mapping):
            raise ValueError("V1 frame has no robot_state mapping")
        field = "qpos_real" if self.proprio_mode == "qpos_real" else "drive_target"
        value = np.asarray(robot_state.get(field), dtype=np.float32)
        if value.shape != (14,) or not np.isfinite(value).all():
            raise ValueError(f"robot_state.{field} must be finite float[14]")
        return torch.from_numpy(np.array(value, copy=True)).view(1, 14)

    def _read_source_frame(
        self,
        sample_ref: SampleRef,
        reader: V1ReaderLike,
        *,
        include_conditions: bool = False,
    ) -> Mapping[str, Any]:
        proprio_field = (
            "qpos_real" if self.proprio_mode == "qpos_real" else "drive_target"
        )
        return reader.read_frame_fields(
            sample_ref.source_index,
            camera_names=PHYSICAL_CAMERA_NAMES,
            observation_fields=("rgb",),
            private_geometry_fields=(
                (
                    "depth_m",
                    "ray_xy",
                    "ray_valid",
                    "position_valid",
                    "actor_seg_raw",
                    "T_cam_to_world",
                )
                if include_conditions
                else ()
            ),
            timeline_fields=(),
            robot_state_fields=(proprio_field,),
            entity_fields=(),
        )

    def _source_conditions(
        self,
        sample_ref: SampleRef,
        frame: Mapping[str, Any],
    ) -> SourceConditionTensors:
        prepared_frame = frame
        depth_policy = self.source_input_augmentation.depth
        if depth_policy is not None:
            prepared_frame = depth_policy.augment_frame(
                prepared_frame,
                episode_id=sample_ref.episode_id,
                source_index=sample_ref.source_index,
                camera_names=sample_ref.camera_names,
                epoch=self._source_input_epoch,
            )
        segmentation_override = None
        seg_policy = self.source_input_augmentation.segmentation
        if (
            seg_policy is not None
            and not self.source_head_seg_candidates
            and not self.strict_pred_head_only
        ):
            decision = seg_policy.decide(
                episode_id=sample_ref.episode_id,
                source_index=sample_ref.source_index,
                camera_names=sample_ref.camera_names,
                epoch=self._source_input_epoch,
            )
            segmentation_override = decision.predicted_by_camera
        return build_source_condition_tensors(
            prepared_frame,
            camera_names=sample_ref.camera_names,
            episode_identity=sample_ref.episode_id,
            grid_size=self.condition_grid_size,
            reference_camera_name=self.reference_camera_name,
            image_normalization=self.image_normalization,
            segmentation_override_by_camera=segmentation_override,
        )

    def _head_segmentation_candidates(
        self,
        sample_ref: SampleRef,
        conditions: SourceConditionTensors,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return whole-Head GT/Pred candidates; never choose between them.

        GT currently preserves the canonical raw-actor instance RGB encoding,
        including unrelated Head objects.  Pred uses the V3 cache's explicit
        role+track encoding.  The distinction is intentional and versioned;
        a future GT role sidecar can replace the GT encoder without pretending
        that task roles are inferable from raw actor IDs alone.
        """

        if not self.source_head_seg_candidates:
            raise RuntimeError("Head Seg candidates are not enabled")
        seg_policy = self.source_input_augmentation.segmentation
        if seg_policy is None:
            raise RuntimeError("Head Seg candidates require a cache policy")
        head_index = tuple(sample_ref.camera_names).index(
            self.reference_camera_name
        )
        if self.reference_camera_name != COMMON_REFERENCE_CAMERA:
            raise ValueError(
                "Head Seg candidate mode currently requires head_camera as "
                "the reference camera"
            )
        candidate = seg_policy.load_head_role_track_candidate(
            episode_id=sample_ref.episode_id,
            source_index=sample_ref.source_index,
            camera_names=sample_ref.camera_names,
        )
        if not candidate.available:
            reason = candidate.provenance.get("reason", "unknown")
            raise RuntimeError(
                "strict predicted Head Seg candidate unavailable: "
                f"episode={sample_ref.episode_id}, "
                f"source_index={sample_ref.source_index}, reason={reason}"
            )
        assert candidate.role_id is not None
        assert candidate.track_id is not None
        pred_hwc = encode_role_track_segmentation(
            candidate.role_id,
            candidate.track_id,
            episode_identity=sample_ref.episode_id,
            camera_name=self.reference_camera_name,
        )
        rows, columns = uniform_cell_center_indices(
            tuple(int(value) for value in pred_hwc.shape[:2]),
            self.condition_grid_size,
        )
        pred_hwc = pred_hwc[rows[:, None], columns[None, :], :]
        pred = torch.from_numpy(
            np.ascontiguousarray(np.moveaxis(pred_hwc, -1, 0), dtype=np.float32)
        )
        gt = conditions.source_seg_condition[head_index].clone()
        return gt, pred

    def _strict_predicted_head_segmentation(
        self,
        sample_ref: SampleRef,
        conditions: SourceConditionTensors,
    ) -> torch.Tensor:
        """Return one semantic-free predicted Head instance RGB condition.

        Role IDs are validated against track IDs but never enter the encoded
        feature.  A missing or corrupt cache is a hard error: this mode has no
        GT fallback and does not consult the manifest's historical ``p_gt``.
        """

        if not self.strict_pred_head_only:
            raise RuntimeError("strict predicted Head Seg is not enabled")
        if self.reference_camera_name != COMMON_REFERENCE_CAMERA:
            raise ValueError(
                "strict_pred_head_only requires head_camera as the reference"
            )
        seg_policy = self.source_input_augmentation.segmentation
        if seg_policy is None:
            raise RuntimeError(
                "strict_pred_head_only requires a predicted Seg cache policy"
            )
        candidate = seg_policy.load_head_role_track_candidate(
            episode_id=sample_ref.episode_id,
            source_index=sample_ref.source_index,
            camera_names=sample_ref.camera_names,
        )
        if not candidate.available:
            reason = candidate.provenance.get("reason", "unknown")
            raise RuntimeError(
                "strict predicted Head Seg unavailable with no GT fallback: "
                f"episode={sample_ref.episode_id}, "
                f"source_index={sample_ref.source_index}, reason={reason}"
            )
        assert candidate.role_id is not None
        assert candidate.track_id is not None
        pred_hwc = encode_predicted_track_segmentation(
            candidate.role_id,
            candidate.track_id,
            episode_identity=sample_ref.episode_id,
            camera_name=self.reference_camera_name,
        )
        rows, columns = uniform_cell_center_indices(
            tuple(int(value) for value in pred_hwc.shape[:2]),
            self.condition_grid_size,
        )
        pred_hwc = np.array(
            pred_hwc[rows[:, None], columns[None, :], :],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        head_index = tuple(sample_ref.camera_names).index(
            self.reference_camera_name
        )
        head_valid = (
            conditions.source_condition_valid[head_index, 0].numpy()
        )
        pred_hwc[~head_valid] = 0.0
        return torch.from_numpy(
            np.ascontiguousarray(np.moveaxis(pred_hwc, -1, 0))
        )

    def _validate_last_command_proprio(
        self,
        sample_ref: SampleRef,
        reader: V1ReaderLike,
        proprio: torch.Tensor,
    ) -> None:
        if self.proprio_mode != "last_command":
            return
        _, applied_stop = reader.saved_frame_control_span(sample_ref.source_index)
        if applied_stop <= 0:
            raise ValueError(
                "last_command proprio requires a previously applied control"
            )
        expected = np.asarray(
            reader.read_control_stream_fields(
                applied_stop - 1,
                applied_stop,
                fields=("drive_target",),
            )["drive_target"],
            dtype=np.float32,
        )
        if expected.shape != (1, 14) or not np.array_equal(
            proprio.numpy(), expected
        ):
            raise ValueError(
                "saved-frame drive_target differs from last applied control"
            )

    def _context(
        self, sample_ref: SampleRef, reader: V1ReaderLike
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        resolution = self._language_resolution(sample_ref, reader)
        prompt = self.prompt_template.format(task=resolution.instruction)
        if self.context_encoder is None:
            raise RuntimeError(
                "text context was deferred to RobotVideoDataset for this source"
            )
        if self.requires_fork_cow_workers:
            preload_view = getattr(self.context_encoder, "preloaded_view", None)
            if not callable(preload_view):
                raise RuntimeError(
                    "fork-COW text encoder must provide preloaded_view()"
                )
            encoded = _encoded_context(
                preload_view(prompt),
                output_dtype=self._context_output_torch_dtype,
            )
            context, mask = encoded
            clone_output = False
        else:
            try:
                encoded = self._context_cache.pop(prompt)
            except KeyError:
                try:
                    raw_context = self.context_encoder(prompt)
                except FileNotFoundError as exc:
                    prompt_key = _sha256_bytes(prompt.encode("utf-8"))
                    raise FileNotFoundError(
                        "missing prompt embedding for "
                        f"prompt_key={prompt_key} "
                        f"episode={sample_ref.episode_path} "
                        f"source_index={sample_ref.source_index}: {exc}"
                    ) from exc
                encoded = _encoded_context(
                    raw_context,
                    output_dtype=self._context_output_torch_dtype,
                )
            context, mask = self._lru_store(
                self._context_cache,
                prompt,
                encoded,
                limit=self._max_cached_contexts,
            )
            clone_output = True
        cache_path_method = getattr(self.context_encoder, "cache_path", None)
        cache_key = None
        if callable(cache_path_method):
            cache_key = Path(cache_path_method(prompt)).name
        provenance: dict[str, object] = {
            "language_source_kind": resolution.source_kind,
            "language_scope_id": resolution.scope_id,
            "language_binding_sha256": resolution.binding_sha256,
            "language_instruction_sha256": resolution.instruction_sha256,
            "language_prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            "text_embedding_cache_key": cache_key,
        }
        if clone_output:
            context = context.clone()
            mask = mask.clone()
        return context, mask, provenance

    def load_model_inputs(self, sample_ref: SampleRef) -> ModelInputs:
        reader = self._reader(sample_ref)
        frame = self._read_source_frame(
            sample_ref, reader, include_conditions=True
        )
        proprio = self._proprio_from_frame(frame)
        self._validate_last_command_proprio(sample_ref, reader, proprio)
        context, context_mask, provenance = self._context(sample_ref, reader)
        conditions = self._source_conditions(sample_ref, frame)
        source_head_seg_gt_condition = None
        source_head_seg_pred_condition = None
        if self.source_head_seg_candidates:
            (
                source_head_seg_gt_condition,
                source_head_seg_pred_condition,
            ) = self._head_segmentation_candidates(sample_ref, conditions)
            head_index = tuple(sample_ref.camera_names).index(
                self.reference_camera_name
            )
            head_only_seg = torch.zeros_like(
                conditions.source_seg_condition
            )
            head_only_seg[head_index] = source_head_seg_gt_condition
            conditions = replace(
                conditions,
                source_seg_condition=head_only_seg,
            )
        elif self.strict_pred_head_only:
            strict_head_seg = self._strict_predicted_head_segmentation(
                sample_ref,
                conditions,
            )
            head_index = tuple(sample_ref.camera_names).index(
                self.reference_camera_name
            )
            head_only_seg = torch.zeros_like(
                conditions.source_seg_condition
            )
            head_only_seg[head_index] = strict_head_seg
            conditions = replace(
                conditions,
                source_seg_condition=head_only_seg,
            )
        augmentation_enabled = self.source_input_augmentation.enabled
        depth_policy = self.source_input_augmentation.depth
        edge_depth_enabled = bool(
            depth_policy is not None
            and depth_policy.math_schema_version == DEPTH_EDGE_NOISE_MATH_SCHEMA
        )
        return ModelInputs(
            source_rgb=conditions.source_rgb,
            source_depth_m=conditions.source_depth_m,
            source_depth_valid=conditions.source_depth_valid,
            source_seg_condition=conditions.source_seg_condition,
            source_xyz_ref=conditions.source_xyz_ref,
            source_condition_valid=conditions.source_condition_valid,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            provenance=provenance,
            condition_provenance={
                "source_condition_schema_version": (
                    SOURCE_CONDITION_EDGE_AUGMENTED_SCHEMA_VERSION
                    if edge_depth_enabled
                    else (
                        SOURCE_CONDITION_AUGMENTED_SCHEMA_VERSION
                        if augmentation_enabled
                        else SOURCE_CONDITION_SCHEMA_VERSION
                    )
                ),
                "source_condition_reference_camera": (
                    self.reference_camera_name
                ),
                "source_condition_grid_size": self.condition_grid_size,
                "source_seg_encoding": (
                    PREDICTED_TRACK_SEGMENTATION_ENCODING
                    if self.strict_pred_head_only
                    else (
                        "head_gt_actor_rgb_plus_pred_"
                        f"{ROLE_TRACK_SEGMENTATION_ENCODING}_candidates_v1"
                    )
                    if self.source_head_seg_candidates
                    else (
                        SOURCE_SEG_QUALITY_MIX_ENCODING
                        if self.source_input_augmentation.segmentation is not None
                        else SOURCE_SEG_ENCODING
                    )
                ),
                "source_seg_color_seed": SOURCE_SEG_COLOR_SEED,
                "source_depth_semantics": (
                    SOURCE_DEPTH_EDGE_AUGMENTED_SEMANTICS
                    if edge_depth_enabled
                    else (
                        SOURCE_DEPTH_AUGMENTED_SEMANTICS
                        if depth_policy is not None
                        else SOURCE_DEPTH_SEMANTICS
                    )
                ),
            },
            source_head_seg_gt_condition=(
                source_head_seg_gt_condition
            ),
            source_head_seg_pred_condition=(
                source_head_seg_pred_condition
            ),
        )

    def load_source_conditions(
        self,
        sample_ref: SampleRef,
    ) -> SourceConditionTensors:
        """Load only source-time geometry conditions for offline statistics.

        This deliberately skips language encoding, proprioception validation,
        language/proprio target preparation, future RGB, future geometry,
        actions, and the MoMap target builder. It
        is the narrow source-only boundary used by fixed-global XYZ
        calibration tools.
        """

        reader = self._reader(sample_ref)
        frame = self._read_source_frame(
            sample_ref,
            reader,
            include_conditions=True,
        )
        return self._source_conditions(sample_ref, frame)

    def load_instruction(self, sample_ref: SampleRef) -> str:
        """Resolve one episode-level instruction without reading RGB/geometry."""

        reader = self._reader(sample_ref)
        return self._language_resolution(sample_ref, reader).instruction

    def load_fastwam_raw_sample(
        self,
        sample_ref: SampleRef,
        *,
        sample_index: int,
    ) -> dict[str, Any]:
        """Adapt one V1 source frame to the official processor input contract.

        Only current RGB, current proprioception, language and the future
        action label leave this method.  Phase-2 geometry remains behind the
        separate ``load_private_geometry`` boundary.
        """

        reader = self._reader(sample_ref)
        frame = self._read_source_frame(sample_ref, reader)
        proprio = self._proprio_from_frame(frame)
        self._validate_last_command_proprio(sample_ref, reader, proprio)
        action_target = self.load_action_target(sample_ref)
        instruction = self.load_instruction(sample_ref)
        return {
            "idx": int(sample_index),
            "task": instruction,
            "images": self._read_raw_images(frame),
            "state": {"default": proprio},
            "action": {"default": action_target.action},
            "image_is_pad": torch.zeros((1,), dtype=torch.bool),
            "state_is_pad": torch.zeros((1,), dtype=torch.bool),
            "action_is_pad": action_target.action_is_pad,
        }

    def load_action_target(self, sample_ref: SampleRef) -> ActionTarget:
        reader = self._reader(sample_ref)
        selection = self.resolve_action_selection(sample_ref)
        start = selection.indices[0]
        stop = selection.indices[-1] + 1
        values = np.asarray(
            reader.read_control_stream_fields(
                start,
                stop,
                fields=("drive_target",),
            )["drive_target"],
            dtype=np.float32,
        )
        relative = np.asarray(selection.indices, dtype=np.int64) - start
        action = values[relative]
        expected = (self.index.config.action.horizon, 14)
        if action.shape != expected or not np.isfinite(action).all():
            raise ValueError(
                f"future action must be finite {expected}, got {action.shape}"
            )
        tensor = torch.from_numpy(np.array(action, copy=True))
        return ActionTarget(
            action=tensor,
            action_is_pad=torch.zeros((tensor.shape[0],), dtype=torch.bool),
        )

    def load_private_geometry(self, sample_ref: SampleRef) -> PrivateGeometry:
        """Return an opaque locator, never simulator arrays or model inputs."""

        return PrivateGeometry(
            payload={
                "episode_path": str(Path(sample_ref.episode_path).resolve()),
                "source_index": int(sample_ref.source_index),
                "target_indices": tuple(int(x) for x in sample_ref.target_indices),
                "camera_names": tuple(sample_ref.camera_names),
                "coordinate_frame": sample_ref.coordinate_frame,
            }
        )

    def close(self) -> None:
        self._pool.close()
        self.source_input_augmentation.close()
        self._metadata_cache.clear()
        self._control_timestamp_cache.clear()
        self._language_resolution_cache.clear()
        self._context_cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_pool"] = _PIDReaderPool(self._reader_factory, self._pool.max_open_files)
        state["_metadata_cache"] = OrderedDict()
        state["_control_timestamp_cache"] = OrderedDict()
        state["_language_resolution_cache"] = OrderedDict()
        state["_context_cache"] = OrderedDict()
        state["_cache_pid"] = os.getpid()
        return state

    def __del__(self) -> None:  # pragma: no cover - best effort at shutdown.
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "ContextEncoder",
    "DEFAULT_PROMPT",
    "FASTWAM_CAMERA_KEY_BY_PHYSICAL",
    "LanguageResolver",
    "ResolvedLanguage",
    "V1CanonicalSampleSource",
]
