"""Construct the RoboTwin episode reader, motion targets and training dataset."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .collator import MoMapCollator
from .dataset import RoboTwinMoMapDataset
from .prompt_index import RouteBPromptIndexResolver
from .source_conditions import COMMON_REFERENCE_CAMERA
from .text_cache import CachedTextContextEncoder
from .v1_index import (
    PHYSICAL_CAMERA_NAMES,
    ActionSamplingConfig,
    ReaderFactory,
    V1IndexConfig,
    V1SampleIndex,
    build_v1_sample_index,
)
from .v1_source import (
    DEFAULT_PROMPT,
    ContextEncoder,
    LanguageResolver,
    V1CanonicalSampleSource,
)
from .v1_target_builder import ServiceFactory, V1CPUMoMapTargetBuilder
from .v1_target_builder import (
    TARGET_REPRESENTATION_CANONICAL_DXYZ_ALL_VIEWS,
)


@dataclass(frozen=True)
class V1MoMapComponents:
    """Reusable V1 components without imposing an outer Dataset API."""

    source: V1CanonicalSampleSource
    target_builder: V1CPUMoMapTargetBuilder


def _required_paths(
    values: Sequence[str | Path] | None,
    *,
    name: str,
) -> tuple[str | Path, ...]:
    if values is None:
        raise ValueError(f"{name} must be provided")
    if isinstance(values, (str, Path)):
        raise TypeError(f"{name} must be a sequence of paths, not one path string")
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not str(value).strip() for value in result):
        raise ValueError(f"{name} contains an empty path")
    return result


def _index_config(
    *,
    offsets: Sequence[int] | None,
    future_horizons_seconds: Sequence[float] | None,
    future_offset_basis: str | None,
    camera_names: Sequence[str],
    action_mode: str,
    action_horizon: int,
    action_rate_hz: float,
    action_timestamp_tolerance_sec: float,
    proprio_mode: str,
    coordinate_frame: str,
    require_raw_v1: bool,
) -> V1IndexConfig:
    if offsets is None:
        raise ValueError(
            "`offsets` must be explicitly provided; V1 has no implicit K"
        )
    if future_offset_basis is None:
        raise ValueError(
            "`future_offset_basis` must be explicitly provided with `offsets`"
        )
    return V1IndexConfig(
        future_offsets=tuple(offsets),
        future_horizons_seconds=(
            None
            if future_horizons_seconds is None
            else tuple(future_horizons_seconds)
        ),
        future_offset_basis=future_offset_basis,
        camera_names=tuple(camera_names),
        action=ActionSamplingConfig(
            mode=action_mode,
            horizon=action_horizon,
            baseline_rate_hz=action_rate_hz,
            timestamp_tolerance_sec=action_timestamp_tolerance_sec,
        ),
        proprio_mode=proprio_mode,
        coordinate_frame=coordinate_frame,
        require_raw_v1=require_raw_v1,
    )


def _build_components_from_index(
    index: V1SampleIndex,
    *,
    prompt_index_file: str | Path | None,
    text_embedding_cache_dir: str | Path | None,
    text_embedding_preload_fork_cow: bool,
    text_embedding_preload_max_bytes: int | None,
    text_embedding_preload_max_node_bytes: int | None,
    text_embedding_preload_rank_stagger_seconds: float,
    offsets: Sequence[int] | None,
    grid_size: Sequence[int],
    coordinate_frame: str,
    reference_camera_name: str,
    target_backend: str,
    normalization: str,
    proprio_mode: str,
    image_normalization: str,
    evaluate_visibility: bool,
    depth_tolerance_m: float,
    pixel_boundary_ambiguity_px: float,
    target_output_dtype: str,
    target_representation: str,
    context_len: int,
    context_hidden_size: int,
    context_output_dtype: str,
    prompt_template: str,
    max_open_files: int,
    max_open_episodes: int,
    max_cached_contexts: int,
    reader_factory: ReaderFactory | None,
    service_factory: ServiceFactory | None,
    language_resolver: LanguageResolver
    | Callable[[Mapping[str, Any]], str]
    | None,
    context_encoder: ContextEncoder | None,
    defer_text_context: bool,
    pretrained_norm_stats: Mapping[str, Any] | None,
    seg_quality_mix_manifest: str | Path | None,
    seg_quality_mix_seed: int,
    source_head_seg_candidates: bool,
    strict_pred_head_only: bool,
    depth_noise: Mapping[str, Any] | None,
    depth_noise_seed: int,
    max_open_seg_cache_files: int,
    wrist_flow_max_magnitude_native_px: float | None,
    wrist_flow_gt_background_filter: bool = False,
) -> V1MoMapComponents:
    if len(index) <= 0:
        raise ValueError("V1 Dataset must contain at least one legal sample")
    if offsets is None:
        raise ValueError("`offsets` must be explicitly provided")
    if not isinstance(defer_text_context, bool):
        raise TypeError("defer_text_context must be bool")
    if target_backend != "cpu_dataloader":
        raise NotImplementedError(
            "only target_backend='cpu_dataloader' is implemented"
        )
    if normalization != "none":
        raise NotImplementedError(
            "DataLoader normalization must remain 'none'"
        )
    expected_coordinate_frame = f"{reference_camera_name}@source_t0"
    if coordinate_frame != expected_coordinate_frame:
        raise ValueError(
            "the V1 CPU MoMap builder requires reference frame "
            f"{expected_coordinate_frame!r}"
        )
    # Accepted only because the generic runtime injects this constructor field
    # into multiple Dataset types. It never supplies a fallback normalization.
    del pretrained_norm_stats

    resolved_language = language_resolver
    if resolved_language is None:
        if prompt_index_file is None or not str(prompt_index_file).strip():
            raise ValueError(
                "prompt_index_file is required when language_resolver is not injected"
            )
        resolved_language = RouteBPromptIndexResolver(prompt_index_file)

    resolved_context = context_encoder
    if not isinstance(text_embedding_preload_fork_cow, bool):
        raise TypeError("text_embedding_preload_fork_cow must be bool")
    preload_options_requested = (
        text_embedding_preload_max_bytes is not None
        or text_embedding_preload_max_node_bytes is not None
        or text_embedding_preload_rank_stagger_seconds != 0.0
    )
    if not text_embedding_preload_fork_cow and preload_options_requested:
        raise ValueError(
            "preload bounds/stagger require "
            "text_embedding_preload_fork_cow=true"
        )
    if text_embedding_preload_fork_cow and (
        context_encoder is not None or defer_text_context
    ):
        raise ValueError(
            "text embedding preload requires the native non-deferred encoder"
        )
    if resolved_context is None and not defer_text_context:
        if text_embedding_cache_dir is None or not str(
            text_embedding_cache_dir
        ).strip():
            raise ValueError(
                "text_embedding_cache_dir is required when context_encoder is not injected"
            )
        if text_embedding_preload_fork_cow:
            coverage_provider = getattr(
                resolved_language,
                "unique_instructions",
                None,
            )
            if not callable(coverage_provider):
                raise ValueError(
                    "text embedding preload requires prompt-index coverage"
                )
            prompts = tuple(
                prompt_template.format(task=instruction)
                for instruction in coverage_provider()
            )
            from .text_cache_preload import (
                ForkCowPreloadedTextContextEncoder,
            )

            resolved_context = ForkCowPreloadedTextContextEncoder(
                text_embedding_cache_dir,
                context_len=context_len,
                hidden_size=context_hidden_size,
                prompts=prompts,
                preload_max_bytes=text_embedding_preload_max_bytes,
                preload_max_node_bytes=(
                    text_embedding_preload_max_node_bytes
                ),
                preload_rank_stagger_seconds=(
                    text_embedding_preload_rank_stagger_seconds
                ),
            )
        else:
            resolved_context = CachedTextContextEncoder(
                text_embedding_cache_dir,
                context_len=context_len,
                hidden_size=context_hidden_size,
            )

    source_kwargs: dict[str, Any] = {
        "language_resolver": resolved_language,
        "context_encoder": resolved_context,
        "reader_factory": reader_factory,
        "proprio_mode": proprio_mode,
        "image_normalization": image_normalization,
        "condition_grid_size": tuple(grid_size),
        "reference_camera_name": reference_camera_name,
        "prompt_template": prompt_template,
        "max_open_files": max_open_files,
        "max_cached_contexts": max_cached_contexts,
        "context_output_dtype": context_output_dtype,
        "seg_quality_mix_manifest": seg_quality_mix_manifest,
        "seg_quality_mix_seed": seg_quality_mix_seed,
        "source_head_seg_candidates": source_head_seg_candidates,
        "strict_pred_head_only": strict_pred_head_only,
        "depth_noise": depth_noise,
        "depth_noise_seed": depth_noise_seed,
        "max_open_seg_cache_files": max_open_seg_cache_files,
    }
    if defer_text_context:
        source_kwargs["defer_text_context"] = True
    source = V1CanonicalSampleSource(index, **source_kwargs)
    target_builder = V1CPUMoMapTargetBuilder(
        grid_size=tuple(grid_size),
        evaluate_visibility=evaluate_visibility,
        depth_tolerance_m=depth_tolerance_m,
        pixel_boundary_ambiguity_px=pixel_boundary_ambiguity_px,
        output_dtype=target_output_dtype,
        target_representation=target_representation,
        wrist_flow_gt_background_filter=wrist_flow_gt_background_filter,
        wrist_flow_max_magnitude_native_px=(
            wrist_flow_max_magnitude_native_px
        ),
        max_open_episodes=max_open_episodes,
        reference_camera_name=reference_camera_name,
        service_factory=service_factory,
    )
    return V1MoMapComponents(source=source, target_builder=target_builder)


def build_v1_momap_components(
    *,
    episode_paths: Sequence[str | Path] | None = None,
    prompt_index_file: str | Path | None = None,
    text_embedding_cache_dir: str | Path | None = None,
    text_embedding_preload_fork_cow: bool = False,
    text_embedding_preload_max_bytes: int | None = None,
    text_embedding_preload_max_node_bytes: int | None = None,
    text_embedding_preload_rank_stagger_seconds: float = 0.0,
    offsets: Sequence[int] | None = None,
    future_horizons_seconds: Sequence[float] | None = None,
    future_offset_basis: str | None = None,
    camera_names: Sequence[str] = PHYSICAL_CAMERA_NAMES,
    grid_size: Sequence[int] = (240, 320),
    coordinate_frame: str = "head_camera@source_t0",
    reference_camera_name: str = COMMON_REFERENCE_CAMERA,
    target_backend: str = "cpu_dataloader",
    normalization: str = "none",
    action_mode: str = "baseline_rate_resampled@15Hz",
    action_horizon: int = 32,
    action_rate_hz: float = 15.0,
    action_timestamp_tolerance_sec: float = 1.0e-9,
    proprio_mode: str = "last_command_drive_target",
    image_normalization: str = "minus_one_one",
    evaluate_visibility: bool = False,
    depth_tolerance_m: float = 0.005,
    pixel_boundary_ambiguity_px: float = 0.0,
    target_output_dtype: str = "float32",
    target_representation: str = (
        TARGET_REPRESENTATION_CANONICAL_DXYZ_ALL_VIEWS
    ),
    context_len: int = 128,
    context_hidden_size: int = 4096,
    context_output_dtype: str = "float32",
    prompt_template: str = DEFAULT_PROMPT,
    require_raw_v1: bool = True,
    max_open_files: int = 8,
    max_open_episodes: int = 4,
    max_cached_contexts: int = 32,
    reader_factory: ReaderFactory | None = None,
    service_factory: ServiceFactory | None = None,
    language_resolver: LanguageResolver
    | Callable[[Mapping[str, Any]], str]
    | None = None,
    context_encoder: ContextEncoder | None = None,
    defer_text_context: bool = False,
    pretrained_norm_stats: Mapping[str, Any] | None = None,
    seg_quality_mix_manifest: str | Path | None = None,
    seg_quality_mix_seed: int = 43017,
    source_head_seg_candidates: bool = False,
    strict_pred_head_only: bool = False,
    depth_noise: Mapping[str, Any] | None = None,
    depth_noise_seed: int = 43018,
    max_open_seg_cache_files: int = 8,
    wrist_flow_max_magnitude_native_px: float | None = None,
    wrist_flow_gt_background_filter: bool = False,
) -> V1MoMapComponents:
    """Build a direct-HDF5 Dataset for smoke/debug use."""

    resolved_episode_paths = _required_paths(
        episode_paths, name="episode_paths"
    )
    config = _index_config(
        offsets=offsets,
        future_horizons_seconds=future_horizons_seconds,
        future_offset_basis=future_offset_basis,
        camera_names=camera_names,
        action_mode=action_mode,
        action_horizon=action_horizon,
        action_rate_hz=action_rate_hz,
        action_timestamp_tolerance_sec=action_timestamp_tolerance_sec,
        proprio_mode=proprio_mode,
        coordinate_frame=coordinate_frame,
        require_raw_v1=require_raw_v1,
    )
    index = build_v1_sample_index(
        config=config,
        episode_paths=resolved_episode_paths,
        reader_factory=reader_factory,
    )
    return _build_components_from_index(
        index,
        prompt_index_file=prompt_index_file,
        text_embedding_cache_dir=text_embedding_cache_dir,
        text_embedding_preload_fork_cow=text_embedding_preload_fork_cow,
        text_embedding_preload_max_bytes=text_embedding_preload_max_bytes,
        text_embedding_preload_max_node_bytes=(
            text_embedding_preload_max_node_bytes
        ),
        text_embedding_preload_rank_stagger_seconds=(
            text_embedding_preload_rank_stagger_seconds
        ),
        offsets=offsets,
        grid_size=grid_size,
        coordinate_frame=coordinate_frame,
        reference_camera_name=reference_camera_name,
        target_backend=target_backend,
        normalization=normalization,
        proprio_mode=proprio_mode,
        image_normalization=image_normalization,
        evaluate_visibility=evaluate_visibility,
        depth_tolerance_m=depth_tolerance_m,
        pixel_boundary_ambiguity_px=pixel_boundary_ambiguity_px,
        target_output_dtype=target_output_dtype,
        target_representation=target_representation,
        context_len=context_len,
        context_hidden_size=context_hidden_size,
        context_output_dtype=context_output_dtype,
        prompt_template=prompt_template,
        max_open_files=max_open_files,
        max_open_episodes=max_open_episodes,
        max_cached_contexts=max_cached_contexts,
        reader_factory=reader_factory,
        service_factory=service_factory,
        language_resolver=language_resolver,
        context_encoder=context_encoder,
        defer_text_context=defer_text_context,
        pretrained_norm_stats=pretrained_norm_stats,
        seg_quality_mix_manifest=seg_quality_mix_manifest,
        seg_quality_mix_seed=seg_quality_mix_seed,
        source_head_seg_candidates=source_head_seg_candidates,
        strict_pred_head_only=strict_pred_head_only,
        depth_noise=depth_noise,
        depth_noise_seed=depth_noise_seed,
        max_open_seg_cache_files=max_open_seg_cache_files,
        wrist_flow_gt_background_filter=wrist_flow_gt_background_filter,
        wrist_flow_max_magnitude_native_px=(
            wrist_flow_max_magnitude_native_px
        ),
    )


def create_v1_momap_dataset(**kwargs: Any) -> RoboTwinMoMapDataset:
    """Load the geometry, observation and action windows for training."""

    validate_collator_samples = kwargs.pop("validate_collator_samples", True)
    components = build_v1_momap_components(**kwargs)
    return RoboTwinMoMapDataset(
        components.source,
        components.target_builder,
        collator=MoMapCollator(validate_samples=validate_collator_samples),
    )
