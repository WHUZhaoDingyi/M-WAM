"""Basic, method-agnostic data boundary for future MoMap-WAM training.

Formal publication helpers remain available from this package for backwards
compatibility, but are loaded only when one of those symbols is requested.
Ordinary lightweight Dataset imports therefore do not pull in ``formal_v1``.
"""

from importlib import import_module

from .collator import MoMapCollator
from .dataset import RoboTwinMoMapDataset, finiteize_invalid_momap
from .factory import (
    V1MoMapComponents,
    build_fixed_v1_momap_components,
    build_formal_v1_momap_components,
    build_v1_momap_components,
    create_fixed_v1_momap_dataset,
    create_formal_v1_momap_dataset,
    create_v1_momap_dataset,
)
from .interfaces import CanonicalSampleSource, MoMapTargetBuilder
from .prompt_index import (
    PROMPT_INDEX_SCHEMA_VERSION,
    RouteBPromptIndexResolver,
    prompt_key,
    task_info_key,
)
from .schema import (
    ACTION_TARGET_KEYS,
    FINAL_ALLOWED_KEYS,
    FINAL_OPTIONAL_KEYS,
    FINAL_PERMITTED_KEYS,
    FINAL_REQUIRED_KEYS,
    HEAD_SEG_CANDIDATE_KEYS,
    METADATA_KEY,
    MODEL_INPUT_KEYS,
    SOURCE_CONDITION_KEYS,
    SOURCE_CONDITION_PROVENANCE_FIELDS,
    MOMAP_MASK_KEYS,
    MOMAP_COMPONENT_VALID_KEY,
    MOMAP_LOSS_VALID_KEY,
    MOMAP_TARGET_KEYS,
    OPTIONAL_TENSOR_SAMPLE_KEYS,
    TENSOR_SAMPLE_KEYS,
    ActionTarget,
    ModelInputs,
    MoMapTarget,
    PrivateGeometry,
    SampleRef,
    assert_no_private_or_unknown_fields,
    training_sample_to_dict,
    validate_training_sample,
)
from .source_conditions import (
    COMMON_REFERENCE_CAMERA,
    COMMON_REFERENCE_FRAME,
    SOURCE_CONDITION_SCHEMA_VERSION,
    SOURCE_DEPTH_SEMANTICS,
    SOURCE_SEG_COLOR_SEED,
    SOURCE_SEG_ENCODING,
    ROLE_TRACK_SEGMENTATION_COLOR_SEED,
    ROLE_TRACK_SEGMENTATION_ENCODING,
    PREDICTED_TRACK_SEGMENTATION_ENCODING,
    SourceConditionTensors,
    build_source_condition_tensors,
    encode_predicted_track_segmentation,
    encode_role_track_segmentation,
    encode_episode_scoped_segmentation,
    reconstruct_xyz_from_depth_ray,
    rotate_vectors_to_reference,
    transform_points_to_reference,
)
from .text_cache import CachedTextContextEncoder
from .v1_index import (
    PHYSICAL_CAMERA_NAMES,
    ActionHorizonUnavailable,
    ActionSamplingConfig,
    ActionSelection,
    FASTWAM_K4_HYBRID_FUTURE_HORIZONS_SECONDS,
    FASTWAM_K4_HYBRID_LOGICAL_FUTURE_OFFSETS,
    V1EpisodeRecord,
    V1IndexConfig,
    V1SampleIndex,
    build_v1_sample_index,
    project_v1_sample_index_future_contract,
    resolve_action_selection,
)
from .v1_source import ResolvedLanguage, V1CanonicalSampleSource
from .v1_target_builder import (
    TARGET_REPRESENTATION_CANONICAL_DXYZ_ALL_VIEWS,
    TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV,
    TARGET_REPRESENTATIONS,
    TargetServiceContractError,
    V1CPUMoMapTargetBuilder,
)


_FORMAL_V1_EXPORTS = frozenset(
    {
        "DATASET_RECEIPT_SCHEMA",
        "SOURCE_INVENTORY_SCHEMA",
        "SPLIT_MANIFEST_SCHEMA",
        "FormalV1Artifacts",
        "FormalV1ContractError",
        "build_formal_source_inventory",
        "build_formal_v1_sample_index",
        "formal_v1_index_config",
        "load_formal_split_episode_records",
        "verify_and_freeze_formal_v1",
    }
)


def __getattr__(name: str):
    if name not in _FORMAL_V1_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.formal_v1")
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "ACTION_TARGET_KEYS",
    "FINAL_ALLOWED_KEYS",
    "FINAL_OPTIONAL_KEYS",
    "FINAL_PERMITTED_KEYS",
    "FINAL_REQUIRED_KEYS",
    "HEAD_SEG_CANDIDATE_KEYS",
    "METADATA_KEY",
    "MODEL_INPUT_KEYS",
    "SOURCE_CONDITION_KEYS",
    "SOURCE_CONDITION_PROVENANCE_FIELDS",
    "MOMAP_MASK_KEYS",
    "MOMAP_COMPONENT_VALID_KEY",
    "MOMAP_LOSS_VALID_KEY",
    "MOMAP_TARGET_KEYS",
    "OPTIONAL_TENSOR_SAMPLE_KEYS",
    "TENSOR_SAMPLE_KEYS",
    "ActionTarget",
    "COMMON_REFERENCE_CAMERA",
    "COMMON_REFERENCE_FRAME",
    "ActionSamplingConfig",
    "ActionHorizonUnavailable",
    "ActionSelection",
    "FASTWAM_K4_HYBRID_FUTURE_HORIZONS_SECONDS",
    "FASTWAM_K4_HYBRID_LOGICAL_FUTURE_OFFSETS",
    "CachedTextContextEncoder",
    "CanonicalSampleSource",
    "ModelInputs",
    "MoMapCollator",
    "MoMapTarget",
    "MoMapTargetBuilder",
    "PrivateGeometry",
    "PROMPT_INDEX_SCHEMA_VERSION",
    "RoboTwinMoMapDataset",
    "RouteBPromptIndexResolver",
    "ResolvedLanguage",
    "SampleRef",
    "SOURCE_CONDITION_SCHEMA_VERSION",
    "SOURCE_DEPTH_SEMANTICS",
    "SOURCE_SEG_COLOR_SEED",
    "SOURCE_SEG_ENCODING",
    "ROLE_TRACK_SEGMENTATION_COLOR_SEED",
    "ROLE_TRACK_SEGMENTATION_ENCODING",
    "PREDICTED_TRACK_SEGMENTATION_ENCODING",
    "SourceConditionTensors",
    "TargetServiceContractError",
    "TARGET_REPRESENTATION_CANONICAL_DXYZ_ALL_VIEWS",
    "TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV",
    "TARGET_REPRESENTATIONS",
    "DATASET_RECEIPT_SCHEMA",
    "SOURCE_INVENTORY_SCHEMA",
    "SPLIT_MANIFEST_SCHEMA",
    "FormalV1Artifacts",
    "FormalV1ContractError",
    "V1MoMapComponents",
    "V1CPUMoMapTargetBuilder",
    "V1CanonicalSampleSource",
    "V1EpisodeRecord",
    "V1IndexConfig",
    "V1SampleIndex",
    "PHYSICAL_CAMERA_NAMES",
    "assert_no_private_or_unknown_fields",
    "build_v1_momap_components",
    "build_fixed_v1_momap_components",
    "build_formal_source_inventory",
    "build_formal_v1_momap_components",
    "build_formal_v1_sample_index",
    "build_source_condition_tensors",
    "encode_role_track_segmentation",
    "encode_predicted_track_segmentation",
    "build_v1_sample_index",
    "project_v1_sample_index_future_contract",
    "create_v1_momap_dataset",
    "create_fixed_v1_momap_dataset",
    "create_formal_v1_momap_dataset",
    "finiteize_invalid_momap",
    "encode_episode_scoped_segmentation",
    "load_formal_split_episode_records",
    "prompt_key",
    "resolve_action_selection",
    "reconstruct_xyz_from_depth_ray",
    "rotate_vectors_to_reference",
    "training_sample_to_dict",
    "transform_points_to_reference",
    "validate_training_sample",
    "formal_v1_index_config",
    "task_info_key",
    "verify_and_freeze_formal_v1",
]
