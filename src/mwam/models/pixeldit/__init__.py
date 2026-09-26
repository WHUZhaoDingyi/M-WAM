"""PixelDiT world model, K-flow contract, and one-way Action MoT bridge."""

from importlib import import_module

from .conditioning import (
    CANONICAL_REFERENCE_FRAME,
    WORLD_CONDITIONING_V2,
    WorldConditioningContract,
)
from .factory import create_fastwam_pixeldit
from .fastwam_pixeldit import FastWAMPixelDiT
from .geometry_normalization import (
    PER_SAMPLE_P99_GEOMETRY_NORMALIZATION_MODE,
    PER_SAMPLE_P99_GEOMETRY_NORMALIZATION_SCHEMA_VERSION,
    GeometryNormalization,
    GeometryNormalizationContract,
    PerSampleP99GeometryNormalizationContract,
    SOURCE_XYZ_MEAN_STD_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_MODE,
    SOURCE_XYZ_MEAN_STD_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION,
    SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract,
)
from .kflow import (
    KFLOW_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL16_RAW_FLOW_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_K1_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
    LEGACY_DPIX24_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
    LEGACY_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
    ROBOTWIN_FASTWAM_HEAD_SIZE,
    ROBOTWIN_FASTWAM_MOSAIC_SIZE,
    ROBOTWIN_FASTWAM_VIEW_MODE,
    ROBOTWIN_FASTWAM_WRIST_SIZE,
    ROBOTWIN_NATIVE_CAMERA_SIZE,
    SUPPORTED_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSIONS,
    robotwin_fastwam_view_mosaic,
    KFLOW_CHANNEL_ORDER,
    ROBOTWIN_CAMERA_ORDER,
    KFlowContract,
    MoMapBatchAdapter,
    pack_kflow,
    robotwin_view_mosaic,
    unpack_kflow,
)
from .model import (
    FORMAL_FULL_RANK_PIXELDIT_REINITIALIZED_KEYS,
    FORMAL_KFLOW_PIXEL_HIDDEN_SIZE,
    FORMAL_LOCAL_GLOBAL_FUSION_REINITIALIZED_KEYS,
    FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE,
    H32S1_DPIX96_PIXEL_HIDDEN_SIZE,
    H32S1_LOCAL96_RAW_FLOW_RESIDUAL_RESET_POLICY,
    FULL_RANK_LOCAL_RESET_POLICY,
    K1_RAW_FLOW_RESIDUAL_RESET_POLICY,
    K1_RAW_FLOW_RESIDUAL_SEMANTIC_REINITIALIZED_KEYS,
    LEGACY_DPIX24_PIXEL_HIDDEN_SIZE,
    LEGACY_LOCAL_ONLY_OUTPUT_HEAD,
    LOCAL24_RAW_FLOW_RESIDUAL_RESET_POLICY,
    LOCAL_GLOBAL_FUSION_NEW_HEAD_KEYS,
    LOCAL_GLOBAL_FUSION_INITIALIZATION,
    LOCAL_GLOBAL_FUSION_RESET_POLICY,
    LOCAL_GLOBAL_FUSION_SHAPE_REINITIALIZED_KEYS,
    LOCAL_GLOBAL_FUSION_TARGET_ONLY_KEYS,
    LOCAL_GLOBAL_FUSION_TRUNK_REINITIALIZED_KEYS,
    MOMAP_LOCAL_GLOBAL_FUSION_OUTPUT_HEAD,
    MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD,
    MoMapLocalGlobalFusionHead,
    MoMapRawFlowResidualPath,
    PixDiT,
    PixelDiT,
    full_rank_local_reinitialized_keys,
    RAW_FLOW_RESIDUAL_NOISE_FLOOR,
)
from .momap_flow import (
    MoMapFlowMatchingBatch,
    MoMapFlowMatchingObjective,
    MoMapVelocityLoss,
)
from .normalization import FixedFeatureNormalizationContract
from .pixeldit_aligned_action import (
    PixelDiTAlignedActionBlock,
    PixelDiTAlignedActionDiT,
    PixelDiTTemporalActionDiT,
    fixed_1d_sincos_position_embedding,
    fuse_world_action_native_kv,
)
from .pixeldit_aligned_action_mot import PixelDiTAlignedActionMoT
from .pixeldit_action_mot import PixelDiTActionMoT, WorldNativeKVAdapter, WorldToActionAdapter


_LAZY_EXPORTS = {
    "BoundFeatureStatisticsArtifact": (
        "feature_statistics_artifact",
        "BoundFeatureStatisticsArtifact",
    ),
    "FORMAL_STATISTICS_QUALIFICATION": (
        "feature_statistics_artifact",
        "FORMAL_STATISTICS_QUALIFICATION",
    ),
    "GeometryStatisticsAccumulator": (
        "geometry_statistics",
        "GeometryStatisticsAccumulator",
    ),
    "FixedFeatureStatisticsAccumulator": (
        "formal_statistics",
        "FixedFeatureStatisticsAccumulator",
    ),
    "StatisticsProviderResult": (
        "formal_statistics",
        "StatisticsProviderResult",
    ),
    "build_formal_v1_statistics_provider": (
        "formal_statistics",
        "build_formal_v1_statistics_provider",
    ),
    "load_statistics_aggregate": (
        "formal_statistics",
        "load_statistics_aggregate",
    ),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "CANONICAL_REFERENCE_FRAME",
    "BoundFeatureStatisticsArtifact",
    "FastWAMPixelDiT",
    "FixedFeatureStatisticsAccumulator",
    "FixedFeatureNormalizationContract",
    "FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL16_RAW_FLOW_RESIDUAL_ABLATION_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION",
    "FORMAL_STATISTICS_QUALIFICATION",
    "GeometryNormalization",
    "GeometryNormalizationContract",
    "GeometryStatisticsAccumulator",
    "FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_K1_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION",
    "FASTWAM_FULLRES_KFLOW_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION",
    "FORMAL_FULL_RANK_PIXELDIT_REINITIALIZED_KEYS",
    "H32S1_DPIX96_PIXEL_HIDDEN_SIZE",
    "H32S1_LOCAL96_RAW_FLOW_RESIDUAL_RESET_POLICY",
    "FORMAL_KFLOW_PIXEL_HIDDEN_SIZE",
    "FORMAL_LOCAL_GLOBAL_FUSION_REINITIALIZED_KEYS",
    "FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE",
    "FULL_RANK_LOCAL_RESET_POLICY",
    "K1_RAW_FLOW_RESIDUAL_RESET_POLICY",
    "K1_RAW_FLOW_RESIDUAL_SEMANTIC_REINITIALIZED_KEYS",
    "KFLOW_ARCHITECTURE_VERSION",
    "KFLOW_CHANNEL_ORDER",
    "KFlowContract",
    "LEGACY_DPIX24_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION",
    "LEGACY_DPIX24_PIXEL_HIDDEN_SIZE",
    "LEGACY_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION",
    "LEGACY_LOCAL_ONLY_OUTPUT_HEAD",
    "LOCAL24_RAW_FLOW_RESIDUAL_RESET_POLICY",
    "LOCAL_GLOBAL_FUSION_NEW_HEAD_KEYS",
    "LOCAL_GLOBAL_FUSION_INITIALIZATION",
    "LOCAL_GLOBAL_FUSION_RESET_POLICY",
    "LOCAL_GLOBAL_FUSION_SHAPE_REINITIALIZED_KEYS",
    "LOCAL_GLOBAL_FUSION_TARGET_ONLY_KEYS",
    "LOCAL_GLOBAL_FUSION_TRUNK_REINITIALIZED_KEYS",
    "MOMAP_LOCAL_GLOBAL_FUSION_OUTPUT_HEAD",
    "MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD",
    "MoMapBatchAdapter",
    "MoMapFlowMatchingBatch",
    "MoMapFlowMatchingObjective",
    "MoMapVelocityLoss",
    "MoMapLocalGlobalFusionHead",
    "MoMapRawFlowResidualPath",
    "PixDiT",
    "PixelDiT",
    "PixelDiTAlignedActionBlock",
    "PixelDiTAlignedActionDiT",
    "PixelDiTTemporalActionDiT",
    "PixelDiTAlignedActionMoT",
    "PixelDiTActionMoT",
    "RAW_FLOW_RESIDUAL_NOISE_FLOOR",
    "PER_SAMPLE_P99_GEOMETRY_NORMALIZATION_MODE",
    "PER_SAMPLE_P99_GEOMETRY_NORMALIZATION_SCHEMA_VERSION",
    "PerSampleP99GeometryNormalizationContract",
    "SOURCE_XYZ_MEAN_STD_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_MODE",
    "SOURCE_XYZ_MEAN_STD_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION",
    "SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract",
    "ROBOTWIN_CAMERA_ORDER",
    "StatisticsProviderResult",
    "WorldNativeKVAdapter",
    "WorldToActionAdapter",
    "WORLD_CONDITIONING_V2",
    "WorldConditioningContract",
    "ROBOTWIN_FASTWAM_HEAD_SIZE",
    "ROBOTWIN_FASTWAM_MOSAIC_SIZE",
    "ROBOTWIN_FASTWAM_VIEW_MODE",
    "ROBOTWIN_FASTWAM_WRIST_SIZE",
    "ROBOTWIN_NATIVE_CAMERA_SIZE",
    "SUPPORTED_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSIONS",
    "build_formal_v1_statistics_provider",
    "create_fastwam_pixeldit",
    "fixed_1d_sincos_position_embedding",
    "full_rank_local_reinitialized_keys",
    "fuse_world_action_native_kv",
    "load_statistics_aggregate",
    "pack_kflow",
    "robotwin_fastwam_view_mosaic",
    "robotwin_view_mosaic",
    "unpack_kflow",
]
