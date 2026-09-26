"""Model-local factory for the MoMaps-conditioned FastWAM PixelDiT.

This keeps MoMap-specific assembly out of ``mwam.runtime``.  Hydra may point
directly at :func:`create_fastwam_pixeldit`; the old runtime entrypoint can
remain a thin lazy compatibility import.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from ..wan22.action_dit import ActionDiT
from ..wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from .conditioning import WorldConditioningContract
from .fastwam_pixeldit import (
    FLOW_SUPERVISION_ALL_MOSAIC,
    FLOW_SUPERVISION_HEAD_TILE_ONLY,
    FastWAMPixelDiT,
)
from .hybrid_fastwam_pixeldit import HybridFastWAMPixelDiT
from .hybrid_cw_pixeldit import HybridCWPixelDiT
from .hybrid_cw_action_mot import (
    HYBRID_CW_KV_SOURCE_AC,
    HYBRID_CW_KV_SOURCE_ACW,
    HYBRID_CW_NATIVE_KV_FUSION_VERSION,
    HybridCWActionMoT,
    initialize_action_from_hybrid_world,
)
from .hybrid_normalization import (
    HYBRID_HEAD_WRIST_NORMALIZATION_SCHEMA_VERSION,
    HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_SCHEMA_VERSION,
    HybridHeadWristNormalizationContract,
)
from .hybrid_normalization_artifact import (
    load_bound_hybrid_k4_normalization_asset,
)
from .geometry_normalization import (
    FASTWAM_HEAD_TILE_FLOW_VALID_VALUES_POOLED_HORIZONS,
    HYBRID_FLOW_POPULATION_ALL_VALID,
    FIXED_GLOBAL_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION,
    GEOMETRY_NORMALIZATION_SCHEMA_VERSION,
    HYBRID_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION,
    PER_SAMPLE_P99_GEOMETRY_NORMALIZATION_SCHEMA_VERSION,
    SOURCE_XYZ_MEAN_STD_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION,
    GeometryNormalization,
    GeometryNormalizationContract,
    FixedGlobalSourceP99FlowXYZMeanStdGeometryNormalizationContract,
    HybridSourceP99FlowXYZMeanStdGeometryNormalizationContract,
    PerSampleP99GeometryNormalizationContract,
    SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract,
)
from .all_mosaic_flow_statistics import (
    ALL_MOSAIC_FLOW_XYZ_CALIBRATION_MODEL_VISIBLE_POPULATION,
    BoundAllMosaicFlowXYZMeanStdStatisticsArtifact,
)
from .head_flow_statistics import (
    FORMAL_TRAIN_SPLIT_STRATIFIED_CALIBRATION_QUALIFICATION,
    BoundHeadTileFlowXYZMeanStdStatisticsArtifact,
)
from .kflow import (
    FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL16_RAW_FLOW_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_K1_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_LOCAL24_FUSION_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
    FASTWAM_FULLRES_KFLOW_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
    LEGACY_DPIX24_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
    LEGACY_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
    KFlowContract,
    MoMapBatchAdapter,
)
from .method_contract import (
    FORMAL_METHOD_CONTRACT_V1,
    FORMAL_METHOD_CONTRACT_V2,
    FORMAL_METHOD_CONTRACT_V3,
    FORMAL_METHOD_CONTRACT_V4,
    FORMAL_METHOD_CONTRACT_V6,
    FORMAL_METHOD_CONTRACT_V7,
    FORMAL_METHOD_CONTRACT_V8,
    FORMAL_METHOD_CONTRACT_V9,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2,
    resolve_formal_method_contract,
)
from .model import (
    FORMAL_KFLOW_PIXEL_HIDDEN_SIZE,
    FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE,
    H32S1_DPIX96_PIXEL_HIDDEN_SIZE,
    LEGACY_DPIX24_PIXEL_HIDDEN_SIZE,
    LEGACY_LOCAL_ONLY_OUTPUT_HEAD,
    MOMAP_LOCAL_GLOBAL_FUSION_OUTPUT_HEAD,
    MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD,
    PixDiT,
)
from .normalization import FixedFeatureNormalizationContract
from .pixeldit_action_mot import PixelDiTActionMoT
from .pixeldit_aligned_action import (
    PixelDiTAlignedActionDiT,
    PixelDiTTemporalActionDiT,
)
from .pixeldit_aligned_action_mot import PixelDiTAlignedActionMoT


def _plain_dict(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        raise ValueError(f"`{name}` is required.")
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(
            value,
            resolve=True,
            enum_to_str=True,
        )
        if not isinstance(value, dict):
            raise TypeError(f"`{name}` must resolve to a mapping.")
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError(f"`{name}` must be mapping-like, got {type(value).__name__}.")


def _construct_dataclass(value: Any, cls: type, *, name: str):
    if isinstance(value, cls):
        return value
    raw = _plain_dict(value, name=name)
    fields = set(cls.__dataclass_fields__)
    derived = {
        "k",
        "flow_channels",
        "condition_channels",
        "feature_dim",
        "offset_center",
        "target_scale",
        "source_scale",
        "flow_mean_channels_meters",
        "flow_std_channels_meters",
    }
    unexpected = sorted(set(raw) - fields - derived)
    if unexpected:
        raise ValueError(f"`{name}` has unexpected keys {unexpected}.")
    for key in derived:
        raw.pop(key, None)
    try:
        return cls(**raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `{name}`: {exc}") from exc


def _resolve_geometry(
    contract: Any,
    artifact_path: str | os.PathLike[str] | None,
) -> GeometryNormalization:
    if (contract is None) == (artifact_path is None):
        raise ValueError(
            "Provide exactly one of `normalization_contract` or "
            "`normalization_artifact_path`."
        )
    if artifact_path is not None:
        path = Path(artifact_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(
                f"geometry normalization is not a regular file: {path}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read geometry normalization JSON {path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"geometry normalization JSON root must be a mapping: {path}"
            )
        contract = payload.get("normalization_contract", payload)
    if isinstance(
        contract,
        (
            GeometryNormalizationContract,
            PerSampleP99GeometryNormalizationContract,
            HybridSourceP99FlowXYZMeanStdGeometryNormalizationContract,
            FixedGlobalSourceP99FlowXYZMeanStdGeometryNormalizationContract,
            SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract,
        ),
    ):
        return contract
    raw = _plain_dict(contract, name="normalization_contract")
    schema_version = raw.get(
        "schema_version",
        GEOMETRY_NORMALIZATION_SCHEMA_VERSION,
    )
    if schema_version == GEOMETRY_NORMALIZATION_SCHEMA_VERSION:
        # Runtime accepts legacy wrappers but selects only model-math fields.
        runtime_fields = {
            "schema_version",
            "mode",
            "statistics_id",
            "future_offsets",
            "xyz_center",
            "xyz_scale",
            "offset_scale",
            "reference_frame",
            "source_unit",
            "channel_order",
            "reduction",
            "k",
            "offset_center",
        }
        raw = {key: value for key, value in raw.items() if key in runtime_fields}
        contract_type = GeometryNormalizationContract
    elif schema_version == PER_SAMPLE_P99_GEOMETRY_NORMALIZATION_SCHEMA_VERSION:
        contract_type = PerSampleP99GeometryNormalizationContract
    elif (
        schema_version
        == HYBRID_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION
    ):
        contract_type = HybridSourceP99FlowXYZMeanStdGeometryNormalizationContract
    elif (
        schema_version
        == FIXED_GLOBAL_SOURCE_P99_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION
    ):
        contract_type = FixedGlobalSourceP99FlowXYZMeanStdGeometryNormalizationContract
    elif (
        schema_version
        == SOURCE_XYZ_MEAN_STD_FLOW_XYZ_MEAN_STD_GEOMETRY_NORMALIZATION_SCHEMA_VERSION
    ):
        contract_type = SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract
    else:
        raise ValueError(
            "Unsupported geometry normalization schema_version "
            f"{schema_version!r}."
        )
    return _construct_dataclass(
        raw,
        contract_type,
        name="normalization_contract",
    )


def _feature_contract_from_mapping(
    value: Any,
    *,
    kind: str,
) -> FixedFeatureNormalizationContract:
    raw = _plain_dict(value, name=f"{kind}_normalization_contract")
    feature_dim = raw.pop("feature_dim", 14)
    if feature_dim != 14:
        raise ValueError(
            f"{kind} normalization feature_dim must be 14, got {feature_dim!r}."
        )
    # Historical wrappers may contain a small contract digest. Runtime relies
    # on the actual mean/std values and validates them directly.
    raw.pop("digest", None)
    resolved = _construct_dataclass(
        raw,
        FixedFeatureNormalizationContract,
        name=f"{kind}_normalization_contract",
    )
    if resolved.feature_kind != kind:
        raise ValueError(
            f"{kind} normalization feature_kind mismatch: "
            f"{resolved.feature_kind!r}."
        )
    return resolved


def _resolve_feature_contract(
    contract: Any,
    artifact_path: str | os.PathLike[str] | None,
    *,
    kind: str,
) -> tuple[FixedFeatureNormalizationContract, dict[str, Any]]:
    if (contract is None) == (artifact_path is None):
        raise ValueError(
            f"Provide exactly one inline or artifact-backed {kind} normalization."
        )
    if artifact_path is not None:
        path = Path(artifact_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(
                f"{kind} normalization is not a regular file: {path}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read {kind} normalization JSON {path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"{kind} normalization JSON root must be a mapping: {path}"
            )
        # Existing assets may use the historical statistics wrapper. Training
        # consumes only the numerical normalization contract.
        resolved = _feature_contract_from_mapping(
            payload.get("normalization_contract", payload),
            kind=kind,
        )
        source = {
            "kind": "json_artifact",
            "path": str(path),
            "schema_version": resolved.schema_version,
            "statistics_id": resolved.statistics_id,
            "feature_kind": resolved.feature_kind,
            "feature_dim": resolved.feature_dim,
        }
    else:
        resolved = _feature_contract_from_mapping(contract, kind=kind)
        source = {
            "kind": "inline",
            "statistics_id": resolved.statistics_id,
            "feature_kind": resolved.feature_kind,
            "feature_dim": resolved.feature_dim,
        }
    return resolved, source


def _scheduler(value: Any, *, name: str) -> WanContinuousFlowMatchScheduler:
    if isinstance(value, WanContinuousFlowMatchScheduler):
        return value
    raw = _plain_dict(value, name=name)
    expected = {"num_train_timesteps", "shift", "eps"}
    if set(raw) != expected:
        raise ValueError(
            f"`{name}` requires exactly {sorted(expected)}, got {sorted(raw)}."
        )
    scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=int(raw["num_train_timesteps"]),
        shift=float(raw["shift"]),
        eps=float(raw["eps"]),
    )
    if (
        scheduler.num_train_timesteps <= 0
        or not math.isfinite(scheduler.shift)
        or scheduler.shift <= 0.0
        or not math.isfinite(scheduler.eps)
        or scheduler.eps <= 0.0
    ):
        raise ValueError(f"`{name}` values must be finite and positive.")
    return scheduler


def _load_intent(
    *,
    skip: bool | None,
    path: str | os.PathLike[str] | None,
    name: str,
) -> str | None:
    if not isinstance(skip, bool):
        raise ValueError(f"`skip_{name}_load_from_pretrain` must be explicit bool.")
    normalized = None if path is None else os.fspath(path).strip()
    if skip:
        if normalized:
            raise ValueError(f"Do not provide `{name}_pretrained_path` when skipped.")
        return None
    if not normalized:
        raise ValueError(f"`{name}_pretrained_path` is required unless skipped.")
    return normalized


def create_fastwam_pixeldit(
    pixeldit_config,
    action_dit_config,
    kflow_contract,
    method_contract,
    normalization_contract=None,
    normalization_artifact_path: str | None = None,
    action_horizon: int = 32,
    action_normalization_contract=None,
    action_normalization_artifact_path: str | None = None,
    proprio_normalization_contract=None,
    proprio_normalization_artifact_path: str | None = None,
    pixeldit_pretrained_path: str | None = None,
    pixeldit_weight_source: str = "auto",
    action_dit_pretrained_path: str | None = None,
    skip_pixeldit_load_from_pretrain: bool | None = None,
    skip_action_load_from_pretrain: bool | None = None,
    proprio_dim: int | None = None,
    train_world_scheduler=None,
    infer_world_scheduler=None,
    train_action_scheduler=None,
    infer_action_scheduler=None,
    timestep_coupling: str = "independent",
    flow_supervision_scope: str = "all_mosaic",
    flow_statistics_provenance_contract=None,
    loss=None,
    mot=None,
    aligned_action_dit_config=None,
    aligned_mot=None,
    action_initialization_seed: int | None = None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> FastWAMPixelDiT:
    """Build one exact versioned common-reference MoMap architecture."""

    resolved_method = resolve_formal_method_contract(
        _plain_dict(method_contract, name="method_contract")
    )
    resolved_kflow = _construct_dataclass(
        kflow_contract,
        KFlowContract,
        name="kflow_contract",
    )
    v7_concat_architectures = {
        FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H8S4_LOCAL16_RAW_FLOW_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
        FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
    }
    if resolved_kflow.architecture_version in v7_concat_architectures:
        expected_method = FORMAL_METHOD_CONTRACT_V2
    elif (
        resolved_kflow.architecture_version
        == FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION
    ):
        expected_method = None
        expected_aligned_methods = (
            FORMAL_METHOD_CONTRACT_V8,
            FORMAL_METHOD_CONTRACT_V9,
        )
    elif (
        resolved_kflow.architecture_version
        == FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION
    ):
        expected_method = FORMAL_METHOD_CONTRACT_V3
    elif (
        resolved_kflow.architecture_version
        in {
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
        }
    ):
        expected_method = FORMAL_METHOD_CONTRACT_V6
    elif (
        resolved_kflow.architecture_version
        == FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
    ):
        expected_method = FORMAL_METHOD_CONTRACT_V7
    elif (
        resolved_kflow.architecture_version
        == FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_ARCHITECTURE_VERSION
    ):
        expected_method = FORMAL_METHOD_CONTRACT_V4
    else:
        expected_method = FORMAL_METHOD_CONTRACT_V1
    if expected_method is None and resolved_method not in expected_aligned_methods:
        raise ValueError(
            "KFlow aligned-linear14 architecture requires one of "
            + repr(
                [
                    value["schema_version"]
                    for value in expected_aligned_methods
                ]
            )
            + "."
        )
    if expected_method is not None and resolved_method != expected_method:
        raise ValueError(
            "KFlow architecture/method_contract mismatch: architecture "
            f"{resolved_kflow.architecture_version!r} requires "
            f"{expected_method['schema_version']!r}."
        )
    if not resolved_kflow.uses_common_reference:
        raise ValueError(
            "The production factory requires the exact V2 common-reference contract."
        )
    geometry = _resolve_geometry(
        normalization_contract,
        normalization_artifact_path,
    )
    if (
        geometry.future_offsets != resolved_kflow.future_offsets
        or geometry.reference_frame != resolved_kflow.coordinate_frame
        or geometry.source_unit != resolved_kflow.unit
        or geometry.channel_order != resolved_kflow.channel_order
    ):
        raise ValueError(
            "Geometry normalization is not bound to the resolved KFlow contract."
        )
    if isinstance(
        geometry,
        FixedGlobalSourceP99FlowXYZMeanStdGeometryNormalizationContract,
    ):
        geometry_source: dict[str, Any] = {
            "kind": (
                "json_artifact_with_fixed_global_source_scale"
                if normalization_artifact_path is not None
                else "inline_with_fixed_global_source_scale"
            ),
            "schema_version": geometry.schema_version,
            "mode": geometry.mode,
            "statistics_id": geometry.statistics_id,
            "source_scale_kind": geometry.source_scale_kind,
            "requires_statistics_artifact": True,
        }
    elif isinstance(
        geometry,
        SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract,
    ):
        geometry_source = {
            "kind": "bound_source_xyz_and_flow_xyz_mean_std_statistics",
            "schema_version": geometry.schema_version,
            "mode": geometry.mode,
            "statistics_id": geometry.statistics_id,
            "source_statistics_id": geometry.source_statistics_id,
            "flow_statistics_id": geometry.flow_statistics_id,
            "requires_statistics_artifact": True,
        }
    elif isinstance(
        geometry,
        HybridSourceP99FlowXYZMeanStdGeometryNormalizationContract,
    ):
        geometry_source: dict[str, Any] = {
            "kind": (
                "json_artifact_with_dynamic_source_scale"
                if normalization_artifact_path is not None
                else "inline_with_dynamic_source_scale"
            ),
            "schema_version": geometry.schema_version,
            "mode": geometry.mode,
            "statistics_id": geometry.statistics_id,
            "requires_statistics_artifact": True,
        }
    elif isinstance(geometry, PerSampleP99GeometryNormalizationContract):
        geometry_source = {
            "kind": "dynamic_source_observation_contract",
            "schema_version": geometry.schema_version,
            "mode": geometry.mode,
            "requires_statistics_artifact": False,
        }
    else:
        geometry_source = {
            "kind": (
                "json_artifact"
                if normalization_artifact_path is not None
                else "inline"
            ),
            "statistics_id": geometry.statistics_id,
        }
    if normalization_artifact_path is not None:
        geometry_source["path"] = str(
            Path(normalization_artifact_path).expanduser().resolve(strict=True)
        )
    action_norm, action_norm_source = _resolve_feature_contract(
        action_normalization_contract,
        action_normalization_artifact_path,
        kind="action",
    )
    if proprio_dim is None:
        if (
            proprio_normalization_contract is not None
            or proprio_normalization_artifact_path is not None
        ):
            raise ValueError("Proprio normalization requires `proprio_dim`.")
        proprio_norm = None
        proprio_norm_source = None
    else:
        if int(proprio_dim) != 14:
            raise ValueError("Formal proprio_dim is fixed at 14.")
        proprio_norm, proprio_norm_source = _resolve_feature_contract(
            proprio_normalization_contract,
            proprio_normalization_artifact_path,
            kind="proprio",
        )
    if int(action_horizon) != 32:
        raise ValueError("Formal action_horizon is fixed at 32.")

    if timestep_coupling != resolved_method["train_timestep_coupling"]:
        raise ValueError(
            "`timestep_coupling` does not match the frozen method_contract."
        )

    schedulers = {
        "train_world": _scheduler(
            train_world_scheduler,
            name="train_world_scheduler",
        ),
        "infer_world": _scheduler(
            infer_world_scheduler,
            name="infer_world_scheduler",
        ),
        "train_action": _scheduler(
            train_action_scheduler,
            name="train_action_scheduler",
        ),
        "infer_action": _scheduler(
            infer_action_scheduler,
            name="infer_action_scheduler",
        ),
    }
    if len({id(value) for value in schedulers.values()}) != 4:
        raise ValueError("World/action train/infer schedulers must be independent.")

    condition_contract = WorldConditioningContract()
    world_config = _plain_dict(pixeldit_config, name="pixeldit_config")
    if (
        resolved_kflow.architecture_version
        in {
            FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
        }
    ):
        versioned_world = {
            "pixel_hidden_size": H32S1_DPIX96_PIXEL_HIDDEN_SIZE,
            "require_full_rank_output": True,
            "momap_output_head_type": (
                MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
            ),
            "momap_output_hidden_size": FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE,
        }
    elif (
        resolved_kflow.architecture_version in {
            FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
        }
    ):
        versioned_world = {
            "pixel_hidden_size": LEGACY_DPIX24_PIXEL_HIDDEN_SIZE,
            "require_full_rank_output": True,
            "momap_output_head_type": (
                MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
            ),
            "momap_output_hidden_size": FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE,
        }
    elif (
        resolved_kflow.architecture_version in {
            FASTWAM_FULLRES_KFLOW_LOCAL24_FUSION_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION,
        }
    ):
        versioned_world = {
            "pixel_hidden_size": LEGACY_DPIX24_PIXEL_HIDDEN_SIZE,
            "require_full_rank_output": True,
            "momap_output_head_type": MOMAP_LOCAL_GLOBAL_FUSION_OUTPUT_HEAD,
            "momap_output_hidden_size": FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE,
        }
    elif (
        resolved_kflow.architecture_version
        in {
            FASTWAM_FULLRES_KFLOW_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_K1_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION,
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL16_RAW_FLOW_RESIDUAL_ABLATION_ARCHITECTURE_VERSION,
        }
    ):
        versioned_world = {
            "pixel_hidden_size": FORMAL_KFLOW_PIXEL_HIDDEN_SIZE,
            "require_full_rank_output": True,
            "momap_output_head_type": (
                MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
            ),
            "momap_output_hidden_size": FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE,
        }
    elif (
        resolved_kflow.architecture_version
        == FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION
    ):
        versioned_world = {
            "pixel_hidden_size": FORMAL_KFLOW_PIXEL_HIDDEN_SIZE,
            "require_full_rank_output": True,
            "momap_output_head_type": MOMAP_LOCAL_GLOBAL_FUSION_OUTPUT_HEAD,
            "momap_output_hidden_size": FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE,
        }
    elif (
        resolved_kflow.architecture_version
        == LEGACY_DPIX24_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION
    ):
        versioned_world = {
            "pixel_hidden_size": LEGACY_DPIX24_PIXEL_HIDDEN_SIZE,
            "require_full_rank_output": True,
            "momap_output_head_type": LEGACY_LOCAL_ONLY_OUTPUT_HEAD,
            "momap_output_hidden_size": LEGACY_DPIX24_PIXEL_HIDDEN_SIZE,
        }
    elif (
        resolved_kflow.architecture_version
        == LEGACY_FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION
    ):
        versioned_world = {
            "pixel_hidden_size": 16,
            "require_full_rank_output": False,
            "momap_output_head_type": LEGACY_LOCAL_ONLY_OUTPUT_HEAD,
            "momap_output_hidden_size": 16,
        }
    else:
        versioned_world = {}
    fixed_world = {
        "conditioning_contract": condition_contract,
        "flow_channels": resolved_kflow.flow_channels,
        "out_channels": resolved_kflow.flow_channels,
        "in_channels": (
            condition_contract.condition_channels
            + resolved_kflow.flow_channels
        ),
        "patch_depth": 26,
        "pixel_depth": 4,
        **versioned_world,
    }
    for key, expected in fixed_world.items():
        if key in world_config and world_config[key] != expected:
            raise ValueError(
                f"PixelDiT fixed contract drift for {key}: "
                f"{world_config[key]!r} vs {expected!r}."
            )
        world_config[key] = expected
    legacy_action_config = _plain_dict(
        action_dit_config,
        name="action_dit_config",
    )
    aligned_action_method = resolved_method in (
        FORMAL_METHOD_CONTRACT_V8,
        FORMAL_METHOD_CONTRACT_V9,
    )
    if aligned_action_method:
        action_config = _plain_dict(
            aligned_action_dit_config,
            name="aligned_action_dit_config",
        )
    else:
        if aligned_action_dit_config is not None or aligned_mot is not None:
            raise ValueError(
                "Aligned Action config fields are supported only by V8/V9."
            )
        action_config = legacy_action_config
    expected_action_layers = (
        30
        if resolved_method in (
            FORMAL_METHOD_CONTRACT_V2,
            FORMAL_METHOD_CONTRACT_V3,
            FORMAL_METHOD_CONTRACT_V6,
            FORMAL_METHOD_CONTRACT_V8,
            FORMAL_METHOD_CONTRACT_V9,
        )
        else 26
    )
    if int(action_config.get("num_layers", -1)) != expected_action_layers:
        raise ValueError(
            "Versioned ActionDiT depth mismatch: architecture "
            f"{resolved_kflow.architecture_version!r} requires "
            f"{expected_action_layers} layers."
        )
    if int(action_config.get("action_dim", -1)) != 14:
        raise ValueError("Formal ActionDiT action_dim must be 14.")
    text_dim = int(action_config.get("text_dim", -1))
    if text_dim <= 0:
        raise ValueError("ActionDiT text_dim must be positive.")

    pixel_path = _load_intent(
        skip=skip_pixeldit_load_from_pretrain,
        path=pixeldit_pretrained_path,
        name="pixeldit",
    )
    action_path = _load_intent(
        skip=skip_action_load_from_pretrain,
        path=action_dit_pretrained_path,
        name="action",
    )
    world = PixDiT(**world_config).to(device=device, dtype=model_dtype)
    pixel_report = None
    if pixel_path is not None:
        pixel_report = world.load_pretrained(
            pixel_path,
            weight_source=pixeldit_weight_source,
            map_location="cpu",
        )
    mot_config = _plain_dict(
        aligned_mot if aligned_action_method else mot,
        name=(
            "aligned_mot"
            if aligned_action_method
            else "mot"
        ),
    )
    if aligned_action_method:
        if action_path is not None:
            raise ValueError(
                "Aligned Action initialization must come from the fresh-stage "
                "random/World-transfer policy, not an Action checkpoint path."
            )
        if (
            isinstance(action_initialization_seed, bool)
            or not isinstance(action_initialization_seed, int)
            or action_initialization_seed < 0
            or action_initialization_seed >= 2**32
        ):
            raise ValueError(
                "V8/V9 requires action_initialization_seed in uint32 range."
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(action_initialization_seed)
            action = PixelDiTAlignedActionDiT(**action_config)
        action = action.to(device=device, dtype=model_dtype)
        dit = PixelDiTAlignedActionMoT(
            world,
            action,
            **mot_config,
        ).to(device=device, dtype=model_dtype)
    else:
        if action_initialization_seed is not None:
            raise ValueError(
                "action_initialization_seed is supported only by V8/V9."
            )
        action = ActionDiT.from_pretrained(
            action_dit_config=action_config,
            action_dit_pretrained_path=action_path,
            skip_dit_load_from_pretrain=action_path is None,
            device=device,
            torch_dtype=model_dtype,
        )
        dit = PixelDiTActionMoT(
            world,
            action,
            **mot_config,
        ).to(device=device, dtype=model_dtype)
    loss_config = _plain_dict(loss, name="loss")
    if set(loss_config) != {"lambda_momap", "lambda_action"}:
        raise ValueError("`loss` requires lambda_momap and lambda_action.")
    model = FastWAMPixelDiT(
        dit,
        kflow_contract=resolved_kflow,
        normalization_contract=geometry,
        action_horizon=32,
        action_normalization_contract=action_norm,
        text_dim=text_dim,
        batch_adapter=MoMapBatchAdapter(
            resolved_kflow,
            action_label_kind=action_norm.semantic_kind,
            action_is_high_level_policy_action=False,
            action_normalization=action_norm.source_normalization,
            proprio_mode=(
                "none" if proprio_norm is None else proprio_norm.semantic_kind
            ),
            proprio_normalization=(
                "none"
                if proprio_norm is None
                else proprio_norm.source_normalization
            ),
        ),
        proprio_dim=proprio_dim,
        proprio_normalization_contract=proprio_norm,
        train_world_scheduler=schedulers["train_world"],
        infer_world_scheduler=schedulers["infer_world"],
        train_action_scheduler=schedulers["train_action"],
        infer_action_scheduler=schedulers["infer_action"],
        timestep_coupling=timestep_coupling,
        loss_lambda_momap=float(loss_config["lambda_momap"]),
        loss_lambda_action=float(loss_config["lambda_action"]),
        flow_supervision_scope=flow_supervision_scope,
        flow_statistics_provenance_contract=(
            flow_statistics_provenance_contract
        ),
        method_contract=resolved_method,
    ).to(device=device, dtype=model_dtype)
    model.pixeldit_load_report = pixel_report
    model.action_dit_load_report = getattr(action, "pretrained_load_report", None)
    model.pretrained_load_reports = {
        "pixeldit": model.pixeldit_load_report,
        "action_dit": model.action_dit_load_report,
    }
    model.method_contract = resolved_method
    model.normalization_contract_source = geometry_source
    model.normalization_contract_sources = {
        "geometry": geometry_source,
        "action": action_norm_source,
        "proprio": proprio_norm_source,
    }
    return model


def create_head_only_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build the formal V5 model with head-tile-only Flow supervision.

    This is deliberately a production factory, not the deterministic
    single-window/fixed-objective debug wrapper.  The base model therefore
    retains per-forward random Gaussian noise and random FM timesteps.
    """

    if kwargs.get("normalization_contract") is not None:
        raise ValueError(
            "Head-only formal training requires the frozen Flow XYZ mean/std "
            "JSON asset; do not supply an inline normalization contract."
        )
    artifact_path = kwargs.get("normalization_artifact_path")
    if artifact_path is None or not os.fspath(artifact_path).strip():
        raise ValueError(
            "Head-only formal training requires `normalization_artifact_path`."
        )
    for name, value in (
        ("expected_split_manifest_sha256", expected_split_manifest_sha256),
        ("expected_dataset_receipt_sha256", expected_dataset_receipt_sha256),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Head-only formal training requires non-empty `{name}`."
            )
    if "flow_supervision_scope" in kwargs:
        raise ValueError(
            "The head-only factory freezes `flow_supervision_scope`; do not "
            "override it."
        )
    if "flow_statistics_provenance_contract" in kwargs:
        raise ValueError(
            "The head-only factory derives statistics provenance from the "
            "bound formal wrapper; do not override it."
        )
    artifact = BoundHeadTileFlowXYZMeanStdStatisticsArtifact.load_json(
        artifact_path,
        require_formal=True,
        allow_formal_calibration=True,
        expected_split_manifest_sha256=expected_split_manifest_sha256.strip(),
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256.strip(),
    )
    if (
        artifact.qualification
        == FORMAL_TRAIN_SPLIT_STRATIFIED_CALIBRATION_QUALIFICATION
    ):
        calibration = artifact.accumulator_receipt.get("calibration")
        if not isinstance(calibration, Mapping):
            raise ValueError(
                "Head-only production calibration artifact lacks sampling provenance."
            )
        required_calibration = {
            "selection_algorithm": (
                "global_runtime_index_equal_width_cell_centers_v1"
            ),
            "population_scope": "qualified_formal_train_runtime_index",
            "selected_sample_count": 4096,
            "shard_count": 16,
            "selected_population_exact_cover": True,
            "full_population_exact_cover": False,
            "model_visible_population": (
                "head_camera source-point resize 240x320->256x320; "
                "valid K1-K8 XYZ pooled"
            ),
            "qualification": (
                FORMAL_TRAIN_SPLIT_STRATIFIED_CALIBRATION_QUALIFICATION
            ),
        }
        for name, expected in required_calibration.items():
            if calibration.get(name) != expected:
                raise ValueError(
                    "Head-only production calibration provenance "
                    f"{name}={calibration.get(name)!r}, expected {expected!r}."
                )
        for name in (
            "selected_global_indices_sha256",
            "selected_sample_refs_sha256",
            "sampling_plan_sha256",
            "runtime_index_sha256",
        ):
            value = calibration.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"Head-only production calibration {name} is invalid."
                )
    original_artifact_path = os.fspath(artifact_path)
    kwargs["normalization_artifact_path"] = None
    kwargs["normalization_contract"] = artifact.normalization_contract
    model = create_fastwam_pixeldit(
        *args,
        **kwargs,
        flow_supervision_scope=FLOW_SUPERVISION_HEAD_TILE_ONLY,
        flow_statistics_provenance_contract=artifact.provenance_contract(),
    )
    if not isinstance(
        model.normalization_contract,
        HybridSourceP99FlowXYZMeanStdGeometryNormalizationContract,
    ):
        raise TypeError(
            "Head-only formal training requires the hybrid source-P99 / Flow "
            "XYZ mean-std normalization contract."
        )
    if (
        model.normalization_contract.flow_population
        != FASTWAM_HEAD_TILE_FLOW_VALID_VALUES_POOLED_HORIZONS
    ):
        raise ValueError(
            "Head-only formal training requires Flow XYZ statistics computed "
            "from the head-tile valid population, got "
            f"{model.normalization_contract.flow_population!r}."
        )
    geometry_source = {
        "kind": "bound_formal_head_tile_flow_xyz_mean_std_statistics_artifact",
        "path": str(Path(original_artifact_path).expanduser().resolve(strict=True)),
        "schema_version": artifact.schema_version,
        "qualification": artifact.qualification,
        "split_name": artifact.split_name,
        "split_manifest_sha256": artifact.split_manifest_sha256,
        "dataset_receipt_sha256": artifact.dataset_receipt_sha256,
        "artifact_digest": artifact.artifact_digest,
        "statistics_id": artifact.normalization_contract.statistics_id,
    }
    model.normalization_contract_source = geometry_source
    model.normalization_contract_sources["geometry"] = geometry_source
    return model


def _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    expected_architecture_version: str,
    method_label: str,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    allow_formal_flow_calibration: bool = True,
    strict_statistics_provenance: bool = True,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build one architecture with all-mosaic Flow mean/std supervision."""

    if kwargs.get("normalization_contract") is not None:
        raise ValueError(
            f"{method_label} all-mosaic training requires the frozen Flow XYZ "
            "mean/std JSON asset; do not supply an inline contract."
        )
    artifact_path = kwargs.get("normalization_artifact_path")
    if artifact_path is None or not os.fspath(artifact_path).strip():
        raise ValueError(
            f"{method_label} all-mosaic training requires "
            "normalization_artifact_path."
        )
    if not isinstance(strict_statistics_provenance, bool):
        raise TypeError("strict_statistics_provenance must be bool.")
    if strict_statistics_provenance:
        for name, value in (
            ("expected_split_manifest_sha256", expected_split_manifest_sha256),
            ("expected_dataset_receipt_sha256", expected_dataset_receipt_sha256),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{method_label} all-mosaic training requires non-empty {name}."
                )
    if "flow_supervision_scope" in kwargs:
        raise ValueError(
            f"The {method_label} all-mosaic factory freezes "
            "flow_supervision_scope; "
            "do not override it."
        )
    if "flow_statistics_provenance_contract" in kwargs:
        raise ValueError(
            f"The {method_label} all-mosaic factory derives statistics provenance from "
            "the bound formal wrapper; do not override it."
        )

    if not isinstance(allow_formal_flow_calibration, bool):
        raise TypeError("allow_formal_flow_calibration must be bool.")
    expected_split = (
        expected_split_manifest_sha256.strip()
        if strict_statistics_provenance
        and isinstance(expected_split_manifest_sha256, str)
        else None
    )
    expected_receipt = (
        expected_dataset_receipt_sha256.strip()
        if strict_statistics_provenance
        and isinstance(expected_dataset_receipt_sha256, str)
        else None
    )
    artifact = BoundAllMosaicFlowXYZMeanStdStatisticsArtifact.load_json(
        artifact_path,
        require_formal=strict_statistics_provenance,
        allow_formal_calibration=allow_formal_flow_calibration,
        expected_split_manifest_sha256=expected_split,
        expected_dataset_receipt_sha256=expected_receipt,
    )
    if (
        strict_statistics_provenance
        and artifact.qualification
        == FORMAL_TRAIN_SPLIT_STRATIFIED_CALIBRATION_QUALIFICATION
    ):
        calibration = artifact.accumulator_receipt.get("calibration")
        if not isinstance(calibration, Mapping):
            raise ValueError(
                f"{method_label} all-mosaic calibration artifact lacks "
                "sampling provenance."
            )
        required_calibration = {
            "selection_algorithm": (
                "global_runtime_index_equal_width_cell_centers_v1"
            ),
            "population_scope": "qualified_formal_train_runtime_index",
            "selected_sample_count": 4096,
            "shard_count": 16,
            "selected_population_exact_cover": True,
            "full_population_exact_cover": False,
            "model_visible_population": (
                ALL_MOSAIC_FLOW_XYZ_CALIBRATION_MODEL_VISIBLE_POPULATION
            ),
            "qualification": (
                FORMAL_TRAIN_SPLIT_STRATIFIED_CALIBRATION_QUALIFICATION
            ),
        }
        for name, expected in required_calibration.items():
            if calibration.get(name) != expected:
                raise ValueError(
                    f"{method_label} all-mosaic calibration provenance "
                    f"{name}={calibration.get(name)!r}, expected {expected!r}."
                )
        for name in (
            "selected_global_indices_sha256",
            "selected_sample_refs_sha256",
            "sampling_plan_sha256",
            "runtime_index_sha256",
        ):
            value = calibration.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"{method_label} all-mosaic calibration {name} is invalid."
                )

    original_artifact_path = os.fspath(artifact_path)
    combined_contract = None
    source_contract = None
    original_source_artifact_path = None
    if source_xyz_normalization_artifact_path is not None:
        original_source_artifact_path = os.fspath(
            source_xyz_normalization_artifact_path
        )
        if not original_source_artifact_path.strip():
            raise ValueError(
                f"{method_label} source XYZ mean/std artifact path is empty."
            )
        source_path = Path(original_source_artifact_path).expanduser().resolve(
            strict=True
        )
        source_contract = GeometryNormalizationContract.load_json(source_path)
        flow_contract = artifact.normalization_contract
        if strict_statistics_provenance:
            source_identity = {
                "split_name": source_contract.split_name,
                "split_manifest_sha256": source_contract.split_manifest_sha256,
                "dataset_receipt_sha256": source_contract.dataset_receipt_sha256,
                "future_offsets": source_contract.future_offsets,
                "reference_frame": source_contract.reference_frame,
                "source_unit": source_contract.source_unit,
                "channel_order": source_contract.channel_order,
            }
            expected_identity = {
                "split_name": "train",
                "split_manifest_sha256": expected_split,
                "dataset_receipt_sha256": expected_receipt,
                "future_offsets": flow_contract.future_offsets,
                "reference_frame": flow_contract.reference_frame,
                "source_unit": flow_contract.source_unit,
                "channel_order": flow_contract.channel_order,
            }
            combined_split_manifest_sha256 = expected_split
            combined_dataset_receipt_sha256 = expected_receipt
        else:
            source_identity = {
                "split_name": source_contract.split_name,
                "future_offsets": source_contract.future_offsets,
                "reference_frame": source_contract.reference_frame,
                "source_unit": source_contract.source_unit,
                "channel_order": source_contract.channel_order,
            }
            expected_identity = {
                "split_name": artifact.split_name,
                "future_offsets": flow_contract.future_offsets,
                "reference_frame": flow_contract.reference_frame,
                "source_unit": flow_contract.source_unit,
                "channel_order": flow_contract.channel_order,
            }
            combined_split_manifest_sha256 = artifact.split_manifest_sha256
            combined_dataset_receipt_sha256 = artifact.dataset_receipt_sha256
        if source_identity != expected_identity:
            raise ValueError(
                f"{method_label} Source XYZ and Flow XYZ statistics are not "
                f"semantically compatible: source={source_identity!r}, "
                f"expected={expected_identity!r}."
            )
        assert combined_split_manifest_sha256 is not None
        assert combined_dataset_receipt_sha256 is not None
        combined_contract = (
            SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract(
                statistics_id=(
                    f"{source_contract.statistics_id}__{flow_contract.statistics_id}"
                ),
                source_statistics_id=source_contract.statistics_id,
                flow_statistics_id=flow_contract.statistics_id,
                split_manifest_sha256=combined_split_manifest_sha256,
                dataset_receipt_sha256=combined_dataset_receipt_sha256,
                future_offsets=flow_contract.future_offsets,
                source_mean_xyz_meters=source_contract.xyz_center,
                source_std_xyz_meters=source_contract.xyz_scale,
                flow_mean_xyz_meters=flow_contract.flow_mean_xyz_meters,
                flow_std_xyz_meters=flow_contract.flow_std_xyz_meters,
                flow_population=flow_contract.flow_population,
            )
        )
    kwargs["normalization_artifact_path"] = None
    kwargs["normalization_contract"] = (
        combined_contract
        if combined_contract is not None
        else artifact.normalization_contract
    )
    model = create_fastwam_pixeldit(
        *args,
        **kwargs,
        flow_supervision_scope=FLOW_SUPERVISION_ALL_MOSAIC,
        flow_statistics_provenance_contract=artifact.provenance_contract(),
    )
    if (
        model.kflow_contract.architecture_version
        != expected_architecture_version
    ):
        raise ValueError(
            f"{method_label} all-mosaic Flow mean/std factory requires "
            f"architecture {expected_architecture_version!r}, got "
            f"{model.kflow_contract.architecture_version!r}."
        )
    expected_normalization_type = (
        SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract
        if combined_contract is not None
        else HybridSourceP99FlowXYZMeanStdGeometryNormalizationContract
    )
    if not isinstance(model.normalization_contract, expected_normalization_type):
        raise TypeError(
            f"{method_label} all-mosaic training produced unexpected geometry "
            f"normalization {type(model.normalization_contract).__name__}."
        )
    if (
        model.normalization_contract.flow_population
        != HYBRID_FLOW_POPULATION_ALL_VALID
    ):
        raise ValueError(
            f"{method_label} all-mosaic training requires statistics over all "
            "valid mosaic "
            f"Flow values, got {model.normalization_contract.flow_population!r}."
        )
    geometry_source = {
        "kind": (
            "source_xyz_mean_std_and_all_mosaic_flow_xyz_mean_std"
            if source_contract is not None and not strict_statistics_provenance
            else (
                "bound_formal_source_xyz_mean_std_and_all_mosaic_flow_xyz_mean_std"
                if source_contract is not None
                else "bound_formal_all_mosaic_flow_xyz_mean_std_statistics_artifact"
            )
        ),
        "path": str(Path(original_artifact_path).expanduser().resolve(strict=True)),
        "schema_version": artifact.schema_version,
        "qualification": artifact.qualification,
        "split_name": artifact.split_name,
        "split_manifest_sha256": artifact.split_manifest_sha256,
        "dataset_receipt_sha256": artifact.dataset_receipt_sha256,
        "artifact_digest": artifact.artifact_digest,
        "statistics_id": artifact.normalization_contract.statistics_id,
    }
    if source_contract is not None:
        assert original_source_artifact_path is not None
        geometry_source.update(
            {
                "source_xyz_path": str(
                    Path(original_source_artifact_path)
                    .expanduser()
                    .resolve(strict=True)
                ),
                "flow_xyz_path": geometry_source["path"],
                "source_statistics_id": source_contract.statistics_id,
                "source_artifact_digest": source_contract.digest,
                "source_mean_xyz_meters": list(source_contract.xyz_center),
                "source_std_xyz_meters": list(source_contract.xyz_scale),
                "flow_mean_xyz_meters": list(
                    artifact.normalization_contract.flow_mean_xyz_meters
                ),
                "flow_std_xyz_meters": list(
                    artifact.normalization_contract.flow_std_xyz_meters
                ),
            }
        )
    model.normalization_contract_source = geometry_source
    model.normalization_contract_sources["geometry"] = geometry_source
    return model


def create_v5_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build V5/no-residual with all-mosaic Flow mean/std supervision."""

    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
        method_label="V5 no-residual",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        **kwargs,
    )


def create_v6_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build V6 residual with all-mosaic Flow mean/std supervision."""

    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION
        ),
        method_label="V6 residual",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        **kwargs,
    )


def create_v5_all_mosaic_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build V5/no-residual with separate source and Flow XYZ z-scores."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError("V5 Source XYZ mean/std artifact path is required.")
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=FASTWAM_FULLRES_KFLOW_ARCHITECTURE_VERSION,
        method_label="V5 no-residual Source/Flow mean-std",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=(
            source_xyz_normalization_artifact_path
        ),
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )


def create_v5_local24_all_mosaic_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build V5 Local24/no-residual with separate Source/Flow XYZ z-scores."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError("V5 Local24 Source XYZ mean/std artifact path is required.")
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_LOCAL24_FUSION_ARCHITECTURE_VERSION
        ),
        method_label="V5 Local24 no-residual Source/Flow mean-std",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=(
            source_xyz_normalization_artifact_path
        ),
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )


def create_v6_all_mosaic_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build V6 residual with separate source and Flow XYZ z-scores."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError("V6 Source XYZ mean/std artifact path is required.")
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION
        ),
        method_label="V6 residual Source/Flow mean-std",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=(
            source_xyz_normalization_artifact_path
        ),
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )


def create_v7_h8s4_local24_raw_flow_residual_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build V7 H8-S4 Local24 raw residual with Source/Flow XYZ z-scores."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError("V7 H8-S4 Source XYZ mean/std artifact path is required.")
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_LOCAL24_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION
        ),
        method_label="V7 H8-S4 Local24 raw residual Source/Flow mean-std",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=(
            source_xyz_normalization_artifact_path
        ),
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )


def create_v7_h8s4_local24_raw_flow_residual_action30_pit_kv_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build V7 H8-S4 Local24 residual with PiT-only Action30 tail K/V."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError("V7 Action30 PiT-KV Source XYZ artifact is required.")
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_PIT_KV_ARCHITECTURE_VERSION
        ),
        method_label="V7 H8-S4 Local24 Action30 PiT-only K/V",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=(
            source_xyz_normalization_artifact_path
        ),
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )


def create_v7_h8s4_local24_raw_flow_residual_action30_all_native_kv_mot_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build V7 World with all-30 Global/PiT native-K/V mixed Action MoT."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError("V7 all-30 native-K/V MoT Source XYZ artifact is required.")
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
        ),
        method_label="V7 H8-S4 Local24 all-30 native-K/V MoT",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=(
            source_xyz_normalization_artifact_path
        ),
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )



def create_v7_h8s4_local24_no_residual_action30_all_native_kv_mot_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build Local24/no-residual World with all-30 native-K/V Action MoT."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError(
            "V7 no-residual all-30 native-K/V MoT Source XYZ artifact is required."
        )
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
        ),
        method_label="V7 H8-S4 Local24 no-residual all-30 native-K/V MoT",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=(
            source_xyz_normalization_artifact_path
        ),
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )


def create_v8_h8s4_local24_no_residual_action30_linear14_aligned_native_kv_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build the frozen-World V8 Linear14 aligned native-K/V model."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError(
            "V8 Linear14 aligned native-K/V Source XYZ artifact is required."
        )
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION
        ),
        method_label=(
            "V8 H8-S4 Local24 no-residual Linear14 aligned native-K/V"
        ),
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=(
            source_xyz_normalization_artifact_path
        ),
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )


def create_v9_h8s4_local24_no_residual_action30_linear14_aligned_native_kv_cosine_anneal_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build V9 with the controlled 500-update native-K/V cosine anneal."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError(
            "V9 Linear14 aligned native-K/V Source XYZ artifact is required."
        )
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ACTION30_ALIGNED_LINEAR14_NATIVE_KV_ARCHITECTURE_VERSION
        ),
        method_label=(
            "V9 H8-S4 Local24 no-residual Linear14 aligned native-K/V cosine anneal"
        ),
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=(
            source_xyz_normalization_artifact_path
        ),
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )


def create_v7_h8s4_local24_raw_flow_residual_action26_global_native_kv_mot_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build Action26 with Global native-K/V MoT and World-only PiT."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError(
            "V7 Action26 Global-native-K/V Source XYZ artifact is required."
        )
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
        ),
        method_label=(
            "V7 H8-S4 Local24 Action26 Global native-K/V; PiT World-only"
        ),
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=(
            source_xyz_normalization_artifact_path
        ),
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )


def create_v7_h8s4_local24_raw_flow_residual_action26_global_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build paired V7 World with only 26 Global-to-Action bridges."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError("V7 Action26 Global-only Source XYZ artifact is required.")
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_RAW_FLOW_RESIDUAL_ACTION26_GLOBAL_ARCHITECTURE_VERSION
        ),
        method_label="V7 H8-S4 Local24 Action26 Global-only",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=(
            source_xyz_normalization_artifact_path
        ),
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )


def create_v7_h32s1_local96_raw_flow_residual_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build V7 H32-S1 Local96 from one H32-bound combined XYZ artifact."""

    model = create_fastwam_pixeldit(
        *args,
        flow_supervision_scope=FLOW_SUPERVISION_ALL_MOSAIC,
        **kwargs,
    )
    if model.kflow_contract.architecture_version != (
        FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION
    ):
        raise ValueError("V7 H32-S1 factory received a different architecture.")
    if not isinstance(
        model.normalization_contract,
        SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract,
    ):
        raise TypeError("V7 H32-S1 requires one combined Source/Flow XYZ contract.")
    if model.normalization_contract.flow_channels != 96:
        raise ValueError("V7 H32-S1 normalization must broadcast XYZ3 to 96 channels.")
    return model


def create_v7_h32s1_local96_raw_flow_residual_action30_all_native_kv_mot_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build H32-S1 Local96 with all-30 Global/PiT native-K/V Action MoT."""

    model = create_fastwam_pixeldit(
        *args,
        flow_supervision_scope=FLOW_SUPERVISION_ALL_MOSAIC,
        **kwargs,
    )
    if model.kflow_contract.architecture_version != (
        FASTWAM_FULLRES_KFLOW_H32S1_LOCAL96_RAW_FLOW_RESIDUAL_ACTION30_ALL_NATIVE_KV_MOT_ARCHITECTURE_VERSION
    ):
        raise ValueError(
            "V7 H32-S1 all-native factory received a different architecture."
        )
    if not isinstance(
        model.normalization_contract,
        SourceXYZMeanStdFlowXYZMeanStdGeometryNormalizationContract,
    ):
        raise TypeError(
            "V7 H32-S1 all-native requires one combined Source/Flow XYZ contract."
        )
    if model.normalization_contract.flow_channels != 96:
        raise ValueError(
            "V7 H32-S1 all-native normalization must broadcast XYZ3 to 96 channels."
        )
    return model
def create_v7_h8s4_local16_raw_flow_residual_ablation_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build V7 H8-S4 residual-only Local16 ablation."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError("V7 Local16 residual Source XYZ artifact is required.")
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL16_RAW_FLOW_RESIDUAL_ABLATION_ARCHITECTURE_VERSION
        ),
        method_label="V7 H8-S4 Local16 residual ablation",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=source_xyz_normalization_artifact_path,
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )


def create_v7_h8s4_local24_no_residual_ablation_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build V7 H8-S4 Local24-only no-residual ablation."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError("V7 Local24 no-residual Source XYZ artifact is required.")
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_H8S4_LOCAL24_NO_RESIDUAL_ABLATION_ARCHITECTURE_VERSION
        ),
        method_label="V7 H8-S4 Local24 no-residual ablation",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=source_xyz_normalization_artifact_path,
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )


def create_k1_v6_all_mosaic_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit(
    *args: Any,
    source_xyz_normalization_artifact_path: str | os.PathLike[str] | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_dataset_receipt_sha256: str | None = None,
    **kwargs: Any,
) -> FastWAMPixelDiT:
    """Build the versioned K=1 V6 residual Source/Flow mean-std experiment."""

    if source_xyz_normalization_artifact_path is None:
        raise ValueError("K1 V6 Source XYZ mean/std artifact path is required.")
    return _create_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit(
        *args,
        expected_architecture_version=(
            FASTWAM_FULLRES_KFLOW_K1_RAW_FLOW_RESIDUAL_ARCHITECTURE_VERSION
        ),
        method_label="K1 V6 residual Source/Flow mean-std",
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_dataset_receipt_sha256=expected_dataset_receipt_sha256,
        source_xyz_normalization_artifact_path=(
            source_xyz_normalization_artifact_path
        ),
        allow_formal_flow_calibration=True,
        strict_statistics_provenance=False,
        **kwargs,
    )


def create_hybrid_k4_fastwam_pixeldit(
    *,
    pixeldit_config,
    aligned_action_dit_config,
    kflow_contract,
    method_contract,
    normalization_contract=None,
    normalization_artifact_path: str | None = None,
    action_normalization_contract=None,
    action_normalization_artifact_path: str | None = None,
    proprio_normalization_contract=None,
    proprio_normalization_artifact_path: str | None = None,
    proprio_dim: int = 14,
    pixeldit_pretrained_path: str,
    pixeldit_weight_source: str = "auto",
    action_initialization_seed: int = 43,
    train_world_scheduler=None,
    infer_world_scheduler=None,
    train_action_scheduler=None,
    infer_action_scheduler=None,
    timestep_coupling: str = "independent",
    loss=None,
    kv_source: str,
    action_tokenization: str = "joint14",
    model_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> HybridFastWAMPixelDiT:
    """Build a fresh ImageNet-initialized K4 Hybrid C/W + Action model."""

    resolved_method = resolve_formal_method_contract(
        _plain_dict(method_contract, name="method_contract")
    )
    expected_methods = {
        (HYBRID_CW_KV_SOURCE_ACW, "joint14"): (
            FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW,
            FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2,
        ),
        (HYBRID_CW_KV_SOURCE_AC, "joint14"): (
            FORMAL_METHOD_CONTRACT_K4_HYBRID_AC,
            FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2,
        ),
        (HYBRID_CW_KV_SOURCE_AC, "temporal32_rope"): (
            FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2,
        ),
    }.get((kv_source, action_tokenization))
    if expected_methods is None:
        raise ValueError(
            "Unsupported Hybrid Action layout: "
            f"kv_source={kv_source!r}, action_tokenization={action_tokenization!r}."
        )
    if resolved_method not in expected_methods:
        raise ValueError("Hybrid kv_source and method_contract disagree.")
    resolved_kflow = _construct_dataclass(
        kflow_contract,
        KFlowContract,
        name="kflow_contract",
    )
    if (normalization_contract is None) == (normalization_artifact_path is None):
        raise ValueError(
            "Hybrid K4 requires exactly one of `normalization_contract` or "
            "`normalization_artifact_path`."
        )
    if normalization_artifact_path is not None:
        geometry, geometry_source = load_bound_hybrid_k4_normalization_asset(
            normalization_artifact_path
        )
    else:
        geometry = _construct_dataclass(
            normalization_contract,
            HybridHeadWristNormalizationContract,
            name="normalization_contract",
        )
        geometry_source = {
            "kind": "inline_fixed_heterogeneous_k4",
            "schema_version": geometry.schema_version,
            "statistics_id": geometry.statistics_id,
        }
    if (
        geometry.future_offsets != resolved_kflow.future_offsets
        or geometry.reference_frame != resolved_kflow.coordinate_frame
        or geometry.source_unit != resolved_kflow.unit
        or geometry.channel_order != resolved_kflow.channel_order
    ):
        raise ValueError("Hybrid normalization is not bound to the KFlow contract.")
    expected_normalization_schema = (
        HYBRID_HEAD_WRIST_POOLED_NORMALIZATION_SCHEMA_VERSION
        if resolved_method
        in (
            FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2,
            FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2,
            FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2,
        )
        else HYBRID_HEAD_WRIST_NORMALIZATION_SCHEMA_VERSION
    )
    if geometry.schema_version != expected_normalization_schema:
        raise ValueError(
            "Hybrid method contract and normalization schema disagree: "
            f"{resolved_method['schema_version']!r} requires "
            f"{expected_normalization_schema!r}, got {geometry.schema_version!r}."
        )
    action_norm, action_norm_source = _resolve_feature_contract(
        action_normalization_contract,
        action_normalization_artifact_path,
        kind="action",
    )
    if int(proprio_dim) != 14:
        raise ValueError("Hybrid proprio_dim is fixed at 14.")
    proprio_norm, proprio_norm_source = _resolve_feature_contract(
        proprio_normalization_contract,
        proprio_normalization_artifact_path,
        kind="proprio",
    )
    if timestep_coupling != "independent":
        raise ValueError("Hybrid World/Action training timesteps are independent.")
    schedulers = {
        "train_world": _scheduler(train_world_scheduler, name="train_world_scheduler"),
        "infer_world": _scheduler(infer_world_scheduler, name="infer_world_scheduler"),
        "train_action": _scheduler(
            train_action_scheduler,
            name="train_action_scheduler",
        ),
        "infer_action": _scheduler(
            infer_action_scheduler,
            name="infer_action_scheduler",
        ),
    }
    if not isinstance(pixeldit_pretrained_path, str) or not pixeldit_pretrained_path:
        raise ValueError("Hybrid World requires the ImageNet PixelDiT checkpoint path.")
    world_config = _plain_dict(pixeldit_config, name="pixeldit_config")
    fixed_world = {
        "condition_channels": 9,
        "k": 4,
        "hidden_size": 1152,
        "pixel_hidden_size": 16,
        "patch_depth": 26,
        "pixel_depth": 4,
        "patch_size": 16,
        "num_groups": 16,
    }
    for key, expected in fixed_world.items():
        if key in world_config and world_config[key] != expected:
            raise ValueError(
                f"Hybrid World fixed contract drift for {key}: "
                f"{world_config[key]!r} != {expected!r}."
            )
        world_config[key] = expected
    world = HybridCWPixelDiT(**world_config).to(device=device, dtype=model_dtype)
    pixel_report = world.load_pretrained(
        pixeldit_pretrained_path,
        weight_source=pixeldit_weight_source,
        map_location="cpu",
    )

    action_config = _plain_dict(
        aligned_action_dit_config,
        name="aligned_action_dit_config",
    )
    fixed_action = {
        "action_horizon": 32,
        "action_dim": 14,
        "hidden_size": 1152,
        "num_heads": 16,
        "num_layers": 30,
        "text_dim": 4096,
    }
    for key, expected in fixed_action.items():
        if key in action_config and action_config[key] != expected:
            raise ValueError(
                f"Hybrid Action fixed contract drift for {key}: "
                f"{action_config[key]!r} != {expected!r}."
            )
        action_config[key] = expected
    if (
        isinstance(action_initialization_seed, bool)
        or not isinstance(action_initialization_seed, int)
        or not 0 <= action_initialization_seed < 2**32
    ):
        raise ValueError("action_initialization_seed must be a uint32 integer.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(action_initialization_seed)
        action_class = (
            PixelDiTTemporalActionDiT
            if action_tokenization == "temporal32_rope"
            else PixelDiTAlignedActionDiT
        )
        action = action_class(**action_config)
    action.native_kv_fusion_version = HYBRID_CW_NATIVE_KV_FUSION_VERSION
    action = action.to(device=device, dtype=model_dtype)
    action_report = initialize_action_from_hybrid_world(world, action)
    action.pretrained_load_report = action_report
    dit = HybridCWActionMoT(
        world,
        action,
        kv_source=kv_source,
        use_gradient_checkpointing=bool(
            world_config.get("use_gradient_checkpointing", False)
        ),
        validate_finite=bool(world_config.get("validate_finite", False)),
    ).to(device=device, dtype=model_dtype)
    loss_config = _plain_dict(loss, name="loss")
    if set(loss_config) != {"lambda_momap", "lambda_action"}:
        raise ValueError("Hybrid `loss` requires lambda_momap and lambda_action.")
    model = HybridFastWAMPixelDiT(
        dit,
        kflow_contract=resolved_kflow,
        normalization_contract=geometry,
        action_horizon=32,
        action_normalization_contract=action_norm,
        text_dim=4096,
        batch_adapter=MoMapBatchAdapter(
            resolved_kflow,
            action_label_kind=action_norm.semantic_kind,
            action_is_high_level_policy_action=False,
            action_normalization=action_norm.source_normalization,
            proprio_mode=proprio_norm.semantic_kind,
            proprio_normalization=proprio_norm.source_normalization,
        ),
        proprio_dim=14,
        proprio_normalization_contract=proprio_norm,
        train_world_scheduler=schedulers["train_world"],
        infer_world_scheduler=schedulers["infer_world"],
        train_action_scheduler=schedulers["train_action"],
        infer_action_scheduler=schedulers["infer_action"],
        timestep_coupling="independent",
        loss_lambda_momap=float(loss_config["lambda_momap"]),
        loss_lambda_action=float(loss_config["lambda_action"]),
        method_contract=resolved_method,
    ).to(device=device, dtype=model_dtype)
    model.pixeldit_load_report = pixel_report
    model.action_dit_load_report = action_report
    model.pretrained_load_reports = {
        "pixeldit": pixel_report,
        "action_dit": action_report,
    }
    model.normalization_contract_source = geometry_source
    model.normalization_contract_sources = {
        "geometry": model.normalization_contract_source,
        "action": action_norm_source,
        "proprio": proprio_norm_source,
    }
    return model


__all__ = [
    "create_hybrid_k4_fastwam_pixeldit",
    "create_k1_v6_all_mosaic_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
    "create_fastwam_pixeldit",
    "create_head_only_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v5_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v5_all_mosaic_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v5_local24_all_mosaic_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v6_all_mosaic_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v7_h8s4_local24_raw_flow_residual_action30_all_native_kv_mot_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v7_h8s4_local24_no_residual_action30_all_native_kv_mot_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v8_h8s4_local24_no_residual_action30_linear14_aligned_native_kv_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v6_all_mosaic_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v7_h8s4_local24_raw_flow_residual_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v7_h8s4_local24_raw_flow_residual_action30_pit_kv_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v7_h8s4_local24_raw_flow_residual_action26_global_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v7_h32s1_local96_raw_flow_residual_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v7_h32s1_local96_raw_flow_residual_action30_all_native_kv_mot_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v7_h8s4_local16_raw_flow_residual_ablation_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
    "create_v7_h8s4_local24_no_residual_ablation_source_xyz_mean_std_flow_xyz_mean_std_fastwam_pixeldit",
]
