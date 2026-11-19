"""Build the RoboTwin joint action and 3D motion model."""
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from ..wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
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
from .kflow import KFlowContract, MoMapBatchAdapter
from .method_contract import (
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2,
    FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2,
    resolve_formal_method_contract,
)
from .normalization import FixedFeatureNormalizationContract
from .pixeldit_aligned_action import (
    PixelDiTAlignedActionDiT,
    PixelDiTTemporalActionDiT,
)


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
