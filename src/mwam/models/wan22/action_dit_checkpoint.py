"""ActionDiT checkpoint loading and offline backbone conversion utilities.

Layer-count adaptation is a checkpoint preparation concern, not part of the
ActionDiT forward path. Keeping checkpoint I/O here leaves the model module
focused on its definition while retaining FastWAM's strict loading behavior.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from mwam.utils.checkpoint_integrity import validate_floating_state_dict


UNIFORM_ENDPOINTS_30_TO_26_V1 = "uniform_endpoints_30_to_26_v1"
CONVERSION_SCHEMA_VERSION = "action_dit_backbone_layer_conversion_v1"


def uniform_endpoints_30_to_26() -> tuple[int, ...]:
    """Return the frozen endpoint-preserving 30→26 layer selection."""

    mapping = tuple(round(target_layer * 29 / 25) for target_layer in range(26))
    if (
        len(mapping) != 26
        or mapping[0] != 0
        or mapping[-1] != 29
        or len(set(mapping)) != 26
        or any(right <= left for left, right in zip(mapping, mapping[1:]))
    ):
        raise RuntimeError("invalid internal 30-to-26 ActionDiT layer mapping")
    return mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_payload(payload: Any) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not isinstance(payload, dict):
        raise TypeError(
            "ActionDiT backbone payload must be a dict, "
            f"got {type(payload).__name__}"
        )
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise TypeError("ActionDiT backbone payload `meta` must be a dict")
    if isinstance(meta.get("num_layers"), bool) or int(meta.get("num_layers", -1)) != 30:
        raise ValueError(
            "30-to-26 conversion requires `meta.num_layers=30`, "
            f"got {meta.get('num_layers')!r}"
        )
    state = payload.get("backbone_state_dict")
    if not isinstance(state, dict):
        raise TypeError(
            "ActionDiT backbone payload `backbone_state_dict` must be a dict"
        )
    if any(not isinstance(key, str) for key in state):
        raise TypeError("ActionDiT backbone state keys must all be strings")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"ActionDiT backbone state {key!r} must be a tensor, "
                f"got {type(value).__name__}"
            )
    return meta, state


def _split_state(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    non_blocks: dict[str, torch.Tensor] = {}
    blocks: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        if not key.startswith("blocks."):
            non_blocks[key] = value
            continue
        parts = key.split(".", 2)
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2]:
            raise ValueError(f"malformed ActionDiT block key: {key!r}")
        layer = int(parts[1])
        if not 0 <= layer < 30:
            raise ValueError(f"ActionDiT source block index out of range: {layer}")
        blocks.setdefault(layer, {})[parts[2]] = value

    if set(blocks) != set(range(30)):
        raise ValueError(
            "ActionDiT source payload must contain exactly blocks 0..29; "
            f"missing={sorted(set(range(30)) - set(blocks))}, "
            f"unexpected={sorted(set(blocks) - set(range(30)))}"
        )
    reference_suffixes = set(blocks[0])
    if not reference_suffixes:
        raise ValueError("ActionDiT source block 0 is empty")
    for layer in range(30):
        suffixes = set(blocks[layer])
        if suffixes != reference_suffixes:
            raise ValueError(
                "ActionDiT block schemas differ: "
                f"block0_vs_block{layer}; "
                f"missing={sorted(reference_suffixes - suffixes)}, "
                f"unexpected={sorted(suffixes - reference_suffixes)}"
            )
    return non_blocks, blocks


def convert_30_to_26_payload(
    payload: Any,
    *,
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one strict 30-layer backbone payload into a strict 26-layer one."""

    meta, state = _require_payload(payload)
    non_blocks, blocks = _split_state(state)
    mapping = uniform_endpoints_30_to_26()

    converted_state: dict[str, torch.Tensor] = {
        key: value.detach().to(device="cpu").contiguous()
        for key, value in sorted(non_blocks.items())
    }
    for target_layer, source_layer in enumerate(mapping):
        for suffix, value in sorted(blocks[source_layer].items()):
            converted_state[f"blocks.{target_layer}.{suffix}"] = (
                value.detach().to(device="cpu").contiguous()
            )

    dropped = tuple(layer for layer in range(30) if layer not in set(mapping))
    converted_meta = dict(meta)
    converted_meta["num_layers"] = 26
    report: dict[str, Any] = {
        "schema_version": CONVERSION_SCHEMA_VERSION,
        "mapping": UNIFORM_ENDPOINTS_30_TO_26_V1,
        "source_num_layers": 30,
        "target_num_layers": 26,
        "target_to_source_layers": list(mapping),
        "dropped_source_layers": list(dropped),
        "optimizer_state_policy": "reset_required",
    }
    if source_identity is not None:
        report["source_identity"] = dict(source_identity)

    source_policy = payload.get("policy")
    converted_policy: dict[str, Any] = {
        "layer_conversion": UNIFORM_ENDPOINTS_30_TO_26_V1,
        "optimizer_state_policy": "reset_required",
    }
    if isinstance(source_policy, Mapping):
        converted_policy["source_policy"] = dict(source_policy)

    return {
        "policy": converted_policy,
        "backbone_state_dict": converted_state,
        "meta": converted_meta,
        "conversion_report": report,
    }


def convert_30_to_26_file(
    source_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Load, convert and save a payload without initializing ActionDiT."""

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source ActionDiT payload does not exist: {source}")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output ActionDiT payload already exists (refusing overwrite): {output}"
        )
    if output == source:
        raise ValueError("source and output ActionDiT payload paths must differ")

    payload = torch.load(
        source,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    source_identity = {
        "filename": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }
    converted = convert_30_to_26_payload(
        payload,
        source_identity=source_identity,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output)
    return dict(converted["conversion_report"])


def load_action_dit_pretrained(
    *,
    action_dit_cls: type[Any],
    action_dit_config: dict[str, Any],
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    logger: Any,
) -> Any:
    """Load one strict ActionDiT backbone payload into a new model instance."""

    if action_dit_config is None:
        raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
    if skip_dit_load_from_pretrain:
        logger.info(
            "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
            "initializing action expert randomly and expecting checkpoint override."
        )
        return action_dit_cls(**action_dit_config).to(device=device, dtype=torch_dtype)
    if not action_dit_pretrained_path:
        logger.info("No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.")
        return action_dit_cls(**action_dit_config).to(device=device, dtype=torch_dtype)
    p = Path(action_dit_pretrained_path).expanduser()
    if not p.is_absolute():
        from mwam.utils.paths import asset_root
        p = asset_root() / p
    p = p.resolve(strict=True)
    if not p.is_file():
        raise FileNotFoundError(
            f"`action_dit_pretrained_path` is not a regular file: {p}"
        )
    checkpoint_identity = {
        "path": str(p),
        "size_bytes": int(p.stat().st_size),
    }
    action_dit_pretrained_path = str(p)

    action_cfg = dict(action_dit_config)
    action_expert = action_dit_cls(**action_cfg).to(device=device, dtype=torch_dtype)
    action_state = action_expert.state_dict()
    expected_backbone_keys = action_dit_cls.backbone_key_set(action_state.keys())

    # Eight data-parallel ranks load the same immutable payload.  Memory
    # mapping avoids materializing eight independent CPU copies before the
    # selected tensors are cast into their rank-local modules.
    payload = torch.load(
        action_dit_pretrained_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
        )

    policy = payload.get("policy", {})
    if not isinstance(policy, Mapping):
        raise TypeError("ActionDiT checkpoint `policy` must be a mapping.")
    if policy:
        logger.info("ActionDiT backbone payload policy: %s", dict(policy))

    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise TypeError("ActionDiT checkpoint `meta` must be a mapping.")
    expected_meta = {
        "hidden_dim": int(action_cfg["hidden_dim"]),
        "ffn_dim": int(action_cfg["ffn_dim"]),
        "num_layers": int(action_cfg["num_layers"]),
        "num_heads": int(action_cfg["num_heads"]),
        "attn_head_dim": int(action_cfg["attn_head_dim"]),
        "text_dim": int(action_cfg["text_dim"]),
        "freq_dim": int(action_cfg["freq_dim"]),
        "eps": float(action_cfg["eps"]),
    }
    for key in action_dit_cls.ACTION_BACKBONE_META_KEYS:
        if key not in meta:
            raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
        expected_value = expected_meta[key]
        got_value = meta[key]
        if isinstance(got_value, bool) or type(got_value) not in (int, float):
            raise TypeError(
                f"`meta.{key}` in {action_dit_pretrained_path} must be a "
                f"built-in finite number, got {type(got_value).__name__}"
            )
        got_float = float(got_value)
        if not math.isfinite(got_float):
            raise ValueError(
                f"`meta.{key}` in {action_dit_pretrained_path} must be finite"
            )
        if key == "eps":
            if abs(got_float - float(expected_value)) > 1e-12:
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )
        elif not got_float.is_integer() or int(got_float) != int(expected_value):
            raise ValueError(
                f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                f"expected {expected_value}, got {got_value}"
            )

    conversion_report = payload.get("conversion_report")
    layer_conversion = policy.get("layer_conversion")
    if conversion_report is not None or layer_conversion is not None:
        if not isinstance(conversion_report, Mapping):
            raise TypeError(
                "Converted ActionDiT checkpoint `conversion_report` must "
                "be a mapping."
            )
        mapping = list(uniform_endpoints_30_to_26())
        dropped = [layer for layer in range(30) if layer not in set(mapping)]
        expected_conversion = {
            "schema_version": CONVERSION_SCHEMA_VERSION,
            "mapping": UNIFORM_ENDPOINTS_30_TO_26_V1,
            "source_num_layers": 30,
            "target_num_layers": 26,
            "target_to_source_layers": mapping,
            "dropped_source_layers": dropped,
            "optimizer_state_policy": "reset_required",
        }
        for key, expected_value in expected_conversion.items():
            if conversion_report.get(key) != expected_value:
                raise ValueError(
                    f"ActionDiT conversion_report.{key} mismatch: "
                    f"expected {expected_value!r}, got "
                    f"{conversion_report.get(key)!r}"
                )
        if (
            int(expected_meta["num_layers"]) != 26
            or layer_conversion != UNIFORM_ENDPOINTS_30_TO_26_V1
            or policy.get("optimizer_state_policy") != "reset_required"
        ):
            raise ValueError("ActionDiT conversion policy/meta mismatch.")
        # Historical source identity is provenance-only metadata. The
        # loaded 26-layer tensor names and shapes are validated below.

    backbone_state_dict = payload.get("backbone_state_dict")
    if not isinstance(backbone_state_dict, dict):
        raise ValueError(
            f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
            f"got {type(backbone_state_dict)}"
        )

    state_integrity = validate_floating_state_dict(
        backbone_state_dict,
        name="backbone_state_dict",
    )
    provided_keys = set(backbone_state_dict.keys())
    missing_keys = sorted(expected_backbone_keys - provided_keys)
    unexpected_keys = sorted(provided_keys - expected_backbone_keys)
    if missing_keys or unexpected_keys:
        raise ValueError(
            "Action backbone key mismatch in preprocessed payload. "
            f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
            f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
        )

    merged_state = dict(action_state)
    for key in expected_backbone_keys:
        value = backbone_state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                f"got {type(value)}"
            )
        target = merged_state[key]
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
            )
        merged_state[key] = value.to(device=target.device, dtype=target.dtype)

    action_expert.load_state_dict(merged_state, strict=True)
    logger.info(
        "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
        action_dit_pretrained_path,
        len(expected_backbone_keys),
        list(action_dit_cls.ACTION_BACKBONE_SKIP_PREFIXES),
    )
    optimizer_state_policy = policy.get("optimizer_state_policy")
    action_expert.pretrained_load_report = {
        "schema_version": "action_dit_pretrained_load_report_v2",
        "checkpoint_identity": dict(checkpoint_identity),
        "meta": dict(expected_meta),
        "policy": copy.deepcopy(dict(policy)),
        "conversion_report": (
            None
            if conversion_report is None
            else copy.deepcopy(dict(conversion_report))
        ),
        "optimizer_state_policy": (
            "not_provided"
            if optimizer_state_policy is None
            else str(optimizer_state_policy)
        ),
        "loaded_key_count": len(expected_backbone_keys),
        "loaded_numel": int(state_integrity["numel"]),
        "source_dtypes": list(state_integrity["dtypes"]),
        "random_initialization_kept": list(action_dit_cls.ACTION_BACKBONE_SKIP_PREFIXES),
    }
    return action_expert.to(device=device, dtype=torch_dtype)

__all__ = [
    "CONVERSION_SCHEMA_VERSION",
    "UNIFORM_ENDPOINTS_30_TO_26_V1",
    "convert_30_to_26_file",
    "convert_30_to_26_payload",
    "load_action_dit_pretrained",
    "uniform_endpoints_30_to_26",
]
