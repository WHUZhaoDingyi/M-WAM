"""Audited paired initialization for the PixelDiT-aligned Action expert.

The random and World-transfer experiments start from the same seeded Action
state.  The transfer policy is deliberately narrow:

* World ``patch_blocks.0..25`` replace Action ``blocks.0..25`` in full;
* World ``pixel_blocks.0..3.attn.*`` replace only Action
  ``blocks.26..29.attn.*``;
* every other Action tensor remains byte-identical to the seeded random base.

The caller remains responsible for constructing both Action experts with the
same RNG seed.  The returned receipt exposes the random-base digest so two
independent runs can prove that invariant before training begins.
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
import torch.nn as nn

from mwam.utils.checkpoint_integrity import canonical_json_contract


ACTION14_RANDOM_INITIALIZATION = "same_seed_random_base_v1"
ACTION14_WORLD_TRANSFER_INITIALIZATION = (
    "world_step008119_global26_full_pit4_attention_v1"
)
ACTION14_INITIALIZATION_MODES = frozenset(
    {
        ACTION14_RANDOM_INITIALIZATION,
        ACTION14_WORLD_TRANSFER_INITIALIZATION,
    }
)
ACTION14_INITIALIZATION_RECEIPT_SCHEMA_VERSION = (
    "pixeldit_aligned_action_initialization_receipt_v1"
)
ACTION14_WORLD_TRANSFER_POLICY = (
    "patch_blocks_0_25_full_pixel_blocks_0_3_attention_only_v1"
)
ACTION14_RANDOM_BASE_POLICY = "caller_seeded_action_state_preserved_v1"

FROZEN_WORLD_SOURCE_FORMAT = "fastwam_pixeldit_momap_v2"
FROZEN_WORLD_SOURCE_STEP = 8119
FROZEN_WORLD_SOURCE_SHA256 = (
    "1756782d409aae5fd908eee1adbbc470854ffcc7d1974fbe8dec09415f025bf4"
)
FROZEN_WORLD_SOURCE_SIZE_BYTES = 4_315_646_932

WORLD_PATCH_BLOCK_PREFIX = "dit.world_expert.patch_blocks."
WORLD_PIXEL_BLOCK_PREFIX = "dit.world_expert.pixel_blocks."
ACTION_BLOCK_PREFIX = "blocks."
GLOBAL_BLOCK_COUNT = 26
PIXEL_BLOCK_COUNT = 4
ACTION_BLOCK_COUNT = GLOBAL_BLOCK_COUNT + PIXEL_BLOCK_COUNT
ACTION_NON_BLOCK_KEYS = frozenset(
    {
        "action_encoder.bias",
        "action_encoder.weight",
        "context_projection.bias",
        "context_projection.weight",
        "t_embedder.mlp.0.bias",
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.2.bias",
        "t_embedder.mlp.2.weight",
        "action_decoder.bias",
        "action_decoder.weight",
    }
)

_REQUIRED_PAYLOAD_KEYS = frozenset(
    {"format", "training_contract", "model_state", "step"}
)
_ALLOWED_PAYLOAD_KEYS = _REQUIRED_PAYLOAD_KEYS | frozenset(
    {"optimizer", "initialization", "optimizer_policy"}
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _require_tensor(
    value: Any,
    *,
    name: str,
    validate_finite: bool = True,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"`{name}` must be a torch.Tensor.")
    if value.layout != torch.strided:
        raise TypeError(f"`{name}` must use strided layout, got {value.layout}.")
    if not torch.is_floating_point(value):
        raise TypeError(f"`{name}` must be floating point, got {value.dtype}.")
    if value.numel() <= 0:
        raise ValueError(f"`{name}` must not be empty.")
    if validate_finite and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"`{name}` contains NaN or Inf.")
    return value


def tensor_content_sha256(tensor: torch.Tensor, *, name: str) -> str:
    """Hash exact tensor metadata and bytes, including BF16 tensors."""

    checked = _require_tensor(tensor, name=name)
    cpu = checked.detach().to(device="cpu").contiguous()
    metadata = {
        "dtype": _dtype_name(cpu),
        "shape": [int(size) for size in cpu.shape],
    }
    digest = hashlib.sha256()
    digest.update(b"fastwam-tensor-content-v1\0")
    digest.update(_canonical_json_bytes(metadata))
    digest.update(b"\0")
    # NumPy cannot expose bfloat16 directly.  Viewing as uint8 preserves the
    # exact storage bytes and works for every supported floating dtype.
    byte_view = cpu.view(torch.uint8).reshape(-1).numpy()
    digest.update(memoryview(byte_view))
    return digest.hexdigest()


def _state_schema(
    state: Mapping[str, torch.Tensor],
    *,
    name: str,
    validate_finite: bool,
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise TypeError(f"`{name}` must be a mapping.")
    entries: list[dict[str, Any]] = []
    for key in sorted(state):
        if not isinstance(key, str) or not key:
            raise TypeError(f"`{name}` keys must be non-empty strings.")
        tensor = _require_tensor(
            state[key],
            name=f"{name}[{key}]",
            validate_finite=validate_finite,
        )
        entries.append(
            {
                "key": key,
                "shape": [int(size) for size in tensor.shape],
                "dtype": _dtype_name(tensor),
                "numel": int(tensor.numel()),
            }
        )
    if not entries:
        raise ValueError(f"`{name}` must contain at least one tensor.")
    contract = canonical_json_contract(
        entries,
        domain="pixeldit-aligned-action-tensor-schema-v1",
        name=name,
    )
    return {
        "tensor_count": len(entries),
        "numel": sum(int(entry["numel"]) for entry in entries),
        "schema_sha256": contract["sha256"],
        "entries": entries,
    }


def _state_content_sha256(
    key_receipts: list[dict[str, Any]],
    *,
    hash_field: str,
) -> str:
    entries = [
        {
            "key": item["target_key"],
            "shape": item["target_shape"],
            "dtype": item["target_dtype"],
            "tensor_sha256": item[hash_field],
        }
        for item in key_receipts
    ]
    return str(
        canonical_json_contract(
            entries,
            domain="pixeldit-aligned-action-state-content-v1",
            name=hash_field,
        )["sha256"]
    )


def _joint_position_buffer(
    action_expert: nn.Module,
    action_state: Mapping[str, torch.Tensor],
    *,
    validate_finite: bool,
) -> torch.Tensor:
    non_state_buffers = {
        name: value
        for name, value in action_expert.named_buffers()
        if name not in action_state
    }
    if set(non_state_buffers) != {"joint_pos_embed"}:
        raise ValueError(
            "PixelDiT-aligned Action must expose exactly one non-state buffer "
            "named `joint_pos_embed`; observed="
            f"{sorted(non_state_buffers)}."
        )
    position = _require_tensor(
        non_state_buffers["joint_pos_embed"],
        name="action_expert.joint_pos_embed",
        validate_finite=validate_finite,
    )
    action_dim = getattr(action_expert, "action_dim", None)
    hidden_size = getattr(action_expert, "hidden_size", None)
    if (
        isinstance(action_dim, bool)
        or not isinstance(action_dim, int)
        or isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
    ):
        raise TypeError(
            "PixelDiT-aligned Action must expose integer `action_dim` and "
            "`hidden_size` attributes."
        )
    expected_shape = (1, action_dim, hidden_size)
    if tuple(position.shape) != expected_shape:
        raise ValueError(
            "Action joint-position buffer shape mismatch: expected "
            f"{expected_shape}, got {tuple(position.shape)}."
        )
    return position


def _split_indexed_blocks(
    state: Mapping[str, torch.Tensor],
    *,
    prefix: str,
    name: str,
) -> dict[int, dict[str, torch.Tensor]]:
    blocks: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        parts = remainder.split(".", 1)
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1]:
            raise ValueError(f"Malformed `{name}` key: {key!r}.")
        index = int(parts[0])
        suffix = parts[1]
        if suffix in blocks.setdefault(index, {}):
            raise ValueError(f"Duplicate `{name}` key: {key!r}.")
        blocks[index][suffix] = value
    return blocks


def _require_exact_indices(
    blocks: Mapping[int, Any],
    *,
    expected_count: int,
    name: str,
) -> None:
    expected = set(range(expected_count))
    observed = set(blocks)
    if observed != expected:
        raise ValueError(
            f"`{name}` must contain exactly blocks 0..{expected_count - 1}; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}."
        )


def _require_same_tensor_schema(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    source_key: str,
    target_key: str,
    validate_finite: bool,
) -> None:
    source = _require_tensor(
        source,
        name=f"source_state[{source_key}]",
        validate_finite=validate_finite,
    )
    target = _require_tensor(
        target,
        name=f"action_state[{target_key}]",
        validate_finite=validate_finite,
    )
    if source.shape != target.shape or source.dtype != target.dtype:
        raise ValueError(
            "World-to-Action tensor schema mismatch: "
            f"{source_key} shape={tuple(source.shape)} dtype={source.dtype} -> "
            f"{target_key} shape={tuple(target.shape)} dtype={target.dtype}."
        )


def _validate_source_identity(source_identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source_identity, Mapping):
        raise TypeError("`source_identity` must be a mapping.")
    required = {"path", "size_bytes", "sha256"}
    missing = sorted(required - set(source_identity))
    unexpected = sorted(set(source_identity) - required)
    if missing or unexpected:
        raise ValueError(
            "Source checkpoint identity key mismatch: "
            f"missing={missing}, unexpected={unexpected}."
        )
    path = source_identity["path"]
    size_bytes = source_identity["size_bytes"]
    sha256 = source_identity["sha256"]
    if not isinstance(path, str) or not path:
        raise TypeError("`source_identity.path` must be a non-empty string.")
    if size_bytes != FROZEN_WORLD_SOURCE_SIZE_BYTES:
        raise ValueError(
            "Frozen World checkpoint size mismatch: expected "
            f"{FROZEN_WORLD_SOURCE_SIZE_BYTES}, got {size_bytes!r}."
        )
    if sha256 != FROZEN_WORLD_SOURCE_SHA256:
        raise ValueError(
            "Frozen World checkpoint SHA256 mismatch: expected "
            f"{FROZEN_WORLD_SOURCE_SHA256}, got {sha256!r}."
        )
    return {
        "path": path,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _validate_source_payload(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise TypeError("`source_payload` must be a mapping.")
    missing = sorted(_REQUIRED_PAYLOAD_KEYS - set(payload))
    unexpected = sorted(set(payload) - _ALLOWED_PAYLOAD_KEYS)
    if missing or unexpected:
        raise ValueError(
            "Source checkpoint top-level key mismatch: "
            f"missing={missing}, unexpected={unexpected}."
        )
    if payload["format"] != FROZEN_WORLD_SOURCE_FORMAT:
        raise ValueError(
            "Frozen World checkpoint format mismatch: expected "
            f"{FROZEN_WORLD_SOURCE_FORMAT!r}, got {payload['format']!r}."
        )
    if payload["step"] != FROZEN_WORLD_SOURCE_STEP:
        raise ValueError(
            "Frozen World checkpoint step mismatch: expected "
            f"{FROZEN_WORLD_SOURCE_STEP}, got {payload['step']!r}."
        )
    training_contract = payload["training_contract"]
    if not isinstance(training_contract, Mapping):
        raise TypeError("`source_payload.training_contract` must be a mapping.")
    training_contract_receipt = canonical_json_contract(
        training_contract,
        domain="pixeldit-action-world-source-training-contract-v1",
        name="source_payload.training_contract",
    )
    source_state = payload["model_state"]
    if not isinstance(source_state, Mapping):
        raise TypeError("`source_payload.model_state` must be a mapping.")
    for key, value in source_state.items():
        if not isinstance(key, str) or not key:
            raise TypeError("Source model-state keys must be non-empty strings.")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Source model-state entry {key!r} must be a tensor.")
    return source_state, training_contract_receipt


def initialize_pixeldit_aligned_action(
    action_expert: nn.Module,
    *,
    source_payload: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    mode: str,
    audit_content: bool = True,
) -> dict[str, Any]:
    """Apply one fail-closed paired initialization and return its receipt.

    All schema and content checks complete before any Action tensor is changed.
    ``mode=ACTION14_RANDOM_INITIALIZATION`` validates the same transfer source
    but leaves the entire seeded Action state untouched.
    """

    if not isinstance(action_expert, nn.Module):
        raise TypeError("`action_expert` must be an nn.Module.")
    if not isinstance(audit_content, bool):
        raise TypeError("`audit_content` must be boolean.")
    if mode not in ACTION14_INITIALIZATION_MODES:
        raise ValueError(
            f"Unsupported Action initialization mode {mode!r}; expected one of "
            f"{sorted(ACTION14_INITIALIZATION_MODES)}."
        )

    identity = _validate_source_identity(source_identity)
    source_state, training_contract_receipt = _validate_source_payload(
        source_payload
    )
    action_state = action_expert.state_dict()
    if not isinstance(action_state, Mapping):
        raise TypeError("`action_expert.state_dict()` must return a mapping.")
    joint_position = _joint_position_buffer(
        action_expert,
        action_state,
        validate_finite=audit_content,
    )
    random_position_hash = (
        tensor_content_sha256(
            joint_position,
            name="action_random_base[joint_pos_embed]",
        )
        if audit_content
        else None
    )

    source_global = _split_indexed_blocks(
        source_state,
        prefix=WORLD_PATCH_BLOCK_PREFIX,
        name="World patch blocks",
    )
    source_pixel = _split_indexed_blocks(
        source_state,
        prefix=WORLD_PIXEL_BLOCK_PREFIX,
        name="World pixel blocks",
    )
    action_blocks = _split_indexed_blocks(
        action_state,
        prefix=ACTION_BLOCK_PREFIX,
        name="Action blocks",
    )
    _require_exact_indices(
        source_global,
        expected_count=GLOBAL_BLOCK_COUNT,
        name="World patch blocks",
    )
    _require_exact_indices(
        source_pixel,
        expected_count=PIXEL_BLOCK_COUNT,
        name="World pixel blocks",
    )
    _require_exact_indices(
        action_blocks,
        expected_count=ACTION_BLOCK_COUNT,
        name="Action blocks",
    )
    observed_non_block_keys = {
        key for key in action_state if not key.startswith(ACTION_BLOCK_PREFIX)
    }
    if observed_non_block_keys != ACTION_NON_BLOCK_KEYS:
        raise ValueError(
            "PixelDiT-aligned Action non-block key mismatch: "
            f"missing={sorted(ACTION_NON_BLOCK_KEYS - observed_non_block_keys)}, "
            f"unexpected={sorted(observed_non_block_keys - ACTION_NON_BLOCK_KEYS)}."
        )
    reference_action_suffixes = set(action_blocks[0])
    if not reference_action_suffixes:
        raise ValueError("Action block 0 must not be empty.")
    for block_index in range(1, ACTION_BLOCK_COUNT):
        observed_suffixes = set(action_blocks[block_index])
        if observed_suffixes != reference_action_suffixes:
            raise ValueError(
                f"Action block {block_index} key schema mismatch: "
                f"missing={sorted(reference_action_suffixes - observed_suffixes)}, "
                f"unexpected={sorted(observed_suffixes - reference_action_suffixes)}."
            )

    transfer_map: dict[str, str] = {}
    for block_index in range(GLOBAL_BLOCK_COUNT):
        source_suffixes = set(source_global[block_index])
        target_suffixes = set(action_blocks[block_index])
        if source_suffixes != target_suffixes:
            raise ValueError(
                f"Global block {block_index} key schema mismatch: "
                f"missing={sorted(target_suffixes - source_suffixes)}, "
                f"unexpected={sorted(source_suffixes - target_suffixes)}."
            )
        for suffix in sorted(target_suffixes):
            source_key = (
                f"{WORLD_PATCH_BLOCK_PREFIX}{block_index}.{suffix}"
            )
            target_key = f"{ACTION_BLOCK_PREFIX}{block_index}.{suffix}"
            _require_same_tensor_schema(
                source_state[source_key],
                action_state[target_key],
                source_key=source_key,
                target_key=target_key,
                validate_finite=audit_content,
            )
            transfer_map[target_key] = source_key

    for pixel_index in range(PIXEL_BLOCK_COUNT):
        source_attention_suffixes = {
            suffix
            for suffix in source_pixel[pixel_index]
            if suffix.startswith("attn.")
        }
        target_index = GLOBAL_BLOCK_COUNT + pixel_index
        target_attention_suffixes = {
            suffix
            for suffix in action_blocks[target_index]
            if suffix.startswith("attn.")
        }
        if not source_attention_suffixes:
            raise ValueError(f"World PiT block {pixel_index} has no attention tensors.")
        if source_attention_suffixes != target_attention_suffixes:
            missing_attention = sorted(
                target_attention_suffixes - source_attention_suffixes
            )
            unexpected_attention = sorted(
                source_attention_suffixes - target_attention_suffixes
            )
            raise ValueError(
                f"PiT-to-Action attention block {pixel_index} key schema mismatch: "
                f"missing={missing_attention}, "
                f"unexpected={unexpected_attention}."
            )
        for suffix in sorted(target_attention_suffixes):
            source_key = (
                f"{WORLD_PIXEL_BLOCK_PREFIX}{pixel_index}.{suffix}"
            )
            target_key = f"{ACTION_BLOCK_PREFIX}{target_index}.{suffix}"
            _require_same_tensor_schema(
                source_state[source_key],
                action_state[target_key],
                source_key=source_key,
                target_key=target_key,
                validate_finite=audit_content,
            )
            transfer_map[target_key] = source_key

    selected_source = {
        source_key: source_state[source_key]
        for source_key in sorted(set(transfer_map.values()))
    }
    source_schema = _state_schema(
        selected_source,
        name="selected_world_transfer_state",
        validate_finite=audit_content,
    )
    action_schema = _state_schema(
        action_state,
        name="action_random_base_state",
        validate_finite=audit_content,
    )

    key_receipts: list[dict[str, Any]] = []
    for target_key in sorted(action_state):
        target = _require_tensor(
            action_state[target_key],
            name=f"action_state[{target_key}]",
            validate_finite=audit_content,
        )
        candidate_source_key = transfer_map.get(target_key)
        source_key = (
            candidate_source_key
            if mode == ACTION14_WORLD_TRANSFER_INITIALIZATION
            else None
        )
        disposition = "retained_random_base"
        if source_key is not None:
            disposition = (
                "transferred_global_full"
                if target_key.startswith("blocks.")
                and int(target_key.split(".", 2)[1]) < GLOBAL_BLOCK_COUNT
                else "transferred_pit_attention"
            )
        key_receipt = {
            "target_key": target_key,
            "source_key": source_key,
            "candidate_source_key": candidate_source_key,
            "disposition": disposition,
            "target_shape": [int(size) for size in target.shape],
            "target_dtype": _dtype_name(target),
        }
        if audit_content:
            key_receipt.update(
                {
                    "random_base_sha256": tensor_content_sha256(
                    target,
                    name=f"action_random_base[{target_key}]",
                    ),
                    "source_sha256": (
                        None
                        if candidate_source_key is None
                        else tensor_content_sha256(
                            source_state[candidate_source_key],
                            name=(
                                "selected_world_state"
                                f"[{candidate_source_key}]"
                            ),
                        )
                    ),
                }
            )
        key_receipts.append(key_receipt)

    if mode == ACTION14_WORLD_TRANSFER_INITIALIZATION:
        with torch.no_grad():
            for target_key, source_key in sorted(transfer_map.items()):
                action_state[target_key].copy_(
                    source_state[source_key].to(
                        device=action_state[target_key].device,
                        dtype=action_state[target_key].dtype,
                    )
                )

    final_state = action_expert.state_dict()
    if set(final_state) != set(action_state):
        raise RuntimeError("Action state keys changed during initialization.")
    initialized_position = _joint_position_buffer(
        action_expert,
        final_state,
        validate_finite=audit_content,
    )
    initialized_position_hash = (
        tensor_content_sha256(
            initialized_position,
            name="action_initialized[joint_pos_embed]",
        )
        if audit_content
        else None
    )
    if audit_content and initialized_position_hash != random_position_hash:
        raise RuntimeError("Action joint-position embedding changed unexpectedly.")
    if audit_content:
        for item in key_receipts:
            target_key = item["target_key"]
            final_hash = tensor_content_sha256(
                final_state[target_key],
                name=f"action_initialized[{target_key}]",
            )
            item["initialized_sha256"] = final_hash
            if (
                mode == ACTION14_WORLD_TRANSFER_INITIALIZATION
                and item["source_key"]
            ):
                if final_hash != item["source_sha256"]:
                    raise RuntimeError(
                        "Transferred Action tensor hash mismatch for "
                        f"{target_key}."
                    )
            elif final_hash != item["random_base_sha256"]:
                raise RuntimeError(
                    "Random-base Action tensor changed unexpectedly: "
                    f"{target_key}."
                )

    transferred = [item for item in key_receipts if item["source_key"]]
    retained = [item for item in key_receipts if not item["source_key"]]
    random_base_digest = None
    initialized_digest = None
    random_complete_digest = None
    initialized_complete_digest = None
    if audit_content:
        random_base_digest = _state_content_sha256(
            key_receipts,
            hash_field="random_base_sha256",
        )
        initialized_digest = _state_content_sha256(
            key_receipts,
            hash_field="initialized_sha256",
        )
        random_complete_digest = canonical_json_contract(
            {
                "state_sha256": random_base_digest,
                "joint_pos_embed_sha256": random_position_hash,
            },
            domain="pixeldit-aligned-action-complete-state-v1",
            name="action_random_complete_state",
        )["sha256"]
        initialized_complete_digest = canonical_json_contract(
            {
                "state_sha256": initialized_digest,
                "joint_pos_embed_sha256": initialized_position_hash,
            },
            domain="pixeldit-aligned-action-complete-state-v1",
            name="action_initialized_complete_state",
        )["sha256"]
    receipt: dict[str, Any] = {
        "schema_version": ACTION14_INITIALIZATION_RECEIPT_SCHEMA_VERSION,
        "audit_content": audit_content,
        "audit_level": (
            "full_per_tensor_content_sha256_v1"
            if audit_content
            else "schema_only_no_tensor_content_scan_v1"
        ),
        "mode": mode,
        "transition_id": mode,
        "policy": {
            "random_base": ACTION14_RANDOM_BASE_POLICY,
            "world_transfer": ACTION14_WORLD_TRANSFER_POLICY,
            "global_blocks": [0, GLOBAL_BLOCK_COUNT - 1],
            "pit_blocks": [0, PIXEL_BLOCK_COUNT - 1],
            "pit_target_blocks": [GLOBAL_BLOCK_COUNT, ACTION_BLOCK_COUNT - 1],
        },
        "source": {
            "format": FROZEN_WORLD_SOURCE_FORMAT,
            "step": FROZEN_WORLD_SOURCE_STEP,
            "checkpoint_identity": identity,
            "training_contract_sha256": training_contract_receipt["sha256"],
            "selected_tensor_schema_sha256": source_schema["schema_sha256"],
            "selected_tensor_count": source_schema["tensor_count"],
            "selected_numel": source_schema["numel"],
        },
        "action": {
            "tensor_schema_sha256": action_schema["schema_sha256"],
            "tensor_count": action_schema["tensor_count"],
            "numel": action_schema["numel"],
            "random_base_state_sha256": random_base_digest,
            "initialized_state_sha256": initialized_digest,
            "random_base_complete_sha256": random_complete_digest,
            "initialized_complete_sha256": initialized_complete_digest,
            "joint_pos_embed": {
                "persistent": False,
                "shape": [int(size) for size in joint_position.shape],
                "dtype": _dtype_name(joint_position),
                "random_base_sha256": random_position_hash,
                "initialized_sha256": initialized_position_hash,
            },
            "transferred_tensor_count": len(transferred),
            "transferred_numel": sum(
                math.prod(item["target_shape"])
                for item in transferred
            ),
            "retained_random_tensor_count": len(retained),
            "retained_random_numel": sum(
                math.prod(item["target_shape"])
                for item in retained
            ),
        },
        "keys": key_receipts,
    }
    receipt_contract = canonical_json_contract(
        receipt,
        domain="pixeldit-aligned-action-initialization-receipt-v1",
        name="action_initialization_receipt",
    )
    receipt["receipt_sha256"] = receipt_contract["sha256"]
    return receipt


def validate_paired_action_initialization_receipts(
    random_receipt: Mapping[str, Any],
    world_transfer_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that a random/transfer pair shared one exact random Action base."""

    for name, receipt, expected_mode in (
        ("random_receipt", random_receipt, ACTION14_RANDOM_INITIALIZATION),
        (
            "world_transfer_receipt",
            world_transfer_receipt,
            ACTION14_WORLD_TRANSFER_INITIALIZATION,
        ),
    ):
        if not isinstance(receipt, Mapping):
            raise TypeError(f"`{name}` must be a mapping.")
        if receipt.get("schema_version") != (
            ACTION14_INITIALIZATION_RECEIPT_SCHEMA_VERSION
        ):
            raise ValueError(f"`{name}` has an unsupported schema version.")
        if receipt.get("mode") != expected_mode:
            raise ValueError(
                f"`{name}` mode mismatch: expected {expected_mode!r}, "
                f"got {receipt.get('mode')!r}."
            )
        if receipt.get("audit_content") is not True or receipt.get(
            "audit_level"
        ) != "full_per_tensor_content_sha256_v1":
            raise ValueError(
                f"`{name}` must be a full per-tensor content audit receipt."
            )

    random_action = random_receipt.get("action")
    transfer_action = world_transfer_receipt.get("action")
    random_source = random_receipt.get("source")
    transfer_source = world_transfer_receipt.get("source")
    if not all(
        isinstance(value, Mapping)
        for value in (
            random_action,
            transfer_action,
            random_source,
            transfer_source,
        )
    ):
        raise TypeError("Paired receipts must contain Action and source mappings.")
    assert isinstance(random_action, Mapping)
    assert isinstance(transfer_action, Mapping)
    assert isinstance(random_source, Mapping)
    assert isinstance(transfer_source, Mapping)

    compared_action_fields = (
        "tensor_schema_sha256",
        "tensor_count",
        "numel",
        "random_base_state_sha256",
        "random_base_complete_sha256",
        "joint_pos_embed",
    )
    action_mismatches = {
        field: (random_action.get(field), transfer_action.get(field))
        for field in compared_action_fields
        if random_action.get(field) != transfer_action.get(field)
    }
    if action_mismatches:
        raise ValueError(
            "Paired Action experiments do not share the same seeded random "
            f"base: {action_mismatches}."
        )

    compared_source_fields = (
        "format",
        "step",
        "checkpoint_identity",
        "training_contract_sha256",
        "selected_tensor_schema_sha256",
        "selected_tensor_count",
        "selected_numel",
    )
    source_mismatches = {
        field: (random_source.get(field), transfer_source.get(field))
        for field in compared_source_fields
        if random_source.get(field) != transfer_source.get(field)
    }
    if source_mismatches:
        raise ValueError(
            "Paired Action experiments do not use the same frozen World "
            f"source: {source_mismatches}."
        )
    if random_action.get("initialized_state_sha256") != random_action.get(
        "random_base_state_sha256"
    ):
        raise ValueError("Random branch did not preserve its complete Action base.")
    if random_action.get("initialized_complete_sha256") != random_action.get(
        "random_base_complete_sha256"
    ):
        raise ValueError(
            "Random branch did not preserve its non-persistent Action buffers."
        )
    if transfer_action.get("transferred_tensor_count", 0) <= 0:
        raise ValueError("World-transfer branch did not transfer any tensors.")

    comparison = {
        "schema_version": "pixeldit_aligned_action_paired_init_check_v1",
        "random_base_state_sha256": random_action[
            "random_base_state_sha256"
        ],
        "source_checkpoint_sha256": random_source["checkpoint_identity"][
            "sha256"
        ],
        "source_step": random_source["step"],
        "random_receipt_sha256": random_receipt.get("receipt_sha256"),
        "world_transfer_receipt_sha256": world_transfer_receipt.get(
            "receipt_sha256"
        ),
        "passed": True,
    }
    comparison["comparison_sha256"] = canonical_json_contract(
        comparison,
        domain="pixeldit-aligned-action-paired-init-check-v1",
        name="paired_action_initialization_check",
    )["sha256"]
    return comparison


def write_action_initialization_receipt(
    receipt: Mapping[str, Any],
    path: str | os.PathLike[str],
) -> Path:
    """Create one immutable JSON receipt, refusing an existing destination."""

    if not isinstance(receipt, Mapping):
        raise TypeError("`receipt` must be a mapping.")
    expected_digest = receipt.get("receipt_sha256")
    without_digest = dict(receipt)
    without_digest.pop("receipt_sha256", None)
    observed_digest = canonical_json_contract(
        without_digest,
        domain="pixeldit-aligned-action-initialization-receipt-v1",
        name="action_initialization_receipt",
    )["sha256"]
    if expected_digest != observed_digest:
        raise ValueError(
            "Action initialization receipt digest mismatch: "
            f"expected={expected_digest!r}, observed={observed_digest!r}."
        )
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(receipt),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return destination


__all__ = [
    "ACTION14_INITIALIZATION_MODES",
    "ACTION14_INITIALIZATION_RECEIPT_SCHEMA_VERSION",
    "ACTION14_RANDOM_INITIALIZATION",
    "ACTION14_WORLD_TRANSFER_INITIALIZATION",
    "ACTION14_WORLD_TRANSFER_POLICY",
    "FROZEN_WORLD_SOURCE_FORMAT",
    "FROZEN_WORLD_SOURCE_SHA256",
    "FROZEN_WORLD_SOURCE_SIZE_BYTES",
    "FROZEN_WORLD_SOURCE_STEP",
    "initialize_pixeldit_aligned_action",
    "tensor_content_sha256",
    "validate_paired_action_initialization_receipts",
    "write_action_initialization_receipt",
]
