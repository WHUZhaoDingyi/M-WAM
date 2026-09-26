"""Lockstep Hybrid C/W World-to-Action execution.

The wrapper exposes two immutable experiments:

* ``action_condition_world``: Action attends to its own K/V plus C and W K/V.
* ``action_condition``: Action attends to its own K/V plus C K/V only.

Both variants use direct token concatenation.  The historical World-token
count correction and its 500-step anneal are deliberately absent.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.nn as nn

from .hybrid_cw_pixeldit import HybridCWPixelDiT
from .pixeldit_action_mot import PixelDiTActionMoT
from .pixeldit_aligned_action import (
    PixelDiTAlignedActionDiT,
    PixelDiTTemporalActionDiT,
)
from .pixeldit_aligned_action_mot import PixelDiTAlignedActionMoT


HYBRID_CW_KV_SOURCE_ACW = "action_condition_world"
HYBRID_CW_KV_SOURCE_AC = "action_condition"
HYBRID_CW_KV_SOURCES = frozenset(
    {HYBRID_CW_KV_SOURCE_ACW, HYBRID_CW_KV_SOURCE_AC}
)
HYBRID_CW_NATIVE_KV_FUSION_VERSION = "direct_concat_no_count_bias_v1"
HYBRID_CW_ACW_TOPOLOGY_VERSION = (
    "hybrid_cw_g26_p4_action30_joint14_action_condition_world_v1"
)
HYBRID_CW_AC_TOPOLOGY_VERSION = (
    "hybrid_cw_g26_p4_action30_joint14_action_condition_v1"
)
HYBRID_CW_AC_TEMPORAL32_TOPOLOGY_VERSION = (
    "hybrid_cw_g26_p4_action30_time32_action_condition_v1"
)


def initialize_action_from_hybrid_world(
    world_expert: HybridCWPixelDiT,
    action_expert: PixelDiTAlignedActionDiT,
) -> dict[str, Any]:
    """Copy the ImageNet-compatible World trunk into the aligned Action trunk.

    Global blocks 0..25 are copied in full.  Only native-attention tensors can
    be copied from PiT blocks 0..3 into Action blocks 26..29 because PiT local
    MLP/AdaLN widths differ from the 1152-wide Action blocks.  Action I/O,
    context projection, timestep embedder and the Action token I/O/position
    encoding stay at their Action-specific initialization.
    """

    if not isinstance(world_expert, HybridCWPixelDiT):
        raise TypeError("`world_expert` must be HybridCWPixelDiT.")
    if not isinstance(action_expert, PixelDiTAlignedActionDiT):
        raise TypeError("`action_expert` must be PixelDiTAlignedActionDiT.")
    if world_expert.patch_depth != 26 or world_expert.pixel_depth != 4:
        raise ValueError("Hybrid Action transfer requires World depth 26+4.")
    if len(action_expert.blocks) != 30:
        raise ValueError("Hybrid Action transfer requires exactly 30 Action blocks.")

    world_state = world_expert.state_dict()
    action_state = action_expert.state_dict()
    transfer_map: dict[str, str] = {}
    for layer_index in range(26):
        source_prefix = f"patch_blocks.{layer_index}."
        target_prefix = f"blocks.{layer_index}."
        source_suffixes = {
            key[len(source_prefix) :]
            for key in world_state
            if key.startswith(source_prefix)
        }
        target_suffixes = {
            key[len(target_prefix) :]
            for key in action_state
            if key.startswith(target_prefix)
        }
        if source_suffixes != target_suffixes:
            raise RuntimeError(
                f"Hybrid Global-to-Action schema mismatch at layer {layer_index}: "
                f"missing={sorted(target_suffixes - source_suffixes)}, "
                f"unexpected={sorted(source_suffixes - target_suffixes)}."
            )
        for suffix in sorted(target_suffixes):
            transfer_map[target_prefix + suffix] = source_prefix + suffix

    for pixel_index in range(4):
        source_prefix = f"pixel_blocks.{pixel_index}.attn."
        target_prefix = f"blocks.{26 + pixel_index}.attn."
        source_suffixes = {
            key[len(source_prefix) :]
            for key in world_state
            if key.startswith(source_prefix)
        }
        target_suffixes = {
            key[len(target_prefix) :]
            for key in action_state
            if key.startswith(target_prefix)
        }
        if not source_suffixes or source_suffixes != target_suffixes:
            raise RuntimeError(
                f"Hybrid PiT-Attention-to-Action schema mismatch at layer "
                f"{pixel_index}: missing={sorted(target_suffixes - source_suffixes)}, "
                f"unexpected={sorted(source_suffixes - target_suffixes)}."
            )
        for suffix in sorted(target_suffixes):
            transfer_map[target_prefix + suffix] = source_prefix + suffix

    shape_mismatches = {
        target_key: {
            "source": list(world_state[source_key].shape),
            "target": list(action_state[target_key].shape),
        }
        for target_key, source_key in transfer_map.items()
        if world_state[source_key].shape != action_state[target_key].shape
    }
    if shape_mismatches:
        raise RuntimeError(
            f"Hybrid World-to-Action tensor shape mismatch: {shape_mismatches}."
        )

    with torch.no_grad():
        for target_key, source_key in sorted(transfer_map.items()):
            action_state[target_key].copy_(
                world_state[source_key].to(
                    device=action_state[target_key].device,
                    dtype=action_state[target_key].dtype,
                )
            )

    global_keys = sorted(
        key
        for key in transfer_map
        if key.startswith("blocks.") and int(key.split(".")[1]) < 26
    )
    pit_attention_keys = sorted(set(transfer_map) - set(global_keys))
    transferred_numel = sum(action_state[key].numel() for key in transfer_map)
    return {
        "schema_version": "hybrid_cw_imagenet_to_action_initialization_v1",
        "policy": "global26_full_plus_pit4_attention_only_v1",
        "global_full_keys": global_keys,
        "pit_attention_keys": pit_attention_keys,
        "transferred_tensor_count": len(transfer_map),
        "transferred_numel": int(transferred_numel),
        "random_action_specific_prefixes": [
            "action_encoder.",
            "action_decoder.",
            "context_projection.",
            "t_embedder.",
            "blocks.26..29 non-attention tensors",
            "action position encoding (non-persistent buffer)",
        ],
    }


class HybridCWActionMoT(PixelDiTAlignedActionMoT):
    """One-way Hybrid World native-K/V to joint14 or temporal32 Action tokens."""

    INTERACTION_MODE = "momap_to_action"
    ACTION_DIM = 14
    TRAJECTORY_TOKENIZATION = "joint14"
    AC_TOPOLOGY_VERSION = HYBRID_CW_AC_TOPOLOGY_VERSION
    KV_SOURCES = HYBRID_CW_KV_SOURCES
    NATIVE_KV_FUSION_VERSION = HYBRID_CW_NATIVE_KV_FUSION_VERSION
    ACTION30_TOPOLOGY_VERSIONS = (
        PixelDiTActionMoT.ACTION30_TOPOLOGY_VERSIONS
        | {
            HYBRID_CW_ACW_TOPOLOGY_VERSION,
            HYBRID_CW_AC_TOPOLOGY_VERSION,
            HYBRID_CW_AC_TEMPORAL32_TOPOLOGY_VERSION,
        }
    )

    def __init__(
        self,
        world_expert: HybridCWPixelDiT,
        action_expert: PixelDiTAlignedActionDiT,
        *,
        kv_source: str,
        interaction_mode: str = INTERACTION_MODE,
        use_gradient_checkpointing: bool = False,
        validate_finite: bool = False,
    ) -> None:
        # Do not construct legacy adapters or bind the legacy count schedule.
        nn.Module.__init__(self)
        if not isinstance(world_expert, HybridCWPixelDiT):
            raise TypeError("`world_expert` must be HybridCWPixelDiT.")
        if not isinstance(action_expert, PixelDiTAlignedActionDiT):
            raise TypeError("`action_expert` must be PixelDiTAlignedActionDiT.")
        if kv_source not in self.KV_SOURCES:
            raise ValueError(
                f"`kv_source` must be one of {sorted(self.KV_SOURCES)}, got "
                f"{kv_source!r}."
            )
        if interaction_mode != self.INTERACTION_MODE:
            raise ValueError(
                "Hybrid C/W Action always uses one-way World-to-Action K/V."
            )

        self.world_expert = world_expert
        self.action_expert = action_expert
        self._kv_source = str(kv_source)
        self._action_tokenization = (
            "temporal32"
            if isinstance(action_expert, PixelDiTTemporalActionDiT)
            else self.TRAJECTORY_TOKENIZATION
        )
        if (
            self._action_tokenization == "temporal32"
            and self.kv_source != HYBRID_CW_KV_SOURCE_AC
        ):
            raise ValueError(
                "Temporal32 Hybrid Action currently supports A+C K/V only."
            )
        self._action_world_visibility = (
            "c_and_w"
            if self.kv_source == HYBRID_CW_KV_SOURCE_ACW
            else "c_only"
        )
        # The Action block has one implementation; this field is checkpoint
        # metadata.  Hybrid execution always invokes it with correction scale 0.
        self.action_expert.native_kv_fusion_version = (
            self.NATIVE_KV_FUSION_VERSION
        )
        self.validate_finite = bool(validate_finite)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self._interaction_mode = self.INTERACTION_MODE
        self._topology_version = (
            HYBRID_CW_AC_TEMPORAL32_TOPOLOGY_VERSION
            if self._action_tokenization == "temporal32"
            else (
                HYBRID_CW_ACW_TOPOLOGY_VERSION
                if self.kv_source == HYBRID_CW_KV_SOURCE_ACW
                else self.AC_TOPOLOGY_VERSION
            )
        )
        self._bridge_semantics_version = (
            "action_q_reads_action_self_plus_"
            f"{self.kv_source}_native_kv_direct_concat_no_count_bias_v1"
        )
        self._native_kv_fusion_schedule = {
            "mode": "static_direct_concat_no_count_bias_v1",
            "target_native_kv_fusion": self.NATIVE_KV_FUSION_VERSION,
        }
        self._native_kv_fusion_transition = None
        self._attention_runtime_global_step: Optional[int] = None

        self.world_hidden_size = self._positive_int_attr(world_expert, "hidden_size")
        self.action_hidden_size = self._positive_int_attr(action_expert, "hidden_size")
        self.text_dim = self._positive_int_attr(action_expert, "text_dim")
        self.world_depth = self._resolve_world_depth(world_expert)
        self.pixel_depth = self._positive_int_attr(world_expert, "pixel_depth")
        self.action_depth = self._resolve_action_depth(action_expert)
        if (self.world_depth, self.pixel_depth, self.action_depth) != (26, 4, 30):
            raise ValueError(
                "Hybrid lockstep requires World 26+4 and Action 30 blocks, got "
                f"{self.world_depth}+{self.pixel_depth} and {self.action_depth}."
            )
        if action_expert.action_horizon != 32 or action_expert.action_dim != self.ACTION_DIM:
            raise ValueError(f"Hybrid Action requires horizon 32 and {self.ACTION_DIM}-D controls.")
        if self.world_hidden_size != self.action_hidden_size:
            raise ValueError("Hybrid direct K/V requires equal World/Action widths.")

        world_blocks = [*world_expert.patch_blocks, *world_expert.pixel_blocks]
        for layer_index, (world_block, action_block) in enumerate(
            zip(world_blocks, action_expert.blocks, strict=True)
        ):
            world_layout = (
                self._positive_int_attr(world_block.attn, "num_heads"),
                self._positive_int_attr(world_block.attn, "head_dim"),
            )
            action_layout = (
                self._positive_int_attr(action_block.attn, "num_heads"),
                self._positive_int_attr(action_block.attn, "head_dim"),
            )
            if world_layout != action_layout:
                raise ValueError(
                    f"Hybrid layer {layer_index} World/Action head layouts differ: "
                    f"{world_layout} vs {action_layout}."
                )

        self.mot_dim = self.action_hidden_size
        self.mot_num_heads = self._positive_int_attr(action_expert, "num_heads")
        self.world_text_projection = nn.Linear(
            self.text_dim, self.world_hidden_size, bias=True
        )
        nn.init.xavier_uniform_(self.world_text_projection.weight)
        nn.init.zeros_(self.world_text_projection.bias)
        self.world_to_action_adapters = nn.ModuleList()
        self.world_native_kv_adapters = nn.ModuleList()
        self.pit_native_kv_adapters = nn.ModuleList()

    @property
    def native_kv_fusion_version(self) -> str:
        return self.NATIVE_KV_FUSION_VERSION

    @property
    def kv_source(self) -> str:
        return self._kv_source

    @property
    def action_world_visibility(self) -> str:
        return self._action_world_visibility

    @property
    def action_tokenization(self) -> str:
        return self._action_tokenization

    @property
    def bridge_source_schedule(self) -> tuple[tuple[str, int], ...]:
        stem = (
            "condition_world"
            if self.kv_source == HYBRID_CW_KV_SOURCE_ACW
            else "condition"
        )
        global_sources = tuple(
            (f"global_{stem}_native_kv_direct_no_bias", index)
            for index in range(self.FORMAL_PATCH_DEPTH)
        )
        pixel_sources = tuple(
            (f"pit_{stem}_native_kv_direct_no_bias", index)
            for index in range(self.FORMAL_PIXEL_DEPTH)
        )
        return global_sources + pixel_sources

    def set_attention_runtime_step(self, global_step: Optional[int]) -> None:
        """Accept trainer bookkeeping; it cannot change Hybrid attention math."""

        if global_step is not None:
            self._require_nonnegative_step(global_step, name="global_step")
        self._attention_runtime_global_step = global_step

    def native_kv_fusion_transition_contract(
        self, *, require_bound: bool = False
    ) -> None:
        del require_bound
        return None

    @staticmethod
    def _validate_stream_kv(
        value: Any, *, source_name: str
    ) -> dict[str, torch.Tensor]:
        if not isinstance(value, Mapping) or set(value) != {"k", "v"}:
            raise TypeError(f"{source_name} K/V must contain exactly k and v.")
        result = {"k": value["k"], "v": value["v"]}
        if any(
            not isinstance(tensor, torch.Tensor) or tensor.ndim != 4
            for tensor in result.values()
        ):
            raise ValueError(f"{source_name} K/V must be [B,N,heads,head_dim].")
        if result["k"].shape != result["v"].shape:
            raise ValueError(f"{source_name} K/V shapes must match.")
        return result

    def _select_native_kv(
        self,
        native_kv: Any,
        *,
        freeze_world: bool,
        source_name: str,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(native_kv, Mapping) or set(native_kv) != {"c", "w"}:
            raise TypeError(f"{source_name} must expose exactly C and W K/V.")
        c = self._validate_stream_kv(native_kv["c"], source_name=f"{source_name} C")
        w = self._validate_stream_kv(native_kv["w"], source_name=f"{source_name} W")
        if c["k"].shape != w["k"].shape:
            raise ValueError(f"{source_name} C/W K/V shapes must match.")
        if self.kv_source == HYBRID_CW_KV_SOURCE_ACW:
            selected = {
                name: torch.cat([c[name], w[name]], dim=1)
                for name in ("k", "v")
            }
        else:
            selected = c
        if freeze_world:
            selected = {name: tensor.detach() for name, tensor in selected.items()}
        return selected

    def forward_world_only(
        self,
        world_condition: torch.Tensor,
        noised_flow: torch.Tensor,
        t_world: torch.Tensor,
        world_context: torch.Tensor,
        world_context_mask: torch.Tensor,
        *,
        source_support_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        world_language = self._world_language_condition(
            world_context, world_context_mask
        )
        velocity = self.world_expert(
            world_condition,
            noised_flow,
            t_world,
            source_support_mask=source_support_mask,
            extra_condition=world_language,
        )
        if velocity.shape != noised_flow.shape:
            raise ValueError("Hybrid World velocity must match its 12-channel carrier.")
        self._require_finite(velocity, "world_velocity")
        return velocity

    def forward(
        self,
        world_condition: torch.Tensor,
        noised_flow: torch.Tensor,
        t_world: torch.Tensor,
        noised_action: torch.Tensor,
        t_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        *,
        source_support_mask: Optional[torch.Tensor] = None,
        world_context: Optional[torch.Tensor] = None,
        world_context_mask: Optional[torch.Tensor] = None,
        freeze_world: bool = False,
        return_world_velocity: bool = True,
        attention_runtime_global_step: Optional[int] = None,
    ) -> dict[str, Any]:
        if not isinstance(freeze_world, bool) or not isinstance(
            return_world_velocity, bool
        ):
            raise TypeError("Hybrid freeze/return flags must be boolean.")
        batch_size = int(world_condition.shape[0]) if world_condition.ndim == 4 else -1
        expected_condition = (batch_size, self.world_expert.clean_channels)
        if (
            world_condition.ndim != 4
            or world_condition.shape[1] != self.world_expert.clean_channels
        ):
            raise ValueError(
                f"Hybrid C must start [B,{expected_condition[1]}], got "
                f"{tuple(world_condition.shape)}."
            )
        if (
            noised_flow.ndim != 4
            or noised_flow.shape[0] != batch_size
            or noised_flow.shape[1] != self.world_expert.flow_channels
            or noised_flow.shape[2:] != world_condition.shape[2:]
        ):
            raise ValueError("Hybrid W must be [B,12,H,W] aligned with C.")
        if noised_action.shape != (
            batch_size,
            self.action_expert.action_horizon,
            self.action_expert.action_dim,
        ):
            raise ValueError("Hybrid Action must be [B,32,14].")
        for name, timestep in (("t_world", t_world), ("t_action", t_action)):
            if timestep.ndim != 1 or timestep.shape[0] != batch_size:
                raise ValueError(f"`{name}` must be [B].")
        if world_context is None or world_context_mask is None:
            raise ValueError(
                "World text must be separate so Action proprio cannot leak into C/W."
            )
        self.set_attention_runtime_step(attention_runtime_global_step)

        world_grad_context = torch.no_grad if freeze_world else nullcontext
        with world_grad_context():
            world_language = self._world_language_condition(
                world_context, world_context_mask
            )
            world_state = self.world_expert.prepare_tokens(
                world_condition,
                noised_flow,
                t_world,
                source_support_mask=source_support_mask,
                extra_condition=world_language,
            )
        action_state = self.action_expert.pre_dit(
            noised_action, t_action, context, context_mask
        )
        interaction_count = 0
        selected_world_tokens = None

        for layer_index in range(26):
            with world_grad_context():
                _, _, streams = self.world_expert.forward_patch_block(
                    layer_index, world_state, return_native_kv=True
                )
            native_kv = self._select_native_kv(
                streams,
                freeze_world=freeze_world,
                source_name=f"Global layer {layer_index}",
            )
            selected_world_tokens = int(native_kv["k"].shape[1])
            self._require_finite(native_kv["k"], f"global_native_k_{layer_index}")
            self._require_finite(native_kv["v"], f"global_native_v_{layer_index}")
            self.action_expert.forward_block(
                layer_index,
                action_state,
                world_native_kv=native_kv,
                world_key_valid=None,
                world_logit_correction_scale=0.0,
            )
            interaction_count += 1

        with world_grad_context():
            self.world_expert.finish_patch_blocks(world_state)
            self.world_expert.prepare_pixel_tokens(world_state)
        for pixel_index in range(4):
            with world_grad_context():
                _, _, streams = self.world_expert.forward_pixel_block(
                    pixel_index, world_state, return_native_kv=True
                )
            native_kv = self._select_native_kv(
                streams,
                freeze_world=freeze_world,
                source_name=f"PiT layer {pixel_index}",
            )
            selected_world_tokens = int(native_kv["k"].shape[1])
            self._require_finite(native_kv["k"], f"pit_native_k_{pixel_index}")
            self._require_finite(native_kv["v"], f"pit_native_v_{pixel_index}")
            self.action_expert.forward_block(
                26 + pixel_index,
                action_state,
                world_native_kv=native_kv,
                world_key_valid=None,
                world_logit_correction_scale=0.0,
            )
            interaction_count += 1

        world_velocity = None
        if return_world_velocity:
            with world_grad_context():
                world_velocity = self.world_expert.final_head(world_state)
        action_velocity = self.action_expert.post_dit(
            action_state["tokens"], action_state
        )
        if world_velocity is not None and world_velocity.shape != noised_flow.shape:
            raise RuntimeError("Hybrid World output shape drifted.")
        if action_velocity.shape != noised_action.shape:
            raise RuntimeError("Hybrid Action output shape drifted.")
        if world_velocity is not None:
            self._require_finite(world_velocity, "world_velocity")
        self._require_finite(action_velocity, "action_velocity")
        return {
            "world_velocity": world_velocity,
            "action_velocity": action_velocity,
            "interaction_mode": self.interaction_mode,
            "interaction_count": interaction_count,
            "kv_source": self.kv_source,
            "selected_world_token_count": selected_world_tokens,
            "native_kv_fusion_version": self.native_kv_fusion_version,
            "world_logit_correction_scale": 0.0,
        }


__all__ = [
    "HYBRID_CW_AC_TOPOLOGY_VERSION",
    "HYBRID_CW_AC_TEMPORAL32_TOPOLOGY_VERSION",
    "HYBRID_CW_ACW_TOPOLOGY_VERSION",
    "HYBRID_CW_KV_SOURCE_AC",
    "HYBRID_CW_KV_SOURCE_ACW",
    "HYBRID_CW_KV_SOURCES",
    "HYBRID_CW_NATIVE_KV_FUSION_VERSION",
    "HybridCWActionMoT",
    "initialize_action_from_hybrid_world",
]
