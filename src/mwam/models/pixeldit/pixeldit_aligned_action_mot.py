"""Lockstep PixelDiT World to joint-trajectory Action execution.

This wrapper is a ``PixelDiTActionMoT`` subclass solely so the existing formal
FastWAM wrapper accepts it at its public type boundary. It intentionally does
not call the legacy constructor: the aligned Action blocks consume same-width
World native K/V directly and therefore own no bridge or K/V adapter weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.nn as nn

from .pixeldit_action_mot import PixelDiTActionMoT
from .pixeldit_aligned_action import (
    COSINE_ANNEAL_NATIVE_KV_FUSION_VERSION,
    LEGACY_NATIVE_KV_FUSION_VERSION,
    PixelDiTAlignedActionDiT,
    cosine_count_correction_scale,
)


class PixelDiTAlignedActionMoT(PixelDiTActionMoT):
    """One-way, 26-Global plus 4-PiT native-K/V lockstep wrapper."""

    TOPOLOGY_VERSION = "world_g26_p4_action30_linear14_aligned_native_kv_v8"
    BRIDGE_SEMANTICS_VERSION = (
        "world_global26_then_pit4_native_kv_direct_concat_action_self_kv_"
        "log14_over_valid_world_token_count_v1"
    )
    COSINE_ANNEAL_BRIDGE_SEMANTICS_VERSION = (
        "world_global26_then_pit4_native_kv_direct_concat_action_self_kv_"
        "log14_over_valid_world_token_count_cosine_anneal_to_none_v1"
    )
    NATIVE_KV_COUNT_CORRECTION_TRANSITION_SCHEMA_VERSION = (
        "fastwam_native_kv_count_correction_transition_v1"
    )
    NATIVE_KV_COUNT_CORRECTION_SCHEDULE_SCHEMA_VERSION = (
        "fastwam_native_kv_count_correction_schedule_v1"
    )
    COSINE_ANNEAL_SCHEDULE_MODE = "cosine_anneal_to_none_v1"
    COSINE_ANNEAL_TRANSITION_STEPS = 500
    COSINE_ANNEAL_TRANSITION_CURVE = "cosine"
    INTERACTION_MODE = "momap_to_action"
    ACTION30_TOPOLOGY_VERSIONS = (
        PixelDiTActionMoT.ACTION30_TOPOLOGY_VERSIONS | {TOPOLOGY_VERSION}
    )

    def __init__(
        self,
        world_expert: nn.Module,
        action_expert: PixelDiTAlignedActionDiT,
        *,
        topology_version: str = TOPOLOGY_VERSION,
        bridge_semantics_version: str = BRIDGE_SEMANTICS_VERSION,
        interaction_mode: str = INTERACTION_MODE,
        native_kv_fusion_schedule: Optional[Mapping[str, Any]] = None,
        use_gradient_checkpointing: bool = False,
        validate_finite: bool = False,
    ) -> None:
        # Calling PixelDiTActionMoT.__init__ would instantiate legacy adapters
        # and require WAN-style Action blocks. Only initialize nn.Module here.
        nn.Module.__init__(self)
        if topology_version != self.TOPOLOGY_VERSION:
            raise ValueError(
                "Aligned Action MoT requires topology "
                f"{self.TOPOLOGY_VERSION!r}, got {topology_version!r}."
            )
        if interaction_mode != self.INTERACTION_MODE:
            raise ValueError(
                "Aligned Action MoT always consumes World native K/V; expected "
                f"{self.INTERACTION_MODE!r}, got {interaction_mode!r}."
            )
        if not isinstance(action_expert, PixelDiTAlignedActionDiT):
            raise TypeError(
                "`action_expert` must be PixelDiTAlignedActionDiT, got "
                f"{type(action_expert).__name__}."
            )
        resolved_schedule = self._resolve_native_kv_fusion_schedule(
            native_kv_fusion_schedule
        )
        expected_bridge_semantics = (
            self.COSINE_ANNEAL_BRIDGE_SEMANTICS_VERSION
            if resolved_schedule["mode"] == self.COSINE_ANNEAL_SCHEDULE_MODE
            else self.BRIDGE_SEMANTICS_VERSION
        )
        if bridge_semantics_version != expected_bridge_semantics:
            raise ValueError(
                "Aligned Action MoT bridge semantics mismatch: expected "
                f"{expected_bridge_semantics!r}, got "
                f"{bridge_semantics_version!r}."
            )
        expected_action_fusion = resolved_schedule["target_native_kv_fusion"]
        if action_expert.native_kv_fusion_version != expected_action_fusion:
            raise ValueError(
                "Aligned Action native-K/V fusion mismatch: expected "
                f"{expected_action_fusion!r}, got "
                f"{action_expert.native_kv_fusion_version!r}."
            )

        self.world_expert = world_expert
        self.action_expert = action_expert
        self.validate_finite = bool(validate_finite)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self._topology_version = self.TOPOLOGY_VERSION
        self._bridge_semantics_version = expected_bridge_semantics
        self._interaction_mode = self.INTERACTION_MODE
        self._native_kv_fusion_schedule = resolved_schedule
        self._native_kv_fusion_transition: Optional[dict[str, Any]] = None
        self._attention_runtime_global_step: Optional[int] = None

        self.world_hidden_size = self._positive_int_attr(
            self.world_expert,
            "hidden_size",
        )
        self.action_hidden_size = self._positive_int_attr(
            self.action_expert,
            "hidden_size",
        )
        self.text_dim = self._positive_int_attr(self.action_expert, "text_dim")
        self.world_depth = self._resolve_world_depth(self.world_expert)
        self.pixel_depth = self._positive_int_attr(self.world_expert, "pixel_depth")
        self.action_depth = self._resolve_action_depth(self.action_expert)
        if self.world_depth != self.FORMAL_PATCH_DEPTH:
            raise ValueError(
                "Aligned Action MoT requires 26 World Global blocks, got "
                f"{self.world_depth}."
            )
        if self.pixel_depth != self.FORMAL_PIXEL_DEPTH:
            raise ValueError(
                "Aligned Action MoT requires four World PiT blocks, got "
                f"{self.pixel_depth}."
            )
        if self.action_depth != self.FORMAL_PATCH_DEPTH + self.FORMAL_PIXEL_DEPTH:
            raise ValueError(
                "Aligned Action MoT requires 30 Action blocks, got "
                f"{self.action_depth}."
            )
        if self.action_expert.action_horizon != 32:
            raise ValueError("Aligned Action MoT action horizon must be 32.")
        if self.action_expert.action_dim != 14:
            raise ValueError("Aligned Action MoT requires exactly 14 joint tokens.")
        if self.world_hidden_size != self.action_hidden_size:
            raise ValueError(
                "Direct native-K/V fusion requires equal World/Action hidden sizes, "
                f"got {self.world_hidden_size} and {self.action_hidden_size}."
            )

        required_world_methods = (
            "prepare_tokens",
            "forward_patch_block",
            "finish_patch_blocks",
            "prepare_pixel_tokens",
            "forward_pixel_block",
            "forward_pixel_modules",
            "final_head",
        )
        for name in required_world_methods:
            if not callable(getattr(self.world_expert, name, None)):
                raise TypeError(
                    f"{type(self.world_expert).__name__} must implement `{name}(...)`."
                )
        if not hasattr(self.world_expert, "pixel_blocks") or len(
            self.world_expert.pixel_blocks
        ) != self.FORMAL_PIXEL_DEPTH:
            raise TypeError("World expert must expose exactly four PiT blocks.")

        world_blocks = [
            *self.world_expert.patch_blocks,
            *self.world_expert.pixel_blocks,
        ]
        for layer_index, (world_block, action_block) in enumerate(
            zip(world_blocks, self.action_expert.blocks, strict=True)
        ):
            if not hasattr(world_block, "attn") or not hasattr(action_block, "attn"):
                raise TypeError(
                    f"World and Action layer {layer_index} must expose `attn`."
                )
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
                    "Direct native-K/V fusion requires identical head layouts at "
                    f"layer {layer_index}, got World {world_layout} and "
                    f"Action {action_layout}."
                )

        self.mot_dim = self.action_hidden_size
        self.mot_num_heads = self._positive_int_attr(
            self.action_expert,
            "num_heads",
        )
        self.world_text_projection = nn.Linear(
            self.text_dim,
            self.world_hidden_size,
            bias=True,
        )
        nn.init.xavier_uniform_(self.world_text_projection.weight)
        nn.init.zeros_(self.world_text_projection.bias)

        # These attributes remain explicit ModuleLists for state/contract
        # inspection. Direct same-width K/V fusion has no trainable adapters.
        self.world_to_action_adapters = nn.ModuleList()
        self.world_native_kv_adapters = nn.ModuleList()
        self.pit_native_kv_adapters = nn.ModuleList()

    @classmethod
    def _resolve_native_kv_fusion_schedule(
        cls,
        value: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Validate the static V9 schedule declaration without binding its start."""

        if value is None:
            return {
                "mode": "static_log_count_corrected_v1",
                "target_native_kv_fusion": LEGACY_NATIVE_KV_FUSION_VERSION,
            }
        if not isinstance(value, Mapping):
            raise TypeError("`native_kv_fusion_schedule` must be a mapping or None.")
        raw = dict(value)
        expected = {
            "schema_version": cls.NATIVE_KV_COUNT_CORRECTION_SCHEDULE_SCHEMA_VERSION,
            "mode": cls.COSINE_ANNEAL_SCHEDULE_MODE,
            "transition_steps": cls.COSINE_ANNEAL_TRANSITION_STEPS,
            "transition_curve": cls.COSINE_ANNEAL_TRANSITION_CURVE,
            "source_native_kv_fusion": LEGACY_NATIVE_KV_FUSION_VERSION,
            "target_native_kv_fusion": COSINE_ANNEAL_NATIVE_KV_FUSION_VERSION,
        }
        if raw != expected:
            missing = sorted(set(expected) - set(raw))
            unexpected = sorted(set(raw) - set(expected))
            mismatched = {
                key: {"expected": expected[key], "got": raw.get(key)}
                for key in sorted(set(expected) & set(raw))
                if raw[key] != expected[key]
            }
            raise ValueError(
                "native_kv_fusion_schedule must exactly match V9 semantics: "
                f"missing={missing}, unexpected={unexpected}, "
                f"mismatched={mismatched}."
            )
        return expected

    @staticmethod
    def _require_nonnegative_step(value: int, *, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"`{name}` must be an integer.")
        if value < 0:
            raise ValueError(f"`{name}` must be non-negative.")
        return int(value)

    @property
    def native_kv_fusion_version(self) -> str:
        return str(self.action_expert.native_kv_fusion_version)

    def configure_controlled_native_kv_count_correction_anneal(
        self,
        *,
        source_global_step: int,
        transition_steps: int,
        transition_curve: str,
        transition_id: str,
        source_native_kv_fusion: str,
        target_native_kv_fusion: str,
    ) -> dict[str, Any]:
        """Bind the one auditable V8-to-V9 count-correction transition."""

        if self._native_kv_fusion_schedule["mode"] != self.COSINE_ANNEAL_SCHEDULE_MODE:
            raise RuntimeError(
                "Controlled count-correction anneal requires the V9 cosine "
                "native_kv_fusion_schedule."
            )
        bound_step = self._require_nonnegative_step(
            source_global_step,
            name="source_global_step",
        )
        if transition_steps != self.COSINE_ANNEAL_TRANSITION_STEPS:
            raise ValueError(
                "V9 native-K/V transition requires exactly "
                f"{self.COSINE_ANNEAL_TRANSITION_STEPS} optimizer steps, got "
                f"{transition_steps}."
            )
        if transition_curve != self.COSINE_ANNEAL_TRANSITION_CURVE:
            raise ValueError(
                "V9 native-K/V transition requires curve "
                f"{self.COSINE_ANNEAL_TRANSITION_CURVE!r}, got "
                f"{transition_curve!r}."
            )
        if not isinstance(transition_id, str) or not transition_id:
            raise ValueError("`transition_id` must be a non-empty string.")
        if source_native_kv_fusion != LEGACY_NATIVE_KV_FUSION_VERSION:
            raise ValueError(
                "Controlled count-correction transition source fusion mismatch: "
                f"{source_native_kv_fusion!r}."
            )
        if target_native_kv_fusion != COSINE_ANNEAL_NATIVE_KV_FUSION_VERSION:
            raise ValueError(
                "Controlled count-correction transition target fusion mismatch: "
                f"{target_native_kv_fusion!r}."
            )
        if self.native_kv_fusion_version != target_native_kv_fusion:
            raise ValueError(
                "Action native-K/V fusion and transition target disagree: "
                f"{self.native_kv_fusion_version!r} vs "
                f"{target_native_kv_fusion!r}."
            )
        requested = {
            "schema_version": self.NATIVE_KV_COUNT_CORRECTION_TRANSITION_SCHEMA_VERSION,
            "transition_id": transition_id,
            "source_global_step": bound_step,
            "transition_steps": transition_steps,
            "transition_curve": transition_curve,
            "source_native_kv_fusion": source_native_kv_fusion,
            "target_native_kv_fusion": target_native_kv_fusion,
        }
        if (
            self._native_kv_fusion_transition is not None
            and self._native_kv_fusion_transition != requested
        ):
            raise RuntimeError(
                "Native-K/V count-correction transition was already bound to a "
                "different contract."
            )
        self._native_kv_fusion_transition = requested
        return dict(requested)

    def set_attention_runtime_step(self, global_step: Optional[int]) -> None:
        """Set the pre-forward successful-optimizer count on every rank.

        The trainer calls this immediately before model forward. Passing ``None``
        clears the runtime binding; evaluation then uses final FastWAM-style zero
        correction unless a checkpoint step is explicitly supplied.
        """

        if global_step is None:
            self._attention_runtime_global_step = None
            return
        self._attention_runtime_global_step = self._require_nonnegative_step(
            global_step,
            name="global_step",
        )


    def _world_logit_correction_scale(
        self,
        *,
        attention_runtime_global_step: Optional[int],
    ) -> float:
        if self._native_kv_fusion_schedule["mode"] != self.COSINE_ANNEAL_SCHEDULE_MODE:
            return 1.0
        transition = self._native_kv_fusion_transition
        runtime_step = (
            self._attention_runtime_global_step
            if attention_runtime_global_step is None
            else self._require_nonnegative_step(
                attention_runtime_global_step,
                name="attention_runtime_global_step",
            )
        )
        # Inference without a supplied checkpoint step is final direct concat.
        # Training must bind both contracts so microbatches cannot fade by error.
        if transition is None or runtime_step is None:
            if self.training:
                raise RuntimeError(
                    "V9 native-K/V cosine anneal requires a bound transition and "
                    "the current global optimizer step during training."
                )
            return 0.0
        return cosine_count_correction_scale(
            runtime_step,
            transition_start_global_step=int(transition["source_global_step"]),
            transition_optimizer_steps=int(transition["transition_steps"]),
        )

    @property
    def bridge_source_schedule(self) -> tuple[tuple[str, int], ...]:
        if self._native_kv_fusion_schedule["mode"] == self.COSINE_ANNEAL_SCHEDULE_MODE:
            global_kind = "global_native_kv_direct_count_correction_anneal_to_none"
            pixel_kind = "pit_native_kv_direct_count_correction_anneal_to_none"
        else:
            global_kind = "global_native_kv_direct_count_corrected"
            pixel_kind = "pit_native_kv_direct_count_corrected"
        global_sources = tuple(
            (global_kind, layer_index)
            for layer_index in range(self.FORMAL_PATCH_DEPTH)
        )
        pixel_sources = tuple(
            (pixel_kind, pixel_index)
            for pixel_index in range(self.FORMAL_PIXEL_DEPTH)
        )
        return global_sources + pixel_sources

    @staticmethod
    def _detached_native_kv(
        native_kv: Any,
        *,
        freeze_world: bool,
        source_name: str,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(native_kv, dict) or set(native_kv) != {"k", "v"}:
            raise TypeError(f"{source_name} native K/V must contain exactly k and v.")
        result: dict[str, torch.Tensor] = {}
        for name in ("k", "v"):
            value = native_kv[name]
            if not isinstance(value, torch.Tensor) or value.ndim != 4:
                raise ValueError(
                    f"{source_name} native {name.upper()} must be [B,N,H,D]."
                )
            result[name] = value.detach() if freeze_world else value
        return result

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
        if not isinstance(freeze_world, bool):
            raise TypeError("`freeze_world` must be a boolean.")
        if not isinstance(return_world_velocity, bool):
            raise TypeError("`return_world_velocity` must be a boolean.")
        expected_condition_channels = int(self.world_expert.clean_channels)
        if (
            world_condition.ndim != 4
            or world_condition.shape[1] != expected_condition_channels
        ):
            raise ValueError(
                "`world_condition` channel contract mismatch: expected "
                f"[B,{expected_condition_channels},H,W], got "
                f"{tuple(world_condition.shape)}."
            )
        if (
            noised_flow.ndim != 4
            or noised_flow.shape[0] != world_condition.shape[0]
            or noised_flow.shape[1] != int(self.world_expert.flow_channels)
            or noised_flow.shape[2:] != world_condition.shape[2:]
        ):
            raise ValueError(
                "`noised_flow` must be [B,3K,H,W] and match world_condition, got "
                f"{tuple(noised_flow.shape)} for {tuple(world_condition.shape)}."
            )
        if noised_action.ndim != 3:
            raise ValueError(
                f"`noised_action` must be [B,T,A], got {tuple(noised_action.shape)}."
            )
        batch_size = int(world_condition.shape[0])
        if noised_action.shape[0] != batch_size:
            raise ValueError("World/action batch sizes must match.")
        if t_world.ndim != 1 or t_world.shape[0] != batch_size:
            raise ValueError(
                f"`t_world` must be [B]={batch_size}, got {tuple(t_world.shape)}."
            )
        if t_action.ndim != 1 or t_action.shape[0] != batch_size:
            raise ValueError(
                f"`t_action` must be [B]={batch_size}, got {tuple(t_action.shape)}."
            )
        if world_context is None or world_context_mask is None:
            raise ValueError(
                "World language conditioning must be supplied separately from "
                "Action context so proprio cannot leak into PixelDiT."
            )
        world_logit_correction_scale = self._world_logit_correction_scale(
            attention_runtime_global_step=attention_runtime_global_step,
        )

        world_grad_context = torch.no_grad if freeze_world else nullcontext
        with world_grad_context():
            world_language = self._world_language_condition(
                world_context,
                world_context_mask,
            )
            world_state = self.world_expert.prepare_tokens(
                world_condition,
                noised_flow,
                t_world,
                source_support_mask=source_support_mask,
                extra_condition=world_language,
            )
        action_state = self.action_expert.pre_dit(
            action_tokens=noised_action,
            timestep=t_action,
            context=context,
            context_mask=context_mask,
        )
        source_patch_valid = world_state["patch_key_valid"]
        interaction_count = 0

        for layer_index in range(self.FORMAL_PATCH_DEPTH):
            with world_grad_context():
                world_result = self.world_expert.forward_patch_block(
                    layer_index,
                    world_state,
                    return_native_kv=True,
                )
            if not isinstance(world_result, tuple) or len(world_result) != 2:
                raise TypeError(
                    "Aligned Global lockstep requires (hidden, native_kv)."
                )
            _, native_kv_value = world_result
            native_kv = self._detached_native_kv(
                native_kv_value,
                freeze_world=freeze_world,
                source_name=f"Global layer {layer_index}",
            )
            self._require_finite(native_kv["k"], f"global_native_k_{layer_index}")
            self._require_finite(native_kv["v"], f"global_native_v_{layer_index}")
            self.action_expert.forward_block(
                layer_index,
                action_state,
                world_native_kv=native_kv,
                world_key_valid=source_patch_valid,
                world_logit_correction_scale=world_logit_correction_scale,
            )
            interaction_count += 1

        with world_grad_context():
            self.world_expert.finish_patch_blocks(world_state)
            self.world_expert.prepare_pixel_tokens(world_state)
        for pixel_index in range(self.FORMAL_PIXEL_DEPTH):
            with world_grad_context():
                pixel_result = self.world_expert.forward_pixel_block(
                    pixel_index,
                    world_state,
                    return_native_kv=True,
                )
            if not isinstance(pixel_result, tuple) or len(pixel_result) != 2:
                raise TypeError("Aligned PiT lockstep requires (compact, native_kv).")
            compact, native_kv_value = pixel_result
            self._require_finite(compact, f"pit_compact_{pixel_index}")
            native_kv = self._detached_native_kv(
                native_kv_value,
                freeze_world=freeze_world,
                source_name=f"PiT layer {pixel_index}",
            )
            self._require_finite(native_kv["k"], f"pit_native_k_{pixel_index}")
            self._require_finite(native_kv["v"], f"pit_native_v_{pixel_index}")
            self.action_expert.forward_block(
                self.FORMAL_PATCH_DEPTH + pixel_index,
                action_state,
                world_native_kv=native_kv,
                world_key_valid=source_patch_valid,
                world_logit_correction_scale=world_logit_correction_scale,
            )
            interaction_count += 1

        world_velocity = None
        if return_world_velocity:
            with world_grad_context():
                world_velocity = self.world_expert.final_head(world_state)
        action_velocity = self.action_expert.post_dit(
            action_state["tokens"],
            action_state,
        )
        if world_velocity is not None and world_velocity.shape != noised_flow.shape:
            raise ValueError(
                "PixelDiT velocity must match packed noised flow, got "
                f"{tuple(world_velocity.shape)} vs {tuple(noised_flow.shape)}."
            )
        if action_velocity.shape != noised_action.shape:
            raise ValueError(
                "Action velocity must match noised action, got "
                f"{tuple(action_velocity.shape)} vs {tuple(noised_action.shape)}."
            )
        if world_velocity is not None:
            self._require_finite(world_velocity, "world_velocity")
        self._require_finite(action_velocity, "action_velocity")
        return {
            "world_velocity": world_velocity,
            "action_velocity": action_velocity,
            "interaction_mode": self.interaction_mode,
            "interaction_count": interaction_count,
            "world_logit_correction_scale": world_logit_correction_scale,
        }


__all__ = ["PixelDiTAlignedActionMoT"]
