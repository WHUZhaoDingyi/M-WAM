from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..wan22.wan_video_dit import flash_attention, modulate, rope_apply


class WorldToActionAdapter(nn.Module):
    """One independent MoMap-hidden to Action-hidden cross-attention bridge."""

    def __init__(
        self,
        *,
        world_hidden_size: int,
        action_hidden_size: int,
        mot_dim: int,
        mot_num_heads: int,
        norm_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.world_hidden_size = int(world_hidden_size)
        self.action_hidden_size = int(action_hidden_size)
        self.mot_dim = int(mot_dim)
        self.mot_num_heads = int(mot_num_heads)
        self.head_dim = (
            self.mot_dim // self.mot_num_heads if self.mot_num_heads > 0 else 0
        )
        self.norm_eps = float(norm_eps)
        if self.world_hidden_size <= 0 or self.action_hidden_size <= 0:
            raise ValueError("World/action bridge hidden sizes must be positive.")
        if self.mot_dim <= 0 or self.mot_num_heads <= 0:
            raise ValueError("`mot_dim` and `mot_num_heads` must be positive.")
        if self.mot_dim % self.mot_num_heads:
            raise ValueError(
                f"`mot_dim`={self.mot_dim} must be divisible by "
                f"`mot_num_heads`={self.mot_num_heads}."
            )
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError(
                f"Bridge norm epsilon must be finite and positive, got {norm_eps}."
            )

        self.world_norm = nn.LayerNorm(
            self.world_hidden_size,
            eps=self.norm_eps,
        )
        self.action_norm = nn.LayerNorm(
            self.action_hidden_size,
            eps=self.norm_eps,
        )
        self.q_proj = nn.Linear(self.action_hidden_size, self.mot_dim, bias=False)
        self.k_proj = nn.Linear(self.world_hidden_size, self.mot_dim, bias=False)
        self.v_proj = nn.Linear(self.world_hidden_size, self.mot_dim, bias=False)
        self.out_proj = nn.Linear(self.mot_dim, self.action_hidden_size, bias=False)
        # The exact zero makes momap_to_action initialization identical to the
        # no-interaction ActionDiT while still giving this gate a first-step grad.
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        action_hidden: torch.Tensor,
        world_hidden: torch.Tensor,
        source_patch_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            action_hidden.ndim != 3
            or action_hidden.shape[-1] != self.action_hidden_size
        ):
            raise ValueError(
                "`action_hidden` must be [B,T,action_hidden_size], got "
                f"{tuple(action_hidden.shape)}."
            )
        if world_hidden.ndim != 3 or world_hidden.shape[-1] != self.world_hidden_size:
            raise ValueError(
                "`world_hidden` must be [B,N,world_hidden_size], got "
                f"{tuple(world_hidden.shape)}."
            )
        if action_hidden.shape[0] != world_hidden.shape[0]:
            raise ValueError("World/action bridge batch sizes must match.")
        if action_hidden.device != world_hidden.device:
            raise ValueError("World/action bridge tensors must share a device.")

        batch_size, action_length, _ = action_hidden.shape
        world_length = int(world_hidden.shape[1])
        q = self.q_proj(self.action_norm(action_hidden))
        k = self.k_proj(self.world_norm(world_hidden))
        v = self.v_proj(self.world_norm(world_hidden))
        q = q.view(
            batch_size,
            action_length,
            self.mot_num_heads,
            self.head_dim,
        ).transpose(1, 2)
        k = k.view(
            batch_size,
            world_length,
            self.mot_num_heads,
            self.head_dim,
        ).transpose(1, 2)
        v = v.view(
            batch_size,
            world_length,
            self.mot_num_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention_mask = None
        if source_patch_valid is not None:
            if source_patch_valid.shape != (batch_size, world_length):
                raise ValueError(
                    "`source_patch_valid` must be [B,N], got "
                    f"{tuple(source_patch_valid.shape)}."
                )
            if source_patch_valid.dtype != torch.bool:
                raise TypeError("`source_patch_valid` must use torch.bool.")
            if not bool(source_patch_valid.any(dim=1).all().item()):
                raise ValueError(
                    "Every sample must expose at least one source-supported patch."
                )
            attention_mask = source_patch_valid.to(device=action_hidden.device)[
                :, None, None, :
            ]

        mixed = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
        )
        mixed = mixed.transpose(1, 2).reshape(
            batch_size,
            action_length,
            self.mot_dim,
        )
        update = self.out_proj(mixed).to(dtype=action_hidden.dtype)
        gate = self.residual_gate.to(dtype=action_hidden.dtype)
        return action_hidden + gate * update


class WorldNativeKVAdapter(nn.Module):
    """Adapt one Global/PiT native K/V bank to Action attention width."""

    def __init__(
        self,
        *,
        world_attn_dim: int,
        action_attn_dim: int,
    ) -> None:
        super().__init__()
        self.world_attn_dim = int(world_attn_dim)
        self.action_attn_dim = int(action_attn_dim)
        if self.world_attn_dim <= 0 or self.action_attn_dim <= 0:
            raise ValueError("World/Action attention dimensions must be positive.")
        self.k_proj = nn.Linear(
            self.world_attn_dim,
            self.action_attn_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.world_attn_dim,
            self.action_attn_dim,
            bias=False,
        )

    def forward(
        self,
        native_k: torch.Tensor,
        native_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if native_k.ndim != 4 or native_v.ndim != 4:
            raise ValueError("World native K/V must be [B,N,heads,head_dim].")
        if native_k.shape != native_v.shape:
            raise ValueError("World native K/V shapes must match.")
        if native_k.device != native_v.device or native_k.dtype != native_v.dtype:
            raise ValueError("World native K/V must share dtype/device.")
        if int(native_k.shape[-2] * native_k.shape[-1]) != self.world_attn_dim:
            raise ValueError(
                "World native K/V flattened width mismatch: "
                f"{tuple(native_k.shape)} vs {self.world_attn_dim}."
            )
        batch_size, sequence_length = native_k.shape[:2]
        flat_k = native_k.reshape(batch_size, sequence_length, self.world_attn_dim)
        flat_v = native_v.reshape(batch_size, sequence_length, self.world_attn_dim)
        return self.k_proj(flat_k), self.v_proj(flat_v)



class PixelDiTActionMoT(nn.Module):
    """Versioned one-way MoMap-to-Action lockstep architectures."""

    FORMAL_PATCH_DEPTH = 26
    FORMAL_PIXEL_DEPTH = 4
    LEGACY_TOPOLOGY_VERSION = "world_g26_p4_action26_global_bridge_v1"
    ACTION30_TOPOLOGY_VERSION = "world_g26_p4_action30_global_pixel_bridge_v1"
    ACTION30_PIT_ONLY_TOPOLOGY_VERSION = (
        "world_g26_p4_action30_global_then_pixel_kv_bridge_v2"
    )
    ACTION30_PIT_NATIVE_KV_TOPOLOGY_VERSION = (
        "world_g26_p4_action30_global_then_pit_native_kv_mot_v3"
    )
    ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION = (
        "world_g26_p4_action30_all_world_native_kv_mot_v4"
    )
    ACTION26_GLOBAL_NATIVE_KV_TOPOLOGY_VERSION = (
        "world_g26_p4_action26_global_native_kv_mot_v5"
    )
    INTERACTION_MODES = frozenset({"none", "momap_to_action"})
    BRIDGE_SEMANTICS_VERSION = (
        "world_patch_then_action_self_attn_residual_then_"
        "momap_to_action_bridge_then_action_text_cross_attn_then_action_ffn_v1"
    )
    ACTION30_BRIDGE_SEMANTICS_VERSION = (
        "global_0_25_then_final_global_concat_pit_compact_0_3_"
        "to_action_26_29_v1"
    )
    ACTION30_PIT_ONLY_BRIDGE_SEMANTICS_VERSION = (
        "global_0_25_then_corresponding_pit_compact_0_3_"
        "to_action_26_29_kv_v2"
    )
    ACTION30_PIT_NATIVE_KV_BRIDGE_SEMANTICS_VERSION = (
        "global_0_25_post_action_self_bridge_then_pit_native_kv_0_3_"
        "concat_action_self_kv_to_action_26_29_mixed_attention_v3"
    )
    ACTION30_ALL_NATIVE_KV_BRIDGE_SEMANTICS_VERSION = (
        "global_native_kv_0_25_then_pit_native_kv_0_3_"
        "concat_action_self_kv_to_action_0_29_mixed_attention_v4"
    )
    ACTION26_GLOBAL_NATIVE_KV_BRIDGE_SEMANTICS_VERSION = (
        "global_native_kv_0_25_concat_action_self_kv_to_action_0_25_"
        "mixed_attention_pit_world_only_v5"
    )
    ACTION30_TOPOLOGY_VERSIONS = frozenset(
        {
            ACTION30_TOPOLOGY_VERSION,
            ACTION30_PIT_ONLY_TOPOLOGY_VERSION,
            ACTION30_PIT_NATIVE_KV_TOPOLOGY_VERSION,
            ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION,
        }
    )
    _TOPOLOGY_SPECS = {
        LEGACY_TOPOLOGY_VERSION: {
            "action_depth": FORMAL_PATCH_DEPTH,
            "bridge_semantics_version": BRIDGE_SEMANTICS_VERSION,
        },
        ACTION30_TOPOLOGY_VERSION: {
            "action_depth": FORMAL_PATCH_DEPTH + FORMAL_PIXEL_DEPTH,
            "bridge_semantics_version": ACTION30_BRIDGE_SEMANTICS_VERSION,
        },
        ACTION30_PIT_ONLY_TOPOLOGY_VERSION: {
            "action_depth": FORMAL_PATCH_DEPTH + FORMAL_PIXEL_DEPTH,
            "bridge_semantics_version": (
                ACTION30_PIT_ONLY_BRIDGE_SEMANTICS_VERSION
            ),
        },
        ACTION30_PIT_NATIVE_KV_TOPOLOGY_VERSION: {
            "action_depth": FORMAL_PATCH_DEPTH + FORMAL_PIXEL_DEPTH,
            "bridge_semantics_version": (
                ACTION30_PIT_NATIVE_KV_BRIDGE_SEMANTICS_VERSION
            ),
        },
        ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION: {
            "action_depth": FORMAL_PATCH_DEPTH + FORMAL_PIXEL_DEPTH,
            "bridge_semantics_version": (
                ACTION30_ALL_NATIVE_KV_BRIDGE_SEMANTICS_VERSION
            ),
        },
        ACTION26_GLOBAL_NATIVE_KV_TOPOLOGY_VERSION: {
            "action_depth": FORMAL_PATCH_DEPTH,
            "bridge_semantics_version": (
                ACTION26_GLOBAL_NATIVE_KV_BRIDGE_SEMANTICS_VERSION
            ),
        },
    }

    def __init__(
        self,
        world_expert: nn.Module,
        action_expert: nn.Module,
        *,
        mot_dim: Optional[int] = None,
        mot_num_heads: Optional[int] = None,
        k_norm_eps: float = 1.0e-6,
        interaction_mode: str = "momap_to_action",
        topology_version: str = LEGACY_TOPOLOGY_VERSION,
        bridge_semantics_version: str = BRIDGE_SEMANTICS_VERSION,
        use_gradient_checkpointing: bool = False,
        validate_finite: bool = False,
    ) -> None:
        super().__init__()
        self.world_expert = world_expert
        self.action_expert = action_expert
        self.validate_finite = bool(validate_finite)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        if topology_version not in self._TOPOLOGY_SPECS:
            raise ValueError(
                "Unsupported `topology_version`: expected one of "
                f"{sorted(self._TOPOLOGY_SPECS)}, got {topology_version!r}."
            )
        self._topology_version = str(topology_version)
        topology_spec = self._TOPOLOGY_SPECS[self._topology_version]

        self.world_hidden_size = self._positive_int_attr(
            world_expert,
            "hidden_size",
        )
        self.action_hidden_size = self._positive_int_attr(
            action_expert,
            "hidden_dim",
        )
        self.text_dim = self._positive_int_attr(action_expert, "text_dim")
        self.world_text_projection = nn.Linear(
            self.text_dim, self.world_hidden_size, bias=True
        )
        nn.init.xavier_uniform_(self.world_text_projection.weight)
        nn.init.zeros_(self.world_text_projection.bias)
        self.world_depth = self._resolve_world_depth(world_expert)
        self.pixel_depth = self._positive_int_attr(world_expert, "pixel_depth")
        self.action_depth = self._resolve_action_depth(action_expert)
        if self.world_depth != self.FORMAL_PATCH_DEPTH:
            raise ValueError(
                "Formal MoMap-WAM requires exactly 26 PixelDiT patch blocks, "
                f"got {self.world_depth}."
            )
        expected_action_depth = int(topology_spec["action_depth"])
        if self.action_depth != expected_action_depth:
            raise ValueError(
                f"Topology {self._topology_version!r} requires exactly "
                f"{expected_action_depth} ActionDiT blocks, got "
                f"{self.action_depth}."
            )
        if self.pixel_depth != self.FORMAL_PIXEL_DEPTH:
            raise ValueError(
                "Formal MoMap-WAM keeps exactly four PixelDiT pixel modules "
                f"outside MoT, got {self.pixel_depth}."
            )
        required_world_methods = [
            "prepare_tokens",
            "forward_patch_block",
            "finish_patch_blocks",
            "forward_pixel_modules",
            "final_head",
        ]
        if self._topology_version in self.ACTION30_TOPOLOGY_VERSIONS:
            required_world_methods.extend(
                ["prepare_pixel_tokens", "forward_pixel_block"]
            )
        for name in required_world_methods:
            if not callable(getattr(world_expert, name, None)):
                raise TypeError(
                    f"{type(world_expert).__name__} must implement `{name}(...)`."
                )

        if interaction_mode not in self.INTERACTION_MODES:
            raise ValueError(
                f"`interaction_mode` must be one of "
                f"{sorted(self.INTERACTION_MODES)}, got {interaction_mode!r}."
            )
        expected_bridge_semantics = str(
            topology_spec["bridge_semantics_version"]
        )
        if bridge_semantics_version != expected_bridge_semantics:
            raise ValueError(
                "Unsupported `bridge_semantics_version` for topology "
                f"{self._topology_version!r}: expected "
                f"{expected_bridge_semantics!r}, got "
                f"{bridge_semantics_version!r}."
            )
        self._bridge_semantics_version = expected_bridge_semantics
        self.interaction_mode = interaction_mode
        resolved_mot_dim = self.action_hidden_size if mot_dim is None else int(mot_dim)
        if mot_num_heads is None:
            action_heads = self._positive_int_attr(action_expert, "num_heads")
            resolved_mot_heads = math.gcd(resolved_mot_dim, action_heads)
        else:
            resolved_mot_heads = int(mot_num_heads)
        self.mot_dim = resolved_mot_dim
        self.mot_num_heads = resolved_mot_heads

        uses_global_native_kv = self._topology_version in {
            self.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION,
            self.ACTION26_GLOBAL_NATIVE_KV_TOPOLOGY_VERSION,
        }
        uses_pixel_native_kv = (
            self._topology_version
            == self.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
        )
        uses_tail_native_kv = (
            self._topology_version
            == self.ACTION30_PIT_NATIVE_KV_TOPOLOGY_VERSION
        )
        legacy_bridge_count = (
            0
            if uses_global_native_kv
            else self.FORMAL_PATCH_DEPTH
            if uses_tail_native_kv
            else self.action_depth
        )
        self.world_to_action_adapters = nn.ModuleList(
            [
                WorldToActionAdapter(
                    world_hidden_size=self.world_hidden_size,
                    action_hidden_size=self.action_hidden_size,
                    mot_dim=self.mot_dim,
                    mot_num_heads=self.mot_num_heads,
                    norm_eps=k_norm_eps,
                )
                for _ in range(legacy_bridge_count)
            ]
        )

        self.world_native_kv_adapters = nn.ModuleList()
        if uses_global_native_kv:
            if (
                not hasattr(self.world_expert, "patch_blocks")
                or len(self.world_expert.patch_blocks) != self.FORMAL_PATCH_DEPTH
            ):
                raise TypeError(
                    "Global-native-K/V topology requires 26 Global blocks."
                )
            world_blocks = list(self.world_expert.patch_blocks)
            if uses_pixel_native_kv:
                if (
                    not hasattr(self.world_expert, "pixel_blocks")
                    or len(self.world_expert.pixel_blocks)
                    != self.FORMAL_PIXEL_DEPTH
                ):
                    raise TypeError(
                        "All-native-K/V topology requires four PiT blocks."
                    )
                world_blocks.extend(self.world_expert.pixel_blocks)
            for layer_index, world_block in enumerate(world_blocks):
                if not hasattr(world_block, "attn"):
                    raise TypeError("Every World block must expose native attention.")
                action_block = self.action_expert.blocks[layer_index]
                if not hasattr(action_block, "self_attn"):
                    raise TypeError("Every Action block must expose self_attn.")
                self.world_native_kv_adapters.append(
                    WorldNativeKVAdapter(
                        world_attn_dim=self._positive_int_attr(
                            world_block.attn,
                            "dim",
                        ),
                        action_attn_dim=self._positive_int_attr(
                            action_block.self_attn,
                            "attn_hidden_dim",
                        ),
                    )
                )

        self.pit_native_kv_adapters = nn.ModuleList()
        if uses_tail_native_kv:
            for pixel_index, pixel_block in enumerate(self.world_expert.pixel_blocks):
                action_block = self.action_expert.blocks[
                    self.FORMAL_PATCH_DEPTH + pixel_index
                ]
                self.pit_native_kv_adapters.append(
                    WorldNativeKVAdapter(
                        world_attn_dim=self._positive_int_attr(
                            pixel_block.attn,
                            "dim",
                        ),
                        action_attn_dim=self._positive_int_attr(
                            action_block.self_attn,
                            "attn_hidden_dim",
                        ),
                    )
                )

    @staticmethod
    def _positive_int_attr(module: nn.Module, name: str) -> int:
        if not hasattr(module, name):
            raise TypeError(
                f"{type(module).__name__} must define integer attribute `{name}`."
            )
        value = int(getattr(module, name))
        if value <= 0:
            raise ValueError(
                f"{type(module).__name__}.{name} must be positive, got {value}."
            )
        return value

    @classmethod
    def _resolve_world_depth(cls, world_expert: nn.Module) -> int:
        depth = cls._positive_int_attr(world_expert, "patch_depth")
        if not hasattr(world_expert, "patch_blocks"):
            raise TypeError("PixelDiT must expose `patch_blocks`.")
        if len(world_expert.patch_blocks) != depth:
            raise ValueError(
                "PixelDiT `patch_depth` disagrees with `len(patch_blocks)`: "
                f"{depth} vs {len(world_expert.patch_blocks)}."
            )
        return depth

    @staticmethod
    def _resolve_action_depth(action_expert: nn.Module) -> int:
        if not hasattr(action_expert, "blocks"):
            raise TypeError("ActionDiT must expose `blocks`.")
        depth = len(action_expert.blocks)
        if depth <= 0:
            raise ValueError("ActionDiT must contain at least one block.")
        if not callable(getattr(action_expert, "pre_dit", None)):
            raise TypeError("ActionDiT must implement `pre_dit(...)`.")
        if not callable(getattr(action_expert, "post_dit", None)):
            raise TypeError("ActionDiT must implement `post_dit(...)`.")
        return depth

    @property
    def interaction_mode(self) -> str:
        return self._interaction_mode

    @interaction_mode.setter
    def interaction_mode(self, value: str) -> None:
        if value not in self.INTERACTION_MODES:
            raise ValueError(
                f"`interaction_mode` must be one of "
                f"{sorted(self.INTERACTION_MODES)}, got {value!r}."
            )
        self._interaction_mode = value

    @property
    def bridge_semantics_version(self) -> str:
        """Immutable execution-order identity bound into training checkpoints."""

        return self._bridge_semantics_version

    @property
    def topology_version(self) -> str:
        """Immutable topology identity bound into training checkpoints."""

        return self._topology_version

    @property
    def bridge_source_schedule(self) -> tuple[tuple[str, int], ...]:
        """Return the exact source kind/index consumed by every bridge."""

        global_sources = tuple(
            ("global", index) for index in range(self.FORMAL_PATCH_DEPTH)
        )
        if self._topology_version == self.LEGACY_TOPOLOGY_VERSION:
            return global_sources
        if (
            self._topology_version
            == self.ACTION26_GLOBAL_NATIVE_KV_TOPOLOGY_VERSION
        ):
            return tuple(
                ("global_native_kv_mixed_with_action_self_kv", index)
                for index in range(self.FORMAL_PATCH_DEPTH)
            )
        if self._topology_version == self.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION:
            native_global_sources = tuple(
                ("global_native_kv_mixed_with_action_self_kv", index)
                for index in range(self.FORMAL_PATCH_DEPTH)
            )
            native_pixel_sources = tuple(
                ("pit_native_kv_mixed_with_action_self_kv", index)
                for index in range(self.FORMAL_PIXEL_DEPTH)
            )
            return native_global_sources + native_pixel_sources
        if self._topology_version == self.ACTION30_PIT_NATIVE_KV_TOPOLOGY_VERSION:
            pixel_source_kind = "pit_native_kv_mixed_with_action_self_kv"
        elif self._topology_version == self.ACTION30_PIT_ONLY_TOPOLOGY_VERSION:
            pixel_source_kind = "pit_compact"
        else:
            pixel_source_kind = "final_global_concat_pit_compact"
        pixel_sources = tuple(
            (pixel_source_kind, index) for index in range(self.FORMAL_PIXEL_DEPTH)
        )
        return global_sources + pixel_sources

    @property
    def layer_map(self) -> tuple[int, ...]:
        """Compatibility metadata; Action execution is identity by topology."""

        return tuple(range(self.action_depth))

    @property
    def world_k_norm(self) -> nn.Module:
        """Return the K-normalizer actually used by the first World layer."""

        if self.world_to_action_adapters:
            return self.world_to_action_adapters[0].world_norm
        return self.world_expert.patch_blocks[0].attn.k_norm

    def _require_finite(self, tensor: torch.Tensor, name: str) -> None:
        if self.validate_finite and not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"`{name}` contains NaN or Inf.")

    @staticmethod
    def _split_modulation(
        block: nn.Module,
        t_mod: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        has_sequence_modulation = t_mod.ndim == 4
        chunk_dim = 2 if has_sequence_modulation else 1
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        chunks = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_sequence_modulation:
            chunks = tuple(value.squeeze(2) for value in chunks)
        return chunks

    def _action_self_attention_phase(
        self,
        block: nn.Module,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self._split_modulation(block, action_t_mod)
        residual = action_tokens
        attention_input = modulate(
            block.norm1(action_tokens),
            shift_msa,
            scale_msa,
        )
        q_action = block.self_attn.norm_q(block.self_attn.q(attention_input))
        k_action = block.self_attn.norm_k(block.self_attn.k(attention_input))
        v_action = block.self_attn.v(attention_input)
        q_action = rope_apply(q_action, action_freqs, block.num_heads)
        k_action = rope_apply(k_action, action_freqs, block.num_heads)
        attention_output = flash_attention(
            q=q_action,
            k=k_action,
            v=v_action,
            num_heads=block.num_heads,
        )
        action_tokens = block.gate(
            residual,
            gate_msa,
            block.self_attn.o(attention_output),
        )
        return action_tokens, shift_mlp, scale_mlp, gate_mlp

    def _action_world_native_kv_attention_phase(
        self,
        block: nn.Module,
        adapter: WorldNativeKVAdapter,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        world_native_k: torch.Tensor,
        world_native_v: torch.Tensor,
        source_patch_valid: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self._split_modulation(block, action_t_mod)
        residual = action_tokens
        attention_input = modulate(
            block.norm1(action_tokens),
            shift_msa,
            scale_msa,
        )
        q_action = block.self_attn.norm_q(block.self_attn.q(attention_input))
        k_action = block.self_attn.norm_k(block.self_attn.k(attention_input))
        v_action = block.self_attn.v(attention_input)
        q_action = rope_apply(q_action, action_freqs, block.num_heads)
        k_action = rope_apply(k_action, action_freqs, block.num_heads)

        k_world, v_world = adapter(world_native_k, world_native_v)
        if k_world.shape[:2] != world_native_k.shape[:2] or v_world.shape != k_world.shape:
            raise RuntimeError("Adapted World native K/V shape contract drifted.")
        if k_world.shape[0] != action_tokens.shape[0]:
            raise ValueError("World and Action native K/V batch sizes must match.")
        if k_world.shape[-1] != k_action.shape[-1]:
            raise RuntimeError(
                "Adapted World K/V width must match Action self-attention width."
            )
        k_mixed = torch.cat([k_world, k_action], dim=1)
        v_mixed = torch.cat([v_world, v_action], dim=1)

        mixed_mask = None
        if source_patch_valid is not None:
            expected_mask_shape = (k_world.shape[0], k_world.shape[1])
            if source_patch_valid.shape != expected_mask_shape:
                raise ValueError(
                    "World native-K/V source mask must be [B,N], got "
                    f"{tuple(source_patch_valid.shape)} vs {expected_mask_shape}."
                )
            if source_patch_valid.dtype != torch.bool:
                raise TypeError("World native-K/V source mask must use torch.bool.")
            # PixDiT.prepare_tokens validates non-empty source support once before
            # the 30-layer loop. Repeating .item() here would force one GPU/CPU
            # synchronization per Action layer.
            action_key_valid = torch.ones(
                action_tokens.shape[:2],
                dtype=torch.bool,
                device=action_tokens.device,
            )
            mixed_mask = torch.cat(
                [
                    source_patch_valid.to(device=action_tokens.device),
                    action_key_valid,
                ],
                dim=1,
            )[:, None, None, :]

        attention_output = flash_attention(
            q=q_action,
            k=k_mixed,
            v=v_mixed,
            num_heads=block.num_heads,
            ctx_mask=mixed_mask,
        )
        action_tokens = block.gate(
            residual,
            gate_msa,
            block.self_attn.o(attention_output),
        )
        return action_tokens, shift_mlp, scale_mlp, gate_mlp


    @staticmethod
    def _action_text_and_ffn_phase(
        block: nn.Module,
        action_tokens: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if context is not None:
            if context_mask is not None and context_mask.ndim == 3:
                context_mask = context_mask.unsqueeze(1)
            action_tokens = action_tokens + block.cross_attn(
                block.norm3(action_tokens),
                context,
                ctx_mask=context_mask,
            )
        mlp_input = modulate(
            block.norm2(action_tokens),
            shift_mlp,
            scale_mlp,
        )
        return block.gate(
            action_tokens,
            gate_mlp,
            block.ffn(mlp_input),
        )

    def _run_action_self_phase(
        self,
        block: nn.Module,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            bool(getattr(self.action_expert, "use_gradient_checkpointing", False))
            and self.training
        ):
            return checkpoint(
                lambda tokens, freqs, t_mod: self._action_self_attention_phase(
                    block,
                    tokens,
                    freqs,
                    t_mod,
                ),
                action_tokens,
                action_freqs,
                action_t_mod,
                use_reentrant=False,
            )
        return self._action_self_attention_phase(
            block,
            action_tokens,
            action_freqs,
            action_t_mod,
        )

    def _run_action_world_native_kv_phase(
        self,
        block: nn.Module,
        adapter: WorldNativeKVAdapter,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        world_native_k: torch.Tensor,
        world_native_v: torch.Tensor,
        source_patch_valid: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            bool(getattr(self.action_expert, "use_gradient_checkpointing", False))
            and self.training
        ):
            return checkpoint(
                lambda tokens, freqs, t_mod, native_k, native_v: (
                    self._action_world_native_kv_attention_phase(
                        block,
                        adapter,
                        tokens,
                        freqs,
                        t_mod,
                        native_k,
                        native_v,
                        source_patch_valid,
                    )
                ),
                action_tokens,
                action_freqs,
                action_t_mod,
                world_native_k,
                world_native_v,
                use_reentrant=False,
            )
        return self._action_world_native_kv_attention_phase(
            block,
            adapter,
            action_tokens,
            action_freqs,
            action_t_mod,
            world_native_k,
            world_native_v,
            source_patch_valid,
        )


    def _run_bridge(
        self,
        adapter: WorldToActionAdapter,
        action_tokens: torch.Tensor,
        world_hidden: torch.Tensor,
        source_patch_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(
                lambda action, world: adapter(
                    action,
                    world,
                    source_patch_valid,
                ),
                action_tokens,
                world_hidden,
                use_reentrant=False,
            )
        return adapter(action_tokens, world_hidden, source_patch_valid)

    def _run_action_post_phase(
        self,
        block: nn.Module,
        action_tokens: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            bool(getattr(self.action_expert, "use_gradient_checkpointing", False))
            and self.training
        ):
            return checkpoint(
                lambda tokens, shift, scale, gate: self._action_text_and_ffn_phase(
                    block,
                    tokens,
                    shift,
                    scale,
                    gate,
                    context,
                    context_mask,
                ),
                action_tokens,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return self._action_text_and_ffn_phase(
            block,
            action_tokens,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            context,
            context_mask,
        )

    def _world_language_condition(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            context.ndim != 3
            or context.shape[2] != self.text_dim
            or context_mask.shape != context.shape[:2]
        ):
            raise ValueError(
                "World language requires context [B,L,text_dim] and mask [B,L]."
            )
        if context_mask.dtype != torch.bool:
            raise TypeError("World language context_mask must use torch.bool.")
        if not bool(context_mask.any(dim=1).all().item()):
            raise ValueError("Every sample needs one valid world-language token.")
        mask = context_mask.to(device=context.device).unsqueeze(-1)
        pooled = torch.where(mask, context, torch.zeros_like(context)).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp_min(1).to(dtype=context.dtype)
        return self.world_text_projection(pooled)

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
        """Run only PixelDiT; ActionDiT and all bridge adapters stay untouched."""

        world_language = self._world_language_condition(
            world_context,
            world_context_mask,
        )
        world_velocity = self.world_expert(
            world_condition,
            noised_flow,
            t_world,
            source_support_mask=source_support_mask,
            extra_condition=world_language,
        )
        if world_velocity.shape != noised_flow.shape:
            raise ValueError(
                "PixelDiT velocity must match packed noised flow, got "
                f"{tuple(world_velocity.shape)} vs {tuple(noised_flow.shape)}."
            )
        self._require_finite(world_velocity, "world_velocity")
        return world_velocity

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
        action_pre = self.action_expert.pre_dit(
            action_tokens=noised_action,
            timestep=t_action,
            context=context,
            context_mask=context_mask,
        )
        required_pre_keys = {
            "tokens",
            "freqs",
            "t_mod",
            "context",
            "context_mask",
        }
        missing_pre_keys = required_pre_keys - set(action_pre)
        if missing_pre_keys:
            raise ValueError(
                f"ActionDiT `pre_dit` missing keys: {sorted(missing_pre_keys)}."
            )
        action_tokens = action_pre["tokens"]
        action_freqs = action_pre["freqs"]
        action_t_mod = action_pre["t_mod"]
        action_context = action_pre["context"]
        action_context_mask = action_pre["context_mask"]
        source_patch_valid = world_state["patch_key_valid"]

        interaction_count = 0
        use_global_native_kv = self._topology_version in {
            self.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION,
            self.ACTION26_GLOBAL_NATIVE_KV_TOPOLOGY_VERSION,
        }
        use_all_native_kv = (
            self._topology_version
            == self.ACTION30_ALL_NATIVE_KV_TOPOLOGY_VERSION
        )
        use_tail_native_kv = (
            self._topology_version
            == self.ACTION30_PIT_NATIVE_KV_TOPOLOGY_VERSION
        )
        for layer_index in range(self.FORMAL_PATCH_DEPTH):
            # World executes first and never consumes action state.
            if use_global_native_kv:
                with world_grad_context():
                    world_result = self.world_expert.forward_patch_block(
                        layer_index,
                        world_state,
                        return_native_kv=True,
                    )
                if not isinstance(world_result, tuple) or len(world_result) != 2:
                    raise TypeError(
                        "All-native-K/V topology requires (hidden, native_kv) "
                        "from every Global block."
                    )
                world_hidden, world_native_kv = world_result
                if (
                    not isinstance(world_native_kv, dict)
                    or set(world_native_kv) != {"k", "v"}
                ):
                    raise TypeError(
                        "Global native-K/V topology requires exactly k and v."
                    )
                for name in ("k", "v"):
                    self._require_finite(
                        world_native_kv[name],
                        f"global_native_{name}_{layer_index}",
                    )
                if freeze_world:
                    world_native_kv = {
                        name: value.detach()
                        for name, value in world_native_kv.items()
                    }
            else:
                with world_grad_context():
                    world_hidden = self.world_expert.forward_patch_block(
                        layer_index,
                        world_state,
                    )
                if freeze_world:
                    world_hidden = world_hidden.detach()
            block = self.action_expert.blocks[layer_index]
            if use_global_native_kv and self.interaction_mode == "momap_to_action":
                (
                    action_tokens,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self._run_action_world_native_kv_phase(
                    block,
                    self.world_native_kv_adapters[layer_index],
                    action_tokens,
                    action_freqs,
                    action_t_mod,
                    world_native_kv["k"],
                    world_native_kv["v"],
                    source_patch_valid,
                )
                interaction_count += 1
            else:
                (
                    action_tokens,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self._run_action_self_phase(
                    block,
                    action_tokens,
                    action_freqs,
                    action_t_mod,
                )
                if self.interaction_mode == "momap_to_action":
                    action_tokens = self._run_bridge(
                        self.world_to_action_adapters[layer_index],
                        action_tokens,
                        world_hidden,
                        source_patch_valid,
                    )
                    interaction_count += 1
            action_tokens = self._run_action_post_phase(
                block,
                action_tokens,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                action_context,
                action_context_mask,
            )
            self._require_finite(
                action_tokens,
                f"action_tokens_after_layer_{layer_index}",
            )

        with world_grad_context():
            self.world_expert.finish_patch_blocks(world_state)
        if self._topology_version in {
            self.LEGACY_TOPOLOGY_VERSION,
            self.ACTION26_GLOBAL_NATIVE_KV_TOPOLOGY_VERSION,
        }:
            with world_grad_context():
                self.world_expert.forward_pixel_modules(world_state)
        else:
            final_global = world_state.get("finished_patch_tokens")
            if not isinstance(final_global, torch.Tensor) or final_global.ndim != 3:
                raise ValueError(
                    "PixelDiT `finish_patch_blocks` must expose "
                    "`finished_patch_tokens` as [B,L,D]."
                )
            if freeze_world:
                final_global = final_global.detach()
            with world_grad_context():
                self.world_expert.prepare_pixel_tokens(world_state)
            for pixel_index in range(self.FORMAL_PIXEL_DEPTH):
                use_world_native_kv = use_all_native_kv or use_tail_native_kv
                with world_grad_context():
                    if use_world_native_kv:
                        pixel_result = self.world_expert.forward_pixel_block(
                            pixel_index,
                            world_state,
                            return_native_kv=True,
                        )
                    else:
                        pixel_result = self.world_expert.forward_pixel_block(
                            pixel_index,
                            world_state,
                        )
                world_native_kv = None
                if use_world_native_kv:
                    if not isinstance(pixel_result, tuple) or len(pixel_result) != 2:
                        raise TypeError(
                            "PixelDiT native-K/V topology requires "
                            "(compact, native_kv) from forward_pixel_block."
                        )
                    compact, world_native_kv = pixel_result
                else:
                    compact = pixel_result
                if not isinstance(compact, torch.Tensor):
                    raise TypeError(
                        "PixelDiT `forward_pixel_block` must return compact tokens."
                    )
                if compact.shape != final_global.shape:
                    raise ValueError(
                        "PiT compact tokens must align with final global tokens "
                        "as [B,L,D], got "
                        f"{tuple(compact.shape)} vs {tuple(final_global.shape)}."
                    )
                if (
                    compact.device != final_global.device
                    or compact.dtype != final_global.dtype
                ):
                    raise ValueError(
                        "PiT compact and final global tokens must share "
                        "dtype/device."
                    )
                if freeze_world:
                    compact = compact.detach()
                self._require_finite(compact, f"pit_compact_{pixel_index}")
                action_layer_index = self.FORMAL_PATCH_DEPTH + pixel_index
                block = self.action_expert.blocks[action_layer_index]
                if use_world_native_kv:
                    if (
                        not isinstance(world_native_kv, dict)
                        or set(world_native_kv) != {"k", "v"}
                    ):
                        raise TypeError(
                            "PiT native-K/V topology requires exactly k and v."
                        )
                    for name in ("k", "v"):
                        self._require_finite(
                            world_native_kv[name],
                            f"pit_native_{name}_{pixel_index}",
                        )
                    if freeze_world:
                        world_native_kv = {
                            name: value.detach()
                            for name, value in world_native_kv.items()
                        }
                    if self.interaction_mode == "momap_to_action":
                        (
                            action_tokens,
                            shift_mlp,
                            scale_mlp,
                            gate_mlp,
                        ) = self._run_action_world_native_kv_phase(
                            block,
                            (
                                self.world_native_kv_adapters[action_layer_index]
                                if use_all_native_kv
                                else self.pit_native_kv_adapters[pixel_index]
                            ),
                            action_tokens,
                            action_freqs,
                            action_t_mod,
                            world_native_kv["k"],
                            world_native_kv["v"],
                            source_patch_valid,
                        )
                        interaction_count += 1
                    else:
                        (
                            action_tokens,
                            shift_mlp,
                            scale_mlp,
                            gate_mlp,
                        ) = self._run_action_self_phase(
                            block,
                            action_tokens,
                            action_freqs,
                            action_t_mod,
                        )
                else:
                    if (
                        self._topology_version
                        == self.ACTION30_PIT_ONLY_TOPOLOGY_VERSION
                    ):
                        tail_bridge_source = compact
                        tail_source_patch_valid = source_patch_valid
                    else:
                        tail_bridge_source = torch.cat(
                            [final_global, compact],
                            dim=1,
                        )
                        tail_source_patch_valid = (
                            None
                            if source_patch_valid is None
                            else torch.cat(
                                [source_patch_valid, source_patch_valid],
                                dim=1,
                            )
                        )
                    (
                        action_tokens,
                        shift_mlp,
                        scale_mlp,
                        gate_mlp,
                    ) = self._run_action_self_phase(
                        block,
                        action_tokens,
                        action_freqs,
                        action_t_mod,
                    )
                    if self.interaction_mode == "momap_to_action":
                        action_tokens = self._run_bridge(
                            self.world_to_action_adapters[action_layer_index],
                            action_tokens,
                            tail_bridge_source,
                            tail_source_patch_valid,
                        )
                        interaction_count += 1
                action_tokens = self._run_action_post_phase(
                    block,
                    action_tokens,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    action_context,
                    action_context_mask,
                )
                self._require_finite(
                    action_tokens,
                    f"action_tokens_after_layer_{action_layer_index}",
                )
        world_velocity = None
        if return_world_velocity:
            with world_grad_context():
                world_velocity = self.world_expert.final_head(world_state)
        action_velocity = self.action_expert.post_dit(action_tokens, action_pre)
        if world_velocity is not None and world_velocity.shape != noised_flow.shape:
            raise ValueError(
                "PixelDiT velocity must match packed noised flow, got "
                f"{tuple(world_velocity.shape)} vs {tuple(noised_flow.shape)}."
            )
        if action_velocity.shape != noised_action.shape:
            raise ValueError(
                "ActionDiT velocity must match noised action, got "
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
        }


__all__ = ["PixelDiTActionMoT", "WorldNativeKVAdapter", "WorldToActionAdapter"]
