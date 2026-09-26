"""PixelDiT-aligned Action DiTs for joint and temporal token layouts.

The original aligned layout uses fourteen joint tokens with the 32-step horizon
as channels and a fixed joint-position embedding.  The temporal sibling keeps
the native ``[B,32,14]`` layout, projects each 14-D step independently, and
applies one-dimensional temporal RoPE to Action Q/K.

The transformer blocks retain the parameter structure of PixelDiT's global
``AugmentedDiTBlock``. This makes the first 26 blocks shape-compatible with the
World global blocks, while the attention submodules of the final four blocks are
shape-compatible with the World PiT attention submodules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import AugmentedDiTBlock
from .modules import TimestepConditioner, apply_adaln, apply_rotary_emb


def fixed_1d_sincos_position_embedding(
    num_positions: int,
    embed_dim: int,
) -> torch.Tensor:
    """Return deterministic position embeddings with shape ``[1,N,D]``."""

    if isinstance(num_positions, bool) or not isinstance(num_positions, int):
        raise TypeError("`num_positions` must be an integer.")
    if isinstance(embed_dim, bool) or not isinstance(embed_dim, int):
        raise TypeError("`embed_dim` must be an integer.")
    if num_positions <= 0:
        raise ValueError("`num_positions` must be positive.")
    if embed_dim <= 0 or embed_dim % 2:
        raise ValueError("`embed_dim` must be a positive even integer.")

    positions = torch.arange(num_positions, dtype=torch.float32)
    omega = torch.arange(embed_dim // 2, dtype=torch.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000.0**omega)
    angles = torch.outer(positions.to(dtype=torch.float64), omega)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return embedding.to(dtype=torch.float32).unsqueeze(0)


def precompute_1d_rope_frequencies(
    head_dim: int,
    sequence_length: int,
    *,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Return complex 1-D RoPE frequencies for ``[B,T,H,head_dim]`` Q/K."""

    if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError("`head_dim` must be a positive integer.")
    if head_dim % 2:
        raise ValueError("`head_dim` must be even for 1-D RoPE.")
    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length <= 0
    ):
        raise ValueError("`sequence_length` must be a positive integer.")
    if not math.isfinite(theta) or theta <= 0.0:
        raise ValueError("`theta` must be finite and positive.")
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32)
            / float(head_dim)
        )
    )
    angles = torch.outer(
        torch.arange(sequence_length, dtype=torch.float32),
        frequencies,
    )
    return torch.polar(torch.ones_like(angles), angles)


LEGACY_NATIVE_KV_FUSION_VERSION = (
    "world_then_action_token_concat_log_count_corrected_v1"
)
COSINE_ANNEAL_NATIVE_KV_FUSION_VERSION = (
    "world_then_action_token_concat_log_count_cosine_anneal_to_none_v1"
)


def cosine_count_correction_scale(
    global_optimizer_step: int,
    *,
    transition_start_global_step: int,
    transition_optimizer_steps: int,
) -> float:
    """Return the deterministic log-count-correction multiplier in ``[0,1]``.

    ``global_optimizer_step`` is the number of successful optimizer updates
    completed before the current forward. Thus ``S`` retains the legacy
    correction, and the forward at ``S + 500`` and later has zero correction.
    """

    fields = {
        "global_optimizer_step": global_optimizer_step,
        "transition_start_global_step": transition_start_global_step,
        "transition_optimizer_steps": transition_optimizer_steps,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in fields.values()
    ):
        raise TypeError(
            "Count-correction schedule fields must be integers, got "
            f"{fields}."
        )
    if global_optimizer_step < 0 or transition_start_global_step < 0:
        raise ValueError("Optimizer steps must be non-negative.")
    if transition_optimizer_steps <= 0:
        raise ValueError("`transition_optimizer_steps` must be positive.")

    if global_optimizer_step <= transition_start_global_step:
        return 1.0
    elapsed = global_optimizer_step - transition_start_global_step
    if elapsed >= transition_optimizer_steps:
        return 0.0
    progress = float(elapsed) / float(transition_optimizer_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def fuse_world_action_native_kv(
    action_k: torch.Tensor,
    action_v: torch.Tensor,
    world_k: torch.Tensor,
    world_v: torch.Tensor,
    world_key_valid: Optional[torch.Tensor] = None,
    *,
    world_logit_correction_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate World and Action K/V with a scheduled count correction.

    K/V layout is ``[B,N,heads,head_dim]``. World keys precede Action keys on
    the token axis. The returned additive SDPA bias gives every valid World key
    ``scale * log(N_action / N_world_valid)``. ``scale=1`` preserves V8;
    ``scale=0`` is FastWAM-style direct concatenation with only invalid World
    keys masked to ``-inf``. Action keys always receive zero bias.
    """

    tensors = {
        "action_k": action_k,
        "action_v": action_v,
        "world_k": world_k,
        "world_v": world_v,
    }
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 4:
            raise ValueError(f"`{name}` must be [B,N,heads,head_dim].")
        if not value.is_floating_point():
            raise TypeError(f"`{name}` must be floating point.")
    if action_k.shape != action_v.shape:
        raise ValueError("Action native K/V shapes must match.")
    if world_k.shape != world_v.shape:
        raise ValueError("World native K/V shapes must match.")
    if action_k.shape[0] != world_k.shape[0]:
        raise ValueError("World and Action native K/V batch sizes must match.")
    if action_k.shape[2:] != world_k.shape[2:]:
        raise ValueError(
            "World and Action native K/V head layouts must match, got "
            f"{tuple(world_k.shape[2:])} and {tuple(action_k.shape[2:])}."
        )
    reference = action_k
    if any(
        value.device != reference.device or value.dtype != reference.dtype
        for value in tensors.values()
    ):
        raise ValueError("World and Action native K/V must share dtype/device.")
    if isinstance(world_logit_correction_scale, bool) or not isinstance(
        world_logit_correction_scale,
        (int, float),
    ):
        raise TypeError("`world_logit_correction_scale` must be a real scalar.")
    correction_scale = float(world_logit_correction_scale)
    if not math.isfinite(correction_scale) or not 0.0 <= correction_scale <= 1.0:
        raise ValueError("`world_logit_correction_scale` must be finite in [0,1].")

    batch_size, action_length = action_k.shape[:2]
    world_length = int(world_k.shape[1])
    if action_length <= 0 or world_length <= 0:
        raise ValueError("World and Action K/V banks must both contain tokens.")
    if world_key_valid is None:
        world_valid = torch.ones(
            (batch_size, world_length),
            dtype=torch.bool,
            device=reference.device,
        )
    else:
        if not isinstance(world_key_valid, torch.Tensor):
            raise TypeError("`world_key_valid` must be a torch.Tensor or None.")
        if world_key_valid.shape != (batch_size, world_length):
            raise ValueError(
                "`world_key_valid` must be [B,N_world], got "
                f"{tuple(world_key_valid.shape)} vs {(batch_size, world_length)}."
            )
        if world_key_valid.dtype != torch.bool:
            raise TypeError("`world_key_valid` must use torch.bool.")
        world_valid = world_key_valid.to(device=reference.device)

    valid_count = world_valid.sum(dim=1)
    if not bool((valid_count > 0).all().item()):
        raise ValueError("Every sample must expose at least one valid World key.")

    correction = torch.log(
        torch.full(
            (batch_size,),
            float(action_length),
            device=reference.device,
            dtype=torch.float32,
        )
        / valid_count.to(dtype=torch.float32)
    )
    correction = correction * correction_scale
    world_bias = correction[:, None].expand(batch_size, world_length)
    world_bias = world_bias.masked_fill(~world_valid, float("-inf"))
    action_bias = torch.zeros(
        (batch_size, action_length),
        device=reference.device,
        dtype=torch.float32,
    )
    attention_bias = torch.cat([world_bias, action_bias], dim=1)
    attention_bias = attention_bias[:, None, None, :].to(dtype=reference.dtype)
    return (
        torch.cat([world_k, action_k], dim=1),
        torch.cat([world_v, action_v], dim=1),
        attention_bias,
    )


class PixelDiTAlignedActionBlock(AugmentedDiTBlock):
    """Global-style PixelDiT block with caller-selected Action positioning."""

    def _action_attention(
        self,
        x: torch.Tensor,
        *,
        world_native_kv: Optional[Mapping[str, torch.Tensor]],
        world_key_valid: Optional[torch.Tensor],
        world_logit_correction_scale: float,
        action_freqs: Optional[torch.Tensor],
        return_kv: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, sequence_length, channels = x.shape
        qkv = self.attn.qkv(x).reshape(
            batch_size,
            sequence_length,
            3,
            self.attn.num_heads,
            self.attn.head_dim,
        )
        query, action_k, action_v = qkv.unbind(dim=2)
        query = self.attn.q_norm(query)
        action_k = self.attn.k_norm(action_k)

        if action_freqs is not None:
            query, action_k = apply_rotary_emb(query, action_k, action_freqs)

        # FP32 norm parameters can promote Q/K under mixed precision.
        # Native attention banks use the linear projection compute dtype.
        if torch.is_autocast_enabled(x.device.type):
            query = query.to(dtype=qkv.dtype)
            action_k = action_k.to(dtype=qkv.dtype)
            action_v = action_v.to(dtype=qkv.dtype)
            if world_native_kv is not None:
                world_native_kv = {
                    key: value.to(dtype=qkv.dtype)
                    for key, value in world_native_kv.items()
                }

        # Joint14 callers pass no frequencies because joint identity comes from
        # their fixed absolute embedding. Temporal32 callers pass 1D RoPE.
        attention_bias = None
        mixed_k = action_k
        mixed_v = action_v
        if world_native_kv is not None:
            if not isinstance(world_native_kv, Mapping) or set(world_native_kv) != {
                "k",
                "v",
            }:
                raise ValueError("`world_native_kv` must contain exactly `k` and `v`.")
            mixed_k, mixed_v, attention_bias = fuse_world_action_native_kv(
                action_k,
                action_v,
                world_native_kv["k"],
                world_native_kv["v"],
                world_key_valid,
                world_logit_correction_scale=world_logit_correction_scale,
            )
        elif world_key_valid is not None:
            raise ValueError("`world_key_valid` requires `world_native_kv`.")

        attention = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            mixed_k.transpose(1, 2),
            mixed_v.transpose(1, 2),
            attn_mask=attention_bias,
            dropout_p=self.attn.attn_drop.p if self.training else 0.0,
        )
        output = attention.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            channels,
        )
        output = self.attn.proj_drop(self.attn.proj(output))
        if return_kv:
            return output, {"k": action_k, "v": action_v}
        return output

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        *,
        world_native_kv: Optional[Mapping[str, torch.Tensor]] = None,
        world_key_valid: Optional[torch.Tensor] = None,
        world_logit_correction_scale: float = 1.0,
        action_freqs: Optional[torch.Tensor] = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"Action block input must be [B,J,D], got {tuple(x.shape)}.")
        if condition.ndim == 2:
            condition = condition.unsqueeze(1)
        if condition.shape != (x.shape[0], 1, x.shape[2]):
            raise ValueError(
                "Action AdaLN condition must be [B,D] or [B,1,D], got "
                f"{tuple(condition.shape)} for tokens {tuple(x.shape)}."
            )
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(condition).chunk(6, dim=-1)
        attention_input = apply_adaln(self.norm1(x), shift_msa, scale_msa)
        attention_result = self._action_attention(
            attention_input,
            world_native_kv=world_native_kv,
            world_key_valid=world_key_valid,
            world_logit_correction_scale=world_logit_correction_scale,
            action_freqs=action_freqs,
            return_kv=return_kv,
        )
        if return_kv:
            if not isinstance(attention_result, tuple):
                raise RuntimeError("Action native K/V return contract drifted.")
            attention_output, action_native_kv = attention_result
        else:
            if not isinstance(attention_result, torch.Tensor):
                raise RuntimeError("Action attention tensor return contract drifted.")
            attention_output = attention_result
            action_native_kv = None
        x = x + gate_msa * attention_output
        x = x + gate_mlp * self.mlp(
            apply_adaln(self.norm2(x), shift_mlp, scale_mlp)
        )
        if return_kv:
            assert action_native_kv is not None
            return x, action_native_kv
        return x


class PixelDiTAlignedActionDiT(nn.Module):
    """Action velocity model using fourteen joint-trajectory spatial tokens."""

    TOKEN_SEMANTICS_VERSION = "joint14_full_horizon32_spatial_tokens_v1"
    POSITION_ENCODING_VERSION = "fixed_joint_index_1d_sincos_no_action_rope_v1"
    NATIVE_KV_FUSION_VERSION = LEGACY_NATIVE_KV_FUSION_VERSION

    def __init__(
        self,
        *,
        action_horizon: int = 32,
        action_dim: int = 14,
        hidden_size: int = 1152,
        num_heads: int = 16,
        num_layers: int = 30,
        text_dim: int = 4096,
        mlp_ratio: float = 4.0,
        use_gradient_checkpointing: bool = False,
        validate_finite: bool = False,
        native_kv_fusion_version: str = LEGACY_NATIVE_KV_FUSION_VERSION,
    ) -> None:
        super().__init__()
        integer_fields = {
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "text_dim": text_dim,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_fields.values()
        ):
            raise ValueError(
                "Action dimensions must be positive integers, got "
                f"{integer_fields}."
            )
        if hidden_size % num_heads:
            raise ValueError("`hidden_size` must be divisible by `num_heads`.")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("`mlp_ratio` must be finite and positive.")
        if native_kv_fusion_version not in {
            LEGACY_NATIVE_KV_FUSION_VERSION,
            COSINE_ANNEAL_NATIVE_KV_FUSION_VERSION,
        }:
            raise ValueError(
                "Unsupported aligned Action native-K/V fusion version: "
                f"{native_kv_fusion_version!r}."
            )

        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        # Compatibility aliases used by existing FastWAM wrappers/contracts.
        self.hidden_dim = self.hidden_size
        self.num_heads = int(num_heads)
        self.attn_head_dim = self.hidden_size // self.num_heads
        self.num_layers = int(num_layers)
        self.text_dim = int(text_dim)
        self.mlp_ratio = float(mlp_ratio)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.validate_finite = bool(validate_finite)
        self.native_kv_fusion_version = str(native_kv_fusion_version)

        self.action_encoder = nn.Linear(self.action_horizon, self.hidden_size)
        self.context_projection = nn.Linear(self.text_dim, self.hidden_size)
        self.t_embedder = TimestepConditioner(self.hidden_size)
        self.blocks = nn.ModuleList(
            [
                PixelDiTAlignedActionBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.action_decoder = nn.Linear(self.hidden_size, self.action_horizon)
        self.register_buffer(
            "joint_pos_embed",
            fixed_1d_sincos_position_embedding(
                self.action_dim,
                self.hidden_size,
            ),
            persistent=False,
        )
        self._initialize_action_specific_weights()

    def _initialize_action_specific_weights(self) -> None:
        nn.init.xavier_uniform_(self.action_encoder.weight)
        nn.init.zeros_(self.action_encoder.bias)
        nn.init.xavier_uniform_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.xavier_uniform_(self.action_decoder.weight)
        nn.init.zeros_(self.action_decoder.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)

    @staticmethod
    def _prepare_timestep(
        timestep: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        value = torch.as_tensor(timestep, device=device).reshape(-1)
        if value.numel() == 1 and batch_size > 1:
            value = value.expand(batch_size)
        if value.numel() != batch_size:
            raise ValueError(
                f"Expected one Action timestep per sample ({batch_size}), "
                f"got {value.numel()}."
            )
        if not value.is_floating_point():
            value = value.float()
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("Action timestep must contain only finite values.")
        if bool(((value < 0.0) | (value > 1.0)).any().item()):
            raise ValueError(
                "Aligned Action timestep must be WAN noise fraction in [0, 1]."
            )
        return value

    def _pool_context(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not isinstance(context, torch.Tensor)
            or context.ndim != 3
            or context.shape[0] != batch_size
            or context.shape[2] != self.text_dim
        ):
            raise ValueError(
                "Action context must be [B,L,text_dim], got "
                f"{getattr(context, 'shape', None)}."
            )
        if not context.is_floating_point() or not bool(torch.isfinite(context).all()):
            raise ValueError("Action context must be finite floating point.")
        if context_mask is None:
            prepared_mask = torch.ones(
                context.shape[:2],
                dtype=torch.bool,
                device=context.device,
            )
        else:
            if context_mask.shape != context.shape[:2]:
                raise ValueError(
                    "`context_mask` must be [B,L], got "
                    f"{tuple(context_mask.shape)} vs {tuple(context.shape[:2])}."
                )
            if context_mask.dtype != torch.bool:
                raise TypeError("`context_mask` must use torch.bool.")
            prepared_mask = context_mask.to(device=context.device)
        valid_count = prepared_mask.sum(dim=1, keepdim=True)
        if not bool((valid_count > 0).all().item()):
            raise ValueError("Every sample must expose at least one context token.")
        mask = prepared_mask.unsqueeze(-1)
        pooled = torch.where(mask, context, torch.zeros_like(context)).sum(dim=1)
        pooled = pooled / valid_count.to(dtype=context.dtype)
        return pooled, prepared_mask

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if not isinstance(action_tokens, torch.Tensor) or action_tokens.ndim != 3:
            raise ValueError("`action_tokens` must be [B,T,A].")
        expected = (self.action_horizon, self.action_dim)
        if tuple(action_tokens.shape[1:]) != expected:
            raise ValueError(
                "Action shape must match [B,T,A] contract: expected "
                f"{expected}, got {tuple(action_tokens.shape[1:])}."
            )
        if not action_tokens.is_floating_point() or not bool(
            torch.isfinite(action_tokens).all()
        ):
            raise ValueError("Action tokens must be finite floating point.")
        batch_size = int(action_tokens.shape[0])
        timestep = self._prepare_timestep(
            timestep,
            batch_size=batch_size,
            device=action_tokens.device,
        )
        pooled_context, prepared_mask = self._pool_context(
            context,
            context_mask,
            batch_size=batch_size,
        )

        encoder_weight = self.action_encoder.weight
        trajectories = action_tokens.transpose(1, 2).to(
            device=encoder_weight.device,
            dtype=encoder_weight.dtype,
        )
        tokens = self.action_encoder(trajectories)
        tokens = tokens + self.joint_pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        time_embedding = self.t_embedder(
            timestep.to(device=tokens.device, dtype=torch.float32)
        ).unsqueeze(1)
        context_embedding = self.context_projection(
            pooled_context.to(
                device=self.context_projection.weight.device,
                dtype=self.context_projection.weight.dtype,
            )
        ).unsqueeze(1)
        condition = F.silu(time_embedding + context_embedding)
        if self.validate_finite and not bool(
            torch.isfinite(tokens).all() and torch.isfinite(condition).all()
        ):
            raise FloatingPointError("Prepared Action tokens/condition are not finite.")
        return {
            "tokens": tokens,
            "condition": condition,
            "time_embedding": time_embedding,
            "context_embedding": context_embedding,
            "context_mask": prepared_mask,
            "meta": {
                "batch_size": batch_size,
                "action_horizon": self.action_horizon,
                "joint_tokens": self.action_dim,
                "token_semantics": self.TOKEN_SEMANTICS_VERSION,
                "position_encoding": self.POSITION_ENCODING_VERSION,
            },
            "next_block": 0,
        }

    def forward_block(
        self,
        layer_index: int,
        state: dict[str, Any],
        *,
        world_native_kv: Optional[Mapping[str, torch.Tensor]] = None,
        world_key_valid: Optional[torch.Tensor] = None,
        world_logit_correction_scale: float = 1.0,
        return_native_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(
                f"Action layer {layer_index} is outside [0,{self.num_layers})."
            )
        expected_index = int(state.get("next_block", -1))
        if layer_index != expected_index:
            raise ValueError(
                "Action blocks must execute once in order: "
                f"expected {expected_index}, got {layer_index}."
            )
        block = self.blocks[layer_index]
        has_world_native_kv = world_native_kv is not None
        if has_world_native_kv:
            if not isinstance(world_native_kv, Mapping) or set(world_native_kv) != {
                "k",
                "v",
            }:
                raise ValueError("`world_native_kv` must contain exactly `k` and `v`.")
            world_k: Optional[torch.Tensor] = world_native_kv["k"]
            world_v: Optional[torch.Tensor] = world_native_kv["v"]
        else:
            world_k = None
            world_v = None
        action_freqs = state.get("action_freqs")

        def _run(
            tokens: torch.Tensor,
            condition: torch.Tensor,
            checkpoint_world_k: Optional[torch.Tensor],
            checkpoint_world_v: Optional[torch.Tensor],
            checkpoint_world_key_valid: Optional[torch.Tensor],
            checkpoint_action_freqs: Optional[torch.Tensor],
        ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
            checkpoint_world_native_kv = (
                {"k": checkpoint_world_k, "v": checkpoint_world_v}
                if has_world_native_kv
                else None
            )
            return block(
                tokens,
                condition,
                world_native_kv=checkpoint_world_native_kv,
                world_key_valid=checkpoint_world_key_valid,
                world_logit_correction_scale=world_logit_correction_scale,
                action_freqs=checkpoint_action_freqs,
                return_kv=return_native_kv,
            )

        if self.use_gradient_checkpointing and self.training:
            result = checkpoint(
                _run,
                state["tokens"],
                state["condition"],
                world_k,
                world_v,
                world_key_valid,
                action_freqs,
                use_reentrant=False,
            )
        else:
            result = _run(
                state["tokens"],
                state["condition"],
                world_k,
                world_v,
                world_key_valid,
                action_freqs,
            )
        if return_native_kv:
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("Action block native K/V return contract drifted.")
            tokens, native_kv = result
        else:
            if not isinstance(result, torch.Tensor):
                raise RuntimeError("Action block tensor return contract drifted.")
            tokens = result
            native_kv = None
        if self.validate_finite and not bool(torch.isfinite(tokens).all()):
            raise FloatingPointError(f"Action block {layer_index} produced NaN or Inf.")
        state["tokens"] = tokens
        state["next_block"] = layer_index + 1
        if return_native_kv:
            assert native_kv is not None
            return tokens, native_kv
        return tokens

    def post_dit(
        self,
        tokens: torch.Tensor,
        pre_state: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        if tokens.shape[1:] != (self.action_dim, self.hidden_size):
            raise ValueError(
                "Action decoder expects [B,A,D], got " f"{tuple(tokens.shape)}."
            )
        if (
            pre_state is not None
            and int(pre_state.get("next_block", -1)) != self.num_layers
        ):
            raise ValueError("Action decoder requires all transformer blocks to finish.")
        velocity = self.action_decoder(tokens).transpose(1, 2).contiguous()
        if self.validate_finite and not bool(torch.isfinite(velocity).all()):
            raise FloatingPointError("Action decoder produced NaN or Inf.")
        return velocity

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        *,
        world_native_kv_by_layer: Optional[
            Sequence[Optional[Mapping[str, torch.Tensor]]]
        ] = None,
        world_key_valid: Optional[torch.Tensor] = None,
        world_logit_correction_scale: float = 1.0,
    ) -> torch.Tensor:
        state = self.pre_dit(
            action_tokens,
            timestep,
            context,
            context_mask,
        )
        if world_native_kv_by_layer is None:
            native_kv_sequence: Sequence[Optional[Mapping[str, torch.Tensor]]] = (
                [None] * self.num_layers
            )
            if world_key_valid is not None:
                raise ValueError(
                    "`world_key_valid` requires `world_native_kv_by_layer`."
                )
        else:
            if len(world_native_kv_by_layer) != self.num_layers:
                raise ValueError(
                    "Expected one World native K/V bank per Action layer, got "
                    f"{len(world_native_kv_by_layer)} vs {self.num_layers}."
                )
            native_kv_sequence = world_native_kv_by_layer
        for layer_index, world_native_kv in enumerate(native_kv_sequence):
            self.forward_block(
                layer_index,
                state,
                world_native_kv=world_native_kv,
                world_key_valid=world_key_valid,
                world_logit_correction_scale=world_logit_correction_scale,
            )
        return self.post_dit(state["tokens"], state)


class PixelDiTTemporalActionDiT(PixelDiTAlignedActionDiT):
    """PixelDiT Action DiT with one temporal token per low-level action step.

    The input/output action tensor remains ``[B,32,14]``.  Unlike the aligned
    joint-token implementation, it is never transposed: every time step is
    projected from 14 controls to one 1152-wide token and receives 1-D RoPE.
    The 30 Transformer blocks remain shape-compatible with Hybrid World so the
    existing World-to-Action transfer remains valid.
    """

    TOKEN_SEMANTICS_VERSION = "time32_per_step_action14_tokens_v1"
    POSITION_ENCODING_VERSION = "temporal_1d_rope_index_0_31_v1"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.action_encoder = nn.Linear(self.action_dim, self.hidden_size)
        self.action_decoder = nn.Linear(self.hidden_size, self.action_dim)
        nn.init.xavier_uniform_(self.action_encoder.weight)
        nn.init.zeros_(self.action_encoder.bias)
        nn.init.xavier_uniform_(self.action_decoder.weight)
        nn.init.zeros_(self.action_decoder.bias)
        del self._buffers["joint_pos_embed"]
        # Keep only integer positions as module state. Registering the complex
        # RoPE table directly makes ``Module.to(dtype=bfloat16)`` discard its
        # imaginary component. Integer buffers retain their dtype, and the
        # small complex table is rebuilt in float32 on the active device.
        self.register_buffer(
            "temporal_position_ids",
            torch.arange(self.action_horizon, dtype=torch.int64),
            persistent=False,
        )

    @property
    def temporal_rope(self) -> torch.Tensor:
        frequencies = 1.0 / (
            10000.0
            ** (
                torch.arange(
                    0,
                    self.attn_head_dim,
                    2,
                    dtype=torch.float32,
                    device=self.temporal_position_ids.device,
                )
                / float(self.attn_head_dim)
            )
        )
        angles = torch.outer(
            self.temporal_position_ids.to(dtype=torch.float32),
            frequencies,
        )
        return torch.polar(torch.ones_like(angles), angles)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if not isinstance(action_tokens, torch.Tensor) or action_tokens.ndim != 3:
            raise ValueError("`action_tokens` must be [B,T,A].")
        expected = (self.action_horizon, self.action_dim)
        if tuple(action_tokens.shape[1:]) != expected:
            raise ValueError(
                "Action shape must match [B,T,A] contract: expected "
                f"{expected}, got {tuple(action_tokens.shape[1:])}."
            )
        if not action_tokens.is_floating_point() or not bool(
            torch.isfinite(action_tokens).all()
        ):
            raise ValueError("Action tokens must be finite floating point.")
        batch_size = int(action_tokens.shape[0])
        timestep = self._prepare_timestep(
            timestep,
            batch_size=batch_size,
            device=action_tokens.device,
        )
        pooled_context, prepared_mask = self._pool_context(
            context,
            context_mask,
            batch_size=batch_size,
        )
        encoder_weight = self.action_encoder.weight
        tokens = self.action_encoder(
            action_tokens.to(
                device=encoder_weight.device,
                dtype=encoder_weight.dtype,
            )
        )
        time_embedding = self.t_embedder(
            timestep.to(device=tokens.device, dtype=torch.float32)
        ).unsqueeze(1)
        context_embedding = self.context_projection(
            pooled_context.to(
                device=self.context_projection.weight.device,
                dtype=self.context_projection.weight.dtype,
            )
        ).unsqueeze(1)
        condition = F.silu(time_embedding + context_embedding)
        action_freqs = self.temporal_rope.to(device=tokens.device)
        if self.validate_finite and not bool(
            torch.isfinite(tokens).all() and torch.isfinite(condition).all()
        ):
            raise FloatingPointError("Prepared temporal Action tokens/condition are not finite.")
        return {
            "tokens": tokens,
            "condition": condition,
            "time_embedding": time_embedding,
            "context_embedding": context_embedding,
            "context_mask": prepared_mask,
            "action_freqs": action_freqs,
            "meta": {
                "batch_size": batch_size,
                "action_horizon": self.action_horizon,
                "time_tokens": self.action_horizon,
                "token_semantics": self.TOKEN_SEMANTICS_VERSION,
                "position_encoding": self.POSITION_ENCODING_VERSION,
            },
            "next_block": 0,
        }

    def post_dit(
        self,
        tokens: torch.Tensor,
        pre_state: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        if tokens.shape[1:] != (self.action_horizon, self.hidden_size):
            raise ValueError(
                "Temporal Action decoder expects [B,T,D], got "
                f"{tuple(tokens.shape)}."
            )
        if (
            pre_state is not None
            and int(pre_state.get("next_block", -1)) != self.num_layers
        ):
            raise ValueError("Action decoder requires all transformer blocks to finish.")
        velocity = self.action_decoder(tokens).contiguous()
        if self.validate_finite and not bool(torch.isfinite(velocity).all()):
            raise FloatingPointError("Temporal Action decoder produced NaN or Inf.")
        return velocity


__all__ = [
    "COSINE_ANNEAL_NATIVE_KV_FUSION_VERSION",
    "LEGACY_NATIVE_KV_FUSION_VERSION",
    "PixelDiTAlignedActionBlock",
    "PixelDiTAlignedActionDiT",
    "PixelDiTTemporalActionDiT",
    "cosine_count_correction_scale",
    "fixed_1d_sincos_position_embedding",
    "fuse_world_action_native_kv",
    "precompute_1d_rope_frequencies",
]
