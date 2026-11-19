"""PixelDiT world expert adapted to MoMaps geometry-conditioned K-flow.

The dual-level architecture and upstream-compatible parameter names follow
NVIDIA PixelDiT c2i at commit
``41f73006ae532b0b41fee72b181dc22891a5a01a``:
https://github.com/NVlabs/PixelDiT

FastWAM's adaptation is deliberately asymmetric:

* patch/global tokens see ``RGB + segmentation + canonical XYZ + noisy offset``;
* pixel/local tokens see only packed horizon-major XYZ flow;
* the velocity head predicts the same configured ``3*K`` flow channels.

Raw depth is never a direct PixelDiT channel. RGB-only and per-view source-camera
contracts are intentionally unsupported.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mwam.utils.weights import validate_floating_state_dict

from .conditioning import WorldConditioningContract
from .modules import (
    ClassEmbedder,
    FeedForward,
    FinalLayer,
    MLP,
    RMSNorm,
    RotaryAttention,
    TimestepConditioner,
    apply_adaln,
    get_2d_sincos_pos_embed,
    precompute_freqs_cis_2d,
)


OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE = 16
FORMAL_KFLOW_PIXEL_HIDDEN_SIZE = 16
LEGACY_DPIX24_PIXEL_HIDDEN_SIZE = 24
H32S1_DPIX96_PIXEL_HIDDEN_SIZE = 96
LEGACY_LOCAL_ONLY_OUTPUT_HEAD = "local_only_v1"
MOMAP_LOCAL_GLOBAL_FUSION_OUTPUT_HEAD = "local_global_fusion_v1"
MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD = (
    "local_global_raw_flow_residual_v1"
)
FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE = 128
FULL_RANK_LOCAL_RESET_POLICY = (
    "official_dpix16_to_kflow_dpix24_reinitialize_exact_shape_mismatches_v2"
)
LOCAL_GLOBAL_FUSION_RESET_POLICY = (
    "official_pixeldit_to_kflow_p16_local_global_fusion128_v1"
)
RAW_FLOW_RESIDUAL_RESET_POLICY = (
    "official_pixeldit_to_kflow_p16_local_global_fusion128_raw24_residual_v1"
)
K1_RAW_FLOW_RESIDUAL_RESET_POLICY = (
    "official_pixeldit_to_kflow_k1_p16_local_global_fusion128_raw3_residual_v1"
)
LOCAL24_FUSION_RESET_POLICY = (
    "official_pixeldit_p16_to_kflow_p24_local_global_fusion128_v1"
)
LOCAL24_RAW_FLOW_RESIDUAL_RESET_POLICY = (
    "official_pixeldit_p16_to_kflow_p24_local_global_fusion128_"
    "raw24_residual_v1"
)
H32S1_LOCAL96_RAW_FLOW_RESIDUAL_RESET_POLICY = (
    "official_pixeldit_p16_to_kflow_h32s1_p96_local_global_fusion128_"
    "raw96_residual_v1"
)
LOCAL_GLOBAL_FUSION_INITIALIZATION = (
    "branch_xavier_fuse_xavier_final_xavier_gain_1e-2_bias_zero_v1"
)
RAW_FLOW_RESIDUAL_INITIALIZATION = (
    "fusion_gain_1e-2_plus_raw24_sigma_floor_1e-2_zero_residual_v2"
)
RAW_FLOW_RESIDUAL_NOISE_FLOOR = 1.0e-2
LOCAL_GLOBAL_FUSION_TARGET_ONLY_KEYS = (
    "final_layer.fusion_norm.weight",
    "final_layer.fusion_projection.bias",
    "final_layer.fusion_projection.weight",
    "final_layer.global_norm.weight",
    "final_layer.global_to_pixels.bias",
    "final_layer.global_to_pixels.weight",
    "final_layer.local_projection.bias",
    "final_layer.local_projection.weight",
)
LOCAL_GLOBAL_FUSION_NEW_HEAD_KEYS = tuple(
    sorted(
        {
            *LOCAL_GLOBAL_FUSION_TARGET_ONLY_KEYS,
            "final_layer.linear.bias",
            "final_layer.linear.weight",
        }
    )
)
RAW_FLOW_RESIDUAL_TARGET_ONLY_KEYS = (
    "raw_flow_residual.channel_delta.weight",
    "raw_flow_residual.timestep_modulation.bias",
    "raw_flow_residual.timestep_modulation.weight",
)
RAW_FLOW_RESIDUAL_NEW_HEAD_KEYS = tuple(
    sorted(
        {
            *LOCAL_GLOBAL_FUSION_NEW_HEAD_KEYS,
            *RAW_FLOW_RESIDUAL_TARGET_ONLY_KEYS,
        }
    )
)
LOCAL_GLOBAL_FUSION_SHAPE_REINITIALIZED_KEYS = (
    "final_layer.linear.bias",
    "final_layer.linear.weight",
    "pixel_embedder.proj.weight",
    "s_embedder.proj.weight",
)
K1_RAW_FLOW_RESIDUAL_SEMANTIC_REINITIALIZED_KEYS = (
    "final_layer.linear.bias",
    "pixel_embedder.proj.weight",
)
LOCAL_GLOBAL_FUSION_TRUNK_REINITIALIZED_KEYS = (
    "pixel_embedder.proj.weight",
    "s_embedder.proj.weight",
)
FORMAL_LOCAL_GLOBAL_FUSION_REINITIALIZED_KEYS = tuple(
    sorted(
        {
            *LOCAL_GLOBAL_FUSION_SHAPE_REINITIALIZED_KEYS,
            *LOCAL_GLOBAL_FUSION_TARGET_ONLY_KEYS,
        }
    )
)
_LOCAL_GLOBAL_FUSION_OUTPUT_HEADS = frozenset(
    {
        MOMAP_LOCAL_GLOBAL_FUSION_OUTPUT_HEAD,
        MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD,
    }
)
_PIXEL_BLOCK_LOCAL_STATE_SUFFIXES = (
    "adaLN_modulation.0.bias",
    "adaLN_modulation.0.weight",
    "compress_to_attn.weight",
    "expand_from_attn.bias",
    "expand_from_attn.weight",
    "mlp.fc1.bias",
    "mlp.fc1.weight",
    "mlp.fc2.bias",
    "mlp.fc2.weight",
    "norm1.weight",
    "norm2.weight",
)


def full_rank_local_reinitialized_keys(pixel_depth: int = 4) -> tuple[str, ...]:
    """Return exactly the tensors whose shapes change from p16 to p24.

    The local attention operates at the unchanged patch hidden width, so its
    qkv/projection/norm tensors remain shape compatible and retain upstream
    weights.  Only the joint stem/head and width-dependent local tensors reset.
    """

    if isinstance(pixel_depth, bool) or not isinstance(pixel_depth, int):
        raise TypeError("`pixel_depth` must be an integer.")
    if pixel_depth <= 0:
        raise ValueError("`pixel_depth` must be positive.")
    keys = {
        "s_embedder.proj.weight",
        "pixel_embedder.proj.bias",
        "pixel_embedder.proj.weight",
        "final_layer.norm.weight",
        "final_layer.linear.bias",
        "final_layer.linear.weight",
    }
    for layer_index in range(pixel_depth):
        keys.update(
            f"pixel_blocks.{layer_index}.{suffix}"
            for suffix in _PIXEL_BLOCK_LOCAL_STATE_SUFFIXES
        )
    return tuple(sorted(keys))


FORMAL_FULL_RANK_PIXELDIT_REINITIALIZED_KEYS = full_rank_local_reinitialized_keys(4)


class PatchTokenEmbedder(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[type[nn.Module]] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)
        self.proj = nn.Linear(self.in_chans, self.embed_dim, bias=bias)
        self.norm = (
            norm_layer(self.embed_dim) if norm_layer is not None else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class AugmentedDiTBlock(nn.Module):
    """Patch-level DiT block that can expose its native self-attention K/V."""

    def __init__(
        self,
        hidden_size: int,
        groups: int,
        mlp_ratio: float = 4.0,
        adaLN_modulation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RotaryAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = (
            adaLN_modulation
            if adaLN_modulation is not None
            else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(condition).chunk(6, dim=-1)

        attention_input = apply_adaln(self.norm1(x), shift_msa, scale_msa)
        if return_kv:
            attention_output, kv_cache = self.attn(
                attention_input,
                pos,
                mask=mask,
                return_kv=True,
            )
        else:
            attention_output = self.attn(attention_input, pos, mask=mask)
            kv_cache = None

        x = x + gate_msa * attention_output
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        if return_kv:
            assert kv_cache is not None
            return x, kv_cache
        return x


class PixelTokenEmbedder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size_output: int,
        use_pixel_abs_pos: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size_output = int(hidden_size_output)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.proj = nn.Linear(self.in_channels, self.hidden_size_output, bias=True)
        self._pos_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _fetch_pixel_pos(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (height, width)
        if key not in self._pos_cache:
            position = get_2d_sincos_pos_embed(self.hidden_size_output, height, width)
            self._pos_cache[key] = torch.from_numpy(position)
        return self._pos_cache[key].to(device=device, dtype=dtype)

    def forward(
        self,
        inputs: torch.Tensor,
        img_height: int,
        img_width: int,
        patch_size: int,
    ) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(
                "PixelTokenEmbedder expects [B,C,H,W], "
                f"got shape {tuple(inputs.shape)}."
            )
        batch_size, channels, height, width = inputs.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} pixel channels, got {channels}."
            )
        if (height, width) != (img_height, img_width):
            raise ValueError(
                f"Image shape mismatch: tensor={(height, width)}, declared={(img_height, img_width)}."
            )
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"Image {(height, width)} must be divisible by patch size {patch_size}."
            )

        patch_height = height // patch_size
        patch_width = width // patch_size
        pixels_per_patch = patch_size * patch_size
        x = inputs.permute(0, 2, 3, 1).contiguous()
        x = self.proj(x)
        if self.use_pixel_abs_pos:
            position = self._fetch_pixel_pos(height, width, x.device, x.dtype)
            x = x + position.view(height, width, self.hidden_size_output).unsqueeze(0)
        x = x.view(
            batch_size,
            patch_height,
            patch_size,
            patch_width,
            patch_size,
            self.hidden_size_output,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(
            batch_size * patch_height * patch_width,
            pixels_per_patch,
            self.hidden_size_output,
        )


class PiTBlock(nn.Module):
    """Pixel-level block with patch-token-conditioned pixel-wise AdaLN."""

    def __init__(
        self,
        pixel_hidden_size: int,
        patch_hidden_size: int,
        patch_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_hidden_size: Optional[int] = None,
        attn_num_heads: Optional[int] = None,
        adaln_post_modulation: bool = False,
    ):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)
        self.context_dim = int(patch_hidden_size)
        self.patch_size = int(patch_size)
        self.attn_dim = (
            int(attn_hidden_size) if attn_hidden_size is not None else self.context_dim
        )
        self.num_heads = (
            int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        )
        if self.attn_dim % self.num_heads:
            raise ValueError(
                f"Pixel attention dim {self.attn_dim} must divide into {self.num_heads} heads."
            )

        pixels_per_patch = self.patch_size * self.patch_size
        self.compress_to_attn = nn.Linear(
            pixels_per_patch * self.pixel_dim,
            self.attn_dim,
            bias=True,
        )
        self.expand_from_attn = nn.Linear(
            self.attn_dim,
            pixels_per_patch * self.pixel_dim,
            bias=True,
        )
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(
            self.attn_dim, num_heads=self.num_heads, qkv_bias=False
        )
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio, drop=0.0)
        self.adaln_post_modulation = bool(adaln_post_modulation)
        num_modulations = 4 if self.adaln_post_modulation else 6
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(
                self.context_dim,
                num_modulations * self.pixel_dim * pixels_per_patch,
                bias=True,
            )
        )
        self._pos_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _fetch_pos(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width)
        if key not in self._pos_cache:
            self._pos_cache[key] = precompute_freqs_cis_2d(
                self.attn_dim // self.num_heads,
                height,
                width,
            )
        return self._pos_cache[key].to(device=device)

    def forward(
        self,
        x: torch.Tensor,
        patch_condition: torch.Tensor,
        image_height: int,
        image_width: int,
        patch_size: int,
        mask: Optional[torch.Tensor] = None,
        *,
        return_compact: bool = False,
        return_native_kv: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        batch_patch_count, pixels_per_patch, channels = x.shape
        if channels != self.pixel_dim:
            raise ValueError(f"Expected pixel dim {self.pixel_dim}, got {channels}.")
        if patch_size != self.patch_size:
            raise ValueError(
                f"PiTBlock was built for patch size {self.patch_size}, got {patch_size}."
            )
        if image_height % patch_size or image_width % patch_size:
            raise ValueError("Image dimensions must be divisible by patch size.")

        patch_height = image_height // patch_size
        patch_width = image_width // patch_size
        sequence_length = patch_height * patch_width
        if batch_patch_count % sequence_length:
            raise ValueError(
                f"Cannot infer batch size from {batch_patch_count} patch groups "
                f"and sequence length {sequence_length}."
            )
        batch_size = batch_patch_count // sequence_length
        expected_pixels = patch_size * patch_size
        if pixels_per_patch != expected_pixels:
            raise ValueError(
                f"Expected {expected_pixels} pixels per patch, got {pixels_per_patch}."
            )

        num_modulations = 4 if self.adaln_post_modulation else 6
        modulation = self.adaLN_modulation(patch_condition).view(
            batch_patch_count,
            pixels_per_patch,
            num_modulations * self.pixel_dim,
        )
        if self.adaln_post_modulation:
            scale_attn, shift_attn, scale_mlp, shift_mlp = modulation.chunk(4, dim=-1)
            attention_input = self.norm1(x)
        else:
            (
                shift_attn,
                scale_attn,
                gate_attn,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = modulation.chunk(6, dim=-1)
            attention_input = apply_adaln(self.norm1(x), shift_attn, scale_attn)

        compressed = self.compress_to_attn(
            attention_input.reshape(batch_patch_count, -1)
        ).view(batch_size, sequence_length, self.attn_dim)
        position = self._fetch_pos(patch_height, patch_width, x.device)
        native_kv = None
        if return_native_kv:
            attention_result = self.attn(
                compressed,
                position,
                mask=mask,
                return_kv=True,
            )
            if not isinstance(attention_result, tuple) or len(attention_result) != 2:
                raise RuntimeError("PiT attention native K/V contract drifted.")
            attention_output, native_kv = attention_result
        else:
            attention_output = self.attn(compressed, position, mask=mask)
        expanded = self.expand_from_attn(
            attention_output.reshape(batch_patch_count, self.attn_dim)
        ).view(batch_patch_count, pixels_per_patch, self.pixel_dim)

        if self.adaln_post_modulation:
            x = x + expanded * (1 + scale_attn) + shift_attn
            mlp_output = self.mlp(self.norm2(x))
            output = x + mlp_output * (1 + scale_mlp) + shift_mlp
        else:
            x = x + gate_attn * expanded
            mlp_output = self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
            output = x + gate_mlp * mlp_output
        if return_compact and return_native_kv:
            assert native_kv is not None
            return output, attention_output, native_kv
        if return_compact:
            return output, attention_output
        if return_native_kv:
            assert native_kv is not None
            return output, native_kv
        return output


class MoMapLocalGlobalFusionHead(nn.Module):
    """Fuse aligned global patch and local pixel states into MoMap velocity.

    ``global_tokens`` use row-major patch order ``[B,L,Dg]``. ``local_tokens``
    use exactly the matching ``[B,L,pixel_row,pixel_col,Dl]`` flattening.  The
    global projection owns a distinct output slice for every intra-patch pixel,
    so expanding one patch token remains position aware without assuming an
    image height, width, or patch-grid resolution.
    """

    def __init__(
        self,
        *,
        global_hidden_size: int,
        local_hidden_size: int,
        output_hidden_size: int,
        out_channels: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        dimensions = {
            "global_hidden_size": global_hidden_size,
            "local_hidden_size": local_hidden_size,
            "output_hidden_size": output_hidden_size,
            "out_channels": out_channels,
            "patch_size": patch_size,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions.values()
        ):
            raise ValueError(
                "MoMap fusion-head dimensions must be positive integers, got "
                f"{dimensions}."
            )
        self.global_hidden_size = int(global_hidden_size)
        self.local_hidden_size = int(local_hidden_size)
        self.output_hidden_size = int(output_hidden_size)
        self.out_channels = int(out_channels)
        self.patch_size = int(patch_size)
        self.pixels_per_patch = self.patch_size**2

        # Keep the upstream ``final_layer.norm`` state-dict key loadable.
        self.norm = RMSNorm(self.local_hidden_size, eps=1e-6)
        self.global_norm = RMSNorm(self.global_hidden_size, eps=1e-6)
        self.global_to_pixels = nn.Linear(
            self.global_hidden_size,
            self.pixels_per_patch * self.output_hidden_size,
            bias=True,
        )
        self.local_projection = nn.Linear(
            self.local_hidden_size,
            self.output_hidden_size,
            bias=True,
        )
        self.fusion_norm = RMSNorm(2 * self.output_hidden_size, eps=1e-6)
        self.fusion_projection = nn.Linear(
            2 * self.output_hidden_size,
            self.output_hidden_size,
            bias=True,
        )
        self.linear = nn.Linear(
            self.output_hidden_size,
            self.out_channels,
            bias=True,
        )

    def forward(
        self,
        global_tokens: torch.Tensor,
        local_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(global_tokens, torch.Tensor) or not isinstance(
            local_tokens, torch.Tensor
        ):
            raise TypeError("Fusion-head global/local inputs must be tensors.")
        if global_tokens.ndim != 3:
            raise ValueError(
                "Fusion-head global tokens must be [B,L,Dg], got "
                f"{tuple(global_tokens.shape)}."
            )
        if local_tokens.ndim != 4:
            raise ValueError(
                "Fusion-head local tokens must be [B,L,P^2,Dl], got "
                f"{tuple(local_tokens.shape)}."
            )
        batch_size, sequence_length, global_width = global_tokens.shape
        expected_local = (
            batch_size,
            sequence_length,
            self.pixels_per_patch,
            self.local_hidden_size,
        )
        if global_width != self.global_hidden_size:
            raise ValueError(
                "Fusion-head global width mismatch: expected "
                f"{self.global_hidden_size}, got {global_width}."
            )
        if tuple(local_tokens.shape) != expected_local:
            raise ValueError(
                "Fusion-head local ordering/shape mismatch: expected "
                f"{expected_local}, got {tuple(local_tokens.shape)}."
            )
        if (
            global_tokens.device != local_tokens.device
            or global_tokens.dtype != local_tokens.dtype
        ):
            raise ValueError("Fusion-head global/local tensors must share dtype/device.")
        if not global_tokens.is_floating_point():
            raise TypeError("Fusion-head global/local tensors must be floating point.")

        global_pixels = self.global_to_pixels(
            self.global_norm(global_tokens)
        ).view(
            batch_size,
            sequence_length,
            self.pixels_per_patch,
            self.output_hidden_size,
        )
        local_pixels = self.local_projection(self.norm(local_tokens))
        fused = torch.cat([global_pixels, local_pixels], dim=-1)
        fused = F.silu(self.fusion_projection(self.fusion_norm(fused)))
        return self.linear(fused)


class MoMapRawFlowResidualPath(nn.Module):
    """Preserve all raw flow channels with solver-stable noise preconditioning.

    The V5 local/global trunk may still compress ``noised_flow`` while learning
    semantic residuals. This independent path keeps every packed KFlow channel
    at full rank and initializes its gain as ``1 / (sigma + 0.01)``. Under an
    Euler schedule this makes pure-noise residual independent of step count.
    A full 1x1 channel correction and timestep gain/shift remain trainable.
    """

    def __init__(self, *, flow_channels: int, time_hidden_size: int) -> None:
        super().__init__()
        dimensions = {
            "flow_channels": flow_channels,
            "time_hidden_size": time_hidden_size,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions.values()
        ):
            raise ValueError(
                "Raw-flow residual dimensions must be positive integers, got "
                f"{dimensions}."
            )
        self.flow_channels = int(flow_channels)
        self.time_hidden_size = int(time_hidden_size)
        self.noise_floor = RAW_FLOW_RESIDUAL_NOISE_FLOOR
        self.channel_delta = nn.Conv2d(
            self.flow_channels,
            self.flow_channels,
            kernel_size=1,
            bias=False,
        )
        self.timestep_modulation = nn.Linear(
            self.time_hidden_size,
            2 * self.flow_channels,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 全秩预条件起点：不再要求 16-D local trunk 重建 24-D 高斯噪声。
        nn.init.zeros_(self.channel_delta.weight)
        nn.init.zeros_(self.timestep_modulation.weight)
        nn.init.zeros_(self.timestep_modulation.bias)

    def forward(
        self,
        noised_flow: torch.Tensor,
        time_embedding: torch.Tensor,
        noise_fraction: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(noised_flow, torch.Tensor) or noised_flow.ndim != 4:
            raise ValueError(
                "Raw-flow residual input must be [B,C,H,W], got "
                "{}.".format(getattr(noised_flow, "shape", None))
            )
        if noised_flow.shape[1] != self.flow_channels:
            raise ValueError(
                "Raw-flow residual channel mismatch: expected "
                f"{self.flow_channels}, got {noised_flow.shape[1]}."
            )
        if not isinstance(time_embedding, torch.Tensor):
            raise TypeError("Raw-flow timestep embedding must be a tensor.")
        if time_embedding.ndim == 3 and time_embedding.shape[1] == 1:
            time_embedding = time_embedding[:, 0]
        expected_time_shape = (
            noised_flow.shape[0],
            self.time_hidden_size,
        )
        if tuple(time_embedding.shape) != expected_time_shape:
            raise ValueError(
                "Raw-flow timestep embedding must be [B,D], got "
                f"{tuple(time_embedding.shape)}; expected {expected_time_shape}."
            )
        if (
            noised_flow.device != time_embedding.device
            or noised_flow.dtype != time_embedding.dtype
        ):
            raise ValueError(
                "Raw-flow input and timestep embedding must share dtype/device."
            )
        if not noised_flow.is_floating_point():
            raise TypeError("Raw-flow residual inputs must be floating point.")
        if (
            not isinstance(noise_fraction, torch.Tensor)
            or noise_fraction.ndim != 1
            or noise_fraction.shape[0] != noised_flow.shape[0]
        ):
            raise ValueError(
                "Raw-flow noise fraction must be [B], got "
                "{}.".format(getattr(noise_fraction, "shape", None))
            )
        if noise_fraction.device != noised_flow.device:
            raise ValueError("Raw-flow noise fraction must share the input device.")
        if not noise_fraction.is_floating_point():
            raise TypeError("Raw-flow noise fraction must be floating point.")
        noise_fraction_fp32 = noise_fraction.float()
        if not bool(torch.isfinite(noise_fraction_fp32).all().item()):
            raise ValueError("Raw-flow noise fraction must be finite.")
        if bool(
            ((noise_fraction_fp32 < 0.0) | (noise_fraction_fp32 > 1.0))
            .any()
            .item()
        ):
            raise ValueError("Raw-flow noise fraction must lie in [0,1].")

        gain_delta, shift = self.timestep_modulation(
            F.silu(time_embedding)
        ).chunk(2, dim=-1)
        base_gain = (noise_fraction_fp32 + self.noise_floor).reciprocal().to(
            dtype=noised_flow.dtype
        )
        gain = base_gain[:, None, None, None] + gain_delta[:, :, None, None]
        shift = shift[:, :, None, None]
        return (
            noised_flow * gain
            + self.channel_delta(noised_flow)
            + shift
        )


class PixDiT(nn.Module):
    """MoMaps-conditioned PixelDiT velocity model for packed K-flow."""

    _ALLOWED_PRETRAINED_SHAPE_MISMATCH = frozenset(
        LOCAL_GLOBAL_FUSION_SHAPE_REINITIALIZED_KEYS
    )

    def __init__(
        self,
        in_channels: int = 12,
        num_groups: int = 16,
        hidden_size: int = 1152,
        pixel_hidden_size: int = 16,
        patch_depth: int = 26,
        pixel_depth: int = 4,
        patch_size: int = 16,
        num_classes: int = 1000,
        use_pixel_abs_pos: bool = True,
        pit_adaln_post_modulation: bool = False,
        flow_channels: int = 3,
        out_channels: int = 3,
        use_gradient_checkpointing: bool = False,
        require_full_rank_output: bool = False,
        momap_output_head_type: str = LEGACY_LOCAL_ONLY_OUTPUT_HEAD,
        momap_output_hidden_size: Optional[int] = None,
        conditioning_contract: Optional[WorldConditioningContract] = None,
    ):
        super().__init__()
        self.conditioning_contract = (
            WorldConditioningContract()
            if conditioning_contract is None
            else conditioning_contract
        )
        if not isinstance(self.conditioning_contract, WorldConditioningContract):
            raise TypeError(
                "`conditioning_contract` must be WorldConditioningContract."
            )
        self.in_channels = int(in_channels)
        self.flow_channels = int(flow_channels)
        self.out_channels = int(out_channels)
        self.clean_channels = self.in_channels - self.flow_channels
        self.rgb_channels = self.conditioning_contract.rgb_channels
        self.segmentation_channels = self.conditioning_contract.segmentation_channels
        self.canonical_xyz_channels = self.conditioning_contract.canonical_xyz_channels
        self.raw_depth_channels = self.conditioning_contract.raw_depth_channels
        self.k = self.flow_channels // 3 if self.flow_channels > 0 else 0
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.num_classes = int(num_classes)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.pit_adaln_post_modulation = bool(pit_adaln_post_modulation)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        if not isinstance(require_full_rank_output, bool):
            raise TypeError("`require_full_rank_output` must be boolean.")
        self.require_full_rank_output = require_full_rank_output
        self.momap_output_head_type = str(momap_output_head_type)
        supported_output_heads = {
            LEGACY_LOCAL_ONLY_OUTPUT_HEAD,
            MOMAP_LOCAL_GLOBAL_FUSION_OUTPUT_HEAD,
            MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD,
        }
        if self.momap_output_head_type not in supported_output_heads:
            raise ValueError(
                "Unsupported `momap_output_head_type` "
                f"{self.momap_output_head_type!r}; expected one of "
                f"{sorted(supported_output_heads)}."
            )
        if momap_output_hidden_size is None:
            resolved_output_hidden_size = (
                self.pixel_hidden_size
                if self.momap_output_head_type == LEGACY_LOCAL_ONLY_OUTPUT_HEAD
                else FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE
            )
        else:
            if (
                isinstance(momap_output_hidden_size, bool)
                or not isinstance(momap_output_hidden_size, int)
                or momap_output_hidden_size <= 0
            ):
                raise ValueError(
                    "`momap_output_hidden_size` must be a positive integer or "
                    f"None, got {momap_output_hidden_size!r}."
                )
            resolved_output_hidden_size = int(momap_output_hidden_size)
        self.momap_output_hidden_size = resolved_output_hidden_size
        if (
            self.momap_output_head_type
            == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
        ):
            self.momap_output_head_initialization = (
                RAW_FLOW_RESIDUAL_INITIALIZATION
            )
            self.raw_flow_noise_floor: float | None = (
                RAW_FLOW_RESIDUAL_NOISE_FLOOR
            )
        elif self.momap_output_head_type == MOMAP_LOCAL_GLOBAL_FUSION_OUTPUT_HEAD:
            self.momap_output_head_initialization = (
                LOCAL_GLOBAL_FUSION_INITIALIZATION
            )
            self.raw_flow_noise_floor = None
        else:
            self.momap_output_head_initialization = "legacy_final_linear_zero_v1"
            self.raw_flow_noise_floor = None
        if (
            self.momap_output_head_type == LEGACY_LOCAL_ONLY_OUTPUT_HEAD
            and self.momap_output_hidden_size != self.pixel_hidden_size
        ):
            raise ValueError(
                "Legacy local-only output requires momap_output_hidden_size == "
                f"pixel_hidden_size, got {self.momap_output_hidden_size} vs "
                f"{self.pixel_hidden_size}."
            )

        if self.flow_channels <= 0 or self.flow_channels % 3:
            raise ValueError(
                "`flow_channels` must be positive packed XYZ channels (3*K), "
                f"got {self.flow_channels}."
            )
        if self.out_channels != self.flow_channels:
            raise ValueError(
                "PixelDiT velocity channels must equal packed flow channels, "
                f"got out={self.out_channels}, flow={self.flow_channels}."
            )
        if (
            self.require_full_rank_output
            and self.momap_output_hidden_size < self.out_channels
        ):
            raise ValueError(
                "Full-rank PixelDiT output requires momap_output_hidden_size >= "
                "out_channels, got "
                f"{self.momap_output_hidden_size} < {self.out_channels}."
            )
        if self.clean_channels != self.conditioning_contract.condition_channels:
            raise ValueError(
                "PixelDiT condition channels do not match the explicit "
                f"{self.conditioning_contract.mode!r} contract: "
                f"in={self.in_channels}, flow={self.flow_channels}, "
                f"condition={self.clean_channels}, expected="
                f"{self.conditioning_contract.condition_channels}."
            )
        positive_fields = {
            "hidden_size": self.hidden_size,
            "pixel_hidden_size": self.pixel_hidden_size,
            "momap_output_hidden_size": self.momap_output_hidden_size,
            "patch_depth": self.patch_depth,
            "pixel_depth": self.pixel_depth,
            "patch_size": self.patch_size,
            "num_groups": self.num_groups,
            "num_classes": self.num_classes,
        }
        invalid_fields = {
            name: value for name, value in positive_fields.items() if value <= 0
        }
        if invalid_fields:
            raise ValueError(
                "PixelDiT dimensions/depths must be positive, got " f"{invalid_fields}."
            )
        if self.hidden_size % self.num_groups:
            raise ValueError("hidden_size must be divisible by num_groups.")
        self.head_dim = self.hidden_size // self.num_groups
        if self.head_dim % 4:
            raise ValueError(
                "PixelDiT attention head_dim must be divisible by 4 for 2-D RoPE."
            )
        if self.use_pixel_abs_pos and self.pixel_hidden_size % 4:
            raise ValueError(
                "pixel_hidden_size must be divisible by 4 when absolute positions are enabled."
            )

        self.pixel_embedder = PixelTokenEmbedder(
            self.flow_channels,
            self.pixel_hidden_size,
            use_pixel_abs_pos=self.use_pixel_abs_pos,
        )
        self.s_embedder = PatchTokenEmbedder(
            self.in_channels * self.patch_size**2,
            self.hidden_size,
            bias=True,
        )
        self.t_embedder = TimestepConditioner(self.hidden_size)
        self.y_embedder = ClassEmbedder(self.num_classes + 1, self.hidden_size)
        if self.momap_output_head_type == LEGACY_LOCAL_ONLY_OUTPUT_HEAD:
            self.final_layer: nn.Module = FinalLayer(
                self.pixel_hidden_size,
                self.out_channels,
            )
        else:
            self.final_layer = MoMapLocalGlobalFusionHead(
                global_hidden_size=self.hidden_size,
                local_hidden_size=self.pixel_hidden_size,
                output_hidden_size=self.momap_output_hidden_size,
                out_channels=self.out_channels,
                patch_size=self.patch_size,
            )
        self.raw_flow_residual: MoMapRawFlowResidualPath | None
        if (
            self.momap_output_head_type
            == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
        ):
            self.raw_flow_residual = MoMapRawFlowResidualPath(
                flow_channels=self.flow_channels,
                time_hidden_size=self.hidden_size,
            )
        else:
            self.raw_flow_residual = None
        self.patch_blocks = nn.ModuleList(
            [
                AugmentedDiTBlock(self.hidden_size, self.num_groups)
                for _ in range(self.patch_depth)
            ]
        )
        self.pixel_blocks = nn.ModuleList(
            [
                PiTBlock(
                    self.pixel_hidden_size,
                    self.hidden_size,
                    patch_size=self.patch_size,
                    num_heads=self.num_groups,
                    mlp_ratio=4.0,
                    adaln_post_modulation=self.pit_adaln_post_modulation,
                )
                for _ in range(self.pixel_depth)
            ]
        )
        self._pos_cache: dict[tuple[int, int], torch.Tensor] = {}
        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.s_embedder.proj.weight)
        nn.init.zeros_(self.s_embedder.proj.bias)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.momap_output_head_type in _LOCAL_GLOBAL_FUSION_OUTPUT_HEADS:
            if not isinstance(self.final_layer, MoMapLocalGlobalFusionHead):
                raise RuntimeError("Fusion output-head module/type binding drifted.")
            nn.init.xavier_uniform_(self.final_layer.global_to_pixels.weight)
            nn.init.zeros_(self.final_layer.global_to_pixels.bias)
            nn.init.xavier_uniform_(self.final_layer.local_projection.weight)
            nn.init.zeros_(self.final_layer.local_projection.bias)
            nn.init.xavier_uniform_(self.final_layer.fusion_projection.weight)
            nn.init.zeros_(self.final_layer.fusion_projection.bias)
            # Near-zero preserves a gentle pretrained start while remaining a
            # genuine two-input function with gradients through both branches.
            nn.init.xavier_uniform_(self.final_layer.linear.weight, gain=1.0e-2)
        else:
            nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        if self.raw_flow_residual is not None:
            self.raw_flow_residual.reset_parameters()
        for block in self.patch_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)
        for block in self.pixel_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)

    def _fetch_pos(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width)
        if key not in self._pos_cache:
            self._pos_cache[key] = precompute_freqs_cis_2d(
                self.head_dim,
                height,
                width,
            )
        return self._pos_cache[key].to(device=device)

    def _prepare_timestep(
        self,
        timestep: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        timestep = torch.as_tensor(timestep, device=device).reshape(-1)
        if timestep.numel() == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if timestep.numel() != batch_size:
            raise ValueError(
                f"Expected one timestep per sample ({batch_size}), got {timestep.numel()}."
            )
        if not timestep.is_floating_point():
            timestep = timestep.float()
        if not torch.isfinite(timestep).all():
            raise ValueError("PixelDiT timestep must contain only finite values.")
        if bool(((timestep < 0) | (timestep > 1)).any().item()):
            raise ValueError("PixelDiT timestep must lie in [0, 1].")
        return timestep

    def _prepare_class_labels(
        self,
        class_labels: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if class_labels is None:
            return torch.full(
                (batch_size,),
                self.num_classes,
                device=device,
                dtype=torch.long,
            )
        labels = torch.as_tensor(class_labels, device=device, dtype=torch.long).reshape(
            -1
        )
        if labels.numel() == 1 and batch_size > 1:
            labels = labels.expand(batch_size)
        if labels.numel() != batch_size:
            raise ValueError(
                f"Expected one class label per sample ({batch_size}), got {labels.numel()}."
            )
        if torch.any(labels < 0) or torch.any(labels > self.num_classes):
            raise ValueError(
                f"Class labels must be in [0, {self.num_classes}], including the null class."
            )
        return labels

    def _validate_inputs(
        self,
        clean_image: torch.Tensor,
        noised_flow: torch.Tensor,
    ) -> tuple[int, int, int]:
        if clean_image.ndim != 4 or noised_flow.ndim != 4:
            raise ValueError(
                "clean_image and noised_flow must both have shape [B,C,H,W], got "
                f"{tuple(clean_image.shape)} and {tuple(noised_flow.shape)}."
            )
        if (
            clean_image.shape[0] != noised_flow.shape[0]
            or clean_image.shape[2:] != noised_flow.shape[2:]
        ):
            raise ValueError(
                "clean_image and noised_flow must share batch/spatial dimensions, got "
                f"{tuple(clean_image.shape)} and {tuple(noised_flow.shape)}."
            )
        if clean_image.shape[1] != self.clean_channels:
            raise ValueError(
                f"Expected {self.clean_channels} clean-image channels, got {clean_image.shape[1]}."
            )
        if noised_flow.shape[1] != self.flow_channels:
            raise ValueError(
                f"Expected {self.flow_channels} noised-flow channels, got {noised_flow.shape[1]}."
            )
        if (
            clean_image.device != noised_flow.device
            or clean_image.dtype != noised_flow.dtype
        ):
            raise ValueError("clean_image and noised_flow must share device and dtype.")
        if not clean_image.is_floating_point():
            raise TypeError(
                "clean_image and noised_flow must be floating-point tensors, "
                f"got {clean_image.dtype}."
            )
        batch_size, _, height, width = noised_flow.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input {(height, width)} must be divisible by patch size {self.patch_size}."
            )
        return batch_size, height, width

    def patchify(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convert ``[B,C,H,W]`` to channel-major non-overlapping patches."""

        if not isinstance(inputs, torch.Tensor):
            raise TypeError("`inputs` must be a torch.Tensor.")
        if inputs.ndim != 4:
            raise ValueError(
                f"`inputs` must be [B,C,H,W], got {tuple(inputs.shape)}."
            )
        if not inputs.is_floating_point():
            raise TypeError(
                f"`inputs` must be floating point, got {inputs.dtype}."
            )
        _, _, height, width = inputs.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input {(height, width)} must be divisible by patch size "
                f"{self.patch_size}."
            )
        return F.unfold(
            inputs,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)

    def unpatchify(
        self,
        patches: torch.Tensor,
        *,
        output_size: tuple[int, int],
        channels: int,
    ) -> torch.Tensor:
        """Inverse of :meth:`patchify` for non-overlapping channel-major patches."""

        if not isinstance(patches, torch.Tensor):
            raise TypeError("`patches` must be a torch.Tensor.")
        if patches.ndim != 3:
            raise ValueError(
                f"`patches` must be [B,N,C*P*P], got {tuple(patches.shape)}."
            )
        if not patches.is_floating_point():
            raise TypeError(
                f"`patches` must be floating point, got {patches.dtype}."
            )
        if (
            not isinstance(output_size, tuple)
            or len(output_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in output_size
            )
        ):
            raise ValueError(
                f"`output_size` must be two positive integers, got {output_size!r}."
            )
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError(f"`channels` must be a positive integer, got {channels!r}.")
        height, width = output_size
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Output {(height, width)} must be divisible by patch size "
                f"{self.patch_size}."
            )
        expected_sequence = (height // self.patch_size) * (
            width // self.patch_size
        )
        expected_features = channels * self.patch_size**2
        if patches.shape[1:] != (expected_sequence, expected_features):
            raise ValueError(
                "`patches` shape disagrees with output geometry: expected "
                f"[B,{expected_sequence},{expected_features}], got "
                f"{tuple(patches.shape)}."
            )
        return F.fold(
            patches.transpose(1, 2),
            output_size=output_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def _prepare_source_support_mask(
        self,
        source_support_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if source_support_mask is None:
            return None, None
        if not isinstance(source_support_mask, torch.Tensor):
            raise TypeError("`source_support_mask` must be a torch.Tensor.")
        if source_support_mask.ndim == 3:
            source_support_mask = source_support_mask.unsqueeze(1)
        if source_support_mask.shape != (batch_size, 1, height, width):
            raise ValueError(
                "`source_support_mask` must be [B,1,H,W] on the source grid, "
                f"got {tuple(source_support_mask.shape)}."
            )
        if source_support_mask.dtype != torch.bool:
            raise TypeError("`source_support_mask` must use torch.bool.")
        source_support_mask = source_support_mask.to(device=device)
        patch_valid = F.max_pool2d(
            source_support_mask.float(),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).to(dtype=torch.bool)
        patch_valid = patch_valid.flatten(1)
        if not bool(patch_valid.any(dim=1).all().item()):
            raise ValueError(
                "Every sample must expose at least one source-supported patch."
            )
        # SDPA boolean masks use True for allowed positions.  All patch queries
        # may attend, but only source-observable patch keys are exposed.
        attention_mask = patch_valid[:, None, None, :]
        return patch_valid, attention_mask

    def prepare_tokens(
        self,
        clean_image: torch.Tensor,
        noised_flow: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        *,
        source_support_mask: Optional[torch.Tensor] = None,
        extra_condition: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Prepare the single live PixelDiT token state for lockstep execution."""

        batch_size, height, width = self._validate_inputs(clean_image, noised_flow)
        timestep = self._prepare_timestep(timestep, batch_size, noised_flow.device)
        labels = self._prepare_class_labels(
            class_labels,
            batch_size,
            noised_flow.device,
        )
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        position = self._fetch_pos(patch_height, patch_width, noised_flow.device)
        patch_key_valid, patch_attention_mask = self._prepare_source_support_mask(
            source_support_mask,
            batch_size=batch_size,
            height=height,
            width=width,
            device=noised_flow.device,
        )

        joint_input = torch.cat([clean_image, noised_flow], dim=1)
        patches = self.patchify(joint_input)
        patch_tokens = self.s_embedder(patches)
        time_embedding = self.t_embedder(timestep).view(
            batch_size,
            1,
            self.hidden_size,
        )
        class_embedding = self.y_embedder(labels).view(
            batch_size,
            1,
            self.hidden_size,
        )
        if extra_condition is None:
            extra = torch.zeros_like(time_embedding)
        else:
            if (
                not isinstance(extra_condition, torch.Tensor)
                or extra_condition.shape != (batch_size, self.hidden_size)
            ):
                raise ValueError(
                    "`extra_condition` must be [B,hidden_size], got "
                    f"{getattr(extra_condition, 'shape', None)}."
                )
            if not extra_condition.is_floating_point() or not bool(
                torch.isfinite(extra_condition).all().item()
            ):
                raise ValueError("`extra_condition` must be finite floating point.")
            extra = extra_condition.to(noised_flow).unsqueeze(1)
        condition = F.silu(time_embedding + class_embedding + extra)
        return {
            "patch_tokens": patch_tokens,
            "condition": condition,
            "position": position,
            "time_embedding": time_embedding,
            # 调度标量保持 FP32。若先转 BF16，末端很小的 sigma 可能量化为 0，
            # 从而把预条件增益错误地推到 noise-floor 上限。
            "noise_fraction": 1.0 - timestep.float(),
            "noised_flow": noised_flow,
            "batch_size": batch_size,
            "height": height,
            "width": width,
            "patch_key_valid": patch_key_valid,
            "patch_attention_mask": patch_attention_mask,
            "next_patch_block": 0,
        }

    def forward_patch_block(
        self,
        layer_index: int,
        state: dict[str, Any],
        *,
        return_native_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Advance one Global block and optionally expose its native K/V."""

        if layer_index < 0 or layer_index >= self.patch_depth:
            raise IndexError(
                f"PixelDiT patch layer index {layer_index} is outside "
                f"[0,{self.patch_depth})."
            )
        expected_index = int(state.get("next_patch_block", -1))
        if layer_index != expected_index:
            raise ValueError(
                "PixelDiT patch blocks must execute once in order: "
                f"expected {expected_index}, got {layer_index}."
            )
        block = self.patch_blocks[layer_index]
        patch_tokens = state["patch_tokens"]
        condition = state["condition"]
        position = state["position"]
        attention_mask = state["patch_attention_mask"]

        def _run(
            tokens: torch.Tensor,
            block_condition: torch.Tensor,
            block_position: torch.Tensor,
        ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
            return block(
                tokens,
                block_condition,
                block_position,
                mask=attention_mask,
                return_kv=return_native_kv,
            )

        if self.use_gradient_checkpointing and self.training:
            block_result = checkpoint(
                _run,
                patch_tokens,
                condition,
                position,
                use_reentrant=False,
            )
        else:
            block_result = _run(patch_tokens, condition, position)
        native_kv = None
        if return_native_kv:
            if not isinstance(block_result, tuple) or len(block_result) != 2:
                raise RuntimeError("Global block native K/V return contract drifted.")
            patch_tokens, native_kv = block_result
            if not isinstance(native_kv, dict) or set(native_kv) != {"k", "v"}:
                raise RuntimeError("Global native K/V must contain exactly k and v.")
            expected_shape = (
                int(state["batch_size"]),
                int(patch_tokens.shape[1]),
                block.attn.num_heads,
                block.attn.head_dim,
            )
            for name in ("k", "v"):
                value = native_kv[name]
                if value.shape != expected_shape:
                    raise RuntimeError(
                        f"Global native {name.upper()} has invalid shape: "
                        f"{tuple(value.shape)} vs {expected_shape}."
                    )
        else:
            if not isinstance(block_result, torch.Tensor):
                raise RuntimeError("Global block tensor return contract drifted.")
            patch_tokens = block_result
        state["patch_tokens"] = patch_tokens
        state["next_patch_block"] = layer_index + 1
        if return_native_kv:
            assert native_kv is not None
            return patch_tokens, native_kv
        return patch_tokens

    def finish_patch_blocks(self, state: dict[str, Any]) -> torch.Tensor:
        """Finalize patch tokens into per-patch conditioning for pixel modules."""

        completed = int(state.get("next_patch_block", -1))
        if completed != self.patch_depth:
            raise ValueError(
                f"PixelDiT requires {self.patch_depth} patch blocks before "
                f"pixel modules, completed {completed}."
            )
        patch_tokens = F.silu(state["time_embedding"] + state["patch_tokens"])
        sequence_length = int(patch_tokens.shape[1])
        patch_condition = patch_tokens.reshape(
            int(state["batch_size"]) * sequence_length,
            self.hidden_size,
        )
        state["finished_patch_tokens"] = patch_tokens
        state["sequence_length"] = sequence_length
        state["patch_condition"] = patch_condition
        return patch_condition

    def prepare_pixel_tokens(self, state: dict[str, Any]) -> torch.Tensor:
        """Embed noisy KFlow and initialize the ordered local-block state."""

        if "patch_condition" not in state:
            raise ValueError("Call `finish_patch_blocks` before pixel modules.")
        if "pixel_tokens" in state or "next_pixel_block" in state:
            raise ValueError(
                "PixelDiT pixel tokens are already prepared for this state."
            )
        height = int(state["height"])
        width = int(state["width"])
        pixel_tokens = self.pixel_embedder(
            state["noised_flow"],
            img_height=height,
            img_width=width,
            patch_size=self.patch_size,
        )
        state["pixel_tokens"] = pixel_tokens
        state["next_pixel_block"] = 0
        return pixel_tokens

    def forward_pixel_block(
        self,
        layer_index: int,
        state: dict[str, Any],
        *,
        return_native_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Advance one local block and optionally expose its native attention K/V."""

        if layer_index < 0 or layer_index >= self.pixel_depth:
            raise IndexError(
                f"PixelDiT pixel layer index {layer_index} is outside "
                f"[0,{self.pixel_depth})."
            )
        if "pixel_tokens" not in state:
            raise ValueError("Call `prepare_pixel_tokens` before pixel blocks.")
        expected_index = int(state.get("next_pixel_block", -1))
        if layer_index != expected_index:
            raise ValueError(
                "PixelDiT pixel blocks must execute once in order: "
                f"expected {expected_index}, got {layer_index}."
            )
        block = self.pixel_blocks[layer_index]
        result = block(
            state["pixel_tokens"],
            state["patch_condition"],
            int(state["height"]),
            int(state["width"]),
            self.patch_size,
            mask=state["patch_attention_mask"],
            return_compact=True,
            return_native_kv=return_native_kv,
        )
        expected_result_length = 3 if return_native_kv else 2
        if not isinstance(result, tuple) or len(result) != expected_result_length:
            raise RuntimeError("PiTBlock compact return contract drifted.")
        pixel_tokens, compact = result[:2]
        expected_compact_shape = (
            int(state["batch_size"]),
            int(state["sequence_length"]),
            block.attn_dim,
        )
        if compact.shape != expected_compact_shape:
            raise RuntimeError(
                "PiTBlock returned invalid compact tokens: "
                f"{tuple(compact.shape)} vs {expected_compact_shape}."
            )
        state["pixel_tokens"] = pixel_tokens
        state["next_pixel_block"] = layer_index + 1
        if return_native_kv:
            native_kv = result[2]
            if not isinstance(native_kv, dict) or set(native_kv) != {"k", "v"}:
                raise RuntimeError("PiTBlock native K/V must contain exactly k and v.")
            expected_native_shape = (
                int(state["batch_size"]),
                int(state["sequence_length"]),
                block.num_heads,
                block.attn_dim // block.num_heads,
            )
            for name in ("k", "v"):
                value = native_kv[name]
                if value.shape != expected_native_shape:
                    raise RuntimeError(
                        f"PiTBlock native {name.upper()} has invalid shape: "
                        f"{tuple(value.shape)} vs {expected_native_shape}."
                    )
                if value.device != compact.device or value.dtype != compact.dtype:
                    raise RuntimeError(
                        f"PiTBlock native {name.upper()} must share compact "
                        "dtype/device."
                    )
            return compact, native_kv
        return compact

    def forward_pixel_modules(self, state: dict[str, Any]) -> torch.Tensor:
        """Run all local pixel modules through the ordered compact API."""

        self.prepare_pixel_tokens(state)
        for layer_index in range(self.pixel_depth):
            self.forward_pixel_block(layer_index, state)
        return state["pixel_tokens"]

    def final_head(self, state: dict[str, Any]) -> torch.Tensor:
        """Fuse/project aligned tokens and fold packed ``[B,3K,H,W]``."""

        if "pixel_tokens" not in state:
            raise ValueError("Call `forward_pixel_modules` before `final_head`.")
        if "finished_patch_tokens" not in state:
            raise ValueError("Call `finish_patch_blocks` before `final_head`.")
        batch_size = int(state["batch_size"])
        height = int(state["height"])
        width = int(state["width"])
        sequence_length = int(state["sequence_length"])
        pixels_per_patch = self.patch_size * self.patch_size
        local_tokens = state["pixel_tokens"].reshape(
            batch_size,
            sequence_length,
            pixels_per_patch,
            self.pixel_hidden_size,
        )
        if self.momap_output_head_type in _LOCAL_GLOBAL_FUSION_OUTPUT_HEADS:
            if not isinstance(self.final_layer, MoMapLocalGlobalFusionHead):
                raise RuntimeError("Fusion output-head module/type binding drifted.")
            global_tokens = state["finished_patch_tokens"]
            pixel_velocity = self.final_layer(global_tokens, local_tokens)
        else:
            if not isinstance(self.final_layer, FinalLayer):
                raise RuntimeError("Legacy output-head module/type binding drifted.")
            pixel_velocity = self.final_layer(local_tokens)
        if pixel_velocity.shape != (
            batch_size,
            sequence_length,
            pixels_per_patch,
            self.out_channels,
        ):
            raise RuntimeError(
                "PixelDiT output head returned an invalid aligned shape: "
                f"{tuple(pixel_velocity.shape)}."
            )
        velocity_patches = pixel_velocity.permute(0, 1, 3, 2).reshape(
            batch_size,
            sequence_length,
            self.out_channels * pixels_per_patch,
        )
        velocity = self.unpatchify(
            velocity_patches,
            output_size=(height, width),
            channels=self.out_channels,
        )
        if self.raw_flow_residual is not None:
            velocity = velocity + self.raw_flow_residual(
                state["noised_flow"],
                state["time_embedding"],
                state["noise_fraction"],
            )
        state["velocity"] = velocity
        return velocity

    def forward(
        self,
        clean_image: torch.Tensor,
        noised_flow: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        *,
        source_support_mask: Optional[torch.Tensor] = None,
        extra_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict packed K-flow velocity without retaining layer-wise caches."""

        state = self.prepare_tokens(
            clean_image,
            noised_flow,
            timestep,
            class_labels,
            source_support_mask=source_support_mask,
            extra_condition=extra_condition,
        )
        for layer_index in range(self.patch_depth):
            self.forward_patch_block(layer_index, state)
        self.finish_patch_blocks(state)
        self.forward_pixel_modules(state)
        return self.final_head(state)

    @staticmethod
    def _read_checkpoint(
        checkpoint: str | os.PathLike[str] | Mapping[str, Any],
        map_location: str | torch.device,
    ) -> tuple[Mapping[str, Any], str, dict[str, Any] | None]:
        if isinstance(checkpoint, (str, os.PathLike)):
            resolved = Path(checkpoint).expanduser().resolve(strict=True)
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"PixelDiT checkpoint is not a regular file: {resolved}"
                )
            source_name = str(resolved)
            checkpoint_file = {
                "path": source_name,
                "size_bytes": int(resolved.stat().st_size),
            }
            payload = torch.load(
                source_name,
                map_location=map_location,
                weights_only=True,
                mmap=True,
            )
        elif isinstance(checkpoint, Mapping):
            payload = checkpoint
            source_name = "<mapping>"
            checkpoint_file = None
        else:
            raise TypeError(
                "`checkpoint` must be a path or mapping, "
                f"got {type(checkpoint).__name__}."
            )
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"Checkpoint payload must be a mapping, got {type(payload).__name__}."
            )
        state_dict = payload.get("state_dict", payload)
        if not isinstance(state_dict, Mapping):
            raise TypeError("Checkpoint `state_dict` must be a mapping.")
        return state_dict, source_name, checkpoint_file

    @staticmethod
    def _select_denoiser_state(
        state_dict: Mapping[str, Any],
        weight_source: str,
    ) -> tuple[dict[str, torch.Tensor], str]:
        if weight_source not in {"auto", "ema", "denoiser"}:
            raise ValueError("weight_source must be one of: auto, ema, denoiser.")
        prefixes = {
            "auto": ("ema_denoiser.", "denoiser."),
            "ema": ("ema_denoiser.",),
            "denoiser": ("denoiser.",),
        }[weight_source]
        for prefix in prefixes:
            selected = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if isinstance(key, str) and key.startswith(prefix)
            }
            if selected:
                if not all(
                    isinstance(value, torch.Tensor) for value in selected.values()
                ):
                    raise TypeError(
                        f"All values under prefix {prefix!r} must be tensors."
                    )
                return selected, prefix

        if weight_source == "auto" and all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state_dict.items()
        ):
            return dict(state_dict), ""
        available = sorted(
            {
                key.split(".", 1)[0]
                for key in state_dict
                if isinstance(key, str) and "." in key
            }
        )
        raise ValueError(
            f"Could not find requested {weight_source!r} PixelDiT weights; "
            f"available top-level prefixes include {available}."
        )

    def _full_rank_pretrained_shape_contract(
        self,
    ) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
        """Freeze exact official-p16 and KFlow-p24 shapes for every reset."""

        source_pixel_width = OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE
        pixels_per_patch = self.patch_size**2
        num_modulations = 4 if self.pit_adaln_post_modulation else 6
        source_shapes: dict[str, tuple[int, ...]] = {
            "s_embedder.proj.weight": (
                self.hidden_size,
                3 * pixels_per_patch,
            ),
            "pixel_embedder.proj.weight": (source_pixel_width, 3),
            "pixel_embedder.proj.bias": (source_pixel_width,),
            "final_layer.norm.weight": (source_pixel_width,),
            "final_layer.linear.weight": (3, source_pixel_width),
            "final_layer.linear.bias": (3,),
        }
        source_mlp_width = 4 * source_pixel_width
        source_modulation_width = (
            num_modulations * source_pixel_width * pixels_per_patch
        )
        for layer_index in range(self.pixel_depth):
            prefix = f"pixel_blocks.{layer_index}"
            source_shapes.update(
                {
                    f"{prefix}.compress_to_attn.weight": (
                        self.hidden_size,
                        pixels_per_patch * source_pixel_width,
                    ),
                    f"{prefix}.expand_from_attn.weight": (
                        pixels_per_patch * source_pixel_width,
                        self.hidden_size,
                    ),
                    f"{prefix}.expand_from_attn.bias": (
                        pixels_per_patch * source_pixel_width,
                    ),
                    f"{prefix}.norm1.weight": (source_pixel_width,),
                    f"{prefix}.norm2.weight": (source_pixel_width,),
                    f"{prefix}.mlp.fc1.weight": (
                        source_mlp_width,
                        source_pixel_width,
                    ),
                    f"{prefix}.mlp.fc1.bias": (source_mlp_width,),
                    f"{prefix}.mlp.fc2.weight": (
                        source_pixel_width,
                        source_mlp_width,
                    ),
                    f"{prefix}.mlp.fc2.bias": (source_pixel_width,),
                    f"{prefix}.adaLN_modulation.0.weight": (
                        source_modulation_width,
                        self.hidden_size,
                    ),
                    f"{prefix}.adaLN_modulation.0.bias": (
                        source_modulation_width,
                    ),
                }
            )

        expected_keys = set(full_rank_local_reinitialized_keys(self.pixel_depth))
        if set(source_shapes) != expected_keys:
            raise RuntimeError(
                "Internal PixelDiT full-rank shape contract key drift: "
                f"expected={sorted(expected_keys)}, observed={sorted(source_shapes)}."
            )
        target_state = self.state_dict()
        return {
            key: (source_shapes[key], tuple(target_state[key].shape))
            for key in sorted(expected_keys)
        }

    def load_pretrained(
        self,
        checkpoint: str | os.PathLike[str] | Mapping[str, Any],
        *,
        weight_source: str = "auto",
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Strictly load upstream weights with explicit condition/head reset.

        V5 keeps the upstream p16 trunks, resets the two expanded input stems,
        and initializes only the new fusion-head parameters.  Legacy same-width
        adaptation and V4's exact p16-to-p24 reset remain selectable.  Every
        skipped/target-only tensor is reported; all other drift fails closed.
        """
        raw_state, source_name, checkpoint_file = self._read_checkpoint(
            checkpoint,
            map_location,
        )
        source_state, prefix = self._select_denoiser_state(raw_state, weight_source)
        source_integrity = validate_floating_state_dict(
            source_state,
            name="PixelDiT selected state_dict",
        )
        target_state = self.state_dict()
        source_keys = set(source_state)
        target_keys = set(target_state)

        source_pixel_projection = source_state.get("pixel_embedder.proj.weight")
        source_output_bias = source_state.get("final_layer.linear.bias")
        if (
            not isinstance(source_pixel_projection, torch.Tensor)
            or source_pixel_projection.ndim != 2
            or not isinstance(source_output_bias, torch.Tensor)
            or source_output_bias.ndim != 1
        ):
            raise RuntimeError(
                "PixelDiT checkpoint lacks valid local input/output projections."
            )
        source_pixel_hidden_size = int(source_pixel_projection.shape[0])
        source_flow_channels = int(source_pixel_projection.shape[1])
        source_output_channels = int(source_output_bias.shape[0])
        fusion_same_width_pretrained_adaptation = (
            self.momap_output_head_type in _LOCAL_GLOBAL_FUSION_OUTPUT_HEADS
            and source_pixel_hidden_size == OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE
            and self.pixel_hidden_size == OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE
            and source_flow_channels == 3
            and self.flow_channels == 24
            and source_output_channels == 3
            and self.out_channels == 24
            and self.momap_output_hidden_size == FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE
        )
        fusion_local24_pretrained_adaptation = (
            self.momap_output_head_type in _LOCAL_GLOBAL_FUSION_OUTPUT_HEADS
            and source_pixel_hidden_size == OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE
            and self.pixel_hidden_size == LEGACY_DPIX24_PIXEL_HIDDEN_SIZE
            and source_flow_channels == 3
            and self.flow_channels == 24
            and source_output_channels == 3
            and self.out_channels == 24
            and self.momap_output_hidden_size == FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE
        )
        fusion_local24_raw_flow_pretrained_adaptation = (
            fusion_local24_pretrained_adaptation
            and self.momap_output_head_type
            == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
        )
        fusion_h32s1_local96_raw_flow_pretrained_adaptation = (
            self.momap_output_head_type
            == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
            and source_pixel_hidden_size == OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE
            and self.pixel_hidden_size == H32S1_DPIX96_PIXEL_HIDDEN_SIZE
            and source_flow_channels == 3
            and self.flow_channels == 96
            and source_output_channels == 3
            and self.out_channels == 96
            and self.momap_output_hidden_size == FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE
        )
        fusion_full_rank_pretrained_adaptation = (
            fusion_local24_pretrained_adaptation
            or fusion_h32s1_local96_raw_flow_pretrained_adaptation
        )
        fusion_k1_raw3_pretrained_adaptation = (
            self.momap_output_head_type
            == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
            and source_pixel_hidden_size == OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE
            and self.pixel_hidden_size == OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE
            and source_flow_channels == 3
            and self.flow_channels == 3
            and source_output_channels == 3
            and self.out_channels == 3
            and self.momap_output_hidden_size == FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE
        )
        fusion_pretrained_adaptation = (
            fusion_same_width_pretrained_adaptation
            or fusion_local24_pretrained_adaptation
            or fusion_h32s1_local96_raw_flow_pretrained_adaptation
            or fusion_k1_raw3_pretrained_adaptation
        )
        if (
            self.momap_output_head_type in _LOCAL_GLOBAL_FUSION_OUTPUT_HEADS
            and not fusion_pretrained_adaptation
        ):
            raise RuntimeError(
                "Fusion-head PixelDiT only accepts the versioned official "
                "p16/3-channel checkpoint adaptation."
            )

        expected_target_only = set()
        if fusion_pretrained_adaptation:
            expected_target_only.update(LOCAL_GLOBAL_FUSION_TARGET_ONLY_KEYS)
            if (
                self.momap_output_head_type
                == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
            ):
                expected_target_only.update(RAW_FLOW_RESIDUAL_TARGET_ONLY_KEYS)
        missing_from_source = set(target_keys - source_keys)
        unexpected_in_source = set(source_keys - target_keys)
        if missing_from_source != expected_target_only or unexpected_in_source:
            raise RuntimeError(
                "PixelDiT checkpoint key mismatch before loading: "
                f"missing={sorted(missing_from_source)}, "
                f"unexpected={sorted(unexpected_in_source)}, "
                f"expected_target_only={sorted(expected_target_only)}."
            )

        shape_mismatches = {
            key: {
                "checkpoint_shape": tuple(source_state[key].shape),
                "model_shape": tuple(target_state[key].shape),
            }
            for key in sorted(target_keys & source_keys)
            if source_state[key].shape != target_state[key].shape
        }
        width_changed = source_pixel_hidden_size != self.pixel_hidden_size
        full_rank_width_expansion = (
            self.momap_output_head_type == LEGACY_LOCAL_ONLY_OUTPUT_HEAD
            and self.require_full_rank_output
            and source_pixel_hidden_size == OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE
            and self.pixel_hidden_size == LEGACY_DPIX24_PIXEL_HIDDEN_SIZE
            and source_flow_channels == 3
            and self.flow_channels == 24
            and source_output_channels == 3
            and self.out_channels == 24
        )
        if width_changed and not (
            full_rank_width_expansion or fusion_full_rank_pretrained_adaptation
        ):
            raise RuntimeError(
                "Unsupported PixelDiT local-width checkpoint adaptation: "
                f"pixel_hidden_size {source_pixel_hidden_size} -> "
                f"{self.pixel_hidden_size}. Only the versioned official p16/3ch "
                "to full-rank KFlow p24/24ch policy is allowed."
            )

        if fusion_full_rank_pretrained_adaptation:
            expected_shapes = self._full_rank_pretrained_shape_contract()
            shape_reset_keys = set(expected_shapes)
            missing_reset_keys = sorted(
                (shape_reset_keys - source_keys) | (shape_reset_keys - target_keys)
            )
            if missing_reset_keys:
                raise RuntimeError(
                    "PixelDiT Local24 fusion reset contract references missing keys: "
                    f"{missing_reset_keys}."
                )
            observed_mismatches = set(shape_mismatches)
            if observed_mismatches != shape_reset_keys:
                raise RuntimeError(
                    "PixelDiT Local24 fusion reset set does not exactly match the "
                    "versioned p16-to-p24 shape contract: "
                    f"missing={sorted(shape_reset_keys - observed_mismatches)}, "
                    f"unexpected={sorted(observed_mismatches - shape_reset_keys)}."
                )
            for key, (expected_source, expected_target) in expected_shapes.items():
                source_shape = tuple(source_state[key].shape)
                target_shape = tuple(target_state[key].shape)
                if source_shape != expected_source or target_shape != expected_target:
                    raise RuntimeError(
                        f"PixelDiT Local24 fusion reset shape drift for {key}: "
                        f"{source_shape} -> {target_shape}; expected "
                        f"{expected_source} -> {expected_target}."
                    )
            target_only_keys = expected_target_only
            reinitialized_keys = shape_reset_keys | target_only_keys
            reset_policy = (
                H32S1_LOCAL96_RAW_FLOW_RESIDUAL_RESET_POLICY
                if fusion_h32s1_local96_raw_flow_pretrained_adaptation
                else (
                    LOCAL24_RAW_FLOW_RESIDUAL_RESET_POLICY
                    if fusion_local24_raw_flow_pretrained_adaptation
                    else LOCAL24_FUSION_RESET_POLICY
                )
            )
        elif fusion_k1_raw3_pretrained_adaptation:
            expected_shape_mismatches = {
                "final_layer.linear.weight",
                "s_embedder.proj.weight",
            }
            observed_mismatches = set(shape_mismatches)
            if observed_mismatches != expected_shape_mismatches:
                raise RuntimeError(
                    "PixelDiT K1 fusion shape mismatch set drift: "
                    f"missing={sorted(expected_shape_mismatches - observed_mismatches)}, "
                    f"unexpected={sorted(observed_mismatches - expected_shape_mismatches)}."
                )
            semantic_reset_keys = set(
                K1_RAW_FLOW_RESIDUAL_SEMANTIC_REINITIALIZED_KEYS
            )
            shape_reset_keys = expected_shape_mismatches | semantic_reset_keys
            missing_reset_keys = sorted(
                (shape_reset_keys - source_keys) | (shape_reset_keys - target_keys)
            )
            if missing_reset_keys:
                raise RuntimeError(
                    "PixelDiT K1 fusion reset contract references missing keys: "
                    f"{missing_reset_keys}."
                )
            expected_reinitialized_shapes = {
                "s_embedder.proj.weight": (
                    (self.hidden_size, 3 * self.patch_size**2),
                    (self.hidden_size, self.in_channels * self.patch_size**2),
                ),
                "pixel_embedder.proj.weight": (
                    (OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE, 3),
                    (self.pixel_hidden_size, self.flow_channels),
                ),
                "final_layer.linear.weight": (
                    (3, OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE),
                    (self.out_channels, self.momap_output_hidden_size),
                ),
                "final_layer.linear.bias": ((3,), (self.out_channels,)),
            }
            for key, (expected_source, expected_target) in (
                expected_reinitialized_shapes.items()
            ):
                source_shape = tuple(source_state[key].shape)
                target_shape = tuple(target_state[key].shape)
                if source_shape != expected_source or target_shape != expected_target:
                    raise RuntimeError(
                        f"PixelDiT K1 fusion reset shape drift for {key}: "
                        f"{source_shape} -> {target_shape}; expected "
                        f"{expected_source} -> {expected_target}."
                    )
            target_only_keys = expected_target_only
            reinitialized_keys = shape_reset_keys | target_only_keys
            reset_policy = K1_RAW_FLOW_RESIDUAL_RESET_POLICY
        elif fusion_pretrained_adaptation:
            expected_shape_reset = set(self._ALLOWED_PRETRAINED_SHAPE_MISMATCH)
            observed_mismatches = set(shape_mismatches)
            if observed_mismatches != expected_shape_reset:
                raise RuntimeError(
                    "PixelDiT fusion reset set does not exactly match the "
                    "versioned official-to-fusion shape contract: "
                    f"missing={sorted(expected_shape_reset - observed_mismatches)}, "
                    f"unexpected={sorted(observed_mismatches - expected_shape_reset)}."
                )
            expected_reinitialized_shapes = {
                "s_embedder.proj.weight": (
                    (self.hidden_size, 3 * self.patch_size**2),
                    (self.hidden_size, self.in_channels * self.patch_size**2),
                ),
                "pixel_embedder.proj.weight": (
                    (OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE, 3),
                    (self.pixel_hidden_size, self.flow_channels),
                ),
                "final_layer.linear.weight": (
                    (3, OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE),
                    (self.out_channels, self.momap_output_hidden_size),
                ),
                "final_layer.linear.bias": ((3,), (self.out_channels,)),
            }
            for key, (expected_source, expected_target) in (
                expected_reinitialized_shapes.items()
            ):
                source_shape = tuple(source_state[key].shape)
                target_shape = tuple(target_state[key].shape)
                if source_shape != expected_source or target_shape != expected_target:
                    raise RuntimeError(
                        f"PixelDiT fusion reset shape drift for {key}: "
                        f"{source_shape} -> {target_shape}; expected "
                        f"{expected_source} -> {expected_target}."
                    )
            shape_reset_keys = expected_shape_reset
            target_only_keys = expected_target_only
            reinitialized_keys = shape_reset_keys | target_only_keys
            reset_policy = (
                RAW_FLOW_RESIDUAL_RESET_POLICY
                if self.momap_output_head_type
                == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
                else LOCAL_GLOBAL_FUSION_RESET_POLICY
            )
        elif full_rank_width_expansion:
            expected_shapes = self._full_rank_pretrained_shape_contract()
            shape_reset_keys = set(expected_shapes)
            missing_reset_keys = sorted(
                (shape_reset_keys - source_keys) | (shape_reset_keys - target_keys)
            )
            if missing_reset_keys:
                raise RuntimeError(
                    "PixelDiT full-rank reset contract references missing keys: "
                    f"{missing_reset_keys}."
                )
            observed_mismatches = set(shape_mismatches)
            if observed_mismatches != shape_reset_keys:
                raise RuntimeError(
                    "PixelDiT full-rank reset set does not exactly match the "
                    "versioned p16-to-p24 shape contract: "
                    f"missing={sorted(shape_reset_keys - observed_mismatches)}, "
                    f"unexpected={sorted(observed_mismatches - shape_reset_keys)}."
                )
            for key, (expected_source, expected_target) in expected_shapes.items():
                source_shape = tuple(source_state[key].shape)
                target_shape = tuple(target_state[key].shape)
                if source_shape != expected_source or target_shape != expected_target:
                    raise RuntimeError(
                        f"PixelDiT full-rank reset shape drift for {key}: "
                        f"{source_shape} -> {target_shape}; expected "
                        f"{expected_source} -> {expected_target}."
                    )
            target_only_keys = set()
            reinitialized_keys = shape_reset_keys
            reset_policy = FULL_RANK_LOCAL_RESET_POLICY
        else:
            shape_reset_keys = set(shape_mismatches)
            target_only_keys = set()
            reinitialized_keys = shape_reset_keys
            reset_policy = "official_3ch_stems_and_velocity_head_only_v2"

        allowed_mismatches = (
            shape_reset_keys
            if full_rank_width_expansion or fusion_pretrained_adaptation
            else set(self._ALLOWED_PRETRAINED_SHAPE_MISMATCH)
        )
        disallowed = sorted(set(shape_mismatches) - allowed_mismatches)
        if disallowed:
            raise RuntimeError(
                "PixelDiT checkpoint has disallowed shape mismatches: "
                + ", ".join(
                    f"{key}: {shape_mismatches[key]['checkpoint_shape']} -> "
                    f"{shape_mismatches[key]['model_shape']}"
                    for key in disallowed
                )
            )

        if not full_rank_width_expansion and not fusion_pretrained_adaptation:
            expected_reinitialized_shapes = {
                "s_embedder.proj.weight": (
                    (self.hidden_size, 3 * self.patch_size**2),
                    (self.hidden_size, self.in_channels * self.patch_size**2),
                ),
                "pixel_embedder.proj.weight": (
                    (self.pixel_hidden_size, 3),
                    (self.pixel_hidden_size, self.flow_channels),
                ),
                "final_layer.linear.weight": (
                    (3, self.pixel_hidden_size),
                    (self.out_channels, self.pixel_hidden_size),
                ),
                "final_layer.linear.bias": ((3,), (self.out_channels,)),
            }
            for key in sorted(shape_mismatches):
                source_shape = tuple(source_state[key].shape)
                target_shape = tuple(target_state[key].shape)
                expected_source_shape, expected_target_shape = (
                    expected_reinitialized_shapes[key]
                )
                if (
                    source_shape != expected_source_shape
                    or target_shape != expected_target_shape
                ):
                    raise RuntimeError(
                        f"Allowed K-flow reinitialization for {key} has unexpected "
                        f"shapes {source_shape} -> {target_shape}; expected "
                        f"{expected_source_shape} -> {expected_target_shape}."
                    )

        loadable = {
            key: value
            for key, value in source_state.items()
            if key in target_keys and key not in shape_reset_keys
        }
        incompatible = self.load_state_dict(loadable, strict=False)
        if (
            set(incompatible.missing_keys) != reinitialized_keys
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "Unexpected result from strict PixelDiT load: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}."
            )

        report = {
            "schema_version": (
                (
                    (
                        "pixeldit_pretrained_load_report_v9"
                        if fusion_local24_raw_flow_pretrained_adaptation
                        else "pixeldit_pretrained_load_report_v7"
                    )
                    if fusion_full_rank_pretrained_adaptation
                    else (
                        "pixeldit_pretrained_load_report_v6"
                        if self.momap_output_head_type
                        == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
                        else "pixeldit_pretrained_load_report_v5"
                    )
                )
                if fusion_pretrained_adaptation
                else (
                    "pixeldit_pretrained_load_report_v4"
                    if full_rank_width_expansion
                    else "pixeldit_pretrained_load_report_v3"
                )
            ),
            "checkpoint": source_name,
            "checkpoint_identity": checkpoint_file,
            "selected_prefix": prefix,
            "loaded_key_count": len(loadable),
            "loaded_numel": sum(value.numel() for value in loadable.values()),
            "source_tensor_count": int(source_integrity["tensor_count"]),
            "source_dtypes": list(source_integrity["dtypes"]),
            "skipped_shape_mismatches": shape_mismatches,
            "reinitialized_keys": sorted(reinitialized_keys),
            "reinitialization_reason": (
                "official_3ch_to_"
                f"{self.conditioning_contract.mode}_packed_kflow"
            ),
            "reset_policy": reset_policy,
            "conditioning_contract": self.conditioning_contract.as_dict(),
            "missing_keys": sorted(incompatible.missing_keys),
            "unexpected_keys": sorted(incompatible.unexpected_keys),
        }
        if fusion_k1_raw3_pretrained_adaptation:
            report["schema_version"] = "pixeldit_pretrained_load_report_v8"
        if fusion_h32s1_local96_raw_flow_pretrained_adaptation:
            report["schema_version"] = "pixeldit_pretrained_load_report_v10"
            report["semantic_reinitialized_keys"] = sorted(
                K1_RAW_FLOW_RESIDUAL_SEMANTIC_REINITIALIZED_KEYS
            )
        if fusion_pretrained_adaptation:
            report.update(
                {
                    "momap_output_head_type": self.momap_output_head_type,
                    "momap_output_hidden_size": self.momap_output_hidden_size,
                    "momap_output_head_initialization": (
                        self.momap_output_head_initialization
                    ),
                    "source_pixel_hidden_size": source_pixel_hidden_size,
                    "target_pixel_hidden_size": self.pixel_hidden_size,
                    "required_output_rank": self.out_channels,
                    "output_projection_max_rank": min(
                        self.momap_output_hidden_size,
                        self.out_channels,
                    ),
                    "shape_reinitialized_keys": sorted(shape_reset_keys),
                    "trunk_reinitialized_keys": (
                        sorted(
                            set(shape_reset_keys)
                            - {
                                "final_layer.norm.weight",
                                "final_layer.linear.bias",
                                "final_layer.linear.weight",
                            }
                        )
                        if fusion_full_rank_pretrained_adaptation
                        else list(LOCAL_GLOBAL_FUSION_TRUNK_REINITIALIZED_KEYS)
                    ),
                    "target_only_reinitialized_keys": sorted(target_only_keys),
                    "new_head_reinitialized_keys": list(
                        RAW_FLOW_RESIDUAL_NEW_HEAD_KEYS
                        if self.momap_output_head_type
                        == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
                        else LOCAL_GLOBAL_FUSION_NEW_HEAD_KEYS
                    ),
                }
            )
        elif full_rank_width_expansion:
            report.update(
                {
                    "source_pixel_hidden_size": source_pixel_hidden_size,
                    "target_pixel_hidden_size": self.pixel_hidden_size,
                    "required_output_rank": self.out_channels,
                    "output_projection_max_rank": min(
                        self.pixel_hidden_size,
                        self.out_channels,
                    ),
                }
            )
        return report

    def pretrained_adaptation_contract(self) -> dict[str, Any]:
        """Describe the only parameters allowed to reset from official weights."""

        if self.momap_output_head_type in _LOCAL_GLOBAL_FUSION_OUTPUT_HEADS:
            k1_raw3_fusion = (
                self.momap_output_head_type
                == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
                and self.pixel_hidden_size == OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE
                and self.flow_channels == 3
                and self.out_channels == 3
            )
            local24_fusion = (
                self.momap_output_head_type in _LOCAL_GLOBAL_FUSION_OUTPUT_HEADS
                and self.pixel_hidden_size == LEGACY_DPIX24_PIXEL_HIDDEN_SIZE
            )
            local24_raw_flow = (
                local24_fusion
                and self.momap_output_head_type
                == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
            )
            h32s1_local96_raw_flow = (
                self.momap_output_head_type
                == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
                and self.pixel_hidden_size == H32S1_DPIX96_PIXEL_HIDDEN_SIZE
                and self.flow_channels == 96
                and self.out_channels == 96
            )
            full_rank_fusion = local24_fusion or h32s1_local96_raw_flow
            if full_rank_fusion:
                shape_contract = self._full_rank_pretrained_shape_contract()
                shape_reinitialized = set(shape_contract)
                source_shapes = {
                    key: list(shape_contract[key][0])
                    for key in shape_reinitialized
                }
                trunk_reinitialized = sorted(
                    shape_reinitialized
                    - {
                        "final_layer.norm.weight",
                        "final_layer.linear.bias",
                        "final_layer.linear.weight",
                    }
                )
            else:
                shape_reinitialized = set(self._ALLOWED_PRETRAINED_SHAPE_MISMATCH)
                source_shapes = {
                    "s_embedder.proj.weight": [
                        self.hidden_size,
                        3 * self.patch_size**2,
                    ],
                    "pixel_embedder.proj.weight": [
                        OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE,
                        3,
                    ],
                    "final_layer.linear.weight": [
                        3,
                        OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE,
                    ],
                    "final_layer.linear.bias": [3],
                }
                trunk_reinitialized = list(
                    LOCAL_GLOBAL_FUSION_TRUNK_REINITIALIZED_KEYS
                )
            target_only = set(LOCAL_GLOBAL_FUSION_TARGET_ONLY_KEYS)
            if (
                self.momap_output_head_type
                == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
            ):
                target_only.update(RAW_FLOW_RESIDUAL_TARGET_ONLY_KEYS)
            reinitialized = tuple(sorted(shape_reinitialized | target_only))
            if h32s1_local96_raw_flow:
                adaptation_version = "pixeldit_pretrained_adaptation_v9"
            elif local24_raw_flow:
                adaptation_version = "pixeldit_pretrained_adaptation_v8"
            elif k1_raw3_fusion:
                adaptation_version = "pixeldit_pretrained_adaptation_v7"
            elif local24_fusion:
                adaptation_version = "pixeldit_pretrained_adaptation_v6"
            elif (
                self.momap_output_head_type
                == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
            ):
                adaptation_version = "pixeldit_pretrained_adaptation_v5"
            else:
                adaptation_version = "pixeldit_pretrained_adaptation_v4"
            if h32s1_local96_raw_flow:
                adaptation_reset_policy = H32S1_LOCAL96_RAW_FLOW_RESIDUAL_RESET_POLICY
            elif local24_raw_flow:
                adaptation_reset_policy = LOCAL24_RAW_FLOW_RESIDUAL_RESET_POLICY
            elif k1_raw3_fusion:
                adaptation_reset_policy = K1_RAW_FLOW_RESIDUAL_RESET_POLICY
            elif local24_fusion:
                adaptation_reset_policy = LOCAL24_FUSION_RESET_POLICY
            elif (
                self.momap_output_head_type
                == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
            ):
                adaptation_reset_policy = RAW_FLOW_RESIDUAL_RESET_POLICY
            else:
                adaptation_reset_policy = LOCAL_GLOBAL_FUSION_RESET_POLICY
            return {
                "version": adaptation_version,
                "conditioning": self.conditioning_contract.as_dict(),
                "flow_channels": self.flow_channels,
                "output_channels": self.out_channels,
                "pixel_hidden_size": self.pixel_hidden_size,
                "momap_output_head_type": self.momap_output_head_type,
                "momap_output_hidden_size": self.momap_output_hidden_size,
                "momap_output_head_initialization": (
                    self.momap_output_head_initialization
                ),
                "required_output_rank": self.out_channels,
                "output_projection_max_rank": min(
                    self.momap_output_hidden_size,
                    self.out_channels,
                ),
                "reset_policy": adaptation_reset_policy,
                "source_pixel_hidden_size": OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE,
                "allowed_reinitialized_keys": list(reinitialized),
                "shape_reinitialized_keys": sorted(shape_reinitialized),
                "trunk_reinitialized_keys": list(trunk_reinitialized),
                "target_only_reinitialized_keys": sorted(target_only),
                "new_head_reinitialized_keys": list(
                    RAW_FLOW_RESIDUAL_NEW_HEAD_KEYS
                    if self.momap_output_head_type
                    == MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD
                    else LOCAL_GLOBAL_FUSION_NEW_HEAD_KEYS
                ),
                "source_shapes": source_shapes,
                "target_shapes": {
                    key: list(self.state_dict()[key].shape)
                    for key in reinitialized
                },
            }

        if (
            self.momap_output_head_type == LEGACY_LOCAL_ONLY_OUTPUT_HEAD
            and self.pixel_hidden_size == LEGACY_DPIX24_PIXEL_HIDDEN_SIZE
            and self.require_full_rank_output
        ):
            reinitialized = full_rank_local_reinitialized_keys(self.pixel_depth)
            shape_contract = self._full_rank_pretrained_shape_contract()
            return {
                "version": "pixeldit_pretrained_adaptation_v3",
                "conditioning": self.conditioning_contract.as_dict(),
                "flow_channels": self.flow_channels,
                "output_channels": self.out_channels,
                "pixel_hidden_size": self.pixel_hidden_size,
                "required_output_rank": self.out_channels,
                "output_projection_max_rank": min(
                    self.pixel_hidden_size,
                    self.out_channels,
                ),
                "reset_policy": FULL_RANK_LOCAL_RESET_POLICY,
                "source_pixel_hidden_size": OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE,
                "allowed_reinitialized_keys": list(reinitialized),
                "source_shapes": {
                    key: list(shape_contract[key][0]) for key in reinitialized
                },
                "target_shapes": {
                    key: list(shape_contract[key][1]) for key in reinitialized
                },
            }

        return {
            "version": "pixeldit_pretrained_adaptation_v2",
            "conditioning": self.conditioning_contract.as_dict(),
            "flow_channels": self.flow_channels,
            "output_channels": self.out_channels,
            "allowed_reinitialized_keys": sorted(
                self._ALLOWED_PRETRAINED_SHAPE_MISMATCH
            ),
            "source_shapes": {
                "s_embedder.proj.weight": [
                    self.hidden_size,
                    3 * self.patch_size**2,
                ],
                "pixel_embedder.proj.weight": [self.pixel_hidden_size, 3],
                "final_layer.linear.weight": [3, self.pixel_hidden_size],
                "final_layer.linear.bias": [3],
            },
            "target_shapes": {
                key: list(self.state_dict()[key].shape)
                for key in sorted(self._ALLOWED_PRETRAINED_SHAPE_MISMATCH)
            },
        }


PixelDiT = PixDiT


__all__ = [
    "AugmentedDiTBlock",
    "FORMAL_FULL_RANK_PIXELDIT_REINITIALIZED_KEYS",
    "FORMAL_KFLOW_PIXEL_HIDDEN_SIZE",
    "FORMAL_LOCAL_GLOBAL_FUSION_REINITIALIZED_KEYS",
    "FORMAL_MOMAP_OUTPUT_HIDDEN_SIZE",
    "FULL_RANK_LOCAL_RESET_POLICY",
    "K1_RAW_FLOW_RESIDUAL_RESET_POLICY",
    "K1_RAW_FLOW_RESIDUAL_SEMANTIC_REINITIALIZED_KEYS",
    "LEGACY_DPIX24_PIXEL_HIDDEN_SIZE",
    "LEGACY_LOCAL_ONLY_OUTPUT_HEAD",
    "LOCAL_GLOBAL_FUSION_NEW_HEAD_KEYS",
    "LOCAL24_FUSION_RESET_POLICY",
    "LOCAL24_RAW_FLOW_RESIDUAL_RESET_POLICY",
    "LOCAL_GLOBAL_FUSION_INITIALIZATION",
    "LOCAL_GLOBAL_FUSION_RESET_POLICY",
    "LOCAL_GLOBAL_FUSION_SHAPE_REINITIALIZED_KEYS",
    "LOCAL_GLOBAL_FUSION_TARGET_ONLY_KEYS",
    "LOCAL_GLOBAL_FUSION_TRUNK_REINITIALIZED_KEYS",
    "MOMAP_LOCAL_GLOBAL_FUSION_OUTPUT_HEAD",
    "MOMAP_LOCAL_GLOBAL_RAW_FLOW_RESIDUAL_OUTPUT_HEAD",
    "MoMapLocalGlobalFusionHead",
    "MoMapRawFlowResidualPath",
    "OFFICIAL_PIXELDIT_PIXEL_HIDDEN_SIZE",
    "PatchTokenEmbedder",
    "PiTBlock",
    "PixDiT",
    "PixelDiT",
    "PixelTokenEmbedder",
    "RAW_FLOW_RESIDUAL_INITIALIZATION",
    "RAW_FLOW_RESIDUAL_NEW_HEAD_KEYS",
    "RAW_FLOW_RESIDUAL_NOISE_FLOOR",
    "RAW_FLOW_RESIDUAL_RESET_POLICY",
    "RAW_FLOW_RESIDUAL_TARGET_ONLY_KEYS",
    "full_rank_local_reinitialized_keys",
]
