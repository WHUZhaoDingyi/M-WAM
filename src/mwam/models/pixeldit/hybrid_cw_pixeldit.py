"""Hybrid condition/world PixelDiT core for full three-camera 3D flow.

The public flow carrier remains horizon-major ``[B, 3*K, H, W]``.  For the
fixed K=4 RoboTwin layout, the upper two thirds carry Head ``dX,dY,dZ`` and the
lower third carries Wrist ``dX,dY,dZ``.  Internally the carrier is split into a
12-channel Head stream and an 12-channel Wrist stream.  Their patch tokens are
concatenated spatially, never along the feature dimension.

Every transformer layer owns one parameter set and advances two residual
streams with the directed visibility rule::

    C <- C
    W <- [C, W]

This file is intentionally independent of the historical single-stream
``PixDiT`` implementation so existing V8/V9 checkpoints retain their exact
model mathematics.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mwam.utils.checkpoint_integrity import validate_floating_state_dict

from .conditioning import WorldConditioningContract
from .model import (
    MoMapLocalGlobalFusionHead,
    PatchTokenEmbedder,
    PiTBlock,
    PixelTokenEmbedder,
    PixDiT,
)
from .modules import (
    ClassEmbedder,
    FeedForward,
    RMSNorm,
    RotaryAttention,
    TimestepConditioner,
    apply_adaln,
    get_2d_sincos_pos_embed,
    precompute_freqs_cis_2d,
)


HYBRID_CW_STREAM_SEMANTICS = "condition_c_and_allxyz_world_w_v1"
HYBRID_CW_ATTENTION_SEMANTICS = "c_from_c_w_from_cw_shared_parameters_v1"
HYBRID_CW_CARRIER_SEMANTICS = "k4_xyz_head_dxyz_wrist_12ch_mosaic_v1"


def _project_attention(
    attention: RotaryAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=attention.attn_drop.p if attention.training else 0.0,
    )
    output = output.transpose(1, 2).reshape(
        query.shape[0], query.shape[1], attention.dim
    )
    return attention.proj_drop(attention.proj(output))


class HybridCWGlobalBlock(nn.Module):
    """One ImageNet-compatible Global block shared by the C and W streams."""

    def __init__(self, hidden_size: int, groups: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RotaryAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @staticmethod
    def _modulations(
        module: nn.Module, condition: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return module(condition).chunk(6, dim=-1)

    def forward_cw(
        self,
        c: torch.Tensor,
        w: torch.Tensor,
        condition_c: torch.Tensor,
        condition_w: torch.Tensor,
        position: torch.Tensor,
        *,
        return_native_kv: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if c.shape != w.shape or c.ndim != 3:
            raise ValueError(
                "Hybrid Global C/W tokens must share [B,N,D], got "
                f"{tuple(c.shape)} and {tuple(w.shape)}."
            )
        c_mod = self._modulations(self.adaLN_modulation, condition_c)
        w_mod = self._modulations(self.adaLN_modulation, condition_w)
        c_input = apply_adaln(self.norm1(c), c_mod[0], c_mod[1])
        w_input = apply_adaln(self.norm1(w), w_mod[0], w_mod[1])

        q_c, k_c, v_c = self.attn.project_qkv(c_input, position)
        q_w, k_w, v_w = self.attn.project_qkv(w_input, position)
        c_attention = _project_attention(self.attn, q_c, k_c, v_c)
        w_attention = _project_attention(
            self.attn,
            q_w,
            torch.cat([k_c, k_w], dim=1),
            torch.cat([v_c, v_w], dim=1),
        )

        c = c + c_mod[2] * c_attention
        w = w + w_mod[2] * w_attention
        c = c + c_mod[5] * self.mlp(
            apply_adaln(self.norm2(c), c_mod[3], c_mod[4])
        )
        w = w + w_mod[5] * self.mlp(
            apply_adaln(self.norm2(w), w_mod[3], w_mod[4])
        )
        if return_native_kv:
            return c, w, k_c, v_c, k_w, v_w
        return c, w


class HybridCWPiTBlock(PiTBlock):
    """One ImageNet-compatible PiT block shared by C and W pixel streams."""

    def forward_cw(
        self,
        c: torch.Tensor,
        w: torch.Tensor,
        patch_condition_c: torch.Tensor,
        patch_condition_w: torch.Tensor,
        image_height: int,
        image_width: int,
        patch_size: int,
        *,
        return_native_kv: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if c.shape != w.shape or c.ndim != 3:
            raise ValueError(
                "Hybrid PiT C/W tokens must share [B*N,P^2,D], got "
                f"{tuple(c.shape)} and {tuple(w.shape)}."
            )
        batch_patch_count, pixels_per_patch, channels = c.shape
        if channels != self.pixel_dim:
            raise ValueError(
                f"Hybrid PiT expected pixel width {self.pixel_dim}, got {channels}."
            )
        if patch_size != self.patch_size:
            raise ValueError(
                f"Hybrid PiT was built for P={self.patch_size}, got {patch_size}."
            )
        patch_height = image_height // patch_size
        patch_width = image_width // patch_size
        sequence_length = patch_height * patch_width
        if batch_patch_count % sequence_length:
            raise ValueError("Hybrid PiT cannot infer batch size from patch groups.")
        batch_size = batch_patch_count // sequence_length
        if pixels_per_patch != patch_size * patch_size:
            raise ValueError("Hybrid PiT pixels-per-patch mismatch.")
        expected_condition = (batch_patch_count, self.context_dim)
        if patch_condition_c.shape != expected_condition or patch_condition_w.shape != expected_condition:
            raise ValueError(
                "Hybrid PiT patch conditions must be [B*N,D], got "
                f"{tuple(patch_condition_c.shape)} and "
                f"{tuple(patch_condition_w.shape)}."
            )

        modulation_count = 4 if self.adaln_post_modulation else 6

        def prepare(
            x: torch.Tensor, patch_condition: torch.Tensor
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            modulation = self.adaLN_modulation(patch_condition).view(
                batch_patch_count,
                pixels_per_patch,
                modulation_count * self.pixel_dim,
            )
            parts = modulation.chunk(modulation_count, dim=-1)
            if self.adaln_post_modulation:
                attention_input = self.norm1(x)
            else:
                attention_input = apply_adaln(self.norm1(x), parts[0], parts[1])
            compressed = self.compress_to_attn(
                attention_input.reshape(batch_patch_count, -1)
            ).view(batch_size, sequence_length, self.attn_dim)
            return compressed, parts

        compressed_c, c_mod = prepare(c, patch_condition_c)
        compressed_w, w_mod = prepare(w, patch_condition_w)
        position = self._fetch_pos(patch_height, patch_width, c.device)
        q_c, k_c, v_c = self.attn.project_qkv(compressed_c, position)
        q_w, k_w, v_w = self.attn.project_qkv(compressed_w, position)
        compact_c = _project_attention(self.attn, q_c, k_c, v_c)
        compact_w = _project_attention(
            self.attn,
            q_w,
            torch.cat([k_c, k_w], dim=1),
            torch.cat([v_c, v_w], dim=1),
        )
        expanded_c = self.expand_from_attn(
            compact_c.reshape(batch_patch_count, self.attn_dim)
        ).view(batch_patch_count, pixels_per_patch, self.pixel_dim)
        expanded_w = self.expand_from_attn(
            compact_w.reshape(batch_patch_count, self.attn_dim)
        ).view(batch_patch_count, pixels_per_patch, self.pixel_dim)

        def finish(
            x: torch.Tensor,
            expanded: torch.Tensor,
            modulation: tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            if self.adaln_post_modulation:
                scale_attn, shift_attn, scale_mlp, shift_mlp = modulation
                x = x + expanded * (1 + scale_attn) + shift_attn
                return x + self.mlp(self.norm2(x)) * (1 + scale_mlp) + shift_mlp
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation
            del shift_attn, scale_attn
            x = x + gate_attn * expanded
            return x + gate_mlp * self.mlp(
                apply_adaln(self.norm2(x), shift_mlp, scale_mlp)
            )

        c = finish(c, expanded_c, c_mod)
        w = finish(w, expanded_w, w_mod)
        if return_native_kv:
            return c, w, compact_c, compact_w, k_c, v_c, k_w, v_w
        return c, w, compact_c, compact_w


class HybridCWPixelDiT(nn.Module):
    """K=4 PixelDiT with clean C and heterogeneous Head/Wrist W streams."""

    STREAM_SEMANTICS_VERSION = HYBRID_CW_STREAM_SEMANTICS
    ATTENTION_SEMANTICS_VERSION = HYBRID_CW_ATTENTION_SEMANTICS
    CARRIER_SEMANTICS_VERSION = HYBRID_CW_CARRIER_SEMANTICS

    def __init__(
        self,
        *,
        condition_channels: int = 9,
        k: int = 4,
        hidden_size: int = 1152,
        pixel_hidden_size: int = 16,
        patch_depth: int = 26,
        pixel_depth: int = 4,
        patch_size: int = 16,
        num_groups: int = 16,
        num_classes: int = 1000,
        use_pixel_abs_pos: bool = True,
        pit_adaln_post_modulation: bool = False,
        momap_output_hidden_size: int = 128,
        use_gradient_checkpointing: bool = False,
        validate_finite: bool = False,
    ) -> None:
        super().__init__()
        if condition_channels != 9:
            raise ValueError("Hybrid C is fixed to RGB3+Seg3+XYZ3 = 9 channels.")
        if k != 4:
            raise ValueError("Hybrid World is fixed to K=4.")
        dimensions = {
            "hidden_size": hidden_size,
            "pixel_hidden_size": pixel_hidden_size,
            "patch_depth": patch_depth,
            "pixel_depth": pixel_depth,
            "patch_size": patch_size,
            "num_groups": num_groups,
            "num_classes": num_classes,
            "momap_output_hidden_size": momap_output_hidden_size,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions.values()
        ):
            raise ValueError(f"Hybrid PixelDiT dimensions must be positive: {dimensions}.")
        if hidden_size % num_groups:
            raise ValueError("hidden_size must be divisible by num_groups.")
        if (hidden_size // num_groups) % 4:
            raise ValueError("Hybrid Global head_dim must be divisible by four for 2-D RoPE.")
        if use_pixel_abs_pos and pixel_hidden_size % 4:
            raise ValueError("pixel_hidden_size must divide by four for 2-D absolute position.")

        self.clean_channels = int(condition_channels)
        self.conditioning_contract = WorldConditioningContract()
        if self.conditioning_contract.condition_channels != self.clean_channels:
            raise RuntimeError("Hybrid C channels disagree with conditioning contract.")
        self.rgb_channels = self.conditioning_contract.rgb_channels
        self.segmentation_channels = self.conditioning_contract.segmentation_channels
        self.canonical_xyz_channels = self.conditioning_contract.canonical_xyz_channels
        self.raw_depth_channels = self.conditioning_contract.raw_depth_channels
        self.k = int(k)
        self.flow_channels = 3 * self.k
        self.out_channels = self.flow_channels
        self.in_channels = self.clean_channels + self.flow_channels
        self.head_flow_channels = 3 * self.k
        self.wrist_flow_channels = 3 * self.k
        self.hidden_size = int(hidden_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.patch_size = int(patch_size)
        self.num_groups = int(num_groups)
        self.num_classes = int(num_classes)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.pit_adaln_post_modulation = bool(pit_adaln_post_modulation)
        self.momap_output_hidden_size = int(momap_output_hidden_size)
        self.require_full_rank_output = True
        self.momap_output_head_type = (
            "hybrid_allxyz_local_global_fusion_v1"
        )
        self.momap_output_head_initialization = "xavier_gain_1e-2_v1"
        self.raw_flow_noise_floor = None
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.validate_finite = bool(validate_finite)
        self.head_dim = self.hidden_size // self.num_groups

        pixels_per_patch = self.patch_size**2
        self.c_patch_embedder = PatchTokenEmbedder(
            self.clean_channels * pixels_per_patch, self.hidden_size
        )
        self.w1_patch_embedder = PatchTokenEmbedder(
            self.head_flow_channels * pixels_per_patch, self.hidden_size
        )
        self.w2_patch_embedder = PatchTokenEmbedder(
            self.wrist_flow_channels * pixels_per_patch, self.hidden_size
        )
        # Position is added only after W1/W2 have been restored to the complete
        # full-mosaic patch order.
        self.c_pixel_embedder = PixelTokenEmbedder(
            self.clean_channels, self.pixel_hidden_size, use_pixel_abs_pos=False
        )
        self.w1_pixel_embedder = PixelTokenEmbedder(
            self.head_flow_channels, self.pixel_hidden_size, use_pixel_abs_pos=False
        )
        self.w2_pixel_embedder = PixelTokenEmbedder(
            self.wrist_flow_channels, self.pixel_hidden_size, use_pixel_abs_pos=False
        )
        self.t_embedder = TimestepConditioner(self.hidden_size)
        self.y_embedder = ClassEmbedder(self.num_classes + 1, self.hidden_size)
        self.patch_blocks = nn.ModuleList(
            [
                HybridCWGlobalBlock(self.hidden_size, self.num_groups)
                for _ in range(self.patch_depth)
            ]
        )
        self.pixel_blocks = nn.ModuleList(
            [
                HybridCWPiTBlock(
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
        self.head_final_layer = MoMapLocalGlobalFusionHead(
            global_hidden_size=self.hidden_size,
            local_hidden_size=self.pixel_hidden_size,
            output_hidden_size=self.momap_output_hidden_size,
            out_channels=self.head_flow_channels,
            patch_size=self.patch_size,
        )
        self.wrist_final_layer = MoMapLocalGlobalFusionHead(
            global_hidden_size=self.hidden_size,
            local_hidden_size=self.pixel_hidden_size,
            output_hidden_size=self.momap_output_hidden_size,
            out_channels=self.wrist_flow_channels,
            patch_size=self.patch_size,
        )
        self._patch_pos_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._pixel_pos_cache: dict[tuple[int, int], torch.Tensor] = {}
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for embedder in (
            self.c_patch_embedder,
            self.w1_patch_embedder,
            self.w2_patch_embedder,
            self.c_pixel_embedder,
            self.w1_pixel_embedder,
            self.w2_pixel_embedder,
        ):
            nn.init.xavier_uniform_(embedder.proj.weight)
            nn.init.zeros_(embedder.proj.bias)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.patch_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)
        for block in self.pixel_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)
        for head in (self.head_final_layer, self.wrist_final_layer):
            nn.init.xavier_uniform_(head.global_to_pixels.weight)
            nn.init.zeros_(head.global_to_pixels.bias)
            nn.init.xavier_uniform_(head.local_projection.weight)
            nn.init.zeros_(head.local_projection.bias)
            nn.init.xavier_uniform_(head.fusion_projection.weight)
            nn.init.zeros_(head.fusion_projection.bias)
            nn.init.xavier_uniform_(head.linear.weight, gain=1.0e-2)
            nn.init.zeros_(head.linear.bias)

    @staticmethod
    def _layout(height: int, width: int, patch_size: int) -> dict[str, int]:
        if height % 3 or width % 2:
            raise ValueError(
                "Hybrid mosaic must have a 2:1 Head/Wrist height ratio and even width."
            )
        head_height = 2 * (height // 3)
        wrist_height = height // 3
        if any(value % patch_size for value in (height, width, head_height, wrist_height)):
            raise ValueError("Hybrid mosaic regions must all be divisible by patch_size.")
        full_width = width // patch_size
        head_patches = (head_height // patch_size) * full_width
        wrist_patches = (wrist_height // patch_size) * full_width
        return {
            "height": height,
            "width": width,
            "head_height": head_height,
            "wrist_height": wrist_height,
            "patch_height": height // patch_size,
            "patch_width": full_width,
            "head_patches": head_patches,
            "wrist_patches": wrist_patches,
            "patches": head_patches + wrist_patches,
        }

    @staticmethod
    def _patchify(inputs: torch.Tensor, patch_size: int) -> torch.Tensor:
        return F.unfold(inputs, kernel_size=patch_size, stride=patch_size).transpose(1, 2)

    @staticmethod
    def _unpatchify(
        patches: torch.Tensor,
        *,
        output_size: tuple[int, int],
        channels: int,
        patch_size: int,
    ) -> torch.Tensor:
        return F.fold(
            patches.transpose(1, 2),
            output_size=output_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def split_flow_carrier(
        self, noised_flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = noised_flow.shape
        if channels != self.flow_channels:
            raise ValueError(
                f"Hybrid carrier requires {self.flow_channels} channels, got {channels}."
            )
        layout = self._layout(height, width, self.patch_size)
        structured = noised_flow.contiguous().view(
            batch_size, self.k, 3, height, width
        )
        w1 = structured[:, :, :, : layout["head_height"], :].contiguous().view(
            batch_size,
            self.head_flow_channels,
            layout["head_height"],
            width,
        )
        w2 = structured[:, :, :, layout["head_height"] :, :].contiguous().view(
            batch_size,
            self.wrist_flow_channels,
            layout["wrist_height"],
            width,
        )
        return w1, w2

    def merge_velocity_carrier(
        self,
        head_velocity: torch.Tensor,
        wrist_velocity: torch.Tensor,
        *,
        full_height: int,
    ) -> torch.Tensor:
        batch_size, _, head_height, width = head_velocity.shape
        wrist_height = int(wrist_velocity.shape[-2])
        if head_height + wrist_height != full_height:
            raise ValueError("Hybrid Head/Wrist output heights do not recover the mosaic.")
        if head_velocity.shape != (
            batch_size,
            self.head_flow_channels,
            head_height,
            width,
        ) or wrist_velocity.shape != (
            batch_size,
            self.wrist_flow_channels,
            wrist_height,
            width,
        ):
            raise ValueError("Hybrid output-head channel/spatial contract mismatch.")
        structured = head_velocity.new_zeros(
            (batch_size, self.k, 3, full_height, width)
        )
        structured[:, :, :, :head_height, :] = head_velocity.view(
            batch_size, self.k, 3, head_height, width
        )
        structured[:, :, :, head_height:, :] = wrist_velocity.view(
            batch_size, self.k, 3, wrist_height, width
        )
        return structured.view(batch_size, self.flow_channels, full_height, width)

    def _fetch_patch_pos(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width)
        if key not in self._patch_pos_cache:
            self._patch_pos_cache[key] = precompute_freqs_cis_2d(
                self.head_dim, height, width
            )
        return self._patch_pos_cache[key].to(device=device)

    def _fetch_grouped_pixel_pos(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (height, width)
        if key not in self._pixel_pos_cache:
            position = torch.from_numpy(
                get_2d_sincos_pos_embed(self.pixel_hidden_size, height, width)
            ).view(height, width, self.pixel_hidden_size)
            ph, pw, p = height // self.patch_size, width // self.patch_size, self.patch_size
            position = position.view(ph, p, pw, p, self.pixel_hidden_size)
            position = position.permute(0, 2, 1, 3, 4).contiguous()
            self._pixel_pos_cache[key] = position.view(
                ph * pw, p * p, self.pixel_hidden_size
            )
        return self._pixel_pos_cache[key].to(device=device, dtype=dtype)

    @staticmethod
    def _prepare_vector(
        value: torch.Tensor, *, batch_size: int, device: torch.device, name: str
    ) -> torch.Tensor:
        vector = torch.as_tensor(value, device=device).reshape(-1)
        if vector.numel() == 1 and batch_size > 1:
            vector = vector.expand(batch_size)
        if vector.numel() != batch_size:
            raise ValueError(f"{name} must provide one value per sample.")
        vector = vector.float()
        if not bool(torch.isfinite(vector).all().item()):
            raise ValueError(f"{name} must be finite.")
        if bool(((vector < 0) | (vector > 1)).any().item()):
            raise ValueError(f"{name} must lie in [0,1].")
        return vector

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
        if clean_image.ndim != 4 or noised_flow.ndim != 4:
            raise ValueError("Hybrid C and flow must be [B,C,H,W].")
        if clean_image.shape[0] != noised_flow.shape[0] or clean_image.shape[2:] != noised_flow.shape[2:]:
            raise ValueError("Hybrid C and flow must share batch/spatial dimensions.")
        if clean_image.shape[1] != self.clean_channels:
            raise ValueError(f"Hybrid C must have {self.clean_channels} channels.")
        if noised_flow.shape[1] != self.flow_channels:
            raise ValueError(f"Hybrid flow must have {self.flow_channels} channels.")
        if clean_image.device != noised_flow.device or clean_image.dtype != noised_flow.dtype:
            raise ValueError("Hybrid C and flow must share device/dtype.")
        if not clean_image.is_floating_point():
            raise TypeError("Hybrid inputs must be floating point.")
        if self.validate_finite and not bool(
            torch.isfinite(clean_image).all() and torch.isfinite(noised_flow).all()
        ):
            raise FloatingPointError("Hybrid inputs contain NaN or Inf.")
        batch_size, _, height, width = clean_image.shape
        layout = self._layout(height, width, self.patch_size)
        t_w = self._prepare_vector(
            timestep, batch_size=batch_size, device=noised_flow.device, name="t_W"
        )
        t_c = torch.ones_like(t_w)
        if class_labels is None:
            labels = torch.full(
                (batch_size,), self.num_classes, device=noised_flow.device, dtype=torch.long
            )
        else:
            labels = torch.as_tensor(
                class_labels, device=noised_flow.device, dtype=torch.long
            ).reshape(-1)
            if labels.numel() == 1 and batch_size > 1:
                labels = labels.expand(batch_size)
            if labels.numel() != batch_size or bool(
                ((labels < 0) | (labels > self.num_classes)).any().item()
            ):
                raise ValueError("Hybrid class labels are outside the configured range.")
        if source_support_mask is not None:
            if source_support_mask.ndim == 3:
                source_support_mask = source_support_mask.unsqueeze(1)
            if source_support_mask.shape != (batch_size, 1, height, width) or source_support_mask.dtype != torch.bool:
                raise ValueError("source_support_mask must be bool[B,1,H,W].")
        w1, w2 = self.split_flow_carrier(noised_flow)
        c_tokens = self.c_patch_embedder(self._patchify(clean_image, self.patch_size))
        w1_tokens = self.w1_patch_embedder(self._patchify(w1, self.patch_size))
        w2_tokens = self.w2_patch_embedder(self._patchify(w2, self.patch_size))
        w_tokens = torch.cat([w1_tokens, w2_tokens], dim=1)
        expected = (batch_size, layout["patches"], self.hidden_size)
        if c_tokens.shape != expected or w_tokens.shape != expected:
            raise RuntimeError(
                f"Hybrid Global tokenization drifted: C={tuple(c_tokens.shape)}, "
                f"W={tuple(w_tokens.shape)}, expected={expected}."
            )
        time_c = self.t_embedder(t_c).view(batch_size, 1, self.hidden_size)
        time_w = self.t_embedder(t_w).view(batch_size, 1, self.hidden_size)
        class_embedding = self.y_embedder(labels).view(batch_size, 1, self.hidden_size)
        if extra_condition is None:
            extra = torch.zeros_like(time_c)
        else:
            if extra_condition.shape != (batch_size, self.hidden_size):
                raise ValueError("extra_condition must be [B,hidden_size].")
            extra = extra_condition.to(noised_flow).unsqueeze(1)
        condition_c = F.silu(time_c + class_embedding + extra)
        condition_w = F.silu(time_w + class_embedding + extra)
        all_valid = torch.ones(
            (batch_size, layout["patches"]), dtype=torch.bool, device=noised_flow.device
        )
        return {
            "c_patch_tokens": c_tokens,
            "w_patch_tokens": w_tokens,
            "condition_c": condition_c,
            "condition_w": condition_w,
            "time_embedding_c": time_c,
            "time_embedding_w": time_w,
            "position": self._fetch_patch_pos(
                layout["patch_height"], layout["patch_width"], noised_flow.device
            ),
            "c_key_valid": all_valid,
            "w_key_valid": all_valid.clone(),
            "noised_flow": noised_flow,
            "clean_image": clean_image,
            "w1": w1,
            "w2": w2,
            "batch_size": batch_size,
            "layout": layout,
            "next_patch_block": 0,
        }

    def forward_patch_block(
        self,
        layer_index: int,
        state: dict[str, Any],
        *,
        return_native_kv: bool = False,
    ) -> tuple[Any, ...]:
        if layer_index != int(state.get("next_patch_block", -1)):
            raise ValueError("Hybrid Global blocks must execute once in order.")
        block = self.patch_blocks[layer_index]

        def run(
            c: torch.Tensor,
            w: torch.Tensor,
            condition_c: torch.Tensor,
            condition_w: torch.Tensor,
            position: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            return block.forward_cw(
                c,
                w,
                condition_c,
                condition_w,
                position,
                return_native_kv=return_native_kv,
            )

        if self.use_gradient_checkpointing and self.training:
            result = checkpoint(
                run,
                state["c_patch_tokens"],
                state["w_patch_tokens"],
                state["condition_c"],
                state["condition_w"],
                state["position"],
                use_reentrant=False,
            )
        else:
            result = run(
                state["c_patch_tokens"],
                state["w_patch_tokens"],
                state["condition_c"],
                state["condition_w"],
                state["position"],
            )
        state["c_patch_tokens"], state["w_patch_tokens"] = result[:2]
        state["next_patch_block"] = layer_index + 1
        if return_native_kv:
            native = {
                "c": {"k": result[2], "v": result[3]},
                "w": {"k": result[4], "v": result[5]},
            }
            return state["c_patch_tokens"], state["w_patch_tokens"], native
        return state["c_patch_tokens"], state["w_patch_tokens"]

    def finish_patch_blocks(self, state: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if int(state.get("next_patch_block", -1)) != self.patch_depth:
            raise ValueError("Hybrid Global stack is incomplete.")
        finished_c = F.silu(state["time_embedding_c"] + state["c_patch_tokens"])
        finished_w = F.silu(state["time_embedding_w"] + state["w_patch_tokens"])
        batch_size = int(state["batch_size"])
        state["finished_c_patch_tokens"] = finished_c
        state["finished_w_patch_tokens"] = finished_w
        state["c_patch_condition"] = finished_c.reshape(-1, self.hidden_size)
        state["w_patch_condition"] = finished_w.reshape(-1, self.hidden_size)
        if finished_c.shape[0] != batch_size:
            raise RuntimeError("Hybrid finished patch batch drifted.")
        return state["c_patch_condition"], state["w_patch_condition"]

    def prepare_pixel_tokens(self, state: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if "c_patch_condition" not in state:
            raise ValueError("Call finish_patch_blocks before Hybrid PiT.")
        layout = state["layout"]
        batch_size = int(state["batch_size"])
        c = self.c_pixel_embedder(
            state["clean_image"],
            img_height=layout["height"],
            img_width=layout["width"],
            patch_size=self.patch_size,
        ).view(batch_size, layout["patches"], self.patch_size**2, self.pixel_hidden_size)
        w1 = self.w1_pixel_embedder(
            state["w1"],
            img_height=layout["head_height"],
            img_width=layout["width"],
            patch_size=self.patch_size,
        ).view(batch_size, layout["head_patches"], self.patch_size**2, self.pixel_hidden_size)
        w2 = self.w2_pixel_embedder(
            state["w2"],
            img_height=layout["wrist_height"],
            img_width=layout["width"],
            patch_size=self.patch_size,
        ).view(batch_size, layout["wrist_patches"], self.patch_size**2, self.pixel_hidden_size)
        w = torch.cat([w1, w2], dim=1)
        if self.use_pixel_abs_pos:
            position = self._fetch_grouped_pixel_pos(
                layout["height"],
                layout["width"],
                device=c.device,
                dtype=c.dtype,
            ).unsqueeze(0)
            c = c + position
            w = w + position
        state["c_pixel_tokens"] = c.view(
            batch_size * layout["patches"], self.patch_size**2, self.pixel_hidden_size
        )
        state["w_pixel_tokens"] = w.view(
            batch_size * layout["patches"], self.patch_size**2, self.pixel_hidden_size
        )
        state["next_pixel_block"] = 0
        return state["c_pixel_tokens"], state["w_pixel_tokens"]

    def forward_pixel_block(
        self,
        layer_index: int,
        state: dict[str, Any],
        *,
        return_native_kv: bool = False,
    ) -> tuple[Any, ...]:
        if layer_index != int(state.get("next_pixel_block", -1)):
            raise ValueError("Hybrid PiT blocks must execute once in order.")
        block = self.pixel_blocks[layer_index]
        layout = state["layout"]
        image_height = int(layout["height"])
        image_width = int(layout["width"])
        patch_size = int(self.patch_size)

        def run(
            c: torch.Tensor,
            w: torch.Tensor,
            patch_condition_c: torch.Tensor,
            patch_condition_w: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            return block.forward_cw(
                c,
                w,
                patch_condition_c,
                patch_condition_w,
                image_height,
                image_width,
                patch_size,
                return_native_kv=return_native_kv,
            )

        if self.use_gradient_checkpointing and self.training:
            result = checkpoint(
                run,
                state["c_pixel_tokens"],
                state["w_pixel_tokens"],
                state["c_patch_condition"],
                state["w_patch_condition"],
                use_reentrant=False,
            )
        else:
            result = run(
                state["c_pixel_tokens"],
                state["w_pixel_tokens"],
                state["c_patch_condition"],
                state["w_patch_condition"],
            )
        state["c_pixel_tokens"], state["w_pixel_tokens"] = result[:2]
        state["next_pixel_block"] = layer_index + 1
        if return_native_kv:
            native = {
                "c": {"k": result[4], "v": result[5]},
                "w": {"k": result[6], "v": result[7]},
            }
            return result[2], result[3], native
        return result[2], result[3]

    def final_head(self, state: dict[str, Any]) -> torch.Tensor:
        if int(state.get("next_pixel_block", -1)) != self.pixel_depth:
            raise ValueError("Hybrid PiT stack is incomplete.")
        layout = state["layout"]
        batch_size = int(state["batch_size"])
        local = state["w_pixel_tokens"].view(
            batch_size,
            layout["patches"],
            self.patch_size**2,
            self.pixel_hidden_size,
        )
        global_tokens = state["finished_w_patch_tokens"]
        head_patches = self.head_final_layer(
            global_tokens[:, : layout["head_patches"]],
            local[:, : layout["head_patches"]],
        )
        wrist_patches = self.wrist_final_layer(
            global_tokens[:, layout["head_patches"] :],
            local[:, layout["head_patches"] :],
        )
        head_velocity = self._unpatchify(
            head_patches.permute(0, 1, 3, 2).reshape(
                batch_size,
                layout["head_patches"],
                self.head_flow_channels * self.patch_size**2,
            ),
            output_size=(layout["head_height"], layout["width"]),
            channels=self.head_flow_channels,
            patch_size=self.patch_size,
        )
        wrist_velocity = self._unpatchify(
            wrist_patches.permute(0, 1, 3, 2).reshape(
                batch_size,
                layout["wrist_patches"],
                self.wrist_flow_channels * self.patch_size**2,
            ),
            output_size=(layout["wrist_height"], layout["width"]),
            channels=self.wrist_flow_channels,
            patch_size=self.patch_size,
        )
        velocity = self.merge_velocity_carrier(
            head_velocity, wrist_velocity, full_height=layout["height"]
        )
        if self.validate_finite and not bool(torch.isfinite(velocity).all()):
            raise FloatingPointError("Hybrid World velocity contains NaN or Inf.")
        state["head_velocity"] = head_velocity
        state["wrist_velocity"] = wrist_velocity
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
        state = self.prepare_tokens(
            clean_image,
            noised_flow,
            timestep,
            class_labels,
            source_support_mask=source_support_mask,
            extra_condition=extra_condition,
        )
        for index in range(self.patch_depth):
            self.forward_patch_block(index, state)
        self.finish_patch_blocks(state)
        self.prepare_pixel_tokens(state)
        for index in range(self.pixel_depth):
            self.forward_pixel_block(index, state)
        return self.final_head(state)

    def load_pretrained(
        self,
        checkpoint_value: str | os.PathLike[str] | Mapping[str, Any],
        *,
        weight_source: str = "auto",
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load the shared ImageNet core and RGB slices with an explicit receipt."""

        source_state_raw, source_name, checkpoint_file = PixDiT._read_checkpoint(
            checkpoint_value, map_location
        )
        source_state, selected_prefix = PixDiT._select_denoiser_state(
            source_state_raw, weight_source
        )
        validate_floating_state_dict(source_state, name="Hybrid PixelDiT source")
        target = self.state_dict()
        exact_prefixes = ("patch_blocks.", "pixel_blocks.", "t_embedder.", "y_embedder.")
        exact_keys = sorted(
            key for key in target if key.startswith(exact_prefixes)
        )
        missing = [key for key in exact_keys if key not in source_state]
        mismatched = {
            key: [list(source_state[key].shape), list(target[key].shape)]
            for key in exact_keys
            if key in source_state and source_state[key].shape != target[key].shape
        }
        if missing or mismatched:
            raise RuntimeError(
                f"Hybrid ImageNet core mismatch: missing={missing}, shapes={mismatched}."
            )
        source_patch_weight = source_state.get("s_embedder.proj.weight")
        source_patch_bias = source_state.get("s_embedder.proj.bias")
        source_pixel_weight = source_state.get("pixel_embedder.proj.weight")
        source_pixel_bias = source_state.get("pixel_embedder.proj.bias")
        required_shapes = {
            "s_embedder.proj.weight": (
                self.hidden_size,
                3 * self.patch_size**2,
            ),
            "s_embedder.proj.bias": (self.hidden_size,),
            "pixel_embedder.proj.weight": (self.pixel_hidden_size, 3),
            "pixel_embedder.proj.bias": (self.pixel_hidden_size,),
        }
        source_stems = {
            "s_embedder.proj.weight": source_patch_weight,
            "s_embedder.proj.bias": source_patch_bias,
            "pixel_embedder.proj.weight": source_pixel_weight,
            "pixel_embedder.proj.bias": source_pixel_bias,
        }
        invalid_stems = {
            name: None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            for name, value in source_stems.items()
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != required_shapes[name]
        }
        if invalid_stems:
            raise RuntimeError(f"Hybrid ImageNet RGB stem mismatch: {invalid_stems}.")

        with torch.no_grad():
            for key in exact_keys:
                target[key].copy_(source_state[key].to(target[key]))
            assert isinstance(source_patch_weight, torch.Tensor)
            assert isinstance(source_patch_bias, torch.Tensor)
            assert isinstance(source_pixel_weight, torch.Tensor)
            assert isinstance(source_pixel_bias, torch.Tensor)
            self.c_patch_embedder.proj.weight.zero_()
            self.c_patch_embedder.proj.weight[:, : 3 * self.patch_size**2].copy_(
                source_patch_weight.to(self.c_patch_embedder.proj.weight)
            )
            self.c_patch_embedder.proj.bias.copy_(
                source_patch_bias.to(self.c_patch_embedder.proj.bias)
            )
            self.c_pixel_embedder.proj.weight.zero_()
            self.c_pixel_embedder.proj.weight[:, :3].copy_(
                source_pixel_weight.to(self.c_pixel_embedder.proj.weight)
            )
            self.c_pixel_embedder.proj.bias.copy_(
                source_pixel_bias.to(self.c_pixel_embedder.proj.bias)
            )
        copied_targets = set(exact_keys) | {
            "c_patch_embedder.proj.weight",
            "c_patch_embedder.proj.bias",
            "c_pixel_embedder.proj.weight",
            "c_pixel_embedder.proj.bias",
        }
        return {
            "schema_version": "hybrid_cw_pixeldit_imagenet_load_report_v1",
            "checkpoint": source_name,
            "checkpoint_identity": checkpoint_file,
            "selected_prefix": selected_prefix,
            "exact_loaded_keys": exact_keys,
            "rgb_slice_loaded_keys": [
                "c_patch_embedder.proj.weight",
                "c_pixel_embedder.proj.weight",
            ],
            "rgb_bias_loaded_keys": [
                "c_patch_embedder.proj.bias",
                "c_pixel_embedder.proj.bias",
            ],
            "zero_initialized_extra_c_inputs": {
                "global": "C channels 3..8",
                "pixel": "C channels 3..8",
            },
            "new_initialized_keys": sorted(set(target) - copied_targets),
        }


__all__ = [
    "HYBRID_CW_ATTENTION_SEMANTICS",
    "HYBRID_CW_CARRIER_SEMANTICS",
    "HYBRID_CW_STREAM_SEMANTICS",
    "HybridCWGlobalBlock",
    "HybridCWPiTBlock",
    "HybridCWPixelDiT",
]
