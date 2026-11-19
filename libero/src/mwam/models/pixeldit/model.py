from __future__ import annotations


from typing import Optional

import torch

import torch.nn as nn


from .modules import MLP, RMSNorm, RotaryAttention, apply_adaln, get_2d_sincos_pos_embed, precompute_freqs_cis_2d

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

