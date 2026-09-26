"""Core modules for the FastWAM PixelDiT world expert.

The architecture and parameter names follow NVIDIA's public PixelDiT c2i
implementation at commit ``41f73006ae532b0b41fee72b181dc22891a5a01a``:
https://github.com/NVlabs/PixelDiT

This module keeps only the PyTorch/NumPy model primitives needed by FastWAM.
The task-specific conditioning and output contract live in ``model.py``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_2d_sincos_pos_embed(embed_dim: int, height: int, width: int) -> np.ndarray:
    """Return a flattened 2-D sine/cosine positional embedding."""
    if embed_dim % 4 != 0:
        raise ValueError(f"`embed_dim` must be divisible by 4, got {embed_dim}.")

    grid_y = np.arange(height, dtype=np.float32)
    grid_x = np.arange(width, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    return np.concatenate(
        [
            _get_1d_sincos_pos_embed(embed_dim // 2, grid_x.reshape(-1)),
            _get_1d_sincos_pos_embed(embed_dim // 2, grid_y.reshape(-1)),
        ],
        axis=1,
    )


def _get_1d_sincos_pos_embed(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"`embed_dim` must be even, got {embed_dim}.")
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000**omega)
    angles = np.einsum("m,d->md", positions.astype(np.float64), omega)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def apply_adaln(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = int(frequency_embedding_size)

    @staticmethod
    def timestep_embedding(timestep: torch.Tensor, dim: int, max_period: int = 10) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=timestep.device)
            / half
        )
        angles = timestep[..., None].float() * frequencies[None, ...]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        embedding = self.timestep_embedding(timestep, self.frequency_embedding_size)
        embedding = embedding.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(embedding)


class ClassEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = int(num_classes)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


class FeedForward(nn.Module):
    """SwiGLU feed-forward module used by patch-level PixelDiT blocks."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def precompute_freqs_cis_2d(
    dim: int,
    height: int,
    width: int,
    theta: float = 10000.0,
    scale: float = 16.0,
) -> torch.Tensor:
    """Create complex 2-D RoPE frequencies for a flattened spatial grid."""
    if dim % 4 != 0:
        raise ValueError(f"RoPE head dimension must be divisible by 4, got {dim}.")
    x_positions = torch.linspace(0, scale, width)
    y_positions = torch.linspace(0, scale, height)
    y_positions, x_positions = torch.meshgrid(y_positions, x_positions, indexing="ij")
    frequencies = 1.0 / (
        theta ** (torch.arange(0, dim, 4, dtype=torch.float32)[: dim // 4] / dim)
    )
    x_angles = torch.outer(x_positions.reshape(-1), frequencies)
    y_angles = torch.outer(y_positions.reshape(-1), frequencies)
    x_cis = torch.polar(torch.ones_like(x_angles), x_angles)
    y_cis = torch.polar(torch.ones_like(y_angles), y_angles)
    return torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1).reshape(
        height * width, -1
    )


def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2-D RoPE to ``[B, N, heads, head_dim]`` query/key tensors."""
    if query.shape != key.shape:
        raise ValueError(
            f"Query/key shapes must match, got {tuple(query.shape)} and {tuple(key.shape)}."
        )
    if query.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE head dimension must be even, got {query.shape[-1]}.")
    if freqs_cis.shape[0] != query.shape[1]:
        raise ValueError(
            "RoPE sequence length mismatch: "
            f"freqs={freqs_cis.shape[0]} versus tokens={query.shape[1]}."
        )

    frequencies = freqs_cis.to(device=query.device)[None, :, None, :]
    query_complex = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    key_complex = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    query_out = torch.view_as_real(query_complex * frequencies).flatten(-2)
    key_out = torch.view_as_real(key_complex * frequencies).flatten(-2)
    return query_out.to(dtype=query.dtype), key_out.to(dtype=key.dtype)


class RotaryAttention(nn.Module):
    """PixelDiT attention with an optional non-detached K/V cache."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = RMSNorm,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"`dim` ({dim}) must be divisible by `num_heads` ({num_heads}).")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def project_qkv(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project Q/K/V and return layout ``[B, N, heads, head_dim]``."""
        batch_size, sequence_length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = apply_rotary_emb(query, key, freqs_cis=pos)
        return query, key, value

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, sequence_length, channels = x.shape
        query, key, value = self.project_qkv(x, pos)
        attention = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        output = attention.transpose(1, 2).reshape(batch_size, sequence_length, channels)
        output = self.proj_drop(self.proj(output))
        if return_kv:
            return output, {"k": key, "v": value}
        return output


class MLP(nn.Module):
    """GELU MLP used by pixel-level PiT blocks."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


__all__ = [
    "ClassEmbedder",
    "FeedForward",
    "FinalLayer",
    "MLP",
    "RMSNorm",
    "RotaryAttention",
    "TimestepConditioner",
    "apply_adaln",
    "get_2d_sincos_pos_embed",
    "precompute_freqs_cis_2d",
]
