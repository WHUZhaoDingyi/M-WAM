from __future__ import annotations


import torch


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

