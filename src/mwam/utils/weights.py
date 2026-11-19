"""Validate the floating-point tensors in PixelDiT initialization weights."""
from collections.abc import Mapping
from typing import Any

import torch


def validate_floating_state_dict(
    state: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    """Validate that every state value is a finite real floating tensor."""

    if not isinstance(state, Mapping):
        raise TypeError(f"`{name}` must be a mapping.")
    tensor_count = 0
    numel = 0
    dtypes: set[str] = set()
    for key, value in state.items():
        if not isinstance(key, str):
            raise TypeError(f"`{name}` keys must be strings, got {type(key).__name__}.")
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"`{name}[{key}]` must be torch.Tensor, "
                f"got {type(value).__name__}."
            )
        if not torch.is_floating_point(value):
            raise TypeError(
                f"`{name}[{key}]` must have a real floating dtype, got {value.dtype}."
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"`{name}[{key}]` contains NaN or Inf.")
        tensor_count += 1
        numel += int(value.numel())
        dtypes.add(str(value.dtype).removeprefix("torch."))
    return {
        "tensor_count": tensor_count,
        "numel": numel,
        "dtypes": sorted(dtypes),
    }
