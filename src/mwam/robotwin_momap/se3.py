"""Shared fail-closed validation for rigid homogeneous transforms."""

from __future__ import annotations

from typing import Any

import numpy as np


SE3_VALIDATION_SCHEMA_VERSION = "robotwin_momap_se3_validation_v1"
SE3_LAST_ROW_ATOL = 1.0e-5
SE3_ROTATION_ATOL = 2.0e-4


def require_se3(
    value: Any,
    name: str,
    *,
    valid_mask: Any | None = None,
) -> np.ndarray:
    """Return ``float32`` SE(3) matrices after strict rigid-body checks.

    ``value`` may be one transform ``[4,4]`` or a batch ``[...,4,4]``.  When
    ``valid_mask`` is supplied, only rows marked true are required to contain
    a valid transform; this preserves the raw-episode convention that invalid
    entity rows may contain arbitrary placeholders.  A marked-valid row must
    be finite, have the exact homogeneous form within the declared tolerance,
    and contain a proper (determinant +1) orthonormal rotation.
    """

    matrices = np.asarray(value, dtype=np.float32)
    if matrices.ndim < 2 or matrices.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must end in [4,4], got {matrices.shape}")

    batch_shape = matrices.shape[:-2]
    if valid_mask is None:
        selected = np.ones(batch_shape or (), dtype=bool)
    else:
        selected = np.asarray(valid_mask, dtype=bool)
        if selected.shape != batch_shape:
            raise ValueError(
                f"{name} valid_mask shape={selected.shape}, expected={batch_shape}"
            )

    flattened = matrices.reshape(-1, 4, 4)
    selected_flat = np.asarray(selected, dtype=bool).reshape(-1)
    checked = flattened[selected_flat]
    if checked.size == 0:
        return matrices
    if not np.isfinite(checked).all():
        raise ValueError(f"{name} contains NaN/Inf at valid rows")

    expected_last_row = np.asarray(
        [0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    if not np.allclose(
        checked[:, 3, :],
        expected_last_row[None],
        rtol=0.0,
        atol=SE3_LAST_ROW_ATOL,
    ):
        raise ValueError(f"{name} has an invalid homogeneous last row")

    rotations = checked[:, :3, :3]
    gram = np.matmul(np.swapaxes(rotations, -1, -2), rotations)
    identity = np.eye(3, dtype=np.float32)[None]
    determinants = np.linalg.det(rotations)
    if not np.allclose(
        gram,
        identity,
        rtol=0.0,
        atol=SE3_ROTATION_ATOL,
    ) or not np.allclose(
        determinants,
        np.ones_like(determinants),
        rtol=0.0,
        atol=SE3_ROTATION_ATOL,
    ):
        raise ValueError(
            f"{name} rotation is not a proper orthonormal matrix"
        )
    return matrices


__all__ = [
    "SE3_LAST_ROW_ATOL",
    "SE3_ROTATION_ATOL",
    "SE3_VALIDATION_SCHEMA_VERSION",
    "require_se3",
]
