"""Deterministic instance palette encoding."""
from __future__ import annotations
import hashlib
import numpy as np

SOURCE_SEG_COLOR_SEED = "fastwam_momap_seg_condition_v1"


def encode_episode_scoped_segmentation(
    actor_seg_raw: np.ndarray,
    *,
    episode_identity: str,
    color_seed: str = SOURCE_SEG_COLOR_SEED,
) -> np.ndarray:
    """Encode instances as deterministic RGB features in ``[-1,1]``.

    Raw actor IDs are categorical and episode-local, so they must not be fed
    to a network as continuous magnitudes.  The episode identity is part of
    the hash: equal IDs are stable across views of one episode, while no color
    identity is promised across episodes.  Raw ID zero is always black.
    """

    segmentation = np.asarray(actor_seg_raw)
    if segmentation.ndim < 1 or not np.issubdtype(
        segmentation.dtype, np.integer
    ):
        raise TypeError("actor_seg_raw must be an integer array")
    if not str(episode_identity).strip():
        raise ValueError("episode_identity must not be empty")
    if not str(color_seed).strip():
        raise ValueError("color_seed must not be empty")

    encoded = np.zeros((*segmentation.shape, 3), dtype=np.float32)
    for raw_id in np.unique(segmentation):
        integer_id = int(raw_id)
        if integer_id == 0:
            continue
        digest = hashlib.sha256(
            (
                f"{color_seed}\0{episode_identity}\0{integer_id}"
            ).encode("utf-8")
        ).digest()
        rgb = np.frombuffer(digest[:3], dtype=np.uint8).copy()
        # Preserve black exclusively for background, even in the 1/2^24 hash
        # collision case.
        if not bool(rgb.any()):
            rgb[0] = 1
        encoded[segmentation == raw_id] = rgb.astype(np.float32) / 127.5 - 1.0
    return encoded

