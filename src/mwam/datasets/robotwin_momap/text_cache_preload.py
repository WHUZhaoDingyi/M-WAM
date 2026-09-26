"""Explicit, bounded Fork-COW preload for Wan text embeddings.

Ordinary training uses CachedTextContextEncoder and never scans the embedding
bank. This strategy is selected only by the matching factory flag; it validates
the same tensor contract, loads one contiguous CPU bank in each rank parent,
and rejects pickle-based worker startup so workers must inherit the bank
through Linux fork. No receipt or content-digest gate is involved.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .text_cache import CachedTextContextEncoder

logger = logging.getLogger(__name__)


class ForkCowPreloadedTextContextEncoder(CachedTextContextEncoder):
    """Preload the selected cache bank once for forked DataLoader workers."""

    ACCESS_SCHEMA_VERSION = "robotwin_text_embedding_fork_cow_bank_v1"

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        context_len: int = 128,
        hidden_size: int = 4096,
        prompts: Sequence[str],
        preload_max_bytes: int | None,
        preload_max_node_bytes: int | None,
        preload_rank_stagger_seconds: float = 0.0,
    ) -> None:
        super().__init__(
            cache_dir,
            context_len=context_len,
            hidden_size=hidden_size,
        )
        self.preload_max_bytes = self._positive_bound(
            preload_max_bytes,
            name="preload_max_bytes",
        )
        self.preload_max_node_bytes = self._positive_bound(
            preload_max_node_bytes,
            name="preload_max_node_bytes",
        )
        if (
            isinstance(preload_rank_stagger_seconds, bool)
            or not isinstance(preload_rank_stagger_seconds, (int, float))
            or not math.isfinite(float(preload_rank_stagger_seconds))
            or float(preload_rank_stagger_seconds) < 0.0
        ):
            raise ValueError(
                "preload_rank_stagger_seconds must be finite and non-negative"
            )
        self.preload_rank_stagger_seconds = float(
            preload_rank_stagger_seconds
        )
        self._preloaded_context_bank: torch.Tensor | None = None
        self._preloaded_mask_bank: torch.Tensor | None = None
        self._preloaded_rows: dict[str, int] = {}
        self._preload_binding: dict[str, Any] | None = None
        self._preload_owner_pid = os.getpid()

        entries = self._cache_entries(prompts)
        local_world_size, local_rank = self._runtime_local_topology()
        upper_rank_bytes = len(entries) * self.context_len * (
            self.hidden_size * torch.empty(
                (), dtype=torch.float32
            ).element_size()
            + torch.empty((), dtype=torch.bool).element_size()
        )
        self._enforce_bounds(
            rank_bytes=upper_rank_bytes,
            node_bytes=upper_rank_bytes * local_world_size,
        )
        stagger_seconds = local_rank * self.preload_rank_stagger_seconds
        if stagger_seconds:
            logger.info(
                "Text embedding preload stagger: local_rank=%d/%d seconds=%.3f",
                local_rank,
                local_world_size,
                stagger_seconds,
            )
            time.sleep(stagger_seconds)
        self._preload(
            entries,
            local_world_size=local_world_size,
            local_rank=local_rank,
        )

    @staticmethod
    def _positive_bound(value: int | None, *, name: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{name} must be an explicit positive integer")
        return value

    @staticmethod
    def _runtime_local_topology() -> tuple[int, int]:
        try:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        except ValueError as exc:
            raise ValueError(
                "LOCAL_WORLD_SIZE and LOCAL_RANK must be integers for "
                "text embedding preload"
            ) from exc
        if (
            local_world_size <= 0
            or local_rank < 0
            or local_rank >= local_world_size
        ):
            raise ValueError(
                "invalid local preload topology: "
                f"LOCAL_RANK={local_rank} "
                f"LOCAL_WORLD_SIZE={local_world_size}"
            )
        return local_world_size, local_rank

    def _cache_entries(
        self,
        prompts: Sequence[str],
    ) -> tuple[tuple[str, Path], ...]:
        if isinstance(prompts, (str, bytes)) or not isinstance(
            prompts, Sequence
        ):
            raise TypeError("preload prompts must be a sequence of strings")
        selected: dict[str, Path] = {}
        for prompt in prompts:
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(
                    "preload prompts must contain non-empty strings"
                )
            key = self.prompt_key(prompt)
            selected[key] = self.cache_path(prompt)
        entries = sorted(selected.items())
        if not entries:
            raise ValueError("preload prompts must not be empty")
        for key, path in entries:
            if not path.is_file():
                raise FileNotFoundError(
                    "missing selected text embedding for "
                    f"prompt_key={key}: {path}"
                )
        return tuple(entries)

    def _enforce_bounds(self, *, rank_bytes: int, node_bytes: int) -> None:
        if rank_bytes > self.preload_max_bytes:
            raise MemoryError(
                "text embedding preload exceeds preload_max_bytes: "
                f"projected={rank_bytes} limit={self.preload_max_bytes}"
            )
        if node_bytes > self.preload_max_node_bytes:
            raise MemoryError(
                "text embedding preload exceeds preload_max_node_bytes: "
                f"projected={node_bytes} "
                f"limit={self.preload_max_node_bytes}"
            )

    def _load_validated(
        self,
        prompt_key: str,
        cache_path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            payload = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )
        except OSError as exc:
            raise ValueError(
                f"cannot read text embedding prompt_key={prompt_key}: "
                f"{cache_path}: {exc}"
            ) from exc
        context, mask = self._validate_payload(
            payload,
            cache_path=cache_path,
            prompt_key=prompt_key,
            check_finite=True,
        )
        return (
            context.detach().cpu().contiguous(),
            mask.detach().cpu().contiguous(),
        )

    def _preload(
        self,
        entries: tuple[tuple[str, Path], ...],
        *,
        local_world_size: int,
        local_rank: int,
    ) -> None:
        first_context, first_mask = self._load_validated(*entries[0])
        context_dtype = first_context.dtype
        rank_bytes = len(entries) * self.context_len * (
            self.hidden_size * first_context.element_size()
            + first_mask.element_size()
        )
        node_bytes = rank_bytes * local_world_size
        self._enforce_bounds(rank_bytes=rank_bytes, node_bytes=node_bytes)

        context_bank = torch.empty(
            (len(entries), self.context_len, self.hidden_size),
            dtype=context_dtype,
            device="cpu",
        )
        mask_bank = torch.empty(
            (len(entries), self.context_len),
            dtype=torch.bool,
            device="cpu",
        )
        rows: dict[str, int] = {}
        for row, (prompt_key, cache_path) in enumerate(entries):
            if row == 0:
                context, mask = first_context, first_mask
            else:
                context, mask = self._load_validated(
                    prompt_key,
                    cache_path,
                )
            if context.dtype != context_dtype:
                raise ValueError(
                    "Fork-COW text preload requires one homogeneous context "
                    f"dtype; got {context.dtype} after {context_dtype}"
                )
            context_bank[row].copy_(context)
            mask_bank[row].copy_(mask)
            rows[prompt_key] = row

        actual_bytes = (
            context_bank.numel() * context_bank.element_size()
            + mask_bank.numel() * mask_bank.element_size()
        )
        if actual_bytes != rank_bytes:
            raise AssertionError(
                "text embedding preload memory accounting drifted"
            )
        self._preloaded_context_bank = context_bank
        self._preloaded_mask_bank = mask_bank
        self._preloaded_rows = rows
        self._preload_binding = {
            "schema_version": self.ACCESS_SCHEMA_VERSION,
            "mode": "rank_parent_preload_fork_cow",
            "entry_count": len(entries),
            "context_dtype": str(context_dtype).removeprefix("torch."),
            "rank_bank_nbytes": actual_bytes,
            "projected_node_bank_nbytes": node_bytes,
            "local_rank": local_rank,
            "local_world_size": local_world_size,
        }
        logger.info(
            "Text embedding preload COMPLETE: local_rank=%d/%d entries=%d "
            "rank_bank_bytes=%d projected_node_bank_bytes=%d",
            local_rank,
            local_world_size,
            len(entries),
            actual_bytes,
            node_bytes,
        )

    @property
    def requires_fork_cow_workers(self) -> bool:
        return True

    def preloaded_view(self, prompt: str) -> dict[str, torch.Tensor]:
        prompt_key = self.prompt_key(prompt)
        row = self._preloaded_rows.get(prompt_key)
        if row is None:
            raise FileNotFoundError(
                "prompt is not present in the preloaded text bank: "
                f"prompt_key={prompt_key}"
            )
        assert self._preloaded_context_bank is not None
        assert self._preloaded_mask_bank is not None
        return {
            "context": self._preloaded_context_bank[row],
            "mask": self._preloaded_mask_bank[row],
        }

    def __call__(self, prompt: str) -> dict[str, torch.Tensor]:
        view = self.preloaded_view(prompt)
        return {
            "context": view["context"].clone(),
            "mask": view["mask"].clone(),
        }

    def preload_binding(self) -> dict[str, Any]:
        assert self._preload_binding is not None
        return dict(self._preload_binding)

    def __getstate__(self) -> dict[str, Any]:
        raise RuntimeError(
            "preloaded text embeddings must be inherited by fork workers; "
            "pickle/spawn worker transfer is disabled"
        )


__all__ = ["ForkCowPreloadedTextContextEncoder"]
