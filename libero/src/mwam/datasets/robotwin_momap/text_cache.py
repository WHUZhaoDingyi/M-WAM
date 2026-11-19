"""On-demand access to precomputed FastWAM/Wan text embeddings.

The training path intentionally treats the prompt bank as a model input, not
as an audit artifact. A prompt's SHA256 is used only as the existing filename
key. No inventory scan, content digest, or online encoder fallback is
performed here.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


class CachedTextContextEncoder:
    """Load and validate one existing Wan2.2 T5 context on first use.

    Validation is cached against a cheap filesystem identity. Re-reading an
    unchanged file therefore avoids repeating the finite-value scan, while a
    replaced file is validated again. Tensors are cloned before being returned
    because downstream adapters may mutate padding in place.
    """

    CACHE_TAG = "wan22ti2v5b"
    REQUIRED_KEYS = frozenset({"context", "mask"})
    ALLOWED_CONTEXT_DTYPES = frozenset({torch.bfloat16, torch.float32})

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        context_len: int = 128,
        hidden_size: int = 4096,
    ) -> None:
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError("text embedding cache_dir must be provided")
        if isinstance(context_len, bool) or not isinstance(context_len, int):
            raise TypeError("context_len must be an integer")
        if context_len <= 0:
            raise ValueError("context_len must be positive")
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
            raise TypeError("hidden_size must be an integer")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(
                f"missing text embedding cache directory: {self.cache_dir}"
            )
        self.context_len = context_len
        self.hidden_size = hidden_size
        self._validated_file_identities: dict[
            str, tuple[int, int, int, int]
        ] = {}
        self._validation_pid = os.getpid()

    @staticmethod
    def prompt_key(prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def cache_path(self, prompt: str) -> Path:
        key = self.prompt_key(prompt)
        return self.cache_dir / (
            f"{key}.t5_len{self.context_len}.{self.CACHE_TAG}.pt"
        )

    @staticmethod
    def _file_identity(path: Path) -> tuple[int, int, int, int]:
        metadata = path.stat()
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
        )

    def _reset_validation_cache_after_fork(self) -> None:
        current_pid = os.getpid()
        if current_pid != self._validation_pid:
            self._validated_file_identities.clear()
            self._validation_pid = current_pid

    def _validate_payload(
        self,
        payload: Any,
        *,
        cache_path: Path,
        prompt_key: str,
        check_finite: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(payload, Mapping):
            raise TypeError(f"text cache payload must be a mapping: {cache_path}")
        actual_keys = frozenset(str(key) for key in payload)
        if actual_keys != self.REQUIRED_KEYS:
            raise ValueError(
                "text cache keys must be exactly ['context', 'mask']; "
                f"got {sorted(actual_keys)} in {cache_path}"
            )

        context = payload["context"]
        mask = payload["mask"]
        if not isinstance(context, torch.Tensor):
            raise TypeError(f"cached context must be a tensor: {cache_path}")
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"cached mask must be a tensor: {cache_path}")
        expected_context_shape = (self.context_len, self.hidden_size)
        if tuple(context.shape) != expected_context_shape:
            raise ValueError(
                f"cached context for prompt_key={prompt_key} must be "
                f"{expected_context_shape}, got {tuple(context.shape)} in "
                f"{cache_path}"
            )
        if context.dtype not in self.ALLOWED_CONTEXT_DTYPES:
            raise TypeError(
                "cached context must use torch.float32 or torch.bfloat16, "
                f"got {context.dtype} for prompt_key={prompt_key}"
            )
        if tuple(mask.shape) != (self.context_len,):
            raise ValueError(
                f"cached mask for prompt_key={prompt_key} must be "
                f"[{self.context_len}], got {tuple(mask.shape)} in {cache_path}"
            )
        if mask.dtype != torch.bool:
            raise TypeError(
                f"cached mask must use torch.bool, got {mask.dtype} "
                f"for prompt_key={prompt_key}"
            )
        if check_finite and not bool(torch.isfinite(context).all().item()):
            raise ValueError(
                f"cached context contains NaN/Inf for prompt_key={prompt_key}: "
                f"{cache_path}"
            )
        return context, mask

    def __call__(self, prompt: str) -> dict[str, torch.Tensor]:
        self._reset_validation_cache_after_fork()
        prompt_key = self.prompt_key(prompt)
        cache_path = self.cache_path(prompt)
        if not cache_path.is_file():
            raise FileNotFoundError(
                "missing text embedding for "
                f"prompt_key={prompt_key}: {cache_path}. "
                "Generate the real prompt embedding explicitly; online text "
                "encoding fallback is disabled."
            )

        try:
            file_identity = self._file_identity(cache_path)
            payload: Any = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )
        except OSError as exc:
            raise ValueError(
                f"cannot read text embedding prompt_key={prompt_key}: "
                f"{cache_path}: {exc}"
            ) from exc

        check_finite = (
            self._validated_file_identities.get(prompt_key) != file_identity
        )
        context, mask = self._validate_payload(
            payload,
            cache_path=cache_path,
            prompt_key=prompt_key,
            check_finite=check_finite,
        )
        if check_finite:
            self._validated_file_identities[prompt_key] = file_identity

        return {
            "context": context.detach().cpu().contiguous().clone(),
            "mask": mask.detach().cpu().contiguous().clone(),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_validated_file_identities"] = {}
        state["_validation_pid"] = os.getpid()
        return state


