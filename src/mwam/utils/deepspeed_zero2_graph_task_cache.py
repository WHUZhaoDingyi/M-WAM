"""Opt-in compatibility workaround for a DeepSpeed ZeRO-1/2 hook hot path.

DeepSpeed 0.18.5 asks ``count_used_parameters_in_backward`` from every
trainable-parameter gradient hook.  The helper scans every trainable
parameter, so a model with ``P`` parameter tensors pays ``O(P ** 2)`` Python
work for each backward.  The answer is invariant for one autograd graph task
and one DeepSpeed-owned parameter sequence, which makes that pair a safe cache
key.

This module deliberately does not patch site-packages.  Installation is
opt-in, version/API gated, process-local, bounded, and must happen before the
DeepSpeed engine creates its gradient hooks.
"""

from __future__ import annotations

import functools
import inspect
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


SUPPORTED_DEEPSPEED_VERSION = "0.18.5"
DEFAULT_MAX_CACHE_ENTRIES = 64
_PATCH_MARKER = "_fastwam_deepspeed_zero2_graph_task_cache"


@dataclass(frozen=True)
class DeepSpeedGraphTaskCacheReport:
    enabled: bool
    deepspeed_version: str | None
    zero_stage: int | None
    max_cache_entries: int | None


@dataclass
class _CacheEntry:
    # Keep a strong reference so a live cache entry cannot suffer Python id
    # reuse if a different parameter sequence is allocated later.
    parameters: object
    count: int


class _GraphTaskMemoizedParameterCounter:
    """Memoize one used-parameter count per nested autograd graph task.

    A bounded global cache (rather than thread-local state) lets hooks for the
    same graph safely share the result even if PyTorch dispatches them through
    different Python threads.  The re-entrant lock and graph-task key preserve
    interleaved outer/inner backward calls.
    """

    def __init__(
        self,
        original: Callable[[Sequence[Any]], int],
        current_graph_task_id: Callable[[], int],
        *,
        max_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
    ) -> None:
        if not callable(original):
            raise TypeError("original used-parameter counter must be callable")
        if not callable(current_graph_task_id):
            raise TypeError("current_graph_task_id must be callable")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("max_entries must be an integer")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.original = original
        self.current_graph_task_id = current_graph_task_id
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._cache: OrderedDict[tuple[int, int], _CacheEntry] = OrderedDict()

    def __call__(self, parameters: Sequence[Any]) -> int:
        graph_task_id = self.current_graph_task_id()
        if isinstance(graph_task_id, bool) or not isinstance(graph_task_id, int):
            raise RuntimeError(
                "PyTorch returned a non-integer autograd graph task id"
            )

        # Preserve DeepSpeed's own outside-backward failure instead of hiding
        # it behind a cache entry.
        if graph_task_id < 0:
            return self.original(parameters)

        key = (graph_task_id, id(parameters))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and cached.parameters is parameters:
                self._cache.move_to_end(key)
                return cached.count

            count = self.original(parameters)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise RuntimeError(
                    "DeepSpeed used-parameter counter returned an invalid value: "
                    f"{count!r}"
                )
            self._cache[key] = _CacheEntry(parameters=parameters, count=count)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
            return count

    def clear(self) -> None:
        """Discard results from completed backward graph tasks."""

        with self._lock:
            self._cache.clear()


def _canonical_distributed_type(value: object) -> str:
    raw = getattr(value, "name", value)
    normalized = str(raw).strip().upper()
    if "." in normalized:
        normalized = normalized.rsplit(".", 1)[-1]
    return normalized


def _validate_original_binding(
    *,
    stage_module: object,
    utils_module: object,
) -> Callable[[Sequence[Any]], int]:
    stage_counter = getattr(
        stage_module, "count_used_parameters_in_backward", None
    )
    utils_counter = getattr(
        utils_module, "count_used_parameters_in_backward", None
    )
    if stage_counter is None or utils_counter is None:
        raise RuntimeError(
            "DeepSpeed 0.18.5 used-parameter counter API is unavailable"
        )

    existing = getattr(stage_counter, _PATCH_MARKER, None)
    if existing is not None:
        if isinstance(existing, _GraphTaskMemoizedParameterCounter):
            return stage_counter
        raise RuntimeError(
            "DeepSpeed used-parameter counter has an unknown existing patch"
        )

    if stage_counter is not utils_counter:
        raise RuntimeError(
            "DeepSpeed stage_1_and_2 counter binding differs from runtime.utils; "
            "refusing an unverified patch"
        )
    if not callable(stage_counter):
        raise RuntimeError("DeepSpeed used-parameter counter is not callable")
    if getattr(stage_counter, "__name__", None) != (
        "count_used_parameters_in_backward"
    ):
        raise RuntimeError("DeepSpeed used-parameter counter name changed")
    if getattr(stage_counter, "__module__", None) != "deepspeed.runtime.utils":
        raise RuntimeError("DeepSpeed used-parameter counter module changed")
    try:
        signature = inspect.signature(stage_counter)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot inspect the DeepSpeed used-parameter counter signature"
        ) from exc
    parameters = list(signature.parameters.values())
    if len(parameters) != 1 or parameters[0].name != "parameters":
        raise RuntimeError(
            "DeepSpeed used-parameter counter signature changed: "
            f"{signature}"
        )
    return stage_counter


def _install_with_modules(
    *,
    deepspeed_module: object,
    stage_module: object,
    utils_module: object,
    torch_module: object,
    max_cache_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
) -> DeepSpeedGraphTaskCacheReport:
    version = str(getattr(deepspeed_module, "__version__", ""))
    if version != SUPPORTED_DEEPSPEED_VERSION:
        raise RuntimeError(
            "FastWAM DeepSpeed graph-task cache supports exactly DeepSpeed "
            f"{SUPPORTED_DEEPSPEED_VERSION}, observed {version or '<missing>'}"
        )

    check_internal_apis = getattr(
        utils_module, "check_internal_apis_for_count_used_parameters", None
    )
    if not callable(check_internal_apis) or not check_internal_apis():
        raise RuntimeError(
            "Required PyTorch internal APIs for DeepSpeed used-parameter "
            "counting are unavailable"
        )
    torch_c = getattr(torch_module, "_C", None)
    current_graph_task_id = getattr(torch_c, "_current_graph_task_id", None)
    if not callable(current_graph_task_id):
        raise RuntimeError("PyTorch _current_graph_task_id API is unavailable")

    original = _validate_original_binding(
        stage_module=stage_module,
        utils_module=utils_module,
    )
    existing = getattr(original, _PATCH_MARKER, None)
    if isinstance(existing, _GraphTaskMemoizedParameterCounter):
        return DeepSpeedGraphTaskCacheReport(
            enabled=True,
            deepspeed_version=version,
            zero_stage=2,
            max_cache_entries=existing.max_entries,
        )

    memoized_counter = _GraphTaskMemoizedParameterCounter(
        original,
        current_graph_task_id,
        max_entries=max_cache_entries,
    )

    @functools.wraps(original)
    def cached_count_used_parameters_in_backward(parameters: Sequence[Any]) -> int:
        return memoized_counter(parameters)

    setattr(cached_count_used_parameters_in_backward, _PATCH_MARKER, memoized_counter)
    setattr(
        stage_module,
        "count_used_parameters_in_backward",
        cached_count_used_parameters_in_backward,
    )
    return DeepSpeedGraphTaskCacheReport(
        enabled=True,
        deepspeed_version=version,
        zero_stage=2,
        max_cache_entries=max_cache_entries,
    )


def install_deepspeed_zero2_graph_task_cache(
    *,
    max_cache_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
) -> DeepSpeedGraphTaskCacheReport:
    """Install the guarded process-local patch without editing DeepSpeed."""

    import deepspeed
    import deepspeed.runtime.utils as deepspeed_utils
    import deepspeed.runtime.zero.stage_1_and_2 as deepspeed_stage_1_and_2
    import torch

    return _install_with_modules(
        deepspeed_module=deepspeed,
        stage_module=deepspeed_stage_1_and_2,
        utils_module=deepspeed_utils,
        torch_module=torch,
        max_cache_entries=max_cache_entries,
    )


def reset_deepspeed_zero2_graph_task_cache() -> None:
    """Clear the installed cache before starting a new backward call."""

    import deepspeed.runtime.zero.stage_1_and_2 as deepspeed_stage_1_and_2

    binding = getattr(
        deepspeed_stage_1_and_2,
        "count_used_parameters_in_backward",
        None,
    )
    counter = getattr(binding, _PATCH_MARKER, None)
    if not isinstance(counter, _GraphTaskMemoizedParameterCounter):
        raise RuntimeError(
            "DeepSpeed graph-task cache reset requested without the verified patch"
        )
    counter.clear()


def configure_deepspeed_zero2_graph_task_cache(
    *,
    enabled: bool,
    distributed_type: object,
    zero_stage: object,
) -> DeepSpeedGraphTaskCacheReport:
    """Fail closed unless an explicit, supported ZeRO-2 runtime opts in."""

    if not isinstance(enabled, bool):
        raise TypeError("deepspeed_zero2_graph_task_cache_enabled must be boolean")
    if not enabled:
        return DeepSpeedGraphTaskCacheReport(
            enabled=False,
            deepspeed_version=None,
            zero_stage=None,
            max_cache_entries=None,
        )
    if _canonical_distributed_type(distributed_type) != "DEEPSPEED":
        raise RuntimeError(
            "DeepSpeed graph-task cache was enabled outside a DeepSpeed runtime"
        )
    if isinstance(zero_stage, bool):
        raise RuntimeError("DeepSpeed graph-task cache requires ZeRO stage 2")
    try:
        parsed_zero_stage = int(zero_stage)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "DeepSpeed graph-task cache requires an integer ZeRO stage"
        ) from exc
    if parsed_zero_stage != 2:
        raise RuntimeError(
            "DeepSpeed graph-task cache supports only ZeRO stage 2, observed "
            f"{parsed_zero_stage}"
        )
    return install_deepspeed_zero2_graph_task_cache()
