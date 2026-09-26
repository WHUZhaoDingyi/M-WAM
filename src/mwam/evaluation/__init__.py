"""Standalone evaluation utilities."""

from importlib import import_module


_MOMAP_EVALUATION_EXPORTS = frozenset(
    {
        "MOMAP_HEIGHT",
        "MOMAP_WIDTH",
        "MOMAP_VIEW_NAMES",
        "MOMAP_VIEW_SLICES",
        "RobotwinMoMapEvaluationAccumulator",
    }
)


def __getattr__(name: str):
    if name not in _MOMAP_EVALUATION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.robotwin_momap")
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "MOMAP_HEIGHT",
    "MOMAP_WIDTH",
    "MOMAP_VIEW_NAMES",
    "MOMAP_VIEW_SLICES",
    "RobotwinMoMapEvaluationAccumulator",
]
