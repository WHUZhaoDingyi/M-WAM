"""Learning-rate and exposure contract for a fresh optimizer after expansion."""
from __future__ import annotations

import math
from collections.abc import Mapping


def validate_reset(value, *, max_steps, batch_size, accumulation):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("sample_progress_reset must be a mapping")
    result = dict(value)
    expected = {"source_step", "source_max_steps", "source_warmup_steps",
                "source_batch", "target_batch", "rewarmup_steps"}
    extension = {"cumulative_target_step", "extension_start_step"}
    if set(result) not in (expected, expected | extension):
        raise ValueError("sample_progress_reset fields differ")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
           for v in result.values()):
        raise ValueError("sample_progress_reset requires nonnegative integers")
    if not (result["source_warmup_steps"] <= result["source_step"]
            < result["source_max_steps"]):
        raise ValueError("source cursor must be in cosine decay")
    if (result["source_batch"], result["target_batch"]) != (384, 768):
        raise ValueError("reset stage requires batch 384 -> 768")
    if batch_size != 24 or accumulation != 1:
        raise ValueError("reset stage requires B24/GA1")
    remaining = (result["source_max_steps"] - result["source_step"]) * 384
    if "cumulative_target_step" in result:
        if result["cumulative_target_step"] != 30000:
            raise ValueError("only the approved cumulative30000 target is supported")
        if max_steps != result["cumulative_target_step"] - result["source_step"]:
            raise ValueError("extended max_steps must reach cumulative target")
        if not 0 < result["extension_start_step"] < math.ceil(remaining / 768):
            raise ValueError("extension cursor must bind an unfinished original stage")
    elif max_steps != math.ceil(remaining / 768):
        raise ValueError("reset stage max_steps must preserve remaining exposure")
    if not 0 < result["rewarmup_steps"] < max_steps:
        raise ValueError("invalid reset warmup")
    return result


def reset_learning_rate(step, contract, *, peak, floor, start):
    """After rewarmup, exactly follow the source cosine at equal sample exposure."""
    step = max(int(step), 0)
    if "cumulative_target_step" in contract:
        anchor = max(contract["extension_start_step"], contract["rewarmup_steps"])
        original = {k: v for k, v in contract.items()
                    if k not in {"cumulative_target_step", "extension_start_step"}}
        if step <= anchor:
            return reset_learning_rate(step, original, peak=peak, floor=floor, start=start)
        anchor_lr = reset_learning_rate(anchor, original, peak=peak, floor=floor, start=start)
        target = contract["cumulative_target_step"] - contract["source_step"]
        progress = min((step - anchor) / (target - anchor), 1.)
        return floor + (anchor_lr - floor) * .5 * (1. + math.cos(math.pi * progress))
    old_step = contract["source_step"] + step * (
        contract["target_batch"] / contract["source_batch"])
    progress = min(max((old_step - contract["source_warmup_steps"]) /
        (contract["source_max_steps"] - contract["source_warmup_steps"]), 0.), 1.)
    scheduled = floor + (peak - floor) * .5 * (1. + math.cos(math.pi * progress))
    alpha = min(step / contract["rewarmup_steps"], 1.)
    return start + alpha * (scheduled - start)


def allows_step_budget_extension(saved, current, migration, *, source_state_dir, source_global_step):
    """Allow only an explicitly source-bound budget/LR extension; all other state is exact."""
    from pathlib import Path
    import copy
    if not isinstance(migration, Mapping) or set(migration) != {
            "source_state_dir", "source_global_step", "cumulative_target_step"}:
        return False
    if source_state_dir is None or source_global_step != migration["source_global_step"]:
        return False
    if Path(migration["source_state_dir"]).resolve() != Path(source_state_dir).resolve():
        return False
    try:
        original = saved["training"]["sample_progress_reset"]
        extended = current["training"]["sample_progress_reset"]
        if "cumulative_target_step" in original or migration["cumulative_target_step"] != 30000:
            return False
        if extended != dict(original, cumulative_target_step=30000,
                            extension_start_step=source_global_step):
            return False
        validate_reset(original, max_steps=saved["training"]["max_steps"], batch_size=24, accumulation=1)
        validate_reset(extended, max_steps=current["training"]["max_steps"], batch_size=24, accumulation=1)
        expected = copy.deepcopy(saved)
        expected["training"]["sample_progress_reset"] = extended
        expected["training"]["max_steps"] = 30000 - original["source_step"]
        return expected == current
    except (KeyError, TypeError, ValueError):
        return False
