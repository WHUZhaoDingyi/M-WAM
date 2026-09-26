"""Operational compatibility state for exact Trainer resume."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


RESUME_CONTRACT_SCHEMA_VERSION = "fastwam_fresh_process_resume_contract_v6"
PREVIOUS_RESUME_CONTRACT_SCHEMA_VERSION = (
    "fastwam_fresh_process_resume_contract_v5"
)
LEGACY_RESUME_CONTRACT_SCHEMA_VERSION = (
    "fastwam_fresh_process_resume_contract_v4"
)
_EPISODE_SAMPLING_MODE = "one_random_window_per_episode_per_epoch"
_FAILURE_REPLAY_SAMPLING_MODE = "failure_replay_80pct_initial10"
_FAILURE_REPLAY_TASK_VARIANT_NOINITIAL_SAMPLING_MODE = (
    "failure_replay_80pct_task_variant_noinitial"
)

_SAMPLER_OPERATIONAL_KEYS = frozenset(
    {
        "schema_version",
        "sampling_mode",
        "dataset_length",
        "dataset_split_name",
        "episode_count",
        "clips_per_episode",
        "episode_ranges",
        "seed",
        "batch_size_per_rank",
        "world_size",
        "gradient_accumulation_steps",
        "global_batch_size",
        "global_microbatch_size",
        "effective_global_batch_size",
        "initial_fraction_numerator",
        "initial_fraction_denominator",
        "base_fraction_numerator",
        "replay_fraction_numerator",
        "mix_fraction_denominator",
        "replay_task_component_fraction_numerator",
        "replay_task_component_fraction_denominator",
        "initial_window_fraction_numerator",
        "initial_window_fraction_denominator",
        "initial_pool_size",
        "remaining_pool_size",
        "base_pool_size",
        "base_initial_pool_size",
        "base_remaining_pool_size",
        "replay_task_count",
        "replay_failure_task_count",
        "replay_failure_cell_count",
        "replay_initial_pool_size",
        "replay_remaining_pool_size",
        "replay_task_pool_counts",
        "replay_cell_pool_counts",
        "initial_samples_per_epoch",
        "remaining_samples_per_epoch",
        "base_samples_per_epoch",
        "replay_samples_per_epoch",
        "base_initial_samples_per_epoch",
        "base_remaining_samples_per_epoch",
        "replay_initial_samples_per_epoch",
        "replay_remaining_samples_per_epoch",
        "replay_task_samples_per_epoch",
        "replay_cell_samples_per_epoch",
        "stream_samples_per_epoch",
        "selected_samples_per_epoch",
        "padded_samples_per_epoch",
        "padding_samples_per_epoch",
        "global_batches_per_epoch",
        "optimizer_updates_per_epoch",
        "selection_policy",
        "shuffle_policy",
        "tail_policy",
        "epoch_tail_accumulation_policy",
        "failure_manifest",
        "failure_manifest_sha256",
        "pool_digest_algorithm",
        "base_pool_sha256",
        "replay_pool_sha256",
    }
)
_LEGACY_SAMPLER_IDENTITY_KEYS = frozenset(
    {
        "episode_ranges_binding_schema_version",
        "episode_ranges_digest_algorithm",
        "episode_ranges_identity_sha256",
        "episode_range_boundaries_sha256",
    }
)
_REQUIRED_SAMPLER_KEYS = frozenset(
    {
        "schema_version",
        "dataset_length",
        "seed",
        "batch_size_per_rank",
        "world_size",
        "gradient_accumulation_steps",
        "global_batches_per_epoch",
        "optimizer_updates_per_epoch",
        "tail_policy",
    }
)


def _finite_json_object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    try:
        normalized = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain finite JSON values") from exc
    if not isinstance(normalized, dict):
        raise AssertionError("JSON object normalization returned a non-object")
    return normalized


def _normalized_model_math(value: object, *, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _finite_json_object(value, name=name)


def _normalized_episode_ranges(
    value: object,
    *,
    dataset_length: object,
    episode_count: object,
) -> list[list[int]]:
    if not isinstance(value, list):
        raise TypeError("sampler episode_ranges must be an array")
    if (
        isinstance(dataset_length, bool)
        or not isinstance(dataset_length, int)
        or dataset_length <= 0
    ):
        raise ValueError("sampler dataset_length must be a positive integer")
    if (
        isinstance(episode_count, bool)
        or not isinstance(episode_count, int)
        or episode_count <= 0
    ):
        raise ValueError("sampler episode_count must be a positive integer")
    if len(value) != episode_count:
        raise ValueError("sampler episode_ranges count differs from episode_count")

    normalized: list[list[int]] = []
    expected_start = 0
    for position, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise TypeError("each sampler episode range must be a two-item array")
        start, stop = pair
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(stop, bool)
            or not isinstance(stop, int)
        ):
            raise TypeError("sampler episode range bounds must be integers")
        if start != expected_start:
            raise ValueError(
                "sampler episode ranges must be contiguous and ordered: "
                f"range {position} starts at {start}, expected {expected_start}"
            )
        if stop <= start:
            raise ValueError("sampler episode ranges must be non-empty")
        normalized.append([start, stop])
        expected_start = stop
    if expected_start != dataset_length:
        raise ValueError("sampler episode_ranges do not cover dataset_length")
    return normalized


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_failure_replay_state(state: dict[str, Any]) -> None:
    required = {
        "dataset_split_name",
        "episode_ranges",
        "base_fraction_numerator",
        "replay_fraction_numerator",
        "mix_fraction_denominator",
        "initial_fraction_numerator",
        "initial_fraction_denominator",
        "base_initial_pool_size",
        "base_remaining_pool_size",
        "replay_task_count",
        "replay_initial_pool_size",
        "replay_remaining_pool_size",
        "replay_task_pool_counts",
        "base_samples_per_epoch",
        "replay_samples_per_epoch",
        "base_initial_samples_per_epoch",
        "base_remaining_samples_per_epoch",
        "replay_initial_samples_per_epoch",
        "replay_remaining_samples_per_epoch",
        "initial_samples_per_epoch",
        "remaining_samples_per_epoch",
        "stream_samples_per_epoch",
        "selected_samples_per_epoch",
        "padded_samples_per_epoch",
        "failure_manifest",
        "failure_manifest_sha256",
        "pool_digest_algorithm",
        "base_pool_sha256",
        "replay_pool_sha256",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(
            "failure replay sampler state lacks fields: "
            f"{sorted(missing)}"
        )
    if state["dataset_split_name"] != "train":
        raise ValueError("failure replay sampler must bind the train split")
    if (
        state["base_fraction_numerator"],
        state["replay_fraction_numerator"],
        state["mix_fraction_denominator"],
        state["initial_fraction_numerator"],
        state["initial_fraction_denominator"],
    ) != (4, 1, 5, 1, 10):
        raise ValueError("failure replay sampler fractions differ from 80/20+10/90")

    positive_names = (
        "base_initial_pool_size",
        "base_remaining_pool_size",
        "replay_task_count",
        "replay_initial_pool_size",
        "replay_remaining_pool_size",
        "base_samples_per_epoch",
        "replay_samples_per_epoch",
        "base_initial_samples_per_epoch",
        "base_remaining_samples_per_epoch",
        "replay_initial_samples_per_epoch",
        "replay_remaining_samples_per_epoch",
        "initial_samples_per_epoch",
        "remaining_samples_per_epoch",
        "stream_samples_per_epoch",
        "selected_samples_per_epoch",
        "padded_samples_per_epoch",
    )
    counts = {
        name: _positive_int(state[name], name=f"sampler {name}")
        for name in positive_names
    }
    if (
        counts["base_samples_per_epoch"]
        + counts["replay_samples_per_epoch"]
        != counts["stream_samples_per_epoch"]
        or counts["base_initial_samples_per_epoch"]
        + counts["base_remaining_samples_per_epoch"]
        != counts["base_samples_per_epoch"]
        or counts["replay_initial_samples_per_epoch"]
        + counts["replay_remaining_samples_per_epoch"]
        != counts["replay_samples_per_epoch"]
        or counts["initial_samples_per_epoch"]
        != counts["base_initial_samples_per_epoch"]
        + counts["replay_initial_samples_per_epoch"]
        or counts["remaining_samples_per_epoch"]
        != counts["base_remaining_samples_per_epoch"]
        + counts["replay_remaining_samples_per_epoch"]
        or counts["selected_samples_per_epoch"]
        != counts["stream_samples_per_epoch"]
        or counts["padded_samples_per_epoch"]
        != counts["stream_samples_per_epoch"]
    ):
        raise ValueError("failure replay sampler stratum counts are inconsistent")

    raw_task_counts = state["replay_task_pool_counts"]
    if not isinstance(raw_task_counts, list) or len(raw_task_counts) != counts[
        "replay_task_count"
    ]:
        raise ValueError("failure replay per-task pool counts are invalid")
    task_names: list[str] = []
    total_initial = 0
    total_remaining = 0
    for item in raw_task_counts:
        if not isinstance(item, Mapping) or set(item) != {
            "task_name",
            "initial_pool_size",
            "remaining_pool_size",
            "total_pool_size",
        }:
            raise ValueError("failure replay task pool count fields are invalid")
        task_name = item["task_name"]
        if not isinstance(task_name, str) or not task_name:
            raise ValueError("failure replay task pool name is invalid")
        initial = _positive_int(
            item["initial_pool_size"], name="task initial_pool_size"
        )
        remaining = _positive_int(
            item["remaining_pool_size"], name="task remaining_pool_size"
        )
        total = _positive_int(
            item["total_pool_size"], name="task total_pool_size"
        )
        if initial + remaining != total:
            raise ValueError("failure replay task pool totals are inconsistent")
        task_names.append(task_name)
        total_initial += initial
        total_remaining += remaining
    if task_names != sorted(set(task_names)):
        raise ValueError("failure replay task pool names must be unique and sorted")
    if (
        total_initial != counts["replay_initial_pool_size"]
        or total_remaining != counts["replay_remaining_pool_size"]
    ):
        raise ValueError("failure replay aggregate pool sizes are inconsistent")

    manifest = _finite_json_object(
        state["failure_manifest"], name="failure replay manifest"
    )
    if set(manifest) != {"schema_version", "split", "failures"} or (
        manifest["schema_version"] != "fastwam_failure_replay_manifest_v1"
        or manifest["split"] != "train"
        or not isinstance(manifest["failures"], list)
        or not manifest["failures"]
    ):
        raise ValueError("failure replay manifest binding is invalid")
    digest = hashlib.sha256()
    digest.update(b"fastwam-failure-replay-manifest-v1\0")
    digest.update(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if state["failure_manifest_sha256"] != digest.hexdigest():
        raise ValueError("failure replay manifest SHA256 is inconsistent")
    if state["pool_digest_algorithm"] != "ordered_named_int64le_sha256_v1":
        raise ValueError("failure replay pool digest algorithm is unsupported")
    if not _is_sha256(state["base_pool_sha256"]) or not _is_sha256(
        state["replay_pool_sha256"]
    ):
        raise ValueError("failure replay pool SHA256 is invalid")


def _validate_failure_replay_task_variant_noinitial_state(
    state: dict[str, Any],
) -> None:
    """Validate the v1 no-initial-window task×variant sampler binding."""

    required = {
        "dataset_split_name",
        "episode_ranges",
        "base_fraction_numerator",
        "replay_fraction_numerator",
        "mix_fraction_denominator",
        "replay_task_component_fraction_numerator",
        "replay_task_component_fraction_denominator",
        "initial_window_fraction_numerator",
        "initial_window_fraction_denominator",
        "base_pool_size",
        "replay_failure_task_count",
        "replay_failure_cell_count",
        "replay_task_pool_counts",
        "replay_cell_pool_counts",
        "base_samples_per_epoch",
        "replay_samples_per_epoch",
        "replay_task_samples_per_epoch",
        "replay_cell_samples_per_epoch",
        "stream_samples_per_epoch",
        "selected_samples_per_epoch",
        "padded_samples_per_epoch",
        "failure_manifest",
        "failure_manifest_sha256",
        "pool_digest_algorithm",
        "base_pool_sha256",
        "replay_pool_sha256",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(
            "task-variant failure replay sampler state lacks fields: "
            f"{sorted(missing)}"
        )
    if state["dataset_split_name"] != "train":
        raise ValueError(
            "task-variant failure replay sampler must bind the train split"
        )
    if (
        state["base_fraction_numerator"],
        state["replay_fraction_numerator"],
        state["mix_fraction_denominator"],
        state["replay_task_component_fraction_numerator"],
        state["replay_task_component_fraction_denominator"],
        state["initial_window_fraction_numerator"],
        state["initial_window_fraction_denominator"],
    ) != (4, 1, 5, 1, 2, 0, 1):
        raise ValueError(
            "task-variant failure replay sampler fractions differ from "
            "80/20 + 50/50 replay + zero initial-window stratum"
        )

    positive_names = (
        "base_pool_size",
        "replay_failure_task_count",
        "replay_failure_cell_count",
        "base_samples_per_epoch",
        "replay_samples_per_epoch",
        "replay_task_samples_per_epoch",
        "replay_cell_samples_per_epoch",
        "stream_samples_per_epoch",
        "selected_samples_per_epoch",
        "padded_samples_per_epoch",
    )
    counts = {
        name: _positive_int(state[name], name=f"sampler {name}")
        for name in positive_names
    }
    if counts["base_pool_size"] != state["dataset_length"]:
        raise ValueError(
            "task-variant failure replay base pool must cover all legal "
            "train windows"
        )
    if (
        counts["base_samples_per_epoch"]
        + counts["replay_samples_per_epoch"]
        != counts["stream_samples_per_epoch"]
        or counts["replay_task_samples_per_epoch"]
        + counts["replay_cell_samples_per_epoch"]
        != counts["replay_samples_per_epoch"]
        or counts["selected_samples_per_epoch"]
        != counts["stream_samples_per_epoch"]
        or counts["padded_samples_per_epoch"]
        != counts["stream_samples_per_epoch"]
    ):
        raise ValueError(
            "task-variant failure replay sampler stratum counts are "
            "inconsistent"
        )

    raw_task_counts = state["replay_task_pool_counts"]
    if not isinstance(raw_task_counts, list) or len(raw_task_counts) != counts[
        "replay_failure_task_count"
    ]:
        raise ValueError(
            "task-variant failure replay per-task pool counts are invalid"
        )
    task_names: list[str] = []
    task_cell_total = 0
    for item in raw_task_counts:
        if not isinstance(item, Mapping) or set(item) != {
            "task_name",
            "failure_cell_count",
            "total_pool_size",
        }:
            raise ValueError(
                "task-variant failure replay task pool count fields are invalid"
            )
        task_name = item["task_name"]
        if not isinstance(task_name, str) or not task_name:
            raise ValueError(
                "task-variant failure replay task pool name is invalid"
            )
        task_cell_total += _positive_int(
            item["failure_cell_count"],
            name="task-variant failure_cell_count",
        )
        _positive_int(
            item["total_pool_size"],
            name="task-variant task total_pool_size",
        )
        task_names.append(task_name)
    if task_names != sorted(set(task_names)):
        raise ValueError(
            "task-variant failure replay task pool names must be unique and "
            "sorted"
        )
    if task_cell_total != counts["replay_failure_cell_count"]:
        raise ValueError(
            "task-variant failure replay task-cell total is inconsistent"
        )

    raw_cell_counts = state["replay_cell_pool_counts"]
    if not isinstance(raw_cell_counts, list) or len(raw_cell_counts) != counts[
        "replay_failure_cell_count"
    ]:
        raise ValueError(
            "task-variant failure replay per-cell pool counts are invalid"
        )
    cells: list[tuple[str, str]] = []
    cell_identities: list[tuple[str, str, str]] = []
    cells_by_task: dict[str, int] = {}
    windows_by_task: dict[str, int] = {}
    for item in raw_cell_counts:
        if not isinstance(item, Mapping) or set(item) != {
            "task_name",
            "variant",
            "task_config",
            "episode_count",
            "total_pool_size",
        }:
            raise ValueError(
                "task-variant failure replay cell pool count fields are invalid"
            )
        task_name = item["task_name"]
        variant = item["variant"]
        task_config = item["task_config"]
        if any(
            not isinstance(value, str) or not value
            for value in (task_name, variant, task_config)
        ):
            raise ValueError(
                "task-variant failure replay cell identity is invalid"
            )
        _positive_int(
            item["episode_count"],
            name="task-variant cell episode_count",
        )
        pool_size = _positive_int(
            item["total_pool_size"],
            name="task-variant cell total_pool_size",
        )
        cells.append((task_name, variant))
        cell_identities.append((task_name, variant, task_config))
        cells_by_task[task_name] = cells_by_task.get(task_name, 0) + 1
        windows_by_task[task_name] = windows_by_task.get(task_name, 0) + pool_size
    if cells != sorted(set(cells)):
        raise ValueError(
            "task-variant failure replay cells must be unique and sorted"
        )
    if [cells_by_task[name] for name in task_names] != [
        item["failure_cell_count"] for item in raw_task_counts
    ]:
        raise ValueError(
            "task-variant failure replay task/cell counts disagree"
        )
    if [windows_by_task[name] for name in task_names] != [
        item["total_pool_size"] for item in raw_task_counts
    ]:
        raise ValueError(
            "task-variant failure replay task/cell pool sizes disagree"
        )

    manifest = _finite_json_object(
        state["failure_manifest"],
        name="task-variant failure replay manifest",
    )
    if (
        set(manifest) != {"schema_version", "split", "failure_cells"}
        or manifest["schema_version"]
        != "fastwam_failure_replay_task_variant_noinitial_manifest_v1"
        or manifest["split"] != "train"
        or not isinstance(manifest["failure_cells"], list)
        or not manifest["failure_cells"]
    ):
        raise ValueError("task-variant failure replay manifest binding is invalid")
    manifest_cells: list[tuple[str, str, str]] = []
    for item in manifest["failure_cells"]:
        if not isinstance(item, Mapping) or set(item) != {
            "task_name",
            "variant",
            "task_config",
        }:
            raise ValueError(
                "task-variant failure replay manifest cell fields are invalid"
            )
        task_name = item["task_name"]
        variant = item["variant"]
        task_config = item["task_config"]
        if any(
            not isinstance(value, str) or not value
            for value in (task_name, variant, task_config)
        ):
            raise ValueError(
                "task-variant failure replay manifest cell identity is invalid"
            )
        manifest_cells.append((task_name, variant, task_config))
    if manifest_cells != cell_identities:
        raise ValueError(
            "task-variant failure replay manifest and cell pools disagree"
        )
    digest = hashlib.sha256()
    digest.update(b"fastwam-failure-replay-task-variant-noinitial-manifest-v1\0")
    digest.update(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if state["failure_manifest_sha256"] != digest.hexdigest():
        raise ValueError(
            "task-variant failure replay manifest SHA256 is inconsistent"
        )
    if state["pool_digest_algorithm"] != "ordered_named_int64le_sha256_v1":
        raise ValueError(
            "task-variant failure replay pool digest algorithm is unsupported"
        )
    if not _is_sha256(state["base_pool_sha256"]) or not _is_sha256(
        state["replay_pool_sha256"]
    ):
        raise ValueError("task-variant failure replay pool SHA256 is invalid")


def operational_sampler_resume_state(value: object) -> dict[str, Any]:
    """Keep only fields that define the sampler's deterministic stream."""

    binding = _finite_json_object(value, name="sampler resume state")
    unexpected = (
        set(binding)
        - _SAMPLER_OPERATIONAL_KEYS
        - _LEGACY_SAMPLER_IDENTITY_KEYS
    )
    if unexpected:
        raise ValueError(
            "sampler resume state has unsupported fields: "
            f"{sorted(unexpected)}"
        )
    missing = _REQUIRED_SAMPLER_KEYS.difference(binding)
    if missing:
        raise ValueError(
            "sampler resume state lacks operational fields: "
            f"{sorted(missing)}"
        )
    result = {
        key: binding[key]
        for key in sorted(_SAMPLER_OPERATIONAL_KEYS.intersection(binding))
    }
    if "episode_ranges" in result:
        result["episode_ranges"] = _normalized_episode_ranges(
            result["episode_ranges"],
            dataset_length=result.get("dataset_length"),
            episode_count=result.get("episode_count"),
        )
    if result.get("sampling_mode") == _FAILURE_REPLAY_SAMPLING_MODE:
        _validate_failure_replay_state(result)
    if (
        result.get("sampling_mode")
        == _FAILURE_REPLAY_TASK_VARIANT_NOINITIAL_SAMPLING_MODE
    ):
        _validate_failure_replay_task_variant_noinitial_state(result)
    return result


def _validate_legacy_episode_identity(
    saved_sampler: Mapping[str, Any],
    current_sampler_binding: object | None,
) -> None:
    saved_identity_keys = _LEGACY_SAMPLER_IDENTITY_KEYS.intersection(
        saved_sampler
    )
    if saved_sampler.get("sampling_mode") != _EPISODE_SAMPLING_MODE:
        return
    if saved_identity_keys and saved_identity_keys != _LEGACY_SAMPLER_IDENTITY_KEYS:
        raise ValueError(
            "legacy episode sampler identity fields are incomplete"
        )
    if not saved_identity_keys:
        return
    if current_sampler_binding is None:
        raise ValueError(
            "legacy episode sampler identity requires the current sampler binding"
        )
    current = _finite_json_object(
        current_sampler_binding,
        name="current sampler binding",
    )
    for key in sorted(_LEGACY_SAMPLER_IDENTITY_KEYS):
        if current.get(key) != saved_sampler.get(key):
            raise ValueError(
                "legacy episode sampler identity differs from the current "
                f"sampler: {key}"
            )


def normalize_saved_resume_contract(
    value: object,
    *,
    current_sampler_binding: object | None = None,
    current_model_math: object | None = None,
    current_data_math: object | None = None,
) -> dict[str, Any]:
    """Normalize current or historical Trainer state to the v6 contract."""

    contract = _finite_json_object(value, name="trainer resume contract")
    schema = contract.get("schema_version")
    current_keys = {
        "schema_version",
        "sampler",
        "runtime_topology",
        "training",
        "model_math",
        "data_math",
    }
    previous_keys = current_keys - {"model_math", "data_math"}
    legacy_keys = {*previous_keys, "dataset"}
    if schema == RESUME_CONTRACT_SCHEMA_VERSION:
        if set(contract) != current_keys:
            raise ValueError("current trainer resume contract keys drifted")
    elif schema == PREVIOUS_RESUME_CONTRACT_SCHEMA_VERSION:
        if set(contract) != previous_keys:
            raise ValueError("previous trainer resume contract keys drifted")
    elif schema == LEGACY_RESUME_CONTRACT_SCHEMA_VERSION:
        if set(contract) != legacy_keys:
            raise ValueError("legacy trainer resume contract keys drifted")
        # Historical dataset/index/source identity was governance evidence,
        # not state needed by model/optimizer/scheduler/sampler restoration.
    else:
        raise ValueError("trainer resume contract schema is unsupported")

    saved_sampler = _finite_json_object(
        contract.get("sampler"),
        name="saved sampler resume state",
    )
    if schema == LEGACY_RESUME_CONTRACT_SCHEMA_VERSION:
        _validate_legacy_episode_identity(
            saved_sampler,
            current_sampler_binding,
        )
    sampler = operational_sampler_resume_state(saved_sampler)
    if (
        schema
        in {
            LEGACY_RESUME_CONTRACT_SCHEMA_VERSION,
            PREVIOUS_RESUME_CONTRACT_SCHEMA_VERSION,
        }
        and sampler.get("sampling_mode") == _EPISODE_SAMPLING_MODE
        and "episode_ranges" not in sampler
    ):
        if current_sampler_binding is None:
            raise ValueError(
                "historical episode resume requires current episode ranges"
            )
        current_sampler = operational_sampler_resume_state(
            current_sampler_binding
        )
        if "episode_ranges" not in current_sampler:
            raise ValueError(
                "current episode sampler binding lacks episode_ranges"
            )
        sampler["episode_ranges"] = current_sampler["episode_ranges"]

    model_math = (
        _normalized_model_math(
            contract.get("model_math"),
            name="saved model math resume contract",
        )
        if schema == RESUME_CONTRACT_SCHEMA_VERSION
        else _normalized_model_math(
            current_model_math,
            name="current model math resume contract",
        )
    )
    data_math = (
        _normalized_model_math(
            contract.get("data_math"),
            name="saved data math resume contract",
        )
        if schema == RESUME_CONTRACT_SCHEMA_VERSION
        else _normalized_model_math(
            current_data_math,
            name="current data math resume contract",
        )
    )
    return {
        "schema_version": RESUME_CONTRACT_SCHEMA_VERSION,
        "sampler": sampler,
        "runtime_topology": _finite_json_object(
            contract.get("runtime_topology"),
            name="resume runtime topology",
        ),
        "training": _finite_json_object(
            contract.get("training"),
            name="resume training state",
        ),
        "model_math": model_math,
        "data_math": data_math,
    }


__all__ = [
    "LEGACY_RESUME_CONTRACT_SCHEMA_VERSION",
    "PREVIOUS_RESUME_CONTRACT_SCHEMA_VERSION",
    "RESUME_CONTRACT_SCHEMA_VERSION",
    "normalize_saved_resume_contract",
    "operational_sampler_resume_state",
]
