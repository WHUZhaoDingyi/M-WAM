"""Fail-closed identity contract for one official RoboTwin source scene."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


SCENE_IDENTITY_CONTRACT_VERSION = "robotwin_source_scene_identity_v1"


class SceneIdentityError(ValueError):
    """Raised when a live replay scene differs from the official episode."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scene_string(value: Any, *, label: str) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string, got {type(value).__name__}")
    return value


def _task_info_string(value: Any, *, label: str) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value
    # Some RoboTwin tasks keep ArmTag objects until JSON serialization.  Only
    # this audited stable representation is accepted; arbitrary objects are
    # never stringified because that could hide scene drift.
    if value.__class__.__name__ == "ArmTag":
        normalized = str(value)
        if normalized in {"left", "right"}:
            return normalized
    raise ValueError(
        f"{label} must be an official string or ArmTag(left/right), "
        f"got {type(value).__name__}"
    )


def normalize_scene_info_record(
    value: Any,
    *,
    label: str = "scene_info episode",
) -> dict[str, Any]:
    """Normalize the three stable fields of an official episode record.

    Clutter order is significant.  Object type and index remain exact strings,
    including zero padding.  Instruction/language fields outside these stable
    fields are intentionally excluded from the identity contract.
    """

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    required = {"cluttered_table_info", "texture_info", "info"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{label} missing required fields: {missing}")

    clutter_value = value["cluttered_table_info"]
    if not isinstance(clutter_value, (list, tuple)):
        raise ValueError(f"{label}.cluttered_table_info must be a sequence")
    clutter: list[dict[str, str]] = []
    for index, item in enumerate(clutter_value):
        item_label = f"{label}.cluttered_table_info[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{item_label} must be an object")
        item_required = {"object_type", "object_index"}
        item_missing = sorted(item_required - set(item))
        if item_missing:
            raise ValueError(f"{item_label} missing fields: {item_missing}")
        clutter.append(
            {
                "object_type": _scene_string(
                    item["object_type"], label=f"{item_label}.object_type"
                ),
                "object_index": _scene_string(
                    item["object_index"], label=f"{item_label}.object_index"
                ),
            }
        )

    texture_value = value["texture_info"]
    if not isinstance(texture_value, Mapping):
        raise ValueError(f"{label}.texture_info must be an object")
    texture_required = {"wall_texture", "table_texture"}
    texture_missing = sorted(texture_required - set(texture_value))
    if texture_missing:
        raise ValueError(f"{label}.texture_info missing fields: {texture_missing}")
    texture: dict[str, str | None] = {}
    for key in ("wall_texture", "table_texture"):
        candidate = texture_value[key]
        texture[key] = (
            None
            if candidate is None
            else _scene_string(candidate, label=f"{label}.texture_info.{key}")
        )

    info_value = value["info"]
    if not isinstance(info_value, Mapping):
        raise ValueError(f"{label}.info must be an object")
    info: dict[str, str] = {}
    for raw_key, raw_value in info_value.items():
        key = _scene_string(raw_key, label=f"{label}.info key")
        info[key] = _task_info_string(
            raw_value, label=f"{label}.info[{key!r}]"
        )

    return {
        "cluttered_table_info": clutter,
        "texture_info": texture,
        "info": info,
    }


def compare_scene_identity_setup(
    expected_scene_info: Mapping[str, Any], live_task_info: Any
) -> dict[str, Any]:
    """Compare clutter and texture immediately after ``setup_demo``."""

    expected = normalize_scene_info_record(expected_scene_info, label="expected scene")
    report: dict[str, Any] = {
        "phase": "after_setup_demo",
        "compared_fields": ["cluttered_table_info", "texture_info"],
        "passed": False,
    }
    try:
        if not isinstance(live_task_info, Mapping):
            raise ValueError("live task_env.info must be an object")
        actual = normalize_scene_info_record(
            {
                "cluttered_table_info": live_task_info.get("cluttered_table_info"),
                "texture_info": live_task_info.get("texture_info"),
                "info": {},
            },
            label="live scene after setup_demo",
        )
        expected_subset = {
            "cluttered_table_info": expected["cluttered_table_info"],
            "texture_info": expected["texture_info"],
        }
        actual_subset = {
            "cluttered_table_info": actual["cluttered_table_info"],
            "texture_info": actual["texture_info"],
        }
        report.update(
            {
                "expected": expected_subset,
                "actual": actual_subset,
                "expected_sha256": canonical_json_sha256(expected_subset),
                "actual_sha256": canonical_json_sha256(actual_subset),
                "passed": actual_subset == expected_subset,
            }
        )
        if not report["passed"]:
            report["error"] = "live clutter/texture differs from official episode"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def compare_scene_identity_replay(
    expected_scene_info: Mapping[str, Any], live_task_info: Any
) -> dict[str, Any]:
    """Compare stable task placeholders after ``play_once``."""

    expected = normalize_scene_info_record(expected_scene_info, label="expected scene")
    report: dict[str, Any] = {
        "phase": "after_play_once",
        "compared_fields": ["info"],
        "ignored_unstable_fields": ["instruction", "language"],
        "passed": False,
    }
    try:
        if not isinstance(live_task_info, Mapping):
            raise ValueError("live task_env.info must be an object")
        actual_record = normalize_scene_info_record(
            {
                "cluttered_table_info": [],
                "texture_info": {"wall_texture": None, "table_texture": None},
                "info": live_task_info.get("info"),
            },
            label="live scene after play_once",
        )
        expected_info = expected["info"]
        actual_info = actual_record["info"]
        report.update(
            {
                "expected": {"info": expected_info},
                "actual": {"info": actual_info},
                "expected_sha256": canonical_json_sha256(expected_info),
                "actual_sha256": canonical_json_sha256(actual_info),
                "passed": actual_info == expected_info,
            }
        )
        if not report["passed"]:
            report["error"] = "live task placeholders differ from official episode"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def build_scene_identity_report(
    *,
    episode_key: str,
    expected_scene_info: Mapping[str, Any],
    pre_replay: Mapping[str, Any],
    post_replay: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected = normalize_scene_info_record(expected_scene_info, label="expected scene")
    actual_normalized = None
    if post_replay is not None:
        setup_actual = pre_replay.get("actual")
        replay_actual = post_replay.get("actual")
        if isinstance(setup_actual, Mapping) and isinstance(replay_actual, Mapping):
            actual_normalized = {
                "cluttered_table_info": setup_actual.get("cluttered_table_info"),
                "texture_info": setup_actual.get("texture_info"),
                "info": replay_actual.get("info"),
            }
    passed = (
        pre_replay.get("passed") is True
        and post_replay is not None
        and post_replay.get("passed") is True
        and actual_normalized == expected
    )
    return {
        "schema_version": SCENE_IDENTITY_CONTRACT_VERSION,
        "episode_key": str(episode_key),
        "expected_normalized": expected,
        "expected_sha256": canonical_json_sha256(expected),
        "actual_normalized": actual_normalized,
        "actual_sha256": (
            canonical_json_sha256(actual_normalized)
            if actual_normalized is not None
            else None
        ),
        "pre_replay": dict(pre_replay),
        "post_replay": dict(post_replay) if post_replay is not None else None,
        "comparison": {
            "exact_clutter_order": True,
            "exact_object_type": True,
            "exact_object_index_including_zero_padding": True,
            "exact_texture": True,
            "exact_stable_task_info": True,
            "ignored_unstable_fields": ["instruction", "language"],
        },
        "passed": passed,
    }

def validate_complete_scene_identity(value: Any) -> dict[str, Any]:
    """Recompute and validate every declaration in a completed identity object."""

    if not isinstance(value, Mapping):
        raise ValueError("source_scene_identity must be an object")
    if value.get("schema_version") != SCENE_IDENTITY_CONTRACT_VERSION:
        raise ValueError(
            f"unsupported source scene identity schema: {value.get('schema_version')!r}"
        )
    expected = normalize_scene_info_record(
        value.get("expected_normalized"), label="scene identity expected"
    )
    actual = normalize_scene_info_record(
        value.get("actual_normalized"), label="scene identity actual"
    )
    expected_hash = canonical_json_sha256(expected)
    actual_hash = canonical_json_sha256(actual)
    if value.get("expected_sha256") != expected_hash:
        raise ValueError("source scene expected_sha256 does not match normalized data")
    if value.get("actual_sha256") != actual_hash:
        raise ValueError("source scene actual_sha256 does not match normalized data")
    if expected != actual or expected_hash != actual_hash:
        raise ValueError("source scene expected and actual records differ")
    if value.get("passed") is not True:
        raise ValueError("source scene identity passed is not true")
    for phase in ("pre_replay", "post_replay"):
        report = value.get(phase)
        if not isinstance(report, Mapping) or report.get("passed") is not True:
            raise ValueError(f"source scene identity {phase} did not pass")
    pre = value["pre_replay"]
    expected_pre = {
        "cluttered_table_info": expected["cluttered_table_info"],
        "texture_info": expected["texture_info"],
    }
    if pre.get("expected") != expected_pre or pre.get("actual") != expected_pre:
        raise ValueError("source scene pre_replay exact objects differ")
    if (
        pre.get("expected_sha256") != canonical_json_sha256(expected_pre)
        or pre.get("actual_sha256") != canonical_json_sha256(expected_pre)
    ):
        raise ValueError("source scene pre_replay hashes differ")
    post = value["post_replay"]
    expected_post = {"info": expected["info"]}
    if post.get("expected") != expected_post or post.get("actual") != expected_post:
        raise ValueError("source scene post_replay exact objects differ")
    if (
        post.get("expected_sha256") != canonical_json_sha256(expected["info"])
        or post.get("actual_sha256") != canonical_json_sha256(expected["info"])
    ):
        raise ValueError("source scene post_replay hashes differ")
    comparison = value.get("comparison")
    required_exact = (
        "exact_clutter_order",
        "exact_object_type",
        "exact_object_index_including_zero_padding",
        "exact_texture",
        "exact_stable_task_info",
    )
    if not isinstance(comparison, Mapping) or any(
        comparison.get(key) is not True for key in required_exact
    ):
        raise ValueError("source scene identity exact comparison contract is incomplete")
    return dict(value)



def require_scene_identity_pass(report: Mapping[str, Any]) -> None:
    if report.get("passed") is not True:
        raise SceneIdentityError(
            "live replay source-scene identity gate failed", report
        )


__all__ = [
    "SCENE_IDENTITY_CONTRACT_VERSION",
    "SceneIdentityError",
    "build_scene_identity_report",
    "canonical_json_sha256",
    "compare_scene_identity_replay",
    "compare_scene_identity_setup",
    "normalize_scene_info_record",
    "require_scene_identity_pass",
    "validate_complete_scene_identity",
]
