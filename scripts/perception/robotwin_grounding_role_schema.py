#!/usr/bin/env python3
"""Deterministic RoboTwin task roles for GroundingDINO/SAM2 caching.

The formal Route-B prompt index contains expanded instructions, but its
``{A}``/``{B}`` task-info slots do not have a task-independent meaning.  This
module therefore uses a versioned, task-name keyed schema instead of guessing
roles from slot order or asking an external language model.

Only the current task name and instruction are consumed.  No RGB frame,
future frame, simulator depth, actor segmentation, or ground-truth action is
read here.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "fastwam_robotwin_grounding_role_schema_v1"
ROLE_NAMES = (
    "manipulated_target",
    "destination_or_reference",
    "tool",
    "robot",
)
ROBOT_PHRASES = (
    "left robot arm",
    "right robot arm",
    "robotic gripper",
    "robot manipulator",
)


@dataclass(frozen=True)
class TaskRoleSpec:
    """One task's deployable object-role contract.

    The phrase tuples are category fallbacks and synonyms, not task-info slot
    names.  ``expected_*`` counts physical instances, so several synonymous
    phrases may legitimately describe one expected instance.
    """

    manipulated_target: tuple[str, ...] = ()
    destination_or_reference: tuple[str, ...] = ()
    tool: tuple[str, ...] = ()
    expected_target: int = 0
    expected_destination_or_reference: int = 0
    expected_tool: int = 0


def _spec(
    *,
    target: Sequence[str] = (),
    destination: Sequence[str] = (),
    tool: Sequence[str] = (),
    target_count: int = 0,
    destination_count: int = 0,
    tool_count: int = 0,
) -> TaskRoleSpec:
    return TaskRoleSpec(
        manipulated_target=tuple(target),
        destination_or_reference=tuple(destination),
        tool=tuple(tool),
        expected_target=target_count,
        expected_destination_or_reference=destination_count,
        expected_tool=tool_count,
    )


# This is intentionally an explicit 50-task table.  Role assignment cannot be
# inferred from Route-B's {A}/{B}/{C}/{D} order: for example move_can_pot uses
# B as the manipulated can and A as the reference pot, whereas
# place_object_basket uses A as the manipulated toy car and B as the basket.
ROLE_SCHEMA: dict[str, TaskRoleSpec] = {
    "adjust_bottle": _spec(target=("bottle",), target_count=1),
    "beat_block_hammer": _spec(
        destination=("block",), tool=("hammer",),
        destination_count=1, tool_count=1,
    ),
    "blocks_ranking_rgb": _spec(
        target=("red block", "green block", "blue block"), target_count=3,
    ),
    "blocks_ranking_size": _spec(
        target=("large block", "medium block", "small block"), target_count=3,
    ),
    "click_alarmclock": _spec(
        target=("alarm clock", "alarm clock button"), target_count=1,
    ),
    "click_bell": _spec(target=("bell", "bell button"), target_count=1),
    "dump_bin_bigbin": _spec(
        target=("table trash bin",), destination=("big bin", "large bin"),
        target_count=1, destination_count=1,
    ),
    "grab_roller": _spec(target=("roller",), target_count=1),
    "handover_block": _spec(target=("red block", "block"), target_count=1),
    "handover_mic": _spec(target=("microphone",), target_count=1),
    "hanging_mug": _spec(
        target=("mug",), destination=("rack",),
        target_count=1, destination_count=1,
    ),
    "lift_pot": _spec(target=("kitchen pot", "pot"), target_count=1),
    "move_can_pot": _spec(
        target=("sauce can", "can"), destination=("kitchen pot", "pot"),
        target_count=1, destination_count=1,
    ),
    "move_pillbottle_pad": _spec(
        target=("pill bottle", "bottle"), destination=("pad", "mat"),
        target_count=1, destination_count=1,
    ),
    "move_playingcard_away": _spec(
        target=("playing cards box", "playing cards", "card box"),
        target_count=1,
    ),
    "move_stapler_pad": _spec(
        target=("stapler",), destination=("mat", "pad"),
        target_count=1, destination_count=1,
    ),
    "open_laptop": _spec(
        target=("laptop", "laptop lid"), target_count=1,
    ),
    "open_microwave": _spec(
        target=("microwave", "microwave door"), target_count=1,
    ),
    "pick_diverse_bottles": _spec(target=("bottle",), target_count=2),
    "pick_dual_bottles": _spec(target=("bottle",), target_count=2),
    "place_a2b_left": _spec(
        target=("tea box",), destination=("stapler",),
        target_count=1, destination_count=1,
    ),
    "place_a2b_right": _spec(
        target=("tea box",), destination=("soap",),
        target_count=1, destination_count=1,
    ),
    "place_bread_basket": _spec(
        target=("bread", "loaf"), destination=("bread basket", "basket"),
        target_count=2, destination_count=1,
    ),
    "place_bread_skillet": _spec(
        target=("bread",), destination=("skillet", "cooking pan"),
        target_count=1, destination_count=1,
    ),
    "place_burger_fries": _spec(
        target=("hamburger", "fries box"), destination=("tray",),
        target_count=2, destination_count=1,
    ),
    "place_can_basket": _spec(
        target=("can",), destination=("basket",),
        target_count=1, destination_count=1,
    ),
    "place_cans_plasticbox": _spec(
        target=("can",), destination=("plastic box",),
        target_count=2, destination_count=1,
    ),
    "place_container_plate": _spec(
        target=("bowl", "container"), destination=("plate",),
        target_count=1, destination_count=1,
    ),
    "place_dual_shoes": _spec(
        target=("shoe",), destination=("shoe box",),
        target_count=2, destination_count=1,
    ),
    "place_empty_cup": _spec(
        target=("cup",), destination=("coaster",),
        target_count=1, destination_count=1,
    ),
    "place_fan": _spec(
        target=("fan",), destination=("mat",),
        target_count=1, destination_count=1,
    ),
    "place_mouse_pad": _spec(
        target=("mouse",), destination=("mat", "mouse pad"),
        target_count=1, destination_count=1,
    ),
    "place_object_basket": _spec(
        target=("toy car",), destination=("basket",),
        target_count=1, destination_count=1,
    ),
    "place_object_scale": _spec(
        target=("stapler",), destination=("electronic scale", "scale"),
        target_count=1, destination_count=1,
    ),
    "place_object_stand": _spec(
        target=("stapler",), destination=("display stand", "stand"),
        target_count=1, destination_count=1,
    ),
    "place_phone_stand": _spec(
        target=("phone",), destination=("phone stand", "phone rack"),
        target_count=1, destination_count=1,
    ),
    "place_shoe": _spec(
        target=("shoe",), destination=("mat",),
        target_count=1, destination_count=1,
    ),
    "press_stapler": _spec(target=("stapler",), target_count=1),
    "put_bottles_dustbin": _spec(
        target=("bottle",), destination=("trash bin", "dustbin"),
        target_count=3, destination_count=1,
    ),
    "put_object_cabinet": _spec(
        target=("toy car",),
        destination=("cabinet", "cabinet drawer"),
        target_count=1,
        destination_count=1,
    ),
    "rotate_qrcode": _spec(
        target=("payment sign", "QR code sign"), target_count=1,
    ),
    "scan_object": _spec(
        target=("tea box",), tool=("barcode scanner", "scanner"),
        target_count=1, tool_count=1,
    ),
    "shake_bottle": _spec(target=("bottle",), target_count=1),
    "shake_bottle_horizontally": _spec(
        target=("bottle",), target_count=1,
    ),
    "stack_blocks_three": _spec(
        target=("red block", "green block", "blue block"), target_count=3,
    ),
    "stack_blocks_two": _spec(
        target=("red block", "green block"), target_count=2,
    ),
    "stack_bowls_three": _spec(target=("bowl",), target_count=3),
    "stack_bowls_two": _spec(target=("bowl",), target_count=2),
    "stamp_seal": _spec(
        destination=("gray mat", "gray pad"), tool=("seal", "stamp"),
        destination_count=1, tool_count=1,
    ),
    "turn_switch": _spec(target=("switch",), target_count=1),
}


EXPECTED_TASK_NAMES = frozenset(
    {
        "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb",
        "blocks_ranking_size", "click_alarmclock", "click_bell",
        "dump_bin_bigbin", "grab_roller", "handover_block", "handover_mic",
        "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
        "move_playingcard_away", "move_stapler_pad", "open_laptop",
        "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
        "place_a2b_left", "place_a2b_right", "place_bread_basket",
        "place_bread_skillet", "place_burger_fries", "place_can_basket",
        "place_cans_plasticbox", "place_container_plate", "place_dual_shoes",
        "place_empty_cup", "place_fan", "place_mouse_pad",
        "place_object_basket", "place_object_scale", "place_object_stand",
        "place_phone_stand", "place_shoe", "press_stapler",
        "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode",
        "scan_object", "shake_bottle", "shake_bottle_horizontally",
        "stack_blocks_three", "stack_blocks_two", "stack_bowls_three",
        "stack_bowls_two", "stamp_seal", "turn_switch",
    }
)


_ALIASES: dict[str, tuple[str, ...]] = {
    "alarm clock": ("alarm clock", "alarm-clock"),
    "alarm clock button": ("alarm clock button", "alarm-clock button"),
    "barcode scanner": ("barcode scanner",),
    "big bin": ("big bin", "bin"),
    "bottle": ("bottle", "bottles", "bottle shape design"),
    "bread basket": ("bread basket", "breadbasket"),
    "cabinet drawer": ("cabinet drawer", "cabinet's drawer"),
    "cooking pan": ("cooking pan", "pan"),
    "display stand": ("display stand", "displaystand"),
    "electronic scale": ("electronic scale", "electronicscale"),
    "fries box": ("fries box", "french fries box"),
    "hamburger": ("hamburger", "hamburg"),
    "kitchen pot": ("kitchen pot", "kitchenpot"),
    "large bin": ("large bin", "bin"),
    "mouse pad": ("mouse pad", "mat"),
    "payment sign": ("payment sign", "paymentsign"),
    "phone stand": ("phone stand", "phone rack", "phonestand"),
    "pill bottle": ("pill bottle", "pillbottle", "bottle"),
    "plastic box": ("plastic box", "plasticbox"),
    "playing cards": ("playing cards", "cards"),
    "playing cards box": ("playing cards box", "box with cards"),
    "QR code sign": ("QR code sign", "QR code", "paymentsign"),
    "robotic gripper": ("robotic gripper", "gripper"),
    "sauce can": ("sauce can",),
    "shoe box": ("shoe box", "shoe-box"),
    "table trash bin": ("table trash bin", "tabletrashbin"),
    "tea box": ("tea box",),
    "toy car": ("toy car", "toycar"),
    "trash bin": ("trash bin", "dustbin", "bin"),
}

_DESCRIPTORS = frozenset(
    {
        "beige", "black", "blue", "brown", "cyan", "gray", "green",
        "grey", "orange", "red", "silver", "teal", "white", "yellow",
        "large", "long", "medium", "small", "thin", "hand-sized",
        "metal", "plastic", "stone", "wood", "wooden", "smooth", "soft",
        "round", "rounded",
    }
)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def _normalized_words(value: str) -> tuple[str, ...]:
    return tuple(word.lower() for word in _TOKEN_PATTERN.findall(value))


def _alias_words(category: str) -> tuple[tuple[str, ...], ...]:
    raw_aliases = (category, *_ALIASES.get(category, ()))
    result: list[tuple[str, ...]] = []
    for alias in raw_aliases:
        words = _normalized_words(alias)
        if words and words not in result:
            result.append(words)
    return tuple(result)


def _described_phrases(instruction: str, category: str) -> tuple[str, ...]:
    """Extract descriptor-rich noun phrases ending in a known category.

    This intentionally performs no free-form language generation.  It only
    keeps adjacent color/material/shape/size words from the frozen expanded
    Route-B instruction and falls back to the schema category.
    """

    words = _normalized_words(instruction)
    results: list[str] = []
    for alias in _alias_words(category):
        width = len(alias)
        for start in range(0, len(words) - width + 1):
            if words[start : start + width] != alias:
                continue
            left = start
            while left > 0 and words[left - 1] in _DESCRIPTORS:
                left -= 1
            phrase = " ".join((*words[left:start], *alias))
            if phrase and phrase not in results:
                results.append(phrase)
    if category not in results:
        results.append(category)
    return tuple(results)


def _role_phrases(instruction: str, categories: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for category in categories:
        for phrase in _described_phrases(instruction, category):
            if phrase not in result:
                result.append(phrase)
    return tuple(result)


def build_grounding_role_prompts(
    task_name: str,
    instruction: str,
) -> dict[str, Any]:
    """Return deterministic role-separated GroundingDINO phrases.

    The returned role lists can be concatenated into one detector text prompt
    for efficiency as long as the caller retains the phrase-to-role mapping.
    SAM2 masks and track IDs must remain role separated.
    """

    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError("task_name must be a non-empty string")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    normalized_task = task_name.strip()
    spec = ROLE_SCHEMA.get(normalized_task)
    if spec is None:
        raise KeyError(f"unsupported RoboTwin task: {normalized_task!r}")
    normalized_instruction = " ".join(instruction.strip().split())

    roles = {
        "manipulated_target": _role_phrases(
            normalized_instruction, spec.manipulated_target
        ),
        "destination_or_reference": _role_phrases(
            normalized_instruction, spec.destination_or_reference
        ),
        "tool": _role_phrases(normalized_instruction, spec.tool),
        "robot": ROBOT_PHRASES,
    }
    expected_count = {
        "manipulated_target": spec.expected_target,
        "destination_or_reference": spec.expected_destination_or_reference,
        "tool": spec.expected_tool,
        "robot": 2,
    }
    phrase_to_roles: dict[str, list[str]] = {}
    combined: list[str] = []
    for role in ROLE_NAMES:
        for phrase in roles[role]:
            phrase_to_roles.setdefault(phrase, []).append(role)
            if phrase not in combined:
                combined.append(phrase)

    return {
        "schema_version": SCHEMA_VERSION,
        "task_name": normalized_task,
        "instruction": normalized_instruction,
        "roles": {key: list(value) for key, value in roles.items()},
        "expected_count": expected_count,
        "grounding_dino_text": " . ".join(combined) + " .",
        "phrase_to_roles": phrase_to_roles,
        "causal_input_contract": {
            "uses_gt": False,
            "uses_future_frames": False,
            "inputs": ["task_name", "current_episode_instruction"],
        },
    }


def validate_role_schema(
    expected_task_names: Iterable[str] = EXPECTED_TASK_NAMES,
) -> dict[str, Any]:
    """Pure self-check for exact coverage and internally valid role counts."""

    expected = frozenset(expected_task_names)
    actual = frozenset(ROLE_SCHEMA)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    errors: list[str] = []
    if missing:
        errors.append(f"missing tasks: {missing}")
    if extra:
        errors.append(f"unexpected tasks: {extra}")
    for task_name, spec in sorted(ROLE_SCHEMA.items()):
        all_categories = (
            *spec.manipulated_target,
            *spec.destination_or_reference,
            *spec.tool,
        )
        if not spec.manipulated_target and not spec.tool:
            errors.append(f"{task_name}: requires at least target or tool")
        if any(not item or item != item.strip() for item in all_categories):
            errors.append(f"{task_name}: category phrases must be non-empty/trimmed")
        counts = (
            spec.expected_target,
            spec.expected_destination_or_reference,
            spec.expected_tool,
        )
        if any(isinstance(value, bool) or value < 0 for value in counts):
            errors.append(f"{task_name}: expected counts must be non-negative")
        if bool(spec.manipulated_target) != (spec.expected_target > 0):
            errors.append(f"{task_name}: target categories/count disagree")
        if bool(spec.destination_or_reference) != (
            spec.expected_destination_or_reference > 0
        ):
            errors.append(f"{task_name}: destination categories/count disagree")
        if bool(spec.tool) != (spec.expected_tool > 0):
            errors.append(f"{task_name}: tool categories/count disagree")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "task_count": len(actual),
        "task_names": sorted(actual),
    }


def validate_prompt_index_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pure validation against already-loaded formal prompt-index records."""

    validate_role_schema()
    if not records:
        raise ValueError("prompt index records must be non-empty")
    observed_tasks: set[str] = set()
    checked = 0
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"prompt record {index} must be a mapping")
        task_name = record.get("task_name")
        instruction = record.get("instruction")
        if not isinstance(task_name, str) or not isinstance(instruction, str):
            raise ValueError(f"prompt record {index} lacks task/instruction")
        result = build_grounding_role_prompts(task_name, instruction)
        if not result["roles"]["manipulated_target"] and not result["roles"]["tool"]:
            raise ValueError(f"prompt record {index} has neither target nor tool")
        observed_tasks.add(task_name)
        checked += 1
    missing = sorted(EXPECTED_TASK_NAMES - observed_tasks)
    extra = sorted(observed_tasks - EXPECTED_TASK_NAMES)
    if missing or extra:
        raise ValueError(
            f"prompt-index task coverage mismatch: missing={missing}, extra={extra}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "prompt_record_count": checked,
        "task_count": len(observed_tasks),
    }


def _load_prompt_records(path: Path) -> Sequence[Mapping[str, Any]]:
    payload = json.loads(path.expanduser().resolve(strict=True).read_text(
        encoding="utf-8"
    ))
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("records"), list
    ):
        raise ValueError("prompt index must be an object containing records")
    return payload["records"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name")
    parser.add_argument("--instruction")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--self-check-prompt-index", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    selected = sum(
        (
            bool(args.task_name or args.instruction),
            bool(args.self_check),
            args.self_check_prompt_index is not None,
        )
    )
    if selected != 1:
        raise SystemExit(
            "choose exactly one of task/instruction, --self-check, or "
            "--self-check-prompt-index"
        )
    if args.task_name or args.instruction:
        if not args.task_name or not args.instruction:
            raise SystemExit("--task-name and --instruction are required together")
        result = build_grounding_role_prompts(args.task_name, args.instruction)
    elif args.self_check_prompt_index is not None:
        result = validate_prompt_index_records(
            _load_prompt_records(args.self_check_prompt_index)
        )
    else:
        result = validate_role_schema()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

