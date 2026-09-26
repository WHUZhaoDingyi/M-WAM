"""Source-bound supervision-only transition, without resetting training state."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def contract_sha256(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def allows_gt_background_migration(saved, current, migration, *, source_state_dir,
                                   source_global_step):
    """Allow exactly 200px -> GT identity mask; every other contract field is equal."""
    if not isinstance(migration, Mapping) or set(migration) != {
        "source_state_dir", "source_global_step", "source_resume_contract_sha256",
        "authorization_reference",
    }:
        return False
    if (source_state_dir is None or source_global_step is None
        or type(migration["source_global_step"]) is not int
        or migration["source_global_step"] != source_global_step
        or not isinstance(migration["authorization_reference"], str)
        or not migration["authorization_reference"].strip()):
        return False
    try:
        bound = Path(migration["source_state_dir"])
        if not bound.is_absolute() or bound.resolve() != Path(source_state_dir).resolve():
            return False
        if contract_sha256(saved) != migration["source_resume_contract_sha256"]:
            return False
        expected = copy.deepcopy(saved)
        target = expected["data_math"]["target_builder"]
        if (target.get("target_representation") != "all_views_dxyz_matched_uv_valid_v1"
            or target.pop("wrist_flow_max_magnitude_native_px", None) != 200.0
            or target.pop("momap_loss_valid_semantics", None)
                != "component_valid_and_native_wrist_duv_l2_gate_v1"
            or "wrist_flow_gt_background_filter" in target
            or "gt_background_entity_names" in target):
            return False
        target.update(wrist_flow_gt_background_filter=True,
                      momap_loss_valid_semantics="component_valid_and_gt_entity_background_exclusion_v1",
                      gt_background_entity_names=["background", "ground", "table", "wall"])
        return expected == current
    except (KeyError, TypeError, ValueError, OSError):
        return False
