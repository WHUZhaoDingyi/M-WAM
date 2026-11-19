"""Frozen scientific-method identity for the Formal MoMap V2 model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FASTWAM_MOMAP_V5_TRAINABLE_PARAMETER_COUNT = 1_851_312_256
FASTWAM_MOMAP_V6_TRAINABLE_PARAMETER_COUNT = 1_851_368_176
FASTWAM_MOMAP_V7_ACTION30_ALL_NATIVE_KV_MOT_TRAINABLE_PARAMETER_COUNT = 2_157_692_550
FASTWAM_MOMAP_V7_ACTION26_GLOBAL_NATIVE_KV_MOT_TRAINABLE_PARAMETER_COUNT = 1_994_978_950


FORMAL_METHOD_CONTRACT_V1: dict[str, str] = {
    "schema_version": "fastwam_momap_v2_method_contract_v1",
    "source_conditioning": (
        "privileged_source_rgb3_instance_seg3_canonical_xyz3_v1"
    ),
    "bridge": (
        "post_block_world_hidden_reprojected_kv_one_way_world_to_action_v1"
    ),
    "train_timestep_coupling": "independent",
    "inference_timestep_coupling": (
        "lockstep_equal_step_count_distinct_schedulers_v1"
    ),
    "world_model_timestep_coordinate": (
        "pixeldit_clean_fraction_from_one_minus_wan_noise_fraction_v1"
    ),
    "action_model_timestep_coordinate": (
        "wan_noise_fraction_0_clean_1_noise_v1"
    ),
}


FORMAL_METHOD_CONTRACT_V2: dict[str, str] = {
    "schema_version": "fastwam_momap_v2_method_contract_v2",
    "source_conditioning": (
        "privileged_source_rgb3_instance_seg3_canonical_xyz3_v1"
    ),
    "topology": "world_g26_p4_action30_global_pixel_bridge_v1",
    "bridge": (
        "global_0_25_then_final_global_concat_pit_compact_0_3_"
        "to_action_26_29_v1"
    ),
    "pixel_compact": "pit_post_attention_pre_expand_patch_token_v1",
    "tail_bridge_source": (
        "final_global_then_corresponding_pit_compact_sequence_concat_v1"
    ),
    "tail_bridge_mask": "source_patch_valid_duplicated_for_global_and_pixel_v1",
    "train_timestep_coupling": "independent",
    "inference_timestep_coupling": (
        "lockstep_equal_step_count_distinct_schedulers_v1"
    ),
    "world_model_timestep_coordinate": (
        "pixeldit_clean_fraction_from_one_minus_wan_noise_fraction_v1"
    ),
    "action_model_timestep_coordinate": (
        "wan_noise_fraction_0_clean_1_noise_v1"
    ),
}


FORMAL_METHOD_CONTRACT_V3: dict[str, str] = {
    "schema_version": "fastwam_momap_v2_method_contract_v3",
    "source_conditioning": (
        "privileged_source_rgb3_instance_seg3_canonical_xyz3_v1"
    ),
    "topology": "world_g26_p4_action30_global_then_pixel_kv_bridge_v2",
    "bridge": (
        "global_0_25_then_corresponding_pit_compact_0_3_"
        "to_action_26_29_kv_v2"
    ),
    "pixel_compact": "pit_post_attention_pre_expand_patch_token_v1",
    "tail_bridge_source": "corresponding_pit_compact_sequence_only_v1",
    "tail_bridge_mask": "source_patch_valid_single_v1",
    "train_timestep_coupling": "independent",
    "inference_timestep_coupling": (
        "lockstep_equal_step_count_distinct_schedulers_v1"
    ),
    "world_model_timestep_coordinate": (
        "pixeldit_clean_fraction_from_one_minus_wan_noise_fraction_v1"
    ),
    "action_model_timestep_coordinate": (
        "wan_noise_fraction_0_clean_1_noise_v1"
    ),
}


FORMAL_METHOD_CONTRACT_V4: dict[str, str] = {
    "schema_version": "fastwam_momap_v2_method_contract_v4",
    "source_conditioning": (
        "privileged_source_rgb3_instance_seg3_canonical_xyz3_v1"
    ),
    "topology": "world_g26_p4_action26_global_bridge_v1",
    "bridge": "global_0_25_to_action_0_25_v1",
    "pixel_compact": "pit_post_attention_pre_expand_patch_token_v1",
    "tail_bridge_source": "none_pixel_modules_remain_world_only_v1",
    "tail_bridge_mask": "none_pixel_modules_remain_world_only_v1",
    "pixel_action_bridge": "none_pixel_modules_remain_world_only_v1",
    "train_timestep_coupling": "independent",
    "inference_timestep_coupling": (
        "lockstep_equal_step_count_distinct_schedulers_v1"
    ),
    "world_model_timestep_coordinate": (
        "pixeldit_clean_fraction_from_one_minus_wan_noise_fraction_v1"
    ),
    "action_model_timestep_coordinate": (
        "wan_noise_fraction_0_clean_1_noise_v1"
    ),
}


FORMAL_METHOD_CONTRACT_V5: dict[str, str] = {
    "schema_version": "fastwam_momap_v2_method_contract_v5",
    "source_conditioning": (
        "privileged_source_rgb3_instance_seg3_canonical_xyz3_v1"
    ),
    "topology": "world_g26_p4_action30_global_then_pit_native_kv_mot_v3",
    "bridge": (
        "global_0_25_post_action_self_bridge_then_pit_native_kv_0_3_"
        "concat_action_self_kv_to_action_26_29_mixed_attention_v3"
    ),
    "pit_native_kv": "pit_qk_norm_2d_rope_k_and_native_v_pre_attention_v1",
    "tail_kv_adaptation": (
        "independent_linear_k_and_v_world1152_to_action3072_v1"
    ),
    "tail_mixed_attention": (
        "action_q_queries_concat_pit_native_then_action_self_kv_v1"
    ),
    "tail_bridge_mask": (
        "source_patch_valid_for_pit_then_all_action_keys_valid_v1"
    ),
    "train_timestep_coupling": "independent",
    "inference_timestep_coupling": (
        "lockstep_equal_step_count_distinct_schedulers_v1"
    ),
    "world_model_timestep_coordinate": (
        "pixeldit_clean_fraction_from_one_minus_wan_noise_fraction_v1"
    ),
    "action_model_timestep_coordinate": (
        "wan_noise_fraction_0_clean_1_noise_v1"
    ),
}

FORMAL_METHOD_CONTRACT_V6: dict[str, str] = {
    "schema_version": "fastwam_momap_v2_method_contract_v6",
    "source_conditioning": (
        "privileged_source_rgb3_instance_seg3_canonical_xyz3_v1"
    ),
    "topology": "world_g26_p4_action30_all_world_native_kv_mot_v4",
    "bridge": (
        "global_native_kv_0_25_then_pit_native_kv_0_3_"
        "concat_action_self_kv_to_action_0_29_mixed_attention_v4"
    ),
    "world_native_kv": (
        "global_or_pit_qk_norm_native_rope_k_and_native_v_pre_attention_v1"
    ),
    "kv_adaptation": (
        "independent_linear_k_and_v_world1152_to_action3072_v1"
    ),
    "mixed_attention": (
        "action_q_queries_concat_world_native_then_action_self_kv_once_per_layer_v1"
    ),
    "bridge_mask": (
        "source_patch_valid_for_world_then_all_action_keys_valid_v1"
    ),
    # Hydra recursively merges the inherited Action30 method mapping. Keep the
    # three former tail-only fields explicit and fail-closed for V6.
    "pixel_compact": "not_consumed_by_action_all_native_kv_v1",
    "tail_bridge_source": "not_applicable_all_30_layers_use_native_kv_v1",
    "tail_bridge_mask": "not_applicable_uniform_world_native_kv_mask_v1",
    "train_timestep_coupling": "independent",
    "inference_timestep_coupling": (
        "lockstep_equal_step_count_distinct_schedulers_v1"
    ),
    "world_model_timestep_coordinate": (
        "pixeldit_clean_fraction_from_one_minus_wan_noise_fraction_v1"
    ),
    "action_model_timestep_coordinate": (
        "wan_noise_fraction_0_clean_1_noise_v1"
    ),
}


FORMAL_METHOD_CONTRACT_V7: dict[str, str] = {
    "schema_version": "fastwam_momap_v2_method_contract_v7",
    "source_conditioning": (
        "privileged_source_rgb3_instance_seg3_canonical_xyz3_v1"
    ),
    "topology": "world_g26_p4_action26_global_native_kv_mot_v5",
    "bridge": (
        "global_native_kv_0_25_concat_action_self_kv_to_action_0_25_"
        "mixed_attention_pit_world_only_v5"
    ),
    "world_native_kv": (
        "global_qk_norm_native_rope_k_and_native_v_pre_attention_v1"
    ),
    "kv_adaptation": (
        "independent_linear_k_and_v_world1152_to_action3072_v1"
    ),
    "mixed_attention": (
        "action_q_queries_concat_world_native_then_action_self_kv_once_per_layer_v1"
    ),
    "bridge_mask": (
        "source_patch_valid_for_world_then_all_action_keys_valid_v1"
    ),
    "pixel_compact": "not_consumed_by_action_pit_world_only_v1",
    "tail_bridge_source": "none_pit_modules_remain_world_only_v1",
    "tail_bridge_mask": "none_pit_modules_remain_world_only_v1",
    "pixel_action_bridge": "none_pit_modules_remain_world_only_v1",
    "train_timestep_coupling": "independent",
    "inference_timestep_coupling": (
        "lockstep_equal_step_count_distinct_schedulers_v1"
    ),
    "world_model_timestep_coordinate": (
        "pixeldit_clean_fraction_from_one_minus_wan_noise_fraction_v1"
    ),
    "action_model_timestep_coordinate": (
        "wan_noise_fraction_0_clean_1_noise_v1"
    ),
}


FORMAL_METHOD_CONTRACT_V8: dict[str, str] = {
    "schema_version": "fastwam_momap_v2_method_contract_v8",
    "source_conditioning": (
        "privileged_source_rgb3_instance_seg3_canonical_xyz3_v1"
    ),
    "topology": "world_g26_p4_action30_linear14_aligned_native_kv_v8",
    "bridge": (
        "global_native_kv_0_25_then_pit_native_kv_0_3_concat_"
        "action_self_kv_equal_width_v8"
    ),
    "world_native_kv": (
        "global_or_pit_qk_norm_native_2d_rope_k_and_native_v_pre_attention_v1"
    ),
    "kv_adaptation": "none_equal_hidden1152_heads16_head_dim72_v1",
    "mixed_attention": (
        "action_no_rope_q_queries_concat_world_native_then_action_self_kv_"
        "once_per_layer_v1"
    ),
    "bridge_mask": (
        "source_patch_valid_for_world_then_all_14_action_keys_valid_v1"
    ),
    "world_key_logit_correction": (
        "per_sample_log_14_over_valid_world_key_count_v1"
    ),
    "action_tokenization": (
        "transpose_b32x14_to_b14x32_then_linear32_to1152_v1"
    ),
    "action_position": (
        "fixed_sincos_actuator_index_0_13_no_temporal_rope_v1"
    ),
    "pixel_compact": "not_consumed_by_action_all_30_layers_use_native_kv_v1",
    "tail_bridge_source": "not_applicable_all_30_layers_use_native_kv_v1",
    "tail_bridge_mask": "not_applicable_uniform_world_native_kv_mask_v1",
    "train_timestep_coupling": "independent",
    "inference_timestep_coupling": (
        "lockstep_equal_step_count_distinct_schedulers_v1"
    ),
    "world_model_timestep_coordinate": (
        "pixeldit_clean_fraction_from_one_minus_wan_noise_fraction_v1"
    ),
    "action_model_timestep_coordinate": (
        "wan_noise_fraction_0_clean_1_noise_v1"
    ),
}


# V9 preserves V8 parameter topology and tensor schemas. Its only model-math
# change is the explicitly controlled 500-optimizer-step fade from the legacy
# count correction to FastWAM-style direct World/Action K/V concatenation.
FORMAL_METHOD_CONTRACT_V9: dict[str, str] = {
    **FORMAL_METHOD_CONTRACT_V8,
    "schema_version": "fastwam_momap_v2_method_contract_v9",
    "bridge": (
        "global_native_kv_0_25_then_pit_native_kv_0_3_concat_"
        "action_self_kv_equal_width_v9"
    ),
    "world_key_logit_correction": (
        "per_sample_log_14_over_valid_world_key_count_cosine_anneal_to_none_"
        "500_optimizer_steps_v1"
    ),
}


_FORMAL_METHOD_CONTRACT_K4_HYBRID_BASE: dict[str, str] = {
    "source_conditioning": (
        "source_rgb3_pred_instance_seg3_noisy_canonical_xyz3_head_only_seg_v1"
    ),
    "world_topology": (
        "shared_weight_c_only_and_w_reads_cw_g26_p4_local16_k4_v1"
    ),
    "world_state": (
        "all_dxyz12_head_wrist_carrier12_v1"
    ),
    "world_attention": "c_queries_c_only_w_queries_concat_cw_v1",
    "world_timestep": "c_clean_fraction_one_w_one_minus_sigma_v1",
    "world_position": "shared_mosaic_spatial_coordinates_no_cw_type_embed_v1",
    "world_loss": (
        "normalized_velocity_half_head_xyz_half_mean_nonempty_wrist_xyz_v1"
    ),
    "world_normalization": (
        "source_xyz_affine_head_dxyz_scale_wrist_dxyz_meter_scale_v1"
    ),
    "action_tokenization": (
        "transpose_b32x14_to_b14x32_then_linear32_to1152_v1"
    ),
    "action_position": "fixed_sincos_actuator_index_0_13_no_temporal_rope_v1",
    "kv_adaptation": "none_equal_hidden1152_heads16_head_dim72_v1",
    "mixed_attention": (
        "action_no_rope_q_queries_concat_selected_world_native_then_"
        "action_self_kv_direct_no_count_bias_v1"
    ),
    "train_timestep_coupling": "independent",
    "inference_timestep_coupling": (
        "lockstep_equal_step_count_distinct_schedulers_v1"
    ),
    "world_model_timestep_coordinate": (
        "pixeldit_clean_fraction_from_one_minus_wan_noise_fraction_v1"
    ),
    "action_model_timestep_coordinate": (
        "wan_noise_fraction_0_clean_1_noise_v1"
    ),
}


FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW: dict[str, str] = {
    "schema_version": "fastwam_momap_k4_hybrid_method_contract_acw_v1",
    **_FORMAL_METHOD_CONTRACT_K4_HYBRID_BASE,
    "topology": "hybrid_cw_g26_p4_action30_joint14_action_condition_world_v1",
    "bridge": "all_30_layers_concat_c_then_w_then_action_self_kv_v1",
    "action_world_visibility": "c_and_w_960_world_keys_v1",
}


FORMAL_METHOD_CONTRACT_K4_HYBRID_AC: dict[str, str] = {
    "schema_version": "fastwam_momap_k4_hybrid_method_contract_ac_v1",
    **_FORMAL_METHOD_CONTRACT_K4_HYBRID_BASE,
    "topology": "hybrid_cw_g26_p4_action30_joint14_action_condition_v1",
    "bridge": "all_30_layers_concat_c_then_action_self_kv_v1",
    "action_world_visibility": "c_only_480_world_keys_v1",
}


# v2 changes only Flow numerical normalization: all K4 offsets share one
# Head-XYZ and one pooled Wrist-XYZ mean/std set.  The architecture, data source,
# attention visibility and loss reduction remain otherwise identical.
FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2: dict[str, str] = {
    **FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW,
    "schema_version": "fastwam_momap_k4_hybrid_method_contract_acw_v2",
    "world_normalization": (
        "source_xyz_affine_head_dxyz_pooled_zscore_"
        "wrist_dxyz_pooled_meter_zscore_v1"
    ),
}


FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2: dict[str, str] = {
    **FORMAL_METHOD_CONTRACT_K4_HYBRID_AC,
    "schema_version": "fastwam_momap_k4_allxyz_method_contract_ac_v1",
    "world_normalization": (
        "source_xyz_affine_head_dxyz_pooled_zscore_"
        "wrist_dxyz_pooled_meter_zscore_v1"
    ),
}


# This branch keeps the Hybrid C/W World unchanged but restores the original
# FastWAM Action time-token axis: 32 tokens, one 14-D control vector per step.
# It is intentionally a new immutable contract rather than a mutation of AC.
FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2: dict[str, str] = {
    **FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2,
    "schema_version": "fastwam_momap_k4_hybrid_method_contract_ac_temporal32_v2",
    "action_tokenization": "direct_b32x14_linear14_to1152_time32_v1",
    "action_position": "temporal_1d_rope_index_0_31_v1",
    "mixed_attention": (
        "action_temporal_rope_q_queries_concat_selected_world_native_then_"
        "action_self_kv_direct_no_count_bias_v1"
    ),
    "topology": "hybrid_cw_g26_p4_action30_time32_action_condition_v1",
    "bridge": "all_30_layers_concat_c_then_action_time32_self_kv_v1",
    "action_world_visibility": "c_only_480_world_keys_v1",
}


def _require_exact_contract(
    raw: dict[str, Any],
    expected: dict[str, str],
    *,
    label: str,
) -> dict[str, str]:
    if raw != expected:
        missing = sorted(set(expected) - set(raw))
        unexpected = sorted(set(raw) - set(expected))
        mismatched = {
            key: {
                "expected": expected[key],
                "got": raw.get(key),
            }
            for key in sorted(set(raw) & set(expected))
            if raw[key] != expected[key]
        }
        raise ValueError(
            f"Formal method_contract must exactly match {label}: "
            f"missing={missing}, unexpected={unexpected}, "
            f"mismatched={mismatched}."
        )
    return dict(expected)


def resolve_formal_method_contract(value: Any) -> dict[str, str]:
    """Return a detached exact supported contract or reject scientific drift."""

    if not isinstance(value, Mapping):
        raise TypeError(
            "`method_contract` must be a mapping, "
            f"got {type(value).__name__}."
        )
    raw = dict(value)
    schema_version = raw.get("schema_version")
    supported_schema_versions = [
        FORMAL_METHOD_CONTRACT_V1["schema_version"],
        FORMAL_METHOD_CONTRACT_V2["schema_version"],
        FORMAL_METHOD_CONTRACT_V3["schema_version"],
        FORMAL_METHOD_CONTRACT_V4["schema_version"],
        FORMAL_METHOD_CONTRACT_V5["schema_version"],
        FORMAL_METHOD_CONTRACT_V6["schema_version"],
        FORMAL_METHOD_CONTRACT_V7["schema_version"],
        FORMAL_METHOD_CONTRACT_V8["schema_version"],
        FORMAL_METHOD_CONTRACT_V9["schema_version"],
        FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW["schema_version"],
        FORMAL_METHOD_CONTRACT_K4_HYBRID_AC["schema_version"],
        FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2["schema_version"],
        FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2["schema_version"],
        FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2["schema_version"],
    ]
    if schema_version == FORMAL_METHOD_CONTRACT_V1["schema_version"]:
        return _require_exact_contract(raw, FORMAL_METHOD_CONTRACT_V1, label="v1")
    if schema_version == FORMAL_METHOD_CONTRACT_V2["schema_version"]:
        return _require_exact_contract(raw, FORMAL_METHOD_CONTRACT_V2, label="v2")
    if schema_version == FORMAL_METHOD_CONTRACT_V3["schema_version"]:
        return _require_exact_contract(raw, FORMAL_METHOD_CONTRACT_V3, label="v3")
    if schema_version == FORMAL_METHOD_CONTRACT_V4["schema_version"]:
        return _require_exact_contract(raw, FORMAL_METHOD_CONTRACT_V4, label="v4")
    if schema_version == FORMAL_METHOD_CONTRACT_V5["schema_version"]:
        return _require_exact_contract(raw, FORMAL_METHOD_CONTRACT_V5, label="v5")
    if schema_version == FORMAL_METHOD_CONTRACT_V6["schema_version"]:
        return _require_exact_contract(raw, FORMAL_METHOD_CONTRACT_V6, label="v6")
    if schema_version == FORMAL_METHOD_CONTRACT_V7["schema_version"]:
        return _require_exact_contract(raw, FORMAL_METHOD_CONTRACT_V7, label="v7")
    if schema_version == FORMAL_METHOD_CONTRACT_V8["schema_version"]:
        return _require_exact_contract(raw, FORMAL_METHOD_CONTRACT_V8, label="v8")
    if schema_version == FORMAL_METHOD_CONTRACT_V9["schema_version"]:
        return _require_exact_contract(raw, FORMAL_METHOD_CONTRACT_V9, label="v9")
    if schema_version == FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW["schema_version"]:
        return _require_exact_contract(
            raw,
            FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW,
            label="k4_hybrid_acw_v1",
        )
    if schema_version == FORMAL_METHOD_CONTRACT_K4_HYBRID_AC["schema_version"]:
        return _require_exact_contract(
            raw,
            FORMAL_METHOD_CONTRACT_K4_HYBRID_AC,
            label="k4_hybrid_ac_v1",
        )
    if schema_version == FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2["schema_version"]:
        return _require_exact_contract(
            raw,
            FORMAL_METHOD_CONTRACT_K4_HYBRID_ACW_V2,
            label="k4_hybrid_acw_v2",
        )
    if schema_version == FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2["schema_version"]:
        return _require_exact_contract(
            raw,
            FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2,
            label="k4_hybrid_ac_v2",
        )
    if (
        schema_version
        == FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2["schema_version"]
    ):
        return _require_exact_contract(
            raw,
            FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_TEMPORAL32_V2,
            label="k4_hybrid_ac_temporal32_v2",
        )
    raise ValueError(
        "Unsupported formal method_contract schema_version: "
        f"{schema_version!r}; expected one of "
        f"{supported_schema_versions!r}."
    )


