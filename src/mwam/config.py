"""One explicit configuration for the all-view 3D / AC main method."""
from pathlib import Path

from omegaconf import OmegaConf


METHOD = {
    "name": "M-WAM", "motion_representation": "all_views_3d",
    "attention": "AC", "action_tokens": "trajectory",
    "noise_sampling": "high_noise_405505",
}
HIGH_NOISE = (
    "mwam.models.wan22.schedulers.scheduler_continuous."
    "WanHighNoise405505UniformWeightFlowMatchScheduler"
)


def load_config(path: str | Path):
    config = OmegaConf.load(path)
    validate_config(config)
    return config


def validate_config(config):
    """Reject a mislabeled ablation before allocating the network."""
    if OmegaConf.to_container(config.method, resolve=True) != METHOD:
        raise ValueError("Expected main method: all-view 3D, AC, trajectory tokens, high noise")
    model = config.model
    from .models.pixeldit.kflow import KFlowContract
    KFlowContract(**OmegaConf.to_container(model.kflow_contract, resolve=True))
    expected = {
        "_target_": "mwam.models.pixeldit.factory.create_hybrid_k4_fastwam_pixeldit",
        "kv_source": "action_condition",
        "action_tokenization": "joint14",
        "timestep_coupling": "independent",
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f"model.{key} must be {value!r}")
    if model.kflow_contract.target_representation != "all_views_dxyz_matched_uv_valid_v1":
        raise ValueError("RoboTwin training uses XYZ displacements in all views")
    if model.kflow_contract.unit != "all_views_canonical_meter":
        raise ValueError("All motion components must be in the canonical metric frame")
    for name in ("train_world_scheduler", "train_action_scheduler"):
        if model[name].get("_target_") != HIGH_NOISE:
            raise ValueError(f"model.{name} must use high-noise 40/55/5 sampling")
    # Reuse the backend's exact contract rather than inventing a second schema.
    from mwam.models.pixeldit.method_contract import (
        FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2, resolve_formal_method_contract,
    )
    contract = resolve_formal_method_contract(OmegaConf.to_container(model.method_contract))
    if contract != FORMAL_METHOD_CONTRACT_K4_HYBRID_AC_V2:
        raise ValueError("Method contract does not match the all-view 3D AC backend")
    return dict(METHOD)
