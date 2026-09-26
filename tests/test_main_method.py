from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mwam.config import HIGH_NOISE, load_config, validate_config
from mwam.observation import TENSOR_KEYS, load_observation, synthetic_observation, validate_observation
from mwam.policy import MWAMPolicy

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def test_main_config_and_three_d_wrist_normalization():
    from mwam.models.pixeldit.factory import _construct_dataclass
    from mwam.models.pixeldit.hybrid_normalization import HybridHeadWristNormalizationContract
    config = load_config(ROOT / "configs/robotwin_3d_ac.yaml")
    norm = _construct_dataclass(config.model.normalization_contract, HybridHeadWristNormalizationContract, name="geometry")
    flow = torch.randn(1, 12, 384, 320)
    normalized = norm.normalize_offset(flow)
    torch.testing.assert_close(norm.denormalize_offset(normalized), flow, rtol=1e-5, atol=1e-6)
    # Wrist Z is a real trained metric component, not zero padding for 2D UV.
    z = flow.reshape(1, 4, 3, 384, 320)[:, :, 2, 256:]
    expected = (z - norm.wrist_flow_mean_xyz_meters[2]) / norm.wrist_flow_std_xyz_meters[2]
    torch.testing.assert_close(normalized.reshape(1, 4, 3, 384, 320)[:, :, 2, 256:], expected)


@pytest.mark.parametrize("key,value", [
    ("model.kv_source", "action_condition_world"),
    ("model.action_tokenization", "temporal32_rope"),
    ("model.kflow_contract.target_representation", "head_dxyz_wrist_duv"),
    ("model.train_world_scheduler._target_", HIGH_NOISE.replace("WanHighNoise405505UniformWeight", "WanContinuous")),
    ("model.train_action_scheduler._target_", HIGH_NOISE.replace("WanHighNoise405505UniformWeight", "WanContinuous")),
])
def test_main_entry_rejects_ablations(key, value):
    config = load_config(ROOT / "configs/robotwin_3d_ac.yaml")
    OmegaConf.update(config, key, value)
    with pytest.raises(ValueError):
        validate_config(config)


def test_high_noise_distribution_and_uniform_loss_weight():
    from mwam.models.wan22.schedulers.scheduler_continuous import WanHighNoise405505UniformWeightFlowMatchScheduler
    scheduler = WanHighNoise405505UniformWeightFlowMatchScheduler()
    torch.manual_seed(7)
    sigma = scheduler.sample_training_t(100000, torch.device("cpu"), torch.float32) / 1000
    observed = torch.stack(((sigma < .95).float().mean(), ((sigma >= .95) & (sigma < 1)).float().mean(), (sigma == 1).float().mean()))
    torch.testing.assert_close(observed, torch.tensor([.4, .55, .05]), atol=.008, rtol=0)
    torch.testing.assert_close(scheduler.training_weight(sigma * 1000), torch.ones_like(sigma))


def test_npz_roundtrip_and_future_rejection(tmp_path):
    observation = synthetic_observation(7)
    path = tmp_path / "observation.npz"
    np.savez(path, **{key: observation[key].numpy() for key in TENSOR_KEYS})
    restored = load_observation(path)
    for key in TENSOR_KEYS:
        torch.testing.assert_close(observation[key], restored[key], rtol=0, atol=0)
    with pytest.raises(ValueError):
        validate_observation({**restored, "target_flow": torch.zeros(1)})
    np.savez(path, **{key: observation[key].numpy() for key in TENSOR_KEYS}, action=np.zeros(1))
    with pytest.raises(ValueError):
        load_observation(path)


def test_policy_passes_source_and_seed_without_modifying_actions():
    # Interface test only; does not stand in for full-model forward parity.
    class NativeSolver:
        def infer_joint(self, observation, **kwargs):
            self.inputs = observation
            self.options = kwargs
            self.action = torch.randn(1, 32, 14)
            return {"action_raw": self.action}
    solver = NativeSolver()
    observation = synthetic_observation(5)
    action = MWAMPolicy(solver).predict_action(observation, num_inference_steps=5, seed=17)
    assert solver.inputs is observation
    assert solver.options == {"num_inference_steps": 5, "seed": 17}
    assert action is solver.action


def test_ac_selects_condition_without_world_kv():
    from mwam.models.pixeldit.hybrid_cw_action_mot import HybridCWActionMoT
    module = HybridCWActionMoT.__new__(HybridCWActionMoT)
    torch.nn.Module.__init__(module)
    module._kv_source = "action_condition"
    c = {name: torch.randn(1, 4, 2, 8) for name in ("k", "v")}
    w = {name: torch.randn(1, 4, 2, 8) * 100 for name in ("k", "v")}
    selected = module._select_native_kv({"c": c, "w": w}, freeze_world=False, source_name="test")
    for name in ("k", "v"):
        assert selected[name] is c[name]
