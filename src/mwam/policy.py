"""Checkpoint construction and a narrow action interface over the native solver."""
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .config import validate_config
from .observation import validate_observation


class MWAMPolicy:
    def __init__(self, model, *, checkpoint_step=None):
        self.model = model
        self.checkpoint_step = checkpoint_step

    @classmethod
    def from_checkpoint(cls, config, checkpoint, *, pixeldit_pretrained, device="cuda", dtype=torch.bfloat16):
        validate_config(config)
        model_config = OmegaConf.create(OmegaConf.to_container(config.model, resolve=True))
        model_config.pixeldit_pretrained_path = str(Path(pixeldit_pretrained).resolve(strict=True))
        checkpoint = Path(checkpoint).resolve(strict=True)
        model = instantiate(model_config, model_dtype=dtype, device=torch.device(device))
        # Keep native strict format, model-math, key, shape and finite checks.
        loaded = model.load_checkpoint(checkpoint)
        step = loaded["step"]
        del loaded
        model.eval()
        return cls(model, checkpoint_step=step)

    @torch.inference_mode()
    def predict_action(self, observation, *, num_inference_steps=5, seed=20260923):
        validate_observation(observation)
        # First release retains the original joint solver for numerical parity.
        # AC prevents A attending W, but this solver still computes W.
        prediction = self.model.infer_joint(
            observation, num_inference_steps=num_inference_steps, seed=seed,
        )
        return prediction["action_raw"]
