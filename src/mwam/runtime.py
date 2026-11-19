"""Build the RoboTwin dataset and M-WAM policy for joint training."""
import os

import torch
from hydra.utils import instantiate

from .trainer import train
from .utils.pytorch_utils import set_global_seed


def run_training(cfg):
    set_global_seed(cfg.seed)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    dtype = {"no": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.mixed_precision]
    model = instantiate(cfg.model, model_dtype=dtype, device=str(device))
    dataset = instantiate(cfg.data.train)
    return train(model, dataset, cfg)
