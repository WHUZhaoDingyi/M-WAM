"""CPU test of one training run and its exported weights."""
from pathlib import Path
import torch
from omegaconf import OmegaConf

from mwam.trainer import train

ROOT = Path(__file__).resolve().parents[1]


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return {"x": torch.tensor([index / 12.0, 1.0]), "y": torch.tensor([index / 6.0])}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))

    def forward(self, batch):
        loss = (self.dit(batch["x"]) - batch["y"]).square().mean()
        return loss, {"loss/total": loss.detach()}



def test_training_exports_updated_weights(tmp_path):
    torch.manual_seed(43)
    model = TinyModel()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = OmegaConf.load(ROOT / "configs/train.yaml")
    cfg.output_dir = str(tmp_path)
    cfg.max_steps = 5
    cfg.batch_size = 2
    cfg.num_workers = 0
    cfg.mixed_precision = "no"
    path = train(model, TinyDataset(), cfg)
    assert path == tmp_path / "model.pt"
    weights = torch.load(path, map_location="cpu", weights_only=True)
    TinyModel().load_state_dict(weights, strict=True)
    assert any(not torch.equal(weights[k], initial[k]) for k in weights)
    assert list(tmp_path.iterdir()) == [path]


def test_bad_architecture_rejected_before_model_allocation():
    import pytest
    from mwam.config import load_config, validate_config
    config = load_config(ROOT / "configs/robotwin_3d_ac.yaml")
    config.model.kflow_contract.architecture_version = "typo"
    with pytest.raises(ValueError):
        validate_config(config)
