"""Real native trainer integration with a tiny CPU fixture (not M-WAM SR)."""
from pathlib import Path
import torch
from omegaconf import OmegaConf

from mwam.trainer import Wan22Trainer
from mwam.utils.pytorch_utils import set_global_seed

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

    def save_checkpoint(self, path, optimizer=None, step=None):
        torch.save({"model": self.state_dict(), "step": step}, path)


def run_probe(path, *, stop=None, resume=None):
    set_global_seed(43)
    cfg = OmegaConf.load(ROOT / "configs/train.yaml")
    cfg.output_dir = str(path)
    cfg.batch_size = 2
    cfg.gradient_accumulation_steps = 2
    cfg.num_workers = 0
    cfg.persistent_workers = False
    cfg.prefetch_factor = None
    cfg.mixed_precision = "no"
    cfg.max_steps = 5
    cfg.log_every = 1
    cfg.save_every = 0
    cfg.stop_after_step = stop
    cfg.resume = str(resume) if resume else None
    cfg.optimizer_parameter_groups = None
    trainer = Wan22Trainer(TinyModel(), TinyDataset(), cfg=cfg)
    trainer.train()
    model = trainer.accelerator.unwrap_model(trainer.model)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    step = trainer.global_step
    trainer.accelerator.free_memory()
    return state, step


def test_full_trainer_save_resume_matches_uninterrupted(tmp_path, monkeypatch):
    # This includes actual AdamW, accumulation, epoch crossing, scheduler,
    # Accelerator state, sampler cursor and dropout RNG restoration.
    torch.set_num_threads(2)
    # CPU test of Accelerator's ordinary backend. Do not import an optional
    # DeepSpeed install (its CPU probe can require /proc outside this sandbox).
    import accelerate.utils.other
    monkeypatch.setattr(accelerate.utils.other, "is_deepspeed_available", lambda: False)
    reference, step = run_probe(tmp_path / "reference")
    assert step == 5
    stopped, step = run_probe(tmp_path / "resumed", stop=2)
    assert step == 2
    state_dir = tmp_path / "resumed/checkpoints/state/step_000002"
    assert (state_dir / "trainer_state.json").is_file()
    resumed, step = run_probe(tmp_path / "resumed", resume=state_dir)
    assert step == 5
    for key in reference:
        torch.testing.assert_close(resumed[key], reference[key], rtol=0, atol=0)
    assert any(not torch.equal(stopped[k], resumed[k]) for k in resumed)


def test_bad_architecture_rejected_before_model_allocation():
    import pytest
    from mwam.config import load_config, validate_config
    config = load_config(ROOT / "configs/robotwin_3d_ac.yaml")
    config.model.kflow_contract.architecture_version = "typo"
    with pytest.raises(ValueError):
        validate_config(config)
