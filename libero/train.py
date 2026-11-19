"""Train the revised LIBERO world/action model with torchrun and PyTorch FSDP2."""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard, register_fsdp_forward_method
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


class ClipBatches:
    """Shuffle global windows once per epoch. Draw position determines the palette."""

    def __init__(self, dataset, seed, batch, rank, world):
        self.dataset, self.seed, self.batch = dataset, seed, batch
        self.rank, self.global_batch = rank, batch * world
        self.steps_per_epoch = math.ceil(len(dataset) / self.global_batch)
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.dataset), generator=generator).tolist()
        for step in range(self.steps_per_epoch):
            start = step * self.global_batch + self.rank * self.batch
            yield [
                self.dataset.training_sample_key(order[i % len(order)], seed=self.seed,
                                                 epoch=self.epoch, draw=i)
                for i in range(start, start + self.batch)
            ]


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def shard_model(model, world):
    mesh = init_device_mesh("cuda", (world,))
    precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
                                     cast_forward_inputs=False)
    options = dict(mesh=mesh, mp_policy=precision, reshard_after_forward=False)
    world_net, action_net = model.dit.world_expert, model.dit.action_expert
    for block in world_net.patch_blocks:
        fully_shard(block, **options)
        register_fsdp_forward_method(block, "world")
    for block in world_net.pixel_blocks:
        fully_shard(block, **options)
    for block in action_net.blocks:
        fully_shard(block, **options)
        register_fsdp_forward_method(block, "action")
    fully_shard(model, **options)
    world_blocks = [*world_net.patch_blocks, *world_net.pixel_blocks]
    ordered = [block for pair in zip(world_blocks, action_net.blocks, strict=True) for block in pair]
    model.set_modules_to_forward_prefetch(ordered[:2])
    for index, block in enumerate(ordered):
        block.set_modules_to_forward_prefetch(ordered[index + 1:index + 3])


def make_scheduler(optimizer, cfg):
    if not 0 <= cfg.warmup_steps < cfg.max_steps:
        raise ValueError("warmup_steps must be in [0, max_steps)")
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_steps - cfg.warmup_steps, eta_min=cfg.minimum_lr)
    if not cfg.warmup_steps:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=cfg.warmup_start_lr / cfg.learning_rate, total_iters=cfg.warmup_steps)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_steps])


def export_weights(output, model):
    # All ranks participate in gathering the FSDP weights onto rank zero.
    weights = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if dist.get_rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
        path = output / "model.pt"
        temporary = output / "model.pt.tmp"
        torch.save(weights, temporary)
        temporary.replace(path)
        print(f"Saved {path}", flush=True)


def train(args, cfg, device):
    from mwam.datasets.libero_momap.standalone import StandaloneLiberoDataset
    from mwam.models.pixeldit.native.revised import create_revised_libero

    rank, world = dist.get_rank(), dist.get_world_size()
    data = args.data.resolve()
    paths = {"manifest_path": data / "episodes.json",
             "normalization_artifact_path": data / "normalization_full_features.json",
             "segmentation_manifest_path": data / "predicted_segmentation_libero.json"}
    dataset = StandaloneLiberoDataset(
        **{k: str(v) for k, v in paths.items()}, split="train", max_open_episodes=4,
        segmentation_mode="predicted", segmentation_palette="clip_draw")
    if not len(dataset):
        raise ValueError("Training dataset is empty")
    # The recorded training code seeds construction with base_seed + rank.
    seed_rng(cfg.seed + rank)
    model = create_revised_libero(
        normalization_artifact_path=str(paths["normalization_artifact_path"]),
        pixeldit_pretrained_path=args.pretrained,
        use_gradient_checkpointing=cfg.activation_checkpointing,
        device="cpu", uniform_noise=False)
    model.set_training_stage("joint")
    model.train().requires_grad_(True)
    shard_model(model, world)
    seed_rng(cfg.seed + 1009 * (rank + 1) + rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=tuple(cfg.betas),
                                  weight_decay=cfg.weight_decay, fused=True)
    scheduler = make_scheduler(optimizer, cfg)
    step = 0
    sampler = ClipBatches(dataset, cfg.seed, cfg.batch_per_rank, rank, world)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=cfg.workers,
                        persistent_workers=cfg.workers > 0, pin_memory=True,
                        generator=torch.Generator().manual_seed(cfg.seed + rank), collate_fn=dataset.collate_fn)
    while step < cfg.max_steps:
        for sample in loader:
            sample = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                      for k, v in sample.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = model(sample)
            loss.backward()
            # All trainable tensors are sharded across a single mesh dimension.
            grads = [p.grad.to_local() for p in model.parameters() if p.grad is not None]
            squared_norm = torch.stack([g.float().square().sum() for g in grads]).sum()
            dist.all_reduce(squared_norm)
            norm = squared_norm.sqrt()
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite joint gradient")
            torch._foreach_mul_(grads, (cfg.max_grad_norm / (norm + 1e-6)).clamp_max(1.0))
            optimizer.step()
            scheduler.step()
            step += 1
            if step == 1 or step % cfg.log_every == 0:
                keys = ("loss/total", "loss/action_weighted", "loss/momap_weighted")
                values = torch.tensor([metrics[k] for k in keys], device=device)
                dist.all_reduce(values)
                if rank == 0:
                    print(json.dumps({"step": step, **dict(zip(keys, (values / world).tolist()))}), flush=True)
            if step == cfg.max_steps:
                break
        sampler.epoch += 1
    export_weights(args.output, model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared LIBERO directory")
    parser.add_argument("--pretrained", type=Path, required=True, help="Official PixelDiT T2I .pth")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/libero_joint.yaml")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the final model.pt weights")
    args = parser.parse_args()
    if not args.pretrained.is_file():
        parser.error("--pretrained must point to the official PixelDiT T2I weights")
    cfg = SimpleNamespace(**yaml.safe_load(args.config.read_text()))
    if min(cfg.batch_per_rank, cfg.max_steps, cfg.log_every) < 1 or cfg.workers < 0:
        parser.error("Invalid batch, worker count, training budget, or logging interval")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    dist.init_process_group("nccl", device_id=device)
    try:
        train(args, cfg, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
