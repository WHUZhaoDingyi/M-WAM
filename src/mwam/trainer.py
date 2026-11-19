"""Joint action and 3D motion training on RoboTwin demonstrations."""
import math
import os
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data import DataLoader

from .utils.pytorch_utils import set_global_seed


def train(model, dataset, cfg):
    if len(dataset) == 0:
        raise ValueError("The training dataset has no action windows.")
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=1,
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True, data_seed=cfg.seed, non_blocking=True,
        ),
    )
    plugin = accelerator.state.deepspeed_plugin
    if plugin is not None:
        plugin.deepspeed_config["gradient_clipping"] = float(cfg.max_grad_norm)

    # The trainable policy consists of the World and Action experts and the state encoder.
    model.eval().requires_grad_(False)
    model.dit.train().requires_grad_(True)
    proprio = getattr(model, "proprio_encoder", None)
    if proprio is not None:
        proprio.train().requires_grad_(True)

    worker_init = set_global_seed(cfg.seed, get_worker_init_fn=True)
    use_fork = cfg.num_workers > 0 and getattr(dataset, "requires_fork_cow_workers", False)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=worker_init, generator=torch.Generator().manual_seed(cfg.seed),
        pin_memory=accelerator.device.type == "cuda",
        multiprocessing_context="fork" if use_fork else None,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    warmup_steps = int(cfg.max_steps * cfg.warmup_ratio)

    def lr_multiplier(step):
        if step < warmup_steps:
            alpha = step / warmup_steps
            lr = cfg.warmup_start_learning_rate + alpha * (cfg.learning_rate - cfg.warmup_start_learning_rate)
        else:
            progress = (step - warmup_steps) / (cfg.max_steps - warmup_steps)
            cosine = (1 + math.cos(math.pi * progress)) / 2
            lr = cfg.cosine_min_learning_rate + cosine * (cfg.learning_rate - cfg.cosine_min_learning_rate)
        return lr / cfg.learning_rate

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    step = epoch = 0
    while step < cfg.max_steps:
        loader.set_epoch(epoch)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                loss, _ = model(batch)
            accelerator.backward(loss)
            if plugin is None:
                accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            if accelerator.optimizer_step_was_skipped:
                continue
            scheduler.step()
            step += 1
            if step == 1 or step % cfg.log_every == 0:
                average = accelerator.reduce(loss.detach().float(), reduction="mean")
                accelerator.print(f"step {step}  loss {average.item():.5f}")
            if step == cfg.max_steps:
                break
        epoch += 1

    weights = accelerator.get_state_dict(model)
    path = Path(cfg.output_dir) / "model.pt"
    if accelerator.is_main_process:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".model.{os.getpid()}.tmp")
        torch.save({k: v.detach().cpu() for k, v in weights.items()}, temporary)
        temporary.replace(path)
    accelerator.wait_for_everyone()
    return path
