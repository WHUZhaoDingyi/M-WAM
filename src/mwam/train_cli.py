"""Data preparation and full native training command line."""
import argparse
import json
import os
from pathlib import Path


COMMANDS = ("prepare-data", "encode-text", "index-segmentation", "check-data", "train", "check-training-config")


def assemble_training_config(args):
    from omegaconf import OmegaConf
    from .config import load_config, validate_config
    model = load_config(args.model_config)
    cfg = OmegaConf.merge(OmegaConf.load(args.config),
                          {"model": model.model, "method": model.method},
                          OmegaConf.load(args.data_config), OmegaConf.from_dotlist(args.set))
    cfg.output_dir = str(args.output_dir.expanduser().resolve())
    cfg.model.pixeldit_pretrained_path = str(args.pixeldit_pretrained.expanduser().resolve())
    if args.resume is not None:
        cfg.resume = str(args.resume.expanduser().resolve())
    validate_config(cfg)
    if cfg.eval_every != 0:
        raise ValueError("MoMap uses standalone validation; keep eval_every=0")
    if cfg.training_stage != "joint" or not cfg.data.train.strict_pred_head_only:
        raise ValueError("Main recipe requires joint training and predicted head segmentation")
    if cfg.max_steps is None or cfg.max_steps < 1 or cfg.batch_size < 1 or cfg.gradient_accumulation_steps < 1:
        raise ValueError("max_steps, batch_size and gradient_accumulation_steps must be positive")
    if cfg.resume is not None and not Path(cfg.resume).is_dir():
        raise ValueError("--resume requires a full checkpoint state directory, not a weights .pt file")
    if not Path(cfg.model.pixeldit_pretrained_path).is_file():
        raise FileNotFoundError(cfg.model.pixeldit_pretrained_path)
    # Reject data/model mismatches before constructing the 1.61B network.
    data, contract = cfg.data.train, cfg.model.kflow_contract
    for data_key, model_key in (("offsets", "future_offsets"), ("future_horizons_seconds", "future_horizons_seconds"),
                                ("future_offset_basis", "future_offset_basis"), ("camera_names", "camera_order"),
                                ("target_representation", "target_representation"), ("coordinate_frame", "coordinate_frame")):
        if data[data_key] != contract[model_key]:
            raise ValueError(f"Data/model mismatch: {data_key}")
    if data.action_horizon != 32 or data.action_rate_hz != 15.0 or data.context_len != 128 or data.context_hidden_size != 4096:
        raise ValueError("Expected action [32,14] at 15Hz and text [128,4096]")
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-data")
    for flag in ("manifest", "template", "text-cache", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--seg-manifest", type=Path)
    p = sub.add_parser("index-segmentation")
    for flag in ("manifest", "cache-dir", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p = sub.add_parser("encode-text")
    for flag in ("prompts", "text-cache", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--text-weights", type=Path)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("check-data")
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batches", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    for command in ("train", "check-training-config"):
        p = sub.add_parser(command)
        for flag in ("config", "model-config", "data-config", "pixeldit-pretrained", "output-dir"):
            p.add_argument("--" + flag, type=Path, required=True)
        p.add_argument("--resume", type=Path)
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if command == "check-training-config":
            p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from .data_tools import prepare_data, encode_text, index_segmentation, check_data, write_json
    if hasattr(args, "threads"):
        import torch
        if args.threads < 1:
            parser.error("threads must be positive")
        torch.set_num_threads(args.threads)
    if args.command == "prepare-data":
        result = prepare_data(args.manifest, args.template, args.text_cache, args.seg_manifest, args.output)
    elif args.command == "index-segmentation":
        result = index_segmentation(args.manifest, args.cache_dir, args.output)
    elif args.command == "encode-text":
        result = encode_text(args.prompts, args.text_cache, args.text_weights, args.tokenizer, args.device, args.batch_size)
        write_json(args.output, result)
    elif args.command == "check-data":
        result = check_data(args.data_config, args.output, args.batches, args.batch_size, args.workers)
    else:
        from omegaconf import OmegaConf
        cfg = assemble_training_config(args)
        if args.command == "check-training-config":
            result = dict(status="TRAINING_CONFIG_VALID", training_executed=False, config=OmegaConf.to_container(cfg, resolve=True))
            write_json(args.output, result)
        else:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Full M-WAM training requires CUDA; use check-data for CPU data preparation")
            output = Path(cfg.output_dir)
            if cfg.resume is None and (output / "checkpoints").exists():
                raise FileExistsError("Existing training run: use --resume or a new output directory")
            output.mkdir(parents=True, exist_ok=True)
            from .runtime import run_training
            trainer = run_training(cfg)
            result = dict(status="TRAINING_COMPLETE" if trainer.global_step == trainer.max_steps else "TRAINING_STOPPED_AT_REQUESTED_STEP",
                          last_recorded_step=trainer.global_step, max_steps=cfg.max_steps, stop_after_step=cfg.get("stop_after_step"),
                          output_dir=str(output), checkpoints=str(output / "checkpoints"))
            if int(os.environ.get("RANK", "0")) == 0:
                write_json(output / "summary.json", result)
    print(result["status"], flush=True)
    print("Result:", str(getattr(args, "output", None) or args.output_dir), flush=True)


if __name__ == "__main__":
    main()
