"""Configuration check, checkpoint loading and single-observation inference."""
import argparse
import json
from pathlib import Path


def main():
    import sys
    from .train_cli import COMMANDS
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        from .train_cli import main as training_main
        return training_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.epilog = "Data/training commands (COMMAND --help): " + ", ".join(COMMANDS)
    parser.add_argument("command", choices=("check-config", "check-load", "example-input", "infer"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pixeldit-pretrained", type=Path)
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--verify-direct", action="store_true", help="Compare with direct native infer_joint using the same seed")
    args = parser.parse_args()
    if args.threads < 1 or (args.steps is not None and args.steps < 1) or (args.seed is not None and args.seed < 0):
        parser.error("threads/steps must be positive; seed must be nonnegative")
    import numpy as np
    import torch
    from .config import load_config
    from .observation import load_observation, synthetic_observation, TENSOR_KEYS
    torch.set_num_threads(args.threads)
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output path; existing results are not overwritten")
    result = {"method": "M-WAM", "status": "CONFIG_VALID", "full_forward_tested": False}
    if args.command == "example-input":
        if output.suffix != ".npz":
            parser.error("example-input requires --output ending in .npz")
        observation = synthetic_observation(0 if args.seed is None else args.seed)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **{k: observation[k].numpy() for k in TENSOR_KEYS})
        print(f"Synthetic interface example only (not evaluation data): {output}")
        return
    if args.config is None:
        parser.error("--config is required")
    config = load_config(args.config)
    result["config"] = str(args.config.resolve())
    if args.command != "check-config":
        if args.checkpoint is None or args.pixeldit_pretrained is None:
            parser.error("--checkpoint and --pixeldit-pretrained are required")
        observation = None
        if args.command == "infer":
            if args.observation is None:
                parser.error("infer requires --observation")
            observation = load_observation(args.observation)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            parser.error("CUDA is unavailable; check-load can use --device cpu")
        from .policy import MWAMPolicy
        policy = MWAMPolicy.from_checkpoint(
            config, args.checkpoint, pixeldit_pretrained=args.pixeldit_pretrained, device=args.device,
        )
        result.update(status="CHECKPOINT_LOADED", checkpoint=str(args.checkpoint.resolve()),
                      checkpoint_step=policy.checkpoint_step, device=args.device,
                      parameter_count=sum(p.numel() for p in policy.model.parameters()))
        if args.command == "infer":
            if output.suffix != ".npz":
                parser.error("infer requires --output ending in .npz")
            seed = config.inference.seed if args.seed is None else args.seed
            steps = config.inference.num_inference_steps if args.steps is None else args.steps
            action = policy.predict_action(observation, num_inference_steps=steps, seed=seed)
            if not torch.isfinite(action).all():
                raise ValueError("Nonfinite predicted action")
            if args.verify_direct:
                with torch.inference_mode():
                    reference = policy.model.infer_joint(observation, num_inference_steps=steps, seed=seed)["action_raw"]
                torch.testing.assert_close(action, reference, rtol=0, atol=0)
                result["direct_solver_bitwise_equal"] = True
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, action_raw=action.float().cpu().numpy())
            result.update(status="INFERENCE_COMPLETE", full_forward_tested=True, seed=seed, steps=steps,
                          action_shape=list(action.shape), observation=str(args.observation.resolve()),
                          future_branch="computed_by_native_joint_solver")
            output = output.with_suffix(".summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['status']}; full forward tested: {result['full_forward_tested']}")
    print(f"Result: {output}")


if __name__ == "__main__":
    main()
