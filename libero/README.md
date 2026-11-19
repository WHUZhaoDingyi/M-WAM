# M-WAM training on LIBERO

Train the revised M-WAM policy on the original LIBERO Spatial, Object, Goal and Long suites. Each clip pairs the current observation with a 32-step action chunk and four dense 3D displacement targets. This package contains the training code for the LIBERO run.

## Model and targets

| | Training recipe |
|---|---|
| Backbone | PixelDiT T2I initialization, global width 1536, World 14 global + 2 pixel blocks, Action 14 global + 2 tail blocks |
| Observation | Two 224 × 224 views stacked vertically. RGB and XYZ in both views, predicted instance segmentation in the main view, zeros in the wrist segmentation channels |
| Context | Cached Gemma text features and one token for the current 8D robot state (position, rotation vector, two finger positions) |
| Actions | 32 × 7 OSC commands, with real action positions contributing to the loss |
| Motion | Source-point XYZ displacement at steps +8, +16, +24 and +32 in the source main-camera frame. Unavailable horizons are masked. Wrist supervision covers task objects and robot parts |
| Objective | Sum of action and 3D motion flow-matching losses. Noise levels are drawn independently. The clean condition features use the World time embedding during training |
| Augmentation | One palette per clip draw. Repeated draws of the same episode receive different palettes |
| Optimizer | AdamW, peak learning rate 1e-4, 1,000-step warmup, cosine decay to 1e-6, 100,000 updates. Reference run used four GPUs, batch 8 per GPU, one optimizer step per batch |

## Inputs and launch

Use Python 3.10 and a CUDA-enabled PyTorch installation compatible with FSDP2. The reference run used PyTorch 2.9.1 with CUDA 12.8 and four 48 GB GPUs.

```bash
python -m pip install -e .
```

Start with exported episode geometry, aligned proprioception, main-camera DINO/SAM2 tracks and the Gemma instruction cache. Create the index described in [training inputs](DATA.md), then build the local manifests and training statistics:

```bash
python prepare_data.py --index data/index.json --out data/ready
```

Initialize the model with `pixeldit_t2i_v1.pth` from [NVIDIA PixelDiT](https://huggingface.co/nvidia/PixelDiT-1300M-1024px/tree/7c63b99a7a399918a1d6478b095698a65f664847). The reference file has SHA-256 `625fd174d6348af3ad1123b281ab817643efa5dc447c2344eeb222e88b4317db`.

```bash
torchrun --standalone --nproc-per-node=4 train.py \
  --data data/ready --pretrained data/pixeldit_t2i_v1.pth --output runs/libero
```

At the final training step, rank zero writes the full model weights to `runs/libero/model.pt` as a PyTorch state dictionary.
