"""Load the Wan text encoder for caching instruction embeddings."""
import torch

from .io import hash_model_file, load_state_dict
from ..wan_video_text_encoder import WanTextEncoder


def load_text_encoder(path, device):
    # The Wan checkpoint is identified by its original names and tensor shapes.
    if hash_model_file(path) != "9c8818c2cbea55eca56c7b447df170da":
        raise ValueError(f"Expected the Wan UMT5-XXL text encoder weights: {path}")
    with torch.device("meta"):
        model = WanTextEncoder()
    state = load_state_dict(path, torch_dtype=torch.bfloat16, device="cpu")
    missing = model.load_state_dict(state, strict=False, assign=True).missing_keys
    if missing:
        model = WanTextEncoder()
        model.load_state_dict(state, strict=False)
    del state
    return model.eval().to(device=device, dtype=torch.bfloat16)
