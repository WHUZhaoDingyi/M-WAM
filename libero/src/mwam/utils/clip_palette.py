"""Worker-independent identities for per-draw clip augmentation.

The sampler supplies the absolute draw position before Accelerate shards the
stream. Prefetch, worker scheduling and a restored sampler cursor therefore
cannot change the palette assigned to a particular training draw.
"""
from dataclasses import dataclass
from operator import index as integer_index


CLIP_PALETTE_SCHEMA = "libero_clip_draw_palette_v1"


@dataclass(frozen=True)
class ClipSampleIndex:
    index: int
    seed: int
    epoch: int
    draw: int

    def __post_init__(self):
        for name in ("index", "seed", "epoch", "draw"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError("Clip draw identity requires integer " + name)
            value = integer_index(value)
            if name != "seed" and value < 0:
                raise ValueError("Clip draw identity requires nonnegative " + name)
            object.__setattr__(self, name, value)

    def palette_identity(self):
        return f"{CLIP_PALETTE_SCHEMA}:seed={self.seed}:epoch={self.epoch}:draw={self.draw}:index={self.index}"


def clip_sample_key(index, *, seed, epoch, draw):
    return ClipSampleIndex(index=index, seed=seed, epoch=epoch, draw=draw)
