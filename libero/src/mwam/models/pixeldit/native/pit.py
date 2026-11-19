"""Native PiT pixel modulation/compression, with directed condition visibility."""
from __future__ import annotations
import torch
from torch import nn
from ..model import PiTBlock
from ..modules import apply_adaln
from ..t2i_mwam import stable_norms
from .attention import attend


class DirectedPiT(nn.Module):
    def __init__(self, pixel_dim=16, context_dim=1536, attn_dim=1152, heads=16, patch_size=16):
        super().__init__()
        self.patch_size, self.pixel_dim, self.attn_dim, self.heads = patch_size, pixel_dim, attn_dim, heads
        for name in ("condition", "target"):
            setattr(self, name, PiTBlock(pixel_dim, context_dim, patch_size=patch_size,
                num_heads=heads, attn_hidden_size=attn_dim, attn_num_heads=heads))
        stable_norms(self)

    def prepare(self, module, pixels, condition, batch, position, identity):
        groups, p2, dim = pixels.shape
        mods = module.adaLN_modulation(condition).reshape(groups, p2, 6 * dim).chunk(6, -1)
        h = apply_adaln(module.norm1(pixels), mods[0], mods[1])
        h = module.compress_to_attn(h.reshape(groups, -1)).reshape(batch, -1, self.attn_dim)
        q, k, v = module.attn.project_qkv(h, position)
        identity = identity.reshape(1, 1, self.heads, self.attn_dim // self.heads).to(v.dtype)
        return q.to(v.dtype) + identity, k.to(v.dtype) + identity, v, mods

    def finish(self, module, pixels, mods, output):
        output = module.attn.proj(output)
        h = module.expand_from_attn(output.reshape(-1, self.attn_dim)).reshape_as(pixels)
        pixels = pixels + mods[2] * h
        return pixels + mods[5] * module.mlp(apply_adaln(module.norm2(pixels), mods[3], mods[4]))

    def forward(self, c, w, cc, wc, batch, position, identities, bidirectional=False):
        qc, kc, vc, cm = self.prepare(self.condition, c, cc, batch, position, identities[0])
        if bidirectional:
            if w is None:
                raise ValueError("Bidirectional PiT attention requires sampled W")
            qw, kw, vw, wm = self.prepare(self.target, w, wc, batch, position, identities[1])
            oc, ow = attend(torch.cat((qc, qw), 1), torch.cat((kc, kw), 1),
                            torch.cat((vc, vw), 1)).split((qc.shape[1], qw.shape[1]), 1)
            return (self.finish(self.condition, c, cm, oc),
                    self.finish(self.target, w, wm, ow), kc, vc, kw, vw)
        c = self.finish(self.condition, c, cm, attend(qc, kc, vc))
        kw = vw = None
        if w is not None:
            qw, kw, vw, wm = self.prepare(self.target, w, wc, batch, position, identities[1])
            w = self.finish(self.target, w, wm, attend(qw, torch.cat((kc, kw), 1), torch.cat((vc, vw), 1)))
        return c, w, kc, vc, kw, vw
