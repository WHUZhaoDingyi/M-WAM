"""T2I-compatible MM-DiT operations and directed C/Y -> W visibility."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from ..modules import FeedForward, apply_adaln, apply_rotary_emb
from ..t2i_mwam import StableRMSNorm


def rope_1d(length, head_dim, device, theta=10000.):
    freq = theta ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    angle = torch.arange(length, device=device).float()[:, None] * freq[None]
    return torch.polar(torch.ones_like(angle), angle)


def attend(q, k, v, valid=None):
    mask = None if valid is None else valid[:, None, None, :]
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return out.transpose(1, 2).flatten(2)


class JointAttention(nn.Module):
    def __init__(self, dim, heads, condition_branch=False):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = dim, heads, dim // heads
        for branch in (("x", "y", "c") if condition_branch else ("x", "y")):
            setattr(self, "qkv_" + branch, nn.Linear(dim, 3 * dim, bias=False))
            setattr(self, "q_norm_" + branch, StableRMSNorm(self.head_dim))
            setattr(self, "k_norm_" + branch, StableRMSNorm(self.head_dim))
            setattr(self, "proj_" + branch, nn.Linear(dim, dim))

    def project(self, x, branch, position, identity=None):
        q, k, v = getattr(self, "qkv_" + branch)(x).reshape(
            *x.shape[:2], 3, self.num_heads, self.head_dim).unbind(2)
        q = getattr(self, "q_norm_" + branch)(q)
        k = getattr(self, "k_norm_" + branch)(k)
        if position is not None:
            q, k = apply_rotary_emb(q, k, position)
        # Identity is looked up once by the World, then reused at every layer.
        # It is added AFTER Q/K normalization and rotary position, never to V.
        if identity is not None:
            identity = identity.reshape(1, 1, self.num_heads, self.head_dim).to(v.dtype)
            q, k = q + identity, k + identity
        return q.to(v.dtype), k.to(v.dtype), v


class MMBlock(nn.Module):
    """Official image/text block, optionally with an independent C copy.

    C and W begin with the same T2I image-branch weights, but have independent
    parameters. Text follows the official evolving-token path.
    """
    def __init__(self, dim=1536, heads=24, condition_branch=False):
        super().__init__()
        self.attn = JointAttention(dim, heads, condition_branch)
        branches = ("x", "y", "c") if condition_branch else ("x", "y")
        for branch in branches:
            setattr(self, "norm_" + branch + "1", StableRMSNorm(dim))
            setattr(self, "norm_" + branch + "2", StableRMSNorm(dim))
            setattr(self, "mlp_" + branch, FeedForward(dim, 4 * dim))
        self.adaLN_modulation_img = nn.Sequential(nn.Linear(dim, 6 * dim))
        self.adaLN_modulation_txt = nn.Sequential(nn.Linear(dim, 6 * dim))
        if condition_branch:
            self.adaLN_modulation_cond = nn.Sequential(nn.Linear(dim, 6 * dim))

    def prepare(self, x, branch, condition, pos, identity=None):
        suffix = {"x": "img", "y": "txt", "c": "cond"}[branch]
        mods = getattr(self, "adaLN_modulation_" + suffix)(condition).chunk(6, -1)
        h = apply_adaln(getattr(self, "norm_" + branch + "1")(x), mods[0], mods[1])
        return self.attn.project(h, branch, pos, identity), mods

    def finish(self, x, branch, mods, attention):
        x = x + mods[2] * getattr(self.attn, "proj_" + branch)(attention)
        h = apply_adaln(getattr(self, "norm_" + branch + "2")(x), mods[3], mods[4])
        return x + mods[5] * getattr(self, "mlp_" + branch)(h)

    def world(self, c, w, y, condition, pos, pos_txt, text_valid, identities, bidirectional=False):
        (qc, kc, vc), cm = self.prepare(c, "c", condition, pos, identities[0])
        (qy, ky, vy), ym = self.prepare(y, "y", condition, pos_txt)
        valid = torch.cat((text_valid, torch.ones(c.shape[:2], device=c.device, dtype=torch.bool)), 1)
        kcy, vcy = torch.cat((ky, kc), 1), torch.cat((vy, vc), 1)
        if bidirectional:
            if w is None:
                raise ValueError("Bidirectional C/Y/W attention requires sampled W")
            (qw, kw, vw), wm = self.prepare(w, "x", condition, pos, identities[1])
            all_valid = torch.cat((valid, torch.ones(w.shape[:2], device=w.device, dtype=torch.bool)), 1)
            # Remove stream-direction restrictions; text padding remains invalid.
            oy, oc, ow = attend(torch.cat((qy, qc, qw), 1),
                torch.cat((kcy, kw), 1), torch.cat((vcy, vw), 1), all_valid).split(
                    (y.shape[1], c.shape[1], w.shape[1]), 1)
            return (self.finish(c, "c", cm, oc), self.finish(w, "x", wm, ow),
                    self.finish(y, "y", ym, oy), kc, vc, ky, vy, kw, vw)
        oy, oc = attend(torch.cat((qy, qc), 1), kcy, vcy, valid).split((y.shape[1], c.shape[1]), 1)
        c, y = self.finish(c, "c", cm, oc), self.finish(y, "y", ym, oy)
        kw = vw = None
        if w is not None:
            (qw, kw, vw), wm = self.prepare(w, "x", condition, pos, identities[1])
            all_valid = torch.cat((valid, torch.ones(w.shape[:2], device=w.device, dtype=torch.bool)), 1)
            ow = attend(qw, torch.cat((kcy, kw), 1), torch.cat((vcy, vw), 1), all_valid)
            w = self.finish(w, "x", wm, ow)
        return c, w, y, kc, vc, ky, vy, kw, vw

    def action(self, x, y, condition, pos_txt, text_valid, external_k, external_v, external_valid):
        # Trajectory identity uses the agreed fixed 1D positional vector at entry.
        (qx, kx, vx), xm = self.prepare(x, "x", condition, None)
        (qy, ky, vy), ym = self.prepare(y, "y", condition, pos_txt)
        valid = torch.cat((text_valid, torch.ones(x.shape[:2], device=x.device, dtype=torch.bool), external_valid), 1)
        k, v = torch.cat((ky, kx, external_k), 1), torch.cat((vy, vx, external_v), 1)
        oy, ox = attend(torch.cat((qy, qx), 1), k, v, valid).split((y.shape[1], x.shape[1]), 1)
        return self.finish(x, "x", xm, ox), self.finish(y, "y", ym, oy)
