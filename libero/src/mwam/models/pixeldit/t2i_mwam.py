from __future__ import annotations


import torch


from .modules import RMSNorm


from .hybrid_fastwam_pixeldit import HybridFastWAMPixelDiT


T2I_SHA256='625fd174d6348af3ad1123b281ab817643efa5dc447c2344eeb222e88b4317db'

class StableRMSNorm(RMSNorm):
    def forward(self,x):
        y=x.float();scale=y.abs().amax(-1,keepdim=True).clamp_min(1).detach();y=y/scale
        y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+(self.variance_epsilon/scale)/scale)
        return self.weight.to(x.dtype)*y.to(x.dtype)

def stable_norms(module):
    for m in module.modules():
        if isinstance(m,RMSNorm):m.__class__=StableRMSNorm

class T2IFastWAM(HybridFastWAMPixelDiT):

    ALLOW_EMPTY_FLOW_VIEWS = True
