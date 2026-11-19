"""Joint LIBERO model with separate World and Action noise levels."""
from dataclasses import dataclass
import json
from pathlib import Path
import torch
from ..normalization import FixedFeatureNormalizationContract
from ..t2i_libero import LiberoXYZNormalization, ACTION, PROPRIO, libero_xyz_kflow


@dataclass(frozen=True)
class LiberoMinMaxNormalization(FixedFeatureNormalizationContract):
    """Affine FastWAM min/max; mean/std store inverse offset and scale.

    Constant dimensions use scale=1 and offset=-minimum, as upstream does.
    std is also the physical velocity error scale used by the joint metrics.
    """
    minimum: tuple = ()
    maximum: tuple = ()
    schema_version: str = 'libero_fastwam_minmax_v1'
    mode: str = 'min/max'

    @property
    def feature_dim(self):
        return {'action':7, 'proprio':8}[self.feature_kind]

    def __post_init__(self):
        if self.schema_version != 'libero_fastwam_minmax_v1' or self.mode != 'min/max':
            raise ValueError('Wrong LIBERO min/max schema')
        if self.semantic_kind != {'action':ACTION, 'proprio':PROPRIO}[self.feature_kind]:
            raise ValueError('Wrong feature semantics')
        if self.source_normalization != 'none' or not self.statistics_id:
            raise ValueError('Raw features and training statistics required')
        lo, hi = torch.tensor(self.minimum,dtype=torch.float64), torch.tensor(self.maximum,dtype=torch.float64)
        if lo.shape != (self.feature_dim,) or hi.shape != lo.shape or not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or (hi<lo).any():
            raise ValueError('Invalid min/max statistics')
        span = hi-lo
        std = torch.where(span<1e-4, 1., span/2)
        mean = torch.where(span<1e-4, lo, (hi+lo)/2)
        if not torch.allclose(torch.tensor(self.mean,dtype=torch.float64),mean,atol=1e-12,rtol=1e-12) or not torch.allclose(torch.tensor(self.std,dtype=torch.float64),std,atol=1e-12,rtol=1e-12):
            raise ValueError('Min/max inverse coefficients differ from upstream rule')
        for key in ('mean','std','minimum','maximum'):
            object.__setattr__(self,key,tuple(float(v) for v in getattr(self,key)))
        digest=self.recompute_digest()
        if self.digest is not None and self.digest!=digest:raise ValueError('Statistics digest mismatch')
        object.__setattr__(self,'digest',digest)

    def _digest_payload(self):
        return dict(super()._digest_payload(),minimum=list(self.minimum),maximum=list(self.maximum),clamp=[-5.,5.])

    def normalize(self,value_raw):
        return super().normalize(value_raw).clamp(-5.,5.)


def create_revised_libero(*, normalization_artifact_path, pixeldit_pretrained_path, device='cuda',
                          use_gradient_checkpointing=True, uniform_noise=False):
    """Build the joint LIBERO model and initialize its backbone from official PixelDiT."""
    from .factory import build

    statistics = json.loads(Path(normalization_artifact_path).read_text())
    if statistics['schema_version'] != 'libero_four_suites_v1':
        raise ValueError('Original LIBERO four-suite training statistics required')
    return build(
        geometry=LiberoXYZNormalization(**statistics['geometry']),
        kflow=libero_xyz_kflow(224),
        action_norm=LiberoMinMaxNormalization(**statistics['action']),
        proprio_norm=LiberoMinMaxNormalization(**statistics['proprio']),
        pixeldit_pretrained_path=pixeldit_pretrained_path, device=device,
        use_gradient_checkpointing=use_gradient_checkpointing,
        uniform_noise=uniform_noise,
    )
