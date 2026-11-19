from dataclasses import dataclass


import torch

from .kflow import KFlowContract

from .conditioning import WorldConditioningContract

from .normalization import FixedFeatureNormalizationContract

from .hybrid_normalization import HybridHeadWristNormalizationContract, _finite_tuple

from .hybrid_fastwam_pixeldit import _masked_region_mean

from .momap_flow import MoMapVelocityLoss

from .t2i_mwam import T2IFastWAM


OFFSETS=(8,16,24,32)

HORIZONS=(.4,.8,1.2,1.6)

CAMERAS=('agentview','robot0_eye_in_hand')

FRAME='agentview@source_t0'

ACTION='libero_osc_delta_xyz_rotation_gripper7_v1'

PROPRIO='libero_eef_xyz_rotvec_gripper_qpos8_v1'

FLOW='libero_two_view_canonical_dxyz_v1'

ARCH='libero_t2i_allxyz_g26p4_a30_v1'

@dataclass(frozen=True)
class LiberoWorldConditioning(WorldConditioningContract):
    reference_frame: str = FRAME
    segmentation_encoding: str = 'current_front_instance_rgb_signed_wrist_zero'
    def __post_init__(self):
        if self.reference_frame != FRAME or (self.rgb_channels, self.segmentation_channels,
                self.canonical_xyz_channels, self.raw_depth_channels) != (3, 3, 3, 0):
            raise ValueError('Wrong LIBERO source condition')

@dataclass(frozen=True)
class LiberoXYZKFlow(KFlowContract):
    def __post_init__(self):
        for key in ('future_offsets','future_horizons_seconds','camera_order'):
            object.__setattr__(self,key,tuple(getattr(self,key)))
        size = 224 if self.view_mode == "libero_two_camera_vertical_448x224_v1" else 128
        for key,value in _kflow_fields(size).items():
            if getattr(self,key)!=value:
                raise ValueError(f'LIBERO XYZ {key} mismatch')
    @property
    def requires_full_rank_pixel_path(self):return True
    @property
    def requires_full_rank_output_path(self):return True

def _kflow_fields(camera_size=128):
    if camera_size not in (128,224):raise ValueError("Unsupported LIBERO camera size")
    return dict(future_offsets=OFFSETS,future_horizons_seconds=HORIZONS,
                future_offset_basis='libero_control_frame@20hz',
                view_mode=f'libero_two_camera_vertical_{camera_size*2}x{camera_size}_v1',camera_order=CAMERAS,
                source_coordinate_frame=FRAME,coordinate_frame=FRAME,
                source_normalization='none',target_representation=FLOW,
                unit='meter',channel_order='k_major_xyz',architecture_version=ARCH)

def libero_xyz_kflow(camera_size=128):return LiberoXYZKFlow(**_kflow_fields(camera_size))

@dataclass(frozen=True)
class LiberoFeatureNormalization(FixedFeatureNormalizationContract):
    schema_version: str='libero_fixed_feature_zscore_v1'
    @property
    def feature_dim(self):return {'action':7,'proprio':8}[self.feature_kind]
    def __post_init__(self):
        if self.schema_version!='libero_fixed_feature_zscore_v1' or self.mode!='fixed_feature_zscore':
            raise ValueError('Wrong LIBERO feature normalization schema')
        if self.feature_kind not in ('action','proprio'):
            raise ValueError('Unknown LIBERO feature')
        if self.semantic_kind!={'action':ACTION,'proprio':PROPRIO}[self.feature_kind]:
            raise ValueError('Wrong LIBERO feature semantics')
        if self.source_normalization!='none' or not self.statistics_id:
            raise ValueError('Raw LIBERO features and train statistics required')
        for k in ('mean','std'):
            object.__setattr__(self,k,_finite_tuple(getattr(self,k),name=k,length=self.feature_dim,positive=k=='std'))
        digest=self.recompute_digest()
        if self.digest is not None and self.digest!=digest:raise ValueError('Feature statistics digest mismatch')
        object.__setattr__(self,'digest',digest)

@dataclass(frozen=True)
class LiberoXYZNormalization(HybridHeadWristNormalizationContract):
    head_height: int=128
    full_height: int=256
    full_width: int=128
    reference_frame: str=FRAME
    source_unit: str='meter'
    schema_version: str='libero_two_view_xyz_pooled_zscore_v1'
    mode: str='libero_train_xyz_flow_pooled_zscore_v1'
    @property
    def uses_pooled_flow_zscore(self):return True
    def __post_init__(self):
        if self.schema_version!='libero_two_view_xyz_pooled_zscore_v1' or self.mode!='libero_train_xyz_flow_pooled_zscore_v1':
            raise ValueError('Wrong LIBERO XYZ statistics schema')
        if not self.statistics_id or tuple(self.future_offsets)!=OFFSETS:
            raise ValueError('Statistics ID and K4 offsets required')
        object.__setattr__(self,'future_offsets',OFFSETS)
        if (self.head_height,self.full_height,self.full_width) not in ((128,256,128),(224,448,224)):
            raise ValueError('Two square views of 128 or 224 pixels required')
        if self.reference_frame!=FRAME or self.source_unit!='meter' or self.channel_order!='k_major_xyz':
            raise ValueError('Metric source-time reference required')
        for k in ('source_mean_xyz_meters','source_std_xyz_meters','head_flow_mean_xyz_meters',
                  'head_flow_std_xyz_meters','wrist_flow_mean_xyz_meters','wrist_flow_std_xyz_meters'):
            object.__setattr__(self,k,_finite_tuple(getattr(self,k),name=k,length=3,positive='_std_' in k))
        for k in ('head_scale_channels_meters','wrist_scale_channels_meters'):
            object.__setattr__(self,k,_finite_tuple(getattr(self,k),name=k,length=0))
        digest=self.recompute_digest()
        if self.digest is not None and self.digest!=digest:raise ValueError('XYZ statistics digest mismatch')
        object.__setattr__(self,'digest',digest)

class LiberoT2IFastWAM(T2IFastWAM):
    def _compute_momap_velocity_loss(self,prediction,target_velocity,*,inputs,timestep_weight):
        h0 = self.normalization_contract.head_height
        shape = (12,2*h0,h0)
        if prediction.shape!=target_velocity.shape or tuple(prediction.shape[1:])!=shape:
            raise ValueError(f'LIBERO velocity must be [B,{shape}]')
        valid=inputs['flow_loss_valid'].to(prediction.device)
        if valid.dtype!=torch.bool or tuple(valid.shape)!=(len(prediction),4,3,2*h0,h0):
            raise ValueError('LIBERO requires component-level XYZ validity')
        error=(prediction.float()-target_velocity.float()).square().reshape_as(valid)
        h,hv=_masked_region_mean(error[:,:,:,:h0],valid[:,:,:,:h0])
        w,wv=_masked_region_mean(error[:,:,:,h0:],valid[:,:,:,h0:])
        if not getattr(self,'ALLOW_EMPTY_FLOW_VIEWS',False) and not bool(hv.any(1).all()):
            raise ValueError('No valid front XYZ supervision')
        present=hv|wv
        value=(h*hv+w*wv)/(hv.float()+wv.float()).clamp_min(1)
        per_sample=(value*present).sum(1)/present.sum(1).clamp_min(1)
        weight=torch.as_tensor(timestep_weight,device=prediction.device,dtype=per_sample.dtype)
        if weight.ndim==0:weight=weight.expand_as(per_sample)
        if weight.shape!=per_sample.shape or not bool(torch.isfinite(weight).all()):raise ValueError('Invalid time weight')
        self._last_hybrid_world_loss_metrics={
            'loss/momap_head_3d_raw':float((h.sum()/hv.sum().clamp_min(1)).detach()),
            'loss/momap_wrist_3d_raw':float((w.sum()/wv.sum().clamp_min(1)).detach()),
            'valid_ratio/head_3d':float(valid[:,:,:,:h0].float().mean()),
            'valid_ratio/wrist_3d':float(valid[:,:,:,h0:].float().mean())}
        return MoMapVelocityLoss(raw=per_sample.mean(),weighted=(per_sample*weight).mean(),
                    per_sample=per_sample,valid_ratio_by_horizon=valid.float().mean((0,2,3,4)),
                    valid_horizon_ratio=present.float().mean(0))
