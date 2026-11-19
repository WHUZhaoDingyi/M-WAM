"""Four-suite LIBERO + Plus, action tails and partial cumulative 3D flow."""
from bisect import bisect_right
from collections import OrderedDict
import hashlib,json,os
from operator import index as integer_index
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .t2i_training import LiberoT2IDataset,foreground_ids
from .segmentation import PredictedSegmentationStore
from mwam.datasets.robotwin_momap.source_conditions import encode_episode_scoped_segmentation
from mwam.datasets.robotwin_momap.gemma_cache import GemmaCachedTextContextEncoder
from mwam.models.pixeldit.t2i_libero import OFFSETS,HORIZONS,CAMERAS,ACTION,PROPRIO,libero_xyz_kflow
from mwam.utils.clip_palette import CLIP_PALETTE_SCHEMA, ClipSampleIndex, clip_sample_key

SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
SCHEMA='libero_plus_four_suites_v2'


def pad_actions(actions,*,domain,horizon=32):
    """Padding in raw control units, BEFORE normalization and diffusion."""
    a=np.asarray(actions,dtype=np.float32)
    dim={'libero':7,'robotwin':14}[domain]
    if a.ndim!=2 or a.shape[1]!=dim or not 0<len(a)<=horizon or not np.isfinite(a).all():
        raise ValueError('Expected 1..horizon real finite actions')
    out=np.repeat(a[-1:],horizon,axis=0)
    if domain=='libero':out[:,:6]=0
    out[:len(a)]=a
    return out,np.arange(horizon)>=len(a)


def raster_resize(value,size,*,bilinear=False):
    """Resample H,W with one nearest mapping for all geometric quantities."""
    x=torch.from_numpy(np.ascontiguousarray(value))
    shape=x.shape;flat=x.reshape(-1,1,*shape[-2:]).float()
    y=F.interpolate(flat,size=(size,size),mode='bilinear' if bilinear else 'nearest',
                    **({'align_corners':False} if bilinear else {}))
    return y.reshape(*shape[:-2],size,size).to(x.dtype).numpy()


def pack_aligned_source(rgb,xyz,valid,actor_ids,poses,identity,size=224):
    # actor_ids is the legacy argument name. In predicted mode the caller
    # supplies DINO/SAM2 track IDs here, never simulator instance IDs.
    # Fresh training passes a clip-draw identity, so this deterministic color
    # encoder changes palettes on every draw without changing the ID masks.
    ref=np.linalg.inv(np.asarray(poses[0],np.float64));canonical=[];images=[];masks=[];segs=[]
    for i in range(2):
        point=np.asarray(xyz[i]);v=np.asarray(valid[i],bool)&np.isfinite(point).all(-1)
        # Nearest source-point sampling is exactly equivalent to sampling
        # depth AND its matching ray/entity with this same mapping, then
        # reconstructing XYZ. Do not interpolate across object boundaries.
        clean=np.where(v[...,None],point,0)
        t=ref@np.asarray(poses[i],np.float64)
        clean=clean@t[:3,:3].T+t[:3,3]
        clean=np.where(v[...,None],clean,0).astype(np.float32)
        canonical.append(raster_resize(np.moveaxis(clean,-1,0),size))
        images.append(raster_resize(np.moveaxis(np.asarray(rgb[i],np.float32),-1,0),size,bilinear=True)/127.5-1)
        masks.append(raster_resize(v,size))
        seg=encode_episode_scoped_segmentation(actor_ids[i],episode_identity=identity) if i==0 else np.zeros((*v.shape,3),np.float32)
        segs.append(raster_resize(np.moveaxis(seg,-1,0),size))
    return dict(source_rgb=torch.from_numpy(np.concatenate(images,axis=1)),
        source_xyz_ref=torch.from_numpy(np.concatenate(canonical,axis=1)),
        source_seg_condition=torch.from_numpy(np.concatenate(segs,axis=1)),
        source_condition_valid=torch.from_numpy(np.concatenate(masks,axis=0)[None]))


def collate(samples):
    batch={k:torch.stack([s[k] for s in samples]) for k in samples[0]}
    c=libero_xyz_kflow(224)
    for key in ('future_offsets','future_horizons_seconds','future_offset_basis','coordinate_frame','unit','source_normalization','target_representation'):
        batch[key]=getattr(c,key)
    batch.update(action_label_kind=ACTION,action_is_high_level_policy_action=False,
                 action_normalization='none',proprio_mode=PROPRIO,proprio_normalization='none')
    return batch


class RevisedLiberoDataset(LiberoT2IDataset):
    collate_fn=staticmethod(collate)
    MANIFEST_SCHEMA=SCHEMA
    TRAINING_DOMAINS=('libero','plus')
    DEFAULT_SOURCE_PERCENTAGES={'base_libero':20,'base_plus':80}
    SAMPLER_TYPE=None

    def __init__(self,manifest_path,normalization_artifact_path=None,split='train',
                 max_open_episodes=2,text_embedding_cache_dir=None,context_encoder=None,
                 source_percentages=None,segmentation_mode='oracle',segmentation_manifest_path=None,
                 segmentation_palette='episode'):
        if segmentation_palette not in ('episode','clip_draw'):
            raise ValueError('LIBERO segmentation_palette must be episode or clip_draw')
        self.segmentation_palette=segmentation_palette
        self._direct_palette_draw=0
        self.training_sample_key=(clip_sample_key if segmentation_palette=='clip_draw' and split=='train' else None)
        if segmentation_mode not in ('oracle','predicted'):
            raise ValueError('LIBERO segmentation_mode must be oracle or predicted')
        if segmentation_mode=='oracle' and segmentation_manifest_path is not None:
            raise ValueError('Oracle segmentation must not silently ignore a predicted manifest')
        self.segmentation_mode=segmentation_mode
        self.segmentation_store=None
        manifest=json.loads(Path(manifest_path).read_text())
        if manifest['schema_version']!=self.MANIFEST_SCHEMA or split not in ('train','val'):raise ValueError('Revised manifest required')
        self.manifest_path=str(Path(manifest_path).resolve())
        self.split_digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        if normalization_artifact_path:
            stats=json.loads(Path(normalization_artifact_path).read_text())
            if stats['schema_version']!=self.MANIFEST_SCHEMA or stats['split_digest']!=self.split_digest:raise ValueError('Statistics/manifest mismatch')
        self.split=split;self.records=[r for r in manifest['episodes'] if r['split']==split]
        if any(r['suite'] not in SUITES for r in self.records):raise ValueError('LIBERO-90 is excluded from this recipe')
        self.starts=[];self.end=[];total=0
        for row in self.records:
            if row['proprio_semantics']!='official_observation' or row['pixel_transform'] not in ('identity','horizontal_flip'):
                raise ValueError('Unverified proprio/image convention')
            if row['action_count']>row['frames']:raise ValueError('Actions without source observations')
            starts=np.asarray(row['source_indices'],np.int64)
            if len(starts) and (starts.min()<0 or starts.max()>=row['action_count'] or len(np.unique(starts))!=len(starts)):
                raise ValueError('Invalid action-boundary source indices')
            self.starts.append(starts);total+=len(starts);self.end.append(total)
        self.indices=None;self.valid_index_digest=self.split_digest
        self.max_open=int(max_open_episodes);self.cache=OrderedDict();self.pid=os.getpid();self.context_cache=OrderedDict()
        self.context_encoder=context_encoder
        self.encoders={k:GemmaCachedTextContextEncoder(v) for k,v in manifest.get('text_caches',{}).items()} if context_encoder is None else {}
        self.proprio_cache=OrderedDict()
        if segmentation_mode=='predicted':
            self.segmentation_store=PredictedSegmentationStore(self.records,segmentation_manifest_path,max_open=self.max_open)
        if split=='train':
            self.source_percentages=dict(source_percentages or self.DEFAULT_SOURCE_PERCENTAGES)
            if list(self.source_percentages)!=['base_'+d for d in self.TRAINING_DOMAINS] or sum(self.source_percentages.values())!=100 or min(self.source_percentages.values())<=0:
                raise ValueError('Explicit positive LIBERO/Plus percentages required')
            domains=[r['domain'] for r in self.records]
            if domains!=sorted(domains,key=self.TRAINING_DOMAINS.index):raise ValueError('Expected contiguous domain partitions')
            self.source_lengths=tuple(sum(len(s) for r,s in zip(self.records,self.starts) if r['domain']==d) for d in self.TRAINING_DOMAINS)
            if min(self.source_lengths)<=0:raise ValueError('Both domains require real samples')
            if self.SAMPLER_TYPE is not None:self.training_sampler_type=self.SAMPLER_TYPE


    def _window_index_and_palette(self,index):
        if isinstance(index,ClipSampleIndex):
            if self.segmentation_palette!='clip_draw' or self.split!='train':
                raise ValueError('Clip draw key requires clip_draw training palette')
            return index.index,index.palette_identity()
        index=integer_index(index)
        if self.segmentation_palette=='episode':
            return index,None
        if self.split!='train':
            return index,f'{CLIP_PALETTE_SCHEMA}:validation:index={index}'
        # Interactive reads/checks can still request a plain window index.
        # Production training receives a sampler key, never this local counter.
        worker=torch.utils.data.get_worker_info()
        worker_id=worker.id if worker is not None else -1
        draw=self._direct_palette_draw
        self._direct_palette_draw+=1
        return index,f'{CLIP_PALETTE_SCHEMA}:direct:worker={worker_id}:draw={draw}:index={index}'

    @staticmethod
    def _source_palette_identity(row,start,palette):
        identity=row['source_group']
        return identity if palette is None else f'{identity}\0clip_start={start}\0{palette}'

    def close(self):
        super().close()
        if self.segmentation_store is not None:self.segmentation_store.close()

    def __getstate__(self):
        state=super().__getstate__();state['proprio_cache']=OrderedDict();return state

    def load_window(self,index,*,include_text):
        index,palette=self._window_index_and_palette(index)
        if not 0<=index<len(self):raise IndexError(index)
        episode=bisect_right(self.end,index);row=self.records[episode]
        start=int(self.starts[episode][index-(self.end[episode-1] if episode else 0)])
        reader,service,handle=self._open(row['geometry_path'])
        if reader.camera_names!=CAMERAS:raise ValueError('Wrong camera order')
        current=reader.read_frame_fields(start,observation_fields=('rgb',),private_geometry_fields=('position_xyz','position_valid','actor_seg_raw','T_cam_to_world'),robot_state_fields=())
        g=[current['private_geometry'][c] for c in CAMERAS]
        flip=row['pixel_transform']=='horizontal_flip'
        def spatial(x):return np.ascontiguousarray(x[:,::-1]) if flip else x
        if self.segmentation_mode=='predicted':
            head_ids=self.segmentation_store.read(row,start,
                expected_shape=current['observations'][CAMERAS[0]]['rgb'].shape[:2])
            condition_ids=[head_ids,np.zeros_like(g[1]['actor_seg_raw'])]
        else:
            condition_ids=[v['actor_seg_raw'] for v in g]
        sample=pack_aligned_source([spatial(current['observations'][c]['rgb']) for c in CAMERAS],
            [spatial(v['position_xyz']) for v in g],[spatial(v['position_valid']) for v in g],
            [spatial(v) for v in condition_ids],[v['T_cam_to_world'] for v in g],
            self._source_palette_identity(row,start,palette))
        available=[o for o in OFFSETS if start+o<row['frames']]
        target=np.zeros((4,3,448,224),np.float32);valid=np.zeros_like(target,bool)
        for view,camera in enumerate(CAMERAS):
            if not available:continue
            t=service.get_momap(camera,start,available,grid='full',reference_camera_name='agentview',representation='canonical_dxyz')
            expected=np.asarray(available)/20
            if not np.allclose(t.dt_seconds,expected,atol=1e-6,rtol=0):raise ValueError('Wrong control/geometry time alignment')
            flow=np.where(t.component_valid,t.momap,0);mask=t.component_valid
            if flip:flow=flow[...,::-1].copy();mask=mask[...,::-1].copy()
            flow=raster_resize(flow,224);mask=raster_resize(mask,224)
            for n,o in enumerate(available):
                k=OFFSETS.index(o);target[k,:,view*224:(view+1)*224]=flow[n];valid[k,:,view*224:(view+1)*224]=mask[n]
        loss_valid=valid.copy()
        foreground=np.isin(spatial(g[1]['actor_seg_raw']),foreground_ids(reader.entity_registry,row['task_objects']))
        loss_valid[:,:,224:] &= raster_resize(foreground,224)[None,None]
        raw=handle['robot_state/action_target'][start:min(start+32,row['action_count'])]
        action,pad=pad_actions(raw,domain='libero')
        path=row['proprio_path']
        if path not in self.proprio_cache:
            self.proprio_cache[path]=np.load(path,mmap_mode='r')
            while len(self.proprio_cache)>self.max_open:self.proprio_cache.popitem(last=False)
        proprio=np.asarray(self.proprio_cache[path][start],np.float32).copy()
        if proprio.shape!=(8,) or not np.isfinite(proprio).all():raise ValueError('Invalid official proprio')
        sample.update(target_flow=torch.from_numpy(target.reshape(12,448,224)),flow_valid=torch.from_numpy(valid.any(1,keepdims=True)),
            flow_component_valid=torch.from_numpy(valid),flow_loss_valid=torch.from_numpy(loss_valid),
            action=torch.from_numpy(action),action_is_pad=torch.from_numpy(pad),proprio=torch.from_numpy(proprio)[None],
            frame_offsets=torch.tensor(OFFSETS),momap_dt_seconds=torch.tensor(HORIZONS,dtype=torch.float64),
            momap_lateness_seconds=torch.zeros(4,dtype=torch.float64))
        # dt slots with invalid masks are requested horizons, not observations.
        if include_text:
            key=(row['domain'],row['instruction'])
            if key not in self.context_cache:
                encoder=self.context_encoder or self.encoders[row['domain']]
                self.context_cache[key]=encoder(row['instruction'])
                while len(self.context_cache)>32:self.context_cache.popitem(last=False)
            text=self.context_cache[key];sample.update(context=text['context'],context_mask=text['mask'])
        return sample
