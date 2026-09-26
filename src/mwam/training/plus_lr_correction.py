"""Source-bound PLUS LR correction; preserve Adam, RNG, sampler and step."""
import copy
import json
import math
from pathlib import Path

from mwam.utils.paths import asset_root

ROOT = asset_root()
RUN = ROOT/'runs/libero_split_16_32_t20260914/plus_formal'
SOURCE_STEP = 1187
SCHEMA = 'fastwam_plus32_lr_correction_20260914_v1'
PEAK = 1e-4
FLOOR = 1e-6
WARMUP_RATIO = 1/30

def contract(cfg):
    raw = cfg.get('plus_lr_correction')
    if raw is None: return None
    expected = dict(schema_version=SCHEMA,source_step=SOURCE_STEP,
                    source_state_dir=str(RUN/f'checkpoints/state/step_{SOURCE_STEP:06d}'))
    if dict(raw) != expected: raise ValueError('PLUS LR correction source contract differs')
    return expected

def allows(saved,current,cfg,source_path,source_step):
    spec=contract(cfg)
    if spec is None: return False
    if source_path is None or Path(source_path).resolve()!=Path(spec['source_state_dir']) or source_step!=SOURCE_STEP:
        return False
    if saved['runtime_topology']['world_size']!=32 or saved['sampler']['batch_size_per_rank']!=24:
        return False
    expected=copy.deepcopy(saved);train=expected['training']
    if (train['max_steps'],train['learning_rate'],train['cosine_min_learning_rate'],train['warmup_ratio'],train['lr_scheduler_type'])!=(30000,FLOOR,FLOOR,0.,'cosine'):
        return False
    if set(train['optimizer_parameter_groups'])!={'world','action_backbone','action_io'}: return False
    for group in train['optimizer_parameter_groups'].values():
        if group!=dict(learning_rate=FLOOR,cosine_min_learning_rate=FLOOR,warmup_start_learning_rate=FLOOR):return False
        group['learning_rate']=PEAK
    train['learning_rate']=PEAK;train['warmup_ratio']=WARMUP_RATIO
    return expected==current

def lr_at(step):
    if step<=1000:return FLOOR+(PEAK-FLOOR)*step/1000
    return FLOOR+(PEAK-FLOOR)*.5*(1+math.cos(math.pi*min((step-1000)/29000,1.)))

def apply_after_resume(trainer):
    spec=contract(trainer.cfg)
    if spec is None:return
    source=trainer.loaded_resume_source
    path=Path(source['source_path']).resolve()
    from .topology_transition import permits_source_path
    if (path.parent!=RUN/'checkpoints/state' and not permits_source_path(trainer,path)) or source['source_global_step']!=trainer.global_step:
        raise ValueError('PLUS correction resume run/cursor differs')
    scheduler=getattr(trainer.scheduler,'scheduler',trainer.scheduler)
    if trainer.global_step>SOURCE_STEP:
        if scheduler.base_lrs!=[PEAK]*3:raise ValueError('Later PLUS resume has uncorrected LR bases')
        return
    if trainer.global_step!=SOURCE_STEP or path!=Path(spec['source_state_dir']):
        raise ValueError('PLUS correction initial source differs')
    if scheduler.last_epoch!=SOURCE_STEP or scheduler.base_lrs!=[FLOOR]*3:
        raise ValueError('Original PLUS scheduler cursor/bases differ')
    groups=trainer.optimizer.param_groups
    if {g['name'] for g in groups}!={'world','action_backbone','action_io'}:raise ValueError('PLUS optimizer groups differ')
    old=[g['lr'] for g in groups]
    scheduler.base_lrs=[PEAK]*3
    new=[PEAK*f(SOURCE_STEP) for f in scheduler.lr_lambdas]
    if any(abs(v-lr_at(SOURCE_STEP))>1e-12 for v in new):raise ValueError('New PLUS scheduler curve differs')
    for group,lr in zip(groups,new,strict=True):
        group['initial_lr']=PEAK;group['lr']=lr
    scheduler._last_lr=new
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        (Path(trainer.output_dir)/'lr_correction_applied.json').write_text(json.dumps(dict(
            **spec,old_lrs=old,new_lrs=new,optimizer_state='preserved',sampler_rng_step='preserved',
            max_steps=30000,warmup_steps=1000,peak=PEAK,floor=FLOOR),indent=2)+'\n')
