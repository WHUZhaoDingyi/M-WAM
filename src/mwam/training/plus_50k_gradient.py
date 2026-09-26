"""Explicit PLUS full-state extension; backward schedule is training metadata."""
import copy
import json
import math
from pathlib import Path
from mwam.models.pixeldit.libero_gradient_schedule import PLUS_SCHEDULE, validate_schedule

from mwam.utils.paths import asset_root

ROOT = asset_root()
RUN=ROOT/'runs/libero_split_16_32_t20260914/plus_formal'
SCHEMA='fastwam_plus_50k_full_state_gradient_v1'

def contract(cfg):
    raw=cfg.get('plus_50k_gradient')
    if raw is None:return None
    spec=dict(raw)
    if set(spec)!={'schema_version','source_step','source_state_dir','gradient_schedule'} or spec['schema_version']!=SCHEMA:
        raise ValueError('Invalid PLUS50K contract')
    step=spec['source_step']
    if type(step) is not int or not 1187<step<20000:raise ValueError('PLUS50K source must precede detach boundary')
    if spec['source_state_dir']!=str(RUN/f'checkpoints/state/step_{step:06d}'):
        raise ValueError('PLUS50K source must belong to the exact PLUS run')
    spec['gradient_schedule']=validate_schedule(spec['gradient_schedule'])
    if spec['gradient_schedule']!=PLUS_SCHEDULE:raise ValueError('PLUS50K gradient schedule differs')
    return spec

def allows(saved,current,cfg,source_path,source_step):
    spec=contract(cfg)
    if spec is None:return False
    if source_path is None or str(Path(source_path).resolve())!=spec['source_state_dir'] or source_step!=spec['source_step']:
        return False
    if saved['runtime_topology']['world_size']!=32 or saved['sampler']['batch_size_per_rank']!=24:return False
    expected=copy.deepcopy(saved);t=expected['training']
    if (t['max_steps'],t['warmup_ratio'],t['learning_rate'],t['cosine_min_learning_rate'],t['lr_scheduler_type'])!=(30000,1/30,1e-4,1e-6,'cosine'):
        return False
    if 'plus_50k_gradient' in t or saved['model_math']['loss']['lambda_action']!=1 or saved['model_math']['loss']['lambda_momap']!=1:
        return False
    t.update(max_steps=50000,warmup_ratio=.02,plus_50k_gradient=spec)
    return expected==current

def lr_at(step):
    if step<=1000:return 1e-6+(1e-4-1e-6)*step/1000
    return 1e-6+(1e-4-1e-6)*.5*(1+math.cos(math.pi*min((step-1000)/49000,1.)))

def apply_after_resume(trainer):
    spec=contract(trainer.cfg)
    if spec is None:return
    step=trainer.global_step;source=trainer.loaded_resume_source
    from .topology_transition import permits_source_path
    if (step<spec['source_step'] or source['source_global_step']!=step
        or (Path(source['source_path']).resolve()!=RUN/f'checkpoints/state/step_{step:06d}'
            and not permits_source_path(trainer,source['source_path']))):
        raise ValueError('PLUS50K restored source/cursor differs')
    scheduler=getattr(trainer.scheduler,'scheduler',trainer.scheduler)
    if scheduler.last_epoch!=step or scheduler.base_lrs!=[1e-4]*3:
        raise ValueError('PLUS50K restored scheduler differs')
    groups=trainer.optimizer.param_groups
    old=[g['lr'] for g in groups]
    new=[1e-4*f(step) for f in scheduler.lr_lambdas]
    if any(abs(v-lr_at(step))>1e-12 for v in new):raise ValueError('PLUS50K LR curve differs')
    if step>spec['source_step'] and any(abs(v-lr_at(step))>1e-12 for v in old):
        raise ValueError('Later PLUS50K resume lost its corrected scheduler state')
    for g,v in zip(groups,new,strict=True):g['lr']=v
    scheduler._last_lr=new
    model=trainer.accelerator.unwrap_model(trainer.model)
    if getattr(model.dit,'action_world_gradient_schedule',None) is not None:
        raise ValueError('Cannot overwrite original LIBERO gradient plan')
    model.dit.plus_gradient_schedule=dict(spec['gradient_schedule'])
    model.set_attention_runtime_step(step)
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        (Path(trainer.output_dir)/f'plus50k_resume_step_{step:06d}.json').write_text(json.dumps(dict(
            contract=spec,resumed_step=step,old_lrs=old,new_lrs=new,adam_rng_sampler='preserved',
            max_steps=50000,action_loss_weight=1,world_loss_weight=1),indent=2)+'\n')
