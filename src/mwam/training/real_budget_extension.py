"""Source-bound real-data budget extension, preserving full training state."""
import copy
import json
from pathlib import Path
from .libero_transfer_extension import lr_at


def contract(cfg):
    raw = cfg.get('real_budget_extension')
    if raw is None:
        return None
    spec = dict(raw)
    if set(spec) != {'schema_version','source_state_dir','source_step','target_step','rewarm_steps','peak_lr','floor_lr'}:
        raise ValueError('Unexpected real extension fields')
    if (spec['schema_version'],spec['source_step'],spec['target_step'],spec['rewarm_steps'],spec['peak_lr'],spec['floor_lr']) != ('real_2000_to_50000_v1',2000,50000,100,1e-4,1e-6):
        raise ValueError('Unexpected real extension schedule')
    source = Path(spec['source_state_dir'])
    if not source.is_absolute() or source.name != 'step_002000':
        raise ValueError('Extension must bind absolute step2000 full state')
    if cfg.get('max_steps') != 50000:
        raise ValueError('Extension requires cumulative50000')
    return spec


def allows(saved,current,cfg,source_path,source_step):
    spec = contract(cfg)
    if spec is None or source_path is None:
        return False
    if str(Path(source_path).resolve()) != spec['source_state_dir'] or source_step != 2000:
        return False
    if saved['runtime_topology']['world_size'] != 16 or saved['sampler']['batch_size_per_rank'] != 4:
        return False
    if saved['data_math'].get('schema') != 'real_three_view_training_v1':
        return False
    old=saved['training']
    if (old['max_steps'],old['warmup_ratio'],old['learning_rate'],old['cosine_min_learning_rate'],old['lr_scheduler_type']) != (2000,.01,1e-4,1e-6,'cosine'):
        return False
    expected=copy.deepcopy(saved)
    expected['training'].update(max_steps=50000,real_budget_extension=spec)
    return expected == current


def apply_after_resume(trainer):
    spec=contract(trainer.cfg)
    if spec is None:
        return
    step=trainer.global_step
    if step < spec['source_step'] or trainer.loaded_resume_source['source_global_step'] != step:
        raise ValueError('Restored extension cursor differs')
    scheduler=getattr(trainer.scheduler,'scheduler',trainer.scheduler)
    if scheduler.last_epoch != step or scheduler.base_lrs != [spec['peak_lr']]*3:
        raise ValueError('Restored extension scheduler differs')
    old=[g['lr'] for g in trainer.optimizer.param_groups]
    new=[base*f(step) for base,f in zip(scheduler.base_lrs,scheduler.lr_lambdas)]
    if any(abs(v-lr_at(step,spec))>1e-12 for v in old+new):
        raise ValueError('Restored LR is not continuous with extension')
    scheduler._last_lr=new
    for g,v in zip(trainer.optimizer.param_groups,new):
        g['lr']=v
    if trainer.accelerator.is_main_process:
        (Path(trainer.output_dir)/f'extension_resume_{step:06d}.json').write_text(json.dumps(dict(contract=spec,source=trainer.loaded_resume_source,step=step,old_lrs=old,new_lrs=new,preserved=['model','Adam','RNG','sampler','global_step']),indent=2))
