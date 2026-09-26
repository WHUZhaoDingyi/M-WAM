"""Opt-in 1500 -> 50000 full-state extension for the Plus-initialized LIBERO run."""
import copy
import json
import math
from pathlib import Path
from mwam.utils.paths import asset_root

RUN = asset_root() / 'runs/libero_plus_transfer'
SCHEMA = 'libero_plus_transfer_50k_v1'

def contract(cfg):
    raw = cfg.get('libero_transfer_extension')
    if raw is None: return None
    spec = dict(raw)
    expected = dict(schema_version=SCHEMA, source_step=1500,
                    source_state_dir=str(RUN/'training/checkpoints/state/step_001500'),
                    target_step=50000, rewarm_steps=50, peak_lr=1e-5, floor_lr=1e-6)
    if spec != expected: raise ValueError('Unexpected LIBERO transfer budget extension')
    if cfg.get('max_steps') != 50000: raise ValueError('Extension requires total 50000 steps')
    return spec

def lr_at(step, spec):
    start = spec['source_step']; warm = spec['rewarm_steps']
    floor, peak = spec['floor_lr'], spec['peak_lr']
    if step <= start: return floor
    if step <= start + warm: return floor + (peak-floor)*(step-start)/warm
    progress = min((step-start-warm)/(spec['target_step']-start-warm), 1.)
    return floor + (peak-floor)*.5*(1+math.cos(math.pi*progress))

def allows(saved, current, cfg, source_path, source_step):
    spec = contract(cfg)
    if spec is None or source_path is None: return False
    if str(Path(source_path).resolve()) != spec['source_state_dir'] or source_step != 1500: return False
    if saved['runtime_topology']['world_size'] != 16 or saved['sampler']['batch_size_per_rank'] != 24: return False
    old = saved['training']
    if (old['max_steps'],old['warmup_ratio'],old['learning_rate'],old['cosine_min_learning_rate'],old['lr_scheduler_type']) != (1500,50/1500,1e-5,1e-6,'cosine'): return False
    if old.get('task_failure_replay',{}).get('method') != 'libero': return False
    expected = copy.deepcopy(saved)
    expected['training'].update(max_steps=50000,warmup_ratio=50/50000,libero_transfer_extension=spec)
    return expected == current

def apply_after_resume(trainer):
    spec = contract(trainer.cfg)
    if spec is None: return
    step = trainer.global_step
    loaded = trainer.loaded_resume_source
    if step < 1500 or loaded['source_global_step'] != step: raise ValueError('Restored cursor differs')
    source = Path(loaded['source_path']).resolve()
    if source not in [RUN/run/f'checkpoints/state/step_{step:06d}' for run in ('training','canary50k')]:
        raise ValueError('Unexpected extension resume source')
    scheduler = getattr(trainer.scheduler,'scheduler',trainer.scheduler)
    if scheduler.last_epoch != step or scheduler.base_lrs != [1e-5]*3: raise ValueError('Scheduler cursor/base differs')
    old = [g['lr'] for g in trainer.optimizer.param_groups]
    expected = lr_at(step,spec)
    actual = [base*f(step) for base,f in zip(scheduler.base_lrs,scheduler.lr_lambdas)]
    if any(abs(v-expected)>1e-12 for v in actual+old): raise ValueError('Restored LR/extension curve differs')
    # Initial LR is continuous at 1500; only future schedule is extended.
    scheduler._last_lr = actual
    for g,v in zip(trainer.optimizer.param_groups,actual): g['lr'] = v
    if trainer.accelerator.is_main_process:
        (Path(trainer.output_dir)/f'extension_resume_{step:06d}.json').write_text(json.dumps(dict(
            contract=spec,source=str(source),step=step,old_lrs=old,new_lrs=actual,
            preserved=['Adam','RNG','sampler','replay','global_step']),indent=2))
