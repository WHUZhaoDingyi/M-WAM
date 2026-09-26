"""Exact LIBERO 16-GPU current-stage extension; preserve model/Adam/replay."""
import copy
import json
from pathlib import Path
from mwam.models.pixeldit.libero_gradient_schedule import validate_schedule
from .plus_50k_gradient import lr_at

from mwam.utils.paths import asset_root

ROOT = asset_root()
RUN = ROOT / 'runs/libero_split_16_32_t20260914/libero_formal'
SCHEMA = 'fastwam_libero_50k_full_state_v1'
OLD_SCHEDULE = dict(schema_version='libero_action_to_world_gradient_v1',
    detach_steps=20000, ramp_steps=333, initial_scale=0.5, final_scale=1.0,
    stage_steps=30000, source_step=30000)
SCHEDULE = dict(OLD_SCHEDULE, schema_version='libero_action_to_world_gradient_50k_v1',
    ramp_steps=1000, stage_steps=50000)

def contract(cfg):
    raw = cfg.get('libero_50k_extension')
    if raw is None:
        return None
    spec = dict(raw)
    expected = dict(schema_version=SCHEMA, source_step=5068,
        source_state_dir=str(RUN/'checkpoints/state/step_005068'),
        gradient_schedule=SCHEDULE)
    if spec != expected:
        raise ValueError('LIBERO50K requires exact 5068 source and approved schedule')
    spec['gradient_schedule'] = validate_schedule(spec['gradient_schedule'])
    return spec

def allows(saved, current, cfg, source_path, source_step):
    spec = contract(cfg)
    if spec is None or source_path is None:
        return False
    if str(Path(source_path).resolve()) != spec['source_state_dir'] or source_step != spec['source_step']:
        return False
    if saved['runtime_topology']['world_size'] != 16 or saved['sampler']['batch_size_per_rank'] != 24:
        return False
    if saved['model_math'].get('libero_action_world_gradient_schedule') != OLD_SCHEDULE:
        return False
    if saved['model_math']['loss']['lambda_action'] != 1 or saved['model_math']['loss']['lambda_momap'] != 1:
        return False
    expected = copy.deepcopy(saved)
    t = expected['training']
    if (t['max_steps'],t['warmup_ratio'],t['learning_rate'],t['cosine_min_learning_rate'],t['lr_scheduler_type']) != (30000,1/30,1e-4,1e-6,'cosine'):
        return False
    if 'libero_50k_extension' in t:
        return False
    t.update(max_steps=50000,warmup_ratio=.02,libero_50k_extension=spec)
    return expected == current

def apply_after_resume(trainer):
    spec = contract(trainer.cfg)
    if spec is None:
        return
    step = trainer.global_step
    source = trainer.loaded_resume_source
    if (step < spec['source_step'] or source['source_global_step'] != step
        or Path(source['source_path']).resolve() != RUN/f'checkpoints/state/step_{step:06d}'):
        raise ValueError('LIBERO50K restored source/cursor differs')
    scheduler = getattr(trainer.scheduler,'scheduler',trainer.scheduler)
    if scheduler.last_epoch != step or scheduler.base_lrs != [1e-4]*3:
        raise ValueError('LIBERO50K restored scheduler differs')
    groups = trainer.optimizer.param_groups
    old = [g['lr'] for g in groups]
    new = [1e-4*f(step) for f in scheduler.lr_lambdas]
    if any(abs(v-lr_at(step)) > 1e-12 for v in new):
        raise ValueError('LIBERO50K LR curve differs')
    if step > spec['source_step'] and any(abs(v-lr_at(step)) > 1e-12 for v in old):
        raise ValueError('Later LIBERO50K resume lost its scheduler state')
    model = trainer.accelerator.unwrap_model(trainer.model)
    if getattr(model.dit,'action_world_gradient_schedule',None) != OLD_SCHEDULE:
        raise ValueError('LIBERO50K cannot replace another model schedule')
    if getattr(model.dit,'plus_gradient_schedule',None) is not None:
        raise ValueError('LIBERO50K cannot modify PLUS')
    for g,v in zip(groups,new,strict=True):
        g['lr'] = v
    scheduler._last_lr = new
    # Training contract is authoritative for this opt-in backward override.
    # Preserve the original model construction contract for strict inference loading.
    model.dit.libero_extended_gradient_schedule = dict(spec['gradient_schedule'])
    model.set_attention_runtime_step(step)
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        (Path(trainer.output_dir)/f'libero50k_resume_step_{step:06d}.json').write_text(
            json.dumps(dict(contract=spec,resumed_stage_step=step,cumulative_step=30000+step,
                old_lrs=old,new_lrs=new,adam_rng_sampler_replay='preserved',
                max_steps=50000,action_own_loss_weight=1,world_loss_weight=1),indent=2)+'\n')

def evaluation_budget(spec):
    """Extend only this exact fresh-stage run, preserving other 30K campaigns."""
    if spec and Path(spec.get('training_run_dir','')) == RUN:
        continuation=RUN.parents[1]/'ddp_continue50k_t20260916/libero/formal_spec.json'
        if continuation.exists():
            ddp=json.loads(continuation.read_text())
            if ddp['source_run']!=str(RUN) or ddp['max_steps']!=ddp['source_step']+50000:
                raise ValueError('DDP evaluation continuation identity changed')
            return ddp['max_steps']
        return 50000
    return 30000
