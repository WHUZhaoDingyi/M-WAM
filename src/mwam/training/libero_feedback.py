"""LIBERO source-bound replay: legacy 80/20 or explicit scene-diverse 50/50.

No LR/optimizer/budget changes. The base padded epoch and cursor stay intact.
Only training demonstration addresses change; rollout trajectories are not labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .feedback_continuation import FeedbackSampler
from mwam.utils.logging_config import get_logger

from mwam.utils.paths import asset_root

ROOT = asset_root()
RUN = ROOT/'runs/libero_ac56_b24_lr2_30k_t20260912/formal_lr2'
MANIFEST = ROOT/'runs/libero_ac_adapt_t20260912/assets_v3/episodes.json'
FEEDBACK = ROOT/'runs/libero_periodic_metrics130_20260913/feedback'
CONFIG_SCHEMA = 'fastwam_libero_task_replay_config_v1'
STAGE_CONFIG_SCHEMA = 'fastwam_libero_task_replay_fresh_stage_v2'
STATE_SCHEMA = 'fastwam_libero_task_replay_state_v1'
FEEDBACK_SCHEMA = 'fastwam_libero_task_replay_feedback_v1'
logger = get_logger(__name__)

def read(path):
    return json.loads(Path(path).read_text())

def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    pending.replace(path)

@lru_cache(maxsize=8)
def _config(path):
    spec = read(path)
    from .plus_task_replay import SCHEMA as PLUS_SCHEMA, SCHEMA20 as PLUS_SCHEMA20, validate_config
    if spec.get('schema_version') in (PLUS_SCHEMA, PLUS_SCHEMA20):
        return validate_config(spec)
    if spec.get('schema_version') == STAGE_CONFIG_SCHEMA:
        required = {'schema_version','initial_source_state_dir','initial_source_step',
                    'feedback_dir','manifest_path','manifest_sha256','training_run_dir',
                    'source_step_offset','bootstrap_feedback_path','bootstrap_sha256'}
        if set(spec) != required or spec['initial_source_step'] not in (0, 30000) or spec['source_step_offset'] != 30000:
            raise ValueError('Invalid LIBERO fresh-stage replay contract')
        if Path(spec['initial_source_state_dir']) != RUN/'checkpoints/state/step_030000':
            raise ValueError('Fresh replay must continue the original 30K source')
        manifest_path = Path(spec['manifest_path'])
        if not manifest_path.is_absolute():
            raise ValueError('Fresh replay manifest path must be absolute')
        manifest = read(manifest_path)
        if manifest.get('schema_version') != 'libero_ac_episode_split_v1':
            raise ValueError('Fresh replay manifest is not a LIBERO AC episode split')
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != spec['manifest_sha256']:
            raise ValueError('Fresh replay manifest digest differs')
        for key in ('feedback_dir','training_run_dir','bootstrap_feedback_path'):
            if not Path(spec[key]).is_absolute(): raise ValueError('Fresh replay paths must be absolute')
        bootstrap = Path(spec['bootstrap_feedback_path'])
        if hashlib.sha256(bootstrap.read_bytes()).hexdigest() != spec['bootstrap_sha256']:
            raise ValueError('Fresh replay bootstrap provenance differs')
        _validate_feedback(read(bootstrap), spec)
        return spec
    if set(spec) != {'schema_version','initial_source_state_dir','initial_source_step','feedback_dir','manifest_path','manifest_sha256'} or spec['schema_version'] != CONFIG_SCHEMA:
        raise ValueError('Invalid LIBERO replay config')
    if Path(spec['initial_source_state_dir']) != RUN/f"checkpoints/state/step_{spec['initial_source_step']:06d}":
        raise ValueError('Replay migration must bind the exact existing LIBERO run')
    if not 0 < spec['initial_source_step'] < 30000:
        raise ValueError('Invalid replay activation source step')
    if Path(spec['manifest_path']) != MANIFEST or Path(spec['feedback_dir']) != FEEDBACK:
        raise ValueError('LIBERO replay dataset/output identity differs')
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != spec['manifest_sha256']:
        raise ValueError('LIBERO replay manifest changed')
    return spec

def contract(cfg):
    path = cfg.get('libero_failure_replay_config')
    if path is None:
        return None
    if not Path(path).is_absolute():
        raise ValueError('LIBERO replay config path must be absolute')
    if cfg.get('feedback_continuation') is not None or cfg.get('adaptive_replay') is not None:
        raise ValueError('Cannot combine replay controllers')
    return copy.deepcopy(_config(str(path)))

def permits_migration(saved, current, spec, source_path, source_step):
    from .plus_task_replay import SCHEMA as PLUS_SCHEMA, SCHEMA20 as PLUS_SCHEMA20, permits_migration as plus_migration
    if spec is not None and spec.get('schema_version') in (PLUS_SCHEMA, PLUS_SCHEMA20):
        return plus_migration(saved, current, spec, source_path, source_step)
    if spec is not None and spec.get('schema_version') == STAGE_CONFIG_SCHEMA:
        return False  # fresh-stage initialization is not full-state migration
    if spec is None or source_path is None or source_step != spec['initial_source_step']:
        return False
    if Path(source_path).resolve() != Path(spec['initial_source_state_dir']).resolve():
        return False
    if saved.get('training', {}).get('libero_failure_replay') is not None:
        return False
    if current.get('runtime_topology', {}).get('world_size') != 56:
        return False
    if current.get('training', {}).get('max_steps') != 30000:
        return False
    expected = copy.deepcopy(saved)
    expected['training']['libero_failure_replay'] = spec
    return expected == current

class LiberoFeedbackSampler(FeedbackSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dataset.split != 'train':
            raise ValueError('Replay must sample train demonstrations only')
        self.feedback_state['schema_version'] = STATE_SCHEMA

    def _ranges(self):
        if self._task_ranges is None:
            raw_bounds = np.asarray([0, *self.dataset.end], dtype=np.int64)
            valid = self.dataset._valid_raw_indices
            bounds = raw_bounds if valid is None else np.searchsorted(valid, raw_bounds)
            if int(bounds[-1]) != len(self.dataset) or len(bounds) != len(self.dataset.records)+1:
                raise ValueError('LIBERO valid-window address ranges differ')
            grouped = {}
            for i, row in enumerate(self.dataset.records):
                if row['split'] != 'train': raise ValueError('Validation episode in replay pool')
                if bounds[i+1] > bounds[i]:
                    grouped.setdefault(row['suite']+'/'+row['task'], []).append((int(bounds[i]),int(bounds[i+1])))
            self._task_ranges = {k: np.asarray(v,dtype=np.int64) for k,v in grouped.items()}
        return self._task_ranges

    def _epoch_indices(self):
        if getattr(self, 'task_replay_spec', None):
            from .task_replay import epoch_indices as task_epoch_indices
            return task_epoch_indices(self)
        if getattr(self, 'plus_task_replay', False):
            from .plus_task_replay import epoch_indices
            return epoch_indices(self)
        policy_path=ROOT/'runs/libero_split_16_32_t20260914/replay_policy.json'
        if not policy_path.exists():return super()._epoch_indices()
        policy=read(policy_path)
        if policy.get('replay_ratio')!=.5:raise ValueError('Expected authorized 50:50 replay')
        from mwam.utils.samplers import StrictResumableEpochSampler
        base=StrictResumableEpochSampler._epoch_indices(self)
        active=self.feedback_state['active']
        if not active or not active['failed_tasks']:return base
        tasks=active['failed_tasks'];pools=self._ranges()
        if set(tasks)-set(pools):raise ValueError('Unknown replay task')
        rng=np.random.default_rng(self.seed+self.effective_epoch+active['source_step']*1000003)
        positions=np.arange(1,len(base),2)
        order=rng.permutation(len(tasks));result=np.asarray(base,dtype=np.int64)
        for i,task in enumerate(tasks):
            chosen=positions[order[np.arange(len(positions))%len(tasks)]==i]
            spans=pools[task]
            # Balance demonstrations/scenes first, then sample their valid windows.
            eps=rng.permutation(len(spans))[np.arange(len(chosen))%len(spans)]
            offsets=np.array([rng.integers(spans[e,1]-spans[e,0]) for e in eps])
            result[chosen]=spans[eps,0]+offsets
        self.feedback_state['sampling_policy']=dict(name='train50_task_balanced_scene_diverse50_v1',replay_ratio=.5,
            source_step=active['source_step'],failed_tasks=len(tasks),demonstrations=sum(len(pools[t]) for t in tasks),
            policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest())
        return result.tolist()

def _validate_feedback(row, spec):
    from .plus_task_replay import SCHEMA as PLUS_SCHEMA, SCHEMA20 as PLUS_SCHEMA20, validate_feedback
    if spec.get('schema_version') in (PLUS_SCHEMA, PLUS_SCHEMA20):
        return validate_feedback(row, spec)
    if row.get('schema_version') != FEEDBACK_SCHEMA or row.get('status') != 'PASS' or row.get('rollouts') not in (130,500):
        raise ValueError('Feedback requires a completed 130-task evaluation')
    step = row['source_step']
    if type(step) is not int or step <= 0 or step % 2000:
        raise ValueError('Feedback checkpoint is not on the 2000-step schedule')
    if row.get('manifest_sha256') != spec['manifest_sha256']:
        raise ValueError('Feedback and training dataset differ')
    tasks = row.get('failed_tasks')
    if not isinstance(tasks,list) or tasks != sorted(set(tasks)) or not all(isinstance(t,str) for t in tasks):
        raise ValueError('Noncanonical failed-task set')
    offset = spec.get('source_step_offset', 0)
    expected = RUN/f'checkpoints/weights/step_{step:06d}.pt'
    if offset and step > offset:
        expected = Path(spec['training_run_dir'])/f'checkpoints/weights/step_{step-offset:06d}.pt'
    if row.get('checkpoint_path') != str(expected):
        raise ValueError('Feedback belongs to another training run')

def initialize_stage_sampler(sampler, spec):
    if spec is None:
        return
    from .plus_task_replay import SCHEMA as PLUS_SCHEMA, SCHEMA20 as PLUS_SCHEMA20
    if spec.get('schema_version') in (PLUS_SCHEMA, PLUS_SCHEMA20):
        sampler.feedback_state['replay_schema'] = spec['schema_version']
        if spec.get('initial_source_step') != 0:
            return
        row = _latest_feedback(spec)
        if row is None:
            raise ValueError('Cannot enable PLUS step0 replay without a completed evaluation')
        label = 'PLUS step0 replay'
    elif spec.get('schema_version') == STAGE_CONFIG_SCHEMA:
        if spec.get('initial_source_step') == 0:
            row = _latest_feedback(spec) or read(spec['bootstrap_feedback_path'])
        else:
            row = read(spec['bootstrap_feedback_path'])
        _validate_feedback(row, spec)
        if spec.get('initial_source_step') != 0 and row['source_step'] > spec['source_step_offset']:
            raise ValueError('Future bootstrap feedback')
        label = 'Bootstrap replay'
    else:
        return
    missing = set(row['failed_tasks']) - set(sampler._ranges())
    if missing:
        raise ValueError(f'{label} task absent from training data: {sorted(missing)}')
    sampler.feedback_state.update(active=dict(row,activation_step=0), pending=None,
        activations=[dict(source_step=row['source_step'],activation_step=0,failed_tasks=row['failed_tasks'])])

def _candidate(spec, active_step, current_step):
    candidates = []
    for path in Path(spec['feedback_dir']).glob('step_*.json'):
        row = read(path); _validate_feedback(row,spec)
        if active_step < row['source_step'] <= current_step: candidates.append(row)
    return max(candidates,key=lambda r:r['source_step'],default=None)

def _latest_feedback(spec):
    candidates = []
    for path in Path(spec['feedback_dir']).glob('step_*.json'):
        row = read(path); _validate_feedback(row,spec); candidates.append(row)
    return max(candidates,key=lambda r:r['source_step'],default=None)

def _collective_tasks(trainer, row):
    missing = sorted(set(row['failed_tasks'])-set(trainer.train_sampler._ranges()))
    if any(trainer._all_gather_rank_objects(missing)):
        raise ValueError('Replay task absent from train split: '+repr(missing))

def _record(trainer):
    if trainer.accelerator.is_main_process:
        put(Path(trainer.output_dir)/'libero_feedback_activations.json',trainer.train_sampler.feedback_state)

def restore(trainer, payload):
    spec = contract(trainer.cfg)
    if spec is None: return
    state = payload.get('libero_failure_replay_state')
    if trainer.cfg.get('task_failure_replay_config'):
        if state is not None:
            trainer.train_sampler.feedback_state = copy.deepcopy(state)
        _record(trainer)
        return
    if state is not None:
        if state.get('schema_version') != STATE_SCHEMA:
            raise ValueError('Invalid saved LIBERO replay state')
        for row in (state.get('active'),state.get('pending')):
            if row is not None:
                _validate_feedback(row,spec); _collective_tasks(trainer,row)
                if spec.get('initial_source_step') != 0 and row['source_step'] > trainer.global_step + spec.get('source_step_offset',0): raise ValueError('Future feedback in saved state')
        trainer.train_sampler.feedback_state = copy.deepcopy(state)
    else:
        if trainer.global_step != spec['initial_source_step']:
            raise ValueError('Non-migration checkpoint lacks LIBERO replay state')
        if spec.get('initial_source_step') == 0:
            row = _latest_feedback(spec) if trainer.accelerator.is_main_process else None
        else:
            row = _candidate(spec,0,trainer.global_step) if trainer.accelerator.is_main_process else None
        row = trainer._all_gather_rank_objects(row)[0]
        if row is None: raise ValueError('Cannot enable replay without a completed evaluation')
        _collective_tasks(trainer,row)
        active = dict(row,activation_step=trainer.global_step)
        trainer.train_sampler.feedback_state.update(active=active,pending=None,activations=[dict(source_step=row['source_step'],activation_step=trainer.global_step,failed_tasks=row['failed_tasks'])])
    _record(trainer)

def boundary(trainer):
    if trainer.cfg.get('task_failure_replay_config'): return False
    spec = contract(trainer.cfg)
    if spec is None or trainer.global_step % 10: return False
    state = trainer.train_sampler.feedback_state
    active_step = (state['active'] or {}).get('source_step',0)
    payload = None
    if trainer.accelerator.is_main_process:
        try: payload = {'candidate':_candidate(spec,active_step,trainer.global_step + spec.get('source_step_offset',0))}
        except Exception as exc: payload = {'candidate':None,'error':str(exc)}
    payload = trainer._all_gather_rank_objects(payload)[0]
    if payload.get('error') and trainer.accelerator.is_main_process:
        logger.warning('LIBERO feedback not accepted: %s',payload['error'])
    row = payload['candidate']
    if row is not None and (state['pending'] is None or row['source_step'] > state['pending']['source_step']):
        state['pending'] = dict(row,activation_step=(trainer.global_step//100+1)*100)
    pending = state['pending']
    if pending is None or trainer.global_step < pending['activation_step']: return False
    _collective_tasks(trainer,pending)
    state['active']=pending;state['pending']=None
    state['activations'].append(dict(source_step=pending['source_step'],activation_step=trainer.global_step,failed_tasks=pending['failed_tasks']))
    _record(trainer)
    return True

def publish_result(result_path, *, stage_config_path=None):
    """Publish only full five-suite results, preserving official initial-state identity."""
    result_path=Path(result_path);result=read(result_path)
    if result.get('status')!='PASS' or result.get('rollouts') not in (130,500): return None
    spec = _config(str(stage_config_path)) if stage_config_path is not None else None
    if spec is not None and spec['schema_version'] != STAGE_CONFIG_SCHEMA:
        raise ValueError('Publisher requires an explicit fresh-stage contract')
    step=result['step']
    from .libero_50k_extension import evaluation_budget
    assert type(step) is int and 0 < step <= evaluation_budget(spec) and step % 2000 == 0
    source_step=step+(spec['source_step_offset'] if spec else 0)
    run=Path(spec['training_run_dir']) if spec else RUN
    output=Path(spec['feedback_dir']) if spec else FEEDBACK
    out=output/f'step_{source_step:06d}.json'
    if out.exists(): return out
    summary_path=Path(result['rollout_summary']);d=read(summary_path);folder=summary_path.parent;inventory=read(folder/'inventory.json')
    expected_path=str(run/f'checkpoints/weights/step_{step:06d}.pt')
    count=result['rollouts']
    assert d['status']=='PASS' and d['completed']==count and d['errors']==0 and d['checkpoint']['step']==step and d['checkpoint']['path']==expected_path
    assert inventory['seed']==43 and inventory['inference_steps']==5 and inventory['replan_steps']==8
    manifest=read(MANIFEST);tasks={(r['suite'],r['task']) for r in manifest['episodes'] if r['split']=='train'}
    rows=inventory['rows'];assert len(tasks)==130 and len(rows)==count and {(r['suite'],r['task']) for r in rows}==tasks
    if count==130:assert all(r['trial']==0 for r in rows)
    else:
        from mwam.training.libero_scene_rotation import bank_path
        bank=bank_path(step,run);assert bank and hashlib.sha256(bank.read_bytes()).hexdigest()==inventory['scene_bank_sha256']
        assert sorted(rows,key=lambda r:r['rollout_id'])==sorted(read(bank)['rows'],key=lambda r:r['rollout_id'])
    expected={r['rollout_id']:r for r in rows};assert len(expected)==count
    starts={};ends={}
    for p in folder.glob('episode_events_rank_*.jsonl'):
        for line in p.read_text().splitlines():
            r=json.loads(line)
            if r['event'] not in ('ROLLOUT_START','ROLLOUT_END'): raise ValueError('Unexpected rollout event')
            target=starts if r['event']=='ROLLOUT_START' else ends
            assert r['rollout_id'] not in target;target[r['rollout_id']]=r
    assert set(starts)==set(ends)==set(expected)
    for rid,r in ends.items():
        e=expected[rid];assert r['status'] in ('SUCCESS','FAIL') and type(r['success']) is bool
        assert r['checkpoint_step']==step and starts[rid]['checkpoint_path']==expected_path
        assert starts[rid]['initial_state_sha256']==e['initial_state_sha256']
        assert (r['suite'],r['task'],r['trial'])==(e['suite'],e['task'],e['trial'])
    assert sum(r['success'] for r in ends.values())==d['successes']==result['successes']
    val=read(result['validation_summary']);assert val['status']=='PASS' and val['sigma']==1 and val['samples']==130 and val['checkpoint']['step']==step
    assert val['checkpoint']['path']==expected_path
    row=dict(schema_version=FEEDBACK_SCHEMA,status='PASS',rollouts=count,source_step=source_step,checkpoint_path=expected_path,
        manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),failed_tasks=sorted({r['suite']+'/'+r['task'] for r in ends.values() if not r['success']}),
        failed_rollouts=sum(not r['success'] for r in ends.values()),metrics_result=str(result_path),metrics_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
    if spec is not None: _validate_feedback(row,spec)
    put(out,row);return out
