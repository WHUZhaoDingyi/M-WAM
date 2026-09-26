"""Source-bound 70/30 task replay; no rollout actions are training labels."""
import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import numpy as np

SCHEMA = 'fastwam_task_replay73_v1'


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n'); tmp.replace(path)


@lru_cache(maxsize=16)
def _spec(path):
    spec = read(path)
    if spec.get('schema_version') != SCHEMA or spec.get('replay_ratio') != .3:
        raise ValueError('Expected explicit task replay 70/30 contract')
    if spec['method'] not in ('ac', 'plus', 'libero') or spec['interval'] not in (500, 1000):
        raise ValueError('Unsupported method/cadence')
    if hashlib.sha256(Path(spec['manifest']).read_bytes()).hexdigest() != spec['manifest_sha256']:
        raise ValueError('Training manifest identity changed')
    return spec


def contract(cfg):
    path = cfg.get('task_failure_replay_config')
    return copy.deepcopy(_spec(str(path))) if path else None


def permits_migration(saved, current, spec, source_path, source_step):
    # Isolate feedback transport only, bound to an exact full-state checkpoint.
    previous = saved.get('training', {}).get('task_failure_replay')
    relocation = (spec or {}).get('feedback_relocation')
    if previous and relocation and source_path:
        if (source_step != relocation.get('source_step') or
            Path(source_path).resolve() != Path(relocation['source_state_dir']).resolve() or
            previous.get('feedback_dir') != relocation.get('previous_feedback_dir') or
            previous.get('feedback_dir') == spec.get('feedback_dir')):
            return False
        expected_spec = copy.deepcopy(previous)
        expected_spec.update(feedback_dir=spec['feedback_dir'], feedback_relocation=relocation)
        expected = copy.deepcopy(saved)
        expected['training']['task_failure_replay'] = expected_spec
        return expected_spec == spec and expected == current
    if not spec or not source_path or source_step != spec['source_step']:
        return False
    if Path(source_path).resolve() != Path(spec['source_state_dir']).resolve():
        return False
    if 'task_failure_replay' in saved['training']:
        return False
    expected = copy.deepcopy(saved)
    expected['training']['task_failure_replay'] = spec
    return expected == current


def validate_feedback(row, spec, step):
    if row.get('schema_version') != SCHEMA or row.get('method') != spec['method']:
        raise ValueError('Wrong task feedback schema/method')
    if row.get('manifest_sha256') != spec['manifest_sha256']:
        raise ValueError('Feedback uses a different training pool')
    if row.get('training_run_dir') != spec['training_run_dir']:
        raise ValueError('Feedback is bound to another training run')
    if row['source_step'] > step or row.get('errors', 0):
        raise ValueError('Future or infrastructure-error feedback')
    tasks = row['failed_tasks']
    if tasks != sorted(set(tasks)) or not all(isinstance(t, str) for t in tasks):
        raise ValueError('Failed tasks must be canonical unique strings')
    if not set(tasks).issubset(spec['task_names']):
        raise ValueError('Feedback task absent from training pool')


def initialize(trainer):
    spec = contract(trainer.cfg)
    if not spec: return
    sampler = trainer.train_sampler
    sampler.task_replay_spec = spec
    sampler.task_replay_output_dir = str(trainer.output_dir)
    row = read(spec['bootstrap_feedback'])
    validate_feedback(row, spec, spec['source_step'])
    sampler.task_replay_state = dict(schema_version=SCHEMA, active=row, activations=[], sampling_policy=None)


def record(trainer):
    if trainer.accelerator.is_main_process:
        put(Path(trainer.output_dir)/'task_replay_state.json', trainer.train_sampler.task_replay_state)


def restore(trainer, payload):
    spec = contract(trainer.cfg)
    if not spec: return
    state = payload.get('task_replay_state')
    if state:
        validate_feedback(state['active'], spec, trainer.global_step)
        trainer.train_sampler.task_replay_state = copy.deepcopy(state)
    else:
        if trainer.global_step != spec['source_step']:
            raise ValueError('Missing task replay state outside the authorized migration')
        trainer.train_sampler.task_replay_state['activations'].append(dict(
            activation_step=trainer.global_step, revision=0, source_step=spec['source_step'], kind='bootstrap'))
    record(trainer)


def sample_addresses(counts, task_names, failed_tasks, size, seed):
    """Uniform full-pool windows; replay balances tasks, then scenes, then windows."""
    counts = np.asarray(counts, dtype=np.int64)
    if np.any(counts <= 0): raise ValueError('Replay domains must have positive windows')
    rng = np.random.default_rng(seed)
    ends = np.cumsum(counts); starts = ends-counts
    flat = rng.integers(0, int(ends[-1]), size=size)
    episodes = np.searchsorted(ends, flat, side='right'); windows = flat-starts[episodes]
    # Three out of ten stream addresses; max rounding error is one address.
    pos = np.flatnonzero((np.arange(1, size+1)*3//10) > (np.arange(size)*3//10))
    groups = {t: np.flatnonzero(np.asarray(task_names) == t) for t in failed_tasks}
    if any(not len(v) for v in groups.values()): raise ValueError('Feedback task absent from training pool')
    if failed_tasks:
        order = rng.permutation(len(failed_tasks))[np.arange(len(pos)) % len(failed_tasks)]
        for i, t in enumerate(failed_tasks):
            dst = pos[order == i]; spans = groups[t]
            chosen = spans[rng.permutation(len(spans))[np.arange(len(dst)) % len(spans)]]
            episodes[dst] = chosen; windows[dst] = rng.integers(0, counts[chosen])
    return episodes, windows, len(pos) if failed_tasks else 0


def epoch_indices(sampler):
    spec = sampler.task_replay_spec; state = sampler.task_replay_state; active = state['active']
    if spec['method'] == 'ac':
        from .scene_replay import load_pool
        rows = load_pool(spec['manifest'])[0]['episodes']
        counts = [r['windows'] for r in rows]; names = [r['task'] for r in rows]; bounds = None
    else:
        ds = sampler.dataset
        raw = np.asarray([0, *ds.end], dtype=np.int64)
        bounds = raw if ds._valid_raw_indices is None else np.searchsorted(ds._valid_raw_indices, raw)
        # Empty episodes are removed only from this sampling domain, not Dataset identity.
        valid = np.flatnonzero(np.diff(bounds) > 0)
        counts = np.diff(bounds)[valid]; names = [ds.records[i]['suite']+'/'+ds.records[i]['task'] for i in valid]
        if any(ds.records[i]['split'] != 'train' or not ds.records[i].get('evaluation_scene') for i in valid):
            raise ValueError('Task73 PLUS requires generated training-scene records only')
        bounds = bounds[valid]
    ep, windows, replay_n = sample_addresses(counts, names, active['failed_tasks'], sampler.total_size,
        sampler.seed + sampler.effective_epoch + active['revision']*1000003)
    policy = dict(requested_replay_ratio=.3, actual_replay_ratio=replay_n/sampler.total_size,
                  full_pool_addresses=sampler.total_size-replay_n, replay_addresses=replay_n,
                  failed_tasks=len(active['failed_tasks']), feedback_revision=active['revision'],
                  source_step=active['source_step'], labels='existing_generated_expert_demonstrations')
    state['sampling_policy'] = policy
    if int(os.environ.get('RANK', '0')) == 0:
        put(Path(sampler.task_replay_output_dir)/'task_replay_sampling.json', policy)
    if spec['method'] == 'ac':
        return [('scene_replay',spec['manifest'],int(e),int(w)) for e,w in zip(ep,windows)]
    return (bounds[ep]+windows).tolist()


def boundary(trainer):
    spec = contract(trainer.cfg)
    if not spec or trainer.global_step % 10: return False
    state = trainer.train_sampler.task_replay_state; message = None
    if trainer.accelerator.is_main_process:
        try:
            path = Path(spec['feedback_dir'])/'latest.json'
            row = read(path) if path.exists() else None
            if row:
                validate_feedback(row, spec, trainer.global_step)
                if row['revision'] <= state['active']['revision']: row = None
            message = dict(candidate=row)
        except Exception as exc: message = dict(candidate=None,error=str(exc))
    message = trainer._all_gather_rank_objects(message)[0]
    if message.get('error'):
        if trainer.accelerator.is_main_process:
            put(Path(trainer.output_dir)/'task_replay_rejection.json', dict(step=trainer.global_step,**message))
        return False
    row = message['candidate']
    if not row: return False
    state['active'] = row
    state['activations'].append(dict(activation_step=trainer.global_step, revision=row['revision'],
        source_step=row['source_step'], bank_index=row.get('bank_index'), failed_tasks=row['failed_tasks']))
    record(trainer)
    return True


def checkpoint_due(trainer):
    spec = contract(trainer.cfg)
    return bool(spec and trainer.global_step > spec['source_step'] and
                (trainer.global_step-spec['source_step']) % spec['interval'] == 0)
