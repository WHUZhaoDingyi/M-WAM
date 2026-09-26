"""PLUS failure-task replay over existing train demonstrations, with versioned evidence."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'libero_plus_train_task_replay50_v1'
SCHEMA20 = 'libero_plus_train_task_replay20_v1'
RATIO_BY_SCHEMA = {SCHEMA: 0.5, SCHEMA20: 0.2}
RENDER_PROTOCOL = 'plus_egl_context_restore_v1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(spec):
    required = {'schema_version', 'initial_source_state_dir', 'initial_source_step',
                'training_run_dir', 'manifest_path', 'manifest_sha256',
                'feedback_dir', 'bank_path', 'bank_sha256', 'replay_ratio'}
    if set(spec) != required or spec['schema_version'] not in RATIO_BY_SCHEMA:
        raise ValueError('Invalid PLUS train-task replay contract')
    if spec['replay_ratio'] != RATIO_BY_SCHEMA[spec['schema_version']]:
        raise ValueError('PLUS replay ratio must match its schema')
    for key in ('initial_source_state_dir', 'training_run_dir', 'manifest_path', 'feedback_dir', 'bank_path'):
        if not Path(spec[key]).is_absolute():
            raise ValueError('PLUS replay paths must be absolute')
    step = spec['initial_source_step']
    if type(step) is not int or step < 0:
        raise ValueError('Invalid PLUS replay source step')
    if step > 0:
        expected = Path(spec['training_run_dir']) / f'checkpoints/state/step_{step:06d}'
        if Path(spec['initial_source_state_dir']).resolve() != expected.resolve():
            raise ValueError('PLUS replay must resume its exact training run')
    elif not Path(spec['initial_source_state_dir']).exists():
        raise ValueError('PLUS replay step0 source state provenance is missing')
    for prefix in ('manifest', 'bank'):
        if digest(spec[prefix + '_path']) != spec[prefix + '_sha256']:
            raise ValueError('PLUS replay asset identity changed')
    manifest = read(spec['manifest_path'])
    if manifest.get('schema_version') != 'libero_ac_episode_split_v1':
        raise ValueError('Invalid PLUS training manifest')
    if len(read(spec['bank_path'])['rows']) != 100:
        raise ValueError('Expected official PLUS fixed100 bank')
    return spec


def permits_migration(saved, current, spec, source_path, source_step):
    if source_path is None or source_step != spec['initial_source_step']:
        return False
    if spec['initial_source_step'] == 0:
        return False
    if Path(source_path).resolve() != Path(spec['initial_source_state_dir']).resolve():
        return False
    if saved.get('training', {}).get('libero_failure_replay') is not None:
        return False
    expected = copy.deepcopy(saved)
    expected['training']['libero_failure_replay'] = spec
    return expected == current


def epoch_indices(sampler):
    from mwam.utils.samplers import StrictResumableEpochSampler
    base = np.asarray(StrictResumableEpochSampler._epoch_indices(sampler), dtype=np.int64)
    active = sampler.feedback_state['active']
    if not active or not active['failed_tasks']:
        return base.tolist()
    tasks = active['failed_tasks']; pools = sampler._ranges()
    if set(tasks) - set(pools):
        raise ValueError('PLUS feedback task absent from train demonstrations')
    rng = np.random.default_rng(sampler.seed + sampler.effective_epoch + active['source_step'] * 1000003)
    schema = sampler.feedback_state.get('replay_schema', SCHEMA)
    ratio = RATIO_BY_SCHEMA.get(schema)
    if ratio is None:
        raise ValueError('Unknown PLUS replay schema: ' + str(schema))
    if ratio == 0.5:
        positions = np.arange(1, len(base), 2)
    else:
        stride = max(2, int(round(1.0 / ratio)))
        positions = np.arange(stride - 1, len(base), stride)
    chosen = rng.permutation(len(tasks))[np.arange(len(positions)) % len(tasks)]
    for i, task in enumerate(tasks):
        dst = positions[chosen == i]; spans = pools[task]
        episodes = rng.permutation(len(spans))[np.arange(len(dst)) % len(spans)]
        offsets = np.asarray([rng.integers(spans[e, 1] - spans[e, 0]) for e in episodes], dtype=np.int64)
        base[dst] = spans[episodes, 0] + offsets
    sampler.feedback_state['sampling_policy'] = dict(name=schema, replay_ratio=ratio,
        source_step=active['source_step'], failed_tasks=len(tasks),
        demonstrations=sum(len(pools[t]) for t in tasks), labels='existing_train_demonstrations')
    return base.tolist()


def validate_feedback(row, spec):
    step = row.get('source_step')
    if row.get('schema_version') != SCHEMA or row.get('status') != 'PASS' or row.get('rollouts') != 100:
        raise ValueError('PLUS replay needs a complete fixed100 evaluation')
    if type(step) is not int or step <= 0:
        raise ValueError('Invalid feedback step')
    if row.get('render_protocol') != RENDER_PROTOCOL:
        raise ValueError('Old or unverified PLUS rendering cannot supply feedback')
    for key in ('manifest_sha256', 'bank_sha256'):
        if row.get(key) != spec[key]:
            raise ValueError('Feedback asset identity differs')
    expected = str(Path(spec['training_run_dir']) / f'checkpoints/weights/step_{step:06d}.pt')
    if row.get('checkpoint_path') != expected:
        # A fresh stage may retain the feedback of its explicitly bound source
        # checkpoint. This exception does not admit feedback from other runs.
        source = Path(spec['initial_source_state_dir'])
        source_weights = source.parent.parent / 'weights' / (source.name + '.pt')
        bootstrap = (spec['initial_source_step'] == 0
                     and source.name == f'step_{step:06d}'
                     and row.get('checkpoint_path') == str(source_weights))
        if not bootstrap:
            raise ValueError('Feedback belongs to another run')
        metadata = read(source / 'checkpoint_meta.json')
        if metadata.get('global_step') != step:
            raise ValueError('Bootstrap feedback differs from its bound source state')
    tasks = row.get('failed_tasks')
    allowed = {r['suite'] + '/' + r['task'] for r in read(spec['manifest_path'])['episodes'] if r['split'] == 'train'}
    if not isinstance(tasks, list) or tasks != sorted(set(tasks)) or set(tasks) - allowed:
        raise ValueError('Feedback tasks are not canonical train task IDs')


def publish(summary_path, spec):
    """Bind new-renderer, complete rollout pairs to the matching train base tasks."""
    summary_path = Path(summary_path); folder = summary_path.parent
    summary = read(summary_path); inventory = read(folder / 'inventory.json')
    checkpoint = summary['checkpoint']; step = checkpoint['step']
    if (summary.get('status'), summary.get('completed'), summary.get('errors')) != ('PASS', 100, 0):
        raise ValueError('Incomplete PLUS rollout')
    if inventory.get('render_protocol') != RENDER_PROTOCOL or not inventory.get('renderer_sha256'):
        raise ValueError('Missing corrected-renderer identity')
    if (inventory.get('seed'), inventory.get('inference_steps'), inventory.get('replan_steps'), inventory.get('proprio_source')) != (43, 5, 8, 'official_observation'):
        raise ValueError('PLUS evaluation protocol differs')
    bank = {r['rollout_id']: r for r in read(spec['bank_path'])['rows']}
    rows = {r['rollout_id']: r for r in inventory['rows']}
    if len(rows) != 100 or set(rows) != set(bank):
        raise ValueError('PLUS evaluation does not cover the exact bank')
    starts = {}; ends = {}
    for path in folder.glob('episode_events_rank_*.jsonl'):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event['event'] not in ('ROLLOUT_START', 'ROLLOUT_END'):
                raise ValueError('Unknown rollout event')
            target = starts if event['event'] == 'ROLLOUT_START' else ends
            if event['rollout_id'] in target:
                raise ValueError('Duplicate rollout event')
            target[event['rollout_id']] = event
    if set(starts) != set(ends) or set(ends) != set(bank):
        raise ValueError('Unpaired PLUS rollout')
    failed = set()
    for rid, original in bank.items():
        row, start, end = rows[rid], starts[rid], ends[rid]
        for key in ('suite', 'task', 'base_task', 'category', 'trial', 'instruction', 'bddl_path', 'initial_state_path'):
            if row.get(key) != original.get(key) or start.get(key) != row.get(key):
                raise ValueError('PLUS scene identity differs: ' + key)
        if start['initial_state_sha256'] != row['initial_state_sha256'] or not row['initial_state_sha256']:
            raise ValueError('PLUS initial state identity differs')
        if start['checkpoint_path'] != checkpoint['path'] or start['checkpoint_step'] != step or end['checkpoint_step'] != step:
            raise ValueError('Mixed PLUS checkpoints')
        if any(end.get(k) != row[k] for k in ('suite', 'task', 'trial')):
            raise ValueError('PLUS END identity differs from its START')
        if type(end['success']) is not bool or end['status'] != ('SUCCESS' if end['success'] else 'FAIL'):
            raise ValueError('Invalid PLUS outcome or infrastructure failure')
        if not end['success']:
            failed.add(original['suite'] + '/' + original['base_task'])
    if sum(e['success'] for e in ends.values()) != summary['successes']:
        raise ValueError('PLUS success count mismatch')
    result = dict(schema_version=SCHEMA, status='PASS', rollouts=100, source_step=step,
        checkpoint_path=checkpoint['path'], manifest_sha256=spec['manifest_sha256'],
        bank_sha256=spec['bank_sha256'], render_protocol=RENDER_PROTOCOL,
        renderer_sha256=inventory['renderer_sha256'], failed_tasks=sorted(failed),
        rollout_summary=str(summary_path), rollout_summary_sha256=digest(summary_path))
    validate_feedback(result, spec)
    dest = Path(spec['feedback_dir']) / f'step_{step:06d}.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if read(dest) != result:
            raise ValueError('Do not overwrite different feedback for the same step')
    else:
        tmp = dest.with_suffix('.tmp'); tmp.write_text(json.dumps(result, indent=2)); tmp.replace(dest)
    return dest
