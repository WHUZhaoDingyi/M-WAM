"""Explicit test-scene expert replay from immutable certified pools.

This is a test-scene overfitting experiment, not held-out validation training.
The ordinary train address stream and resumable cursor remain intact. Depending
on the authorized replay ratio, either odd positions or all positions are
replaced by references to independently certified expert data.
"""
from pathlib import Path
from collections import OrderedDict
import copy
import hashlib
import inspect
import json
import math
import os
import time
import numpy as np
from mwam.utils.samplers import StrictResumableEpochSampler
from mwam.datasets.robotwin_momap.collator import MoMapCollator

SCHEMA = 'robotwin_same_scene_replay50_v1'


def read(path):
    return json.loads(Path(path).read_text())


def contract(cfg):
    raw = cfg.get('scene_replay')
    if raw is None:
        return None
    spec = dict(raw)
    required = {'schema_version','source_state_dir','source_step','additional_steps','warmup_steps',
                'pool_manifest','round_dir','interval','method','replay_ratio'}
    if set(spec) != required or spec['schema_version'] != SCHEMA:
        raise ValueError('Invalid scene replay contract')
    if spec['replay_ratio'] not in (.5, 1.0) or spec['additional_steps'] not in (30000,50000) or spec['interval'] != 2000:
        raise ValueError('Scene replay requires replay_ratio in {0.5, 1.0}, 30000/50000 updates, 2000 cadence')
    if spec['method'] not in ('ac','acw') or spec['warmup_steps'] != 600:
        raise ValueError('Unknown scene replay method or LR warmup')
    if cfg.get('feedback_continuation') is not None or cfg.get('adaptive_replay') is not None:
        raise ValueError('Scene replay cannot mix with old replay policies')
    for key in ('source_state_dir','pool_manifest','round_dir'):
        if not Path(spec[key]).is_absolute():
            raise ValueError(f'{key} must be absolute')
    return spec


def load_pool(path):
    p = Path(path)
    raw = p.read_bytes()
    d = json.loads(raw)
    if d.get('schema_version') != SCHEMA or d.get('status') != 'CERTIFIED' or not d.get('episodes'):
        raise ValueError('A nonempty certified expert replay pool is required')
    for e in d['episodes']:
        if not e.get('expert_success') or not e.get('scene_identity_verified') or int(e.get('windows',0)) <= 0:
            raise ValueError('Unverified or empty expert episode in replay pool')
        if not Path(e['hdf5']).is_file():
            raise FileNotFoundError(e['hdf5'])
    return d, hashlib.sha256(raw).hexdigest()


def validate_task(root, task, profile=None):
    from mwam.utils.state_directory_manifest import audit_state_directory
    spec=contract(task)
    if spec is None:raise ValueError('Missing scene replay contract')
    for k,v in dict(batch_size=24,gradient_accumulation_steps=1,learning_rate=1e-4,
                    cosine_min_learning_rate=1e-6,mixed_precision='bf16',training_stage='joint',
                    strict_resumable_sampler=True,feedback_continuation=None,sample_progress_reset=None,
                    adaptive_replay=None,weights_only_warmstart=None).items():
        if task.get(k)!=v:raise ValueError(f'Scene replay task differs at {k}')
    if task['max_steps']!=spec['source_step']+spec['additional_steps']:raise ValueError('Scene replay budget differs')
    load_pool(spec['pool_manifest'])
    source=audit_state_directory(Path(spec['source_state_dir']))
    source_world=int(source['checkpoint_meta']['world_size'])
    if source['checkpoint_meta']['global_step']!=spec['source_step'] or source_world not in (16,32):
        raise ValueError('Source is not a bound world16/world32 state at source_step')
    resumed=audit_state_directory(Path(task['resume']))
    if int(resumed['checkpoint_meta']['world_size'])!=source_world:raise ValueError('Resume world size differs')
    if Path(task['resume']).resolve()!=Path(spec['source_state_dir']).resolve():
        if resumed['trainer_state']['resume_contract']['training'].get('scene_replay')!=spec:
            raise ValueError('Resumed scene replay contract differs')
    if profile is not None:
        expected_batch=source_world*24
        if tuple(profile[k] for k in ('world_size','batch_size_per_rank','gradient_accumulation_steps','effective_global_batch_size'))!=(source_world,24,1,expected_batch):
            raise ValueError('Scene replay batch/topology differs')
    return source


def lr_at(step, spec):
    local = max(0, int(step)-spec['source_step'])
    if spec['additional_steps'] == 50000:
        original=dict(spec,additional_steps=30000)
        if local <= 4000:
            return lr_at(step,original)
        anchor=lr_at(spec['source_step']+4000,original)
        t=min(1.,(local-4000)/46000)
        return 1e-6+(anchor-1e-6)*.5*(1+math.cos(math.pi*t))
    warmup = spec['warmup_steps']
    if local <= warmup:
        return 1e-6 + (1e-4-1e-6)*local/warmup
    t = min(1., (local-warmup)/(spec['additional_steps']-warmup))
    return 1e-6 + (1e-4-1e-6)*.5*(1+math.cos(math.pi*t))


def permits_migration(saved, current, spec, source_path, source_step):
    if spec is None or source_path is None or source_step != spec['source_step']:
        return False
    if Path(source_path).resolve() != Path(spec['source_state_dir']).resolve():
        return False
    previous_scene = saved.get('training', {}).get('scene_replay')
    if previous_scene is not None:
        expected = copy.deepcopy(saved)
        expected['training'].update(scene_replay=spec, max_steps=spec['source_step']+spec['additional_steps'])
        # Earlier scene-replay runs may carry a historical 32-to-16 topology
        # transition record. Once the bound source checkpoint is already a
        # 16-rank state, clearing that historical field does not change the
        # resumed topology or sampler stream.
        current_training = current.get('training', {})
        if expected['training'].get('topology_transition') != current_training.get('topology_transition'):
            expected['training']['topology_transition'] = current_training.get('topology_transition')
        matches = expected == current
        if not matches and os.environ.get('FASTWAM_DEBUG_SCENE_REPLAY_MIGRATION'):
            out = Path(os.environ.get('FASTWAM_DEBUG_SCENE_REPLAY_MIGRATION'))
            out.mkdir(parents=True, exist_ok=True)
            def summarize(a, b, prefix=''):
                rows = []
                if isinstance(a, dict) and isinstance(b, dict):
                    for key in sorted(set(a) | set(b)):
                        rows.extend(summarize(a.get(key, '<MISSING>'), b.get(key, '<MISSING>'), prefix + '/' + str(key)))
                elif a != b:
                    rows.append({'path': prefix, 'expected': a, 'current': b})
                return rows
            payload = {'spec': spec, 'diffs': summarize(expected, current)[:200]}
            (out / f'scene_replay_migration_mismatch_{os.getpid()}.json').write_text(json.dumps(payload, indent=2, default=str))
        return matches
    expected = copy.deepcopy(saved)
    if expected['training']['max_steps'] != source_step or 'feedback_continuation' not in expected['training']:
        return False
    expected['training'].pop('feedback_continuation')
    expected['training'].update(scene_replay=spec, max_steps=source_step+spec['additional_steps'])
    return expected == current


def coverage_addresses(window_counts, count, *, offset, seed):
    """Visit every scene once per cycle, then advance its window without replacement.

    The global ordinal includes epoch padding, so sampler resume slicing and
    rank sharding retain the exact sequence. No mutable RNG cursor is needed.
    Any N consecutive replay draws cover N scenes; N*max(windows) draws cover
    every window of every scene even when a new pool activates mid-epoch.
    """
    counts=np.asarray(window_counts,dtype=np.int64)
    if not len(counts) or np.any(counts<=0) or count<0 or offset<0:
        raise ValueError('Invalid replay coverage domain')
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(counts))
    starts=np.array([rng.integers(n) for n in counts],dtype=np.int64)
    strides=np.array([rng.integers(1,n+1) for n in counts],dtype=np.int64)
    for i,n in enumerate(counts):
        while math.gcd(int(strides[i]),int(n))!=1:strides[i]=strides[i]%n+1
    ordinals=np.arange(offset,offset+count,dtype=np.int64)
    episodes=order[ordinals%len(counts)]
    windows=(starts[episodes]+(ordinals//len(counts))*strides[episodes])%counts[episodes]
    return episodes,windows


class SceneReplaySampler(StrictResumableEpochSampler):
    def __init__(self, *args, scene_spec, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_ratio = float(scene_spec['replay_ratio'])
        if self.replay_ratio == .5 and self.batch_size % 2:
            raise ValueError('Exact per-rank 1:1 requires an even batch size')
        if self.replay_ratio not in (.5, 1.0):
            raise ValueError('Unsupported scene replay ratio')
        self.scene_spec = scene_spec
        pool, digest = load_pool(scene_spec['pool_manifest'])
        self.scene_replay_state = dict(schema_version=SCHEMA, pool_manifest=scene_spec['pool_manifest'],
            pool_sha256=digest, source_step=scene_spec['source_step'], activations=[], waiting_for=None)

    def _assign_replay_addresses(self, base, chosen, windows):
        if self.replay_ratio == 1.0:
            for j, (episode, index) in enumerate(zip(chosen, windows)):
                base[j] = ('scene_replay',self.scene_replay_state['pool_manifest'],int(episode),int(index))
            return
        for j, (episode, index) in enumerate(zip(chosen, windows)):
            base[2*j+1] = ('scene_replay',self.scene_replay_state['pool_manifest'],int(episode),int(index))

    def _epoch_indices(self):
        if getattr(self, 'task_replay_spec', None):
            from .task_replay import epoch_indices as task_epoch_indices
            return task_epoch_indices(self)
        base = super()._epoch_indices()
        state = self.scene_replay_state
        pool,digest = load_pool(state['pool_manifest'])
        if digest != state['pool_sha256']:
            raise ValueError('Replay pool changed after activation')
        rng = np.random.default_rng(self.seed+self.effective_epoch+state['source_step']*1000003)
        episodes = pool['episodes']
        policy = pool.get('sampling_policy', 'legacy_random_v1')
        replay_count = len(base) if self.replay_ratio == 1.0 else len(base)//2
        if policy in ('scene_window_round_robin_v1', 'coverage70_failure30_v1'):
            if policy == 'coverage70_failure30_v1':
                from .scene_replay_coverage import mixed_addresses
                chosen, windows = mixed_addresses(pool, replay_count,
                    offset=self.effective_epoch*replay_count,
                    seed=self.seed+state['source_step']*1000003)
            else:
                chosen, windows = coverage_addresses(
                    [int(e['windows']) for e in episodes], replay_count,
                    offset=self.effective_epoch*replay_count,
                    seed=self.seed+state['source_step']*1000003)
            self._assign_replay_addresses(base, chosen, windows)
        elif policy == 'legacy_random_v1':
            chosen = rng.integers(len(episodes),size=replay_count)
            windows = [int(rng.integers(episodes[int(episode)]['windows'])) for episode in chosen]
            self._assign_replay_addresses(base, chosen, windows)
        else:
            raise ValueError(f'Unknown replay sampling policy: {policy}')
        return base


class SceneReplayCollator(MoMapCollator):
    # Preserve true per-sample provenance instead of relabelling oracle inputs
    # as predicted Seg/noisy depth merely to satisfy a homogeneous collator.
    _CONTRACT_FIELDS=tuple(k for k in MoMapCollator._CONTRACT_FIELDS if k not in (
        'source_condition_schema_version','source_seg_encoding','source_depth_semantics'))

    def __init__(self, *args, replay_ratio=.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_ratio = float(replay_ratio)

    def __call__(self,samples):
        from mwam.models.pixeldit.kflow import SOURCE_CONDITION_PROVENANCE_KEYS,SUPPORTED_SOURCE_CONDITION_PROVENANCE
        oracle=0
        for sample in samples:
            row=sample['metadata']
            if tuple(row[k] for k in SOURCE_CONDITION_PROVENANCE_KEYS) not in SUPPORTED_SOURCE_CONDITION_PROVENANCE:
                raise ValueError('Unsupported source provenance in scene replay batch')
            oracle+=row['source_condition_schema_version']=='robotwin_momap_source_condition_v1'
        if self.replay_ratio == 1.0:
            if oracle != len(samples):
                raise ValueError('100% scene replay batch must contain only oracle expert inputs')
        elif oracle*2!=len(samples):
            raise ValueError('50% scene replay batch must contain exactly half oracle expert inputs')
        return super().__call__(samples)


class SceneReplayDataset:
    def __init__(self, base, train_cfg, *, replay_ratio=.5):
        self.base = base
        self.train_cfg = train_cfg
        self.cache = OrderedDict()
        self.pool_cache = {}
        self.collate_fn = SceneReplayCollator(validate_samples=base.collate_fn.validate_samples,
                                             replay_ratio=replay_ratio)

    def __len__(self):
        return len(self.base)

    def _cached_pool(self, pool_path):
        path = Path(pool_path)
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        cached = self.pool_cache.get(pool_path)
        if cached is not None:
            if cached[0] != identity:
                raise ValueError('Certified replay pool changed after cache initialization')
            return cached[1]
        pool, digest = load_pool(pool_path)
        after = path.stat()
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('Certified replay pool changed while loading')
        if len(self.pool_cache) >= 4:
            self.pool_cache.pop(next(iter(self.pool_cache)))
        self.pool_cache[pool_path] = (identity, pool, digest)
        return pool

    def __getitem__(self, address):
        if isinstance(address, (int,np.integer)):
            return self.base[int(address)]
        kind,pool_path,episode,index = address
        if kind != 'scene_replay':
            raise ValueError('Unknown replay address')
        key = (pool_path,episode)
        if key not in self.cache:
            from mwam.datasets.robotwin_momap.factory import build_v1_momap_components,create_v1_momap_dataset
            from mwam.datasets.robotwin_momap.text_cache import CachedTextContextEncoder
            pool = self._cached_pool(pool_path) if os.environ.get('FASTWAM_CACHE_REPLAY_POOLS','0')=='1' else load_pool(pool_path)[0]
            e = pool['episodes'][episode]
            supported = inspect.signature(build_v1_momap_components).parameters
            kwargs = {k:v for k,v in self.train_cfg.items() if k in supported}
            kwargs.update(episode_paths=[e['hdf5']], prompt_index_file=None,
                language_resolver=lambda metadata,instruction=e['instruction']: instruction,
                text_embedding_preload_fork_cow=False, text_embedding_preload_max_bytes=None,
                text_embedding_preload_max_node_bytes=None, text_embedding_preload_rank_stagger_seconds=0.,
                strict_pred_head_only=False, seg_quality_mix_manifest=None, depth_noise=None,
                source_head_seg_candidates=False, context_encoder=CachedTextContextEncoder(pool['text_cache'],context_len=128,hidden_size=4096))
            dataset = create_v1_momap_dataset(**kwargs)
            if len(dataset) != e['windows']:
                raise ValueError('Replay window count differs from certified pool')
            self.cache[key] = dataset
            if len(self.cache)>4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return prepare_scene_sample(self.cache[key][index])


def prepare_scene_sample(sample):
    # Match the evaluated oracle-source contract exactly: no wrist Seg input.
    sample['source_seg_condition'][1:]=0
    return sample


def record(trainer):
    if not trainer.accelerator.is_main_process:
        return
    p = Path(trainer.output_dir)/'scene_replay_activations.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(trainer.train_sampler.scene_replay_state,indent=2))
    tmp.replace(p)


def restore(trainer,payload):
    spec = contract(trainer.cfg)
    if spec is None:
        return
    saved = payload.get('scene_replay_state')
    if saved is None:
        if trainer.global_step != spec['source_step']:
            raise ValueError('Continuation lacks scene replay state')
        state=trainer.train_sampler.scene_replay_state
        state['activations']=[dict(source_step=spec['source_step'],activation_step=trainer.global_step,pool_sha256=state['pool_sha256'])]
        scheduler=getattr(trainer.scheduler,'scheduler',trainer.scheduler)
        values=[lr_at(trainer.global_step,spec) for _ in trainer.optimizer.param_groups]
        for group,value in zip(trainer.optimizer.param_groups,values):group['lr']=value
        scheduler._last_lr=values
    else:
        if saved.get('schema_version') != SCHEMA:
            raise ValueError('Invalid saved scene replay state')
        _,digest=load_pool(saved['pool_manifest'])
        if digest!=saved['pool_sha256']:
            raise ValueError('Saved replay manifest changed')
        trainer.train_sampler.scene_replay_state=copy.deepcopy(saved)
    record(trainer)
    if trainer.cfg.get('task_failure_replay_config'): return
    policy_path=Path(spec['round_dir']).parent.parent/'async_policy.json'
    policy=read(policy_path) if policy_path.exists() else {}
    if policy.get('enabled') and policy.get('coverage_publications'):
        # Restore is itself a safe sampler boundary, before the next batch.
        asynchronous_boundary(trainer,spec,policy,force=True)
    else:
        boundary(trainer)


def checkpoint_due(trainer):
    spec=contract(trainer.cfg)
    if spec is None:return False
    local=trainer.global_step-spec['source_step']
    return local>0 and local%500==0


def boundary(trainer):
    if trainer.cfg.get('task_failure_replay_config'): return False
    spec=contract(trainer.cfg)
    if spec is None:return False
    policy_path=Path(spec['round_dir']).parent.parent/'async_policy.json'
    if policy_path.exists() and read(policy_path).get('enabled') is True:
        return asynchronous_boundary(trainer,spec,read(policy_path))
    local=trainer.global_step-spec['source_step']
    if local<=0 or local%spec['interval'] or local>=spec['additional_steps']:return False
    state=trainer.train_sampler.scene_replay_state
    if state['source_step']==trainer.global_step:return False
    # Pause ALL ranks at the evaluated checkpoint: late evaluation may delay
    # wall-clock progress, but can never silently skip a requested replay round.
    state['waiting_for']=trainer.global_step
    record(trainer)
    result=Path(spec['round_dir'])/f"step_{trainer.global_step:06d}.json"
    while not result.exists():time.sleep(5)
    row=read(result)
    if row.get('status')!='CERTIFIED' or row.get('source_step')!=trainer.global_step:
        raise ValueError('Replay round is not certified for this checkpoint')
    _,digest=load_pool(row['pool_manifest'])
    if digest!=row['pool_sha256']:raise ValueError('Round pool hash mismatch')
    agreed=trainer._all_gather_rank_objects(digest)
    if len(set(agreed))!=1:raise ValueError('Ranks disagree on replay pool')
    state.update(pool_manifest=row['pool_manifest'],pool_sha256=digest,source_step=trainer.global_step,waiting_for=None)
    state['activations'].append(dict(source_step=trainer.global_step,activation_step=trainer.global_step,pool_sha256=digest))
    record(trainer)
    return True


def asynchronous_boundary(trainer, spec, policy, *, force=False):
    """Only rank zero chooses a certified pool; missing evaluations never wait."""
    if policy.get('schema_version') != 'robotwin_async_replay_v1':
        raise ValueError('Unknown asynchronous replay policy')
    local=trainer.global_step-spec['source_step']
    if local < 0 or (local % int(policy['pool_poll_steps']) and not force):return False
    state=trainer.train_sampler.scene_replay_state
    candidate=None
    if trainer.accelerator.is_main_process:
        if policy.get('coverage_publications'):
            from .scene_replay_coverage import publication_candidate
            candidate=publication_candidate(policy['coverage_publications'],state,
                                            trainer.global_step,spec['method'])
        else:
            for path in sorted(Path(spec['round_dir']).glob('step_*.json'),reverse=True):
                row=read(path)
                if row.get('status')=='CERTIFIED' and state['source_step'] < row['source_step'] <= trainer.global_step:
                    candidate=dict(row,result_path=str(path));break
    candidate=trainer._all_gather_rank_objects(candidate)[0]
    state['waiting_for']=None
    state['update_mode']='asynchronous'
    if candidate is None:
        record(trainer)
        return False
    pool,digest=load_pool(candidate['pool_manifest'])
    if digest!=candidate['pool_sha256']:raise ValueError('Round pool hash mismatch')
    agreed=trainer._all_gather_rank_objects(digest)
    if len(set(agreed))!=1:raise ValueError('Ranks disagree on replay pool')
    previous=dict(pool_manifest=state['pool_manifest'],pool_sha256=state['pool_sha256'],source_step=state['source_step'])
    state.update(pool_manifest=candidate['pool_manifest'],pool_sha256=digest,source_step=candidate['source_step'])
    if 'pool_revision' in candidate:
        state['pool_revision']=candidate['pool_revision']
    state['activations'].append(dict(source_step=candidate['source_step'],activation_step=trainer.global_step,
        pool_manifest=candidate['pool_manifest'],pool_sha256=digest,epoch=trainer.epoch,batch_in_epoch=trainer.batch_in_epoch,
        previous=previous,result_path=candidate['result_path'],evaluation_lag_steps=trainer.global_step-candidate['source_step']))
    if 'pool_revision' in candidate:
        state['activations'][-1]['pool_revision']=candidate['pool_revision']
    record(trainer)
    return True
