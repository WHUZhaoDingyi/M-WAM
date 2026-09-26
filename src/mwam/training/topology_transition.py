"""Explicit 2026-09-14 ZeRO-2 32 -> 16 continuation, without resetting Adam."""
import copy
import json
from functools import lru_cache
from pathlib import Path

SCHEMA = 'fastwam_zero2_32to16_b24_full_state_v1'

def contract(cfg):
    raw = cfg.get('topology_transition')
    if raw is None:
        return None
    spec = dict(raw)
    if set(spec) != {'schema_version', 'method', 'source_state_dir', 'converted_state_dir', 'output_dir'}:
        raise ValueError('Invalid topology transition fields')
    if spec['schema_version'] != SCHEMA or spec['method'] not in ('ac', 'acw', 'plus'):
        raise ValueError('Unsupported topology transition')
    for key in ('source_state_dir', 'converted_state_dir', 'output_dir'):
        if not Path(spec[key]).is_absolute():
            raise ValueError('Transition paths must be absolute')
    if cfg.get('batch_size') != 24 or cfg.get('gradient_accumulation_steps') != 1:
        raise ValueError('Authorized transition is B24 GA1')
    return spec

@lru_cache(maxsize=8)
def source_payload(path):
    return json.loads((Path(path)/'trainer_state.json').read_text())

def mapped_sampler(saved):
    result = copy.deepcopy(saved)
    if (saved['world_size'], saved['batch_size_per_rank'], saved['gradient_accumulation_steps']) != (32, 24, 1):
        raise ValueError('Expected source B24 GA1 world32')
    batches = (saved['dataset_length'] + 383)//384
    result.update(world_size=16, effective_global_batch_size=384, global_microbatch_size=384,
                  global_batches_per_epoch=batches, optimizer_updates_per_epoch=batches,
                  padded_samples_per_epoch=batches*384)
    return result

def cursor_offset(spec):
    source = source_payload(spec['source_state_dir'])
    sampler = mapped_sampler(source['resume_contract']['sampler'])
    batch = source['batch_in_epoch']*2
    if batch > sampler['global_batches_per_epoch']:
        raise ValueError('Source cursor lies in padding removed by shrinking; explicit remap required')
    return source['epoch']*sampler['optimizer_updates_per_epoch'] + batch - source['global_step']

def transform_payload(source, spec):
    result = copy.deepcopy(source)
    rc = result['resume_contract']
    rc['sampler'] = mapped_sampler(rc['sampler'])
    rc['runtime_topology'].update(world_size=16, node_count=2, ranks_per_node=[8, 8])
    rc['training']['topology_transition'] = spec
    if spec['method'] in ('ac', 'acw'):
        scene = rc['training']['scene_replay']
        if scene['additional_steps'] != 30000 or source['global_step']-scene['source_step'] != 4000:
            raise ValueError('Expected RoboTwin stage4000 of original30K')
        scene['additional_steps'] = 50000
        rc['training']['max_steps'] = scene['source_step'] + 50000
    result['batch_in_epoch'] *= 2
    result['rank_state'] = result['rank_state'][:16]
    for row in result['rank_state']:
        row['world_size'] = 16
    # Runtime evidence remains historical and is explicitly marked as conversion provenance.
    result['topology_conversion_provenance'] = dict(spec, source_runtime_execution=result.pop('runtime_execution', None),
        rng_mapping='source ranks 0..15; topology changes stochastic trajectory',
        optimizer='repartitioned FP32 master, exp_avg, exp_avg_sq, Adam step preserved',
        cursor='same epoch and consumed source addresses; twice as many B384 microbatches')
    return result

def permits_source_path(trainer, path):
    spec = contract(trainer.cfg)
    if spec is None:
        return False
    path = Path(path).resolve()
    return path == Path(spec['converted_state_dir']) or path.parent == Path(spec['output_dir'])/'checkpoints/state'
