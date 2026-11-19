"""Build the training manifests and train-only statistics from exported LIBERO episodes."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mwam.datasets.libero_momap.revised import RevisedLiberoDataset, SUITES, foreground_ids
from mwam.datasets.libero_momap.standalone import SCHEMA
from mwam.datasets.libero_momap.segmentation import CAMERA_POLICY, MANIFEST_SCHEMA, PIXEL_SPACE, inspect_cache, manifest_digest
from mwam.datasets.robotwin_momap.gemma_cache import GemmaCachedTextContextEncoder
from mwam.models.pixeldit.native.revised import LiberoMinMaxNormalization
from mwam.models.pixeldit.t2i_libero import ACTION, PROPRIO, LiberoXYZNormalization


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


class GeometrySamples(RevisedLiberoDataset):
    """Read unnormalized geometry with the exact training masks."""
    MANIFEST_SCHEMA = SCHEMA
    TRAINING_DOMAINS = ('libero',)
    DEFAULT_SOURCE_PERCENTAGES = {'base_libero': 100}
    SAMPLER_TYPE = None


def episode_inputs(index_root, item):
    geometry, proprio, segmentation = (
        (index_root / item[key]).resolve(strict=True) for key in ('geometry', 'proprio', 'segmentation')
    )
    with h5py.File(geometry, 'r') as data:
        metadata = json.loads(data['metadata/episode_json'][()])
        bddl = Path(metadata['bddl_name'])
        suite, task = bddl.parent.name, bddl.stem
        episode = metadata['source_episode']
        if suite not in SUITES or not re.fullmatch(r'demo_[0-9]+', episode) or metadata['fps'] != 20:
            raise ValueError(f'Expected an original LIBERO 20 Hz episode: {geometry}')
        actions = np.asarray(data['robot_state/action_target'], dtype=np.float32)
        frames = data['observations/agentview/rgb'].shape[0]
        if (actions.shape != (frames, 7) or not np.isfinite(actions).all()
                or not np.asarray(data['robot_state/action_target_valid']).all()):
            raise ValueError(f'Invalid OSC actions: {geometry}')
        objects = item['task_objects']
        if not isinstance(objects, list) or not objects:
            raise ValueError(f'Expected a nonempty task_objects list: {geometry}')
        foreground_ids([json.loads(entry) for entry in data['entities/registry_json'][:]], objects)
        excluded = set(metadata.get('excluded_source_indices', []))
        if not excluded.issubset(range(frames)):
            raise ValueError(f'Invalid excluded source index: {geometry}')
        sources = [frame for frame in range(frames) if frame not in excluded]
        shape = list(data['observations/agentview/rgb'].shape[1:3])
    states = np.load(proprio, mmap_mode='r', allow_pickle=False)
    if states.shape != (frames, 8) or not np.isfinite(states).all() or not sources:
        raise ValueError(f'Expected finite current proprio [T,8] and usable frames: {proprio}')
    instruction = item['instruction']
    row = dict(domain='libero', suite=suite, task=task, instruction=instruction, frames=frames,
               geometry_path=str(geometry), proprio_path=str(proprio), proprio_semantics='official_observation',
               action_count=frames, source_group=f'{task}/{episode}', source_episode=episode,
               pixel_transform='horizontal_flip', source_indices=sources, task_objects=objects)
    extrema = {kind: (values.min(0), values.max(0)) for kind, values in
               [('action', actions), ('proprio', states[sources])]}
    entry = dict(frames=frames, native_shape=shape, downstream_pixel_transform='horizontal_flip',
                 path=str(segmentation))
    return row, extrema, entry


def bind_segmentation(row, entry, directory):
    """Validate the mask source and bind a moved cache locally without copying its track raster."""
    geometry = Path(row['geometry_path'])
    source = Path(entry['path'])
    stat = geometry.stat()
    with h5py.File(source, 'r') as cache:
        provenance = str(cache.attrs['provenance_digest'])
        original = str(cache.attrs['geometry_path'])
        if original != str(geometry) or cache.attrs['source_file_mtime_ns'] != stat.st_mtime_ns:
            stamped_sha = cache.attrs.get('source_geometry_sha256')
            if stamped_sha:
                if sha256_file(geometry) != stamped_sha:
                    raise ValueError(f'Mask source SHA256 differs: {source}')
            else:
                # Historical exports identify source geometry by path, byte size and mtime.
                # Moving the root is safe when the task/demo suffix and file identity survive.
                suffix = (row['suite'], row['task'] + '_demo', row['source_episode'], 'episode.hdf5')
                same_relative = Path(original).parts[-4:] == geometry.parts[-4:] == suffix
                if (not same_relative or cache.attrs['source_file_size'] != stat.st_size
                        or cache.attrs['source_file_mtime_ns'] != stat.st_mtime_ns):
                    raise ValueError(f'Cannot establish source identity for moved masks: {source}')
            wrapper = directory / (hashlib.sha256(row['source_group'].encode()).hexdigest() + '.h5')
            with h5py.File(wrapper, 'x') as local:
                for key, value in cache.attrs.items():
                    local.attrs[key] = value
                local.attrs.update(geometry_path=str(geometry), source_file_size=stat.st_size,
                                   source_file_mtime_ns=stat.st_mtime_ns)
                for key in ('camera_name', 'source_index', 'pred'):
                    local[key] = h5py.ExternalLink(str(source), '/' + key)
            entry['path'] = str(wrapper)
    with h5py.File(entry['path'], 'r') as bound:
        inspect_cache(bound, entry, row, {'provenance_digest': provenance})
    return provenance


def feature_contract(kind, extrema, digest):
    low = np.min([pair[0] for pair in extrema], axis=0).astype(np.float64)
    high = np.max([pair[1] for pair in extrema], axis=0).astype(np.float64)
    span = high - low
    return asdict(LiberoMinMaxNormalization(
        feature_kind=kind, semantic_kind={'action': ACTION, 'proprio': PROPRIO}[kind],
        source_normalization='none', statistics_id=digest, minimum=tuple(low), maximum=tuple(high),
        mean=tuple(np.where(span < 1e-4, low, (high + low) / 2)),
        std=tuple(np.where(span < 1e-4, 1.0, span / 2))))


class Moments:
    def __init__(self):
        self.count = np.zeros(3, dtype=np.int64)
        self.total = np.zeros(3)
        self.squares = np.zeros(3)

    def add(self, data, valid):
        valid = np.broadcast_to(valid, data.shape)
        for axis in range(3):
            values = data[axis][valid[axis]].astype(np.float64)
            self.count[axis] += len(values)
            self.total[axis] += values.sum()
            self.squares[axis] += np.square(values).sum()

    def result(self):
        if np.any(self.count == 0):
            raise ValueError('No valid coordinates for geometry statistics')
        mean = self.total / self.count
        return tuple(mean), tuple(np.sqrt(np.maximum(self.squares / self.count - mean ** 2, 1e-12)))


def geometry_contract(directory, digest, budget):
    reader = GeometrySamples(directory / 'episodes.json', context_encoder=False, segmentation_mode='predicted',
                             segmentation_manifest_path=directory / 'predicted_segmentation_libero.json')
    indices = np.random.default_rng(43).choice(len(reader), min(budget, len(reader)), replace=False)
    source, head, wrist = Moments(), Moments(), Moments()
    try:
        for index in indices:
            sample = reader.load_window(int(index), include_text=False)
            source.add(sample['source_xyz_ref'].numpy(), sample['source_condition_valid'].numpy())
            flow = sample['target_flow'].reshape(4, 3, 448, 224).numpy()
            mask = sample['flow_loss_valid'].numpy()
            for target, valid in zip(flow, mask):
                head.add(target[:, :224], valid[:, :224])
                wrist.add(target[:, 224:], valid[:, 224:])
    finally:
        reader.close()
    sm, ss = source.result()
    hm, hs = head.result()
    wm, ws = wrist.result()
    geometry = LiberoXYZNormalization(
        statistics_id=digest, future_offsets=(8, 16, 24, 32),
        source_mean_xyz_meters=sm, source_std_xyz_meters=ss, head_flow_mean_xyz_meters=hm,
        head_flow_std_xyz_meters=hs, wrist_flow_mean_xyz_meters=wm, wrist_flow_std_xyz_meters=ws,
        head_height=224, full_height=448, full_width=224)
    return asdict(geometry), indices.tolist()


def prepare(index_path, final_dir, geometry_windows):
    index = json.loads(index_path.read_text())
    root = index_path.parent
    text_cache = (root / index['text_cache']).resolve(strict=True)
    encoder = GemmaCachedTextContextEncoder(text_cache)
    if final_dir.exists():
        raise FileExistsError(f'Choose a new output directory: {final_dir}')
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'.{final_dir.name}-', dir=final_dir.parent))
    try:
        masks = work / 'masks'
        masks.mkdir()
        inputs = [episode_inputs(root, item) for item in index['episodes']]
        inputs.sort(key=lambda item: (item[0]['task'], item[0]['source_episode']))
        groups = defaultdict(list)
        segmented, provenance = {}, set()
        geometry_paths = set()
        for row, _, entry in inputs:
            if row['geometry_path'] in geometry_paths or row['source_group'] in groups[row['task']]:
                raise ValueError(f"Duplicate episode: {row['source_group']}")
            geometry_paths.add(row['geometry_path'])
            groups[row['task']].append(row['source_group'])
            provenance.add(bind_segmentation(row, entry, masks))
            segmented[row['geometry_path']] = entry
        if len(provenance) != 1 or geometry_windows < 1:
            raise ValueError('Expected one segmentation pipeline and a positive statistics window budget')
        for instruction in sorted({row['instruction'] for row, _, _ in inputs}):
            text = encoder(instruction)
            if not text['mask'].any() or not text['context'][text['mask']].count_nonzero():
                raise ValueError(f'Empty text features for {instruction!r}')
        held_out = set()
        for task, episodes in groups.items():
            if len(episodes) < 2:
                raise ValueError(f'At least two episodes needed for the train/val split: {task}')
            ordered = sorted(episodes, key=lambda key: hashlib.sha256(f'43:{key}'.encode()).digest())
            held_out.update(ordered[:max(1, round(0.1 * len(episodes)))])
        for row, _, _ in inputs:
            row['split'] = 'val' if row['source_group'] in held_out else 'train'
        manifest = dict(schema_version=SCHEMA, complete=True, episodes=[row for row, _, _ in inputs],
                        source_percentages={'base_libero': 100}, text_caches={'libero': str(text_cache)})
        segments = dict(schema_version=MANIFEST_SCHEMA, source='predicted', camera_policy=CAMERA_POLICY,
                        pixel_space=PIXEL_SPACE, causal=True, uses_simulator_segmentation=False,
                        provenance_digest=provenance.pop(), complete=True, episodes=segmented)
        digest = manifest_digest(manifest)
        write_json(work / 'episodes.json', manifest)
        seg_path = work / 'predicted_segmentation_libero.json'
        write_json(seg_path, segments)
        geometry, sampled = geometry_contract(work, digest, geometry_windows)
        train = [extrema for row, extrema, _ in inputs if row['split'] == 'train']
        stats = dict(schema_version=SCHEMA, complete=True, split_digest=digest, geometry=geometry,
                     geometry_statistics={'seed': 43, 'sampled_indices': sampled})
        for kind in ('action', 'proprio'):
            stats[kind] = feature_contract(kind, [values[kind] for values in train], digest)
        write_json(work / 'normalization_full_features.json', stats)
        for entry in segmented.values():
            path = Path(entry['path'])
            if path.parent == masks:
                entry['path'] = str(final_dir / 'masks' / path.name)
        write_json(seg_path, segments)
        os.rename(work, final_dir)
        print(f'{final_dir}: {len(train)} train episodes, {len(held_out)} validation episodes')
    finally:
        if work.exists():
            shutil.rmtree(work)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, type=Path, help='JSON index of exported episodes')
    parser.add_argument('--out', required=True, type=Path, help='New directory for training inputs')
    parser.add_argument('--geometry-windows', default=512, type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare(args.index.resolve(strict=True), args.out.resolve(), args.geometry_windows)
