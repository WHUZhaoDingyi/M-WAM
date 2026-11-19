"""Original four-suite LIBERO with the shared revised observation/target path.

This schema and sampler scope are explicit. A legacy mixed manifest cannot be
turned into a standalone run merely by setting the Plus sampling weight to zero.
"""
import hashlib
import json
from pathlib import Path

from .revised import RevisedLiberoDataset, SUITES


SCHEMA = 'libero_four_suites_v1'


def bind_geometry_storage(records, mapping_path):
    """Bind physical mirrors while preserving the original logical identities."""
    if mapping_path is None:
        return [dict(row) for row in records], None
    document = json.loads(Path(mapping_path).read_text())
    if document.get('schema_version') != 'libero_geometry_storage_map_v1' or document.get('complete') is not True:
        raise ValueError('Complete LIBERO geometry mirror map required')
    result = []
    for original in records:
        row = dict(original)
        entry = document['entries'][row['geometry_path']]
        physical = Path(entry['path'])
        if not physical.is_absolute() or not physical.is_file():
            raise FileNotFoundError('Missing shared LIBERO geometry: ' + str(physical))
        stat = physical.stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry['size'], entry['mtime_ns']):
            raise ValueError('Shared LIBERO geometry identity differs: ' + str(physical))
        row['storage_geometry_path'] = str(physical)
        result.append(row)
    return result, hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


class StandaloneLiberoDataset(RevisedLiberoDataset):
    MANIFEST_SCHEMA = SCHEMA
    TRAINING_DOMAINS = ('libero',)
    DEFAULT_SOURCE_PERCENTAGES = {'base_libero': 100}
    # Let the Trainer select its strict, resumable all-window sampler. There
    # is no inter-domain resampling in an original-LIBERO-only run.
    SAMPLER_TYPE = None

    def __init__(self, manifest_path, normalization_artifact_path=None, split='train',
                 max_open_episodes=2, text_embedding_cache_dir=None, context_encoder=None,
                 source_percentages=None, segmentation_mode='predicted', segmentation_manifest_path=None,
                 segmentation_palette='episode', geometry_storage_map_path=None):
        manifest = json.loads(Path(manifest_path).read_text())
        if (not isinstance(manifest, dict) or manifest.get('schema_version') != SCHEMA
                or manifest.get('complete') is not True):
            raise ValueError('Complete standalone LIBERO manifest required')
        if split not in ('train', 'val'):
            raise ValueError('Standalone LIBERO split must be train or val')
        expected = self.DEFAULT_SOURCE_PERCENTAGES
        if manifest.get('source_percentages') != expected:
            raise ValueError('Standalone manifest must declare 100% base_libero')
        if source_percentages is not None and dict(source_percentages) != expected:
            raise ValueError('Standalone LIBERO requires 100% base_libero')
        rows = manifest.get('episodes')
        if not isinstance(rows, list) or not rows:
            raise ValueError('Standalone LIBERO manifest requires episodes')
        paths = set()
        group_splits = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Standalone episode records must be mappings')
            if row.get('domain') != 'libero':
                raise ValueError('Standalone LIBERO excludes Plus and other domains in every split')
            if row.get('suite') not in SUITES:
                raise ValueError('Standalone LIBERO accepts only the four declared suites')
            if row.get('split') not in ('train', 'val'):
                raise ValueError('Standalone manifest split must be train or val')
            path = row.get('geometry_path')
            if not isinstance(path, str) or not path.strip():
                raise ValueError('Standalone LIBERO requires explicit geometry paths')
            canonical = str(Path(path).expanduser().resolve())
            if canonical in paths:
                raise ValueError('Standalone LIBERO contains a duplicate geometry path')
            paths.add(canonical)
            group = row.get('source_group')
            if not isinstance(group, str) or not group.strip():
                raise ValueError('Standalone LIBERO requires a physical source_group')
            previous = group_splits.setdefault(group, row['split'])
            if previous != row['split']:
                raise ValueError('Standalone physical source_group crosses train/val splits')
            if row.get('proprio_semantics') != 'official_observation' or row.get('pixel_transform') != 'horizontal_flip':
                raise ValueError('Standalone LIBERO requires verified official proprio and base-image convention')
        if not any(row['split'] == 'train' for row in rows):
            raise ValueError('Standalone LIBERO requires real training episodes')
        if not normalization_artifact_path:
            raise ValueError('Complete standalone LIBERO normalization artifact required')
        stats = json.loads(Path(normalization_artifact_path).read_text())
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if (not isinstance(stats, dict) or stats.get('schema_version') != SCHEMA or stats.get('complete') is not True
                or stats.get('split_digest') != digest):
            raise ValueError('Complete standalone statistics must match the manifest')
        if segmentation_mode != 'predicted':
            raise ValueError('Standalone LIBERO requires predicted DINO/SAM2 segmentation')
        super().__init__(manifest_path, normalization_artifact_path, split=split,
                         max_open_episodes=max_open_episodes,
                         text_embedding_cache_dir=text_embedding_cache_dir,
                         context_encoder=context_encoder, source_percentages=source_percentages,
                         segmentation_mode=segmentation_mode,
                         segmentation_manifest_path=segmentation_manifest_path,
                         segmentation_palette=segmentation_palette)
        self.geometry_storage_digest = None
        self.geometry_storage_paths = {}
        if geometry_storage_map_path is not None:
            # Keep manifest identities and their split digest unchanged while
            # resolving the physical shared-mirror location at read time.
            bound, self.geometry_storage_digest = bind_geometry_storage(
                self.records, geometry_storage_map_path)
            self.geometry_storage_paths = {
                row['geometry_path']: row['storage_geometry_path'] for row in bound
            }

    def _open(self, path):
        return super()._open(self.geometry_storage_paths.get(path, path))
