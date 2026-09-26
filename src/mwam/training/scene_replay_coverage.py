"""Resumable 80:20 expert stream: task/config coverage and historical failures.

This operates only within the certified expert replay stream. When scene replay
uses a 50% outer ratio, it controls the expert half of each batch; when the outer
ratio is 100%, it controls the whole batch. No future targets or evaluation
labels are returned to the model; feedback controls addresses only.
"""
from collections import defaultdict
import numpy as np

POLICY = 'coverage80_failure20_v1'


def _group_addresses(entries, eligible, count, offset, seed):
    from .scene_replay import coverage_addresses
    groups = defaultdict(list)
    for i in eligible:
        groups[(entries[i]['task'], entries[i]['task_config'])].append(i)
    ordered = [groups[k] for k in sorted(groups)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ordered))
    ordinal = np.arange(offset, offset + count, dtype=np.int64)
    chosen_group = order[ordinal % len(order)]
    episodes = np.empty(count, dtype=np.int64)
    windows = np.empty(count, dtype=np.int64)
    for g, indices in enumerate(ordered):
        positions = np.flatnonzero(chosen_group == g)
        if not len(positions):
            continue
        first = int(ordinal[positions[0]] // len(order))
        e, w = coverage_addresses([entries[i]['windows'] for i in indices],
                                 len(positions), offset=first, seed=seed + 1009 * (g + 1))
        episodes[positions] = np.asarray(indices)[e]
        windows[positions] = w
    return episodes, windows


def mixed_addresses(pool, count, *, offset, seed):
    """Each five expert draws: four coverage, one historical-failure draw.

    Addressing is a pure function of saved ordinal, seed and immutable pool.
    Epoch and checkpoint splits do not reset either lane's coverage cursor.
    Empty failure pool falls back to 100% coverage; no empty/invalid samples.
    """
    entries = pool['episodes']
    if count < 0 or offset < 0 or not entries:
        raise ValueError('Invalid replay address domain')
    hard = [i for i, e in enumerate(entries) if e.get('historical_failure', False)]
    if not hard:
        return _group_addresses(entries, range(len(entries)), count, offset, seed)
    ordinal = np.arange(offset, offset + count, dtype=np.int64)
    slots = ordinal % 5
    covered = slots < 4
    episodes = np.empty(count, dtype=np.int64)
    windows = np.empty(count, dtype=np.int64)
    for is_coverage, eligible, indices, lane_seed in (
        (True, range(len(entries)), ordinal // 5 * 4 + np.minimum(slots, 4), seed),
        (False, hard, ordinal // 5 + np.maximum(slots - 4, 0), seed + 999983),
    ):
        selected = np.flatnonzero(covered == is_coverage)
        if not len(selected):
            continue
        e, w = _group_addresses(entries, eligible, len(selected),
                                int(indices[selected[0]]), lane_seed)
        episodes[selected], windows[selected] = e, w
    return episodes, windows


def publication_candidate(directory, state, step, method):
    """A publication revision is not an evaluated checkpoint number."""
    import json
    from pathlib import Path
    p = Path(directory) / method / 'latest.json'
    if not p.exists():
        return None
    row = json.loads(p.read_text())
    if (row.get('schema_version') != 'robotwin_coverage_publication_v1'
            or row.get('method') != method or row.get('status') != 'CERTIFIED'):
        raise ValueError('Invalid coverage publication')
    if row['source_step'] > step or row['source_step'] < state['source_step']:
        return None
    if row['pool_revision'] <= state.get('pool_revision', 0):
        return None
    return dict(row, result_path=str(p))
