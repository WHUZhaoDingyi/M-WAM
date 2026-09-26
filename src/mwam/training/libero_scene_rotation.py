"""Freeze fixed-130 plus rotating-370 official LIBERO initial states."""
import hashlib
import json
from pathlib import Path

from mwam.utils.paths import asset_root

ROOT = asset_root()
POLICY=ROOT/'runs/libero_split_16_32_t20260914/replay_policy.json'

def read(path):return json.loads(Path(path).read_text())

def enabled(step, run):
    if not POLICY.exists():return False
    p=read(POLICY)
    return str(run)==p['training_run_dir'] and step>=p['from_step']

def selected_rows(rows, round_index, total=500):
    anchors=[r for r in rows if r['trial']==0]
    others=sorted((r for r in rows if r['trial']!=0),key=lambda r:(r['trial'],r['suite'],r['task']))
    n=total-len(anchors)
    if n<=0 or len(others)<n:raise ValueError('Insufficient distinct initial states')
    chosen=[others[(round_index*n+i)%len(others)] for i in range(n)]
    result=anchors+chosen
    if len({r['rollout_id'] for r in result})!=total:raise ValueError('Duplicate scene')
    return result

def bank_path(step, run):
    if not enabled(step,run):return None
    p=read(POLICY);out=Path(p['plan_dir'])/f'step_{step:06d}.json'
    if out.exists():return out
    inventory=read(p['inventory']);raw=Path(p['inventory']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=p['inventory_sha256']:raise ValueError('Inventory changed')
    rows=selected_rows(inventory['rows'],(step-p['from_step'])//2000)
    d=dict(schema_version='libero_fixed130_rotate370_v1',step=step,rows=rows,
           inventory_sha256=p['inventory_sha256'],anchor_count=130,rotation_count=370,
           experimental_role='evaluation_task_overfit',replay_ratio=.5)
    out.parent.mkdir(parents=True,exist_ok=True)
    # CPU scheduler has the existing leader lease; GPU workers consume its plan.
    with out.open('x') as f:json.dump(d,f,indent=2)
    return out

def prepare_inventory(output):
    import numpy as np
    from libero.libero import benchmark
    from experiments.libero.eval_libero_ac import official_initial_states
    source=ROOT/'runs/libero_split_16_32_t20260914/libero_periodic_metrics130/step_016000/rollout_130/attempt_001/inventory.json'
    rows=[];counts={}
    for old in read(source)['rows']:
        suite=benchmark.get_benchmark_dict()[old['suite']]()
        states=official_initial_states(suite,old['task_id'])
        counts[old['suite']+'/'+old['task']]=len(states)
        for trial,state in enumerate(states):
            row=dict(old,trial=trial,rollout_id=f"{old['suite']}:{old['task']}:init_{trial:03d}",
                     initial_state_sha256=hashlib.sha256(np.asarray(state,dtype=np.float64).tobytes()).hexdigest())
            rows.append(row)
    d=dict(schema_version='libero_official_all_initial_states_v1',tasks=len(counts),scene_count=len(rows),initial_states_per_task=counts,rows=rows)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(d,f,indent=2)
    return d

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare-inventory',type=Path,required=True);a=p.parse_args()
    d=prepare_inventory(a.prepare_inventory);print(json.dumps({k:v for k,v in d.items() if k!='rows'}))
