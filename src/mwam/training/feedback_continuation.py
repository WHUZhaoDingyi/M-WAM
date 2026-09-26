"""Source-bound 30k continuation with asynchronous task-only replay.

Keep the original padded epoch/cursor contract. Replay changes the sampled
addresses, not dataset labels, and its versioned state is saved separately.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from mwam.utils.samplers import (
    StrictResumableEpochSampler, FailureReplayInitialWindowSampler,
)

SCHEMA = "hybrid_feedback_continuation_v1"


def contract(cfg):
    value = cfg.get("feedback_continuation")
    if value is None:
        return None
    value = dict(value)
    required = {"schema_version", "branch_id", "source_state_dir", "source_step",
                "additional_steps", "warmup_steps", "feedback_dir"}
    if set(value) != required or value["schema_version"] != SCHEMA:
        raise ValueError("Invalid feedback continuation contract")
    if (value["additional_steps"], value["warmup_steps"]) != (30000, 600):
        raise ValueError("Continuation requires 30000 updates and 600 warmup")
    if value["source_step"] not in (19913, 20430):
        raise ValueError("Continuation source must be the completed AC/ACW stage")
    for key in ("source_state_dir", "feedback_dir"):
        if not Path(value[key]).is_absolute():
            raise ValueError(f"{key} must be absolute")
    return value


def lr_at(step, spec, *, peak=1e-4, floor=1e-6):
    local = max(0, int(step) - spec["source_step"])
    warmup = spec["warmup_steps"]
    if local <= warmup:
        return floor + (peak - floor) * local / warmup
    progress = min((local - warmup) / (spec["additional_steps"] - warmup), 1.)
    return floor + (peak - floor) * .5 * (1. + math.cos(math.pi * progress))


def permits_migration(saved, current, spec, source_path, source_step):
    if spec is None or source_path is None or source_step != spec["source_step"]:
        return False
    if Path(source_path).resolve() != Path(spec["source_state_dir"]).resolve():
        return False
    expected = copy.deepcopy(saved)
    if expected["training"]["max_steps"] != source_step:
        return False
    expected["training"].pop("sample_progress_reset", None)
    expected["training"].update(max_steps=source_step + 30000,
                                 feedback_continuation=spec)
    return expected == current


class FeedbackSampler(StrictResumableEpochSampler):
    """Exact original stream until activation; every fifth address is replay.

    Task is uniform, then all legal windows of that task are uniform. The
    original stream supplies the other four addresses, including failed tasks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback_state = {"schema_version": SCHEMA, "active": None,
                               "pending": None, "activations": []}
        self._task_ranges = None

    def _ranges(self):
        if self._task_ranges is None:
            ranges = tuple(self.dataset.episode_index_ranges())
            metadata = FailureReplayInitialWindowSampler._load_episode_replay_metadata(self.dataset)
            if len(ranges) != len(metadata):
                raise ValueError("Task metadata and episode ranges differ")
            grouped = {}
            for (start, end), row in zip(ranges, metadata):
                if end > start:
                    grouped.setdefault(row["task_name"], []).append((start, end))
            self._task_ranges = {k: np.asarray(v, dtype=np.int64) for k, v in grouped.items()}
        return self._task_ranges

    def _epoch_indices(self):
        base = super()._epoch_indices()
        active = self.feedback_state["active"]
        if active is None or not active["failed_tasks"]:
            return base
        tasks = active["failed_tasks"]
        pools = self._ranges()
        missing = set(tasks) - set(pools)
        if missing:
            raise ValueError(f"Failed tasks absent from train split: {sorted(missing)}")
        seed = self.seed + self.effective_epoch + int(active["source_step"]) * 1000003
        rng = np.random.default_rng(seed)
        positions = np.arange(4, len(base), 5)
        chosen = rng.integers(len(tasks), size=len(positions))
        result = np.asarray(base, dtype=np.int64)
        for i, task in enumerate(tasks):
            selected = positions[chosen == i]
            spans = pools[task]
            cumulative = np.cumsum(spans[:, 1] - spans[:, 0])
            offsets = rng.integers(int(cumulative[-1]), size=len(selected))
            episode = np.searchsorted(cumulative, offsets, side="right")
            previous = np.concatenate(([0], cumulative[:-1]))
            result[selected] = spans[episode, 0] + offsets - previous[episode]
        return result.tolist()


def restore(trainer, payload):
    spec = contract(trainer.cfg)
    if spec is None:
        return
    state = payload.get("feedback_continuation_state")
    if state is None:
        if trainer.global_step != spec["source_step"]:
            raise ValueError("Continuation checkpoint lacks replay state")
    else:
        if state.get("schema_version") != SCHEMA:
            raise ValueError("Invalid saved feedback state")
        trainer.train_sampler.feedback_state = copy.deepcopy(state)
    # Accelerator restored Adam and scheduler counters exactly. Only refresh
    # the LR at the source boundary using the new source-bound lambda.
    if trainer.global_step == spec["source_step"]:
        scheduler = getattr(trainer.scheduler, "scheduler", trainer.scheduler)
        values = [lr_at(trainer.global_step, spec) for _ in trainer.optimizer.param_groups]
        for group, lr in zip(trainer.optimizer.param_groups, values):
            group["lr"] = lr
        scheduler._last_lr = values


def checkpoint_due(trainer):
    spec = contract(trainer.cfg)
    if spec is None:
        return trainer.save_every > 0 and trainer.global_step % trainer.save_every == 0
    local = trainer.global_step - spec["source_step"]
    return local > 0 and local % 500 == 0


def _candidate(spec, active_step):
    candidates = []
    for path in Path(spec["feedback_dir"]).glob("step_*.json"):
        row = json.loads(path.read_text())
        if row.get("schema_version") != SCHEMA or row.get("branch_id") != spec["branch_id"]:
            raise ValueError("Feedback artifact branch/schema mismatch")
        if row.get("rollouts") != 100 or row.get("status") != "PASS":
            continue
        step = row["source_step"]
        local = step - spec["source_step"]
        if local < 2500 or step <= active_step:
            continue
        tasks = row.get("failed_tasks")
        if not isinstance(tasks, list) or tasks != sorted(set(tasks)):
            raise ValueError("Feedback task list is not canonical")
        candidates.append(row)
    return max(candidates, key=lambda x: x["source_step"], default=None)


def boundary(trainer):
    spec = contract(trainer.cfg)
    if spec is None:
        return False
    local = trainer.global_step - spec["source_step"]
    if local < 2500 or local % 10:
        return False
    sampler = trainer.train_sampler
    state = sampler.feedback_state
    active_step = (state["active"] or {}).get("source_step", spec["source_step"])
    payload = None
    if trainer.accelerator.is_main_process:
        try:
            payload = {"candidate": _candidate(spec, active_step)}
        except Exception as exc:
            # A broken/incomplete evaluation never interrupts training.
            payload = {"candidate": None, "error": str(exc)}
    payload = trainer._all_gather_rank_objects(payload)[0]
    candidate = payload.get("candidate")
    if candidate is not None and candidate["source_step"] <= trainer.global_step:
        pending = state["pending"]
        if pending is None or candidate["source_step"] > pending["source_step"]:
            state["pending"] = dict(candidate, activation_step=spec["source_step"] + (local // 100 + 1) * 100)
    pending = state["pending"]
    if pending is None or trainer.global_step < pending["activation_step"]:
        return False
    # Every rank verifies the same task set before switching its iterator.
    missing = sorted(set(pending["failed_tasks"]) - set(sampler._ranges()))
    errors = trainer._all_gather_rank_objects(missing)
    if any(errors):
        raise ValueError(f"Feedback tasks missing on ranks: {errors}")
    state["active"] = pending
    state["pending"] = None
    state["activations"].append({"source_step": pending["source_step"],
                                 "activation_step": trainer.global_step,
                                 "failed_tasks": pending["failed_tasks"]})
    if trainer.accelerator.is_main_process:
        output = Path(trainer.cfg.output_dir) / "feedback_activations.json"
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output)
    return True


def validate_task(root, task, profile=None):
    """Keep the existing group controller; authorize only the bound new stage."""
    from mwam.utils.state_directory_manifest import audit_state_directory
    spec = contract(task)
    if spec is None:
        raise ValueError("Missing continuation contract")
    required = dict(batch_size=24, gradient_accumulation_steps=1,
                    train_sampling_mode="all_windows_per_epoch", learning_rate=1e-4,
                    cosine_min_learning_rate=1e-6, warmup_ratio=0.,
                    lr_scheduler_type="cosine", mixed_precision="bf16",
                    training_stage="joint", strict_resumable_sampler=True,
                    weights_only_warmstart=None, sample_progress_reset=None,
                    adaptive_replay=None, seed=20260910)
    for key, value in required.items():
        if task.get(key) != value:
            raise ValueError(f"Continuation contract differs at {key}")
    if task["max_steps"] != spec["source_step"] + 30000:
        raise ValueError("Continuation budget differs")
    for group in task["optimizer_parameter_groups"].values():
        if (group["learning_rate"], group["cosine_min_learning_rate"], group["warmup_start_learning_rate"]) != (1e-4, 1e-6, 1e-6):
            raise ValueError("Continuation parameter-group LR differs")
    source = Path(spec["source_state_dir"])
    if not source.is_relative_to(root / "runs"):
        raise ValueError("Source outside runs")
    audit = audit_state_directory(source)
    if (audit["checkpoint_meta"]["global_step"], audit["checkpoint_meta"]["world_size"]) != (spec["source_step"], 32):
        raise ValueError("Source checkpoint does not match completed world32 stage")
    resumed = audit_state_directory(Path(task["resume"]))
    if resumed["checkpoint_meta"]["world_size"] != 32:
        raise ValueError("Resume world size differs")
    if Path(task["resume"]).resolve() != source.resolve():
        if resumed["trainer_state"]["resume_contract"]["training"].get("feedback_continuation") != spec:
            raise ValueError("Resumed continuation identity differs")
    if profile is not None:
        if (profile["world_size"], profile["nproc_per_node"], profile["batch_size_per_rank"], profile["gradient_accumulation_steps"], profile["effective_global_batch_size"]) != (32,8,24,1,768):
            raise ValueError("Continuation topology/batch differs")
        if profile["stop_after_step"] != task["stop_after_step"]:
            raise ValueError("Stop boundary differs")
    return audit


def publish_feedback(branch, result_path):
    """Export only the certified completed rollout cells, never error attempts."""
    if not branch.get("feedback_dir"):
        return
    result_path = Path(result_path)
    result = json.loads(result_path.read_text())
    if result.get("status") != "PASS" or result.get("rollouts") != 100:
        return
    source = result["source_checkpoint"]
    output = Path(branch["feedback_dir"]) / f'step_{source["global_step"]:06d}.json'
    if output.exists():
        return
    summary = json.loads(Path(result["rollout_summary"]).read_text())
    if summary["source_checkpoint"] != source or summary["coverage"]["unique_cells"] != 100:
        raise ValueError("Feedback rollout coverage or checkpoint differs")
    ends = {}
    for receipt in summary["campaign_receipts"]:
        path = Path(receipt["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["file_sha256"]:
            raise ValueError("Feedback campaign receipt changed")
        cells = json.loads(path.read_text())["cells"]
        for cell in cells.values():
            rows = [json.loads(line) for line in Path(cell["events"]).read_text().splitlines() if line.strip()]
            starts = [r for r in rows if r.get("event") == "ROLLOUT_START"]
            terminal = [r for r in rows if r.get("event") == "ROLLOUT_END"]
            if len(starts) != 1 or len(terminal) != 1 or starts[0]["rollout_id"] != terminal[0]["rollout_id"]:
                raise ValueError("Feedback requires one matched START/END per cell")
            row = terminal[0]
            if row["checkpoint_setting"] != source["weights"]["path"] or type(row["success"]) is not bool:
                raise ValueError("Feedback event identity/label differs")
            if row["rollout_id"] in ends:
                raise ValueError("Duplicate feedback rollout")
            ends[row["rollout_id"]] = row
    if len(ends) != 100 or sum(r["success"] for r in ends.values()) != summary["total_successes"]:
        raise ValueError("Feedback event counts differ from certified result")
    payload = dict(schema_version=SCHEMA, branch_id=branch["id"], source_step=source["global_step"],
                   status="PASS", rollouts=100,
                   failed_tasks=sorted({r["task_name"] for r in ends.values() if not r["success"]}),
                   source_checkpoint=source, metrics_result=str(result_path),
                   validation_summary=result["validation_summary"],
                   failed_rollouts=sum(not r["success"] for r in ends.values()))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
