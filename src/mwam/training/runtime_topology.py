"""Runtime topology discovery and validation outside the Trainer core."""

from __future__ import annotations

import socket
from typing import Any, TypeVar

import torch


RUNTIME_TOPOLOGY_SCHEMA_VERSION = "fastwam_runtime_topology_v1"
RUNTIME_EXECUTION_SCHEMA_VERSION = (
    "fastwam_distributed_runtime_evidence_v1"
)
_T = TypeVar("_T")


def all_gather_rank_objects(trainer: Any, value: _T) -> list[_T]:
    world_size = int(trainer.accelerator.num_processes)
    process_index = int(trainer.accelerator.process_index)
    if world_size <= 0:
        raise RuntimeError("Accelerate reported a non-positive world size")
    if process_index < 0 or process_index >= world_size:
        raise RuntimeError(
            "Accelerate process index is outside the reported world size"
        )
    if world_size == 1:
        if process_index != 0:
            raise RuntimeError("Single-rank runtime must use process_index=0")
        return [value]
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        raise RuntimeError(
            "Distributed runtime topology requires an initialized "
            "torch.distributed process group"
        )
    if int(torch.distributed.get_world_size()) != world_size:
        raise RuntimeError(
            "torch.distributed world size differs from Accelerate world size"
        )
    if int(torch.distributed.get_rank()) != process_index:
        raise RuntimeError(
            "torch.distributed rank differs from Accelerate process index"
        )
    gathered: list[_T] = [None] * world_size  # type: ignore[list-item]
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def validated_runtime_topology(topology: object) -> dict[str, Any]:
    if not isinstance(topology, dict) or set(topology) != {
        "schema_version",
        "world_size",
        "node_count",
        "ranks_per_node",
    }:
        raise ValueError("runtime topology has an invalid structure")
    if topology.get("schema_version") != RUNTIME_TOPOLOGY_SCHEMA_VERSION:
        raise ValueError("runtime topology has an unsupported schema version")
    world_size = topology.get("world_size")
    node_count = topology.get("node_count")
    ranks_per_node = topology.get("ranks_per_node")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise ValueError("runtime topology world_size must be positive")
    if (
        isinstance(node_count, bool)
        or not isinstance(node_count, int)
        or node_count <= 0
    ):
        raise ValueError("runtime topology node_count must be positive")
    if (
        not isinstance(ranks_per_node, list)
        or len(ranks_per_node) != node_count
        or any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for count in ranks_per_node
        )
        or sum(ranks_per_node) != world_size
    ):
        raise ValueError(
            "runtime topology ranks_per_node must be positive and sum to "
            "world_size"
        )
    return {
        "schema_version": RUNTIME_TOPOLOGY_SCHEMA_VERSION,
        "world_size": int(world_size),
        "node_count": int(node_count),
        "ranks_per_node": [int(count) for count in ranks_per_node],
    }


def discover_runtime_topology(trainer: Any) -> dict[str, Any]:
    local_hostname = socket.gethostname()
    if not isinstance(local_hostname, str) or not local_hostname:
        raise RuntimeError("Runtime hostname must be a non-empty string")
    rank_hostnames = trainer._all_gather_rank_objects(local_hostname)
    node_index_by_hostname: dict[str, int] = {}
    ranks_per_node: list[int] = []
    for hostname in rank_hostnames:
        if not isinstance(hostname, str) or not hostname:
            raise RuntimeError(
                "Every distributed rank must report a non-empty hostname"
            )
        node_index = node_index_by_hostname.get(hostname)
        if node_index is None:
            node_index = len(ranks_per_node)
            node_index_by_hostname[hostname] = node_index
            ranks_per_node.append(0)
        ranks_per_node[node_index] += 1
    return validated_runtime_topology(
        {
            "schema_version": RUNTIME_TOPOLOGY_SCHEMA_VERSION,
            "world_size": len(rank_hostnames),
            "node_count": len(ranks_per_node),
            "ranks_per_node": ranks_per_node,
        }
    )


def discover_runtime_execution(trainer: Any) -> dict[str, Any]:
    distributed_type = getattr(
        trainer.accelerator.distributed_type,
        "value",
        str(trainer.accelerator.distributed_type),
    )
    distributed_type = str(distributed_type).upper()
    if "." in distributed_type:
        distributed_type = distributed_type.rsplit(".", 1)[-1]

    plugin = getattr(trainer.accelerator.state, "deepspeed_plugin", None)
    config = getattr(plugin, "deepspeed_config", None) if plugin else None
    zero_stage = None
    if isinstance(config, dict):
        zero = config.get("zero_optimization")
        if isinstance(zero, dict):
            zero_stage = zero.get("stage")
    if (
        isinstance(zero_stage, str)
        and zero_stage.strip().lstrip("+-").isdigit()
    ):
        zero_stage = int(zero_stage)

    device = trainer.accelerator.device
    device_type = str(device.type)
    if device.index is not None:
        device_index = int(device.index)
    elif device_type == "cuda":
        device_index = int(torch.cuda.current_device())
    else:
        device_index = -1
    device_name = (
        str(torch.cuda.get_device_name(device_index))
        if device_type == "cuda"
        else "CPU"
    )
    local_record = {
        "global_process_index": int(trainer.accelerator.process_index),
        "local_process_index": int(
            trainer.accelerator.local_process_index
        ),
        "hostname": socket.gethostname(),
        "distributed_type": distributed_type,
        "deepspeed_zero_stage": zero_stage,
        "mixed_precision": str(trainer.accelerator.mixed_precision).lower(),
        "device_type": device_type,
        "device_index": device_index,
        "device_name": device_name,
    }
    gathered_records = trainer._all_gather_rank_objects(local_record)
    for expected_rank, record in enumerate(gathered_records):
        if not isinstance(record, dict):
            raise RuntimeError(
                "Every distributed rank must report runtime execution evidence"
            )
        if record.get("global_process_index") != expected_rank:
            raise RuntimeError(
                "Runtime execution evidence is not ordered by global rank"
            )
    return {
        "schema_version": RUNTIME_EXECUTION_SCHEMA_VERSION,
        "world_size": int(trainer.accelerator.num_processes),
        "ranks": gathered_records,
    }


__all__ = [
    "RUNTIME_EXECUTION_SCHEMA_VERSION",
    "RUNTIME_TOPOLOGY_SCHEMA_VERSION",
    "all_gather_rank_objects",
    "discover_runtime_execution",
    "discover_runtime_topology",
    "validated_runtime_topology",
]
