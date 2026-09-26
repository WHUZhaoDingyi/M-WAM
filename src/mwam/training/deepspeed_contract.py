"""Correctness checks for an effective DeepSpeed training engine."""

from __future__ import annotations

from typing import Any


MOMAP_PRODUCTION_ACCELERATE_CONFIG = (
    "scripts/accelerate_configs/accelerate_zero2_momap_production.yaml"
)


def is_momap_production_accelerate_config(value: object) -> bool:
    """Return whether an explicit config path selects MoMap production."""

    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.strip().replace("\\", "/")
    return (
        normalized == MOMAP_PRODUCTION_ACCELERATE_CONFIG
        or normalized.endswith(
            f"/{MOMAP_PRODUCTION_ACCELERATE_CONFIG}"
        )
    )


def assert_prepared_deepspeed_contract(
    trainer: Any,
    *,
    require_momap_production_zero2: bool = False,
) -> None:
    """Validate effective engine values needed by the selected profile.

    Generic FastWAM binds batch semantics only. The explicit MoMap production
    Accelerate config additionally binds clipping, ZeRO-2, and BF16 overflow.
    """

    trainer.effective_deepspeed_contract = None
    runtime_types = {
        str(record.get("distributed_type", "")).upper()
        for record in trainer.runtime_execution["ranks"]
    }
    if runtime_types != {"DEEPSPEED"}:
        if require_momap_production_zero2:
            raise RuntimeError(
                "MoMap production config requires a DeepSpeed runtime"
            )
        return

    engine = trainer.model
    required_callables = (
        "gradient_clipping",
        "gradient_accumulation_steps",
        "train_micro_batch_size_per_gpu",
        "zero_optimization_stage",
    )
    missing = [
        name
        for name in required_callables
        if not callable(getattr(engine, name, None))
    ]
    if missing:
        raise RuntimeError(
            "Prepared DeepSpeed engine lacks required contract getters: "
            f"{missing}"
        )
    engine_config = getattr(engine, "_config", None)
    bf16_config = getattr(engine_config, "bfloat16_config", None)
    local_contract = {
        "gradient_clipping": float(engine.gradient_clipping()),
        "gradient_accumulation_steps": int(
            engine.gradient_accumulation_steps()
        ),
        "train_micro_batch_size_per_gpu": int(
            engine.train_micro_batch_size_per_gpu()
        ),
        "zero_stage": int(engine.zero_optimization_stage()),
        "bf16_check_grad_overflow": bool(
            getattr(bf16_config, "check_grad_overflow", False)
        ),
    }
    gathered = trainer._all_gather_rank_objects(local_contract)
    if any(record != local_contract for record in gathered):
        raise RuntimeError(
            "DeepSpeed effective optimizer contract differs across ranks"
        )
    trainer.effective_deepspeed_contract = dict(local_contract)
    if local_contract["gradient_accumulation_steps"] != (
        trainer.gradient_accumulation_steps
    ):
        raise RuntimeError("DeepSpeed gradient accumulation contract drifted")
    if local_contract["train_micro_batch_size_per_gpu"] != trainer.batch_size:
        raise RuntimeError("DeepSpeed per-rank microbatch contract drifted")
    if local_contract["zero_stage"] not in {0, 1, 2, 3}:
        raise RuntimeError(
            "DeepSpeed reported an unsupported ZeRO stage: "
            f"{local_contract['zero_stage']}"
        )

    if not require_momap_production_zero2:
        return

    if abs(local_contract["gradient_clipping"] - trainer.max_grad_norm) > 1e-12:
        raise RuntimeError(
            "MoMap production gradient clipping differs from max_grad_norm: "
            f"{local_contract['gradient_clipping']} != {trainer.max_grad_norm}"
        )
    if local_contract["zero_stage"] != 2:
        raise RuntimeError(
            "MoMap production requires DeepSpeed ZeRO stage 2"
        )
    if not local_contract["bf16_check_grad_overflow"]:
        raise RuntimeError(
            "MoMap production requires BF16 gradient overflow checking"
        )
