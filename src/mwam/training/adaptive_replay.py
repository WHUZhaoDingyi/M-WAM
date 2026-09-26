"""Non-blocking, distributed-safe activation of adaptive replay manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from ..utils.logging_config import get_logger
from ..utils.samplers import (
    FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL,
    FailureReplayTaskVariantNoInitialSampler,
    canonical_task_variant_failure_manifest,
    canonical_task_variant_failure_manifest_sha256,
)
from .resume_contract import operational_sampler_resume_state


logger = get_logger(__name__)

ADAPTIVE_REPLAY_TRAINING_CONFIG_SCHEMA_VERSION = (
    "fastwam_adaptive_replay_training_config_v1"
)
ADAPTIVE_REPLAY_STATE_SCHEMA_VERSION = "fastwam_adaptive_replay_state_v1"
ADAPTIVE_REPLAY_ACTIVATION_REQUEST_SCHEMA_VERSION = (
    "fastwam_adaptive_replay_activation_request_v1"
)
ADAPTIVE_REPLAY_ACTIVATION_RECEIPT_SCHEMA_VERSION = (
    "fastwam_adaptive_replay_activation_receipt_v1"
)
ADAPTIVE_I5_EVALUATION_RECEIPT_SCHEMA_VERSION = (
    "fastwam_adaptive_i5_evaluation_receipt_v1"
)


def _safe_id(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    result = value.strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if (
        not result
        or len(result) > 160
        or not result[0].isalnum()
        or any(character not in allowed for character in result)
    ):
        raise ValueError(f"{label} must be a safe path-component identifier")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(value)


def _absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{label} must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    return path.resolve(strict=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path.read_bytes()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular_bytes(path, label=label))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{os.urandom(8).hex()}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class AdaptiveReplayTrainingConfig:
    branch_id: str
    control_dir: Path
    checkpoint_interval: int
    activation_poll_interval_steps: int
    activation_step_multiple: int
    activation_lead_steps: int
    initial_manifest_source_step: int
    launch_sidecar: bool
    evaluation_profile: Path
    controller_poll_seconds: float
    sidecar_script: Path


def validate_adaptive_replay_training_config(
    value: object,
    *,
    sampling_mode: str,
    repository_root: Path,
) -> AdaptiveReplayTrainingConfig | None:
    """Validate the opt-in training/sidecar contract without creating files."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("adaptive_replay must be a mapping or null")
    required = {
        "schema_version",
        "enabled",
        "branch_id",
        "control_dir",
        "checkpoint_interval",
        "activation_poll_interval_steps",
        "activation_step_multiple",
        "activation_lead_steps",
        "initial_manifest_source_step",
        "launch_sidecar",
        "evaluation_profile",
        "controller_poll_seconds",
    }
    optional = {"sidecar_script"}
    if not set(value).issubset(required | optional) or not required.issubset(value):
        raise ValueError(
            "adaptive_replay fields drifted: "
            f"missing={sorted(required - set(value))}, "
            f"unexpected={sorted(set(value) - required - optional)}"
        )
    if value["schema_version"] != ADAPTIVE_REPLAY_TRAINING_CONFIG_SCHEMA_VERSION:
        raise ValueError("adaptive_replay schema_version is unsupported")
    if not isinstance(value["enabled"], bool):
        raise TypeError("adaptive_replay.enabled must be boolean")
    if not value["enabled"]:
        return None
    if sampling_mode != FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL:
        raise ValueError(
            "adaptive replay requires "
            "failure_replay_80pct_task_variant_noinitial"
        )
    launch_sidecar = value["launch_sidecar"]
    if not isinstance(launch_sidecar, bool):
        raise TypeError("adaptive_replay.launch_sidecar must be boolean")
    poll_seconds = value["controller_poll_seconds"]
    if (
        isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, (int, float))
        or float(poll_seconds) < 1.0
        or float(poll_seconds) > 300.0
    ):
        raise ValueError(
            "adaptive_replay.controller_poll_seconds must lie in [1,300]"
        )
    sidecar_raw = value.get(
        "sidecar_script",
        repository_root / "tools/runtime/watch_robotwin_adaptive_replay_i5.py",
    )
    return AdaptiveReplayTrainingConfig(
        branch_id=_safe_id(value["branch_id"], label="adaptive branch_id"),
        control_dir=_absolute_path(
            value["control_dir"], label="adaptive control_dir"
        ),
        checkpoint_interval=_positive_int(
            value["checkpoint_interval"], label="adaptive checkpoint_interval"
        ),
        activation_poll_interval_steps=_positive_int(
            value["activation_poll_interval_steps"],
            label="adaptive activation_poll_interval_steps",
        ),
        activation_step_multiple=_positive_int(
            value["activation_step_multiple"],
            label="adaptive activation_step_multiple",
        ),
        activation_lead_steps=_positive_int(
            value["activation_lead_steps"],
            label="adaptive activation_lead_steps",
        ),
        initial_manifest_source_step=_nonnegative_int(
            value["initial_manifest_source_step"],
            label="adaptive initial_manifest_source_step",
        ),
        launch_sidecar=launch_sidecar,
        evaluation_profile=_absolute_path(
            value["evaluation_profile"],
            label="adaptive evaluation_profile",
        ),
        controller_poll_seconds=float(poll_seconds),
        sidecar_script=_absolute_path(
            sidecar_raw,
            label="adaptive sidecar_script",
        ),
    )


def _resume_trainer_state_path(resume: object) -> Path | None:
    if not isinstance(resume, (str, Path)) or not str(resume).strip():
        return None
    candidate = Path(resume).expanduser()
    if candidate.suffix == ".pt" or not candidate.is_dir():
        return None
    state_path = candidate / "trainer_state.json"
    return state_path if state_path.is_file() and not state_path.is_symlink() else None


def manifest_for_adaptive_resume(
    trainer: Any,
    configured_manifest: object,
) -> object:
    """Recover the active manifest before building a resumed DataLoader."""

    config = getattr(trainer, "adaptive_replay_config", None)
    if not isinstance(config, AdaptiveReplayTrainingConfig):
        return configured_manifest
    state_path = _resume_trainer_state_path(getattr(trainer, "resume", None))
    if state_path is None:
        return configured_manifest
    payload = _read_json_object(state_path, label="resume trainer state")
    contract = payload.get("resume_contract")
    if not isinstance(contract, Mapping):
        return configured_manifest
    sampler = contract.get("sampler")
    if (
        not isinstance(sampler, Mapping)
        or sampler.get("sampling_mode")
        != FAILURE_REPLAY_80PCT_TASK_VARIANT_NOINITIAL
    ):
        return configured_manifest
    adaptive_state = payload.get("adaptive_replay_state")
    if not isinstance(adaptive_state, Mapping):
        return configured_manifest
    if (
        adaptive_state.get("schema_version")
        != ADAPTIVE_REPLAY_STATE_SCHEMA_VERSION
        or adaptive_state.get("branch_id") != config.branch_id
    ):
        raise ValueError("resume adaptive replay state differs from branch config")
    manifest = sampler.get("failure_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("resume sampler binding lacks its active manifest")
    if (
        sampler.get("failure_manifest_sha256")
        != adaptive_state.get("current_manifest_sha256")
    ):
        raise ValueError("resume adaptive state and sampler manifest disagree")
    return dict(manifest)


def is_same_mode_adaptive_replay_resume(trainer: Any) -> bool:
    """Return whether resume already contains this branch's adaptive state."""

    config = getattr(trainer, "adaptive_replay_config", None)
    if not isinstance(config, AdaptiveReplayTrainingConfig):
        return False
    state_path = _resume_trainer_state_path(getattr(trainer, "resume", None))
    if state_path is None:
        return False
    payload = _read_json_object(state_path, label="resume trainer state")
    state = payload.get("adaptive_replay_state")
    if state is None:
        return False
    if (
        not isinstance(state, Mapping)
        or state.get("schema_version") != ADAPTIVE_REPLAY_STATE_SCHEMA_VERSION
        or state.get("branch_id") != config.branch_id
    ):
        raise ValueError("resume adaptive replay state differs from branch config")
    return True


def initialize_adaptive_replay_state(trainer: Any) -> dict[str, Any] | None:
    config = getattr(trainer, "adaptive_replay_config", None)
    if not isinstance(config, AdaptiveReplayTrainingConfig):
        return None
    sampler = getattr(trainer, "train_sampler", None)
    if not isinstance(sampler, FailureReplayTaskVariantNoInitialSampler):
        raise TypeError("adaptive replay requires the task-variant sampler")
    binding = sampler.binding()
    return {
        "schema_version": ADAPTIVE_REPLAY_STATE_SCHEMA_VERSION,
        "branch_id": config.branch_id,
        "current_manifest_sha256": binding["failure_manifest_sha256"],
        "last_source_checkpoint_step": config.initial_manifest_source_step,
        "activations": [],
    }


def adaptive_replay_state_snapshot(trainer: Any) -> dict[str, Any] | None:
    state = getattr(trainer, "adaptive_replay_state", None)
    if state is None:
        return None
    return json.loads(
        json.dumps(state, ensure_ascii=True, sort_keys=True, allow_nan=False)
    )


def restore_adaptive_replay_state(trainer: Any, value: object) -> None:
    config = getattr(trainer, "adaptive_replay_config", None)
    if not isinstance(config, AdaptiveReplayTrainingConfig):
        if value is not None:
            raise ValueError(
                "checkpoint has adaptive replay state but config disables it"
            )
        return
    if value is None:
        return
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "branch_id",
        "current_manifest_sha256",
        "last_source_checkpoint_step",
        "activations",
    }:
        raise ValueError("checkpoint adaptive replay state has an invalid shape")
    if (
        value["schema_version"] != ADAPTIVE_REPLAY_STATE_SCHEMA_VERSION
        or value["branch_id"] != config.branch_id
        or not _is_sha256(value["current_manifest_sha256"])
    ):
        raise ValueError("checkpoint adaptive replay identity is invalid")
    last_step = _nonnegative_int(
        value["last_source_checkpoint_step"],
        label="last_source_checkpoint_step",
    )
    activations = value["activations"]
    if not isinstance(activations, list) or any(
        not isinstance(item, dict) for item in activations
    ):
        raise ValueError("adaptive replay activations must be a list of objects")
    sampler = getattr(trainer, "train_sampler", None)
    if not isinstance(sampler, FailureReplayTaskVariantNoInitialSampler):
        raise TypeError("adaptive replay checkpoint requires task-variant sampler")
    current_sha = sampler.binding()["failure_manifest_sha256"]
    if current_sha != value["current_manifest_sha256"]:
        raise ValueError("restored sampler differs from adaptive replay state")
    trainer.adaptive_replay_state = {
        "schema_version": ADAPTIVE_REPLAY_STATE_SCHEMA_VERSION,
        "branch_id": config.branch_id,
        "current_manifest_sha256": current_sha,
        "last_source_checkpoint_step": last_step,
        "activations": json.loads(json.dumps(activations, sort_keys=True)),
    }


@dataclass(frozen=True)
class AdaptiveReplayBoundaryResult:
    reset_data_iterator: bool = False
    request_id: str | None = None
    activation_step: int | None = None
    manifest_sha256: str | None = None
    changed: bool = False


class AdaptiveReplayActivationRuntime:
    """Poll rank-zero requests and activate one manifest on all ranks."""

    def __init__(self, trainer: Any, config: AdaptiveReplayTrainingConfig):
        self.config = config
        self.branch_root = config.control_dir / "branches" / config.branch_id
        self.request_dir = self.branch_root / "activation_requests"
        self.receipt_dir = self.branch_root / "activation_receipts"
        self.manifest_dir = self.branch_root / "manifests"
        self.evaluation_dir = self.branch_root / "evaluations"
        self._scheduled: dict[str, Any] | None = None
        self._rejected_request_ids: set[str] = set()
        self._invalid_request_paths: set[Path] = set()
        self._sidecar_process: subprocess.Popen[Any] | None = None
        if trainer.accelerator.is_main_process:
            for path in (
                self.request_dir,
                self.receipt_dir,
                self.manifest_dir,
                self.evaluation_dir,
            ):
                path.mkdir(parents=True, exist_ok=True)
        trainer.accelerator.wait_for_everyone()
        if trainer.accelerator.is_main_process and config.launch_sidecar:
            self._launch_sidecar(trainer)

    @classmethod
    def from_trainer(cls, trainer: Any) -> "AdaptiveReplayActivationRuntime | None":
        config = getattr(trainer, "adaptive_replay_config", None)
        if config is None:
            return None
        if not isinstance(config, AdaptiveReplayTrainingConfig):
            raise TypeError("trainer adaptive_replay_config was not validated")
        return cls(trainer, config)

    def _launch_sidecar(self, trainer: Any) -> None:
        if not self.config.sidecar_script.is_file():
            raise FileNotFoundError(
                f"adaptive replay sidecar is missing: {self.config.sidecar_script}"
            )
        if not self.config.evaluation_profile.is_file():
            raise FileNotFoundError(
                "adaptive replay evaluation profile is missing: "
                f"{self.config.evaluation_profile}"
            )
        log_path = Path(trainer.output_dir) / "adaptive_replay_sidecar.log"
        log_stream = log_path.open("ab", buffering=0)
        command = [
            sys.executable,
            str(self.config.sidecar_script),
            "--run-dir",
            str(Path(trainer.output_dir).resolve()),
            "--profile",
            str(self.config.evaluation_profile),
            "--control-dir",
            str(self.config.control_dir),
            "--branch-id",
            self.config.branch_id,
            "--checkpoint-interval",
            str(self.config.checkpoint_interval),
            "--initial-source-step",
            str(self.config.initial_manifest_source_step),
            "--poll-seconds",
            str(self.config.controller_poll_seconds),
            "--parent-pid",
            str(os.getpid()),
        ]
        self._sidecar_process = subprocess.Popen(
            command,
            cwd=str(Path(trainer.output_dir).resolve()),
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_stream.close()
        logger.info(
            "Adaptive replay sidecar launched pid=%d branch=%s log=%s",
            self._sidecar_process.pid,
            self.config.branch_id,
            log_path,
        )

    def _receipt_path(self, request_id: str) -> Path:
        return self.receipt_dir / f"{request_id}.json"

    def _write_activation_receipt(
        self,
        *,
        request: Mapping[str, Any],
        status: str,
        activation_step: int | None,
        changed: bool,
        error: str | None = None,
    ) -> None:
        request_id = str(request["request_id"])
        payload = {
            "schema_version": ADAPTIVE_REPLAY_ACTIVATION_RECEIPT_SCHEMA_VERSION,
            "request_id": request_id,
            "branch_id": self.config.branch_id,
            "status": status,
            "applied": status == "APPLIED",
            "source_checkpoint_step": request.get("source_checkpoint_step"),
            "activation_step": activation_step,
            "manifest_sha256": request.get("manifest", {}).get(
                "canonical_sampler_sha256"
            ),
            "changed": bool(changed),
            "error": error,
        }
        path = self._receipt_path(request_id)
        if path.is_file():
            existing = _read_json_object(path, label="activation receipt")
            if existing == payload:
                return
            if existing.get("status") == "APPLIED" and status == "APPLIED":
                # A resume from a checkpoint older than the receipt may
                # deterministically reapply the request at a new safe boundary.
                payload["previous_application"] = existing
        _atomic_json(path, payload)

    def _validate_evaluation_receipt(
        self,
        receipt: Mapping[str, Any],
        *,
        source_step: int,
        manifest_sha256: str,
    ) -> None:
        if (
            receipt.get("schema_version")
            != ADAPTIVE_I5_EVALUATION_RECEIPT_SCHEMA_VERSION
            or receipt.get("status") != "PASS"
            or receipt.get("passed") is not True
            or receipt.get("branch_id") != self.config.branch_id
        ):
            raise ValueError("I5 evaluation receipt did not PASS for this branch")
        source = receipt.get("source_checkpoint")
        coverage = receipt.get("coverage")
        manifest = receipt.get("manifest")
        if not isinstance(source, Mapping) or source.get("global_step") != source_step:
            raise ValueError("I5 receipt source checkpoint step is inconsistent")
        if not isinstance(coverage, Mapping) or any(
            coverage.get(key) != expected
            for key, expected in (
                ("expected_cells", 100),
                ("start_events", 100),
                ("end_events", 100),
                ("unique_cells", 100),
                ("conflicts", 0),
            )
        ):
            raise ValueError("I5 receipt is not an exact 100/100 cover")
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("canonical_sampler_sha256") != manifest_sha256
        ):
            raise ValueError("I5 receipt and replay manifest SHA disagree")

    def _load_request_bundle(self, path: Path) -> dict[str, Any]:
        request = _read_json_object(path, label="activation request")
        required = {
            "schema_version",
            "request_id",
            "branch_id",
            "source_checkpoint_step",
            "not_before_step",
            "manifest",
            "evaluation",
        }
        if set(request) != required:
            raise ValueError("activation request fields drifted")
        if (
            request["schema_version"]
            != ADAPTIVE_REPLAY_ACTIVATION_REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("activation request schema is unsupported")
        request_id = _safe_id(request["request_id"], label="request_id")
        if path.name != f"{request_id}.json":
            raise ValueError("activation request filename and request_id disagree")
        if request["branch_id"] != self.config.branch_id:
            raise ValueError("activation request targets another branch")
        source_step = _positive_int(
            request["source_checkpoint_step"],
            label="source_checkpoint_step",
        )
        not_before = _positive_int(
            request["not_before_step"], label="not_before_step"
        )
        if not_before <= source_step:
            raise ValueError("not_before_step must follow the source checkpoint")

        manifest_spec = request["manifest"]
        evaluation_spec = request["evaluation"]
        expected_file_fields = {"path", "file_sha256"}
        if (
            not isinstance(manifest_spec, Mapping)
            or set(manifest_spec)
            != expected_file_fields | {"canonical_sampler_sha256"}
            or not isinstance(evaluation_spec, Mapping)
            or set(evaluation_spec) != expected_file_fields
        ):
            raise ValueError("activation artifact bindings have an invalid shape")
        manifest_path = _absolute_path(
            manifest_spec["path"], label="activation manifest path"
        )
        evaluation_path = _absolute_path(
            evaluation_spec["path"], label="evaluation receipt path"
        )
        if manifest_path.parent != self.manifest_dir.resolve(strict=False):
            raise ValueError("activation manifest is outside the branch directory")
        if evaluation_path.parent != self.evaluation_dir.resolve(strict=False):
            raise ValueError("evaluation receipt is outside the branch directory")
        manifest_bytes = _read_regular_bytes(
            manifest_path, label="activation manifest"
        )
        evaluation_bytes = _read_regular_bytes(
            evaluation_path, label="I5 evaluation receipt"
        )
        if (
            _sha256_bytes(manifest_bytes) != manifest_spec["file_sha256"]
            or _sha256_bytes(evaluation_bytes) != evaluation_spec["file_sha256"]
        ):
            raise ValueError("activation artifact file SHA changed")
        try:
            manifest = json.loads(manifest_bytes)
            evaluation = json.loads(evaluation_bytes)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("activation artifact is not valid JSON") from exc
        manifest = canonical_task_variant_failure_manifest(manifest)
        manifest_sha = canonical_task_variant_failure_manifest_sha256(manifest)
        if manifest_sha != manifest_spec["canonical_sampler_sha256"]:
            raise ValueError("activation manifest canonical SHA changed")
        if not isinstance(evaluation, Mapping):
            raise ValueError("I5 evaluation receipt must be an object")
        self._validate_evaluation_receipt(
            evaluation,
            source_step=source_step,
            manifest_sha256=manifest_sha,
        )
        return {
            "request": request,
            "manifest": manifest,
            "manifest_sha256": manifest_sha,
        }

    def _main_scan_request(self, trainer: Any, step: int) -> dict[str, Any] | None:
        state = trainer.adaptive_replay_state
        last_source = int(state["last_source_checkpoint_step"])
        candidates: list[tuple[int, Path, dict[str, Any]]] = []
        for path in sorted(self.request_dir.glob("*.json")):
            if path in self._invalid_request_paths:
                continue
            try:
                bundle = self._load_request_bundle(path)
            except (OSError, TypeError, ValueError) as exc:
                logger.warning("Ignoring invalid adaptive replay request %s: %s", path, exc)
                self._invalid_request_paths.add(path)
                continue
            request = bundle["request"]
            if request["request_id"] in self._rejected_request_ids:
                continue
            source_step = int(request["source_checkpoint_step"])
            if source_step <= last_source:
                continue
            candidates.append((source_step, path, bundle))
        if not candidates:
            return None
        # Prefer the newest completed checkpoint if delayed evaluations ever
        # overlap, so an obsolete replay pool is never applied briefly.
        _, _, bundle = max(candidates, key=lambda item: item[0])
        request = bundle["request"]
        earliest = max(
            int(request["not_before_step"]),
            int(step) + self.config.activation_lead_steps,
        )
        multiple = self.config.activation_step_multiple
        bundle["activation_step"] = ((earliest + multiple - 1) // multiple) * multiple
        return bundle

    def _collect_rank_zero_value(self, trainer: Any, value: object) -> object:
        local = value if trainer.accelerator.is_main_process else None
        gathered = trainer._all_gather_rank_objects(local)
        if not gathered or not isinstance(gathered[0], (dict, type(None))):
            raise RuntimeError("rank zero adaptive replay broadcast is malformed")
        if any(item is not None for item in gathered[1:]):
            raise RuntimeError("non-zero rank authored an adaptive replay command")
        return gathered[0]

    def boundary(self, trainer: Any, *, step: int) -> AdaptiveReplayBoundaryResult:
        if self._scheduled is None and (
            step % self.config.activation_poll_interval_steps == 0
        ):
            candidate = (
                self._main_scan_request(trainer, step)
                if trainer.accelerator.is_main_process
                else None
            )
            self._scheduled = self._collect_rank_zero_value(trainer, candidate)
            if self._scheduled is not None:
                logger.info(
                    "Adaptive replay scheduled request=%s source_step=%d "
                    "activation_step=%d",
                    self._scheduled["request"]["request_id"],
                    self._scheduled["request"]["source_checkpoint_step"],
                    self._scheduled["activation_step"],
                )
        if self._scheduled is None or step < self._scheduled["activation_step"]:
            return AdaptiveReplayBoundaryResult()
        if step != self._scheduled["activation_step"]:
            raise RuntimeError("adaptive replay activation boundary was skipped")

        bundle = self._scheduled
        request = bundle["request"]
        request_id = str(request["request_id"])
        sampler = trainer.train_sampler
        local_error: str | None = None
        replacement = None
        replacement_binding = None
        try:
            if not isinstance(sampler, FailureReplayTaskVariantNoInitialSampler):
                raise TypeError("active sampler cannot accept replay manifests")
            replacement = sampler.build_manifest_replacement(bundle["manifest"])
            replacement.set_epoch(trainer.epoch)
            replacement.set_resume_batch_offset(trainer.batch_in_epoch)
            replacement_binding = operational_sampler_resume_state(
                replacement.binding()
            )
            if (
                replacement_binding["failure_manifest_sha256"]
                != bundle["manifest_sha256"]
            ):
                raise ValueError("replacement sampler manifest SHA differs")
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        reports = trainer._all_gather_rank_objects(
            {"error": local_error, "binding": replacement_binding}
        )
        errors = [
            {"rank": rank, "error": report.get("error")}
            for rank, report in enumerate(reports)
            if report.get("error") is not None
        ]
        bindings = [report.get("binding") for report in reports]
        if errors or any(binding != bindings[0] for binding in bindings[1:]):
            error = json.dumps(
                errors or [{"error": "replacement bindings differ across ranks"}],
                sort_keys=True,
            )
            if trainer.accelerator.is_main_process:
                self._write_activation_receipt(
                    request=request,
                    status="REJECTED",
                    activation_step=step,
                    changed=False,
                    error=error,
                )
            trainer.accelerator.wait_for_everyone()
            self._rejected_request_ids.add(request_id)
            self._scheduled = None
            logger.warning(
                "Adaptive replay request %s rejected without changing sampler: %s",
                request_id,
                error,
            )
            return AdaptiveReplayBoundaryResult(request_id=request_id)

        assert replacement is not None
        previous_sha = sampler.binding()["failure_manifest_sha256"]
        changed = previous_sha != bundle["manifest_sha256"]
        if changed:
            sampler.adopt_manifest_replacement(replacement)
            sampler.set_epoch(trainer.epoch)
            sampler.set_resume_batch_offset(trainer.batch_in_epoch)
        state = trainer.adaptive_replay_state
        state["current_manifest_sha256"] = bundle["manifest_sha256"]
        state["last_source_checkpoint_step"] = int(
            request["source_checkpoint_step"]
        )
        state["activations"].append(
            {
                "request_id": request_id,
                "source_checkpoint_step": int(request["source_checkpoint_step"]),
                "activation_step": int(step),
                "manifest_sha256": bundle["manifest_sha256"],
                "changed": changed,
            }
        )
        if trainer.accelerator.is_main_process:
            self._write_activation_receipt(
                request=request,
                status="APPLIED",
                activation_step=step,
                changed=changed,
            )
        trainer.accelerator.wait_for_everyone()
        self._scheduled = None
        logger.info(
            "Adaptive replay applied request=%s source_step=%d step=%d "
            "changed=%s manifest=%s",
            request_id,
            request["source_checkpoint_step"],
            step,
            changed,
            bundle["manifest_sha256"],
        )
        return AdaptiveReplayBoundaryResult(
            reset_data_iterator=changed,
            request_id=request_id,
            activation_step=step,
            manifest_sha256=bundle["manifest_sha256"],
            changed=changed,
        )


__all__ = [
    "ADAPTIVE_I5_EVALUATION_RECEIPT_SCHEMA_VERSION",
    "ADAPTIVE_REPLAY_ACTIVATION_RECEIPT_SCHEMA_VERSION",
    "ADAPTIVE_REPLAY_ACTIVATION_REQUEST_SCHEMA_VERSION",
    "ADAPTIVE_REPLAY_STATE_SCHEMA_VERSION",
    "ADAPTIVE_REPLAY_TRAINING_CONFIG_SCHEMA_VERSION",
    "AdaptiveReplayActivationRuntime",
    "AdaptiveReplayBoundaryResult",
    "AdaptiveReplayTrainingConfig",
    "adaptive_replay_state_snapshot",
    "initialize_adaptive_replay_state",
    "is_same_mode_adaptive_replay_resume",
    "manifest_for_adaptive_resume",
    "restore_adaptive_replay_state",
    "validate_adaptive_replay_training_config",
]
