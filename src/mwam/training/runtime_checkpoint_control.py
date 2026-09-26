"""Run-local requests for checkpointing at a safe optimizer-step boundary.

The external control plane only publishes a small, session-bound JSON request.
Rank zero claims that request after a successful optimizer update, broadcasts a
small action code, and all ranks reuse the trainer's existing collective
``save_checkpoint`` implementation.  The module never signals a worker or
writes model state itself.
"""

from __future__ import annotations

import json
import os
import re
import stat
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from accelerate.utils import broadcast as accelerate_broadcast


ACTIVE_SCHEMA_VERSION = "fastwam_runtime_checkpoint_active_v1"
REQUEST_SCHEMA_VERSION = "fastwam_runtime_checkpoint_request_v1"
RESPONSE_SCHEMA_VERSION = "fastwam_runtime_checkpoint_response_v1"

ACTION_CHECKPOINT = "checkpoint"
ACTION_CHECKPOINT_AND_STOP = "checkpoint-and-stop"
ACTION_CODES = {
    ACTION_CHECKPOINT: 1,
    ACTION_CHECKPOINT_AND_STOP: 2,
}
ACTION_NAMES_BY_CODE = {value: key for key, value in ACTION_CODES.items()}

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_JSON_BYTES = 1024 * 1024


class RuntimeCheckpointControlError(ValueError):
    """A runtime checkpoint request or control directory is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validated_request_id(value: object) -> str:
    if not isinstance(value, str) or _SAFE_REQUEST_ID.fullmatch(value) is None:
        raise RuntimeCheckpointControlError(
            "request_id must be a safe 1..128 character identifier"
        )
    return value


def _validated_session_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeCheckpointControlError("session_id must be a UUID string")
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise RuntimeCheckpointControlError(
            "session_id must be a UUID string"
        ) from exc


def _validated_action(value: object) -> str:
    if not isinstance(value, str) or value not in ACTION_CODES:
        raise RuntimeCheckpointControlError(
            f"action must be one of {sorted(ACTION_CODES)}"
        )
    return value


def _ensure_real_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise RuntimeCheckpointControlError(
            f"runtime checkpoint path must be a real directory: {path}"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _write_new_file(path: Path, encoded: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write while publishing {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_real_directory(path.parent)
    pending = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        _write_new_file(pending, _json_bytes(payload))
        os.replace(pending, path)
        _fsync_directory(path.parent)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish JSON create-only without exposing a partially written file."""

    _ensure_real_directory(path.parent)
    pending = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        _write_new_file(pending, _json_bytes(payload))
        try:
            os.link(pending, path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"runtime checkpoint path already exists: {path}")
        _fsync_directory(path.parent)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
            raise RuntimeCheckpointControlError(
                f"runtime checkpoint JSON must be a non-empty regular file: {path}"
            )
        if value.st_size > _MAX_JSON_BYTES:
            raise RuntimeCheckpointControlError(
                f"runtime checkpoint JSON is unexpectedly large: {path}"
            )
        raw = b""
        while len(raw) <= _MAX_JSON_BYTES:
            chunk = os.read(descriptor, min(64 * 1024, _MAX_JSON_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(descriptor)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeCheckpointControlError(
            f"runtime checkpoint JSON is invalid: {path}"
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeCheckpointControlError(
            f"runtime checkpoint JSON root must be an object: {path}"
        )
    return parsed


def _control_paths(run_dir: Path) -> dict[str, Path]:
    root = run_dir / "runtime_checkpoint_control"
    return {
        "root": root,
        "active": root / "active.json",
        "pending": root / "pending" / "request.json",
        "inflight": root / "inflight",
        "history": root / "history",
        "responses": root / "responses",
    }


def _initialize_control_directories(run_dir: Path) -> dict[str, Path]:
    paths = _control_paths(run_dir)
    for key in ("root", "pending", "inflight", "history", "responses"):
        path = paths[key]
        _ensure_real_directory(path if key != "pending" else path.parent)
    return paths


@dataclass(frozen=True)
class RuntimeCheckpointRequest:
    request_id: str
    session_id: str
    action: str
    created_at_utc: str
    run_dir: str
    inflight_path: Path


@dataclass(frozen=True)
class RuntimeCheckpointBoundaryResult:
    action: str | None = None
    request_id: str | None = None
    checkpoint_info: Mapping[str, str | None] | None = None
    stop_requested: bool = False
    poll_error: str | None = None


class RuntimeCheckpointControl:
    """Rank-zero filesystem endpoint for one active training session."""

    def __init__(
        self,
        *,
        configured: bool,
        is_main_process: bool,
        run_dir: Path | None = None,
        session_id: str | None = None,
        initialization_error: str | None = None,
    ) -> None:
        self.configured = configured
        self.is_main_process = is_main_process
        self.run_dir = run_dir
        self.session_id = session_id
        self.initialization_error = initialization_error
        self.paths = _control_paths(run_dir) if run_dir is not None else None

    @classmethod
    def from_trainer(cls, trainer: Any) -> "RuntimeCheckpointControl":
        configured = getattr(trainer, "runtime_checkpoint_control_enabled", False)
        if not isinstance(configured, bool):
            raise TypeError("runtime_checkpoint_control_enabled must be a boolean")
        is_main = bool(trainer.accelerator.is_main_process)
        if not configured:
            return cls(configured=False, is_main_process=is_main)

        run_dir = Path(trainer.output_dir).expanduser().resolve()
        if not is_main:
            return cls(
                configured=True,
                is_main_process=False,
                run_dir=run_dir,
            )

        try:
            writer = getattr(trainer, "training_metrics_writer", None)
            session_id = _validated_session_id(
                getattr(writer, "session_id", None)
            )
            paths = _initialize_control_directories(run_dir)
            payload = {
                "schema_version": ACTIVE_SCHEMA_VERSION,
                "session_id": session_id,
                "run_dir": str(run_dir),
                "world_size": int(trainer.accelerator.num_processes),
                "main_pid": os.getpid(),
                "start_step": int(trainer.global_step),
                "started_at_utc": _utc_now(),
                "status": "running",
            }
            _atomic_replace_json(paths["active"], payload)
            return cls(
                configured=True,
                is_main_process=True,
                run_dir=run_dir,
                session_id=session_id,
            )
        except (OSError, TypeError, ValueError) as exc:
            return cls(
                configured=True,
                is_main_process=True,
                run_dir=run_dir,
                initialization_error=str(exc),
            )

    @property
    def ready(self) -> bool:
        return (
            self.configured
            and self.is_main_process
            and self.paths is not None
            and self.session_id is not None
            and self.initialization_error is None
        )

    def _validated_request(self, payload: Mapping[str, Any]) -> dict[str, str]:
        expected = {
            "schema_version",
            "request_id",
            "session_id",
            "action",
            "created_at_utc",
            "run_dir",
        }
        if set(payload) != expected:
            raise RuntimeCheckpointControlError(
                "runtime checkpoint request has an invalid structure"
            )
        if payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
            raise RuntimeCheckpointControlError(
                "runtime checkpoint request has an unsupported schema"
            )
        request_id = _validated_request_id(payload.get("request_id"))
        session_id = _validated_session_id(payload.get("session_id"))
        action = _validated_action(payload.get("action"))
        created_at = payload.get("created_at_utc")
        if not isinstance(created_at, str) or not created_at:
            raise RuntimeCheckpointControlError(
                "runtime checkpoint request created_at_utc is invalid"
            )
        run_dir = payload.get("run_dir")
        if not isinstance(run_dir, str) or self.run_dir is None:
            raise RuntimeCheckpointControlError(
                "runtime checkpoint request run_dir is invalid"
            )
        if run_dir != str(self.run_dir):
            raise RuntimeCheckpointControlError(
                "runtime checkpoint request targets another run directory"
            )
        if session_id != self.session_id:
            raise RuntimeCheckpointControlError(
                "runtime checkpoint request targets another training session"
            )
        return {
            "request_id": request_id,
            "session_id": session_id,
            "action": action,
            "created_at_utc": created_at,
            "run_dir": run_dir,
        }

    def _reject_pending(self, reason: str) -> None:
        assert self.paths is not None
        pending = self.paths["pending"]
        rejection_id = f"rejected-{uuid.uuid4().hex}"
        history = self.paths["history"] / f"{rejection_id}.request.json"
        try:
            os.rename(pending, history)
            _fsync_directory(pending.parent)
            _fsync_directory(history.parent)
        except OSError:
            return
        response = {
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "request_id": rejection_id,
            "session_id": self.session_id,
            "action": None,
            "status": "rejected",
            "completed_at_utc": _utc_now(),
            "global_step": None,
            "weights_path": None,
            "state_path": None,
            "stop_requested": False,
            "error": reason,
        }
        _atomic_create_json(
            self.paths["responses"] / f"{rejection_id}.json",
            response,
        )

    def claim_pending(self) -> RuntimeCheckpointRequest | None:
        if not self.ready:
            return None
        assert self.paths is not None
        pending = self.paths["pending"]
        if not pending.exists():
            return None
        try:
            validated = self._validated_request(_read_json(pending))
        except (OSError, TypeError, ValueError) as exc:
            self._reject_pending(str(exc))
            return None
        inflight = self.paths["inflight"] / (
            f"{validated['request_id']}.request.json"
        )
        if inflight.exists() or inflight.is_symlink():
            self._reject_pending(
                f"inflight request already exists: {validated['request_id']}"
            )
            return None
        os.rename(pending, inflight)
        _fsync_directory(pending.parent)
        _fsync_directory(inflight.parent)
        return RuntimeCheckpointRequest(
            request_id=validated["request_id"],
            session_id=validated["session_id"],
            action=validated["action"],
            created_at_utc=validated["created_at_utc"],
            run_dir=validated["run_dir"],
            inflight_path=inflight,
        )

    def complete(
        self,
        request: RuntimeCheckpointRequest,
        *,
        global_step: int,
        checkpoint_info: Mapping[str, str | None],
    ) -> Path:
        if not self.ready or self.paths is None:
            raise RuntimeCheckpointControlError(
                "rank-zero runtime checkpoint control is not ready"
            )
        if request.session_id != self.session_id:
            raise RuntimeCheckpointControlError(
                "cannot complete a request from another training session"
            )
        response_path = self.paths["responses"] / f"{request.request_id}.json"
        response = {
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "request_id": request.request_id,
            "session_id": request.session_id,
            "action": request.action,
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "global_step": int(global_step),
            "weights_path": checkpoint_info.get("weights_path"),
            "state_path": checkpoint_info.get("state_path"),
            "stop_requested": request.action == ACTION_CHECKPOINT_AND_STOP,
            "error": None,
        }
        if response_path.exists():
            existing = _read_json(response_path)
            stable_keys = set(response).difference({"completed_at_utc"})
            if (
                set(existing) != set(response)
                or not isinstance(existing.get("completed_at_utc"), str)
                or any(existing.get(key) != response[key] for key in stable_keys)
            ):
                raise RuntimeCheckpointControlError(
                    f"response already exists with different content: {response_path}"
                )
        else:
            _atomic_create_json(response_path, response)
        history = self.paths["history"] / f"{request.request_id}.request.json"
        if request.inflight_path.exists():
            if history.exists() or history.is_symlink():
                raise RuntimeCheckpointControlError(
                    f"request history already exists: {history}"
                )
            os.rename(request.inflight_path, history)
            _fsync_directory(request.inflight_path.parent)
            _fsync_directory(history.parent)
        return response_path

    def close(self, *, status: str, global_step: int) -> None:
        if not self.ready or self.paths is None:
            return
        active_path = self.paths["active"]
        try:
            active = _read_json(active_path)
        except (OSError, TypeError, ValueError):
            return
        if active.get("session_id") != self.session_id:
            return
        active.update(
            {
                "status": str(status),
                "final_step": int(global_step),
                "finished_at_utc": _utc_now(),
            }
        )
        _atomic_replace_json(active_path, active)


def _distributed_action_code(
    trainer: Any,
    *,
    local_action_code: int,
    device: torch.device,
) -> int:
    if local_action_code not in {0, *ACTION_NAMES_BY_CODE}:
        raise RuntimeCheckpointControlError(
            f"unsupported local runtime checkpoint action code: {local_action_code}"
        )
    if (
        not bool(trainer.accelerator.is_main_process)
        and local_action_code != 0
    ):
        raise RuntimeCheckpointControlError(
            "only global rank zero may publish a runtime checkpoint action"
        )
    local = torch.tensor(
        [local_action_code],
        device=device,
        dtype=torch.int64,
    )
    world_size = int(trainer.accelerator.num_processes)
    published = (
        local
        if world_size == 1
        else accelerate_broadcast(local, from_process=0)
    )
    if published.numel() != 1:
        raise RuntimeError(
            "runtime checkpoint action broadcast did not return one scalar"
        )
    action_code = int(published.detach().cpu().item())
    if action_code not in {0, *ACTION_NAMES_BY_CODE}:
        raise RuntimeError(
            f"rank zero published an invalid runtime checkpoint action: {action_code}"
        )
    return action_code


def process_runtime_checkpoint_boundary(
    trainer: Any,
    control: RuntimeCheckpointControl,
    *,
    device: torch.device,
) -> RuntimeCheckpointBoundaryResult:
    """Process at most one request after a successful optimizer update."""

    if not control.configured:
        return RuntimeCheckpointBoundaryResult()

    request = None
    poll_error = control.initialization_error
    if control.is_main_process and control.ready:
        try:
            request = control.claim_pending()
        except (OSError, TypeError, ValueError) as exc:
            poll_error = str(exc)
    local_code = 0 if request is None else ACTION_CODES[request.action]
    action_code = _distributed_action_code(
        trainer,
        local_action_code=local_code,
        device=device,
    )
    if action_code == 0:
        return RuntimeCheckpointBoundaryResult(poll_error=poll_error)

    action = ACTION_NAMES_BY_CODE[action_code]
    checkpoint_info = trainer.save_checkpoint()
    if control.is_main_process:
        if request is None:
            raise RuntimeError(
                "rank zero lost the claimed runtime checkpoint request"
            )
        control.complete(
            request,
            global_step=int(trainer.global_step),
            checkpoint_info=checkpoint_info,
        )
    return RuntimeCheckpointBoundaryResult(
        action=action,
        request_id=None if request is None else request.request_id,
        checkpoint_info=checkpoint_info,
        stop_requested=action == ACTION_CHECKPOINT_AND_STOP,
        poll_error=poll_error,
    )


def _validated_active_session(run_dir: Path) -> dict[str, Any]:
    active = _read_json(_control_paths(run_dir)["active"])
    required = {
        "schema_version",
        "session_id",
        "run_dir",
        "world_size",
        "main_pid",
        "start_step",
        "started_at_utc",
        "status",
    }
    optional = {"final_step", "finished_at_utc"}
    if not required.issubset(active) or not set(active).issubset(required | optional):
        raise RuntimeCheckpointControlError(
            "runtime checkpoint active session has an invalid structure"
        )
    if active.get("schema_version") != ACTIVE_SCHEMA_VERSION:
        raise RuntimeCheckpointControlError(
            "runtime checkpoint active session has an unsupported schema"
        )
    active["session_id"] = _validated_session_id(active.get("session_id"))
    if active.get("run_dir") != str(run_dir):
        raise RuntimeCheckpointControlError(
            "runtime checkpoint active session targets another run directory"
        )
    if active.get("status") != "running":
        raise RuntimeCheckpointControlError(
            f"training session is not running: {active.get('status')!r}"
        )
    return active


def submit_runtime_checkpoint_request(
    run_dir: str | os.PathLike[str],
    *,
    action: str,
    request_id: str | None = None,
    expected_session_id: str | None = None,
    execute: bool,
) -> dict[str, Any]:
    """Plan or create one session-bound request in a real run directory."""

    requested = Path(run_dir).expanduser()
    if requested.is_symlink():
        raise RuntimeCheckpointControlError(
            f"run directory must not be a symlink: {requested}"
        )
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeCheckpointControlError(
            f"run directory must be a real directory: {resolved}"
        )
    action = _validated_action(action)
    paths = _control_paths(resolved)
    active = _validated_active_session(resolved)
    session_id = str(active["session_id"])
    if expected_session_id is not None:
        expected = _validated_session_id(expected_session_id)
        if expected != session_id:
            raise RuntimeCheckpointControlError(
                "expected session does not match the active training session"
            )
    request_id = _validated_request_id(request_id or uuid.uuid4().hex)
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "request_id": request_id,
        "session_id": session_id,
        "action": action,
        "created_at_utc": _utc_now(),
        "run_dir": str(resolved),
    }

    response_path = paths["responses"] / f"{request_id}.json"
    if response_path.exists():
        response = _read_json(response_path)
        if (
            response.get("request_id") != request_id
            or response.get("session_id") != session_id
            or response.get("action") != action
        ):
            raise RuntimeCheckpointControlError(
                "request_id already belongs to another completed request"
            )
        return {
            "action": action,
            "run_dir": str(resolved),
            "session_id": session_id,
            "request_id": request_id,
            "request_path": None,
            "response_path": str(response_path),
            "executed": False,
            "already_completed": True,
            "response": response,
        }

    inflight_path = paths["inflight"] / f"{request_id}.request.json"
    if inflight_path.exists():
        inflight = _read_json(inflight_path)
        if (
            inflight.get("session_id") != session_id
            or inflight.get("action") != action
        ):
            raise RuntimeCheckpointControlError(
                "request_id already belongs to another inflight request"
            )
        return {
            "action": action,
            "run_dir": str(resolved),
            "session_id": session_id,
            "request_id": request_id,
            "request_path": str(inflight_path),
            "response_path": str(response_path),
            "executed": False,
            "already_inflight": True,
        }

    pending_path = paths["pending"]
    if pending_path.exists():
        pending = _read_json(pending_path)
        if (
            pending.get("request_id") == request_id
            and pending.get("session_id") == session_id
            and pending.get("action") == action
        ):
            return {
                "action": action,
                "run_dir": str(resolved),
                "session_id": session_id,
                "request_id": request_id,
                "request_path": str(pending_path),
                "response_path": str(response_path),
                "executed": False,
                "already_pending": True,
            }
        raise RuntimeCheckpointControlError(
            "another runtime checkpoint request is already pending"
        )

    other_inflight = []
    for candidate in sorted(paths["inflight"].glob("*.request.json")):
        try:
            payload = _read_json(candidate)
        except (OSError, TypeError, ValueError):
            continue
        if payload.get("session_id") == session_id:
            other_inflight.append(candidate.name)
    if other_inflight:
        raise RuntimeCheckpointControlError(
            "another runtime checkpoint request is already inflight: "
            + ", ".join(other_inflight)
        )

    result = {
        "action": action,
        "run_dir": str(resolved),
        "session_id": session_id,
        "request_id": request_id,
        "request_path": str(pending_path),
        "response_path": str(response_path),
        "executed": False,
    }
    if execute:
        _atomic_create_json(pending_path, request)
        result["executed"] = True
    return result


def wait_for_runtime_checkpoint_response(
    run_dir: str | os.PathLike[str],
    *,
    request_id: str,
    timeout_seconds: float,
    poll_seconds: float = 1.0,
) -> dict[str, Any] | None:
    request_id = _validated_request_id(request_id)
    if timeout_seconds < 0 or poll_seconds <= 0:
        raise RuntimeCheckpointControlError(
            "timeout_seconds must be non-negative and poll_seconds positive"
        )
    resolved = Path(run_dir).expanduser().resolve(strict=True)
    response_path = _control_paths(resolved)["responses"] / f"{request_id}.json"
    deadline = time.monotonic() + timeout_seconds
    while True:
        if response_path.exists():
            return _read_json(response_path)
        if time.monotonic() >= deadline:
            return None
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


__all__ = [
    "ACTION_CHECKPOINT",
    "ACTION_CHECKPOINT_AND_STOP",
    "RuntimeCheckpointBoundaryResult",
    "RuntimeCheckpointControl",
    "RuntimeCheckpointControlError",
    "RuntimeCheckpointRequest",
    "process_runtime_checkpoint_boundary",
    "submit_runtime_checkpoint_request",
    "wait_for_runtime_checkpoint_response",
]
