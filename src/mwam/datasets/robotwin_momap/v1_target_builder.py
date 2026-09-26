"""CPU adapter from V1 canonical episodes to the DataLoader MoMap schema.

The verified :class:`mwam.robotwin_momap.MomapTargetService` owns all V1
HDF5/private-geometry reads and the Phase-2 geometry implementation.  This
module requests every per-camera result in the same head-camera-at-source
reference and adapts it to the explicit multi-view PyTorch target consumed by
:class:`RoboTwinMoMapDataset`.

No missing mask is synthesized.  In particular, the service must expose the
geometry-origin masks ``unmapped`` and ``unsupported`` as well as the
visibility-origin ``depth_conflict`` mask.  Older service versions which drop
those fields are rejected rather than being represented by misleading
all-false tensors.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mwam.robotwin_momap.se3 import SE3_VALIDATION_SCHEMA_VERSION
from mwam.robotwin_momap.target_service import (
    MOMAP_REPRESENTATION_CANONICAL_DXYZ,
    MOMAP_REPRESENTATION_SOURCE_ALIGNED_IMAGE_FLOW_PX,
)

from .schema import MoMapTarget, PrivateGeometry, SampleRef
from .source_conditions import COMMON_REFERENCE_CAMERA


ServiceFactory = Callable[[Path], Any]

_PRIVATE_ADDRESS_KEYS = frozenset(
    {
        "episode_path",
        "source_index",
        "target_indices",
        "camera_names",
        "coordinate_frame",
    }
)

TARGET_REPRESENTATION_CANONICAL_DXYZ_ALL_VIEWS = (
    "canonical_dxyz_all_views_v1"
)
TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV = (
    "all_views_dxyz_matched_uv_valid_v1"
)
TARGET_REPRESENTATIONS = frozenset(
    {
        TARGET_REPRESENTATION_CANONICAL_DXYZ_ALL_VIEWS,
        TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV,
    }
)


class TargetServiceContractError(RuntimeError):
    """The upstream target service omitted or contradicted required labels."""


def _default_service_factory(episode_path: Path) -> Any:
    # Keep the heavy canonical reader/geometry package out of import-time paths
    # which only use the synthetic DataLoader skeleton.
    from mwam.robotwin_momap import MomapTargetService

    return MomapTargetService(episode_path)


def _required_attribute(value: Any, name: str, *, owner: str) -> Any:
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise TargetServiceContractError(
            f"{owner} must expose {name!r}; refusing to fabricate a mask"
        ) from exc


def _numpy_mask(value: Any, name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != np.bool_:
        raise TargetServiceContractError(
            f"{name} must use boolean dtype, got {mask.dtype}"
        )
    if tuple(mask.shape) != expected_shape:
        raise TargetServiceContractError(
            f"{name} shape={tuple(mask.shape)}, expected={expected_shape}"
        )
    return np.array(mask, dtype=np.bool_, order="C", copy=True)


def _torch_mask(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value)


def _close_quietly(service: Any) -> None:
    close = getattr(service, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            # Cleanup must not hide the original exception or make object
            # finalization noisy. Explicit service calls remain fail-fast.
            pass


class V1CPUMoMapTargetBuilder:
    """Build one ``[V,K,3,H,W]`` MoMap target through the CPU V1 service.

    A service is opened lazily per episode and cached inside the current
    process.  A PID guard discards inherited services after ``fork`` so HDF5
    handles are never reused across DataLoader workers.

    ``PrivateGeometry`` carries only an opaque address duplicated from the
    ``SampleRef``.  The two addresses must agree exactly.  The target service
    then reads privileged fields internally and returns supervision only; raw
    geometry arrays never cross this adapter API.
    """

    def __init__(
        self,
        *,
        grid_size: tuple[int, int] = (240, 320),
        evaluate_visibility: bool = False,
        depth_tolerance_m: float = 0.005,
        pixel_boundary_ambiguity_px: float = 0.0,
        output_dtype: str | np.dtype[Any] = np.float32,
        max_open_episodes: int = 4,
        reference_camera_name: str = COMMON_REFERENCE_CAMERA,
        target_representation: str = (
            TARGET_REPRESENTATION_CANONICAL_DXYZ_ALL_VIEWS
        ),
        wrist_flow_max_magnitude_native_px: float | None = None,
        wrist_flow_gt_background_filter: bool = False,
        service_factory: ServiceFactory | None = None,
    ) -> None:
        if (
            not isinstance(grid_size, tuple)
            or len(grid_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in grid_size)
            or any(value <= 0 for value in grid_size)
        ):
            raise ValueError("grid_size must be a pair of positive integers")
        if not isinstance(evaluate_visibility, bool):
            raise TypeError("evaluate_visibility must be boolean")
        if not np.isfinite(depth_tolerance_m) or depth_tolerance_m < 0:
            raise ValueError("depth_tolerance_m must be finite and non-negative")
        if (
            not np.isfinite(pixel_boundary_ambiguity_px)
            or pixel_boundary_ambiguity_px < 0
        ):
            raise ValueError(
                "pixel_boundary_ambiguity_px must be finite and non-negative"
            )
        dtype = np.dtype(output_dtype)
        if dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("output_dtype must be float16 or float32")
        if (
            isinstance(max_open_episodes, bool)
            or not isinstance(max_open_episodes, int)
            or max_open_episodes <= 0
        ):
            raise ValueError("max_open_episodes must be a positive integer")
        if service_factory is not None and not callable(service_factory):
            raise TypeError("service_factory must be callable")
        if not isinstance(reference_camera_name, str) or not reference_camera_name.strip():
            raise TypeError("reference_camera_name must be a non-empty string")
        if target_representation not in TARGET_REPRESENTATIONS:
            raise ValueError(
                "target_representation must be one of "
                f"{sorted(TARGET_REPRESENTATIONS)}, got "
                f"{target_representation!r}"
            )
        if wrist_flow_max_magnitude_native_px is not None:
            if (
                isinstance(wrist_flow_max_magnitude_native_px, bool)
                or not isinstance(
                    wrist_flow_max_magnitude_native_px, (int, float)
                )
                or not np.isfinite(wrist_flow_max_magnitude_native_px)
                or wrist_flow_max_magnitude_native_px <= 0
            ):
                raise ValueError(
                    "wrist_flow_max_magnitude_native_px must be finite and "
                    "positive or None"
                )
            if (
                target_representation
                != TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV
            ):
                raise ValueError(
                    "wrist Flow magnitude gate requires the hybrid Head-dXYZ/"
                    "Wrist-dUV target representation"
                )

        if not isinstance(wrist_flow_gt_background_filter, bool):
            raise TypeError("wrist_flow_gt_background_filter must be boolean")
        if wrist_flow_gt_background_filter and (
            wrist_flow_max_magnitude_native_px is not None
            or target_representation != TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV
        ):
            raise ValueError("GT background filtering requires hybrid targets and no magnitude gate")
        self.wrist_flow_gt_background_filter = wrist_flow_gt_background_filter
        self.grid_size = grid_size
        self.evaluate_visibility = evaluate_visibility
        self.depth_tolerance_m = float(depth_tolerance_m)
        self.pixel_boundary_ambiguity_px = float(pixel_boundary_ambiguity_px)
        self.output_dtype = dtype
        self.max_open_episodes = max_open_episodes
        self.reference_camera_name = reference_camera_name
        self.target_representation = target_representation
        self.wrist_flow_max_magnitude_native_px = (
            None
            if wrist_flow_max_magnitude_native_px is None
            else float(wrist_flow_max_magnitude_native_px)
        )
        self._service_factory = service_factory or _default_service_factory
        self._services: OrderedDict[str, Any] = OrderedDict()
        self._owner_pid = os.getpid()

    def _reset_after_fork(self) -> None:
        current_pid = os.getpid()
        if current_pid == self._owner_pid:
            return
        for service in self._services.values():
            _close_quietly(service)
        self._services.clear()
        self._owner_pid = current_pid

    def resume_binding(self) -> dict[str, Any]:
        factory_module = getattr(
            self._service_factory,
            "__module__",
            type(self._service_factory).__module__,
        )
        factory_name = getattr(
            self._service_factory,
            "__qualname__",
            type(self._service_factory).__qualname__,
        )
        binding = {
            "schema_version": "robotwin_momap_v1_cpu_target_builder_binding_v1",
            "target_builder_adapter": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
            "target_service_factory": f"{factory_module}.{factory_name}",
            "target_service_factory_production_auditable": (
                self._service_factory is _default_service_factory
            ),
            "grid_size": list(self.grid_size),
            "output_dtype": self.output_dtype.name,
            "evaluate_visibility": self.evaluate_visibility,
            "depth_tolerance_m": self.depth_tolerance_m,
            "pixel_boundary_ambiguity_px": self.pixel_boundary_ambiguity_px,
            "reference_camera_name": self.reference_camera_name,
            "target_representation": self.target_representation,
            "max_open_episodes": self.max_open_episodes,
            "se3_validation_schema_version": SE3_VALIDATION_SCHEMA_VERSION,
        }
        if self.wrist_flow_max_magnitude_native_px is not None:
            binding["wrist_flow_max_magnitude_native_px"] = (
                self.wrist_flow_max_magnitude_native_px
            )
            binding["momap_loss_valid_semantics"] = (
                "component_valid_and_native_wrist_duv_l2_gate_v1"
            )
        if self.wrist_flow_gt_background_filter:
            binding["wrist_flow_gt_background_filter"] = True
            binding["momap_loss_valid_semantics"] = (
                "component_valid_and_gt_entity_background_exclusion_v1"
            )
            binding["gt_background_entity_names"] = ["background", "ground", "table", "wall"]
        return binding

    def _service_for(self, episode_path: str | Path) -> Any:
        self._reset_after_fork()
        path = Path(episode_path)
        key = os.fspath(path)
        service = self._services.get(key)
        if service is not None:
            self._services.move_to_end(key)
            return service

        service = self._service_factory(path)
        if not callable(getattr(service, "get_momap", None)):
            _close_quietly(service)
            raise TypeError("service_factory result must provide get_momap()")
        self._services[key] = service
        self._services.move_to_end(key)
        while len(self._services) > self.max_open_episodes:
            _, evicted = self._services.popitem(last=False)
            _close_quietly(evicted)
        return service

    def close(self) -> None:
        """Close every service opened in this process."""

        self._reset_after_fork()
        for service in self._services.values():
            _close_quietly(service)
        self._services.clear()

    def __getstate__(self) -> dict[str, Any]:
        # Spawned workers receive configuration and the factory, never a live
        # HDF5-backed service from the parent process.
        state = dict(self.__dict__)
        state["_services"] = OrderedDict()
        state["_owner_pid"] = os.getpid()
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _adapt_camera_result(
        self,
        result: Any,
        *,
        camera_name: str,
        sample_ref: SampleRef,
        offsets: tuple[int, ...],
        expected_representation: str,
    ) -> dict[str, torch.Tensor]:
        target_count = len(offsets)
        expected_targets = np.asarray(sample_ref.target_indices, dtype=np.int64)
        expected_offsets = np.asarray(offsets, dtype=np.int64)

        if str(_required_attribute(result, "camera_name", owner="target result")) != camera_name:
            raise TargetServiceContractError(
                f"target camera={result.camera_name!r}, requested={camera_name!r}"
            )
        result_source_index = int(
            _required_attribute(result, "source_index", owner="target result")
        )
        if result_source_index != sample_ref.source_index:
            raise TargetServiceContractError("target source_index does not match SampleRef")
        result_targets = np.asarray(
            _required_attribute(result, "target_indices", owner="target result")
        )
        result_offsets = np.asarray(
            _required_attribute(result, "frame_offsets", owner="target result")
        )
        if not np.array_equal(result_targets, expected_targets):
            raise TargetServiceContractError(
                "target_indices do not match SampleRef or preserve requested order"
            )
        if not np.array_equal(result_offsets, expected_offsets):
            raise TargetServiceContractError(
                "frame_offsets do not match target_indices - source_index"
            )
        result_grid = tuple(
            int(value)
            for value in _required_attribute(result, "grid_size", owner="target result")
        )
        if result_grid != self.grid_size:
            raise TargetServiceContractError(
                f"target grid_size={result_grid}, requested={self.grid_size}"
            )

        height, width = self.grid_size
        momap_shape = (target_count, 3, height, width)
        mask_shape = (target_count, 1, height, width)
        momap = np.asarray(_required_attribute(result, "momap", owner="target result"))
        if tuple(momap.shape) != momap_shape:
            raise TargetServiceContractError(
                f"momap shape={tuple(momap.shape)}, expected={momap_shape}"
            )
        if not np.issubdtype(momap.dtype, np.floating):
            raise TargetServiceContractError(
                f"momap must use floating dtype, got {momap.dtype}"
            )
        actual_representation = getattr(result, "representation", None)
        if (
            actual_representation is not None
            and str(actual_representation) != expected_representation
        ):
            raise TargetServiceContractError(
                "target result representation mismatch: "
                f"got {actual_representation!r}, "
                f"expected {expected_representation!r}"
            )

        masks = {
            "geom_valid": _numpy_mask(
                _required_attribute(result, "geom_valid", owner="target result"),
                "geom_valid",
                mask_shape,
            ),
            # These two masks are deliberately required. Older target-service
            # revisions computed them but dropped them from their public
            # result; silently replacing them with zeros would erase unknown
            # geometry from the supervision contract.
            "unsupported": _numpy_mask(
                _required_attribute(result, "unsupported", owner="target result"),
                "unsupported",
                mask_shape,
            ),
            "unmapped": _numpy_mask(
                _required_attribute(result, "unmapped", owner="target result"),
                "unmapped",
                mask_shape,
            ),
        }
        visibility = _required_attribute(result, "visibility", owner="target result")
        for output_name, service_name in (
            ("visibility_evaluated", "evaluated"),
            ("visible", "visible"),
            ("occluded", "occluded"),
            ("out_of_frame", "out_of_frame"),
            ("depth_conflict", "depth_conflict"),
            ("sampling_ambiguous", "sampling_ambiguous"),
        ):
            masks[output_name] = _numpy_mask(
                _required_attribute(
                    visibility, service_name, owner="target result.visibility"
                ),
                output_name,
                mask_shape,
            )

        evaluated = masks["visibility_evaluated"]
        classified = (
            masks["visible"]
            | masks["occluded"]
            | masks["out_of_frame"]
            | masks["depth_conflict"]
        )
        if np.any(classified & ~evaluated):
            raise TargetServiceContractError(
                "visibility classification is true where visibility_evaluated is false"
            )
        class_count = (
            masks["visible"].astype(np.uint8)
            + masks["occluded"].astype(np.uint8)
            + masks["out_of_frame"].astype(np.uint8)
            + masks["depth_conflict"].astype(np.uint8)
        )
        if np.any(class_count > 1):
            raise TargetServiceContractError(
                "visibility classes visible/occluded/out_of_frame/depth_conflict overlap"
            )
        if np.any(evaluated & ~classified):
            raise TargetServiceContractError(
                "evaluated visibility has no explicit classification"
            )
        if np.any(masks["sampling_ambiguous"] & evaluated):
            raise TargetServiceContractError(
                "sampling_ambiguous points must remain visibility-unevaluated"
            )
        if np.any(evaluated & ~masks["geom_valid"]):
            raise TargetServiceContractError(
                "visibility cannot be evaluated where geometry is invalid"
            )
        if np.any(masks["unsupported"] & masks["unmapped"]):
            raise TargetServiceContractError("unsupported and unmapped masks overlap")
        if np.any(
            masks["geom_valid"] & (masks["unsupported"] | masks["unmapped"])
        ):
            raise TargetServiceContractError(
                "geom_valid overlaps unsupported or unmapped geometry"
            )
        if not self.evaluate_visibility and (
            np.any(evaluated)
            or np.any(classified)
            or np.any(masks["sampling_ambiguous"])
        ):
            raise TargetServiceContractError(
                "visibility masks are populated although evaluate_visibility=False"
            )

        dt_seconds = np.asarray(
            _required_attribute(result, "dt_seconds", owner="target result"),
            dtype=np.float64,
        )
        if tuple(dt_seconds.shape) != (target_count,):
            raise TargetServiceContractError(
                f"dt_seconds shape={tuple(dt_seconds.shape)}, expected={(target_count,)}"
            )
        if not np.isfinite(dt_seconds).all() or np.any(dt_seconds < 0):
            raise TargetServiceContractError(
                "dt_seconds must be finite and non-negative for future offsets"
            )

        adapted = {
            "momap": torch.from_numpy(
                np.array(momap, dtype=self.output_dtype, order="C", copy=True)
            ),
            "dt_seconds": torch.from_numpy(
                np.array(dt_seconds, dtype=np.float32, order="C", copy=True)
            ),
        }
        raw_component_valid = getattr(result, "component_valid", None)
        if raw_component_valid is None:
            if expected_representation != MOMAP_REPRESENTATION_CANONICAL_DXYZ:
                raise TargetServiceContractError(
                    "image-flow target result must expose component_valid"
                )
            # Exact legacy compatibility: canonical XYZ has three supervised
            # components wherever the historical geometry mask is valid.
            component_valid = np.broadcast_to(
                masks["geom_valid"], momap_shape
            ).copy()
        else:
            component_valid = _numpy_mask(
                raw_component_valid,
                "component_valid",
                momap_shape,
            )
        if np.any(component_valid & ~np.broadcast_to(
            masks["geom_valid"], momap_shape
        )):
            raise TargetServiceContractError(
                "component_valid must be a subset of geom_valid"
            )
        adapted["component_valid"] = _torch_mask(component_valid)
        if self.wrist_flow_max_magnitude_native_px is not None or self.wrist_flow_gt_background_filter:
            loss_valid = np.array(component_valid, dtype=np.bool_, copy=True)
            if (
                expected_representation
                == MOMAP_REPRESENTATION_SOURCE_ALIGNED_IMAGE_FLOW_PX
                and camera_name != self.reference_camera_name
            ):
                if self.wrist_flow_gt_background_filter:
                    background = _numpy_mask(
                        _required_attribute(result, "source_gt_background", owner="target service"),
                        "source_gt_background", self.grid_size,
                    )
                    gate = ~background[None] & np.isfinite(momap[:, :2]).all(axis=1)
                else:
                    with np.errstate(invalid="ignore", over="ignore"):
                        magnitude = np.hypot(momap[:, 0], momap[:, 1])
                    gate = np.isfinite(magnitude) & (magnitude <= self.wrist_flow_max_magnitude_native_px)
                loss_valid[:, 0] &= gate
                loss_valid[:, 1] &= gate
            adapted["loss_valid"] = _torch_mask(loss_valid)
        adapted.update({name: _torch_mask(mask) for name, mask in masks.items()})
        return adapted

    def build_target(
        self,
        sample_ref: SampleRef,
        geometry: PrivateGeometry,
    ) -> MoMapTarget:
        if not isinstance(sample_ref, SampleRef):
            raise TypeError("sample_ref must be a SampleRef")
        if not isinstance(geometry, PrivateGeometry):
            raise TypeError("geometry must be PrivateGeometry")
        expected_coordinate_frame = f"{self.reference_camera_name}@source_t0"
        if sample_ref.coordinate_frame != expected_coordinate_frame:
            raise ValueError(
                "V1 CPU targets require one common source-time camera frame "
                f"{expected_coordinate_frame!r}; "
                f"got {sample_ref.coordinate_frame!r}"
            )
        if self.reference_camera_name not in sample_ref.camera_names:
            raise ValueError(
                "reference_camera_name must be present in SampleRef.camera_names"
            )

        payload = geometry.payload
        actual_keys = set(payload)
        if actual_keys != _PRIVATE_ADDRESS_KEYS:
            raise TargetServiceContractError(
                "PrivateGeometry must contain only the opaque V1 address: "
                f"missing={sorted(_PRIVATE_ADDRESS_KEYS - actual_keys)}, "
                f"unexpected={sorted(actual_keys - _PRIVATE_ADDRESS_KEYS)}"
            )
        expected_address = {
            "episode_path": os.fspath(
                Path(sample_ref.episode_path).expanduser().resolve()
            ),
            "source_index": sample_ref.source_index,
            "target_indices": tuple(sample_ref.target_indices),
            "camera_names": tuple(sample_ref.camera_names),
            "coordinate_frame": sample_ref.coordinate_frame,
        }
        actual_address = {
            "episode_path": os.fspath(
                Path(payload["episode_path"]).expanduser().resolve()
            ),
            "source_index": payload["source_index"],
            "target_indices": tuple(payload["target_indices"]),
            "camera_names": tuple(payload["camera_names"]),
            "coordinate_frame": payload["coordinate_frame"],
        }
        if actual_address != expected_address:
            mismatches = sorted(
                key
                for key in _PRIVATE_ADDRESS_KEYS
                if actual_address[key] != expected_address[key]
            )
            raise TargetServiceContractError(
                f"PrivateGeometry address disagrees with SampleRef: {mismatches}"
            )

        offsets = tuple(
            target_index - sample_ref.source_index
            for target_index in sample_ref.target_indices
        )
        if any(offset <= 0 for offset in offsets):
            raise ValueError(
                "V1 training targets must be strictly future frames; "
                f"got offsets={offsets}"
            )

        episode_path = actual_address["episode_path"]
        del geometry
        service = self._service_for(episode_path)
        per_view: list[dict[str, torch.Tensor]] = []
        for camera_name in sample_ref.camera_names:
            service_representation = MOMAP_REPRESENTATION_CANONICAL_DXYZ
            if (
                self.target_representation
                == TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV
            ):
                service_representation = (
                    MOMAP_REPRESENTATION_SOURCE_ALIGNED_IMAGE_FLOW_PX
                )
            service_kwargs = {
                "dtype": self.output_dtype,
                "evaluate_visibility": self.evaluate_visibility,
                "depth_tolerance_m": self.depth_tolerance_m,
                "pixel_boundary_ambiguity_px": (
                    self.pixel_boundary_ambiguity_px
                ),
                "reference_camera_name": self.reference_camera_name,
            }
            # Avoid breaking injected legacy service fakes on the unchanged
            # all-XYZ path.  Hybrid mode is explicit and requires the new API.
            if (
                self.target_representation
                == TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV
            ):
                service_kwargs["representation"] = service_representation
            result = service.get_momap(
                camera_name,
                sample_ref.source_index,
                offsets,
                self.grid_size,
                **service_kwargs,
            )
            adapted = self._adapt_camera_result(
                result, camera_name=camera_name, sample_ref=sample_ref,
                offsets=offsets, expected_representation=service_representation,
            )
            if self.target_representation == TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV:
                # Retain the paired 2D run's point eligibility and native-pixel
                # wrist gate, but supervise metric displacement on all XYZ axes.
                xyz_kwargs = dict(service_kwargs)
                xyz_kwargs["representation"] = MOMAP_REPRESENTATION_CANONICAL_DXYZ
                xyz_result = service.get_momap(
                    camera_name, sample_ref.source_index, offsets,
                    self.grid_size, **xyz_kwargs,
                )
                xyz = self._adapt_camera_result(
                    xyz_result, camera_name=camera_name, sample_ref=sample_ref,
                    offsets=offsets,
                    expected_representation=MOMAP_REPRESENTATION_CANONICAL_DXYZ,
                )
                uv_valid = adapted["component_valid"][:, :2].all(dim=1, keepdim=True)
                xyz_valid = xyz["component_valid"].all(dim=1, keepdim=True)
                if bool((uv_valid & ~xyz_valid).any()):
                    raise TargetServiceContractError("UV-valid point lacks canonical XYZ target")
                xyz["component_valid"] = uv_valid.expand_as(xyz["momap"]).clone()
                if "loss_valid" in adapted:
                    point_loss = adapted["loss_valid"][:, :2].all(dim=1, keepdim=True)
                    xyz["loss_valid"] = point_loss.expand_as(xyz["momap"]).clone()
                adapted = xyz
            per_view.append(adapted)

        reference_dt = per_view[0]["dt_seconds"]
        for camera_name, view in zip(sample_ref.camera_names[1:], per_view[1:]):
            if not torch.equal(view["dt_seconds"], reference_dt):
                raise TargetServiceContractError(
                    f"dt_seconds differs between cameras; mismatch at {camera_name!r}"
                )

        def stack(name: str) -> torch.Tensor:
            return torch.stack([view[name] for view in per_view], dim=0)

        return MoMapTarget(
            momap=stack("momap"),
            geom_valid=stack("geom_valid"),
            visibility_evaluated=stack("visibility_evaluated"),
            visible=stack("visible"),
            occluded=stack("occluded"),
            out_of_frame=stack("out_of_frame"),
            depth_conflict=stack("depth_conflict"),
            unsupported=stack("unsupported"),
            unmapped=stack("unmapped"),
            sampling_ambiguous=stack("sampling_ambiguous"),
            dt_seconds=reference_dt,
            component_valid=(
                stack("component_valid")
                if self.target_representation
                == TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV
                else None
            ),
            loss_valid=(
                stack("loss_valid")
                if self.wrist_flow_max_magnitude_native_px is not None or self.wrist_flow_gt_background_filter
                else None
            ),
        )


__all__ = [
    "TARGET_REPRESENTATION_CANONICAL_DXYZ_ALL_VIEWS",
    "TARGET_REPRESENTATION_HEAD_DXYZ_WRIST_DUV",
    "TARGET_REPRESENTATIONS",
    "TargetServiceContractError",
    "V1CPUMoMapTargetBuilder",
]
