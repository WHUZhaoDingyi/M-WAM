"""Hash-bound, fail-closed authorization for canonical production geometry.

Generic readers deliberately support experimental schemas for validation and
visualization.  Production publication must not infer authorization from that
readability.  This module binds the checked-in schema decision by SHA-256 and
also validates the physical HDF5 layout, so changing only ``schema_version``
on a v1/v2/uint16 candidate cannot turn it into a production artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from .raw_episode_hdf5 import RAW_EPISODE_SCHEMA_VERSION


from mwam.utils.paths import asset_root

POLICY_SCHEMA_VERSION = 2
POLICY_FILE = (
    asset_root()
    / "configs/control/robotwin_momap_production_schema_policy.json"
)
# Updated only together with the reviewed policy file.  Runtime loading fails
# closed if either file changes independently.
POLICY_FILE_SHA256 = (
    "f995cb8f54c15c8cfe3cd3ba4cb0853347daa850c021db401c8c03cf558c290e"
)
PRODUCTION_GEOMETRY_ENCODING = "position_xyz_float32_v0"
FORBIDDEN_CANDIDATE_FIELDS = frozenset(
    {
        "depth_m",
        "ray_xy",
        "ray_valid",
        "position_uint16",
        "position_xy_xor",
        "position_z_xor",
    }
)


HDF5_STORAGE_CONTRACT = {
    "allowed_filter_id_sequences": [[], [2, 1]],
    "position_filter_id_sequence": [2, 1],
    "shuffle_filter_id": 2,
    "deflate_filter_id": 1,
    "gzip_level": 1,
    "position_chunks": "[1,H,W,3]",
    "lossy_scaleoffset_allowed": False,
    "unknown_filters_allowed": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_production_schema_policy() -> dict[str, Any]:
    """Load the immutable decision document and verify its exact bytes."""

    actual_sha256 = _sha256(POLICY_FILE)
    if actual_sha256 != POLICY_FILE_SHA256:
        raise RuntimeError(
            "production schema policy SHA-256 differs: "
            f"expected={POLICY_FILE_SHA256} actual={actual_sha256}"
        )
    value = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != (
        POLICY_SCHEMA_VERSION
    ):
        raise RuntimeError("production schema policy version differs")
    authorized = value.get("production_authorized")
    if not isinstance(authorized, dict) or authorized != {
        "raw_episode_schema_version": RAW_EPISODE_SCHEMA_VERSION,
        "geometry_encoding": PRODUCTION_GEOMETRY_ENCODING,
        "position_dataset": "position_xyz",
        "position_dtype": "float32",
    }:
        raise RuntimeError("production schema policy authorization differs")
    denied = value.get("read_only_denied_schemas")
    if not isinstance(denied, list) or RAW_EPISODE_SCHEMA_VERSION in denied:
        raise RuntimeError("production schema policy deny-list is invalid")
    if set(value.get("forbidden_physical_position_fields", ())) != set(
        FORBIDDEN_CANDIDATE_FIELDS
    ):
        raise RuntimeError("production schema policy physical deny-list differs")
    if value.get("hdf5_storage_contract") != HDF5_STORAGE_CONTRACT:
        raise RuntimeError("production schema policy HDF5 storage contract differs")
    return value


def _require_canonical_filter_pipeline(
    dataset: h5py.Dataset, *, label: str
) -> None:
    """Reject lossy or plugin-defined storage hidden behind a float32 dtype."""

    creation = dataset.id.get_create_plist()
    filter_ids = tuple(
        int(creation.get_filter(index)[0])
        for index in range(int(creation.get_nfilters()))
    )
    if dataset.scaleoffset is not None:
        raise ValueError(
            f"production {label} must not use lossy HDF5 scale-offset"
        )
    if filter_ids not in ((), (2, 1)):
        raise ValueError(
            f"production {label} uses unsupported HDF5 filter pipeline "
            f"{filter_ids}; only unfiltered or shuffle+gzip are allowed"
        )
    if filter_ids == ():
        if dataset.compression is not None or dataset.shuffle:
            raise ValueError(
                f"production {label} unfiltered storage metadata differs"
            )
        return
    if (
        dataset.compression != "gzip"
        or int(dataset.compression_opts) != 1
        or dataset.shuffle is not True
    ):
        raise ValueError(
            f"production {label} must use canonical shuffle+gzip level 1"
        )


def _require_canonical_position_filter_pipeline(
    dataset: h5py.Dataset, *, label: str
) -> None:
    """Require the exact lossless filter pipeline emitted by the v0 writer.

    Some small metadata/state datasets are intentionally unfiltered, so the
    recursive file-wide gate permits either no filters or shuffle+gzip1. Raw
    Position dominates dataset capacity and has a stricter canonical contract:
    it must always use HDF5's actual filter order ``shuffle -> deflate`` with
    gzip level 1. A chunked-but-uncompressed FP32 Position is numerically v0,
    but accepting it would silently invalidate the production capacity model.
    """

    creation = dataset.id.get_create_plist()
    filter_ids = tuple(
        int(creation.get_filter(index)[0])
        for index in range(int(creation.get_nfilters()))
    )
    if (
        filter_ids != (2, 1)
        or dataset.scaleoffset is not None
        or dataset.compression != "gzip"
        or int(dataset.compression_opts) != 1
        or dataset.shuffle is not True
    ):
        raise ValueError(
            f"production {label} must use exact shuffle+gzip level 1 filter "
            f"pipeline (2, 1), got filters={filter_ids}, "
            f"compression={dataset.compression!r}, "
            f"compression_opts={dataset.compression_opts!r}, "
            f"shuffle={dataset.shuffle!r}"
        )


def _require_hard_link(parent: h5py.Group, name: str, *, label: str) -> Any:
    """Reject HDF5 ExternalLink/SoftLink indirection for production fields."""

    link = parent.get(name, getlink=True)
    if not isinstance(link, h5py.HardLink):
        raise ValueError(f"production {label} must be an HDF5 HardLink")
    value = parent.get(name, getlink=False)
    if value is None:
        raise ValueError(f"production {label} is missing")
    if isinstance(value, h5py.Dataset):
        if bool(value.is_virtual):
            raise ValueError(f"production {label} must not be a virtual dataset")
        external_count = int(value.id.get_create_plist().get_external_count())
        if external_count:
            raise ValueError(f"production {label} must not use external storage")
        _require_canonical_filter_pipeline(value, label=label)
    return value


def _require_recursive_hard_links(handle: h5py.File) -> None:
    """Require every named object in the production HDF5 to be in-file.

    Checking only the datasets needed by the geometry gate is insufficient:
    an ExternalLink in timeline/control/entity metadata can change without
    changing the canonical HDF5 file hash.  Production therefore has no
    SoftLink/ExternalLink allow-list.  Object addresses prevent hard-link
    aliases (including cycles) from causing unbounded recursion.
    """

    visited_groups: set[int] = set()

    def visit(group: h5py.Group, group_path: str) -> None:
        address = int(h5py.h5o.get_info(group.id).addr)
        if address in visited_groups:
            return
        visited_groups.add(address)
        for name in group.keys():
            child_path = f"{group_path}/{name}" if group_path else f"/{name}"
            value = _require_hard_link(group, name, label=child_path)
            if isinstance(value, h5py.Group):
                visit(value, child_path)

    visit(handle, "")


def require_production_raw_episode_handle(
    handle: h5py.File,
    *,
    expected_camera_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Require an actual v0 FP32-XYZ layout, not merely a v0 root attribute."""

    policy = load_production_schema_policy()
    schema = str(handle.attrs.get("schema_version", ""))
    if schema != RAW_EPISODE_SCHEMA_VERSION:
        raise ValueError(
            "production raw episode requires raw_episode_v0; "
            f"candidate schema {schema!r} is read-only"
        )
    encoding = str(
        handle.attrs.get("geometry_encoding", PRODUCTION_GEOMETRY_ENCODING)
    )
    if encoding != PRODUCTION_GEOMETRY_ENCODING:
        raise ValueError(
            "production raw episode geometry encoding must be FP32 XYZ; "
            f"got {encoding!r}"
        )
    if bool(int(handle.attrs.get("candidate_only", 0))):
        raise ValueError("candidate_only HDF5 cannot enter production")
    if "production_authorized" in handle.attrs and not bool(
        int(handle.attrs["production_authorized"])
    ):
        raise ValueError("production_authorized=false HDF5 cannot enter production")
    if "converted_from_schema" in handle.attrs:
        raise ValueError("converted candidate HDF5 cannot enter production")

    status = str(handle.attrs.get("status", ""))
    frame_count = int(handle.attrs.get("num_frames", -1))
    if status != "complete" or frame_count <= 0:
        raise ValueError(
            f"production raw episode is incomplete: status={status!r}, "
            f"num_frames={frame_count}"
        )
    _require_recursive_hard_links(handle)
    observations = _require_hard_link(handle, "observations", label="observations")
    private_geometry = _require_hard_link(
        handle, "private_geometry", label="private_geometry"
    )
    if not isinstance(observations, h5py.Group) or not isinstance(
        private_geometry, h5py.Group
    ):
        raise ValueError("production raw episode camera roots must be groups")
    observation_names = tuple(observations.keys())
    geometry_names = tuple(private_geometry.keys())
    if set(observation_names) != set(geometry_names) or not geometry_names:
        raise ValueError("production observation/private camera sets differ or are empty")
    if expected_camera_names is not None:
        expected = tuple(str(name) for name in expected_camera_names)
        if geometry_names != expected or observation_names != expected:
            raise ValueError(
                "production camera set/order differs: "
                f"expected={expected!r} actual={geometry_names!r}"
            )

    camera_reports: dict[str, Any] = {}
    for camera_name in geometry_names:
        geometry = _require_hard_link(
            private_geometry, camera_name, label=f"private_geometry/{camera_name}"
        )
        observation = _require_hard_link(
            observations, camera_name, label=f"observations/{camera_name}"
        )
        if not isinstance(geometry, h5py.Group) or not isinstance(
            observation, h5py.Group
        ):
            raise ValueError(f"production camera {camera_name!r} roots must be groups")
        forbidden = sorted(FORBIDDEN_CANDIDATE_FIELDS & set(geometry.keys()))
        if forbidden:
            raise ValueError(
                f"production camera {camera_name!r} contains candidate fields: "
                f"{forbidden}"
            )
        required = {
            "position_xyz",
            "position_valid",
            "actor_seg_raw",
            "K",
            "T_cam_to_world",
        }
        missing = sorted(required - set(geometry.keys()))
        if missing:
            raise ValueError(
                f"production camera {camera_name!r} lacks v0 fields: {missing}"
            )
        position = _require_hard_link(
            geometry, "position_xyz", label=f"{camera_name}.position_xyz"
        )
        expected_position_shape = (
            frame_count,
            int(position.shape[1]) if position.ndim == 4 else -1,
            int(position.shape[2]) if position.ndim == 4 else -1,
            3,
        )
        if (
            position.ndim != 4
            or tuple(position.shape) != expected_position_shape
            or position.dtype != np.dtype("float32")
        ):
            raise ValueError(
                f"production {camera_name}.position_xyz must be "
                f"float32[T,H,W,3], got {position.shape} {position.dtype}"
            )
        height, width = int(position.shape[1]), int(position.shape[2])
        if position.chunks != (1, height, width, 3):
            raise ValueError(
                f"production {camera_name}.position_xyz chunks must be "
                f"(1,H,W,3), got {position.chunks}"
            )
        _require_canonical_position_filter_pipeline(
            position, label=f"{camera_name}.position_xyz"
        )
        valid = _require_hard_link(
            geometry, "position_valid", label=f"{camera_name}.position_valid"
        )
        if valid.shape != (frame_count, height, width) or valid.dtype != np.dtype(
            "uint8"
        ):
            raise ValueError(
                f"production {camera_name}.position_valid layout differs"
            )
        actor_seg = _require_hard_link(
            geometry, "actor_seg_raw", label=f"{camera_name}.actor_seg_raw"
        )
        if actor_seg.shape != valid.shape or actor_seg.dtype not in {
            np.dtype("uint16"),
            np.dtype("uint32"),
        }:
            raise ValueError(
                f"production {camera_name}.actor_seg_raw layout differs"
            )
        intrinsic = _require_hard_link(geometry, "K", label=f"{camera_name}.K")
        if intrinsic.shape != (frame_count, 3, 3) or intrinsic.dtype != np.dtype(
            "float32"
        ):
            raise ValueError(f"production {camera_name}.K layout differs")
        camera_pose = _require_hard_link(
            geometry, "T_cam_to_world", label=f"{camera_name}.T_cam_to_world"
        )
        if camera_pose.shape != (frame_count, 4, 4) or camera_pose.dtype != np.dtype(
            "float32"
        ):
            raise ValueError(
                f"production {camera_name}.T_cam_to_world layout differs"
            )
        rgb = _require_hard_link(observation, "rgb", label=f"{camera_name}.rgb")
        if (
            rgb.shape != (frame_count, height, width, 3)
            or rgb.dtype != np.dtype("uint8")
        ):
            raise ValueError(f"production {camera_name}.rgb layout differs")
        camera_reports[camera_name] = {
            "position_shape": list(position.shape),
            "position_dtype": position.dtype.name,
            "position_valid_shape": list(valid.shape),
        }

    return {
        "passed": True,
        "policy_sha256": POLICY_FILE_SHA256,
        "schema_version": schema,
        "geometry_encoding": encoding,
        "frame_count": frame_count,
        "camera_names": list(geometry_names),
        "cameras": camera_reports,
        "decision": policy["decision"],
    }


def require_production_raw_episode_path(
    path: str | Path,
    *,
    expected_camera_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(
            f"production raw episode must not be a symlink: {candidate}"
        )
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"production raw episode must be a regular file: {resolved}"
        )
    with h5py.File(resolved, "r") as handle:
        return require_production_raw_episode_handle(
            handle, expected_camera_names=expected_camera_names
        )


__all__ = [
    "FORBIDDEN_CANDIDATE_FIELDS",
    "POLICY_FILE",
    "POLICY_FILE_SHA256",
    "PRODUCTION_GEOMETRY_ENCODING",
    "load_production_schema_policy",
    "require_production_raw_episode_handle",
    "require_production_raw_episode_path",
]
