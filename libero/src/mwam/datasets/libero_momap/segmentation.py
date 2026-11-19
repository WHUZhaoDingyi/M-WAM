"""Strict raw-pixel DINO/SAM2 cache contract for LIBERO and LIBERO-Plus.

Caches contain predictions in the original agentview RGB raster. The dataset
applies its existing flip and nearest resize afterwards, together with XYZ/RGB.
No simulator segmentation is substituted when an asset is missing or invalid.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

CACHE_SCHEMA = "mwam.libero-grounded-sam2-seg-cache.v1"
MANIFEST_SCHEMA = "libero_plus_predicted_segmentation_v1"
CAMERA_POLICY = "agentview_only_zero_wrist"
PIXEL_SPACE = "raw_native_rgb"


def instruction_digest(instruction):
    return hashlib.sha256(str(instruction).encode("utf-8")).hexdigest()


def manifest_digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _text(value):
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def _require_flag(value, expected, label):
    if not isinstance(value, (bool, np.bool_)) or bool(value) != expected:
        raise ValueError("Predicted segmentation " + label + " mismatch")


def _required_path(value, label):
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Explicit predicted segmentation " + label + " path required")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError("Missing predicted segmentation " + label + ": " + str(path))
    return path


def _entry(document, row):
    # Manifest keys identify the logical dataset path. A GPU node may mount
    # that path through a directory symlink to an exact shared-disk mirror.
    key = str(Path(row["geometry_path"]).expanduser().absolute())
    if row.get("pixel_transform") not in ("identity", "horizontal_flip"):
        raise ValueError("Predicted segmentation requires the declared LIBERO pixel transform")
    entry = document["episodes"].get(key)
    if not isinstance(entry, dict):
        raise ValueError("Predicted segmentation does not cover episode: " + key)
    if entry.get("frames") != row["frames"]:
        raise ValueError("Predicted segmentation frame count differs: " + key)
    if entry.get("downstream_pixel_transform") != row["pixel_transform"]:
        raise ValueError("Predicted segmentation pixel transform differs: " + key)
    shape = entry.get("native_shape")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape)):
        raise ValueError("Predicted segmentation native shape must be [height,width]")
    _required_path(entry.get("path"), "episode cache")
    return key, entry


def inspect_cache(handle, entry, row, document):
    """Validate provenance and all requested source indices without reading masks."""
    # Keep the same logical identity as _entry; stat/open still follow the
    # host's mount or symlink and validate the actual source file below.
    geometry = _required_path(row.get("storage_geometry_path", row["geometry_path"]), "source geometry").absolute()
    logical_geometry = str(Path(row["geometry_path"]).expanduser().absolute())
    info = geometry.stat()
    expected = {
        "schema_version": CACHE_SCHEMA, "status": "PASS",
        "geometry_path": logical_geometry, "camera_policy": CAMERA_POLICY,
        "pixel_space": PIXEL_SPACE,
        "downstream_pixel_transform": row["pixel_transform"],
        "provenance_digest": document["provenance_digest"],
        "instruction_sha256": instruction_digest(row["instruction"]),
    }
    for key, value in expected.items():
        if _text(handle.attrs.get(key, "")) != value:
            raise ValueError("Predicted segmentation cache provenance differs at " + key)
    for key, value in {"source_file_size": info.st_size,
                       "source_file_mtime_ns": info.st_mtime_ns,
                       "source_frame_count": row["frames"]}.items():
        if handle.attrs.get(key) != value:
            raise ValueError("Predicted segmentation source identity differs at " + key)
    _require_flag(handle.attrs.get("causal"), True, "causal cache")
    _require_flag(handle.attrs.get("uses_simulator_segmentation"), False, "simulator-mask usage")
    for key in ("source_index", "camera_name", "pred/track_id"):
        if key not in handle:
            raise ValueError("Predicted segmentation cache is missing " + key)
    indices = np.asarray(handle["source_index"])
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
            or np.any(indices < 0) or np.any(indices >= row["frames"])
            or np.any(np.diff(indices.astype(np.int64)) <= 0)):
        raise ValueError("Predicted segmentation source indices must be sorted and unique")
    cameras = [_text(v) for v in np.asarray(handle["camera_name"]).reshape(-1)]
    if cameras != ["agentview"]:
        raise ValueError("Predicted segmentation must contain only agentview")
    tracks = handle["pred/track_id"]
    shape = tuple(entry["native_shape"])
    if tracks.shape != (len(indices), 1, *shape) or tracks.dtype.kind not in "iu":
        raise ValueError("Predicted segmentation track raster shape/dtype mismatch")
    with h5py.File(geometry, "r") as source:
        rgb = source["observations/agentview/rgb"]
        if rgb.shape != (row["frames"], *shape, 3):
            raise ValueError("Predicted segmentation native raster differs from original RGB")
    required = np.asarray(row["source_indices"], dtype=np.int64)
    if required.ndim != 1 or np.any(required < 0) or np.any(required >= row["frames"]):
        raise ValueError("Invalid requested segmentation source indices")
    if len(np.setdiff1d(required, indices)):
        raise ValueError("Predicted segmentation does not cover all training source frames")
    return {int(source): i for i, source in enumerate(indices)}


def validate_predicted_segmentation(records, manifest_path, *, check_files=True):
    path = _required_path(manifest_path, "manifest")
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError("Predicted segmentation manifest must be an object")
    for key, value in {"schema_version": MANIFEST_SCHEMA, "source": "predicted",
                       "camera_policy": CAMERA_POLICY, "pixel_space": PIXEL_SPACE}.items():
        if document.get(key) != value:
            raise ValueError("Predicted segmentation manifest differs at " + key)
    _require_flag(document.get("complete"), True, "manifest completion")
    _require_flag(document.get("causal"), True, "manifest causal policy")
    _require_flag(document.get("uses_simulator_segmentation"), False, "manifest simulator-mask usage")
    digest = document.get("provenance_digest")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("Predicted segmentation requires a pipeline provenance SHA256")
    if not isinstance(document.get("episodes"), dict):
        raise ValueError("Predicted segmentation manifest requires episode records")
    for row in records:
        _, entry = _entry(document, row)
        if check_files:
            with h5py.File(entry["path"], "r") as handle:
                inspect_cache(handle, entry, row, document)
    return document


class PredictedSegmentationStore:
    """Per-worker bounded HDF5 handles, validating each opened episode once."""
    def __init__(self, records, manifest_path, max_open=2):
        self.document = validate_predicted_segmentation(records, manifest_path, check_files=False)
        self.digest = manifest_digest(self.document)
        self.max_open = int(max_open)
        if self.max_open < 1:
            raise ValueError("Positive segmentation cache handle limit required")
        self.handles = OrderedDict()
        self.pid = os.getpid()

    def close(self):
        for handle, _ in self.handles.values():
            handle.close()
        self.handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["handles"] = OrderedDict()
        state["pid"] = None
        return state

    def read(self, row, source_index, *, expected_shape):
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        key, entry = _entry(self.document, row)
        if key not in self.handles:
            handle = h5py.File(entry["path"], "r")
            try:
                indices = inspect_cache(handle, entry, row, self.document)
            except Exception:
                handle.close()
                raise
            self.handles[key] = handle, indices
            while len(self.handles) > self.max_open:
                _, (old, _) = self.handles.popitem(last=False)
                old.close()
        self.handles.move_to_end(key)
        handle, indices = self.handles[key]
        if source_index not in indices:
            raise ValueError("Predicted segmentation is missing requested source frame")
        track = np.asarray(handle["pred/track_id"][indices[source_index], 0])
        if tuple(track.shape) != tuple(expected_shape) or np.any(track < 0):
            raise ValueError("Predicted segmentation does not match the current raw RGB raster")
        return track
