"""Separate code location from existing run, policy and weight assets.

Legacy optional replay helpers use this root. Normal training uses explicit
paths in data/train configs; no original cluster paths.json is required.
"""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]


def asset_root():
    raw = os.environ.get("MWAM_ASSET_ROOT", str(CODE_ROOT))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = CODE_ROOT / path
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise NotADirectoryError(path)
    return path
