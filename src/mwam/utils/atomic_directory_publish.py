"""Same-filesystem, create-only publication of a fully built directory."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path


def publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Publish ``source`` atomically while refusing an existing destination.

    Linux ``renameat2(RENAME_NOREPLACE)`` is preferred.  GPFS can reject that
    flag with ``EINVAL``; in that case a deterministic O_EXCL claim serializes
    cooperating publishers and a final adjacent existence check precedes the
    ordinary same-filesystem directory rename.
    """

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source.parent != destination.parent:
        raise ValueError("Pending and final directories must share one parent")
    if not source.is_dir() or source.is_symlink():
        raise ValueError("Pending publication source must be one real directory")
    claim = destination.parent / f".{destination.name}.publish.lock"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(claim, flags, 0o600)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to replace existing output: {destination}"
            )
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                1,
            )
            if result == 0:
                return
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(
                    f"Refusing to replace existing output: {destination}"
                )
            unsupported = {errno.EINVAL, errno.ENOSYS}
            if hasattr(errno, "EOPNOTSUPP"):
                unsupported.add(errno.EOPNOTSUPP)
            if error not in unsupported:
                raise OSError(error, os.strerror(error), str(destination))
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to replace existing output: {destination}"
            )
        os.rename(source, destination)
    finally:
        os.close(descriptor)
        try:
            claim.unlink()
        except FileNotFoundError:
            pass


__all__ = ["publish_directory_noreplace"]
