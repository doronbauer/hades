"""Stage 1: find audio files under the configured directories.

This module only reads the filesystem; it knows nothing about the database.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass

SUPPORTED_FORMATS: dict[str, str] = {
    ".mp3": "MP3",
    ".aif": "AIFF",
    ".aiff": "AIFF",
    ".wav": "WAV",
    ".flac": "FLAC",
    ".m4a": "M4A",
}


@dataclass(frozen=True)
class DiscoveredFile:
    path: str
    format: str
    size: int
    mtime_ns: int
    created: float | None  # st_birthtime where the platform provides it
    device: int
    inode: int

    @property
    def signature(self) -> str:
        return f"{self.size}:{self.mtime_ns}"


def audio_format(filename: str) -> str | None:
    return SUPPORTED_FORMATS.get(os.path.splitext(filename)[1].lower())


def check_root(root: str) -> str | None:
    """Return None if ``root`` can be scanned, otherwise the reason it cannot.

    A root that cannot be scanned must not have its tracks marked missing.
    """
    if not os.path.exists(root):
        return "does not exist (drive disconnected?)"
    if not os.path.isdir(root):
        return "is not a directory"
    volume = _macos_volume(root)
    if volume is not None and not os.path.ismount(volume):
        return f"{volume} is not a mounted volume"
    try:
        with os.scandir(root) as entries:
            next(entries, None)
    except OSError as exc:
        return f"cannot be read ({exc.strerror or exc})"
    return None


def _macos_volume(root: str) -> str | None:
    # A leftover directory under /Volumes is not a mounted drive.
    parts = root.split(os.sep)
    if len(parts) >= 3 and parts[0] == "" and parts[1] == "Volumes":
        return os.sep.join(parts[:3])
    return None


def iter_audio_files(root: str, unreadable: list[str]) -> Iterator[DiscoveredFile]:
    """Yield supported audio files under ``root`` recursively, in a stable order.

    Directory symlinks are not followed; hidden files and directories (including
    macOS ``._`` resource-fork files on external drives) are ignored. Paths that
    could not be read are appended to ``unreadable`` so the caller can avoid
    marking tracks beneath them as missing.
    """

    def on_error(err: OSError) -> None:
        if err.filename:
            unreadable.append(os.fspath(err.filename))

    for dirpath, dirnames, filenames in os.walk(root, onerror=on_error, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            fmt = audio_format(name)
            if fmt is None:
                continue
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue  # deleted mid-scan, or a dangling file symlink
            except OSError:
                unreadable.append(path)
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            yield DiscoveredFile(
                path=path,
                format=fmt,
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
                created=getattr(st, "st_birthtime", None),
                device=st.st_dev,
                inode=st.st_ino,
            )


def is_under(path: str, prefixes: list[str]) -> bool:
    for prefix in prefixes:
        prefix = prefix.rstrip(os.sep)
        if path == prefix or path.startswith(prefix + os.sep):
            return True
    return False
