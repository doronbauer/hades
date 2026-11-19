"""Write playlists as extended M3U files that players and DJ software open.

The default extension is ``.m3u``, because Traktor 3 does not open ``.m3u8``;
``.m3u8`` can be chosen instead. The text is UTF-8 either way. Entries are
absolute paths to the indexed files; nothing is copied.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from hades.playlist import Playlist

EXTENSIONS = (".m3u", ".m3u8")
DEFAULT_EXTENSION = ".m3u"


def export_filename(name: str, extension: str = DEFAULT_EXTENSION) -> str:
    """A safe file name for a playlist name: "Friday: Warmup / Peak" -> "Friday- Warmup - Peak.m3u"."""
    safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "-", name)
    safe = " ".join(safe.split()).strip(" .")
    return (safe or "playlist") + extension


def render_m3u(playlist: Playlist) -> str:
    lines = ["#EXTM3U", f"#PLAYLIST:{playlist.name}"]
    for t in playlist.tracks:
        seconds = round(t.duration) if t.duration else -1
        label = f"{t.artist} - {t.title}" if t.artist and t.title else t.display
        lines += [f"#EXTINF:{seconds},{label}", t.path]
    return "\n".join(lines) + "\n"


def export_m3u(playlist: Playlist, target: Path, extension: str = DEFAULT_EXTENSION) -> Path:
    """Write the playlist to ``target`` (a file, or a directory to use the playlist's name).

    The file is replaced atomically, so a player never sees a half-written playlist.
    """
    path = target / export_filename(playlist.name, extension) \
        if target.is_dir() or not target.suffix else target
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".hades-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render_m3u(playlist))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def remove_export(name: str, directory: Path, extension: str = DEFAULT_EXTENSION) -> None:
    """Delete the export of a playlist that was renamed (if it exists)."""
    (directory / export_filename(name, extension)).unlink(missing_ok=True)
