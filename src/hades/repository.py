"""Persist playlists in the library database (tables ``playlists`` and ``playlist_tracks``)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from hades.db import now_iso
from hades.playlist import UNTITLED, Entry, Playlist
from hades.query import load_tracks
from hades.spec import PlaylistSpec, SpecError, parse_spec


class RepositoryError(Exception):
    """A playlist could not be saved or loaded. Nothing was changed."""


@dataclass(frozen=True)
class PlaylistSummary:
    id: int
    name: str
    track_count: int
    updated_at: str
    generation_status: str


def save_playlist(conn: sqlite3.Connection, playlist: Playlist) -> None:
    """Insert or update the playlist and its order in one transaction."""
    if not playlist.named and playlist.name == UNTITLED:
        raise RepositoryError("name the playlist first, e.g. `name Friday Warmup`")
    now = now_iso()
    params = playlist.spec.model_dump_json() if playlist.spec is not None else None
    try:
        with conn:
            if playlist.id is None or conn.execute(
                    "SELECT 1 FROM playlists WHERE id = ?", (playlist.id,)).fetchone() is None:
                cur = conn.execute(
                    """INSERT INTO playlists (name, created_at, updated_at, original_prompt,
                                              generation_parameters, generation_status)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (playlist.name, now, now, playlist.prompt, params, playlist.status))
                playlist_id = int(cur.lastrowid)  # type: ignore[arg-type]
            else:
                playlist_id = playlist.id
                conn.execute(
                    """UPDATE playlists SET name = ?, updated_at = ?, original_prompt = ?,
                              generation_parameters = ?, generation_status = ?
                       WHERE id = ?""",
                    (playlist.name, now, playlist.prompt, params, playlist.status, playlist_id))
            conn.execute("DELETE FROM playlist_tracks WHERE playlist_id = ?", (playlist_id,))
            conn.executemany(
                "INSERT INTO playlist_tracks (playlist_id, track_id, position, added_at) "
                "VALUES (?, ?, ?, ?)",
                [(playlist_id, e.track.id, pos, e.added_at)
                 for pos, e in enumerate(playlist.entries, start=1)])
    except sqlite3.IntegrityError as exc:
        if "playlists.name" in str(exc):
            raise RepositoryError(f"a playlist named {playlist.name!r} already exists") from None
        raise RepositoryError(f"could not save the playlist: {exc}") from None
    playlist.id = playlist_id
    playlist.dirty = False
    playlist.saved_name = playlist.name


def list_playlists(conn: sqlite3.Connection) -> list[PlaylistSummary]:
    rows = conn.execute(
        """SELECT p.id, p.name, p.updated_at, p.generation_status,
                  (SELECT COUNT(*) FROM playlist_tracks pt WHERE pt.playlist_id = p.id) AS n
           FROM playlists p ORDER BY p.name COLLATE NOCASE""").fetchall()
    return [PlaylistSummary(r["id"], r["name"], r["n"], r["updated_at"], r["generation_status"])
            for r in rows]


def load_playlist(conn: sqlite3.Connection, name: str) -> Playlist:
    row = conn.execute("SELECT * FROM playlists WHERE name = ?", (name.strip(),)).fetchone()
    if row is None:
        raise RepositoryError(f"no saved playlist named {name.strip()!r} (see `playlists`)")
    items = conn.execute(
        "SELECT track_id, added_at FROM playlist_tracks WHERE playlist_id = ? ORDER BY position",
        (row["id"],)).fetchall()
    tracks = load_tracks(conn, [r["track_id"] for r in items])
    spec: PlaylistSpec | None = None
    if row["generation_parameters"]:
        try:
            spec = parse_spec(json.loads(row["generation_parameters"]))
        except (SpecError, json.JSONDecodeError):
            spec = None  # parameters are kept for reference only; an old format is not fatal
    return Playlist(
        name=row["name"], id=row["id"], prompt=row["original_prompt"], spec=spec,
        status=row["generation_status"], named=True, dirty=False, saved_name=row["name"],
        entries=[Entry(tracks[r["track_id"]], r["added_at"]) for r in items],
    )
