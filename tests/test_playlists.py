"""Playlist editing, command parsing, persistence, migration and LLM refinement."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from conftest import FakeLLM, insert_track
from hades.builder import build_playlist
from hades.commands import Command, CommandError, parse_command
from hades.db import MIGRATIONS, connect, schema_version
from hades.llm import refine_playlist
from hades.playlist import UNTITLED, EditError, Playlist
from hades.query import find_candidates, load_track
from hades.repository import RepositoryError, list_playlists, load_playlist, save_playlist
from hades.spec import parse_spec


@pytest.fixture
def tracks(conn, music_dir: Path):
    return [load_track(conn, insert_track(conn, music_dir, artist=f"A{i}", bpm=120 + i,
                                          duration=300 + i)) for i in range(6)]


def make(tracks, n: int = 4) -> Playlist:
    pl = Playlist()
    for t in tracks[:n]:
        pl.add(t)
    return pl


# --------------------------------------------------------------------------- commands


@pytest.mark.parametrize("text, expected", [
    ("name My Playlist", Command("name", text="My Playlist")),
    ("rename  New Name ", Command("rename", text="New Name")),
    ("move 876 to 4", Command("move", track_id=876, position=4)),
    ("MOVE 876 4", Command("move", track_id=876, position=4)),
    ("remove 876", Command("remove", track_ids=(876,))),
    ("rm 876", Command("remove", track_ids=(876,))),
    ("remove 45 7  879", Command("remove", track_ids=(45, 7, 879))),
    ("add 1042", Command("add", track_ids=(1042,))),
    ("add 1042 at 3", Command("add", track_ids=(1042,), position=3)),
    ("add 1042 7 AT 3", Command("add", track_ids=(1042, 7), position=3)),
    ("add 1042 7", Command("add", track_ids=(1042, 7))),
    ("keep", Command("keep")),
    ("save", Command("save")),
    ("playlists", Command("playlists")),
    ("open My Playlist", Command("open", text="My Playlist")),
    ("Create a one-hour playlist of techno", None),
    ("", None),
])
def test_parse_command(text: str, expected: Command | None) -> None:
    assert parse_command(text) == expected


@pytest.mark.parametrize("text", ["move 876", "move abc to 2", "remove", "add x", "name",
                                  "save now", "remove 4 x", "add 4 at", "keep 4"])
def test_bad_command_arguments(text: str) -> None:
    with pytest.raises(CommandError, match="usage:"):
        parse_command(text)


# --------------------------------------------------------------------------- editing


def test_mark_and_keep(tracks) -> None:
    pl = make(tracks)
    a, b, c, d = (t.id for t in pl.tracks)
    with pytest.raises(EditError, match="no tracks are marked"):
        pl.keep_marked()
    assert pl.toggle_mark(a) and pl.toggle_mark(c) and pl.toggle_mark(d)
    assert not pl.toggle_mark(d)  # a second press unmarks
    with pytest.raises(EditError):
        pl.toggle_mark(999999)
    removed = pl.keep_marked()
    assert [t.id for t in removed] == [b, d]
    assert [t.id for t in pl.tracks] == [a, c] and pl.marked == set()


def test_remove_unmarks(tracks) -> None:
    pl = make(tracks)
    first = pl.tracks[0].id
    pl.toggle_mark(first)
    pl.remove(first)
    assert pl.marked == set()


def test_new_playlist_is_untitled(tracks) -> None:
    pl = Playlist()
    assert pl.name == UNTITLED and len(pl) == 0 and not pl.dirty


def test_move_remove_add_by_id(tracks) -> None:
    pl = make(tracks)
    a, b, c, d, e, _ = (t.id for t in tracks)
    pl.move(c, 1)
    assert [t.id for t in pl.tracks] == [c, a, b, d]
    pl.move(c, 4)
    assert [t.id for t in pl.tracks] == [a, b, d, c]
    assert pl.remove(b).id == b
    assert [t.id for t in pl.tracks] == [a, d, c]
    assert pl.add(tracks[4]) == 4
    assert pl.add(tracks[1], 2) == 2
    assert [t.id for t in pl.tracks] == [a, b, d, c, e]
    assert pl.dirty


@pytest.mark.parametrize("edit, message", [
    (lambda pl, ts: pl.move(9999, 1), "not in the playlist"),
    (lambda pl, ts: pl.move(ts[0].id, 5), "between 1 and 4"),
    (lambda pl, ts: pl.move(ts[0].id, 0), "between 1 and 4"),
    (lambda pl, ts: pl.remove(9999), "not in the playlist"),
    (lambda pl, ts: pl.add(ts[0]), "already in the playlist"),
    (lambda pl, ts: pl.add(ts[5], 6), "between 1 and 5"),
    (lambda pl, ts: pl.rename("   "), "must not be empty"),
])
def test_invalid_edits_leave_playlist_unchanged(tracks, edit, message: str) -> None:
    pl = make(tracks)
    before = [t.id for t in pl.tracks]
    with pytest.raises(EditError, match=message):
        edit(pl, tracks)
    assert [t.id for t in pl.tracks] == before


def test_stats_and_violations_recalculate_after_edits(conn, music_dir: Path, tracks) -> None:
    spec = parse_spec({"bpm": {"max": 123}, "track_count": 3})
    result = build_playlist(spec, find_candidates(conn, spec))
    pl = Playlist.from_result(result, "slow ones")
    assert pl.violations() == [] and pl.stats.count == 3
    total = pl.stats.total_seconds

    fast = tracks[5]  # 125 BPM: allowed as a manual edit, but reported
    pl.add(fast)
    assert pl.stats.count == 4 and pl.stats.total_seconds == total + fast.duration
    assert f"#4 (ID {fast.id}): BPM 125 > 123" in pl.violations()
    assert "4 tracks instead of the requested 3" in pl.violations()
    pl.remove(fast.id)
    assert pl.violations() == []


def test_transitions_follow_edits(tracks) -> None:
    pl = make(tracks)
    pl.spec = parse_spec({"ordering": {"bpm_progression": "ascending"}})
    assert all(not t.issues for t in pl.transitions())
    pl.move(tracks[3].id, 1)
    assert "BPM drop -3" in pl.transitions()[0].issues


# --------------------------------------------------------------------------- persistence


def test_save_requires_a_name(conn, tracks) -> None:
    with pytest.raises(RepositoryError, match="name the playlist first"):
        save_playlist(conn, make(tracks))


def test_save_and_reopen(conn, config, tracks) -> None:
    pl = make(tracks)
    pl.prompt = "warmup"
    pl.spec = parse_spec({"bpm": {"max": 130}})
    pl.status = "complete"
    pl.rename("Friday")
    save_playlist(conn, pl)
    assert pl.id is not None and not pl.dirty

    # A fresh connection, as after an application restart.
    other = connect(config.database_path)
    try:
        loaded = load_playlist(other, "friday")
    finally:
        other.close()
    assert loaded.name == "Friday" and loaded.id == pl.id
    assert [t.id for t in loaded.tracks] == [t.id for t in pl.tracks]
    assert [e.added_at for e in loaded.entries] == [e.added_at for e in pl.entries]
    assert loaded.prompt == "warmup" and loaded.spec == pl.spec and loaded.status == "complete"
    assert not loaded.dirty


def test_resave_rename_and_reorder(conn, tracks) -> None:
    pl = make(tracks)
    pl.rename("One")
    save_playlist(conn, pl)
    pl.move(tracks[3].id, 1)
    pl.remove(tracks[1].id)
    pl.rename("Two")
    save_playlist(conn, pl)
    assert [(p.name, p.track_count) for p in list_playlists(conn)] == [("Two", 3)]
    rows = conn.execute("SELECT track_id, position FROM playlist_tracks ORDER BY position")
    assert [tuple(r) for r in rows] == [(tracks[3].id, 1), (tracks[0].id, 2), (tracks[2].id, 3)]
    with pytest.raises(RepositoryError, match="no saved playlist"):
        load_playlist(conn, "One")


def test_duplicate_names_are_rejected_atomically(conn, tracks) -> None:
    first = make(tracks, 2)
    first.rename("Set")
    save_playlist(conn, first)
    second = make(tracks, 3)
    second.rename("SET")
    with pytest.raises(RepositoryError, match="already exists"):
        save_playlist(conn, second)
    assert second.id is None and second.dirty
    assert conn.execute("SELECT COUNT(*) FROM playlist_tracks").fetchone()[0] == 2


def test_schema_enforces_unique_tracks_and_positions(conn, tracks) -> None:
    pl = make(tracks, 2)
    pl.rename("X")
    save_playlist(conn, pl)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO playlist_tracks VALUES (?, ?, 3, 'now')", (pl.id, tracks[0].id))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO playlist_tracks VALUES (?, ?, 1, 'now')", (pl.id, tracks[4].id))
    conn.rollback()


def test_migration_extends_existing_library(tmp_path: Path) -> None:
    db = tmp_path / "old.sqlite3"
    old = sqlite3.connect(db)
    old.executescript(f"BEGIN;\n{MIGRATIONS[0]}\nPRAGMA user_version = 1;\nCOMMIT;")
    old.execute(
        "INSERT INTO tracks (path, filename, directory, root, format, file_size, file_mtime_ns, "
        "file_modified, content_signature, first_indexed, last_seen, effective_bpm) "
        "VALUES ('/m/a.mp3', 'a.mp3', '/m', '/m', 'MP3', 1, 1, 'x', '1:1', 'x', 'x', 128)")
    old.commit()
    old.close()

    conn = connect(db)
    try:
        assert schema_version(conn) == len(MIGRATIONS) == 2
        row = conn.execute("SELECT path, effective_bpm FROM tracks").fetchone()
        assert tuple(row) == ("/m/a.mp3", 128)
        assert conn.execute("SELECT COUNT(*) FROM playlists").fetchone()[0] == 0
    finally:
        conn.close()


# --------------------------------------------------------------------------- refinement


@pytest.fixture
def generated(conn, music_dir: Path):
    for i in range(10):
        insert_track(conn, music_dir, artist=f"A{i}", bpm=120 + i, camelot=f"{1 + i}A")
    spec = parse_spec({"track_count": 4, "bpm": {"max": 127}})
    pool = find_candidates(conn, spec)
    return build_playlist(spec, pool), pool


def test_refinement_is_accepted_when_valid(generated, music_dir: Path) -> None:
    result, pool = generated
    proposal = [t.id for t in result.tracks]  # "already good": an equal score is accepted
    llm = FakeLLM({"track_ids": proposal, "rationale": "keep"})
    outcome = refine_playlist(result, pool.tracks, llm, "four slow ones")
    assert outcome.accepted and [t.id for t in outcome.result.tracks] == proposal
    sent = llm.calls[0][1]
    assert str(music_dir) not in sent and ".mp3" not in sent, "file paths must not be sent"
    assert '"shortlist"' in sent


@pytest.mark.parametrize("make_ids, reason", [
    (lambda ids, all_ids: ids[:3] + [99999], "not in the shortlist"),
    (lambda ids, all_ids: ids[:3] + ids[:1], "repeats tracks"),
    (lambda ids, all_ids: ids[:3], "3 tracks instead of the requested 4"),
    (lambda ids, all_ids: ids[:3] + [max(all_ids)], "not in the shortlist"),  # 129 BPM: not eligible
])
def test_refinement_is_rejected_when_invalid(generated, conn, make_ids, reason: str) -> None:
    result, pool = generated
    ids = [t.id for t in result.tracks]
    all_ids = [r[0] for r in conn.execute("SELECT id FROM tracks")]
    outcome = refine_playlist(result, pool.tracks,
                              FakeLLM({"track_ids": make_ids(ids, all_ids), "rationale": ""}))
    assert not outcome.accepted and reason in outcome.message
    assert outcome.result is result


def test_refinement_that_scores_worse_is_rejected(conn, music_dir: Path) -> None:
    for i, key in enumerate(["8A", "9A", "10A", "3B"]):
        insert_track(conn, music_dir, artist=f"A{i}", camelot=key)
    spec = parse_spec({"track_count": 3, "ordering": {"harmonic_compatibility": True}})
    pool = find_candidates(conn, spec)
    result = build_playlist(spec, pool)
    clash = [result.tracks[0].id, max(t.id for t in pool.tracks), result.tracks[1].id]
    outcome = refine_playlist(result, pool.tracks, FakeLLM({"track_ids": clash, "rationale": ""}))
    assert not outcome.accepted and "score worse" in outcome.message
