"""Stage A: deterministic candidate retrieval."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from conftest import insert_track
from hades.query import (
    artist_matches,
    find_candidates,
    folder_matches,
    genre_matches,
    load_track,
    track_violations,
)
from hades.spec import parse_spec


@pytest.mark.parametrize("track_genre, requested, expected", [
    ("Drum & Bass", ["drum and bass"], True),
    ("Drum'n'Bass", ["Drum & Bass"], True),
    ("DnB", ["drum & bass"], True),
    ("Liquid Drum & Bass", ["dnb"], True),
    ("Minimal Techno", ["techno"], True),
    ("Electronic", ["electro"], False),
    ("Tech House", ["techno"], False),
    ("Electro; Breaks", ["breakbeat"], True),
    ("DARK AMBIENT / DRONE / METAL", ["ambient"], True),
    (None, ["techno"], False),
    ("House", [], True),
])
def test_genre_matching(track_genre: str | None, requested: list[str], expected: bool) -> None:
    assert genre_matches(track_genre, requested) is expected


def test_artist_and_folder_matching() -> None:
    assert artist_matches("Surgeon & Lady Star Garden", ["surgeon"])
    assert not artist_matches("Surgeonics", ["surgeon"])
    assert not artist_matches(None, ["surgeon"])
    assert folder_matches("/m/Bandcamp/2026/new", "bandcamp")
    assert folder_matches("/m/Bandcamp/2026", "Bandcamp/2026")
    assert not folder_matches("/m/Bandcampish", "bandcamp")
    assert folder_matches("/m/Bandcamp/x", "/m/Bandcamp")
    assert not folder_matches("/m/Bandcamp2", "/m/Bandcamp")


def ids(pool) -> list[int]:
    return [t.id for t in pool.tracks]


def test_date_filter_is_calendar_based_on_creation_date(conn, music_dir: Path) -> None:
    march = insert_track(conn, music_dir, created="2026-03-31T12:00:00+00:00")
    april = insert_track(conn, music_dir, created="2026-04-01T12:00:00+00:00")
    undated = insert_track(conn, music_dir, created=None)
    spec = parse_spec({"created_after": "2026-03-31"})
    pool = find_candidates(conn, spec)
    assert ids(pool) == [april]
    assert pool.excluded["no filesystem creation date"] == 1
    assert track_violations(load_track(conn, march), spec) == ["created 2026-03-31"]
    assert track_violations(load_track(conn, undated), spec) == ["no creation date"]

    before = find_candidates(conn, parse_spec({"created_before": "2026-04-01"}))
    assert ids(before) == [march]


def test_bpm_filter_excludes_missing_and_conflicting(conn, music_dir: Path) -> None:
    ok = insert_track(conn, music_dir, bpm=150)
    insert_track(conn, music_dir, bpm=170)
    insert_track(conn, music_dir, bpm=None)
    insert_track(conn, music_dir, bpm=150, bpm_flag="CONFLICT")
    manual = insert_track(conn, music_dir, bpm=155, bpm_flag="CONFLICT", bpm_source="manual")
    halftime = insert_track(conn, music_dir, bpm=158, bpm_flag="HALF_DOUBLE")
    pool = find_candidates(conn, parse_spec({"bpm": {"max": 160}}))
    assert ids(pool) == [ok, manual, halftime]
    assert pool.excluded == {"no BPM": 1, "tag and analyzed BPM conflict": 1}
    # Without a BPM constraint, missing BPM is no reason to exclude.
    assert len(find_candidates(conn, parse_spec({})).tracks) == 6


def test_genre_status_files_and_duration(conn, music_dir: Path) -> None:
    dnb = insert_track(conn, music_dir, genre="Drum & Bass")
    insert_track(conn, music_dir, genre="Techno")
    insert_track(conn, music_dir, genre=None)
    insert_track(conn, music_dir, genre="DnB", status="missing")
    insert_track(conn, music_dir, genre="DnB", create_file=False)
    insert_track(conn, music_dir, genre="DnB", duration=None)
    pool = find_candidates(conn, parse_spec({"genre": ["drum & bass"]}))
    assert ids(pool) == [dnb]
    assert pool.excluded == {"no genre": 1, "file not found on disk": 1, "unknown duration": 1}
    assert any("file not found" in w for w in pool.warnings())


def test_folder_and_artist_filters(conn, music_dir: Path) -> None:
    a = insert_track(conn, music_dir, artist="Surgeon", subdir="Bandcamp")
    b = insert_track(conn, music_dir, artist="Regis", subdir="Bandcamp")
    c = insert_track(conn, music_dir, artist="Surgeon", subdir="Boomkat")
    d = insert_track(conn, music_dir, artist=None, subdir="Boomkat")
    assert ids(find_candidates(conn, parse_spec({"folder": "bandcamp"}))) == [a, b]
    assert ids(find_candidates(conn, parse_spec({"artists": ["surgeon"]}))) == [a, c]
    # Unknown artist: cannot be a requested artist, but is not an excluded one either.
    assert ids(find_candidates(conn, parse_spec({"exclude_artists": ["Surgeon"]}))) == [b, d]


def test_effective_values_and_flags_are_preserved(conn, music_dir: Path) -> None:
    t = load_track(conn, insert_track(conn, music_dir, camelot="8A", key_flag="CONFLICT",
                                      bpm=174, bpm_flag="HALF_DOUBLE"))
    assert t is not None
    assert t.key == "Am" and t.camelot == "8A" and not t.key_reliable
    assert t.bpm == 174 and t.bpm_reliable and t.bpm_flag == "HALF_DOUBLE"
    assert t.created_date == date(2026, 5, 10)
