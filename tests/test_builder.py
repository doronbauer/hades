"""Stage B/D: selection, ordering, validation and duration calculation."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import insert_track
from hades.builder import (
    build_playlist,
    fmt_duration,
    summarize,
    transition,
    validate,
)
from hades.keys import harmonic_compatibility
from hades.query import find_candidates, load_track
from hades.spec import Ordering, parse_spec


def build(conn, data: dict):
    spec = parse_spec(data)
    return build_playlist(spec, find_candidates(conn, spec))


def test_harmonic_compatibility_scores() -> None:
    assert harmonic_compatibility("8A", "8A") == 1.0
    assert harmonic_compatibility("8A", "9A") == 0.9
    assert harmonic_compatibility("12A", "1A") == 0.9
    assert harmonic_compatibility("8A", "8B") == 0.85
    assert harmonic_compatibility("8A", "9B") == 0.5
    assert harmonic_compatibility("8A", "2A") == 0.0
    with pytest.raises(ValueError):
        harmonic_compatibility("8A", "Am")


def test_track_count_and_bpm_limits(conn, music_dir: Path) -> None:
    for bpm in range(120, 141):
        insert_track(conn, music_dir, artist=f"A{bpm}", bpm=bpm)
    result = build(conn, {"track_count": 8, "bpm": {"min": 125, "max": 135}})
    assert len(result.tracks) == 8
    assert all(125 <= t.bpm <= 135 for t in result.tracks)
    assert result.candidate_count == 11
    assert result.unsatisfied == [] and result.status == "complete"


def test_ascending_progression_orders_by_tempo(conn, music_dir: Path) -> None:
    for i, bpm in enumerate([135, 125, 131, 127, 133, 129, 126, 134, 128, 132, 130]):
        insert_track(conn, music_dir, artist=f"A{i}", bpm=bpm)
    result = build(conn, {"track_count": 6,
                          "ordering": {"bpm_progression": "ascending"}})
    bpms = [t.bpm for t in result.tracks]
    assert bpms == sorted(bpms)
    assert bpms[-1] - bpms[0] >= 6, "the set should actually build up, not stay flat"
    desc = build(conn, {"track_count": 6, "ordering": {"bpm_progression": "descending"}})
    assert [t.bpm for t in desc.tracks] == sorted((t.bpm for t in desc.tracks), reverse=True)


def test_harmonic_preference_changes_order(conn, music_dir: Path) -> None:
    for i, key in enumerate(["8A", "3B", "9A", "4B", "10A", "5B"]):
        insert_track(conn, music_dir, artist=f"A{i}", bpm=128, camelot=key)
    result = build(conn, {"track_count": 3, "ordering": {"harmonic_compatibility": True}})
    keys = [t.camelot for t in result.tracks]
    for a, b in zip(keys, keys[1:]):
        assert harmonic_compatibility(a, b) >= 0.5, keys
    assert not any("key clash" in w for w in result.warnings)


def test_unreliable_keys_are_not_treated_as_compatible(conn, music_dir: Path) -> None:
    good = load_track(conn, insert_track(conn, music_dir, camelot="8A"))
    same_but_unreliable = load_track(conn, insert_track(conn, music_dir, camelot="8A",
                                                        key_flag="CONFLICT"))
    neighbour = load_track(conn, insert_track(conn, music_dir, camelot="9A"))
    ordering = Ordering(harmonic_compatibility=True)
    assert transition(good, same_but_unreliable, ordering).score < \
        transition(good, neighbour, ordering).score
    assert "key unknown" in transition(good, same_but_unreliable, ordering).issues


def test_target_duration_with_tolerance(conn, music_dir: Path) -> None:
    for i in range(30):
        insert_track(conn, music_dir, artist=f"A{i}", duration=240 + i * 7)
    result = build(conn, {"duration_minutes": 60})
    total = summarize(result.tracks).total_seconds
    assert 55 * 60 <= total <= 65 * 60
    assert result.unsatisfied == []


def test_maximum_duration_is_never_exceeded(conn, music_dir: Path) -> None:
    for i in range(20):
        insert_track(conn, music_dir, artist=f"A{i}", duration=420)  # 7 min each
    result = build(conn, {"duration_minutes": 60, "duration_mode": "max"})
    total = summarize(result.tracks).total_seconds
    assert total == 8 * 420 <= 3600


def test_default_length_is_reported(conn, music_dir: Path) -> None:
    for i in range(40):
        insert_track(conn, music_dir, artist=f"A{i}")
    result = build(conn, {"genre": ["techno"]})
    assert result.defaults and "60" in result.defaults[0]
    assert result.spec.duration_minutes is None, "the request itself is never altered"
    assert result.effective_spec.duration_minutes == 60
    assert 55 * 60 <= summarize(result.tracks).total_seconds <= 65 * 60


def test_unsatisfiable_request_gives_partial_result(conn, music_dir: Path) -> None:
    for i in range(5):
        insert_track(conn, music_dir, artist=f"A{i}", bpm=170, genre="Drum & Bass")
    result = build(conn, {"genre": ["dnb"], "track_count": 10})
    assert len(result.tracks) == 5
    assert result.status == "partial"
    assert result.unsatisfied == ["5 tracks instead of the requested 10"]

    empty = build(conn, {"genre": ["techno"]})
    assert empty.tracks == [] and "no tracks" in empty.unsatisfied[0]


def test_count_and_max_duration_conflict_keeps_duration(conn, music_dir: Path) -> None:
    for i in range(10):
        insert_track(conn, music_dir, artist=f"A{i}", duration=600)
    result = build(conn, {"track_count": 10, "duration_minutes": 30, "duration_mode": "max"})
    assert summarize(result.tracks).total_seconds <= 1800
    assert any("instead of the requested 10" in u for u in result.unsatisfied)


def test_artist_variety_and_duplicates(conn, music_dir: Path) -> None:
    for i in range(3):
        insert_track(conn, music_dir, artist="Same", title=f"Song {i}", bpm=128)
    insert_track(conn, music_dir, artist="Dupe", title="Hit", bpm=128)
    insert_track(conn, music_dir, artist="dupe", title="HIT", bpm=128)  # same song, other file
    for i in range(3):
        insert_track(conn, music_dir, artist=f"Other {i}", bpm=128)
    result = build(conn, {"track_count": 6})
    names = [(t.artist or "").lower() for t in result.tracks]
    assert names.count("dupe") == 1
    assert all(a != b for a, b in zip(names, names[1:])), names


def test_builder_is_deterministic(conn, music_dir: Path) -> None:
    for i in range(25):
        insert_track(conn, music_dir, artist=f"A{i % 7}", bpm=120 + (i * 7) % 20,
                     camelot=f"{1 + (i * 5) % 12}A", duration=200 + i * 13)
    data = {"duration_minutes": 45, "ordering": {"bpm_progression": "ascending",
                                                 "harmonic_compatibility": True}}
    assert [t.id for t in build(conn, data).tracks] == [t.id for t in build(conn, data).tracks]


def test_validate_reports_every_violation(conn, music_dir: Path) -> None:
    fast = load_track(conn, insert_track(conn, music_dir, bpm=170, duration=600))
    ok = load_track(conn, insert_track(conn, music_dir, bpm=150, duration=600))
    spec = parse_spec({"bpm": {"max": 160}, "track_count": 2, "duration_minutes": 15,
                       "duration_mode": "max"})
    problems = validate([fast, fast, ok], spec)
    assert f"track {fast.id} appears more than once" in problems
    assert f"#1 (ID {fast.id}): BPM 170 > 160" in problems
    assert "3 tracks instead of the requested 2" in problems
    assert any("exceeds the maximum of 15:00" in p for p in problems)


def test_duration_sum_and_formatting(conn, music_dir: Path) -> None:
    a = load_track(conn, insert_track(conn, music_dir, duration=332.4, bpm=125))
    b = load_track(conn, insert_track(conn, music_dir, duration=None, bpm=131))
    s = summarize([a, b])
    assert s.total_seconds == pytest.approx(332.4) and s.unknown_duration == 1
    assert s.bpm_range == "125–131 BPM"
    assert fmt_duration(3674) == "1:01:14" and fmt_duration(332.4) == "5:32"


def test_missing_a_default_length_is_only_a_warning(conn, music_dir: Path) -> None:
    for i in range(3):
        insert_track(conn, music_dir, artist=f"A{i}", genre="Ambient", duration=600)
    result = build(conn, {"genre": ["ambient"]})
    assert len(result.tracks) == 3
    assert result.unsatisfied == [] and result.status == "complete"
    assert any(w.startswith("default length not reached") for w in result.warnings)
