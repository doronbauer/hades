"""Artist/album variety, per-artist limits, "again" variations and M3U export."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from conftest import FakeLLM, insert_track, spec_answer
from hades.builder import build_playlist, build_variation, validate
from hades.export import export_filename, export_m3u, render_m3u
from hades.llm import interpret_request
from hades.playlist import Playlist
from hades.query import artist_names, find_candidates, load_track
from hades.repository import load_playlist
from hades.service import ExportError, generate_again, save_and_export
from hades.spec import parse_spec


def build(conn, data: dict, **kw):
    spec = parse_spec(data)
    return build_playlist(spec, find_candidates(conn, spec), **kw)


def set_album(conn, track_id: int, album: str) -> None:
    conn.execute("UPDATE track_tags SET album = ? WHERE track_id = ?", (album, track_id))
    conn.commit()


@pytest.mark.parametrize("credit, names", [
    ("Clark, Marcel Dettmann", {"clark", "marcel dettmann"}),
    ("Biocym & SHFT", {"biocym", "shft"}),
    ("Phase Fatale feat. P.E.A.R.L.", {"phase fatale", "p e a r l"}),
    ("Surgeon vs. Regis", {"surgeon", "regis"}),
    ("Wax Doctor", {"wax doctor"}),
    (None, set()),
])
def test_artist_names(credit: str | None, names: set[str]) -> None:
    assert artist_names(credit) == names


def test_shared_artist_or_album_never_back_to_back(conn, music_dir: Path) -> None:
    # The real-world case: a DJ's compilation credits every track as "X, DJ".
    comp = [insert_track(conn, music_dir, artist=f"Guest {i}, Marcel Dettmann", bpm=128 + i * 0.1)
            for i in range(4)]
    for t in comp:
        set_album(conn, t, "Mastermix")
    for i in range(4):
        insert_track(conn, music_dir, artist=f"Solo {i}", bpm=128 + i * 0.1)
    result = build(conn, {"track_count": 8})
    for a, b in zip(result.tracks, result.tracks[1:]):
        assert not (a.artist_names & b.artist_names), (a.artist, b.artist)
        assert a.album_key is None or a.album_key != b.album_key
    assert not any("same artist" in w for w in result.warnings)


def test_back_to_back_allowed_only_when_unavoidable(conn, music_dir: Path) -> None:
    for i in range(3):
        insert_track(conn, music_dir, artist="Surgeon", title=f"T{i}")
    result = build(conn, {"artists": ["Surgeon"], "track_count": 3})
    assert len(result.tracks) == 3 and result.unsatisfied == []
    assert sum("same artist" in w for w in result.warnings) == 2


def test_max_tracks_per_artist_is_a_hard_limit(conn, music_dir: Path) -> None:
    for i in range(5):
        insert_track(conn, music_dir, artist=f"Biocym, Guest {i}" if i % 2 else "Biocym")
    for i in range(3):
        insert_track(conn, music_dir, artist=f"Other {i}")
    result = build(conn, {"track_count": 5, "max_tracks_per_artist": 2})
    assert sum("biocym" in t.artist_names for t in result.tracks) == 2
    assert result.unsatisfied == []
    impossible = build(conn, {"track_count": 8, "max_tracks_per_artist": 1})
    assert any("instead of the requested 8" in u for u in impossible.unsatisfied)

    tracks = [load_track(conn, 1), load_track(conn, 2)]  # "Biocym" and "Biocym, Guest 1"
    assert "2 tracks by biocym (at most 1)" in validate(tracks, parse_spec({"max_tracks_per_artist": 1}))


def test_llm_can_set_artist_limit() -> None:
    llm = FakeLLM(spec_answer(max_tracks_per_artist=1))
    spec = interpret_request("every track by a different artist", llm, date(2026, 11, 19)).spec
    assert spec is not None and spec.max_tracks_per_artist == 1
    assert "at most 1 per artist" in spec.hard_constraints()
    assert "max_tracks_per_artist" in llm.calls[0][0]


def test_variation_is_different_valid_and_reproducible(conn, music_dir: Path) -> None:
    for i in range(30):
        insert_track(conn, music_dir, artist=f"A{i}", bpm=124 + i % 10, camelot=f"{1 + i % 12}A")
    spec = parse_spec({"track_count": 8, "ordering": {"bpm_progression": "ascending"}})
    pool = find_candidates(conn, spec)
    first = build_playlist(spec, pool)
    again = build_variation(spec, pool, [t.id for t in first.tracks], seed=1)
    overlap = {t.id for t in again.tracks} & {t.id for t in first.tracks}
    assert len(again.tracks) == 8 and again.unsatisfied == []
    assert len(overlap) <= 4
    assert [t.id for t in build_variation(spec, pool, [t.id for t in first.tracks], seed=1).tracks] \
        == [t.id for t in again.tracks]


def test_generate_again_requires_a_request(conn) -> None:
    with pytest.raises(ValueError, match="nothing to run again"):
        generate_again(conn, Playlist())


# --------------------------------------------------------------------------- export


@pytest.mark.parametrize("name, filename", [
    ("Friday Warmup", "Friday Warmup.m3u"),
    ('Peak: 2/3 "loud"?', "Peak- 2-3 -loud--.m3u"),
    ("...", "playlist.m3u"),
])
def test_export_filename(name: str, filename: str) -> None:
    assert export_filename(name) == filename


def test_m3u_content(conn, music_dir: Path) -> None:
    a = load_track(conn, insert_track(conn, music_dir, artist="Ünïcode", title="Söng", duration=332.6))
    b = load_track(conn, insert_track(conn, music_dir, artist=None, title=None, duration=None))
    pl = Playlist()
    pl.add(a)
    pl.add(b)
    pl.rename("Set")
    assert render_m3u(pl) == (
        "#EXTM3U\n#PLAYLIST:Set\n"
        f"#EXTINF:333,Ünïcode - Söng\n{a.path}\n"
        f"#EXTINF:-1,Track {b.id}\n{b.path}\n")


def test_export_to_file_or_directory(conn, music_dir: Path, tmp_path: Path) -> None:
    pl = Playlist()
    pl.add(load_track(conn, insert_track(conn, music_dir)))
    pl.rename("Mine")
    assert export_m3u(pl, tmp_path / "out") == tmp_path / "out" / "Mine.m3u"
    custom = export_m3u(pl, tmp_path / "x" / "custom.m3u")
    assert custom.read_text(encoding="utf-8").startswith("#EXTM3U")
    assert not list((tmp_path / "x").glob(".hades-*")), "no temporary files left behind"


def test_save_writes_file_and_rename_replaces_it(conn, config, music_dir: Path) -> None:
    pl = Playlist()
    pl.add(load_track(conn, insert_track(conn, music_dir)))
    pl.rename("Before")
    out = config.export_directory
    assert save_and_export(conn, pl, out) == out / "Before.m3u"
    pl.rename("After")
    save_and_export(conn, pl, out)
    assert sorted(p.name for p in out.iterdir()) == ["After.m3u"]
    reopened = load_playlist(conn, "After")
    reopened.rename("Again")
    save_and_export(conn, reopened, out)
    assert sorted(p.name for p in out.iterdir()) == ["Again.m3u"]


def test_export_failure_keeps_database_save(conn, music_dir: Path, tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("a file, not a directory")
    pl = Playlist()
    pl.add(load_track(conn, insert_track(conn, music_dir)))
    pl.rename("Kept")
    with pytest.raises(ExportError, match="saved “Kept” in the database"):
        save_and_export(conn, pl, blocker / "sub")
    assert load_playlist(conn, "Kept").id == pl.id


def test_m3u8_format_option(conn, config, make_config, music_dir: Path) -> None:
    assert config.playlist_format == "m3u" and config.export_extension == ".m3u"
    cfg = make_config(music_dir, extra='playlist_format = "m3u8"\n')
    pl = Playlist()
    pl.add(load_track(conn, insert_track(conn, music_dir, artist="Tièmoko Koné")))
    pl.rename("Accents")
    path = save_and_export(conn, pl, cfg.export_directory, cfg.export_extension)
    assert path.name == "Accents.m3u8"
    assert "Tièmoko Koné" in path.read_bytes().decode("utf-8")
