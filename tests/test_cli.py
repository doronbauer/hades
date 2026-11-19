"""End-to-end CLI tests."""

from __future__ import annotations

import csv
import io
import os
import re
from pathlib import Path

import pytest

import hades.analysis
from conftest import FakeAnalyzer, write_wav
from hades.cli import main


@pytest.fixture
def cli(config, monkeypatch: pytest.MonkeyPatch):
    # Use the fake analyzer so these tests don't depend on Essentia.
    monkeypatch.setattr(hades.analysis, "EssentiaAnalyzer", lambda settings: FakeAnalyzer())

    def run(*args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        monkeypatch.setattr("sys.stdout", out)
        monkeypatch.setattr("sys.stderr", err)
        code = main(["--config", str(config.path), *args])
        return code, out.getvalue(), err.getvalue()

    return run


def test_full_workflow(cli, music_dir: Path, tmp_path: Path) -> None:
    write_wav(music_dir / "a.wav")
    (music_dir / "corrupt.wav").write_bytes(b"\x00" * 64)

    code, out, _ = cli("scan")
    assert code == 0 and "files found      2" in out

    code, out, err = cli("analyze", "--limit", "5")
    assert code == 0
    assert "analyzed   1" in out and "failed     1" in out
    assert "FAILED" in err and "corrupt.wav" in err

    code, out, _ = cli("stats")
    assert code == 0 and "Tracks: 2" in out and "WAV" in out

    code, out, _ = cli("inspect", str(music_dir / "a.wav"))
    assert code == 0
    assert "Audio analysis: fake 1 [ok] (used)" in out
    assert "174.00  [source: analysis]" in out

    code, out, _ = cli("inspect", "1")
    assert code == 0 and "Track #1" in out

    code, _, err = cli("inspect", "/nope.mp3")
    assert code == 1 and "no indexed track" in err

    code, out, _ = cli("conflicts")
    assert code == 0 and "No discrepancies" in out

    target = tmp_path / "out.csv"
    code, _, _ = cli("export", "--format", "csv", "-o", str(target))
    rows = list(csv.DictReader(target.open()))
    assert code == 0 and len(rows) == 2
    good = next(r for r in rows if r["path"].endswith("a.wav"))
    assert good["analyzed_bpm"] == "174.0" and good["effective_key"] == "Am"

    code, out, _ = cli("export", "--fields", "path,effective_bpm")
    assert out.splitlines()[0] == "path,effective_bpm"

    code, _, err = cli("export", "--fields", "bogus")
    assert code == 2 and "unknown field" in err

    code, out, _ = cli("reconcile")
    assert code == 0 and "Reconciled 2 tracks" in out


def test_conflicts_listed(cli, conn, music_dir: Path) -> None:
    write_wav(music_dir / "a.wav")
    cli("scan")
    cli("analyze")
    with conn:
        conn.execute("UPDATE track_tags SET bpm_raw = '87', bpm = 87")
    code, out, _ = cli("reconcile")
    code, out, _ = cli("conflicts", "--field", "bpm")
    assert "HALF_DOUBLE" in out and "87" in out
    code, out, _ = cli("conflicts", "--field", "key")
    assert "No discrepancies" in out


def test_missing_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(tmp_path / "none.toml"), "stats"]) == 2
    assert "config file not found" in capsys.readouterr().err


def test_invalid_limit(cli) -> None:
    with pytest.raises(SystemExit):
        cli("analyze", "--limit", "0")


def test_stats_empty(cli) -> None:
    code, out, _ = cli("stats")
    assert code == 0 and "No tracks indexed yet" in out


@pytest.fixture
def tagged_library(cli, conn, music_dir: Path) -> None:
    """Three analyzed tracks with tags written directly (no ffmpeg needed)."""
    for name in ("a.wav", "b.wav", "c.wav"):
        write_wav(music_dir / name)
    cli("scan")
    cli("analyze")
    tags = {
        "a.wav": ("Björk", "Hyperballad", "1996-06-10", "One Little Indian", "128", "Electronic"),
        "b.wav": ("Burial", "Archangel", "2007", "Hyperdub", "87", "UK Garage"),
        "c.wav": ("Four Tet", "Angel Echoes", None, None, None, None),
    }
    with conn:
        for name, (artist, title, date, label, bpm, genre) in tags.items():
            conn.execute(
                """UPDATE track_tags SET artist = ?, title = ?, date = ?, label = ?,
                       bpm_raw = ?, bpm = ?, genre = ?
                   WHERE track_id = (SELECT id FROM tracks WHERE filename = ?)""",
                (artist, title, date, label, bpm, float(bpm) if bpm else None, genre, name),
            )
    cli("reconcile")


def search_ids(out: str) -> list[str]:
    table = out.split("\n\n")[0].splitlines()[1:]  # rows between header and summary
    return [line.split()[0] for line in table]


def properties(out: str) -> dict[str, str]:
    """Parse `hades list` output ("  label   value") into {label: value}."""
    return dict(
        (m.group(1), m.group(2))
        for m in (re.match(r"^  (\S.*?)\s{2,}(.*)$", line) for line in out.splitlines()[1:])
        if m
    )


def test_search_matches_artist_or_title_case_insensitively(cli, tagged_library) -> None:
    code, out, _ = cli("search", "ANGEL")  # title of b and c
    assert code == 0
    assert out.splitlines()[0].split() == ["ID", "ARTIST", "TITLE"]
    assert "Burial" in out and "Archangel" in out and "Four Tet" in out
    assert "Björk" not in out
    assert len(search_ids(out)) == 2

    code, out, _ = cli("search", "four")  # artist
    assert search_ids(out) == ["3"]

    code, out, _ = cli("search", "björk")  # non-ASCII, different case
    assert "Hyperballad" in out


def test_search_unicode_normalization(cli, tagged_library) -> None:
    import unicodedata

    code, out, _ = cli("search", unicodedata.normalize("NFD", "BJÖRK"))
    assert "Hyperballad" in out


def test_search_no_results_and_empty_query(cli, tagged_library) -> None:
    code, out, _ = cli("search", "zzz")
    assert code == 0 and "No tracks found" in out
    code, _, err = cli("search", "  ")
    assert code == 2 and "must not be empty" in err


def test_list_shows_requested_properties(cli, tagged_library) -> None:
    code, out, _ = cli("list", "2")
    assert code == 0
    assert "Track #2  Burial - Archangel" in out
    lines = properties(out)
    assert list(lines) == ["bpm", "key", "genre", "file created", "publication date", "label",
                           "loudness", "format", "duration"]
    assert lines["duration"] == "00:08"  # synthetic fixtures are 8 s long
    assert "87.00  [tag]  tag and analysis disagree: HALF_DOUBLE" in lines["bpm"]  # fake analyzer: 174
    assert "Am (8A / 1m)  [analysis]" in lines["key"]
    assert "UK Garage" in lines["genre"]
    assert "(on this system)" in lines["file created"]
    assert "2007" in lines["publication date"]
    assert "Hyperdub" in lines["label"]
    assert "-9.0 LUFS" in lines["loudness"]
    assert "WAV" in lines["format"]


def test_list_shows_dash_for_unknown_values(cli, tagged_library) -> None:
    code, out, _ = cli("list", "3")
    assert code == 0
    lines = properties(out)
    assert lines["publication date"] == lines["label"] == lines["genre"] == "-"


def test_list_unknown_or_invalid_id(cli, tagged_library) -> None:
    code, _, err = cli("list", "999")
    assert code == 1 and "no track with ID 999" in err
    with pytest.raises(SystemExit):
        cli("list", "abc")
