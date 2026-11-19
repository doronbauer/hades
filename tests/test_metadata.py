"""Embedded-metadata extraction across formats (fixtures encoded with ffmpeg)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from conftest import encode, needs_ffmpeg, write_wav
from hades.metadata import MetadataError, extract_metadata, normalize_tags
from hades.pipeline import scan

pytestmark = [pytest.mark.ffmpeg, needs_ffmpeg]

ID3_TAGS = {
    "title": "Night Drive (Extended Mix)",
    "artist": "Test Artist",
    "album_artist": "Various",
    "album": "Test EP",
    "genre": "Techno",
    "date": "2021-05-14",
    "TBPM": "87",
    "TKEY": "8A",
    "publisher": "Test Label",
    "TSRC": "GBXXX2100001",
    "track": "3/10",
    "comment": "promo copy",
    "CATALOGNUMBER": "TL001",
}


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "src.wav")


def test_mp3_tags(tmp_path: Path, wav: Path) -> None:
    meta = extract_metadata(str(encode(wav, tmp_path / "t.mp3", ID3_TAGS)))
    t = meta.tags
    assert t.title == "Night Drive (Extended Mix)"
    assert (t.artist, t.album_artist, t.album) == ("Test Artist", "Various", "Test EP")
    assert t.genres == ("Techno",)
    assert (t.date, t.year) == ("2021-05-14", 2021)
    assert (t.bpm_raw, t.key_raw) == ("87", "8A")
    assert (t.label, t.catalog_number, t.isrc) == ("Test Label", "TL001", "GBXXX2100001")
    assert (t.track_number, t.track_total) == (3, 10)
    assert t.comment == "promo copy"
    assert t.version is None  # not derived from the "(Extended Mix)" in the title
    assert meta.technical.sample_rate == 44100
    assert meta.technical.channels == 2
    assert meta.technical.bitrate and meta.technical.bitrate > 0
    assert meta.technical.duration == pytest.approx(8.0, abs=0.1)
    assert meta.raw["bpm"] == ["87"]


def test_flac_vorbis_comments(tmp_path: Path, wav: Path) -> None:
    path = encode(wav, tmp_path / "t.flac", {
        "title": "x", "BPM": "128", "INITIALKEY": "Am", "LABEL": "L", "GENRE": "House;Deep House",
        "MY_CUSTOM_TAG": "kept"})
    meta = extract_metadata(str(path))
    assert meta.tags.bpm_raw == "128"
    assert meta.tags.key_raw == "Am"
    assert meta.tags.label == "L"
    assert meta.tags.genres == ("House", "Deep House")
    assert meta.tags.genre == "House; Deep House"
    assert meta.technical.bit_depth == 16
    assert meta.raw.get("my_custom_tag") == ["kept"]  # unrecognized tags preserved


@pytest.mark.parametrize("ext", ["m4a", "aiff", "wav"])
def test_other_formats_read_without_error(tmp_path: Path, wav: Path, ext: str) -> None:
    meta = extract_metadata(str(encode(wav, tmp_path / f"t.{ext}", {"title": "T", "artist": "A"})))
    assert meta.technical.duration == pytest.approx(8.0, abs=0.1)
    if ext == "m4a":
        assert (meta.tags.title, meta.tags.artist) == ("T", "A")


def test_untagged_file_has_null_fields(tmp_path: Path, wav: Path) -> None:
    t = extract_metadata(str(encode(wav, tmp_path / "Artist - Title (Remix).flac"))).tags
    assert (t.title, t.artist, t.genre, t.bpm_raw, t.key_raw) == (None, None, None, None, None)


def test_unreadable_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MetadataError):
        extract_metadata(str(tmp_path / "does-not-exist.mp3"))


def test_normalize_tags_handles_lists_and_blanks() -> None:
    t = normalize_tags({"artist": ["A", "B", "A"], "genre": ["Techno; techno", " "], "year": ["1999"],
                        "track": ["07"], "bpm": [""]})
    assert t.artist == "A; B"
    assert t.genres == ("Techno",)
    assert t.year == 1999
    assert t.track_number == 7
    assert t.bpm_raw is None


def test_scan_persists_tags(conn, config, music_dir: Path, wav: Path) -> None:
    encode(wav, music_dir / "t.mp3", ID3_TAGS)
    scan(conn, config)
    row = conn.execute("SELECT * FROM track_tags").fetchone()
    assert (row["bpm_raw"], row["bpm"], row["key_raw"], row["key"]) == ("87", 87.0, "8A", "Am")
    assert json.loads(row["raw_json"])["initial_key"] == ["8A"]
    assert row["extractor"].startswith("tinytag")
    genres = conn.execute("SELECT source, genre, rank FROM track_genres").fetchall()
    assert [tuple(g) for g in genres] == [("tag", "Techno", 1)]
    t = conn.execute("SELECT * FROM tracks").fetchone()
    assert (t["effective_bpm"], t["effective_bpm_source"]) == (87.0, "tag")
    assert (t["effective_genre"], t["effective_key"]) == ("Techno", "Am")


# --- WAV files with both an ID3 chunk and a RIFF INFO chunk (as Traktor leaves them) ---


def _id3v23(frames: dict[str, str]) -> bytes:
    body = b"".join(
        fid.encode() + struct.pack(">I", len(text.encode()) + 1) + b"\x00\x00" + b"\x03" + text.encode()
        for fid, text in frames.items()
    )
    n = len(body)
    syncsafe = bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])
    return b"ID3\x03\x00\x00" + syncsafe + body


def _chunk(cid: bytes, data: bytes) -> bytes:
    return cid + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) & 1 else b"")


def make_traktor_wav(path: Path) -> Path:
    """A WAV like Traktor leaves it: ID3 chunk (artist + short title) and a RIFF INFO
    chunk with Traktor's binary NITR analysis blob and a filename-style INAM title."""
    write_wav(path)
    data = path.read_bytes()
    nitr = b"NTKB\x8d\x02\x00data\xff\xfe\x00\x01" * 500
    extra = _chunk(b"ID3 ", _id3v23({"TPE1": "Kommune", "TIT2": "I", "TALB": "Oast"}))
    extra += _chunk(b"LIST", b"INFO" + _chunk(b"NITR", nitr) + _chunk(b"INAM", b"01_Kommune_I\x00"))
    data = data + extra
    path.write_bytes(data[:4] + struct.pack("<I", len(data) - 8) + data[8:])
    return path


def test_read_riff_info(tmp_path: Path) -> None:
    from hades.metadata import read_riff_info

    assert read_riff_info(str(make_traktor_wav(tmp_path / "t.wav"))) == {"title": "01_Kommune_I"}
    assert read_riff_info(str(write_wav(tmp_path / "plain.wav"))) == {}


def test_wav_prefers_riff_info_title_and_keeps_id3_values(tmp_path: Path) -> None:
    meta = extract_metadata(str(make_traktor_wav(tmp_path / "t.wav")))
    assert meta.tags.title == "01_Kommune_I"  # what Traktor shows
    assert meta.tags.artist == "Kommune"  # only in the ID3 chunk; still used
    assert meta.tags.album == "Oast"
    assert sorted(meta.raw["title"]) == ["01_Kommune_I", "I"]  # both preserved


def test_binary_raw_values_are_omitted(tmp_path: Path) -> None:
    meta = extract_metadata(str(make_traktor_wav(tmp_path / "t.wav")))
    (nitr,) = meta.raw["nitr"]
    assert nitr.startswith("<omitted: Native Instruments Traktor analysis data, ")
    assert len(json.dumps(meta.raw)) < 2000


def test_single_valued_fields_take_first_value() -> None:
    t = normalize_tags({"title": ["I", "01_Kommune_I"], "album": ["A", "B"], "artist": ["X", "Y"]})
    assert (t.title, t.album, t.artist) == ("I", "A", "X; Y")
