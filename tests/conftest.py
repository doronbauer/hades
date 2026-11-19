"""Shared fixtures. All audio is synthesized at test time; nothing copyrighted is committed."""

from __future__ import annotations

import shutil
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from hades.analysis import AnalysisError, AnalysisResult
from hades.config import Config, load_config
from hades.db import connect

A_MINOR_TRIAD = (220.0, 261.63, 329.63)  # A3, C4, E4


def synth_track(seconds: float = 8.0, bpm: float = 120.0, sr: int = 44100,
                chord: tuple[float, ...] = A_MINOR_TRIAD) -> np.ndarray:
    """A sustained chord plus a decaying low kick on every beat, as float32 mono."""
    t = np.arange(int(seconds * sr)) / sr
    tones = sum(np.sin(2 * np.pi * f * t) for f in chord) / len(chord)
    beat = 60.0 / bpm
    kick = np.sin(2 * np.pi * 60 * t) * np.exp(-30 * np.mod(t, beat))
    return (0.3 * tones + 0.6 * kick).astype(np.float32)


def write_wav(path: Path, signal: np.ndarray | None = None, sr: int = 44100,
              channels: int = 2) -> Path:
    signal = synth_track(sr=sr) if signal is None else signal
    pcm = (np.clip(signal, -1, 1) * 32767).astype("<i2")
    if channels == 2:
        pcm = np.column_stack((pcm, pcm)).ravel()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def encode(src: Path, dst: Path, tags: dict[str, str] | None = None) -> Path:
    """Convert a WAV into another format with ffmpeg, optionally writing tags."""
    assert FFMPEG is not None
    cmd = [FFMPEG, "-loglevel", "error", "-y", "-i", str(src)]
    for k, v in (tags or {}).items():
        cmd += ["-metadata", f"{k}={v}"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([*cmd, str(dst)], check=True)
    return dst


@pytest.fixture
def music_dir(tmp_path: Path) -> Path:
    d = tmp_path / "music"
    d.mkdir()
    return d


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[..., Config]:
    def _make(*dirs: Path | str, extra: str = "") -> Config:
        listed = ", ".join(f'"{d}"' for d in dirs)
        path = tmp_path / "config.toml"
        path.write_text(f'music_directories = [{listed}]\ndatabase_path = "db/lib.sqlite3"\n{extra}')
        return load_config(path)

    return _make


@pytest.fixture
def config(make_config: Callable[..., Config], music_dir: Path) -> Config:
    return make_config(music_dir)


@pytest.fixture
def conn(config: Config):
    c = connect(config.database_path)
    yield c
    c.close()


class FakeAnalyzer:
    """Deterministic stand-in for Essentia, so pipeline tests stay fast."""

    name = "fake"
    settings = {"revision": 1}

    def __init__(self, bpm: float = 174.0, tonic: str = "A", scale: str = "minor",
                 version: str = "fake 1") -> None:
        self.bpm, self.tonic, self.scale, self.version = bpm, tonic, scale, version
        self.calls: list[str] = []

    def analyze(self, path: str) -> AnalysisResult:
        self.calls.append(path)
        if Path(path).name.startswith("corrupt"):
            raise AnalysisError("could not decode audio")
        return AnalysisResult(
            bpm=self.bpm, bpm_confidence=3.0, key_tonic=self.tonic, key_scale=self.scale,
            key_strength=0.8, loudness_lufs=-9.0, loudness_range_lu=4.0,
            dynamic_complexity=2.0, analyzed_seconds=8.0,
        )


# --------------------------------------------------------------------------- playlist fixtures


def _key_name(camelot: str | None) -> str | None:
    from hades.keys import parse_key

    key = parse_key(camelot) if camelot else None
    return key.name if key else None


def insert_track(conn, music_dir: Path, *, artist: str | None = "Artist", title: str | None = None,
                 bpm: float | None = 128.0, camelot: str | None = "8A", genre: str | None = "Techno",
                 duration: float | None = 300.0, created: str | None = "2026-05-10T12:00:00+00:00",
                 status: str = "available", bpm_flag: str | None = None,
                 bpm_source: str | None = "tag", key_flag: str | None = None,
                 key_source: str | None = "tag", subdir: str = "", create_file: bool = True) -> int:
    """Insert a fully reconciled track row (plus tags) without scanning audio."""
    n = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] + 1
    directory = music_dir / subdir if subdir else music_dir
    path = directory / f"track{n}.mp3"
    if create_file:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00")
    cur = conn.execute(
        """INSERT INTO tracks (path, filename, directory, root, format, file_size, file_mtime_ns,
                               file_modified, file_created, content_signature, duration,
                               first_indexed, last_seen, status, effective_bpm,
                               effective_bpm_source, bpm_flag, effective_key, effective_key_camelot,
                               effective_key_source, key_flag, effective_genre,
                               effective_genre_source)
           VALUES (?, ?, ?, ?, 'MP3', 1, 1, ?, ?, '1:1', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (str(path), path.name, str(directory), str(music_dir), created or "2026-01-01T00:00:00+00:00",
         created, duration, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00", status,
         bpm, bpm_source if bpm is not None else None, bpm_flag, _key_name(camelot), camelot,
         key_source if camelot else None, key_flag, genre, "tag" if genre else None))
    track_id = int(cur.lastrowid)
    conn.execute(
        "INSERT INTO track_tags (track_id, title, artist, genre, raw_json, extractor, extracted_at) "
        "VALUES (?, ?, ?, ?, '{}', 'test', '2026-01-01T00:00:00+00:00')",
        (track_id, title or f"Track {track_id}", artist, genre))
    conn.commit()
    return track_id


class FakeLLM:
    """Returns queued JSON answers and records what it was asked."""

    def __init__(self, *answers: dict) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append((system, user))
        if not self.answers:
            raise AssertionError("FakeLLM: no answer queued")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def spec_answer(**spec) -> dict:
    """An interpretation answer in the LLM's wire format (all fields present)."""
    full = {"genre": [], "created_after": None, "created_before": None, "bpm": None,
            "duration_minutes": None, "duration_mode": None, "duration_tolerance_minutes": None,
            "track_count": None, "folder": None, "artists": [], "exclude_artists": [], "max_tracks_per_artist": None,
            "ordering": {"bpm_progression": "none", "harmonic_compatibility": False}}
    full.update(spec)
    return {"error": None, "spec": full}
