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
