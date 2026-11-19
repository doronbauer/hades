"""Real Essentia analysis on synthetic audio with known tempo and key."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import encode, needs_ffmpeg, synth_track, write_wav
from hades.config import AnalysisSettings

pytest.importorskip("essentia")

from hades.analysis import AnalysisError, AnalysisSkipped, EssentiaAnalyzer  # noqa: E402

pytestmark = pytest.mark.essentia

C_MAJOR_TRIAD = (261.63, 329.63, 392.0)


@pytest.fixture(scope="module")
def analyzer() -> EssentiaAnalyzer:
    return EssentiaAnalyzer(AnalysisSettings())


@pytest.mark.parametrize("bpm", [100.0, 128.0])
def test_tempo_and_key(tmp_path: Path, analyzer: EssentiaAnalyzer, bpm: float) -> None:
    path = write_wav(tmp_path / "a.wav", synth_track(seconds=12, bpm=bpm))
    r = analyzer.analyze(str(path))
    assert r.bpm == pytest.approx(bpm, rel=0.02)
    assert (r.key_tonic, r.key_scale) == ("A", "minor")
    assert r.bpm_confidence is not None
    assert 0 <= r.key_strength <= 1
    assert r.loudness_lufs is not None and -40 < r.loudness_lufs < 0
    assert r.dynamic_complexity is not None
    assert r.analyzed_seconds == pytest.approx(12, abs=0.01)


def test_major_key(tmp_path: Path, analyzer: EssentiaAnalyzer) -> None:
    path = write_wav(tmp_path / "c.wav", synth_track(chord=C_MAJOR_TRIAD))
    r = analyzer.analyze(str(path))
    assert (r.key_tonic, r.key_scale) == ("C", "major")


@pytest.mark.parametrize("sr", [22050, 48000, 96000])
def test_results_independent_of_sample_rate(tmp_path: Path, analyzer: EssentiaAnalyzer, sr: int) -> None:
    ref = analyzer.analyze(str(write_wav(tmp_path / "ref.wav", synth_track(seconds=10))))
    other = analyzer.analyze(str(write_wav(tmp_path / f"{sr}.wav", synth_track(seconds=10, sr=sr), sr=sr)))
    assert other.bpm == pytest.approx(ref.bpm, rel=0.02)
    assert (other.key_tonic, other.key_scale) == (ref.key_tonic, ref.key_scale)
    # Essentia's EBU R128 is wrong at 96 kHz unless resampled; guard the fix.
    assert other.loudness_lufs == pytest.approx(ref.loudness_lufs, abs=0.5)
    assert other.extra["source_sample_rate"] == sr


def test_mono_loudness_measured_as_single_channel(tmp_path: Path, analyzer: EssentiaAnalyzer) -> None:
    signal = synth_track(seconds=10)
    stereo = analyzer.analyze(str(write_wav(tmp_path / "s.wav", signal)))
    mono = analyzer.analyze(str(write_wav(tmp_path / "m.wav", signal, channels=1)))
    assert mono.loudness_lufs == pytest.approx(stereo.loudness_lufs - 3.0, abs=0.2)


@needs_ffmpeg
@pytest.mark.parametrize("ext", ["mp3", "aiff", "flac", "m4a"])
def test_all_formats_decode(tmp_path: Path, analyzer: EssentiaAnalyzer, ext: str) -> None:
    src = write_wav(tmp_path / "src.wav", synth_track(seconds=10))
    r = analyzer.analyze(str(encode(src, tmp_path / f"t.{ext}")))
    assert r.bpm == pytest.approx(120, rel=0.02)
    assert r.key_scale == "minor"


def test_undecodable_file(tmp_path: Path, analyzer: EssentiaAnalyzer) -> None:
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio" * 100)
    with pytest.raises(AnalysisError):
        analyzer.analyze(str(bad))


def test_max_duration(tmp_path: Path) -> None:
    short = EssentiaAnalyzer(AnalysisSettings(max_duration_seconds=5))
    with pytest.raises(AnalysisSkipped):
        short.analyze(str(write_wav(tmp_path / "a.wav")))


def test_version_reflects_settings() -> None:
    a = EssentiaAnalyzer(AnalysisSettings())
    b = EssentiaAnalyzer(AnalysisSettings(key_profile="temperley"))
    assert a.version != b.version
    assert a.version == EssentiaAnalyzer(AnalysisSettings()).version
    assert a.version.startswith("essentia ")
