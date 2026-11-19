"""Stage 3: independent audio analysis.

Analyzers implement the ``Analyzer`` protocol. Results are stored separately
from embedded tags, keyed by ``Analyzer.version`` — which covers the library
version, this module's processing revision and every setting — so any change
to those makes previously analyzed tracks pending again.

A future genre classifier would be a separate analyzer that writes scored rows
to ``track_genres``; it does not need to change this one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from hades.config import AnalysisSettings

# Bump whenever the processing below changes in a way that affects results.
ANALYSIS_REVISION = 1


class AnalysisError(Exception):
    """The track could not be analyzed (e.g. undecodable audio)."""


class AnalysisSkipped(AnalysisError):
    """The track was deliberately not analyzed (e.g. longer than the configured limit)."""


@dataclass(frozen=True)
class AnalysisResult:
    bpm: float | None
    bpm_confidence: float | None
    key_tonic: str | None
    key_scale: str | None
    key_strength: float | None
    loudness_lufs: float | None
    loudness_range_lu: float | None
    dynamic_complexity: float | None
    analyzed_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


class Analyzer(Protocol):
    name: str
    version: str
    settings: dict[str, Any]

    def analyze(self, path: str) -> AnalysisResult: ...


class EssentiaAnalyzer:
    """BPM (RhythmExtractor2013), key (KeyExtractor), loudness (LoudnessEBUR128)
    and dynamic complexity, all computed on audio resampled to one fixed rate.

    Resampling matters: Essentia's EBU R128 filters are only correct at
    44.1/48 kHz (a 96 kHz file measured natively reads ~50 LU too loud), and
    the rhythm extractor assumes 44.1 kHz input.
    """

    name = "essentia"

    def __init__(self, settings: AnalysisSettings) -> None:
        try:
            import essentia
            import essentia.standard as es
            import numpy as np
        except ImportError as exc:
            raise AnalysisError(
                "Essentia is not installed; run `uv sync` to install pinned dependencies"
            ) from exc
        essentia.log.infoActive = False
        essentia.log.warningActive = False
        self._es = es
        self._np = np
        self._sr = settings.sample_rate
        self._bpm_method = settings.bpm_method
        self._key_profile = settings.key_profile
        self._max_seconds = settings.max_duration_seconds
        self.settings: dict[str, Any] = {
            "revision": ANALYSIS_REVISION,
            "sample_rate": settings.sample_rate,
            "bpm_method": settings.bpm_method,
            "key_profile": settings.key_profile,
            "max_duration_seconds": settings.max_duration_seconds,
        }
        digest = hashlib.sha256(json.dumps(self.settings, sort_keys=True).encode()).hexdigest()
        self.version = f"essentia {essentia.__version__}; r{ANALYSIS_REVISION}; {digest[:8]}"

    def analyze(self, path: str) -> AnalysisResult:
        es, np = self._es, self._np
        try:
            audio, sr, channels, _md5, _bitrate, codec = es.AudioLoader(filename=path)()
        except RuntimeError as exc:
            raise AnalysisError(f"could not decode audio: {_essentia_message(exc)}") from exc
        if audio.shape[0] == 0 or sr <= 0:
            raise AnalysisError("no audio samples decoded")
        seconds = audio.shape[0] / sr
        if seconds > self._max_seconds:
            raise AnalysisSkipped(
                f"duration {seconds:.0f}s exceeds max_duration_seconds={self._max_seconds:.0f}"
            )

        # AudioLoader always returns two columns; mono files have a silent right
        # channel, which is exactly how EBU R128 expects a single channel to be measured.
        if sr != self._sr:
            # A fresh Resample per channel: the resampler keeps internal state between calls.
            left, right = (
                es.Resample(inputSampleRate=sr, outputSampleRate=self._sr)(
                    np.ascontiguousarray(audio[:, ch])
                )
                for ch in (0, 1)
            )
            n = min(len(left), len(right))
            audio = np.ascontiguousarray(np.column_stack((left[:n], right[:n])), dtype=np.float32)
        mono = es.MonoMixer()(audio, channels)

        extra: dict[str, Any] = {"codec": codec, "source_sample_rate": float(sr)}

        bpm, _ticks, confidence, _estimates, _intervals = es.RhythmExtractor2013(
            method=self._bpm_method
        )(mono)
        tonic, scale, strength = es.KeyExtractor(
            profileType=self._key_profile, sampleRate=self._sr
        )(mono)
        dynamic_complexity, _ = es.DynamicComplexity(sampleRate=self._sr)(mono)

        loudness: float | None = None
        loudness_range: float | None = None
        try:
            _m, _s, integrated, lra = es.LoudnessEBUR128(sampleRate=self._sr)(audio)
            loudness, loudness_range = float(integrated), float(lra)
        except RuntimeError as exc:
            extra["loudness_error"] = _essentia_message(exc)

        return AnalysisResult(
            bpm=float(bpm) if bpm > 0 else None,
            # Only the multifeature method produces a confidence (range 0–5.32).
            bpm_confidence=float(confidence) if self._bpm_method == "multifeature" else None,
            key_tonic=str(tonic) if scale in ("major", "minor") else None,
            key_scale=str(scale) if scale in ("major", "minor") else None,
            key_strength=float(strength),
            loudness_lufs=_finite(loudness),
            loudness_range_lu=_finite(loudness_range),
            dynamic_complexity=_finite(float(dynamic_complexity)),
            analyzed_seconds=round(seconds, 3),
            extra=extra,
        )


def _finite(value: float | None) -> float | None:
    # Silence yields -inf/NaN loudness; store NULL instead.
    if value is None or value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _essentia_message(exc: Exception) -> str:
    return str(exc).replace("Error while configuring AudioLoader: ", "").strip()
