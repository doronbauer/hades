"""Stage 4: compare embedded tags with analysis results and pick effective values.

Pure functions only. Precedence for effective values: manual > tag > analysis.
A discrepancy flag is always computed from tag vs. analysis, even when a manual
value or the tag wins, so conflicts are never hidden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hades.keys import Key, parse_key

BPM_MIN = 20.0
BPM_MAX = 300.0
BPM_TOLERANCE = 0.02  # relative

# Flags
CONSISTENT = "CONSISTENT"
HALF_DOUBLE = "HALF_DOUBLE"
CONFLICT = "CONFLICT"
RELATIVE = "RELATIVE"  # same notes, different tonic (Am vs C)
PARALLEL = "PARALLEL"  # same tonic, different mode (Bb vs Bbm)
UNRECOGNIZED = "UNRECOGNIZED"
KEY_DISCREPANCIES = (RELATIVE, PARALLEL, CONFLICT, UNRECOGNIZED)
BPM_DISCREPANCIES = (HALF_DOUBLE, CONFLICT)

# Sources
MANUAL = "manual"
TAG = "tag"
ANALYSIS = "analysis"


@dataclass(frozen=True)
class BpmResolution:
    value: float | None
    source: str | None
    flag: str | None


@dataclass(frozen=True)
class KeyResolution:
    value: Key | None
    source: str | None
    flag: str | None


@dataclass(frozen=True)
class GenreResolution:
    value: str | None
    source: str | None


def parse_bpm(raw: str | float | None) -> float | None:
    """Return a plausible tempo from a tag value, or None (e.g. for "0" or "")."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*(?:bpm)?\s*", raw, re.IGNORECASE)
        if not m:
            return None
        value = float(m.group(1).replace(",", "."))
    return value if BPM_MIN <= value <= BPM_MAX else None


def compare_bpm(a: float, b: float) -> str:
    hi, lo = max(a, b), min(a, b)
    if hi - lo <= BPM_TOLERANCE * hi:
        return CONSISTENT
    if abs(hi - 2 * lo) <= BPM_TOLERANCE * hi:
        return HALF_DOUBLE
    return CONFLICT


def resolve_bpm(
    tag: float | None, analyzed: float | None, manual: float | None = None
) -> BpmResolution:
    tag = parse_bpm(tag)
    analyzed = parse_bpm(analyzed)
    flag = compare_bpm(tag, analyzed) if tag is not None and analyzed is not None else None
    if manual is not None:
        return BpmResolution(manual, MANUAL, flag)
    # A valid tag wins even on conflict: never silently "correct" the user's grid.
    if tag is not None:
        return BpmResolution(tag, TAG, flag)
    if analyzed is not None:
        return BpmResolution(analyzed, ANALYSIS, flag)
    return BpmResolution(None, None, None)


def compare_keys(a: Key, b: Key) -> str:
    if a == b:
        return CONSISTENT
    if a.is_relative_of(b):
        return RELATIVE
    if a.is_parallel_of(b):
        return PARALLEL
    return CONFLICT


def resolve_key(
    tag_raw: str | None, analyzed: Key | None, manual: str | None = None
) -> KeyResolution:
    tag = parse_key(tag_raw)
    if tag_raw and tag_raw.strip() and tag is None:
        flag: str | None = UNRECOGNIZED
    elif tag is not None and analyzed is not None:
        flag = compare_keys(tag, analyzed)
    else:
        flag = None

    manual_key = parse_key(manual)
    if manual_key is not None:
        return KeyResolution(manual_key, MANUAL, flag)
    if tag is not None:
        return KeyResolution(tag, TAG, flag)
    if analyzed is not None:
        return KeyResolution(analyzed, ANALYSIS, flag)
    return KeyResolution(None, None, flag)


def resolve_genre(tag: str | None, manual: str | None = None) -> GenreResolution:
    if manual and manual.strip():
        return GenreResolution(manual.strip(), MANUAL)
    if tag and tag.strip():
        return GenreResolution(tag, TAG)
    return GenreResolution(None, None)
