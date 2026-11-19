"""Stage 2: read embedded tags and technical properties (read-only, via TinyTag).

Only values that are actually tagged are returned; nothing is derived from
filenames. Every value TinyTag reports is preserved in ``raw`` (binary blobs,
such as the analysis chunk Traktor embeds in WAV files, are replaced by a
short placeholder).

When a file carries several values for a single-valued field (e.g. a WAV with
both an ID3 chunk and a RIFF INFO chunk), one value is chosen; the others stay
in ``raw``. For WAV files the RIFF INFO chunk wins, because that is what
Traktor reads and writes.
"""

from __future__ import annotations

import re
import struct
import unicodedata
from dataclasses import dataclass
from typing import Any

import tinytag
from tinytag import TinyTag

# Bump when extraction/normalization changes; tracks read by an older revision
# are re-read on the next scan.
EXTRACTOR_REVISION = 2
EXTRACTOR = f"tinytag {tinytag.__version__}; r{EXTRACTOR_REVISION}"

MAX_RAW_TEXT = 10_000

# RIFF INFO sub-chunk -> TinyTag field name
_RIFF_INFO_FIELDS = {
    b"INAM": "title",
    b"IART": "artist",
    b"IPRD": "album",
    b"IGNR": "genre",
    b"ICRD": "year",
    b"ICMT": "comment",
    b"ITRK": "track",
}

# Known binary values, described in the raw-tag placeholder.
_BINARY_DESCRIPTIONS = {"nitr": "Native Instruments Traktor analysis data"}


class MetadataError(Exception):
    """The file's metadata could not be read."""


@dataclass(frozen=True)
class TechnicalInfo:
    duration: float | None
    bitrate: float | None
    sample_rate: int | None
    channels: int | None
    bit_depth: int | None


@dataclass(frozen=True)
class EmbeddedTags:
    title: str | None = None
    artist: str | None = None
    album_artist: str | None = None
    album: str | None = None
    genres: tuple[str, ...] = ()
    date: str | None = None
    year: int | None = None
    label: str | None = None
    catalog_number: str | None = None
    bpm_raw: str | None = None
    key_raw: str | None = None
    remixer: str | None = None
    version: str | None = None
    isrc: str | None = None
    track_number: int | None = None
    track_total: int | None = None
    disc_number: int | None = None
    disc_total: int | None = None
    comment: str | None = None
    composer: str | None = None

    @property
    def genre(self) -> str | None:
        return "; ".join(self.genres) or None


@dataclass(frozen=True)
class ExtractedMetadata:
    technical: TechnicalInfo
    tags: EmbeddedTags
    raw: dict[str, Any]


def extract_metadata(path: str) -> ExtractedMetadata:
    try:
        tag = TinyTag.get(path, image=False)
        raw = tag.as_dict()
        preferred = read_riff_info(path) if path.lower().endswith(".wav") else {}
    except Exception as exc:  # TinyTag raises a variety of errors on corrupt files
        raise MetadataError(f"{type(exc).__name__}: {exc}") from exc

    technical = TechnicalInfo(
        duration=_float(tag.duration),
        bitrate=_float(tag.bitrate),
        sample_rate=_int(tag.samplerate),
        channels=_int(tag.channels),
        bit_depth=_int(tag.bitdepth),
    )
    return ExtractedMetadata(
        technical=technical,
        tags=normalize_tags(raw, preferred),
        raw={k: _sanitize_raw(k, v) for k, v in raw.items()},
    )


def read_riff_info(path: str) -> dict[str, str]:
    """Text fields from a WAV file's RIFF ``LIST/INFO`` chunk, keyed like TinyTag fields.

    Read-only; seeks past the audio data instead of reading it.
    """
    out: dict[str, str] = {}
    with open(path, "rb") as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return out
        while len(h := f.read(8)) == 8:
            chunk_id, size = struct.unpack("<4sI", h)
            start = f.tell()
            if chunk_id == b"LIST" and f.read(4) == b"INFO":
                end = start + size
                pos = start + 4
                while pos + 8 <= end:
                    f.seek(pos)
                    sub_id, sub_size = struct.unpack("<4sI", f.read(8))
                    field = _RIFF_INFO_FIELDS.get(sub_id)
                    if field and field not in out and sub_size <= 65536:
                        text = _decode_info(f.read(sub_size))
                        if text:
                            out[field] = text
                    pos += 8 + sub_size + (sub_size & 1)
            f.seek(start + size + (size & 1))
    return out


def _decode_info(data: bytes) -> str:
    data = data.split(b"\0", 1)[0]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")  # common for INFO chunks written by older software
    return text.strip()


def normalize_tags(raw: dict[str, Any], preferred: dict[str, str] | None = None) -> EmbeddedTags:
    """Map TinyTag's flat ``as_dict()`` output onto the columns we store.

    ``preferred`` holds values that override TinyTag's for the same field
    (the RIFF INFO chunk of WAV files).
    """
    preferred = preferred or {}

    def values(*keys: str) -> list[str]:
        out: list[str] = []
        for key in keys:
            if key in preferred:
                items: list[Any] = [preferred[key]]
            else:
                v = raw.get(key)
                items = v if isinstance(v, list) else [v] if v is not None else []
            for item in items:
                text = str(item).strip()
                if text and text not in out:
                    out.append(text)
        return out

    def joined(*keys: str, sep: str = "; ") -> str | None:
        """Multi-valued fields: keep every distinct value."""
        return sep.join(values(*keys)) or None

    def single(*keys: str) -> str | None:
        """Single-valued fields: the first value; any others remain in ``raw``."""
        return _first(values(*keys))

    date = single("year")
    year_match = re.search(r"\b(1[89]\d\d|2\d\d\d)\b", date or "")

    return EmbeddedTags(
        title=single("title"),
        artist=joined("artist"),
        album_artist=joined("albumartist"),
        album=single("album"),
        genres=_split_genres(values("genre")),
        date=date,
        year=int(year_match.group(1)) if year_match else None,
        label=single("publisher", "label", "organization"),
        catalog_number=single("catalog_number", "catalognumber", "catalog"),
        bpm_raw=single("bpm"),
        key_raw=single("initial_key", "key"),
        remixer=joined("remixer", "mixartist", "remixed_by"),
        version=single("subtitle", "version", "mix"),
        isrc=single("isrc"),
        track_number=_leading_int(single("track")),
        track_total=_leading_int(single("track_total")),
        disc_number=_leading_int(single("disc")),
        disc_total=_leading_int(single("disc_total")),
        # ID3 COMM frames with a language/description come through as "_comment"
        comment=joined("comment", "_comment", sep="\n"),
        composer=joined("composer"),
    )


def _sanitize_raw(key: str, value: Any) -> Any:
    """Replace binary blobs and huge strings so raw tags stay readable and small."""
    if isinstance(value, list):
        return [_sanitize_raw(key, v) for v in value]
    if not isinstance(value, str):
        return value
    if _looks_binary(value):
        what = _BINARY_DESCRIPTIONS.get(key, "binary data")
        return f"<omitted: {what}, {len(value):,} characters>"
    if len(value) > MAX_RAW_TEXT:
        return value[:MAX_RAW_TEXT] + f"… <truncated, {len(value):,} characters>"
    return value


def _looks_binary(text: str) -> bool:
    sample = text[:2000]
    return "�" in sample or any(
        unicodedata.category(ch) == "Cc" and ch not in "\t\n\r" for ch in sample
    )


def _split_genres(raw_genres: list[str]) -> tuple[str, ...]:
    # Split only on ";" — "/" and "," legitimately occur inside genre names.
    out: list[str] = []
    seen: set[str] = set()
    for value in raw_genres:
        for part in value.split(";"):
            part = part.strip()
            if part and part.casefold() not in seen:
                seen.add(part.casefold())
                out.append(part)
    return tuple(out)


def _first(items: list[str]) -> str | None:
    return items[0] if items else None


def _leading_int(value: str | None) -> int | None:
    m = re.match(r"\s*(\d+)", value or "")
    return int(m.group(1)) if m else None


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
