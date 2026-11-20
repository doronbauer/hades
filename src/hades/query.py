"""Stage A: retrieve playlist candidates from SQLite with deterministic filters.

Every filter is a parameterized SQL condition. Genre, artist, folder and date
matching use small Python predicates registered as SQL functions, so the same
code decides membership during retrieval and during later validation
(``track_violations``).

A track whose value for a constrained field is missing never satisfies that
constraint. BPM values that tags and analysis disagree on (flag CONFLICT,
unless set manually) are treated as unknown when a BPM limit applies; the same
holds for unreliable keys in key filters.

The free-form ``filters`` of a spec are evaluated in Python on loaded tracks
(``filters_match``), again by the same code that validates playlists later.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import cached_property
from pathlib import Path

from hades.keys import camelot_parts
from hades.reconcile import CONFLICT, MANUAL, PARALLEL, UNRECOGNIZED
from hades.spec import (
    DATE_FIELDS,
    FIELD_LABELS,
    FLAG_FIELDS,
    KEY_FIELDS,
    TEXT_FIELDS,
    Condition,
    PlaylistSpec,
    describe_group,
)

# Canonical genre -> other spellings. Matching is on whole words, so "techno"
# also matches "Minimal Techno" but "electro" does not match "Electronic".
GENRE_ALIASES: dict[str, tuple[str, ...]] = {
    "drum and bass": ("dnb", "d and b", "drum n bass", "drumnbass", "drum bass"),
    "hip hop": ("hiphop", "rap"),
    "uk garage": ("ukg", "2 step", "2step", "garage"),
    "breakbeat": ("breaks",),
    "rhythm and blues": ("r and b", "rnb"),
    "downtempo": ("down tempo",),
    "electro": ("electro funk",),
    "techno": (),
    "house": (),
    "ambient": (),
}


@dataclass(frozen=True)
class Track:
    id: int
    path: str
    filename: str
    directory: str
    status: str
    artist: str | None
    title: str | None
    duration: float | None
    bpm: float | None
    bpm_source: str | None
    bpm_flag: str | None
    key: str | None
    camelot: str | None
    key_source: str | None
    key_flag: str | None
    genre: str | None
    file_created: str | None
    file_exists: bool = True
    album: str | None = None
    year: int | None = None
    format: str | None = None
    loudness: float | None = None  # integrated LUFS from the latest analysis

    # Cached: the builder asks for these many times per candidate.
    @cached_property
    def artist_names(self) -> frozenset[str]:
        return artist_names(self.artist)

    @cached_property
    def album_key(self) -> str | None:
        return normalize_text(self.album) or None if self.album else None

    @cached_property
    def identity(self) -> tuple[str, str] | None:
        """Normalized (artist, title), to recognize the same song in two files."""
        if self.artist and self.title:
            return normalize_text(self.artist), normalize_text(self.title)
        return None

    @property
    def available(self) -> bool:
        return self.status == "available" and self.file_exists

    @property
    def display(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} — {self.title}"
        return self.title or self.filename

    @property
    def bpm_reliable(self) -> bool:
        return self.bpm is not None and (self.bpm_source == MANUAL or self.bpm_flag != CONFLICT)

    @property
    def key_reliable(self) -> bool:
        if self.camelot is None or camelot_parts(self.camelot) is None:
            return False
        return self.key_source == MANUAL or self.key_flag not in (CONFLICT, PARALLEL, UNRECOGNIZED)

    @property
    def created_date(self) -> date | None:
        return local_date(self.file_created)


@dataclass
class CandidatePool:
    tracks: list[Track]
    # Reason -> number of otherwise-eligible tracks excluded for missing/conflicting data.
    excluded: Counter[str] = field(default_factory=Counter)

    def warnings(self) -> list[str]:
        return [f"{n} track(s) excluded: {reason}" for reason, n in sorted(self.excluded.items())]


# --------------------------------------------------------------------------- matching


def normalize_text(text: str) -> str:
    s = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = s.replace("&", " and ").replace("'n'", " and ").replace("’n’", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\bn\b", "and", s)  # "drum n bass"
    return " ".join(s.split())


def _contains_words(haystack: str, needle: str) -> bool:
    return bool(needle) and f" {needle} " in f" {haystack} "


def genre_variants(requested: str) -> set[str]:
    norm = normalize_text(requested)
    variants = {norm}
    for canonical, aliases in GENRE_ALIASES.items():
        group = {canonical, *(normalize_text(a) for a in aliases)}
        if norm in group or norm.replace(" ", "") in {g.replace(" ", "") for g in group}:
            variants |= group
    return variants


def genre_matches(track_genre: str | None, requested: list[str]) -> bool:
    if not requested:
        return True
    if not track_genre:
        return False
    parts = [normalize_text(p) for p in re.split(r"[;,/|]", track_genre)]
    for wanted in requested:
        for variant in genre_variants(wanted):
            for part in parts:
                if _contains_words(part, variant) or (
                    " " in variant and part.replace(" ", "") == variant.replace(" ", "")
                ):
                    return True
    return False


_ARTIST_SEPARATORS = re.compile(
    r"\s*(?:,|;|/|&|\+|\bx\b|\bvs\.?|\bfeat\.?|\bft\.?|\bfeaturing\b|\bwith\b)\s*",
    re.IGNORECASE)


def artist_names(artist: str | None) -> frozenset[str]:
    """Individual normalized names in a credit: "Clark, Marcel Dettmann" -> {clark, marcel dettmann}."""
    if not artist:
        return frozenset()
    names = {normalize_text(p) for p in _ARTIST_SEPARATORS.split(artist)}
    return frozenset(n for n in names if n)


def artist_matches(track_artist: str | None, names: list[str]) -> bool:
    if not track_artist:
        return False
    hay = normalize_text(track_artist)
    return any(_contains_words(hay, normalize_text(n)) for n in names)


def folder_matches(directory: str, folder: str) -> bool:
    folder = folder.strip()
    if folder.startswith(("/", "~")):
        base = os.path.normpath(os.path.expanduser(folder))
        return directory == base or directory.startswith(base.rstrip("/") + "/")
    wanted = [p.casefold() for p in Path(folder).parts if p not in ("/", "")]
    parts = [p.casefold() for p in Path(directory).parts]
    n = len(wanted)
    return n > 0 and any(parts[i:i + n] == wanted for i in range(len(parts) - n + 1))


def local_date(iso: str | None) -> date | None:
    """Calendar date of an ISO-8601 timestamp in the local timezone."""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).astimezone().date()
    except ValueError:
        return None


def register_functions(conn: sqlite3.Connection) -> None:
    conn.create_function(
        "hades_genre_match", 2, lambda g, req: genre_matches(g, json.loads(req)), deterministic=True)
    conn.create_function(
        "hades_artist_match", 2, lambda a, req: artist_matches(a, json.loads(req)),
        deterministic=True)
    conn.create_function(
        "hades_folder_match", 2, lambda d, f: folder_matches(d, f), deterministic=True)
    conn.create_function(
        "hades_local_date", 1,
        lambda iso: (d.isoformat() if (d := local_date(iso)) else None), deterministic=True)


# --------------------------------------------------------------------------- loading

_SELECT = """
SELECT t.id, t.path, t.filename, t.directory, t.status, t.duration,
       t.effective_bpm, t.effective_bpm_source, t.bpm_flag,
       t.effective_key, t.effective_key_camelot, t.effective_key_source, t.key_flag,
       t.effective_genre, t.file_created, g.artist, g.title, g.album, g.year, t.format,
       (SELECT a.loudness_lufs FROM audio_analysis a
        WHERE a.track_id = t.id AND a.status = 'ok' AND a.content_signature = t.content_signature
        ORDER BY a.analyzed_at DESC, a.id DESC LIMIT 1) AS loudness
FROM tracks t
LEFT JOIN track_tags g ON g.track_id = t.id
"""


def _track(row: sqlite3.Row, check_file: bool = True) -> Track:
    return Track(
        id=row["id"], path=row["path"], filename=row["filename"], directory=row["directory"],
        status=row["status"], artist=row["artist"], title=row["title"],
        duration=row["duration"],
        bpm=row["effective_bpm"], bpm_source=row["effective_bpm_source"], bpm_flag=row["bpm_flag"],
        key=row["effective_key"], camelot=row["effective_key_camelot"],
        key_source=row["effective_key_source"], key_flag=row["key_flag"],
        genre=row["effective_genre"], file_created=row["file_created"],
        file_exists=os.path.isfile(row["path"]) if check_file else True,
        album=row["album"], year=row["year"], format=row["format"], loudness=row["loudness"],
    )


def load_tracks(conn: sqlite3.Connection, ids: list[int]) -> dict[int, Track]:
    """Load tracks by ID (any status). IDs that don't exist are absent from the result."""
    out: dict[int, Track] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(f"{_SELECT} WHERE t.id IN ({marks})", chunk):
            out[row["id"]] = _track(row)
    return out


def load_track(conn: sqlite3.Connection, track_id: int) -> Track | None:
    return load_tracks(conn, [track_id]).get(track_id)


# --------------------------------------------------------------------------- retrieval

_BPM_TRUSTED = f"(t.bpm_flag IS NOT '{CONFLICT}' OR t.effective_bpm_source = '{MANUAL}')"


def _conditions(spec: PlaylistSpec) -> list[tuple[str, str, tuple[object, ...]]]:
    """(name, SQL condition, parameters) for each hard filter."""
    conds: list[tuple[str, str, tuple[object, ...]]] = [
        ("status", "t.status = 'available'", ()),
        ("duration", "t.duration IS NOT NULL AND t.duration > 0", ()),
    ]
    if spec.created_after:
        conds.append(("date", "hades_local_date(t.file_created) > ?",
                      (spec.created_after.isoformat(),)))
    if spec.created_before:
        conds.append(("date", "hades_local_date(t.file_created) < ?",
                      (spec.created_before.isoformat(),)))
    if spec.bpm is not None or spec.ordering.max_bpm_step is not None:
        conds.append(("bpm_known", "t.effective_bpm IS NOT NULL", ()))
        conds.append(("bpm_trusted", _BPM_TRUSTED, ()))
    if spec.bpm is not None:
        if spec.bpm.min is not None:
            conds.append(("bpm", "t.effective_bpm >= ?", (spec.bpm.min,)))
        if spec.bpm.max is not None:
            conds.append(("bpm", "t.effective_bpm <= ?", (spec.bpm.max,)))
    if spec.genre:
        conds.append(("genre", "hades_genre_match(t.effective_genre, ?)",
                      (json.dumps(spec.genre),)))
    if spec.folder:
        conds.append(("folder", "hades_folder_match(t.directory, ?)", (spec.folder,)))
    if spec.artists:
        conds.append(("artist", "hades_artist_match(g.artist, ?)", (json.dumps(spec.artists),)))
    if spec.exclude_artists:
        conds.append(("artist", "NOT COALESCE(hades_artist_match(g.artist, ?), 0)",
                      (json.dumps(spec.exclude_artists),)))
    return conds


# Exclusions worth reporting: (filters set aside, SQL detecting the missing or conflicting
# value, reason). A track is counted if it passes every other filter.
_MISSING_DATA: tuple[tuple[frozenset[str], str, str], ...] = (
    (frozenset({"date"}), "t.file_created IS NULL", "no filesystem creation date"),
    (frozenset({"bpm_known", "bpm_trusted", "bpm"}), "t.effective_bpm IS NULL", "no BPM"),
    (frozenset({"bpm_trusted"}), f"NOT {_BPM_TRUSTED}", "tag and analyzed BPM conflict"),
    (frozenset({"genre"}), "t.effective_genre IS NULL OR trim(t.effective_genre) = ''",
     "no genre"),
    (frozenset({"duration"}), "t.duration IS NULL OR t.duration <= 0", "unknown duration"),
)


def find_candidates(conn: sqlite3.Connection, spec: PlaylistSpec) -> CandidatePool:
    register_functions(conn)
    conds = _conditions(spec)

    def where(items: list[tuple[str, str, tuple[object, ...]]]) -> tuple[str, list[object]]:
        sql = " AND ".join(f"({c})" for _, c, _ in items) or "1"
        return sql, [p for _, _, ps in items for p in ps]

    sql, params = where(conds)
    rows = conn.execute(f"{_SELECT} WHERE {sql} ORDER BY t.id", params).fetchall()
    pool = CandidatePool([])
    for row in rows:
        track = _track(row, check_file=False)
        if not filters_match(track, spec.filters):
            if missing := _missing_for_filters(track, spec.filters):
                pool.excluded[f"no {' or '.join(missing)}"] += 1
            continue
        if os.path.isfile(track.path):
            pool.tracks.append(track)
        else:
            pool.excluded["file not found on disk"] += 1

    # Count tracks that pass every other filter but lack the data a filter needs.
    names = {name for name, _, _ in conds}
    for set_aside, missing_sql, reason in _MISSING_DATA:
        if not set_aside & names:
            continue
        others, others_params = where([c for c in conds if c[0] not in set_aside])
        n = conn.execute(
            f"SELECT COUNT(*) FROM tracks t LEFT JOIN track_tags g ON g.track_id = t.id "
            f"WHERE {others} AND ({missing_sql})", others_params).fetchone()[0]
        if n:
            pool.excluded[reason] += n
    return pool


# --------------------------------------------------------------------------- validation


def track_violations(track: Track, spec: PlaylistSpec) -> list[str]:
    """Hard filters this track does not satisfy (empty if it satisfies all)."""
    out: list[str] = []
    if not track.available:
        out.append("file unavailable" if track.status == "available" else f"status {track.status}")
    if not track.duration or track.duration <= 0:
        out.append("unknown duration")
    created = track.created_date
    if spec.created_after or spec.created_before:
        if created is None:
            out.append("no creation date")
        elif (spec.created_after and created <= spec.created_after) or (
                spec.created_before and created >= spec.created_before):
            out.append(f"created {created}")
    if spec.bpm is not None or spec.ordering.max_bpm_step is not None:
        if track.bpm is None:
            out.append("no BPM")
        elif not track.bpm_reliable:
            out.append("conflicting BPM")
        elif spec.bpm is not None:
            if spec.bpm.min is not None and track.bpm < spec.bpm.min:
                out.append(f"BPM {track.bpm:.0f} < {spec.bpm.min:g}")
            if spec.bpm.max is not None and track.bpm > spec.bpm.max:
                out.append(f"BPM {track.bpm:.0f} > {spec.bpm.max:g}")
    if spec.genre and not genre_matches(track.genre, spec.genre):
        out.append("no genre" if not track.genre else f"genre {track.genre}")
    if spec.folder and not folder_matches(track.directory, spec.folder):
        out.append("outside folder")
    if spec.artists and not artist_matches(track.artist, spec.artists):
        out.append("artist not requested")
    if spec.exclude_artists and artist_matches(track.artist, spec.exclude_artists):
        out.append("excluded artist")
    for group in spec.filters:
        if not any(condition_holds(track, c) for c in group):
            missing = _missing_for_filters(track, [group])
            out.append(f"no {' or '.join(missing)}" if missing else f"not {describe_group(group)}")
    return out


# --------------------------------------------------------------------------- free-form filters


def field_value(track: Track, name: str) -> object:
    """The value a filter condition tests; None if missing or unreliable."""
    if name == "bpm":
        return track.bpm if track.bpm_reliable else None
    if name == "key":
        return track.camelot if track.key_reliable else None
    if name == "duration_minutes":
        return track.duration / 60 if track.duration else None
    if name == "loudness_lufs":
        return track.loudness
    if name == "added":
        return track.created_date
    if name == "folder":
        return track.directory
    if name == "key_reliable":
        return track.key_reliable
    if name == "bpm_reliable":
        return track.bpm_reliable
    return getattr(track, name)  # genre, artist, title, album, year, format


def condition_holds(track: Track, c: Condition) -> bool | None:
    """Whether the track satisfies the condition; None if the value is missing."""
    v = field_value(track, c.field)
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    result = _test(c, v)
    return not result if c.negate else result


def _test(c: Condition, v: object) -> bool:
    f, vals = c.field, c.values
    if f in FLAG_FIELDS:
        return v == vals[0]
    if f in KEY_FIELDS:
        return v in vals
    if f in TEXT_FIELDS:
        text = str(v)
        texts = [str(x) for x in vals]
        if f == "genre":
            return genre_matches(text, texts)
        if f == "artist":
            return artist_matches(text, texts)
        if f == "folder":
            return any(folder_matches(text, x) for x in texts)
        if f == "format":
            return text.casefold() in {x.casefold() for x in texts}
        hay = normalize_text(text)  # title, album: whole words
        return any(_contains_words(hay, normalize_text(x)) for x in texts)
    if f in DATE_FIELDS:
        x: list = [date.fromisoformat(str(a)) for a in vals]
    else:
        x = [float(a) for a in vals]  # type: ignore[arg-type]
        v = float(v)  # type: ignore[arg-type]
    op = c.op
    if op == "between":
        return x[0] <= v <= x[1]  # type: ignore[operator]
    if op in ("eq", "in"):
        if f in DATE_FIELDS:
            return v in x
        return any(round(v) == round(a) for a in x)  # type: ignore[call-overload]
    return {"lt": v < x[0], "le": v <= x[0], "gt": v > x[0],  # type: ignore[operator]
            "ge": v >= x[0]}[op]  # type: ignore[operator]


def filters_match(track: Track, filters: list[list[Condition]]) -> bool:
    """Every group has at least one condition that holds."""
    return all(any(condition_holds(track, c) for c in group) for group in filters)


def _missing_for_filters(track: Track, filters: list[list[Condition]]) -> list[str]:
    """Labels of missing fields that make the track fail, if missing data is the only reason."""
    missing: list[str] = []
    for group in filters:
        results = [condition_holds(track, c) for c in group]
        if any(results):
            continue
        absent = [c for c, r in zip(group, results) if r is None]
        if not absent:
            return []  # fails on actual values
        for c in absent:
            label = {"bpm": "trusted BPM", "key": "certain key"}.get(c.field, FIELD_LABELS[c.field])
            if label not in missing:
                missing.append(label)
    return missing
