"""SQLite connection handling and schema migrations.

The schema version is stored in ``PRAGMA user_version``. To change the schema,
append a new script to ``MIGRATIONS``; never edit a script that has shipped.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_V1 = """
CREATE TABLE tracks (
    id                    INTEGER PRIMARY KEY,
    path                  TEXT NOT NULL UNIQUE,      -- absolute path as found during the scan
    filename              TEXT NOT NULL,
    directory             TEXT NOT NULL,
    root                  TEXT NOT NULL,             -- configured music directory it was found under
    format                TEXT NOT NULL,             -- MP3, AIFF, WAV, FLAC, M4A
    file_size             INTEGER NOT NULL,
    file_mtime_ns         INTEGER NOT NULL,
    file_modified         TEXT NOT NULL,             -- ISO-8601 UTC
    file_created          TEXT,                      -- filesystem birth time; NOT a purchase date
    content_signature     TEXT NOT NULL,             -- "size:mtime_ns"; a change triggers re-processing
    duration              REAL,                      -- seconds
    bitrate               REAL,                      -- kbps
    sample_rate           INTEGER,                   -- Hz
    channels              INTEGER,
    bit_depth             INTEGER,
    first_indexed         TEXT NOT NULL,
    last_seen             TEXT NOT NULL,
    last_seen_run         INTEGER,
    status                TEXT NOT NULL DEFAULT 'available'
                          CHECK (status IN ('available', 'missing', 'error')),
    last_error            TEXT,                      -- metadata read error, when status = 'error'
    metadata_signature    TEXT,                      -- content_signature at last metadata extraction
    metadata_extracted_at TEXT,
    -- Effective (reconciled) values used for future queries
    effective_bpm         REAL,
    effective_bpm_source  TEXT,                      -- manual | tag | analysis
    bpm_flag              TEXT,                      -- CONSISTENT | HALF_DOUBLE | CONFLICT
    effective_key         TEXT,                      -- e.g. "Am", "F#"
    effective_key_camelot TEXT,                      -- e.g. "8A"
    effective_key_source  TEXT,
    key_flag              TEXT,                      -- CONSISTENT | RELATIVE | PARALLEL | CONFLICT | UNRECOGNIZED
    effective_genre       TEXT,
    effective_genre_source TEXT,
    reconciled_at         TEXT
);
CREATE INDEX idx_tracks_status ON tracks(status);
CREATE INDEX idx_tracks_root ON tracks(root);
CREATE INDEX idx_tracks_effective_bpm ON tracks(effective_bpm);
CREATE INDEX idx_tracks_effective_key ON tracks(effective_key);
CREATE INDEX idx_tracks_effective_key_camelot ON tracks(effective_key_camelot);
CREATE INDEX idx_tracks_effective_genre ON tracks(effective_genre);
CREATE INDEX idx_tracks_file_created ON tracks(file_created);
CREATE INDEX idx_tracks_file_modified ON tracks(file_modified);
CREATE INDEX idx_tracks_first_indexed ON tracks(first_indexed);

-- Embedded metadata exactly as tagged (normalized columns + full raw JSON).
CREATE TABLE track_tags (
    track_id        INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    title           TEXT,
    artist          TEXT,
    album_artist    TEXT,
    album           TEXT,
    genre           TEXT,             -- all tagged genres joined with "; "
    date            TEXT,             -- release date/year as tagged
    year            INTEGER,
    label           TEXT,
    catalog_number  TEXT,
    bpm_raw         TEXT,
    bpm             REAL,             -- bpm_raw parsed; NULL if absent or not a plausible tempo
    key_raw         TEXT,
    key             TEXT,             -- key_raw normalized; NULL if unrecognized
    remixer         TEXT,
    version         TEXT,             -- mix/version/subtitle, only when explicitly tagged
    isrc            TEXT,
    track_number    INTEGER,
    track_total     INTEGER,
    disc_number     INTEGER,
    disc_total      INTEGER,
    comment         TEXT,
    composer        TEXT,
    raw_json        TEXT NOT NULL,    -- every value the extractor returned, for traceability
    extractor       TEXT NOT NULL,    -- e.g. "tinytag 2.3.2"
    extracted_at    TEXT NOT NULL
);
CREATE INDEX idx_track_tags_genre ON track_tags(genre);
CREATE INDEX idx_track_tags_bpm ON track_tags(bpm);
CREATE INDEX idx_track_tags_year ON track_tags(year);
CREATE INDEX idx_track_tags_artist ON track_tags(artist);
CREATE INDEX idx_track_tags_label ON track_tags(label);

-- One row per genre per source. v0.1 writes source = 'tag'; a future classifier
-- can write e.g. source = 'model:<name>' with scores without touching the tags.
CREATE TABLE track_genres (
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    source   TEXT NOT NULL,
    genre    TEXT NOT NULL,
    score    REAL,
    rank     INTEGER NOT NULL,
    PRIMARY KEY (track_id, source, genre)
);
CREATE INDEX idx_track_genres_genre ON track_genres(genre);

-- Independent analysis results. One row per track per analyzer version, so
-- results from different algorithms/settings can coexist.
CREATE TABLE audio_analysis (
    id                  INTEGER PRIMARY KEY,
    track_id            INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    analyzer            TEXT NOT NULL,      -- e.g. "essentia"
    analyzer_version    TEXT NOT NULL,      -- library version + pipeline revision + settings hash
    settings_json       TEXT NOT NULL,
    content_signature   TEXT NOT NULL,      -- file signature the analysis was computed from
    status              TEXT NOT NULL CHECK (status IN ('ok', 'error', 'skipped')),
    error               TEXT,
    bpm                 REAL,
    bpm_confidence      REAL,               -- algorithm-specific; not a probability
    key_tonic           TEXT,
    key_scale           TEXT,               -- major | minor
    key                 TEXT,               -- normalized, e.g. "Am"
    key_strength        REAL,               -- algorithm-specific; not a probability
    loudness_lufs       REAL,               -- EBU R128 integrated loudness
    loudness_range_lu   REAL,               -- EBU R128 loudness range
    dynamic_complexity  REAL,               -- Essentia DynamicComplexity, dB
    analyzed_seconds    REAL,
    extra_json          TEXT,
    analyzed_at         TEXT NOT NULL,
    UNIQUE (track_id, analyzer_version)
);
CREATE INDEX idx_audio_analysis_track ON audio_analysis(track_id);
CREATE INDEX idx_audio_analysis_bpm ON audio_analysis(bpm);
CREATE INDEX idx_audio_analysis_key ON audio_analysis(key);

-- Manual corrections take precedence over tags and analysis. No editing UI in v0.1.
CREATE TABLE manual_overrides (
    track_id   INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    field      TEXT NOT NULL CHECK (field IN ('bpm', 'key', 'genre')),
    value      TEXT NOT NULL,
    note       TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (track_id, field)
);

CREATE TABLE scan_runs (
    id                INTEGER PRIMARY KEY,
    kind              TEXT NOT NULL CHECK (kind IN ('scan', 'analyze')),
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    status            TEXT NOT NULL CHECK (status IN ('running', 'completed', 'interrupted', 'failed')),
    discovered        INTEGER NOT NULL DEFAULT 0,
    new               INTEGER NOT NULL DEFAULT 0,
    changed           INTEGER NOT NULL DEFAULT 0,
    skipped           INTEGER NOT NULL DEFAULT 0,   -- unchanged files, or analyses skipped
    missing           INTEGER NOT NULL DEFAULT 0,   -- newly marked missing
    extracted         INTEGER NOT NULL DEFAULT 0,
    analyzed          INTEGER NOT NULL DEFAULT 0,
    failed            INTEGER NOT NULL DEFAULT 0,
    unavailable_roots TEXT,                         -- JSON list of [directory, reason]
    notes             TEXT
);
"""

MIGRATIONS: list[str] = [SCHEMA_V1]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="seconds")


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection) -> None:
    current = schema_version(conn)
    if current > len(MIGRATIONS):
        raise RuntimeError(
            f"database schema version {current} is newer than this hades supports "
            f"({len(MIGRATIONS)}); upgrade hades"
        )
    for version, script in enumerate(MIGRATIONS[current:], start=current + 1):
        try:
            conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {version};\nCOMMIT;")
        except BaseException:
            conn.rollback()
            raise
