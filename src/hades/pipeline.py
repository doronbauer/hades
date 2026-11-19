"""Orchestrates the stages against the database.

* ``scan``: discovery (stage 1) + metadata extraction (stage 2)
* ``analyze``: audio analysis (stage 3)
* both reconcile (stage 4) every track they touch; ``reconcile_all`` redoes it for all.

Every unit of work is committed on its own, so an interrupted run loses at
most the file in progress; the next run picks up whatever is still pending.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field

from hades.analysis import AnalysisError, Analyzer, AnalysisResult, AnalysisSkipped
from hades.config import Config
from hades.db import now_iso, timestamp_iso
from hades.discovery import DiscoveredFile, check_root, is_under, iter_audio_files
from hades.keys import from_essentia, parse_key
from hades.metadata import EXTRACTOR, ExtractedMetadata, MetadataError, extract_metadata
from hades.progress import NullProgress, Progress
from hades.reconcile import parse_bpm, resolve_bpm, resolve_genre, resolve_key

BATCH = 200


@dataclass
class ScanSummary:
    run_id: int
    discovered: int = 0
    new: int = 0
    changed: int = 0
    skipped: int = 0
    missing: int = 0
    extracted: int = 0
    failed: int = 0
    unavailable_roots: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalyzeSummary:
    run_id: int
    pending: int = 0
    analyzed: int = 0
    skipped: int = 0
    failed: int = 0
    analyzer_version: str = ""


# --------------------------------------------------------------------------- runs


def _start_run(conn: sqlite3.Connection, kind: str) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO scan_runs (kind, started_at, status) VALUES (?, ?, 'running')",
            (kind, now_iso()),
        )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def _finish_run(
    conn: sqlite3.Connection, summary: ScanSummary | AnalyzeSummary, status: str
) -> None:
    conn.rollback()
    if isinstance(summary, ScanSummary):
        counts = dict(
            discovered=summary.discovered,
            new=summary.new,
            changed=summary.changed,
            skipped=summary.skipped,
            missing=summary.missing,
            extracted=summary.extracted,
            failed=summary.failed,
        )
        unavailable = json.dumps(summary.unavailable_roots) if summary.unavailable_roots else None
        notes = "\n".join(summary.warnings) or None
    else:
        counts = dict(
            discovered=summary.pending,
            analyzed=summary.analyzed,
            skipped=summary.skipped,
            failed=summary.failed,
        )
        unavailable = None
        notes = f"analyzer: {summary.analyzer_version}"
    assignments = ", ".join(f"{k} = :{k}" for k in counts)
    with conn:
        conn.execute(
            f"UPDATE scan_runs SET {assignments}, finished_at = :finished, status = :status,"
            " unavailable_roots = :unavailable, notes = :notes WHERE id = :id",
            {
                **counts,
                "finished": now_iso(),
                "status": status,
                "unavailable": unavailable,
                "notes": notes,
                "id": summary.run_id,
            },
        )


# --------------------------------------------------------------------------- scan


def scan(
    conn: sqlite3.Connection,
    config: Config,
    *,
    retry_failed: bool = False,
    progress: Progress | None = None,
) -> ScanSummary:
    progress = progress or NullProgress()
    summary = ScanSummary(run_id=_start_run(conn, "scan"))
    try:
        _discover(conn, config, summary, progress)
        _extract_pending(conn, summary, progress, retry_failed)
    except BaseException as exc:
        progress.finish()
        _finish_run(conn, summary, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        raise
    progress.finish()
    _finish_run(conn, summary, "completed")
    return summary


def _discover(
    conn: sqlite3.Connection, config: Config, summary: ScanSummary, progress: Progress
) -> None:
    run_id = summary.run_id
    now = now_iso()
    seen: set[tuple[int, int]] = set()
    # (root, unreadable paths, files found under the root including duplicates)
    scanned: list[tuple[str, list[str], int]] = []

    roots: list[str] = []
    for configured in config.music_directories:
        root = os.path.realpath(configured)
        if root not in roots:
            roots.append(root)

    for root in roots:
        reason = check_root(root)
        if reason is not None:
            summary.unavailable_roots.append((root, reason))
            progress.note(f"Skipping {root}: {reason}. Its tracks are left untouched.")
            continue
        unreadable: list[str] = []
        found = 0
        with conn:
            for f in iter_audio_files(root, unreadable):
                found += 1
                # Overlapping roots, hard links and case-variant paths are the same file.
                if (f.device, f.inode) in seen:
                    continue
                seen.add((f.device, f.inode))
                summary.discovered += 1
                progress.update(summary.discovered, None, f"Discovering {f.path}")
                outcome = _record_discovered(conn, f, root, run_id, now)
                setattr(summary, outcome, getattr(summary, outcome) + 1)
        for path in unreadable:
            progress.note(f"Could not read {path}; tracks under it are left untouched.")
        scanned.append((root, unreadable, found))

    with conn:
        for root, unreadable, found in scanned:
            _mark_missing(conn, root, unreadable, found, summary)


def _record_discovered(
    conn: sqlite3.Connection, f: DiscoveredFile, root: str, run_id: int, now: str
) -> str:
    fs = {
        "path": f.path,
        "filename": os.path.basename(f.path),
        "directory": os.path.dirname(f.path),
        "root": root,
        "format": f.format,
        "file_size": f.size,
        "file_mtime_ns": f.mtime_ns,
        "file_modified": timestamp_iso(f.mtime_ns / 1e9),
        "file_created": timestamp_iso(f.created) if f.created is not None else None,
        "content_signature": f.signature,
        "now": now,
        "run_id": run_id,
    }
    row = conn.execute(
        "SELECT id, content_signature, status, last_error FROM tracks WHERE path = ?", (f.path,)
    ).fetchone()
    if row is None:
        conn.execute(
            """INSERT INTO tracks (path, filename, directory, root, format, file_size,
                   file_mtime_ns, file_modified, file_created, content_signature,
                   first_indexed, last_seen, last_seen_run, status)
               VALUES (:path, :filename, :directory, :root, :format, :file_size,
                   :file_mtime_ns, :file_modified, :file_created, :content_signature,
                   :now, :now, :run_id, 'available')""",
            fs,
        )
        return "new"
    if row["content_signature"] != f.signature:
        # metadata_signature now differs from content_signature -> metadata pending;
        # analyses stored under the old signature no longer count -> analysis pending.
        conn.execute(
            """UPDATE tracks SET root = :root, format = :format, file_size = :file_size,
                   file_mtime_ns = :file_mtime_ns, file_modified = :file_modified,
                   file_created = :file_created, content_signature = :content_signature,
                   last_seen = :now, last_seen_run = :run_id,
                   status = 'available', last_error = NULL
               WHERE path = :path""",
            fs,
        )
        return "changed"
    status = row["status"]
    if status == "missing":
        status = "error" if row["last_error"] else "available"
    conn.execute(
        "UPDATE tracks SET root = ?, last_seen = ?, last_seen_run = ?, status = ? WHERE id = ?",
        (root, now, run_id, status, row["id"]),
    )
    return "skipped"


def _mark_missing(
    conn: sqlite3.Connection,
    root: str,
    unreadable: list[str],
    found: int,
    summary: ScanSummary,
) -> None:
    stale = conn.execute(
        """SELECT id, path FROM tracks
           WHERE root = ? AND status != 'missing' AND last_seen_run IS NOT ?""",
        (root, summary.run_id),
    ).fetchall()
    if not stale:
        return
    if found == 0:
        # An empty mount point or a half-mounted drive looks exactly like "everything
        # was deleted". Refuse to mark a whole root missing on that evidence.
        msg = (
            f"{root} contained no audio files; {len(stale)} previously indexed tracks "
            "were NOT marked missing"
        )
        summary.warnings.append(msg)
        return
    ids = [row["id"] for row in stale if not is_under(row["path"], unreadable)]
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        conn.execute(
            f"UPDATE tracks SET status = 'missing' WHERE id IN ({','.join('?' * len(chunk))})",
            chunk,
        )
    summary.missing += len(ids)


def _extract_pending(
    conn: sqlite3.Connection, summary: ScanSummary, progress: Progress, retry_failed: bool
) -> None:
    if retry_failed:
        with conn:
            conn.execute("UPDATE tracks SET metadata_signature = NULL WHERE status = 'error'")
    # Pending: file changed since tags were read, or tags were read by an older extractor.
    pending = """FROM tracks t WHERE t.status != 'missing'
                 AND (t.metadata_signature IS NULL OR t.metadata_signature != t.content_signature
                      OR (t.status != 'error' AND NOT EXISTS (
                          SELECT 1 FROM track_tags g
                          WHERE g.track_id = t.id AND g.extractor = :extractor)))"""
    params = {"extractor": EXTRACTOR}
    total = conn.execute(f"SELECT COUNT(*) {pending}", params).fetchone()[0]
    done = 0
    last_id = 0
    while True:
        rows = conn.execute(
            f"SELECT t.id, t.path, t.content_signature {pending} AND t.id > :last"
            " ORDER BY t.id LIMIT :batch",
            {**params, "last": last_id, "batch": BATCH},
        ).fetchall()
        if not rows:
            break
        for row in rows:
            last_id = row["id"]
            done += 1
            progress.update(done, total, f"Reading tags {row['path']}")
            try:
                meta = extract_metadata(row["path"])
            except MetadataError as exc:
                summary.failed += 1
                progress.failure(row["path"], str(exc))
                with conn:
                    conn.execute(
                        """UPDATE tracks SET status = 'error', last_error = ?,
                               metadata_signature = ?, metadata_extracted_at = ?
                           WHERE id = ?""",
                        (str(exc), row["content_signature"], now_iso(), row["id"]),
                    )
                continue
            with conn:
                store_metadata(conn, row["id"], row["content_signature"], meta)
                reconcile_track(conn, row["id"])
            summary.extracted += 1


def store_metadata(
    conn: sqlite3.Connection, track_id: int, signature: str, meta: ExtractedMetadata
) -> None:
    t, tech, now = meta.tags, meta.technical, now_iso()
    conn.execute(
        """UPDATE tracks SET duration = ?, bitrate = ?, sample_rate = ?, channels = ?,
               bit_depth = ?, metadata_signature = ?, metadata_extracted_at = ?,
               status = 'available', last_error = NULL
           WHERE id = ?""",
        (tech.duration, tech.bitrate, tech.sample_rate, tech.channels, tech.bit_depth,
         signature, now, track_id),
    )
    tag_key = parse_key(t.key_raw)
    conn.execute(
        """INSERT OR REPLACE INTO track_tags (track_id, title, artist, album_artist, album,
               genre, date, year, label, catalog_number, bpm_raw, bpm, key_raw, key,
               remixer, version, isrc, track_number, track_total, disc_number, disc_total,
               comment, composer, raw_json, extractor, extracted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (track_id, t.title, t.artist, t.album_artist, t.album, t.genre, t.date, t.year,
         t.label, t.catalog_number, t.bpm_raw, parse_bpm(t.bpm_raw), t.key_raw,
         tag_key.name if tag_key else None, t.remixer, t.version, t.isrc, t.track_number,
         t.track_total, t.disc_number, t.disc_total, t.comment, t.composer,
         json.dumps(meta.raw, ensure_ascii=False, sort_keys=True, default=str),
         EXTRACTOR, now),
    )
    conn.execute("DELETE FROM track_genres WHERE track_id = ? AND source = 'tag'", (track_id,))
    conn.executemany(
        "INSERT INTO track_genres (track_id, source, genre, rank) VALUES (?, 'tag', ?, ?)",
        [(track_id, genre, rank) for rank, genre in enumerate(t.genres, start=1)],
    )


# --------------------------------------------------------------------------- analyze


def analyze(
    conn: sqlite3.Connection,
    config: Config,
    analyzer: Analyzer,
    *,
    limit: int | None = None,
    retry_failed: bool = False,
    force: bool = False,
    progress: Progress | None = None,
) -> AnalyzeSummary:
    """Analyze tracks lacking a result for this analyzer version and file content.

    ``retry_failed`` also re-runs tracks whose analysis errored; ``force``
    re-runs everything that is not missing.
    """
    progress = progress or NullProgress()
    summary = AnalyzeSummary(run_id=_start_run(conn, "analyze"), analyzer_version=analyzer.version)
    params = {"version": analyzer.version}
    where = "t.status != 'missing'"
    if not force:
        retry_clause = "AND a.status != 'error'" if retry_failed else ""
        where += f""" AND NOT EXISTS (
            SELECT 1 FROM audio_analysis a
            WHERE a.track_id = t.id AND a.analyzer_version = :version
              AND a.content_signature = t.content_signature {retry_clause})"""
    total = conn.execute(f"SELECT COUNT(*) FROM tracks t WHERE {where}", params).fetchone()[0]
    if limit is not None:
        total = min(total, limit)
    summary.pending = total

    max_seconds = config.analysis.max_duration_seconds
    done = 0
    last_id = 0
    try:
        while done < total:
            rows = conn.execute(
                f"""SELECT t.id, t.path, t.content_signature, t.duration FROM tracks t
                    WHERE {where} AND t.id > :last ORDER BY t.id LIMIT :batch""",
                {**params, "last": last_id, "batch": min(BATCH, total - done)},
            ).fetchall()
            if not rows:
                break
            for row in rows:
                last_id = row["id"]
                done += 1
                progress.update(done, total, f"Analyzing {row['path']}")
                result: AnalysisResult | None = None
                error: str | None = None
                try:
                    # Check the tagged duration first to avoid decoding a 2-hour mix at all.
                    if row["duration"] is not None and row["duration"] > max_seconds:
                        raise AnalysisSkipped(
                            f"duration {row['duration']:.0f}s exceeds "
                            f"max_duration_seconds={max_seconds:.0f}"
                        )
                    result = analyzer.analyze(row["path"])
                    status = "ok"
                    summary.analyzed += 1
                except AnalysisSkipped as exc:
                    status, error = "skipped", str(exc)
                    summary.skipped += 1
                    progress.note(f"Skipped {row['path']}: {exc}")
                except Exception as exc:  # one bad file must not stop the run
                    status = "error"
                    error = str(exc) if isinstance(exc, AnalysisError) else f"{type(exc).__name__}: {exc}"
                    summary.failed += 1
                    progress.failure(row["path"], error)
                with conn:
                    store_analysis(conn, row["id"], row["content_signature"], analyzer, status, result, error)
                    reconcile_track(conn, row["id"])
    except BaseException as exc:
        progress.finish()
        _finish_run(conn, summary, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        raise
    progress.finish()
    _finish_run(conn, summary, "completed")
    return summary


def store_analysis(
    conn: sqlite3.Connection,
    track_id: int,
    signature: str,
    analyzer: Analyzer,
    status: str,
    result: AnalysisResult | None,
    error: str | None,
) -> None:
    key = from_essentia(result.key_tonic, result.key_scale) if result and result.key_tonic and result.key_scale else None
    values = {
        "track_id": track_id,
        "analyzer": analyzer.name,
        "analyzer_version": analyzer.version,
        "settings_json": json.dumps(analyzer.settings, sort_keys=True),
        "content_signature": signature,
        "status": status,
        "error": error,
        "bpm": result.bpm if result else None,
        "bpm_confidence": result.bpm_confidence if result else None,
        "key_tonic": result.key_tonic if result else None,
        "key_scale": result.key_scale if result else None,
        "key": key.name if key else None,
        "key_strength": result.key_strength if result else None,
        "loudness_lufs": result.loudness_lufs if result else None,
        "loudness_range_lu": result.loudness_range_lu if result else None,
        "dynamic_complexity": result.dynamic_complexity if result else None,
        "analyzed_seconds": result.analyzed_seconds if result else None,
        "extra_json": json.dumps(result.extra, sort_keys=True) if result and result.extra else None,
        "analyzed_at": now_iso(),
    }
    columns = ", ".join(values)
    placeholders = ", ".join(f":{c}" for c in values)
    updates = ", ".join(f"{c} = excluded.{c}" for c in values if c not in ("track_id", "analyzer_version"))
    conn.execute(
        f"""INSERT INTO audio_analysis ({columns}) VALUES ({placeholders})
            ON CONFLICT (track_id, analyzer_version) DO UPDATE SET {updates}""",
        values,
    )


# --------------------------------------------------------------------------- reconcile


def reconcile_track(conn: sqlite3.Connection, track_id: int) -> None:
    tags = conn.execute(
        "SELECT bpm, key_raw, genre FROM track_tags WHERE track_id = ?", (track_id,)
    ).fetchone()
    analysis = latest_analysis(conn, track_id)
    overrides = {
        r["field"]: r["value"]
        for r in conn.execute(
            "SELECT field, value FROM manual_overrides WHERE track_id = ?", (track_id,)
        )
    }
    analyzed_key = (
        from_essentia(analysis["key_tonic"], analysis["key_scale"])
        if analysis is not None and analysis["key_tonic"] and analysis["key_scale"]
        else None
    )
    bpm = resolve_bpm(
        tags["bpm"] if tags else None,
        analysis["bpm"] if analysis else None,
        parse_bpm(overrides.get("bpm")),
    )
    key = resolve_key(tags["key_raw"] if tags else None, analyzed_key, overrides.get("key"))
    genre = resolve_genre(tags["genre"] if tags else None, overrides.get("genre"))
    conn.execute(
        """UPDATE tracks SET effective_bpm = ?, effective_bpm_source = ?, bpm_flag = ?,
               effective_key = ?, effective_key_camelot = ?, effective_key_source = ?,
               key_flag = ?, effective_genre = ?, effective_genre_source = ?, reconciled_at = ?
           WHERE id = ?""",
        (bpm.value, bpm.source, bpm.flag,
         key.value.name if key.value else None, key.value.camelot if key.value else None,
         key.source, key.flag, genre.value, genre.source, now_iso(), track_id),
    )


def latest_analysis(conn: sqlite3.Connection, track_id: int) -> sqlite3.Row | None:
    """Most recent successful analysis of the file's *current* content, any analyzer version."""
    return conn.execute(
        """SELECT a.* FROM audio_analysis a JOIN tracks t ON t.id = a.track_id
           WHERE a.track_id = ? AND a.status = 'ok' AND a.content_signature = t.content_signature
           ORDER BY a.analyzed_at DESC, a.id DESC LIMIT 1""",
        (track_id,),
    ).fetchone()


def reconcile_all(conn: sqlite3.Connection, progress: Progress | None = None) -> int:
    progress = progress or NullProgress()
    ids = [r[0] for r in conn.execute("SELECT id FROM tracks ORDER BY id")]
    for i in range(0, len(ids), BATCH):
        with conn:
            for track_id in ids[i : i + BATCH]:
                reconcile_track(conn, track_id)
        progress.update(min(i + BATCH, len(ids)), len(ids), "Reconciling")
    progress.finish()
    return len(ids)
