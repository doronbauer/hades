"""Stage 5: summaries, per-track inspection, conflict listing and CSV export."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import unicodedata
from datetime import datetime
from typing import TextIO

from hades.keys import parse_key
from hades.pipeline import latest_analysis
from hades.reconcile import BPM_DISCREPANCIES, KEY_DISCREPANCIES

# Name -> SQL expression. "a" is the latest successful analysis of the current file content.
EXPORT_FIELDS: dict[str, str] = {
    "id": "t.id",
    "path": "t.path",
    "status": "t.status",
    "format": "t.format",
    "duration": "t.duration",
    "bitrate": "t.bitrate",
    "sample_rate": "t.sample_rate",
    "artist": "g.artist",
    "title": "g.title",
    "album": "g.album",
    "label": "g.label",
    "year": "g.year",
    "tag_genre": "g.genre",
    "tag_bpm_raw": "g.bpm_raw",
    "tag_bpm": "g.bpm",
    "analyzed_bpm": "a.bpm",
    "bpm_confidence": "a.bpm_confidence",
    "effective_bpm": "t.effective_bpm",
    "bpm_source": "t.effective_bpm_source",
    "bpm_flag": "t.bpm_flag",
    "tag_key_raw": "g.key_raw",
    "tag_key": "g.key",
    "analyzed_key": "a.key",
    "key_strength": "a.key_strength",
    "effective_key": "t.effective_key",
    "camelot": "t.effective_key_camelot",
    "open_key": "hades_open_key(t.effective_key)",
    "key_source": "t.effective_key_source",
    "key_flag": "t.key_flag",
    "loudness_lufs": "a.loudness_lufs",
    "loudness_range_lu": "a.loudness_range_lu",
    "dynamic_complexity": "a.dynamic_complexity",
    "analyzer_version": "a.analyzer_version",
    "file_created": "t.file_created",
    "file_modified": "t.file_modified",
    "first_indexed": "t.first_indexed",
    "last_error": "t.last_error",
}

_TRACK_JOINS = """
FROM tracks t
LEFT JOIN track_tags g ON g.track_id = t.id
LEFT JOIN audio_analysis a ON a.id = (
    SELECT a2.id FROM audio_analysis a2
    WHERE a2.track_id = t.id AND a2.status = 'ok' AND a2.content_signature = t.content_signature
    ORDER BY a2.analyzed_at DESC, a2.id DESC LIMIT 1)
"""


# --------------------------------------------------------------------------- stats


def print_stats(conn: sqlite3.Connection, out: TextIO) -> None:
    def scalar(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0] or 0)

    total = scalar("SELECT COUNT(*) FROM tracks")
    out.write(f"Tracks: {total}\n")
    if total == 0:
        out.write("No tracks indexed yet. Run `hades scan`.\n")
        return

    _section(out, "By status")
    for r in conn.execute("SELECT status, COUNT(*) n FROM tracks GROUP BY status ORDER BY n DESC"):
        _row(out, r["status"], r["n"])

    _section(out, "By format")
    for r in conn.execute("SELECT format, COUNT(*) n FROM tracks GROUP BY format ORDER BY n DESC"):
        _row(out, r["format"], r["n"])

    present = "t.status != 'missing'"
    _section(out, "Audio analysis (current file content, not missing)")
    _row(out, "analyzed", scalar(f"""SELECT COUNT(*) {_TRACK_JOINS} WHERE {present} AND a.id IS NOT NULL"""))
    for status in ("error", "skipped"):
        n = scalar(
            f"""SELECT COUNT(DISTINCT t.id) FROM tracks t JOIN audio_analysis x ON x.track_id = t.id
                AND x.content_signature = t.content_signature AND x.status = '{status}'
                WHERE {present} AND NOT EXISTS (SELECT 1 FROM audio_analysis y WHERE y.track_id = t.id
                AND y.status = 'ok' AND y.content_signature = t.content_signature)"""
        )
        _row(out, f"{status} (no successful result)", n)
    _row(out, "never analyzed", scalar(
        f"""SELECT COUNT(*) FROM tracks t WHERE {present} AND NOT EXISTS (
            SELECT 1 FROM audio_analysis x WHERE x.track_id = t.id
            AND x.content_signature = t.content_signature)"""))
    for r in conn.execute(
        "SELECT analyzer_version, COUNT(*) n FROM audio_analysis WHERE status = 'ok' "
        "GROUP BY analyzer_version ORDER BY n DESC"
    ):
        _row(out, f"results from {r['analyzer_version']}", r["n"])

    _section(out, "Missing embedded metadata (tracks with tags read)")
    tagged = f"FROM tracks t JOIN track_tags g ON g.track_id = t.id WHERE {present}"
    _row(out, "tags not read yet / unreadable",
         scalar(f"SELECT COUNT(*) FROM tracks t WHERE {present} AND NOT EXISTS "
                "(SELECT 1 FROM track_tags g WHERE g.track_id = t.id)"))
    for label, column in (("no title", "title"), ("no artist", "artist"), ("no genre", "genre"),
                          ("no BPM tag", "bpm_raw"), ("no key tag", "key_raw"),
                          ("no label", "label"), ("no year", "year")):
        _row(out, label, scalar(f"SELECT COUNT(*) {tagged} AND g.{column} IS NULL"))

    _section(out, "Discrepancies (tag vs. analysis)")
    for column in ("bpm_flag", "key_flag"):
        for r in conn.execute(
            f"SELECT {column} f, COUNT(*) n FROM tracks WHERE {column} IS NOT NULL "
            f"GROUP BY {column} ORDER BY n DESC"
        ):
            _row(out, f"{column.split('_')[0].upper()} {r['f']}", r["n"])

    run = conn.execute("SELECT * FROM scan_runs WHERE kind = 'scan' ORDER BY id DESC LIMIT 1").fetchone()
    if run:
        _section(out, "Last scan")
        _row(out, "run", f"#{run['id']} {run['status']} at {run['started_at']}")
        _row(out, "found / new / changed / unchanged",
             f"{run['discovered']} / {run['new']} / {run['changed']} / {run['skipped']}")
        _row(out, "marked missing / tag errors", f"{run['missing']} / {run['failed']}")
        if run["unavailable_roots"]:
            for root, reason in json.loads(run["unavailable_roots"]):
                _row(out, "unavailable", f"{root} — {reason}")


# --------------------------------------------------------------------------- inspect


def find_track(conn: sqlite3.Connection, ref: str) -> sqlite3.Row | None:
    """Look a track up by numeric id or by path (absolute, relative or ~)."""
    if ref.isdigit():
        row = conn.execute("SELECT * FROM tracks WHERE id = ?", (int(ref),)).fetchone()
        if row:
            return row
    expanded = os.path.abspath(os.path.expanduser(ref))
    for candidate in dict.fromkeys((expanded, os.path.realpath(expanded))):
        row = conn.execute("SELECT * FROM tracks WHERE path = ?", (candidate,)).fetchone()
        if row:
            return row
    return None


def print_track(conn: sqlite3.Connection, track: sqlite3.Row, out: TextIO) -> None:
    t = track
    out.write(f"Track #{t['id']}  [{t['status']}]\n")
    _row(out, "path", t["path"])
    tech = " ".join(
        part
        for part in (
            t["format"],
            f"{t['sample_rate']} Hz" if t["sample_rate"] else None,
            f"{t['channels']} ch" if t["channels"] else None,
            f"{t['bit_depth']}-bit" if t["bit_depth"] else None,
            f"{t['bitrate']:.0f} kbps" if t["bitrate"] else None,
        )
        if part
    )
    _row(out, "format", tech)
    _row(out, "duration", _duration(t["duration"]))
    _row(out, "size", f"{t['file_size']:,} bytes")
    _row(out, "file created", f"{t['file_created']} (filesystem date, not a purchase date)"
         if t["file_created"] else None)
    _row(out, "file modified", t["file_modified"])
    _row(out, "first indexed", t["first_indexed"])
    _row(out, "last seen", t["last_seen"])
    _row(out, "error", t["last_error"])

    tags = conn.execute("SELECT * FROM track_tags WHERE track_id = ?", (t["id"],)).fetchone()
    if tags is None:
        _section(out, "Embedded tags: not read")
    else:
        _section(out, f"Embedded tags ({tags['extractor']}, {tags['extracted_at']})")
        for column in ("title", "artist", "album_artist", "album", "remixer", "version", "genre",
                       "date", "label", "catalog_number", "isrc", "track_number", "composer",
                       "comment"):
            _row(out, column.replace("_", " "), tags[column])
        if tags["bpm_raw"] is not None:
            parsed = "" if tags["bpm"] is not None else "  (not a valid tempo; ignored)"
            _row(out, "bpm", f"{tags['bpm_raw']}{parsed}")
        if tags["key_raw"] is not None:
            key = parse_key(tags["key_raw"])
            norm = f" -> {key.label}" if key else "  (unrecognized; preserved as-is)"
            _row(out, "key", f"{tags['key_raw']}{norm}")
        out.write("  raw tags:\n")
        raw = json.loads(tags["raw_json"])
        for k in sorted(raw):
            out.write(f"    {k}: {json.dumps(raw[k], ensure_ascii=False)}\n")

    rows = conn.execute(
        "SELECT * FROM audio_analysis WHERE track_id = ? ORDER BY analyzed_at DESC, id DESC",
        (t["id"],),
    ).fetchall()
    current = latest_analysis(conn, t["id"])
    if not rows:
        _section(out, "Audio analysis: none yet")
    for a in rows:
        marker = " (used)" if current is not None and a["id"] == current["id"] else ""
        stale = "" if a["content_signature"] == t["content_signature"] else " (stale: file changed since)"
        _section(out, f"Audio analysis: {a['analyzer_version']} [{a['status']}]{marker}{stale}")
        _row(out, "analyzed at", a["analyzed_at"])
        if a["status"] != "ok":
            _row(out, "reason", a["error"])
            continue
        _row(out, "bpm", _fmt(a["bpm"], "{:.2f}") + _paren("confidence", a["bpm_confidence"]))
        key = f"{_key_label(a['key'])}  {a['key_tonic']} {a['key_scale']}" if a["key"] else None
        _row(out, "key", (key or "-") + _paren("strength", a["key_strength"]))
        _row(out, "loudness", f"{_fmt(a['loudness_lufs'], '{:.1f}')} LUFS, "
                              f"range {_fmt(a['loudness_range_lu'], '{:.1f}')} LU")
        _row(out, "dynamic complexity", _fmt(a["dynamic_complexity"], "{:.2f}"))
        _row(out, "analyzed seconds", a["analyzed_seconds"])

    _section(out, "Effective values (manual > tag > analysis)")
    _row(out, "bpm", _effective(_fmt(t["effective_bpm"], "{:.2f}"), t["effective_bpm_source"], t["bpm_flag"]))
    key = _key_label(t["effective_key"])
    _row(out, "key", _effective(key, t["effective_key_source"], t["key_flag"]))
    _row(out, "genre", _effective(t["effective_genre"], t["effective_genre_source"], None))
    for o in conn.execute("SELECT * FROM manual_overrides WHERE track_id = ?", (t["id"],)):
        _row(out, f"override {o['field']}", f"{o['value']} ({o['created_at']})")


# --------------------------------------------------------------------------- search / list


def _fold(value: str | None) -> str | None:
    # Case- and Unicode-insensitive matching ("björk" finds "Björk"; NFC/NFD tags compare equal).
    return unicodedata.normalize("NFKC", value).casefold() if value is not None else None


def search_tracks(conn: sqlite3.Connection, query: str) -> list[sqlite3.Row]:
    """Tracks whose tagged artist or title contains ``query``; missing tracks excluded."""
    conn.create_function("hades_fold", 1, _fold, deterministic=True)
    needle = _fold(query.strip())
    return conn.execute(
        """SELECT t.id, g.artist, g.title FROM tracks t JOIN track_tags g ON g.track_id = t.id
           WHERE t.status != 'missing'
             AND (instr(hades_fold(g.artist), :q) > 0 OR instr(hades_fold(g.title), :q) > 0)
           ORDER BY hades_fold(g.artist), hades_fold(g.title), t.id""",
        {"q": needle},
    ).fetchall()


def print_search(conn: sqlite3.Connection, query: str, out: TextIO) -> int:
    rows = search_tracks(conn, query)
    if not rows:
        out.write(f"No tracks found for {query!r}.\n")
        return 0
    _table(out, ("ID", "ARTIST", "TITLE"),
           [(str(r["id"]), r["artist"] or "-", r["title"] or "-") for r in rows])
    out.write(f"\n{len(rows)} track(s). Use `hades list <ID>` for details.\n")
    return len(rows)


def print_track_properties(conn: sqlite3.Connection, track: sqlite3.Row, out: TextIO) -> None:
    """Short property sheet for one track (effective values; see `inspect` for sources)."""
    t = track
    tags = conn.execute("SELECT * FROM track_tags WHERE track_id = ?", (t["id"],)).fetchone()
    analysis = latest_analysis(conn, t["id"])

    artist = tags["artist"] if tags else None
    title = tags["title"] if tags else None
    out.write(f"Track #{t['id']}  {artist or '-'} - {title or '-'}\n")
    if t["status"] != "available":
        out.write(f"  (status: {t['status']})\n")

    bpm = "-"
    if t["effective_bpm"] is not None:
        bpm = f"{t['effective_bpm']:.2f}  [{t['effective_bpm_source']}]"
        if t["bpm_flag"] in BPM_DISCREPANCIES:
            bpm += f"  tag and analysis disagree: {t['bpm_flag']}"
    key = "-"
    if t["effective_key"] is not None:
        key = f"{_key_label(t['effective_key'])}  [{t['effective_key_source']}]"
        if t["key_flag"] in KEY_DISCREPANCIES:
            key += f"  tag and analysis disagree: {t['key_flag']}"
    loudness = (
        f"{analysis['loudness_lufs']:.1f} LUFS"
        if analysis is not None and analysis["loudness_lufs"] is not None
        else "- (not analyzed)" if analysis is None else "-"
    )

    for label, value in (
        ("bpm", bpm),
        ("key", key),
        ("genre", t["effective_genre"]),
        ("file created", f"{_local_time(t['file_created'])} (on this system)"
         if t["file_created"] else None),
        ("publication date", tags["date"] if tags else None),
        ("label", tags["label"] if tags else None),
        ("loudness", loudness),
        ("format", t["format"]),
        ("duration", _mm_ss(t["duration"])),
    ):
        out.write(f"  {label:<18} {value if value is not None else '-'}\n")


def _mm_ss(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    m, s = divmod(round(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _key_label(name: str | None) -> str | None:
    """Stored key name -> "Bb (6B / 11d)"; unparseable names are shown as stored."""
    key = parse_key(name)
    return key.label if key else name


def _open_key(name: str | None) -> str | None:
    key = parse_key(name)
    return key.open_key if key else None


def _sql_list(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


def _local_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- conflicts


def print_conflicts(conn: sqlite3.Connection, out: TextIO, field: str | None = None) -> int:
    conditions = []
    if field in (None, "bpm"):
        conditions.append(f"t.bpm_flag IN {_sql_list(BPM_DISCREPANCIES)}")
    if field in (None, "key"):
        conditions.append(f"t.key_flag IN {_sql_list(KEY_DISCREPANCIES)}")
    rows = conn.execute(
        f"""SELECT t.id, t.path, g.artist, g.title, g.bpm_raw, a.bpm abpm, t.bpm_flag,
                   g.key_raw, a.key akey, t.key_flag
            {_TRACK_JOINS}
            WHERE t.status != 'missing' AND ({' OR '.join(conditions)})
            ORDER BY t.bpm_flag IS NULL, t.bpm_flag, t.key_flag, g.artist, g.title"""
    ).fetchall()
    if not rows:
        out.write("No discrepancies.\n")
        return 0
    header = ("ID", "TRACK", "TAG BPM", "ANALYZED", "BPM FLAG", "TAG KEY", "ANALYZED", "KEY FLAG")
    table = [
        (
            str(r["id"]),
            _track_label(r),
            r["bpm_raw"] or "",
            _fmt(r["abpm"], "{:.2f}", ""),
            r["bpm_flag"] if r["bpm_flag"] in BPM_DISCREPANCIES else "",
            r["key_raw"] or "",
            _key_label(r["akey"]) or "",
            r["key_flag"] if r["key_flag"] in KEY_DISCREPANCIES else "",
        )
        for r in rows
    ]
    _table(out, header, table)
    out.write(f"\n{len(rows)} track(s) with discrepancies. Use `hades inspect <ID>` for details.\n")
    return len(rows)


# --------------------------------------------------------------------------- export


def export_csv(
    conn: sqlite3.Connection, out: TextIO, fields: list[str] | None = None,
    include_missing: bool = False,
) -> int:
    fields = fields or list(EXPORT_FIELDS)
    unknown = [f for f in fields if f not in EXPORT_FIELDS]
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(unknown)}")
    conn.create_function("hades_open_key", 1, _open_key, deterministic=True)
    select = ", ".join(f"{EXPORT_FIELDS[f]} AS {f}" for f in fields)
    where = "" if include_missing else "WHERE t.status != 'missing'"
    writer = csv.writer(out)
    writer.writerow(fields)
    count = 0
    for row in conn.execute(f"SELECT {select} {_TRACK_JOINS} {where} ORDER BY t.path"):
        writer.writerow(["" if v is None else v for v in row])
        count += 1
    return count


# --------------------------------------------------------------------------- formatting


def _section(out: TextIO, title: str) -> None:
    out.write(f"\n{title}\n")


def _row(out: TextIO, label: str, value: object) -> None:
    if value is None or value == "":
        return
    out.write(f"  {label:<34} {value}\n")


def _fmt(value: float | None, spec: str, default: str = "-") -> str:
    return spec.format(value) if value is not None else default


def _paren(label: str, value: float | None) -> str:
    return f"  ({label} {value:.2f})" if value is not None else ""


def _effective(value: str | None, source: str | None, flag: str | None) -> str:
    if value is None or value == "-":
        text = "-"
    else:
        text = f"{value}  [source: {source}]"
    return f"{text}  FLAG: {flag}" if flag and flag != "CONSISTENT" else text


def _duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    m, s = divmod(round(seconds), 60)
    return f"{m}:{s:02d} ({seconds:.1f} s)"


def _track_label(r: sqlite3.Row) -> str:
    if r["artist"] or r["title"]:
        label = f"{r['artist'] or '?'} - {r['title'] or '?'}"
    else:
        label = os.path.basename(r["path"])
    return label if len(label) <= 50 else label[:49] + "…"


def _table(out: TextIO, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    out.write("  ".join(h.ljust(w) for h, w in zip(header, widths)).rstrip() + "\n")
    for r in rows:
        out.write("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() + "\n")
