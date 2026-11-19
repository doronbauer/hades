"""Command-line interface: ``hades <command>``."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from hades import __version__
from hades.config import CONFIG_ENV_VAR, Config, ConfigError, load_config, resolve_config_path
from hades.db import connect
from hades.progress import Progress

if TYPE_CHECKING:
    from hades.playlist import Playlist


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hades",
        description="Index a local music library and analyze BPM, key and loudness.",
    )
    parser.add_argument(
        "--config",
        help=f"path to the config file (default: ${CONFIG_ENV_VAR}, then ./config.toml)",
    )
    parser.add_argument("--version", action="version", version=f"hades {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("scan", help="discover files and read embedded metadata incrementally")
    p.add_argument("--retry-failed", action="store_true",
                   help="re-read tags of files that previously failed")
    p.set_defaults(handler=cmd_scan)

    p = sub.add_parser("analyze", help="run audio analysis on pending tracks")
    p.add_argument("--limit", type=_positive_int, help="analyze at most N tracks")
    p.add_argument("--retry-failed", action="store_true",
                   help="also retry tracks whose analysis previously failed")
    p.add_argument("--force", action="store_true",
                   help="re-analyze every track, even if a current result exists")
    p.set_defaults(handler=cmd_analyze)

    p = sub.add_parser("reconcile", help="recompute effective values and flags for all tracks")
    p.set_defaults(handler=cmd_reconcile)

    p = sub.add_parser("stats", help="summarize the library")
    p.set_defaults(handler=cmd_stats)

    p = sub.add_parser("inspect", help="show tags, analysis and effective values for a track")
    p.add_argument("track", help="track path or numeric ID")
    p.set_defaults(handler=cmd_inspect)

    p = sub.add_parser("search", help="find tracks by artist or title")
    p.add_argument("query", help="text to look for in artist or title (case-insensitive)")
    p.set_defaults(handler=cmd_search)

    p = sub.add_parser("list", help="show BPM, key, genre, dates, label, loudness and format")
    p.add_argument("track_id", type=_positive_int, help="track ID (see `hades search`)")
    p.set_defaults(handler=cmd_list)

    p = sub.add_parser("conflicts", help="list tag vs. analysis discrepancies")
    p.add_argument("--field", choices=("bpm", "key"), help="only this field")
    p.set_defaults(handler=cmd_conflicts)

    p = sub.add_parser("export", help="export track data for manual evaluation")
    p.add_argument("--format", choices=("csv",), default="csv")
    p.add_argument("-o", "--output", help="output file (default: stdout)")
    p.add_argument("--fields", help="comma-separated list of fields (default: all)")
    p.add_argument("--include-missing", action="store_true", help="include missing tracks")
    p.add_argument("--list-fields", action="store_true", help="list available fields and exit")
    p.set_defaults(handler=cmd_export)

    p = sub.add_parser("ui", help="interactive playlist creator and player (terminal UI)")
    p.set_defaults(handler=cmd_ui)

    p = sub.add_parser("generate", help="create a playlist from a natural-language request")
    p.add_argument("request", help='e.g. "a one-hour techno set with rising tempo"')
    p.add_argument("--refine", action="store_true",
                   help="let the LLM propose a better selection/order (re-validated)")
    p.add_argument("--save", metavar="NAME", help="save the playlist under this name")
    p.set_defaults(handler=cmd_generate)

    p = sub.add_parser("playlists", help="list saved playlists, or show one")
    p.add_argument("name", nargs="?", help="playlist to show")
    p.set_defaults(handler=cmd_playlists)

    return parser


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(resolve_config_path(args.config))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    conn = connect(config.database_path)
    try:
        handler: Callable[[argparse.Namespace, Config, sqlite3.Connection], int] = args.handler
        return handler(args, config, conn)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed work is saved; run the command again to resume.",
              file=sys.stderr)
        return 130
    finally:
        conn.close()


# --------------------------------------------------------------------------- commands


def cmd_scan(args: argparse.Namespace, config: Config, conn: sqlite3.Connection) -> int:
    from hades.pipeline import scan

    s = scan(conn, config, retry_failed=args.retry_failed, progress=Progress())
    print(f"Scan #{s.run_id} complete")
    print(f"  files found      {s.discovered}")
    print(f"  new              {s.new}")
    print(f"  changed          {s.changed}")
    print(f"  unchanged        {s.skipped}")
    print(f"  marked missing   {s.missing}")
    print(f"  tags read        {s.extracted}")
    print(f"  tag read errors  {s.failed}")
    for root, reason in s.unavailable_roots:
        print(f"  UNAVAILABLE      {root}: {reason} (tracks left untouched)")
    for warning in s.warnings:
        print(f"  WARNING          {warning}")
    return 0


def cmd_analyze(args: argparse.Namespace, config: Config, conn: sqlite3.Connection) -> int:
    from hades.analysis import AnalysisError, EssentiaAnalyzer
    from hades.pipeline import analyze

    try:
        analyzer = EssentiaAnalyzer(config.analysis)
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    s = analyze(conn, config, analyzer, limit=args.limit, retry_failed=args.retry_failed,
                force=args.force, progress=Progress())
    print(f"Analysis #{s.run_id} complete ({s.analyzer_version})")
    print(f"  processed  {s.pending}")
    print(f"  analyzed   {s.analyzed}")
    print(f"  skipped    {s.skipped}")
    print(f"  failed     {s.failed}")
    if s.failed:
        print("  Failed tracks keep their error; retry with `hades analyze --retry-failed`.")
    return 0


def cmd_reconcile(args: argparse.Namespace, config: Config, conn: sqlite3.Connection) -> int:
    from hades.pipeline import reconcile_all

    n = reconcile_all(conn, Progress())
    print(f"Reconciled {n} tracks")
    return 0


def cmd_stats(args: argparse.Namespace, config: Config, conn: sqlite3.Connection) -> int:
    from hades.reporting import print_stats

    print(f"Database: {config.database_path}")
    print_stats(conn, sys.stdout)
    return 0


def cmd_inspect(args: argparse.Namespace, config: Config, conn: sqlite3.Connection) -> int:
    from hades.reporting import find_track, print_track

    track = find_track(conn, args.track)
    if track is None:
        print(f"error: no indexed track matches {args.track!r}", file=sys.stderr)
        return 1
    print_track(conn, track, sys.stdout)
    return 0


def cmd_search(args: argparse.Namespace, config: Config, conn: sqlite3.Connection) -> int:
    from hades.reporting import print_search

    if not args.query.strip():
        print("error: search text must not be empty", file=sys.stderr)
        return 2
    print_search(conn, args.query, sys.stdout)
    return 0


def cmd_list(args: argparse.Namespace, config: Config, conn: sqlite3.Connection) -> int:
    from hades.reporting import print_track_properties

    track = conn.execute("SELECT * FROM tracks WHERE id = ?", (args.track_id,)).fetchone()
    if track is None:
        print(f"error: no track with ID {args.track_id}", file=sys.stderr)
        return 1
    print_track_properties(conn, track, sys.stdout)
    return 0


def cmd_conflicts(args: argparse.Namespace, config: Config, conn: sqlite3.Connection) -> int:
    from hades.reporting import print_conflicts

    print_conflicts(conn, sys.stdout, args.field)
    return 0


def cmd_export(args: argparse.Namespace, config: Config, conn: sqlite3.Connection) -> int:
    from hades.reporting import EXPORT_FIELDS, export_csv

    if args.list_fields:
        print("\n".join(EXPORT_FIELDS))
        return 0
    fields = [f.strip() for f in args.fields.split(",") if f.strip()] if args.fields else None
    try:
        if args.output:
            with open(args.output, "w", newline="", encoding="utf-8") as fh:
                n = export_csv(conn, fh, fields, args.include_missing)
            print(f"Exported {n} tracks to {args.output}", file=sys.stderr)
        else:
            export_csv(conn, sys.stdout, fields, args.include_missing)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_ui(args: argparse.Namespace, config: Config, conn: sqlite3.Connection) -> int:
    from hades.llm import make_client
    from hades.player import make_player
    from hades.tui import HadesApp

    app = HadesApp(conn, config.database_path, lambda: make_client(config.llm),
                   make_player(config.player.mpv_path))
    app.run()
    return 0


def cmd_generate(args: argparse.Namespace, config: Config, conn: sqlite3.Connection) -> int:
    from hades.llm import LLMError, make_client, refine_playlist
    from hades.playlist import Playlist
    from hades.repository import RepositoryError, save_playlist
    from hades.service import generate, report_lines

    try:
        client = make_client(config.llm)
        gen = generate(conn, args.request, client)
        if gen.error:
            print(f"error: could not create a playlist: {gen.error}", file=sys.stderr)
            return 1
        assert gen.result is not None and gen.pool is not None
        result = gen.result
        if args.refine:
            outcome = refine_playlist(result, gen.pool.tracks, client, args.request)
            print(f"Refinement {outcome.message}", file=sys.stderr)
            result = outcome.result
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    playlist = Playlist.from_result(result, args.request)
    _print_playlist(playlist)
    print()
    print("\n".join(report_lines(result)))
    if args.save:
        playlist.rename(args.save)
        try:
            save_playlist(conn, playlist)
        except RepositoryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Saved as {playlist.name!r}")
    return 0 if not result.unsatisfied else 3


def cmd_playlists(args: argparse.Namespace, config: Config, conn: sqlite3.Connection) -> int:
    from hades.repository import RepositoryError, list_playlists, load_playlist

    if args.name:
        try:
            playlist = load_playlist(conn, args.name)
        except RepositoryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(playlist.name)
        _print_playlist(playlist)
        for v in playlist.violations():
            print(f"  WARNING  {v}")
        return 0
    items = list_playlists(conn)
    if not items:
        print("No saved playlists.")
    for p in items:
        print(f"{p.name}\t{p.track_count} tracks\t{p.generation_status}\t{p.updated_at}")
    return 0


def _print_playlist(playlist: Playlist) -> None:
    from hades.builder import fmt_duration

    issues = [()] + [t.issues for t in playlist.transitions()]
    for pos, (t, tr) in enumerate(zip(playlist.tracks, issues), start=1):
        bpm = f"{t.bpm:.0f}" if t.bpm is not None else "-"
        key = t.camelot or "-"
        time = fmt_duration(t.duration) if t.duration else "-"
        note = f"  [{', '.join(tr)}]" if tr else ""
        print(f"{pos:>3}  {t.id:>6}  {t.display[:50]:<50}  {bpm:>4}  {key:>3}  {time:>6}{note}")
    s = playlist.stats
    print(f"{s.count} tracks | {fmt_duration(s.total_seconds)} total | {s.bpm_range}")
