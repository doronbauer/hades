"""Generate a playlist from a natural-language request, end to end.

LLM -> PlaylistSpec -> candidates (SQLite) -> deterministic build -> report.
No UI code here, so the CLI and the terminal app share it.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from hades.builder import BuildResult, build_playlist, build_variation, fmt_duration
from hades.export import DEFAULT_EXTENSION, export_filename, export_m3u, remove_export
from hades.llm import LLMClient, interpret_request
from hades.playlist import Playlist
from hades.query import CandidatePool, find_candidates
from hades.repository import save_playlist
from hades.spec import PlaylistSpec


class ExportError(Exception):
    """The playlist was saved in the database, but the playlist file could not be written."""


@dataclass
class Generation:
    request: str
    spec: PlaylistSpec | None = None
    error: str | None = None          # interpretation failed; nothing was built
    pool: CandidatePool | None = None
    result: BuildResult | None = None


def generate(conn: sqlite3.Connection, request: str, client: LLMClient,
             today: date | None = None) -> Generation:
    gen = Generation(request)
    interpretation = interpret_request(request, client, today or date.today())
    if interpretation.spec is None:
        gen.error = interpretation.error
        return gen
    gen.spec = interpretation.spec
    gen.pool = find_candidates(conn, gen.spec)
    gen.result = build_playlist(gen.spec, gen.pool)
    gen.result.warnings[:0] = [f"not understood, ignored: “{u}”"
                               for u in interpretation.unsupported] + list(interpretation.notes)
    return gen


def generate_again(conn: sqlite3.Connection, playlist: Playlist,
                   seed: int | None = None) -> BuildResult:
    """Rebuild a generated playlist's request (no LLM call), preferring other tracks."""
    if not playlist.generated:
        raise ValueError("this playlist was not generated from a request, so there is "
                         "nothing to run again")
    assert playlist.spec is not None
    pool = find_candidates(conn, playlist.spec)
    seed = random.randrange(2**31) if seed is None else seed
    return build_variation(playlist.spec, pool, [t.id for t in playlist.tracks], seed)


def save_and_export(conn: sqlite3.Connection, playlist: Playlist, export_dir: Path,
                    extension: str = DEFAULT_EXTENSION) -> Path:
    """Save to the database, then write ``<export_dir>/<name><extension>``; return the file path.

    After a rename, the file written under the previous name is removed.
    """
    previous = playlist.saved_name
    save_playlist(conn, playlist)
    try:
        path = export_m3u(playlist, export_dir, extension)
        if previous and export_filename(previous) != export_filename(playlist.name):
            remove_export(previous, export_dir, extension)
    except OSError as exc:
        raise ExportError(f"saved “{playlist.name}” in the database, but could not write the "
                          f"playlist file in {export_dir}: {exc.strerror or exc}") from None
    return path


def report_lines(result: BuildResult) -> list[str]:
    """Stage D summary: counts, totals, unmet constraints and warnings."""
    s = result.stats
    spec = result.spec
    lines = [
        f"Constraints: {'; '.join(spec.hard_constraints()) or 'none'}",
        f"Preferences: {'; '.join(spec.soft_preferences()) or 'none'}",
    ]
    lines += [f"Default: {d}" for d in result.defaults]
    lines.append(
        f"{s.count} tracks | {fmt_duration(s.total_seconds)} total | {s.bpm_range} | "
        f"{result.candidate_count} eligible candidates")
    if result.unsatisfied:
        lines.append("UNMET: " + "; ".join(result.unsatisfied))
    else:
        lines.append("All hard constraints satisfied.")
    lines += [f"Warning: {w}" for w in result.warnings]
    return lines
