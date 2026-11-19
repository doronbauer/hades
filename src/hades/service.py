"""Generate a playlist from a natural-language request, end to end.

LLM -> PlaylistSpec -> candidates (SQLite) -> deterministic build -> report.
No UI code here, so the CLI and the terminal app share it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date

from hades.builder import BuildResult, build_playlist, fmt_duration
from hades.llm import LLMClient, interpret_request
from hades.query import CandidatePool, find_candidates
from hades.spec import PlaylistSpec


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
    return gen


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
