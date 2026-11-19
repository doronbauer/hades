"""Stages B and D: select and order tracks, validate the result, summarize it.

The algorithm is a deterministic greedy walk, repeated from several starting
tracks; the best attempt wins. At each step it picks the next track with the
best transition score among those that keep the hard constraints (count,
duration) satisfiable. Soft preferences only influence the score:

* BPM progression: tempo follows a planned ramp from the slow to the fast end
  of the pool (or the reverse); drops and big jumps are penalized.
* Harmonic compatibility: Camelot-wheel neighbours score high. Missing or
  unreliable keys get a neutral score, never a compatible one.
* Variety: repeating an artist is penalized; the same artist+title is never
  used twice.

Ties are broken by track ID, so the same pool and spec always give the same
playlist.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from hades.keys import harmonic_compatibility
from hades.query import CandidatePool, Track, normalize_text, track_violations
from hades.spec import Ordering, PlaylistSpec

DEFAULT_DURATION_MINUTES = 60.0
NEUTRAL_KEY_SCORE = 0.35       # unknown/unreliable key: below any "compatible" score
GOOD_KEY_SCORE = 0.5           # at or above: a transition counts as harmonic
MAX_BPM_STEP_PCT = 6.0         # a bigger tempo change counts as a jump
ATTEMPTS = 8                   # starting tracks tried


@dataclass(frozen=True)
class Transition:
    score: float
    issues: tuple[str, ...]  # human-readable problems, empty if smooth


@dataclass(frozen=True)
class Stats:
    count: int
    total_seconds: float
    bpm_min: float | None
    bpm_max: float | None
    unknown_duration: int

    @property
    def bpm_range(self) -> str:
        if self.bpm_min is None or self.bpm_max is None:
            return "BPM n/a"
        return f"{self.bpm_min:.0f}–{self.bpm_max:.0f} BPM"


@dataclass
class BuildResult:
    spec: PlaylistSpec            # as requested
    effective_spec: PlaylistSpec  # as built: the request plus any defaults listed in `defaults`
    tracks: list[Track]
    candidate_count: int
    unsatisfied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    defaults: list[str] = field(default_factory=list)  # assumptions made for unspecified parameters

    @property
    def stats(self) -> Stats:
        return summarize(self.tracks)

    @property
    def status(self) -> str:
        return "partial" if self.unsatisfied else "complete"


# --------------------------------------------------------------------------- scoring


def summarize(tracks: list[Track]) -> Stats:
    bpms = [t.bpm for t in tracks if t.bpm is not None]
    return Stats(
        count=len(tracks),
        total_seconds=sum(t.duration or 0.0 for t in tracks),
        bpm_min=min(bpms) if bpms else None,
        bpm_max=max(bpms) if bpms else None,
        unknown_duration=sum(1 for t in tracks if not t.duration),
    )


def key_score(a: Track, b: Track) -> float:
    if not (a.key_reliable and b.key_reliable):
        return NEUTRAL_KEY_SCORE
    return harmonic_compatibility(a.camelot, b.camelot)  # type: ignore[arg-type]


def transition(a: Track, b: Track, ordering: Ordering) -> Transition:
    """Score and describe the move from track a to track b (higher is smoother)."""
    issues: list[str] = []
    score = 0.0
    direction = {"ascending": 1, "descending": -1}.get(ordering.bpm_progression, 0)

    if a.bpm is None or b.bpm is None:
        if direction:
            score -= 0.5
            issues.append("BPM unknown")
    else:
        pct = (b.bpm - a.bpm) / a.bpm * 100
        signed = pct * direction if direction else abs(pct)
        if direction and signed < -0.5:
            score -= 1.0 + abs(signed) / 2
            issues.append(f"BPM {'drop' if direction > 0 else 'rise'} {b.bpm - a.bpm:+.0f}")
        if abs(pct) > MAX_BPM_STEP_PCT:
            score -= (abs(pct) - MAX_BPM_STEP_PCT) / 3
            issues.append(f"BPM jump {b.bpm - a.bpm:+.0f}")
        elif not direction:
            score -= abs(pct) / 20  # mild preference for steady tempo

    if ordering.harmonic_compatibility:
        k = key_score(a, b)
        score += k
        if not (a.key_reliable and b.key_reliable):
            issues.append("key unknown")
        elif k < GOOD_KEY_SCORE:
            issues.append(f"key clash {a.camelot}→{b.camelot}")
    return Transition(score, tuple(issues))


def transitions(tracks: list[Track], ordering: Ordering) -> list[Transition]:
    """transitions[i] describes the move into tracks[i + 1]."""
    return [transition(a, b, ordering) for a, b in zip(tracks, tracks[1:])]


def _identity(t: Track) -> tuple[str, str] | None:
    if t.artist and t.title:
        return normalize_text(t.artist), normalize_text(t.title)
    return None


# --------------------------------------------------------------------------- validation


def validate(tracks: list[Track], spec: PlaylistSpec) -> list[str]:
    """Every hard constraint the given playlist violates (empty if none)."""
    out: list[str] = []
    seen: set[int] = set()
    for pos, t in enumerate(tracks, start=1):
        if t.id in seen:
            out.append(f"track {t.id} appears more than once")
        seen.add(t.id)
        for v in track_violations(t, spec):
            out.append(f"#{pos} (ID {t.id}): {v}")
    if spec.track_count is not None and len(tracks) != spec.track_count:
        out.append(f"{len(tracks)} tracks instead of the requested {spec.track_count}")
    lo, hi = spec.duration_bounds_seconds
    total = summarize(tracks).total_seconds
    if hi is not None and total > hi + 1e-6:
        limit = "maximum" if spec.duration_mode == "max" else "upper limit"
        out.append(f"total {fmt_duration(total)} exceeds the {limit} of {fmt_duration(hi)}")
    if lo is not None and total < lo - 1e-6:
        out.append(f"total {fmt_duration(total)} is below the lower limit of {fmt_duration(lo)}")
    return out


def fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# --------------------------------------------------------------------------- building


@dataclass(frozen=True)
class _Targets:
    count: int | None
    lo: float | None       # seconds
    hi: float | None
    goal: float | None     # keep adding tracks while the total is below this


def apply_defaults(spec: PlaylistSpec) -> tuple[PlaylistSpec, list[str]]:
    """Fill in a length when neither a count nor a duration was requested. Filters are never added."""
    if spec.track_count is None and spec.duration_minutes is None:
        effective = spec.model_copy(update={"duration_minutes": DEFAULT_DURATION_MINUTES,
                                            "duration_mode": "target"})
        return effective, [f"no length requested: aiming for about "
                           f"{DEFAULT_DURATION_MINUTES:g} ± {spec.duration_tolerance_minutes:g} min"]
    return spec, []


def _targets(spec: PlaylistSpec) -> _Targets:
    lo, hi = spec.duration_bounds_seconds
    goal = spec.duration_minutes * 60 if spec.duration_minutes is not None else None
    return _Targets(spec.track_count, lo, hi, goal)


def _bpm_plan(pool: list[Track], spec: PlaylistSpec, n: int) -> list[float] | None:
    """Target tempo per position for a requested progression, or None."""
    direction = spec.ordering.bpm_progression
    bpms = sorted(t.bpm for t in pool if t.bpm is not None)
    if direction == "none" or not bpms:
        return None
    lo = bpms[len(bpms) // 10]
    hi = bpms[-1 - len(bpms) // 10]
    if spec.bpm is not None:
        lo = max(lo, spec.bpm.min) if spec.bpm.min is not None else lo
        hi = min(hi, spec.bpm.max) if spec.bpm.max is not None else hi
    if n <= 1:
        plan = [lo]
    else:
        plan = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    return plan if direction == "ascending" else plan[::-1]


class _Attempt:
    def __init__(self, start: Track, pool: list[Track], spec: PlaylistSpec, targets: _Targets,
                 plan: list[float] | None) -> None:
        self.spec, self.targets, self.plan = spec, targets, plan
        self.pool = pool
        self.seq: list[Track] = []
        self.used: set[int] = set()
        self.identities: set[tuple[str, str]] = set()
        self.artists: dict[str, int] = {}
        self.total = 0.0
        self.score = 0.0
        self._add(start, 0.0)

    def _add(self, t: Track, score: float) -> None:
        self.seq.append(t)
        self.used.add(t.id)
        if (ident := _identity(t)) is not None:
            self.identities.add(ident)
        if t.artist:
            a = normalize_text(t.artist)
            self.artists[a] = self.artists.get(a, 0) + 1
        self.total += t.duration or 0.0
        self.score += score

    def _usable(self, t: Track) -> bool:
        return t.id not in self.used and _identity(t) not in self.identities

    def done(self) -> bool:
        if self.targets.count is not None:
            return len(self.seq) >= self.targets.count
        return self.targets.goal is not None and self.total >= self.targets.goal

    def _fits(self, t: Track, remaining: list[float]) -> bool:
        lo, hi = self.targets.lo, self.targets.hi
        total = self.total + (t.duration or 0.0)
        if hi is not None and total > hi:
            return False
        if self.targets.count is not None:
            k = self.targets.count - len(self.seq) - 1  # slots left after this one
            if k > 0 and hi is not None and total + sum(remaining[:k]) > hi:
                return False
            if lo is not None and total + (sum(remaining[-k:]) if k > 0 else 0) < lo:
                return False
        return True

    def _candidate_score(self, t: Track) -> float:
        last = self.seq[-1]
        score = transition(last, t, self.spec.ordering).score
        if self.plan is not None and t.bpm is not None:
            idx = min(len(self.seq), len(self.plan) - 1)
            score -= abs(t.bpm - self.plan[idx]) / max(self.plan[idx], 1) * 100 / 4
        if t.artist:
            a = normalize_text(t.artist)
            if last.artist and normalize_text(last.artist) == a:
                score -= 1.5
            score -= 0.4 * self.artists.get(a, 0)
        return score

    def grow(self) -> None:
        while not self.done():
            usable = [t for t in self.pool if self._usable(t)]
            if not usable:
                return
            remaining = sorted(t.duration or 0.0 for t in usable)
            fitting = [t for t in usable if self._fits(t, remaining)]
            if not fitting:
                # Count + duration can't both be met: keep the duration limit, give up on count.
                hi = self.targets.hi
                fitting = [t for t in usable
                           if hi is None or self.total + (t.duration or 0.0) <= hi]
                if not fitting:
                    return
            best = max(fitting, key=lambda t: (self._candidate_score(t), -t.id))
            self._add(best, self._candidate_score(best))

    def rank(self) -> tuple[float, ...]:
        violations = validate(self.seq, self.spec)
        mean = self.score / max(len(self.seq) - 1, 1)
        return (-len(violations), len(self.seq) if self.targets.count else 0, mean)


def _starts(pool: list[Track], plan: list[float] | None) -> list[Track]:
    if plan is not None:
        first = plan[0]
        ranked = sorted(pool, key=lambda t: (abs((t.bpm if t.bpm is not None else 1e9) - first),
                                             t.id))
        return ranked[:ATTEMPTS]
    by_bpm = sorted(pool, key=lambda t: (t.bpm if t.bpm is not None else 1e9, t.id))
    step = max(len(by_bpm) // ATTEMPTS, 1)
    return by_bpm[::step][:ATTEMPTS]


def build_playlist(requested: PlaylistSpec, pool: CandidatePool) -> BuildResult:
    spec, defaults = apply_defaults(requested)
    targets = _targets(spec)
    candidates = list(pool.tracks)
    result = BuildResult(spec=requested, effective_spec=spec, tracks=[],
                         candidate_count=len(candidates), warnings=pool.warnings(),
                         defaults=defaults)
    # Never trust the pool blindly: drop anything that fails a hard filter.
    candidates = [t for t in candidates if not track_violations(t, spec)]
    if not candidates:
        result.unsatisfied.append("no tracks in the library match the requested filters")
        return result

    durations = [t.duration for t in candidates if t.duration]
    if targets.count is not None:
        n_est = targets.count
    else:
        n_est = max(1, round((targets.goal or 0) / statistics.median(durations)))
    plan = _bpm_plan(candidates, spec, n_est)

    best: _Attempt | None = None
    for start in _starts(candidates, plan):
        if targets.hi is not None and (start.duration or 0) > targets.hi:
            continue
        attempt = _Attempt(start, candidates, spec, targets, plan)
        attempt.grow()
        if best is None or attempt.rank() > best.rank():
            best = attempt
    if best is None:
        result.unsatisfied.append("every matching track is longer than the duration limit")
        return result

    result.tracks = best.seq
    # Only what the user asked for can be unmet; missing a default length is a warning.
    result.unsatisfied.extend(validate(best.seq, requested))
    result.warnings.extend(f"default length not reached: {v}"
                           for v in validate(best.seq, spec) if v not in result.unsatisfied)
    result.warnings.extend(quality_warnings(best.seq, spec))
    return result


def quality_warnings(tracks: list[Track], spec: PlaylistSpec) -> list[str]:
    out: list[str] = []
    for i, tr in enumerate(transitions(tracks, spec.ordering), start=2):
        if tr.issues:
            out.append(f"#{i - 1}→#{i}: {', '.join(tr.issues)}")
    if spec.ordering.harmonic_compatibility:
        n = sum(1 for t in tracks if not t.key_reliable)
        if n:
            out.append(f"{n} track(s) have a missing or unreliable key")
    flagged = sum(1 for t in tracks if t.bpm_flag in ("HALF_DOUBLE",) and t.bpm_source != "manual")
    if flagged:
        out.append(f"{flagged} track(s) have a half/double-time BPM discrepancy")
    return out
