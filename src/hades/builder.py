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

A maximum BPM step is a hard rule: the walk only continues with tracks within
the step (and not against the requested direction), so it may end early.

Ties are broken by track ID, so the same pool and spec always give the same
playlist.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from dataclasses import dataclass, field

from hades.keys import harmonic_compatibility
from hades.query import CandidatePool, Track, track_violations
from hades.spec import Ordering, PlaylistSpec

DEFAULT_DURATION_MINUTES = 60.0
NEUTRAL_KEY_SCORE = 0.35       # unknown/unreliable key: below any "compatible" score
GOOD_KEY_SCORE = 0.5           # at or above: a transition counts as harmonic
MAX_BPM_STEP_PCT = 6.0         # a bigger tempo change counts as a jump
ATTEMPTS = 8                   # starting tracks tried
LARGE_LIST = 80                # above this many tracks: no neighbour swapping
MAX_TRANSITION_WARNINGS = 10   # more are summarized by kind
SAME_ARTIST_PENALTY = 1.5      # back-to-back tracks sharing an artist
SAME_ALBUM_PENALTY = 1.0       # back-to-back tracks from the same album
REPEAT_ARTIST_PENALTY = 0.6    # per earlier track by the same artist anywhere in the set
AVOID_PENALTY = 0.3            # tracks of the previous attempt, when asked to try again
VARIATION = 0.2                # max random bonus per step when a seed is given


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

    if (step := step_violation(a, b, ordering)) is not None:
        issues.append(step)

    shared = a.artist_names & b.artist_names
    if shared:
        score -= SAME_ARTIST_PENALTY
        issues.append("same artist")
    elif a.album_key is not None and a.album_key == b.album_key:
        score -= SAME_ALBUM_PENALTY
        issues.append("same album")

    if ordering.harmonic_compatibility:
        k = key_score(a, b)
        score += k
        if not (a.key_reliable and b.key_reliable):
            issues.append("key unknown")
        elif k < GOOD_KEY_SCORE:
            issues.append(f"key clash {a.camelot}→{b.camelot}")
    return Transition(score, tuple(issues))


def step_violation(a: Track, b: Track, ordering: Ordering) -> str | None:
    """How the move a -> b breaks the maximum BPM step rule, or None if it doesn't."""
    limit = ordering.max_bpm_step
    if limit is None:
        return None
    if a.bpm is None or b.bpm is None:
        return "BPM unknown"
    diff = b.bpm - a.bpm
    if abs(diff) > limit + 1e-9:
        return f"BPM step {diff:+.1f} exceeds {limit:g}"
    direction = {"ascending": 1, "descending": -1}.get(ordering.bpm_progression, 0)
    if diff * direction < -1e-9:
        return f"BPM {'drop' if direction > 0 else 'rise'} {diff:+.1f}"
    return None


def transitions(tracks: list[Track], ordering: Ordering) -> list[Transition]:
    """transitions[i] describes the move into tracks[i + 1]."""
    return [transition(a, b, ordering) for a, b in zip(tracks, tracks[1:])]


def _identity(t: Track) -> tuple[str, str] | None:
    return t.identity


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
    if spec.max_tracks_per_artist is not None:
        counts = Counter(n for t in tracks for n in t.artist_names)
        for name, n in sorted(counts.items()):
            if n > spec.max_tracks_per_artist:
                out.append(f"{n} tracks by {name} (at most {spec.max_tracks_per_artist})")
    for pos, (a, b) in enumerate(zip(tracks, tracks[1:]), start=1):
        if (step := step_violation(a, b, spec.ordering)) is not None:
            out.append(f"#{pos}→#{pos + 1}: {step}")
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
    """Fill in a length when neither a count nor a duration was requested. Filters are never added.

    Not for "all matching tracks", which has no length by definition.
    """
    if spec.selection == "set" and spec.track_count is None and spec.duration_minutes is None:
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


def _stepped_starts(pool: list[Track], spec: PlaylistSpec, n: int,
                    rng: random.Random | None) -> list[tuple[Track, list[float]]]:
    """Starts spread over the tempo range for a progression under a maximum BPM step.

    Each start gets its own ramp that rises (or falls) by at most half the step per
    track, so the walk has slack and sparse tempo regions don't end it at once.
    """
    step = spec.ordering.max_bpm_step
    assert step is not None
    sign = 1 if spec.ordering.bpm_progression == "ascending" else -1
    known = sorted((t for t in pool if t.bpm is not None), key=lambda t: (sign * t.bpm, t.id))
    if not known:
        return []
    far = known[-1 - len(known) // 10].bpm  # the 90th percentile in the walking direction
    if spec.bpm is not None:
        limit = spec.bpm.max if sign > 0 else spec.bpm.min
        if limit is not None:
            far = min(far, limit) if sign > 0 else max(far, limit)
    k = ATTEMPTS * 2
    if rng is None:
        stride = max(len(known) // k, 1)
        starts = known[::stride][:k]
    else:
        starts = rng.sample(known, min(k, len(known)))
    out = []
    for t in starts:
        assert t.bpm is not None and far is not None
        span = max(sign * (far - t.bpm), 0.0)
        slope = min(step / 2, span / (n - 1)) if n > 1 else 0.0
        out.append((t, [t.bpm + sign * slope * i for i in range(max(n, 1))]))
    return out


class _Attempt:
    def __init__(self, start: Track, pool: list[Track], spec: PlaylistSpec, targets: _Targets,
                 plan: list[float] | None, rng: random.Random | None,
                 avoid: frozenset[int]) -> None:
        self.spec, self.targets, self.plan = spec, targets, plan
        self.pool, self.rng, self.avoid = pool, rng, avoid
        self.seq: list[Track] = []
        self.used: set[int] = set()
        self.identities: set[tuple[str, str]] = set()
        self.artists: Counter[str] = Counter()
        self.total = 0.0
        self.score = 0.0
        self._add(start, 0.0)

    def _add(self, t: Track, score: float) -> None:
        self.seq.append(t)
        self.used.add(t.id)
        if (ident := _identity(t)) is not None:
            self.identities.add(ident)
        self.artists.update(t.artist_names)
        self.total += t.duration or 0.0
        self.score += score

    def _usable(self, t: Track) -> bool:
        if t.id in self.used or _identity(t) in self.identities:
            return False
        limit = self.spec.max_tracks_per_artist
        return limit is None or all(self.artists[n] < limit for n in t.artist_names)

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

    def _step_ok(self, t: Track) -> bool:
        return step_violation(self.seq[-1], t, self.spec.ordering) is None

    def _next_to_last_ok(self, t: Track) -> bool:
        """No shared artist and no shared album with the previous track."""
        last = self.seq[-1]
        if t.artist_names & last.artist_names:
            return False
        return t.album_key is None or t.album_key != last.album_key

    def _candidate_score(self, t: Track) -> float:
        score = transition(self.seq[-1], t, self.spec.ordering).score
        if self.plan is not None and t.bpm is not None:
            idx = min(len(self.seq), len(self.plan) - 1)
            score -= abs(t.bpm - self.plan[idx]) / max(self.plan[idx], 1) * 100 / 4
        score -= REPEAT_ARTIST_PENALTY * max((self.artists[n] for n in t.artist_names), default=0)
        if t.id in self.avoid:
            score -= AVOID_PENALTY
        return score

    def grow(self) -> None:
        while not self.done():
            usable = [t for t in self.pool if self._usable(t)]
            if self.spec.ordering.max_bpm_step is not None:
                usable = [t for t in usable if self._step_ok(t)]
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
            # Same artist or album back to back only when nothing else fits.
            fitting = [t for t in fitting if self._next_to_last_ok(t)] or fitting
            scored = []
            for t in fitting:
                base = self._candidate_score(t)
                bonus = self.rng.uniform(0, VARIATION) if self.rng else 0.0
                scored.append((base + bonus, -t.id, base, t))
            _, _, base, best = max(scored, key=lambda x: (x[0], x[1]))
            self._add(best, base)

    def rank(self) -> tuple[float, ...]:
        violations = validate(self.seq, self.spec)
        mean = self.score / max(len(self.seq) - 1, 1)
        # Longer is better when the length is not a matter of duration alone.
        by_length = (self.targets.count is not None or self.spec.selection == "all"
                     or self.spec.ordering.max_bpm_step is not None)
        return (-len(violations), len(self.seq) if by_length else 0, mean)


def _neighbours_clash(a: Track, b: Track) -> bool:
    return bool(a.artist_names & b.artist_names) or (
        a.album_key is not None and a.album_key == b.album_key)


def separate_neighbours(tracks: list[Track], ordering: Ordering) -> list[Track]:
    """Reorder (by swapping pairs) so fewer neighbours share an artist or album.

    Greedy building can leave tracks of one artist for the end, where they have to
    sit together. A swap is made only if it reduces the number of such neighbours;
    among those, the one with the smoothest transitions wins.
    """
    seq = list(tracks)
    if len(seq) > LARGE_LIST:
        return seq  # quadratic in swaps times length: too slow for long lists

    def clashes(s: list[Track]) -> int:
        return sum(_neighbours_clash(a, b) for a, b in zip(s, s[1:]))

    def steps_ok(s: list[Track]) -> bool:
        return all(step_violation(a, b, ordering) is None for a, b in zip(s, s[1:]))

    def smoothness(s: list[Track]) -> float:
        return sum(t.score for t in transitions(s, ordering))

    current = clashes(seq)
    while current:
        best: tuple[int, float, list[Track]] | None = None
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                trial = list(seq)
                trial[i], trial[j] = trial[j], trial[i]
                n = clashes(trial)
                if n < current and (ordering.max_bpm_step is None or steps_ok(trial)):
                    key = (n, smoothness(trial))
                    if best is None or (-key[0], key[1]) > (-best[0], best[1]):
                        best = (key[0], key[1], trial)
        if best is None:
            break
        current, seq = best[0], best[2]
    return seq


def _starts(pool: list[Track], plan: list[float] | None,
            rng: random.Random | None) -> list[Track]:
    if plan is not None:
        first = plan[0]
        ranked = sorted(pool, key=lambda t: (abs((t.bpm if t.bpm is not None else 1e9) - first),
                                             t.id))
    else:
        by_bpm = sorted(pool, key=lambda t: (t.bpm if t.bpm is not None else 1e9, t.id))
        step = max(len(by_bpm) // (ATTEMPTS * 3), 1)
        ranked = by_bpm[::step]
    if rng is None:
        return ranked[:ATTEMPTS] if plan is not None else ranked[::3][:ATTEMPTS]
    choices = ranked[:ATTEMPTS * 3]
    return rng.sample(choices, min(ATTEMPTS, len(choices)))


def build_playlist(requested: PlaylistSpec, pool: CandidatePool, *, seed: int | None = None,
                   avoid: frozenset[int] = frozenset()) -> BuildResult:
    """Build a playlist. The same inputs always give the same result.

    ``seed`` adds bounded randomness so that a repeated request yields a different
    (still valid) playlist; ``avoid`` softly steers away from given track IDs.
    """
    rng = random.Random(seed) if seed is not None else None
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

    if spec.selection == "all" and spec.ordering.bpm_progression != "none":
        return _finish(result, _all_by_tempo(candidates, spec), candidates, requested, spec)

    durations = [t.duration for t in candidates if t.duration]
    if targets.count is not None:
        n_est = targets.count
    elif spec.selection == "all":
        n_est = len(candidates)
    else:
        n_est = max(1, round((targets.goal or 0) / statistics.median(durations)))
    if spec.ordering.max_bpm_step is not None and spec.ordering.bpm_progression != "none":
        starts = _stepped_starts(candidates, spec, n_est, rng)
    else:
        plan = _bpm_plan(candidates, spec, n_est)
        starts = [(t, plan) for t in _starts(candidates, plan, rng)]

    best: _Attempt | None = None
    for start, plan in starts:
        if targets.hi is not None and (start.duration or 0) > targets.hi:
            continue
        attempt = _Attempt(start, candidates, spec, targets, plan, rng, avoid)
        attempt.grow()
        if best is None or attempt.rank() > best.rank():
            best = attempt
    if best is None:
        result.unsatisfied.append("every matching track is longer than the duration limit")
        return result

    return _finish(result, separate_neighbours(best.seq, spec.ordering), candidates,
                   requested, spec)


def _finish(result: BuildResult, tracks: list[Track], candidates: list[Track],
            requested: PlaylistSpec, spec: PlaylistSpec) -> BuildResult:
    result.tracks = tracks
    # Only what the user asked for can be unmet; missing a default length is a warning.
    result.unsatisfied.extend(validate(result.tracks, requested))
    result.warnings.extend(f"default length not reached: {v}"
                           for v in validate(result.tracks, spec) if v not in result.unsatisfied)
    result.warnings.extend(quality_warnings(result.tracks, spec))
    if spec.selection == "all":
        result.warnings.extend(_left_out(candidates, result.tracks, spec))
    return result


def _all_by_tempo(pool: list[Track], spec: PlaylistSpec) -> list[Track]:
    """Every matching track in tempo order: the only order that leaves none behind.

    With a maximum step, the sorted list is cut at gaps wider than the step and the
    longest run is kept. Tracks of equal tempo are ordered for smooth transitions.
    """
    ordering = spec.ordering
    sign = 1 if ordering.bpm_progression == "ascending" else -1
    seq: list[Track] = []
    seen: set[tuple[str, str]] = set()
    artists: Counter[str] = Counter()
    limit = spec.max_tracks_per_artist
    for t in sorted(pool, key=lambda t: (t.bpm is None, sign * (t.bpm or 0.0), t.id)):
        if t.identity is not None and t.identity in seen:
            continue
        if limit is not None and any(artists[n] >= limit for n in t.artist_names):
            continue
        seq.append(t)
        if t.identity is not None:
            seen.add(t.identity)
        artists.update(t.artist_names)
    if ordering.max_bpm_step is not None and seq:
        runs, run = [], [seq[0]]
        for a, b in zip(seq, seq[1:]):
            if step_violation(a, b, ordering) is None:
                run.append(b)
            else:
                runs.append(run)
                run = [b]
        runs.append(run)
        seq = max(runs, key=len)
    out: list[Track] = []
    i = 0
    while i < len(seq):  # reorder each group of equal tempo
        j = i
        while j < len(seq) and seq[j].bpm == seq[i].bpm:
            j += 1
        group = seq[i:j]
        out += _smoothest_order(group, ordering) if len(group) > 2 else group
        i = j
    return out


def _left_out(candidates: list[Track], chosen: list[Track], spec: PlaylistSpec) -> list[str]:
    """Why matching tracks are missing from an "all matching tracks" list."""
    used = {t.id for t in chosen}
    identities = {t.identity for t in chosen if t.identity is not None}
    rest = [t for t in candidates if t.id not in used]
    dupes = sum(1 for t in rest if t.identity in identities)
    out = []
    if dupes:
        out.append(f"{dupes} matching track(s) left out: another file of the same song is listed")
    if other := len(rest) - dupes:
        rules = [r for r, on in (("the BPM step rule", spec.ordering.max_bpm_step is not None),
                                 ("the per-artist limit", spec.max_tracks_per_artist is not None))
                 if on]
        out.append(f"{other} matching track(s) left out by {' and '.join(rules) or 'the rules'}")
    return out


def build_variation(requested: PlaylistSpec, pool: CandidatePool, previous: list[int],
                    seed: int, tries: int = 12) -> BuildResult:
    """A different playlist for the same request, for "try again".

    Builds several seeded variants that softly avoid ``previous`` and returns the
    best one (fewest unmet constraints, then smoothest transitions) among those
    replacing at least half of the previous tracks; if none does, the most different.
    """
    avoid = frozenset(previous)
    rng = random.Random(seed)
    variants = [build_playlist(requested, pool, seed=rng.randrange(2**31), avoid=avoid)
                for _ in range(tries)]

    def overlap(r: BuildResult) -> int:
        return sum(1 for t in r.tracks if t.id in avoid)

    def quality(r: BuildResult) -> tuple[float, ...]:
        smooth = sum(t.score for t in transitions(r.tracks, r.effective_spec.ordering))
        return (-len(r.unsatisfied), smooth / max(len(r.tracks) - 1, 1))

    fresh = [r for r in variants if overlap(r) * 2 <= len(avoid)]
    if fresh:
        return max(fresh, key=quality)
    return min(variants, key=lambda r: (overlap(r), [-q for q in quality(r)]))


# --------------------------------------------------------------------------- fixed track lists

EXACT_ORDER_LIMIT = 12  # up to this many tracks, harmonic ordering tries every order


def order_fixed(tracks: list[Track], harmonic: bool) -> BuildResult:
    """Order exactly the given tracks: by ascending tempo, or for the smoothest transitions.

    Nothing is added or dropped; problems (e.g. unavailable files) are only warnings.
    """
    ordering = Ordering(harmonic_compatibility=True) if harmonic \
        else Ordering(bpm_progression="ascending")
    spec = PlaylistSpec(ordering=ordering)
    if harmonic:
        seq = _smoothest_order(tracks, ordering)
    else:  # unknown tempo last; equal tempos keep the given order
        seq = sorted(tracks, key=lambda t: (t.bpm is None, t.bpm or 0.0))
    result = BuildResult(spec=spec, effective_spec=spec, tracks=seq, candidate_count=len(tracks))
    result.warnings.extend(validate(seq, spec))
    result.warnings.extend(quality_warnings(seq, spec))
    return result


def _smoothest_order(tracks: list[Track], ordering: Ordering) -> list[Track]:
    n = len(tracks)
    if n <= 2:
        return list(tracks)
    score = [[transition(a, b, ordering).score if a is not b else 0.0 for b in tracks]
             for a in tracks]
    path = _best_path(score) if n <= EXACT_ORDER_LIMIT else _good_path(score)
    return [tracks[i] for i in path]


def _path_score(score: list[list[float]], path: list[int]) -> float:
    return sum(score[a][b] for a, b in zip(path, path[1:]))


def _best_path(score: list[list[float]]) -> list[int]:
    """The order with the highest total transition score (Held-Karp, any start and end)."""
    n = len(score)
    full = (1 << n) - 1
    best = [[float("-inf")] * n for _ in range(1 << n)]
    prev = [[-1] * n for _ in range(1 << n)]
    for i in range(n):
        best[1 << i][i] = 0.0
    for mask in range(1, full + 1):
        row = best[mask]
        for last in range(n):
            total = row[last]
            if total == float("-inf"):
                continue
            for nxt in range(n):
                if mask >> nxt & 1:
                    continue
                cand = total + score[last][nxt]
                nmask = mask | 1 << nxt
                if cand > best[nmask][nxt]:
                    best[nmask][nxt], prev[nmask][nxt] = cand, last
    last = max(range(n), key=lambda i: best[full][i])
    path, mask = [], full
    while last != -1:
        path.append(last)
        mask, last = mask & ~(1 << last), prev[mask][last]
    return path[::-1]


def _good_path(score: list[list[float]]) -> list[int]:
    """Greedy walk from every start, improved by reversing segments while that helps."""
    n = len(score)
    best: list[int] | None = None
    for start in range(n):
        path, left = [start], set(range(n)) - {start}
        while left:
            nxt = max(sorted(left), key=lambda j: score[path[-1]][j])
            path.append(nxt)
            left.remove(nxt)
        if best is None or _path_score(score, path) > _path_score(score, best):
            best = path
    assert best is not None
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                trial = best[:i] + best[i:j][::-1] + best[j:]
                if _path_score(score, trial) > _path_score(score, best) + 1e-9:
                    best, improved = trial, True
    return best


def quality_warnings(tracks: list[Track], spec: PlaylistSpec) -> list[str]:
    out: list[str] = []
    for i, tr in enumerate(transitions(tracks, spec.ordering), start=2):
        if tr.issues:
            out.append(f"#{i - 1}→#{i}: {', '.join(tr.issues)}")
    if len(out) > MAX_TRANSITION_WARNINGS:  # long lists: counts per kind; the notes have details
        kinds = Counter(" ".join(issue.split()[:2]) for tr in transitions(tracks, spec.ordering)
                        for issue in tr.issues)
        out = [f"{len(out)} transitions have issues: "
               + ", ".join(f"{kind} ×{n}" for kind, n in kinds.most_common())]
    if spec.ordering.harmonic_compatibility:
        n = sum(1 for t in tracks if not t.key_reliable)
        if n:
            out.append(f"{n} track(s) have a missing or unreliable key")
    flagged = sum(1 for t in tracks if t.bpm_flag in ("HALF_DOUBLE",) and t.bpm_source != "manual")
    if flagged:
        out.append(f"{flagged} track(s) have a half/double-time BPM discrepancy")
    return out
