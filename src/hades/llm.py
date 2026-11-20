"""LLM service: natural language -> validated ``PlaylistSpec``, and optional curation.

The LLM never touches SQL or the database. It returns JSON constrained by a
schema; ``hades.spec`` validates it again, and any refinement it proposes is
re-validated by the deterministic engine before use.

Only track IDs, artists, titles, BPM, keys, genres and durations are ever sent
(no file paths, no audio).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from hades.builder import (
    BuildResult,
    key_score,
    quality_warnings,
    summarize,
    transitions,
    validate,
)
from hades.config import LLMSettings
from hades.query import Track
from hades.spec import (
    FILTER_FIELDS,
    FILTER_OPS,
    PlaylistSpec,
    SpecError,
    describe_group,
    parse_spec,
)

SHORTLIST_SIZE = 60


class LLMError(Exception):
    """The LLM could not be reached or returned something unusable."""


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON object conforming to ``schema``."""
        ...


# --------------------------------------------------------------------------- provider


class AnthropicClient:
    def __init__(self, settings: LLMSettings) -> None:
        import anthropic

        self._anthropic = anthropic
        self.settings = settings
        # None: the SDK's own credential lookup applies.
        self._client = anthropic.Anthropic(api_key=settings.resolved_api_key,
                                           timeout=settings.timeout_seconds)

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        anthropic = self._anthropic
        try:
            response = self._client.beta.messages.create(
                model=self.settings.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": self.settings.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError:
            raise LLMError(
                f"LLM authentication failed: set ${self.settings.api_key_env} "
                "or log in with `ant auth login`") from None
        except anthropic.NotFoundError:
            raise LLMError(f"LLM model not found: {self.settings.model}") from None
        except anthropic.RateLimitError:
            raise LLMError("LLM rate limit reached; try again shortly") from None
        except anthropic.APIStatusError as exc:
            raise LLMError(f"LLM request failed ({exc.status_code}): {exc.message}") from None
        except anthropic.APIConnectionError:
            raise LLMError("could not reach the LLM service (network error or timeout)") from None
        except anthropic.AnthropicError as exc:
            raise LLMError(f"LLM unavailable: {exc}") from None

        if response.stop_reason == "refusal":
            raise LLMError("the LLM declined this request")
        if response.stop_reason == "max_tokens":
            raise LLMError("the LLM response was cut off")
        return _json_object(next((b.text for b in response.content if b.type == "text"), None))


class OpenAIClient:
    def __init__(self, settings: LLMSettings) -> None:
        import openai

        self._openai = openai
        self.settings = settings
        api_key = settings.resolved_api_key
        if api_key is None:
            raise LLMError(f"no OpenAI API key: set llm.api_key in config.toml "
                           f"or ${settings.api_key_env}")
        self._client = openai.OpenAI(api_key=api_key, timeout=settings.timeout_seconds)

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        openai = self._openai
        try:
            response = self._client.responses.create(
                model=self.settings.model,
                instructions=system,
                input=user,
                max_output_tokens=16000,
                reasoning={"effort": self.settings.effort},  # type: ignore[typeddict-item]
                text={"format": {"type": "json_schema", "name": "hades_response",
                                 "schema": schema, "strict": True}},
                store=False,
            )
        except openai.AuthenticationError:
            raise LLMError("LLM authentication failed: check llm.api_key in config.toml "
                           f"or ${self.settings.api_key_env}") from None
        except openai.NotFoundError:
            raise LLMError(f"LLM model not found: {self.settings.model}") from None
        except openai.RateLimitError:
            raise LLMError("LLM rate limit or quota reached; try again shortly") from None
        except openai.APIStatusError as exc:
            raise LLMError(f"LLM request failed ({exc.status_code}): {exc.message}") from None
        except openai.APIConnectionError:
            raise LLMError("could not reach the LLM service (network error or timeout)") from None
        except openai.OpenAIError as exc:
            raise LLMError(f"LLM unavailable: {exc}") from None

        if response.error is not None:
            raise LLMError(f"LLM request failed: {response.error.message}")
        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else None
            raise LLMError("the LLM declined this request" if reason == "content_filter"
                           else f"the LLM response was cut off ({reason or 'incomplete'})")
        for item in response.output:
            for part in getattr(item, "content", None) or []:
                if part.type == "refusal":
                    raise LLMError(f"the LLM declined this request: {part.refusal}")
        return _json_object(response.output_text or None)


def _json_object(text: str | None) -> dict[str, Any]:
    if text is None:
        raise LLMError("the LLM returned no answer")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise LLMError("the LLM returned malformed JSON") from None
    if not isinstance(data, dict):
        raise LLMError("the LLM returned JSON that is not an object")
    return data


def make_client(settings: LLMSettings) -> LLMClient:
    if settings.provider == "anthropic":
        return AnthropicClient(settings)
    if settings.provider == "openai":
        return OpenAIClient(settings)
    raise LLMError(f"unsupported LLM provider: {settings.provider}")


# --------------------------------------------------------------------------- interpretation


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


_STRINGS = {"type": "array", "items": {"type": "string"}}

# Flat on purpose (structured outputs don't allow recursive schemas): a list of
# groups that must all hold, each a list of conditions of which one must hold.
_CONDITION = {
    "type": "object",
    "additionalProperties": False,
    "required": ["field", "op", "values", "negate"],
    "properties": {
        "field": {"type": "string", "enum": list(FILTER_FIELDS)},
        "op": {"type": "string", "enum": list(FILTER_OPS)},
        "values": {"type": "array",
                   "items": {"anyOf": [{"type": "string"}, {"type": "number"},
                                       {"type": "boolean"}]}},
        "negate": {"type": "boolean"},
    },
}

INTERPRET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["error", "unsupported", "spec"],
    "properties": {
        "error": _nullable({"type": "string"}),
        "unsupported": _STRINGS,
        "spec": _nullable({
            "type": "object",
            "additionalProperties": False,
            "required": ["genre", "created_after", "created_before", "bpm", "duration_minutes",
                         "duration_mode", "duration_tolerance_minutes", "track_count", "folder",
                         "artists", "exclude_artists", "max_tracks_per_artist", "selection",
                         "filters", "ordering"],
            "properties": {
                "genre": _STRINGS,
                "created_after": _nullable({"type": "string", "format": "date"}),
                "created_before": _nullable({"type": "string", "format": "date"}),
                "bpm": _nullable({
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["min", "max"],
                    "properties": {"min": _nullable({"type": "number"}),
                                   "max": _nullable({"type": "number"})},
                }),
                "duration_minutes": _nullable({"type": "number"}),
                "duration_mode": _nullable({"type": "string", "enum": ["target", "max"]}),
                "duration_tolerance_minutes": _nullable({"type": "number"}),
                "track_count": _nullable({"type": "integer"}),
                "folder": _nullable({"type": "string"}),
                "artists": _STRINGS,
                "exclude_artists": _STRINGS,
                "max_tracks_per_artist": _nullable({"type": "integer"}),
                "selection": {"type": "string", "enum": ["set", "all"]},
                "filters": {"type": "array", "items": {"type": "array", "items": _CONDITION}},
                "ordering": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["bpm_progression", "harmonic_compatibility", "max_bpm_step"],
                    "properties": {
                        "bpm_progression": {"type": "string",
                                            "enum": ["ascending", "descending", "none"]},
                        "harmonic_compatibility": {"type": "boolean"},
                        "max_bpm_step": _nullable({"type": "number"}),
                    },
                },
            },
        }),
    },
}

INTERPRET_SYSTEM = """\
You translate a DJ's request about their local music library into a structured query. \
Capture everything the request states and nothing it doesn't; never invent a filter.

Selection:
- selection: "all" only when the user explicitly wants every matching track ("all the \
tracks from 2026", "every track under 90 BPM", "list/show me the tracks ...") and gives no \
length. Otherwise "set": a playlist of some length, including whenever the request is \
about how tracks follow each other (tempo progression, steps, transitions) without \
saying "all". With "set" and no length, the app picks a default length.

Simple fields (use null or [] for anything not mentioned):
- genre: genre names exactly as the user said them (e.g. "drum & bass", "techno"). \
Several genres mean "any of these".
- created_after / created_before: EXCLUSIVE calendar-date bounds (YYYY-MM-DD) on when \
the file was added to the library. Words like "added", "new", "got", "bought", or a bare \
period such as "tracks from 2026", refer to this. Interpret calendar periods as whole \
periods: "after March 2026" -> created_after "2026-03-31"; "before March 2026" -> \
created_before "2026-03-01"; "in March 2026" / "from March 2026" -> created_after \
"2026-02-28" and created_before "2026-04-01"; "since 1 May 2026" (inclusive) -> \
created_after "2026-04-30". Resolve relative dates ("this year", "last month") against \
today's date, given below.
- bpm: ONE contiguous range {min, max}; "below 160" -> max 160, "above 125" -> min 125, \
"125 to 135" -> both. For ranges that don't overlap, use filters instead.
- duration_minutes + duration_mode: "a one-hour playlist", "about 90 minutes", "a 90-minute \
set" -> mode "target"; "no longer than an hour", "at most 45 minutes", "under an hour" -> \
mode "max". duration_tolerance_minutes: only if the user states one, else null.
- track_count: only if a number of tracks is requested.
- folder: a directory name or path, only if the user restricts to one.
- artists / exclude_artists: only artists the user names.
- max_tracks_per_artist: only if the user limits repeats: "one track per artist", "no \
artist twice" -> 1; "at most 2 per artist" -> 2. Otherwise null.

filters: for anything the simple fields can't say: alternatives ("or"), negations, and \
these fields. Every group must hold; a group holds if ANY of its conditions holds.
- Fields: bpm, year (release year tag), duration_minutes (track length), loudness_lufs \
(integrated loudness, e.g. -8 is loud, -14 is quiet), added (date the file was added; \
inclusive YYYY-MM-DD values), genre, artist, title, album (whole-word matches), format \
(MP3, FLAC, AIFF, WAV, M4A), folder, key (Camelot like "8A" or names like "A minor"), \
key_reliable / bpm_reliable (true/false: tag and analysis agree).
- ops: lt, le, gt, ge (one value), between (two values, inclusive), eq (one value), in \
(several values). Text and key fields use eq or in and match any listed value. \
negate: true inverts a condition.
- "under 90 or above 150 BPM" -> one group: [bpm lt 90, bpm gt 150]. People also say \
"under 90 and above 150": conditions that no track can meet at once are alternatives, \
so this is the same single group. "tracks in 8A or \
9A, not techno" -> two groups: [key in 8A, 9A] and [genre eq techno, negate]. "added in \
2024 or 2026" -> one group: [added between 2024-01-01 and 2024-12-31, added between \
2026-01-01 and 2026-12-31]. "only tracks with a certain key" -> [key_reliable eq true].

ordering:
- bpm_progression: "ascending" for increasing tempo, "starting slow and getting faster", \
"build up", and for a tempo progression with no direction given ("progress in tempo"); \
"descending" for slowing down or winding down; otherwise "none".
- harmonic_compatibility: true if the user asks for harmonic/key-compatible/smooth \
harmonic transitions or mixing in key.
- max_bpm_step: the largest BPM change allowed between neighbouring tracks, if the user \
limits it ("no more than 2 BPM from track to track" -> 2). Otherwise null.

unsupported: short phrases quoting any part of the request you could not express with \
these fields (e.g. a mood, "vocals only"). Still return a spec for the rest. Never \
resolve a contradiction by silently changing what the user asked for: list the part you \
could not honour here.

Set "error" (and spec null) only if the text is not a request about tracks or playlists \
at all, or if nothing in it can be expressed. Explain the problem in one or two plain \
sentences."""


@dataclass(frozen=True)
class Interpretation:
    spec: PlaylistSpec | None
    error: str | None
    raw: dict[str, Any]
    unsupported: tuple[str, ...] = ()  # parts of the request the spec does not capture
    notes: tuple[str, ...] = ()        # how the request was read where that was not literal


def interpret_request(text: str, client: LLMClient, today: date) -> Interpretation:
    """Ask the LLM for a spec, then validate it deterministically."""
    user = f"Today's date: {today.isoformat()}\n\nRequest: {text.strip()}"
    raw = client.complete_json(INTERPRET_SYSTEM, user, INTERPRET_SCHEMA)
    if raw.get("error"):
        return Interpretation(None, str(raw["error"]), raw)
    data = raw.get("spec")
    if not isinstance(data, dict):
        return Interpretation(None, "the request could not be interpreted as a playlist", raw)
    data = {k: v for k, v in data.items() if v is not None}
    for nested in ("bpm", "ordering"):
        if isinstance(data.get(nested), dict):
            data[nested] = {k: v for k, v in data[nested].items() if v is not None}
    unsupported = tuple(str(u).strip() for u in raw.get("unsupported") or [] if str(u).strip())
    try:
        spec = parse_spec(data)
    except SpecError:
        spec = None
    notes: tuple[str, ...] = ()
    if spec is None or not _satisfiable(spec):
        # "Under 90 and above 150 BPM": ranges no track can meet together mean either/or.
        merged = _ranges_as_alternatives(data)
        if merged is not None:
            try:
                spec = parse_spec(merged)
                notes = tuple(f"ranges that can't hold together read as either/or: "
                              f"{describe_group(g)}" for g in spec.filters if len(g) > 1)
            except SpecError:
                pass
    try:
        spec = spec or parse_spec(data)
    except SpecError as exc:
        return Interpretation(None, f"the request is invalid or contradictory: {exc}", raw,
                              unsupported)
    return Interpretation(spec, None, raw, unsupported, notes)


_RANGE_OPS = {"lt", "le", "gt", "ge", "between", "eq"}


def _interval(c: dict[str, Any]) -> tuple[Any, Any] | None:
    """(low, high) of a numeric/date condition; None for other conditions."""
    if c.get("negate") or c.get("op") not in _RANGE_OPS or not c.get("values"):
        return None
    v = c["values"]
    try:
        if c["field"] == "added":
            v = [date.fromisoformat(str(x)) for x in v]
        elif c["field"] in ("bpm", "year", "duration_minutes", "loudness_lufs"):
            v = [float(x) for x in v]
        else:
            return None
    except (TypeError, ValueError):
        return None
    lo, hi = (None, None)
    if c["op"] in ("lt", "le"):
        hi = v[0]
    elif c["op"] in ("gt", "ge"):
        lo = v[0]
    elif c["op"] == "eq" or len(v) == 2:
        lo, hi = min(v[0], v[-1]), max(v[0], v[-1])
    return lo, hi


def _disjoint(intervals: list[tuple[Any, Any]]) -> bool:
    lows = [lo for lo, _ in intervals if lo is not None]
    highs = [hi for _, hi in intervals if hi is not None]
    return bool(lows and highs) and max(lows) >= min(highs)


def _ranges_as_alternatives(data: dict[str, Any]) -> dict[str, Any] | None:
    """Merge single-condition groups on one field that can't hold together into one group.

    A simple BPM range with its minimum above its maximum counts too. Returns the
    changed data, or None if there is nothing to merge.
    """
    groups = [list(g) for g in data.get("filters") or []]
    bpm = data.get("bpm") or {}
    lo, hi = bpm.get("min"), bpm.get("max")
    out = dict(data)
    if lo is not None and hi is not None and lo > hi:
        out.pop("bpm")
        groups.append([{"field": "bpm", "op": "le", "values": [hi], "negate": False},
                       {"field": "bpm", "op": "ge", "values": [lo], "negate": False}])
    elif bpm and any(len(g) == 1 and g[0].get("field") == "bpm" for g in groups):
        out.pop("bpm")  # take part in the comparison below
        for op, v in (("ge", lo), ("le", hi)):
            if v is not None:
                groups.append([{"field": "bpm", "op": op, "values": [v], "negate": False}])
    by_field: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        if len(g) == 1 and _interval(g[0]) is not None:
            by_field.setdefault(g[0]["field"], []).append(i)
    merged_any = len(groups) != len(data.get("filters") or [])
    drop: set[int] = set()
    for idx in by_field.values():
        if len(idx) > 1 and _disjoint([_interval(groups[i][0]) for i in idx]):  # type: ignore[misc]
            groups[idx[0]] = [groups[i][0] for i in idx]
            drop |= set(idx[1:])
            merged_any = True
    if not merged_any:
        return None
    out["filters"] = [g for i, g in enumerate(groups) if i not in drop]
    return out


def _satisfiable(spec: PlaylistSpec) -> bool:
    """False if single-condition range groups on one field can't all hold."""
    by_field: dict[str, list[tuple[Any, Any]]] = {}
    for g in spec.filters:
        if len(g) == 1 and (iv := _interval(g[0].model_dump(mode="json"))) is not None:
            by_field.setdefault(g[0].field, []).append(iv)
    if spec.bpm is not None:
        by_field.setdefault("bpm", []).append((spec.bpm.min, spec.bpm.max))
    return not any(len(ivs) > 1 and _disjoint(ivs) for ivs in by_field.values())


# --------------------------------------------------------------------------- refinement

REFINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["track_ids", "rationale"],
    "properties": {
        "track_ids": {"type": "array", "items": {"type": "integer"}},
        "rationale": {"type": "string"},
    },
}

REFINE_SYSTEM = """\
You are helping a DJ refine a playlist. You get the request, its hard constraints, the \
current playlist and a shortlist of eligible tracks. Propose an improved selection and \
order, using only track IDs from the shortlist, each at most once. Keep every hard \
constraint (track count, total duration). Improve flow: follow the requested tempo \
progression with small steps, prefer harmonically compatible neighbouring keys (Camelot: \
same number, +/-1, or the A/B relative), and vary artists. If the current playlist is \
already good, return it unchanged. Give a one-sentence rationale."""


@dataclass(frozen=True)
class Refinement:
    accepted: bool
    result: BuildResult
    message: str


def _track_json(t: Track) -> dict[str, Any]:
    return {"id": t.id, "artist": t.artist, "title": t.title,
            "bpm": round(t.bpm, 1) if t.bpm is not None else None,
            "key": t.camelot if t.key_reliable else None, "genre": t.genre,
            "duration_s": round(t.duration) if t.duration else None}


def shortlist(result: BuildResult, pool: list[Track], size: int = SHORTLIST_SIZE) -> list[Track]:
    """The current playlist plus the closest other candidates, at most ``size`` tracks."""
    chosen = {t.id for t in result.tracks}
    stats = summarize(result.tracks)
    lo, hi = stats.bpm_min, stats.bpm_max

    def closeness(t: Track) -> tuple[float, float, int]:
        if t.bpm is None or lo is None or hi is None:
            bpm_gap = 1e6
        else:
            bpm_gap = max(lo - t.bpm, 0.0, t.bpm - hi)
        harmony = max((key_score(t, s) for s in result.tracks), default=0.0)
        return bpm_gap, -harmony, t.id

    extra = sorted((t for t in pool if t.id not in chosen), key=closeness)
    return list(result.tracks) + extra[:max(size - len(result.tracks), 0)]


def refine_playlist(result: BuildResult, pool: list[Track], client: LLMClient,
                    request: str = "") -> Refinement:
    """Ask the LLM for a better selection/order.

    The proposal is accepted only if it passes every hard constraint and its
    transitions score at least as well as the current order.
    """
    if not result.tracks:
        return Refinement(False, result, "nothing to refine")
    if result.spec.selection == "all":
        return Refinement(False, result, "not available for a list of all matching tracks")
    spec = result.spec
    candidates = shortlist(result, pool)
    by_id = {t.id: t for t in candidates}
    user = json.dumps({
        "request": request,
        "hard_constraints": spec.hard_constraints() or ["none"],
        "preferences": spec.soft_preferences() or ["none"],
        "required_track_count": spec.track_count if spec.track_count is not None
        else len(result.tracks),
        "duration_bounds_seconds": result.effective_spec.duration_bounds_seconds,
        "current_playlist": [t.id for t in result.tracks],
        "shortlist": [_track_json(t) for t in candidates],
    }, ensure_ascii=False)
    raw = client.complete_json(REFINE_SYSTEM, user, REFINE_SCHEMA)

    ids = raw.get("track_ids")
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        return Refinement(False, result, "rejected: the LLM returned no valid track list")
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        return Refinement(False, result,
                          f"rejected: IDs not in the shortlist: {', '.join(map(str, unknown))}")
    if len(set(ids)) != len(ids):
        return Refinement(False, result, "rejected: the proposal repeats tracks")
    proposed = [by_id[i] for i in ids]
    violations = validate(proposed, spec)
    if result.defaults and len(proposed) < len(result.tracks):
        violations.append("shortens a playlist whose length was not specified")
    if violations:
        return Refinement(False, result, "rejected: " + "; ".join(violations[:3]))

    old = sum(t.score for t in transitions(result.tracks, spec.ordering))
    new = sum(t.score for t in transitions(proposed, spec.ordering))
    if new < old - 1e-9:
        return Refinement(False, result, f"rejected: transitions score worse ({old:.1f} → "
                                         f"{new:.1f}) than the current order")
    refined = BuildResult(spec=spec, effective_spec=result.effective_spec, tracks=proposed,
                          candidate_count=result.candidate_count,
                          warnings=[w for w in result.warnings if not w.startswith("#")],
                          defaults=result.defaults)
    refined.warnings.extend(quality_warnings(proposed, spec))
    note = str(raw.get("rationale") or "").strip()
    message = f"accepted (transition score {old:.1f} → {new:.1f})"
    return Refinement(True, refined, f"{message}: {note}" if note else message)
