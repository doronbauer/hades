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
from hades.spec import PlaylistSpec, SpecError, parse_spec

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

INTERPRET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["error", "spec"],
    "properties": {
        "error": _nullable({"type": "string"}),
        "spec": _nullable({
            "type": "object",
            "additionalProperties": False,
            "required": ["genre", "created_after", "created_before", "bpm", "duration_minutes",
                         "duration_mode", "duration_tolerance_minutes", "track_count", "folder",
                         "artists", "exclude_artists", "ordering"],
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
                "ordering": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["bpm_progression", "harmonic_compatibility"],
                    "properties": {
                        "bpm_progression": {"type": "string",
                                            "enum": ["ascending", "descending", "none"]},
                        "harmonic_compatibility": {"type": "boolean"},
                    },
                },
            },
        }),
    },
}

INTERPRET_SYSTEM = """\
You translate a DJ's request for a playlist from their local music library into a \
structured query. Output only what the request states; never invent a filter.

Fields (use null or [] for anything the request does not mention):
- genre: genre names exactly as the user said them (e.g. "drum & bass", "techno"). \
Several genres mean "any of these".
- created_after / created_before: EXCLUSIVE calendar-date bounds (YYYY-MM-DD) on when \
the file was added to the library. Words like "added", "new", "got", "bought" refer to this. \
Interpret calendar periods as whole periods: "after March 2026" -> created_after \
"2026-03-31"; "before March 2026" -> created_before "2026-03-01"; "in March 2026" -> \
created_after "2026-02-28" and created_before "2026-04-01"; "since 1 May 2026" (inclusive) \
-> created_after "2026-04-30". Resolve relative dates ("this year", "last month") against \
today's date, given below. Never use release dates.
- bpm: {min, max}; "below 160" -> max 160, "above 125" -> min 125, "125 to 135" -> both.
- duration_minutes + duration_mode: "a one-hour playlist", "about 90 minutes", "a 90-minute \
set" -> mode "target"; "no longer than an hour", "at most 45 minutes", "under an hour" -> \
mode "max". duration_tolerance_minutes: only if the user states one, else null.
- track_count: only if a number of tracks is requested.
- folder: a directory name or path, only if the user restricts to one.
- artists / exclude_artists: only artists the user names.
- ordering.bpm_progression: "ascending" for increasing tempo, "starting slow and getting \
faster", "build up"; "descending" for the reverse; otherwise "none".
- ordering.harmonic_compatibility: true if the user asks for harmonic/key-compatible/\
smooth harmonic transitions or mixing in key.

Set "error" (and spec null) only if the request is not a playlist request, is \
self-contradictory (e.g. "below 120 BPM and above 140 BPM"), or is impossible to express \
with these fields. Explain the problem in one or two plain sentences for the user. \
Never resolve a contradiction by changing what the user asked for."""


@dataclass(frozen=True)
class Interpretation:
    spec: PlaylistSpec | None
    error: str | None
    raw: dict[str, Any]


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
    if isinstance(data.get("bpm"), dict):
        data["bpm"] = {k: v for k, v in data["bpm"].items() if v is not None}
    try:
        return Interpretation(parse_spec(data), None, raw)
    except SpecError as exc:
        return Interpretation(None, f"the request is invalid or contradictory: {exc}", raw)


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
