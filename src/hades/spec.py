"""The structured playlist query specification.

An LLM translates a natural-language request into this shape; everything after
that (retrieval, sequencing, validation) is deterministic and works on a
``PlaylistSpec`` only.

Hard constraints (never silently violated): dates, BPM limits, genres, folder,
artists, a per-artist limit, an explicit track count, total duration, free-form
``filters`` and a maximum BPM step between neighbours. Soft preferences (affect
ordering only): BPM progression and harmonic compatibility, under ``ordering``.

``selection`` is "set" (a playlist of some length; one hour if none is given) or
"all" (every matching track).

``filters`` combine conditions on more fields: every group must hold, and a group
holds if any of its conditions does. "Below 90 or above 150 BPM" is one group of
two conditions; "added in 2024 and not techno" is two groups of one.

Dates are exclusive calendar bounds on the filesystem creation date, in local
time: "after March 2026" is ``created_after = 2026-03-31``.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from hades.keys import parse_key
from hades.reconcile import BPM_MAX, BPM_MIN

DEFAULT_TOLERANCE_MINUTES = 5.0
MAX_TRACK_COUNT = 500


class SpecError(ValueError):
    """The request is invalid or contradictory. The message explains why."""


class BpmRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float | None = Field(default=None, ge=BPM_MIN, le=BPM_MAX)
    max: float | None = Field(default=None, ge=BPM_MIN, le=BPM_MAX)

    @model_validator(mode="after")
    def _ordered(self) -> BpmRange:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"the minimum BPM ({self.min:g}) is higher than the maximum ({self.max:g})"
            )
        return self


# Filterable fields by kind. Missing values never satisfy a condition (negated or not).
NUMERIC_FIELDS = ("bpm", "year", "duration_minutes", "loudness_lufs")
DATE_FIELDS = ("added",)
TEXT_FIELDS = ("genre", "artist", "title", "album", "format", "folder")
KEY_FIELDS = ("key",)
FLAG_FIELDS = ("key_reliable", "bpm_reliable")
FILTER_FIELDS = NUMERIC_FIELDS + DATE_FIELDS + TEXT_FIELDS + KEY_FIELDS + FLAG_FIELDS
FILTER_OPS = ("eq", "in", "lt", "le", "gt", "ge", "between")

FIELD_LABELS = {
    "bpm": "BPM", "year": "release year", "duration_minutes": "length (min)",
    "loudness_lufs": "loudness (LUFS)", "added": "added", "genre": "genre", "artist": "artist",
    "title": "title", "album": "album", "format": "format", "folder": "folder", "key": "key",
    "key_reliable": "key is certain", "bpm_reliable": "BPM is certain",
}
_OP_SYMBOLS = {"eq": "=", "lt": "<", "le": "≤", "gt": ">", "ge": "≥"}

FilterValue = Union[bool, float, str]


class Condition(BaseModel):
    """One test on one field. Text and key conditions match any of their values."""

    model_config = ConfigDict(extra="forbid")

    field: Literal[FILTER_FIELDS]  # type: ignore[valid-type]
    op: Literal[FILTER_OPS]  # type: ignore[valid-type]
    values: list[FilterValue]
    negate: bool = False

    @model_validator(mode="after")
    def _typed(self) -> Condition:
        f, op = self.field, self.op
        if f in TEXT_FIELDS or f in KEY_FIELDS:
            if op not in ("eq", "in"):
                raise ValueError(f"{f} can only be matched against values, not compared ({op})")
            texts = [" ".join(str(v).split()) for v in self.values if not isinstance(v, bool)]
            texts = [t for t in texts if t]
            if f in KEY_FIELDS:
                keys = [parse_key(t) for t in texts]
                bad = [t for t, k in zip(texts, keys) if k is None]
                if bad:
                    raise ValueError(f"unrecognized key: {', '.join(bad)}")
                texts = list(dict.fromkeys(k.camelot for k in keys if k is not None))
            self.values = list(texts)
        elif f in FLAG_FIELDS:
            if op != "eq" or len(self.values) != 1:
                raise ValueError(f"{f} takes a single true/false value")
            v = self.values[0]
            if isinstance(v, str) and v.strip().lower() in ("true", "false"):
                v = v.strip().lower() == "true"
            if not isinstance(v, bool):
                raise ValueError(f"{f} must be true or false")
            self.values = [v]
        else:
            if op == "in":
                op = self.op = "eq" if len(self.values) == 1 else "in"
            parse = _parse_date if f in DATE_FIELDS else _parse_number
            if op == "in" and f in DATE_FIELDS:
                raise ValueError("dates can be compared, not listed")
            values: list[FilterValue] = [parse(f, v) for v in self.values]
            if op == "between":
                if len(values) != 2:
                    raise ValueError(f"{f} between needs exactly two values")
                values = sorted(values)  # type: ignore[type-var]
            elif op != "in" and len(values) != 1:
                raise ValueError(f"{f} {op} needs exactly one value")
            self.values = values
        if not self.values:
            raise ValueError(f"no values given for {f}")
        return self

    def describe(self) -> str:
        label, vals = FIELD_LABELS[self.field], [_fmt(v) for v in self.values]
        if self.field in FLAG_FIELDS:
            text = label if self.values[0] else f"{label.replace(' is ', ' is not ')}"
            return f"not {text}" if self.negate else text
        if self.op == "between":
            text = f"{label} {vals[0]}–{vals[1]}"
        elif self.op in ("eq", "in") and (self.field in TEXT_FIELDS or self.field in KEY_FIELDS):
            text = f"{label}: {' or '.join(vals)}"
        elif self.op == "in":
            text = f"{label} in {', '.join(vals)}"
        else:
            text = f"{label} {_OP_SYMBOLS[self.op]} {vals[0]}"
        return f"not ({text})" if self.negate else text


def _parse_number(field: str, v: FilterValue) -> float:
    if isinstance(v, bool):
        raise ValueError(f"{field} needs a number, not {v}")
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"{field} needs a number, not {v!r}") from None


def _parse_date(field: str, v: FilterValue) -> str:
    try:
        return date.fromisoformat(str(v).strip()).isoformat()
    except ValueError:
        raise ValueError(f"{field} needs a date (YYYY-MM-DD), not {v!r}") from None


def _fmt(v: FilterValue) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def describe_group(group: list[Condition]) -> str:
    return " or ".join(c.describe() for c in group)


class Ordering(BaseModel):
    """How tracks follow each other.

    Progression and harmonic compatibility are preferences. ``max_bpm_step`` is a
    hard rule: neighbours differ by at most that many BPM, and with a progression
    no step may go against its direction.
    """

    model_config = ConfigDict(extra="forbid")

    bpm_progression: Literal["ascending", "descending", "none"] = "none"
    harmonic_compatibility: bool = False
    max_bpm_step: float | None = Field(default=None, gt=0, le=100)


class PlaylistSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    genre: list[str] = Field(default_factory=list)       # any of these genres
    created_after: date | None = None                     # exclusive, local calendar date
    created_before: date | None = None                    # exclusive, local calendar date
    bpm: BpmRange | None = None
    duration_minutes: float | None = Field(default=None, gt=0, le=24 * 60)
    duration_mode: Literal["target", "max"] | None = None
    duration_tolerance_minutes: float = Field(default=DEFAULT_TOLERANCE_MINUTES, ge=0)
    track_count: int | None = Field(default=None, ge=1, le=MAX_TRACK_COUNT)
    folder: str | None = None
    artists: list[str] = Field(default_factory=list)          # only these artists
    exclude_artists: list[str] = Field(default_factory=list)
    max_tracks_per_artist: int | None = Field(default=None, ge=1)  # None: repeats allowed
    selection: Literal["set", "all"] = "set"
    filters: list[list[Condition]] = Field(default_factory=list)  # AND of ORs
    ordering: Ordering = Field(default_factory=Ordering)

    @model_validator(mode="before")
    @classmethod
    def _default_mode(cls, data: object) -> object:
        if isinstance(data, dict) and data.get("duration_minutes") is not None \
                and data.get("duration_mode") is None:
            data = {**data, "duration_mode": "target"}
        return data

    @model_validator(mode="after")
    def _consistent(self) -> PlaylistSpec:
        self.genre = _clean_list(self.genre)
        self.artists = _clean_list(self.artists)
        self.exclude_artists = _clean_list(self.exclude_artists)
        if self.folder is not None and not self.folder.strip():
            self.folder = None
        if self.bpm is not None and self.bpm.min is None and self.bpm.max is None:
            self.bpm = None
        self.filters = [g for g in self.filters if g]
        if self.selection == "all" and (self.track_count is not None
                                        or self.duration_minutes is not None):
            raise ValueError("“all matching tracks” cannot also have a track count or duration")

        if self.duration_mode is not None and self.duration_minutes is None:
            raise ValueError("a duration mode was given without a duration")
        if self.duration_mode is None:
            self.duration_tolerance_minutes = DEFAULT_TOLERANCE_MINUTES
        elif self.duration_tolerance_minutes >= self.duration_minutes:  # type: ignore[operator]
            raise ValueError("the duration tolerance must be smaller than the duration itself")
        if self.created_after and self.created_before:
            if (self.created_before - self.created_after).days <= 1:
                raise ValueError(
                    f"no date is both after {self.created_after} and before {self.created_before}"
                )
        both = {a.casefold() for a in self.artists} & {a.casefold() for a in self.exclude_artists}
        if both:
            raise ValueError(f"artist(s) both requested and excluded: {', '.join(sorted(both))}")
        return self

    # ------------------------------------------------------------------ helpers

    @property
    def duration_bounds_seconds(self) -> tuple[float | None, float | None]:
        """(lower, upper) total-duration bounds in seconds, or (None, None)."""
        if self.duration_minutes is None:
            return None, None
        target = self.duration_minutes * 60
        if self.duration_mode == "max":
            return None, target
        tol = self.duration_tolerance_minutes * 60
        return target - tol, target + tol

    def hard_constraints(self) -> list[str]:
        out: list[str] = []
        if self.selection == "all":
            out.append("all matching tracks")
        if self.genre:
            out.append("genre: " + " or ".join(self.genre))
        if self.created_after:
            out.append(f"file created after {self.created_after}")
        if self.created_before:
            out.append(f"file created before {self.created_before}")
        if self.bpm and self.bpm.min is not None:
            out.append(f"BPM ≥ {self.bpm.min:g}")
        if self.bpm and self.bpm.max is not None:
            out.append(f"BPM ≤ {self.bpm.max:g}")
        if self.track_count is not None:
            out.append(f"{self.track_count} tracks")
        if self.duration_minutes is not None:
            if self.duration_mode == "max":
                out.append(f"at most {self.duration_minutes:g} min")
            else:
                out.append(f"{self.duration_minutes:g} ± {self.duration_tolerance_minutes:g} min")
        if self.folder:
            out.append(f"folder: {self.folder}")
        if self.artists:
            out.append("artist: " + " or ".join(self.artists))
        if self.exclude_artists:
            out.append("excluding: " + ", ".join(self.exclude_artists))
        if self.max_tracks_per_artist is not None:
            out.append(f"at most {self.max_tracks_per_artist} per artist")
        out += [describe_group(g) for g in self.filters]
        if self.ordering.max_bpm_step is not None:
            out.append(f"BPM steps of at most {self.ordering.max_bpm_step:g}")
        return out

    def soft_preferences(self) -> list[str]:
        out: list[str] = []
        if self.ordering.bpm_progression != "none":
            out.append(f"{self.ordering.bpm_progression} BPM")
        if self.ordering.harmonic_compatibility:
            out.append("harmonic transitions")
        return out


def parse_spec(data: object) -> PlaylistSpec:
    """Validate untrusted data (e.g. LLM output); raise SpecError with a readable reason."""
    try:
        return PlaylistSpec.model_validate(data)
    except ValidationError as exc:
        reasons = []
        for err in exc.errors():
            where = ".".join(str(p) for p in err["loc"])
            msg = err["msg"].removeprefix("Value error, ")
            reasons.append(f"{where}: {msg}" if where else msg)
        raise SpecError("; ".join(reasons)) from None


def _clean_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        s = " ".join(item.split())
        if s and s.casefold() not in seen:
            seen.add(s.casefold())
            out.append(s)
    return out
