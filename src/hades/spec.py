"""The structured playlist query specification.

An LLM translates a natural-language request into this shape; everything after
that (retrieval, sequencing, validation) is deterministic and works on a
``PlaylistSpec`` only.

Hard constraints (never silently violated): dates, BPM limits, genres, folder,
artists, an explicit track count, and total duration. Soft preferences (affect
ordering only): BPM progression and harmonic compatibility, under ``ordering``.

Dates are exclusive calendar bounds on the filesystem creation date, in local
time: "after March 2026" is ``created_after = 2026-03-31``.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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


class Ordering(BaseModel):
    """Soft preferences: they shape the order, never which tracks are allowed."""

    model_config = ConfigDict(extra="forbid")

    bpm_progression: Literal["ascending", "descending", "none"] = "none"
    harmonic_compatibility: bool = False


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
