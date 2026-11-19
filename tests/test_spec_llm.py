"""Query specification validation and LLM interpretation (LLM mocked)."""

from __future__ import annotations

import re
from datetime import date

import pytest

from conftest import FakeLLM, spec_answer
from hades.llm import LLMError, interpret_request
from hades.spec import PlaylistSpec, SpecError, parse_spec

TODAY = date(2026, 11, 19)


def test_example_spec_validates() -> None:
    spec = parse_spec({
        "genre": ["Drum & Bass"], "created_after": "2026-03-31", "bpm": {"max": 160},
        "duration_minutes": 60, "duration_mode": "target", "track_count": None,
        "ordering": {"bpm_progression": "ascending", "harmonic_compatibility": True},
    })
    assert spec.created_after == date(2026, 3, 31)
    assert spec.duration_bounds_seconds == (55 * 60, 65 * 60)
    assert "BPM ≤ 160" in spec.hard_constraints()
    assert spec.soft_preferences() == ["ascending BPM", "harmonic transitions"]


def test_duration_modes() -> None:
    assert parse_spec({"duration_minutes": 60}).duration_mode == "target"
    strict = parse_spec({"duration_minutes": 60, "duration_mode": "max"})
    assert strict.duration_bounds_seconds == (None, 3600)
    assert PlaylistSpec().duration_bounds_seconds == (None, None)


@pytest.mark.parametrize("data, message", [
    ({"bpm": {"min": 140, "max": 120}}, "minimum BPM (140) is higher than the maximum (120)"),
    ({"created_after": "2026-04-01", "created_before": "2026-04-02"}, "no date is both"),
    ({"artists": ["Surgeon"], "exclude_artists": ["surgeon"]}, "both requested and excluded"),
    ({"duration_mode": "max"}, "duration mode was given without a duration"),
    ({"track_count": 0}, "track_count"),
    ({"bpm": {"max": 1000}}, "bpm.max"),
    ({"tempo": 120}, "tempo"),
])
def test_invalid_specs_explain_why(data: dict, message: str) -> None:
    with pytest.raises(SpecError, match=re.escape(message)):
        parse_spec(data)


def test_lists_are_cleaned() -> None:
    spec = parse_spec({"genre": [" Techno ", "techno", ""], "folder": "  "})
    assert spec.genre == ["Techno"] and spec.folder is None


def test_interpretation_passes_date_and_request() -> None:
    llm = FakeLLM(spec_answer(genre=["Drum & Bass"], created_after="2026-03-31",
                              bpm={"min": None, "max": 160}, duration_minutes=60,
                              duration_mode="target"))
    result = interpret_request("Create a one-hour playlist of drum & bass tracks below 160 BPM, "
                               "added after March 2026.", llm, TODAY)
    assert result.error is None and result.spec is not None
    assert result.spec.bpm is not None and result.spec.bpm.max == 160 and result.spec.bpm.min is None
    assert result.spec.created_after == date(2026, 3, 31)
    system, user = llm.calls[0]
    assert "2026-11-19" in user and "drum & bass" in user
    assert '"after March 2026" -> created_after "2026-03-31"' in system


def test_llm_reported_contradiction_is_returned() -> None:
    llm = FakeLLM({"error": "You asked for tracks both below 120 and above 140 BPM.", "spec": None})
    result = interpret_request("below 120 and above 140 bpm", llm, TODAY)
    assert result.spec is None and "below 120" in (result.error or "")


def test_contradiction_the_llm_missed_is_caught() -> None:
    llm = FakeLLM(spec_answer(bpm={"min": 140, "max": 120}))
    result = interpret_request("whatever", llm, TODAY)
    assert result.spec is None
    assert "invalid or contradictory" in (result.error or "")
    assert "higher than the maximum" in (result.error or "")


def test_unrecognized_answer_shape() -> None:
    result = interpret_request("hi", FakeLLM({"error": None, "spec": None}), TODAY)
    assert result.spec is None and result.error


def test_llm_errors_propagate() -> None:
    with pytest.raises(LLMError):
        interpret_request("x", FakeLLM(LLMError("offline")), TODAY)
