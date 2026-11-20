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
    llm = FakeLLM(spec_answer(artists=["Surgeon"], exclude_artists=["Surgeon"]))
    result = interpret_request("whatever", llm, TODAY)
    assert result.spec is None
    assert "invalid or contradictory" in (result.error or "")
    assert "both requested and excluded" in (result.error or "")


def test_unrecognized_answer_shape() -> None:
    result = interpret_request("hi", FakeLLM({"error": None, "spec": None}), TODAY)
    assert result.spec is None and result.error


def test_llm_errors_propagate() -> None:
    with pytest.raises(LLMError):
        interpret_request("x", FakeLLM(LLMError("offline")), TODAY)


# --------------------------------------------------------------------------- filters and selection


def test_filter_groups_are_validated_and_described() -> None:
    spec = parse_spec({
        "selection": "all",
        "filters": [
            [{"field": "bpm", "op": "lt", "values": [90]}, {"field": "bpm", "op": "gt", "values": [150]}],
            [{"field": "key", "op": "in", "values": ["A minor", "9a"]}],
            [{"field": "genre", "op": "eq", "values": ["techno"], "negate": True}],
            [{"field": "added", "op": "between", "values": ["2026-12-31", "2026-01-01"]}],
            [],
        ],
        "ordering": {"bpm_progression": "ascending", "max_bpm_step": 2},
    })
    assert len(spec.filters) == 4, "empty groups are dropped"
    assert spec.filters[1][0].values == ["8A", "9A"], "keys become Camelot"
    assert spec.filters[3][0].values == ["2026-01-01", "2026-12-31"], "ranges are put in order"
    assert spec.hard_constraints() == [
        "all matching tracks", "BPM < 90 or BPM > 150", "key: 8A or 9A", "not (genre: techno)",
        "added 2026-01-01–2026-12-31", "BPM steps of at most 2"]
    assert parse_spec(spec.model_dump(mode="json")) == spec, "survives saving"


@pytest.mark.parametrize("condition, message", [
    ({"field": "genre", "op": "lt", "values": ["techno"]}, "can only be matched"),
    ({"field": "key", "op": "eq", "values": ["H major"]}, "unrecognized key"),
    ({"field": "bpm", "op": "between", "values": [120]}, "exactly two values"),
    ({"field": "bpm", "op": "gt", "values": ["fast"]}, "needs a number"),
    ({"field": "added", "op": "gt", "values": ["last year"]}, "needs a date"),
    ({"field": "key_reliable", "op": "eq", "values": ["maybe"]}, "true or false"),
    ({"field": "mood", "op": "eq", "values": ["happy"]}, "field"),
])
def test_invalid_filters_explain_why(condition: dict, message: str) -> None:
    with pytest.raises(SpecError, match=message):
        parse_spec({"filters": [[condition]]})


def test_all_matching_tracks_has_no_length() -> None:
    with pytest.raises(SpecError, match="cannot also have a track count"):
        parse_spec({"selection": "all", "track_count": 10})


def test_interpretation_with_filters_and_unsupported_parts() -> None:
    llm = FakeLLM({**spec_answer(
        selection="all",
        filters=[[{"field": "bpm", "op": "lt", "values": [90], "negate": False},
                  {"field": "bpm", "op": "gt", "values": [150], "negate": False}]],
        ordering={"bpm_progression": "ascending", "harmonic_compatibility": False,
                  "max_bpm_step": None}),
        "unsupported": ["with female vocals"]})
    result = interpret_request("all tracks under 90 or above 150 bpm with female vocals", llm, TODAY)
    assert result.error is None and result.spec is not None
    assert result.spec.selection == "all" and result.spec.ordering.max_bpm_step is None
    assert result.spec.hard_constraints()[1] == "BPM < 90 or BPM > 150"
    assert result.unsupported == ("with female vocals",)
    system, _ = llm.calls[0]
    assert "contradict" not in system.split("Set \"error\"")[1], \
        "non-overlapping ranges must not be taught as a contradiction"


def _cond(field: str, op: str, values: list) -> dict:
    return {"field": field, "op": op, "values": values, "negate": False}


@pytest.mark.parametrize("answer", [
    spec_answer(filters=[[_cond("bpm", "lt", [90])], [_cond("bpm", "gt", [150])]]),
    spec_answer(bpm={"min": 150, "max": 90}),
    spec_answer(bpm={"min": None, "max": 90}, filters=[[_cond("bpm", "gt", [150])]]),
])
def test_ranges_that_cannot_hold_together_become_alternatives(answer: dict) -> None:
    result = interpret_request("tracks under 90bpm and above 150bpm", FakeLLM(answer), TODAY)
    assert result.spec is not None and result.error is None
    assert len(result.spec.filters) == 1 and len(result.spec.filters[0]) == 2
    assert result.spec.bpm is None
    assert result.notes and "either/or" in result.notes[0]


def test_overlapping_ranges_are_kept_as_given() -> None:
    answer = spec_answer(bpm={"min": 120, "max": 130}, filters=[[_cond("bpm", "gt", [125])]])
    result = interpret_request("x", FakeLLM(answer), TODAY)
    assert result.spec is not None and result.notes == ()
    assert result.spec.hard_constraints() == ["BPM ≥ 120", "BPM ≤ 130", "BPM > 125"]
