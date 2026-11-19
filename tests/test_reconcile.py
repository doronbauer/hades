import pytest

from hades.keys import parse_key
from hades.reconcile import (
    CONFLICT,
    CONSISTENT,
    HALF_DOUBLE,
    RELATIVE,
    UNRECOGNIZED,
    parse_bpm,
    resolve_bpm,
    resolve_genre,
    resolve_key,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("128", 128.0), ("128.00", 128.0), ("87,5", 87.5), (" 174 BPM", 174.0), (126.4, 126.4),
     ("0", None), ("", None), ("fast", None), ("1000", None), (None, None)],
)
def test_parse_bpm(raw, expected) -> None:
    assert parse_bpm(raw) == expected


def test_half_time_discrepancy_from_spec() -> None:
    r = resolve_bpm(87.0, 174.0)
    assert r.flag == HALF_DOUBLE
    assert (r.value, r.source) == (87.0, "tag")  # never silently corrected


def test_double_time_other_direction() -> None:
    assert resolve_bpm(140.0, 70.2).flag == HALF_DOUBLE


@pytest.mark.parametrize(("tag", "analyzed"), [(128.0, 128.0), (128.0, 126.0), (126.0, 128.5)])
def test_within_two_percent_is_consistent(tag: float, analyzed: float) -> None:
    assert resolve_bpm(tag, analyzed).flag == CONSISTENT


def test_just_over_two_percent_conflicts() -> None:
    assert resolve_bpm(128.0, 124.0).flag == CONFLICT


def test_conflict_prefers_tag_but_keeps_flag() -> None:
    r = resolve_bpm(128.0, 140.0)
    assert (r.value, r.source, r.flag) == (128.0, "tag", CONFLICT)


def test_single_value_used_without_flag() -> None:
    assert resolve_bpm(None, 174.0) == resolve_bpm(0.0, 174.0)
    r = resolve_bpm(None, 174.0)
    assert (r.value, r.source, r.flag) == (174.0, "analysis", None)
    r = resolve_bpm(128.0, None)
    assert (r.value, r.source, r.flag) == (128.0, "tag", None)
    assert resolve_bpm(None, None).value is None


def test_manual_bpm_wins_and_flag_is_retained() -> None:
    r = resolve_bpm(87.0, 174.0, manual=174.0)
    assert (r.value, r.source, r.flag) == (174.0, "manual", HALF_DOUBLE)


def test_key_notations_compared_after_normalization() -> None:
    r = resolve_key("8A", parse_key("A minor"))
    assert r.flag == CONSISTENT
    assert r.value.name == "Am"


def test_relative_and_conflicting_keys() -> None:
    assert resolve_key("C", parse_key("Am")).flag == RELATIVE
    r = resolve_key("Fm", parse_key("Am"))
    assert (r.flag, r.source, r.value.name) == (CONFLICT, "tag", "Fm")


def test_unrecognized_key_tag_falls_back_to_analysis() -> None:
    r = resolve_key("o", parse_key("Am"))
    assert (r.flag, r.source, r.value.name) == (UNRECOGNIZED, "analysis", "Am")


def test_manual_key() -> None:
    assert resolve_key("Fm", parse_key("Am"), manual="8A").source == "manual"


def test_genre_comes_from_tags_or_manual_only() -> None:
    assert resolve_genre("Techno") == resolve_genre("Techno", None)
    assert resolve_genre("Techno").source == "tag"
    assert resolve_genre("Techno", manual="Minimal").value == "Minimal"
    assert resolve_genre(None).value is None


def test_parallel_keys_flagged_separately() -> None:
    # Traktor 8m (Bb minor) vs analysis Bb major: same tonic, different mode.
    from hades.reconcile import PARALLEL

    r = resolve_key("8m", parse_key("Bb"))
    assert (r.flag, r.source, r.value.name) == (PARALLEL, "tag", "Bbm")
