import pytest

from hades.keys import Key, from_essentia, parse_key

A_MINOR = Key(9, True)
C_MAJOR = Key(0, False)


@pytest.mark.parametrize(
    "text",
    ["Am", "A minor", "A min", "a minor", "am", "8A", "8a", "08A", "1m", "  A   minor "],
)
def test_a_minor_notations(text: str) -> None:
    assert parse_key(text) == A_MINOR


@pytest.mark.parametrize("text", ["C", "C major", "C maj", "8B", "1d"])
def test_c_major_notations(text: str) -> None:
    assert parse_key(text) == C_MAJOR


@pytest.mark.parametrize(
    ("a", "b"),
    [("Bb", "A#"), ("Bbm", "A#m"), ("Db", "C#"), ("F#m", "Gbm"), ("B♭ minor", "Bbm"), ("Cb", "B")],
)
def test_enharmonic_equivalents(a: str, b: str) -> None:
    assert parse_key(a) == parse_key(b)
    assert parse_key(a) is not None


@pytest.mark.parametrize("text", [None, "", "o", "Off", "Am/C", "a", "AM", "13A", "H", "8C", "unknown"])
def test_unrecognized_or_ambiguous(text: str | None) -> None:
    assert parse_key(text) is None


def test_camelot_and_open_key_round_trip() -> None:
    for pc in range(12):
        for minor in (False, True):
            key = Key(pc, minor)
            assert parse_key(key.camelot) == key
            assert parse_key(key.open_key) == key
            assert parse_key(key.name) == key


def test_known_camelot_codes() -> None:
    assert parse_key("Am").camelot == "8A"
    assert parse_key("C").camelot == "8B"
    assert parse_key("Fm").camelot == "4A"
    assert parse_key("F#").camelot == "2B"
    assert parse_key("Ebm").camelot == "2A"


def test_relative_keys() -> None:
    assert A_MINOR.is_relative_of(C_MAJOR)
    assert not A_MINOR.is_relative_of(Key(9, False))


def test_from_essentia() -> None:
    assert from_essentia("A", "minor") == A_MINOR
    assert from_essentia("Bb", "major") == Key(10, False)
    assert from_essentia("A", "") is None


def test_label_shows_camelot_and_open_key() -> None:
    assert parse_key("Bb").label == "Bb (6B / 11d)"
    assert parse_key("8m").label == "Bbm (3A / 8m)"
    assert parse_key("6m").label == "G#m (1A / 6m)"


def test_parallel_keys() -> None:
    assert parse_key("Bb").is_parallel_of(parse_key("Bbm"))
    assert not parse_key("Bb").is_parallel_of(parse_key("Gm"))  # that's the relative minor
