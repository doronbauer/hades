"""Musical-key parsing and normalization.

Recognized notations (all map to the same internal ``Key``):

* Standard: ``Am``, ``A minor``, ``A min``, ``F#``, ``F# major``, ``Bbm``, ``B♭ minor``
* Camelot: ``8A`` (A minor), ``8B`` (C major)
* Open Key: ``1m`` (A minor), ``1d`` (C major)

Anything else (Traktor's ``o``/off-key, ``Am/C``, a bare lowercase ``a``, a
lone uppercase ``M`` suffix…) is treated as unrecognized: the caller keeps the
original text and never guesses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAJOR_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MINOR_NAMES = ("Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm")

_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTALS = {"": 0, "#": 1, "♯": 1, "b": -1, "♭": -1}

_STANDARD_RE = re.compile(
    r"^(?P<note>[A-Ga-g])\s*(?P<acc>[#♯b♭]?)\s*"
    r"(?P<mode>m|min|minor|maj|major|M)?$"
)
_CAMELOT_RE = re.compile(r"^(?P<num>1[0-2]|0?[1-9])\s*(?P<letter>[ABab])$")
_OPENKEY_RE = re.compile(r"^(?P<num>1[0-2]|0?[1-9])\s*(?P<letter>[dmDM])$")


@dataclass(frozen=True)
class Key:
    pitch_class: int  # 0 = C … 11 = B (tonic)
    minor: bool

    @property
    def name(self) -> str:
        return (MINOR_NAMES if self.minor else MAJOR_NAMES)[self.pitch_class]

    @property
    def camelot(self) -> str:
        major_pc = (self.pitch_class + 3) % 12 if self.minor else self.pitch_class
        number = (7 * major_pc + 7) % 12 + 1
        return f"{number}{'A' if self.minor else 'B'}"

    @property
    def open_key(self) -> str:
        major_pc = (self.pitch_class + 3) % 12 if self.minor else self.pitch_class
        number = (7 * major_pc) % 12 + 1
        return f"{number}{'m' if self.minor else 'd'}"

    @property
    def label(self) -> str:
        """Display form with both DJ notations, e.g. "Bb (6B / 11d)" (Camelot / Open Key)."""
        return f"{self.name} ({self.camelot} / {self.open_key})"

    def is_relative_of(self, other: Key) -> bool:
        """Same notes, different tonic: A minor / C major."""
        return self.minor != other.minor and self.camelot[:-1] == other.camelot[:-1]

    def is_parallel_of(self, other: Key) -> bool:
        """Same tonic, different mode: Bb major / Bb minor."""
        return self.minor != other.minor and self.pitch_class == other.pitch_class


_BY_CAMELOT = {Key(pc, minor).camelot: Key(pc, minor) for pc in range(12) for minor in (False, True)}
_BY_OPENKEY = {Key(pc, minor).open_key: Key(pc, minor) for pc in range(12) for minor in (False, True)}


def parse_key(text: str | None) -> Key | None:
    """Parse a key in any recognized notation; return None if unrecognized or ambiguous."""
    if text is None:
        return None
    s = " ".join(text.strip().split())
    if not s:
        return None

    if m := _CAMELOT_RE.match(s):
        return _BY_CAMELOT[f"{int(m['num'])}{m['letter'].upper()}"]
    if m := _OPENKEY_RE.match(s):
        return _BY_OPENKEY[f"{int(m['num'])}{m['letter'].lower()}"]

    m = _STANDARD_RE.match(s)
    if not m:
        return None
    note, acc, mode = m["note"], m["acc"], m["mode"]
    if mode is None and note.islower():
        return None  # some tools write minor keys in lowercase; too ambiguous to guess
    if mode == "M":
        return None  # "AM" is used for both A major and A minor in the wild
    minor = mode in ("m", "min", "minor")
    pc = (_NOTE_PC[note.upper()] + _ACCIDENTALS[acc]) % 12
    return Key(pc, minor)


def from_essentia(tonic: str, scale: str) -> Key | None:
    if scale not in ("major", "minor"):
        return None
    return parse_key(f"{tonic} {scale}")


def camelot_parts(camelot: str) -> tuple[int, str] | None:
    """Split "8A" into (8, "A"); None if not a valid Camelot code."""
    m = re.fullmatch(r"(1[0-2]|[1-9])([AB])", camelot.strip().upper())
    return (int(m[1]), m[2]) if m else None


def harmonic_compatibility(a: str, b: str) -> float:
    """Score a transition between two Camelot keys from 0.0 (clash) to 1.0 (same key).

    Same key 1.0; one step around the wheel 0.9; relative major/minor 0.85;
    one step plus a mode change 0.5; two steps 0.25; anything else 0.0.
    Callers must not pass missing or unreliable keys here; they decide their own score.
    """
    pa, pb = camelot_parts(a), camelot_parts(b)
    if pa is None or pb is None:
        raise ValueError(f"not a Camelot key: {a if pa is None else b!r}")
    step = (pb[0] - pa[0]) % 12
    distance = min(step, 12 - step)
    same_mode = pa[1] == pb[1]
    if distance == 0:
        return 1.0 if same_mode else 0.85
    if distance == 1:
        return 0.9 if same_mode else 0.5
    if distance == 2 and same_mode:
        return 0.25
    return 0.0
