"""An editable playlist: ordered tracks plus where they came from.

Positions are 1-based. Tracks are addressed by their stable library ID, never
by position. Manual edits may break the original generation criteria; they are
allowed, and ``violations()`` reports what no longer holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hades.builder import BuildResult, Stats, Transition, summarize, transitions, validate
from hades.db import now_iso
from hades.query import Track
from hades.spec import Ordering, PlaylistSpec

UNTITLED = "Untitled Playlist"


class EditError(Exception):
    """An edit could not be applied. The playlist is unchanged."""


@dataclass
class Entry:
    track: Track
    added_at: str


@dataclass
class Playlist:
    name: str = UNTITLED
    id: int | None = None                  # database ID once saved
    entries: list[Entry] = field(default_factory=list)
    prompt: str | None = None              # the natural-language request, if generated
    spec: PlaylistSpec | None = None       # requested generation parameters, if generated
    status: str = "manual"                 # complete | partial | manual
    named: bool = False                    # the user has assigned a name
    dirty: bool = False                    # unsaved changes
    saved_name: str | None = None          # name in the database at the last save/load
    marked: set[int] = field(default_factory=set)  # track IDs picked in the UI (not saved)

    @classmethod
    def from_result(cls, result: BuildResult, prompt: str | None) -> Playlist:
        now = now_iso()
        return cls(entries=[Entry(t, now) for t in result.tracks], prompt=prompt,
                   spec=result.spec, status=result.status, dirty=True)

    # ------------------------------------------------------------------ queries

    @property
    def tracks(self) -> list[Track]:
        return [e.track for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def position_of(self, track_id: int) -> int | None:
        for i, e in enumerate(self.entries, start=1):
            if e.track.id == track_id:
                return i
        return None

    @property
    def generated(self) -> bool:
        """Built from a request that can be run again or refined (not hand-picked)."""
        return self.spec is not None and self.status != "manual"

    @property
    def stats(self) -> Stats:
        return summarize(self.tracks)

    @property
    def ordering(self) -> Ordering:
        return self.spec.ordering if self.spec else Ordering()

    def transitions(self) -> list[Transition]:
        """transitions()[i] describes the move into position i + 2."""
        return transitions(self.tracks, self.ordering)

    def violations(self) -> list[str]:
        """Hard constraints of the original request the current playlist violates."""
        if self.spec is None:
            return [f"#{i} (ID {t.id}): file unavailable"
                    for i, t in enumerate(self.tracks, start=1) if not t.available]
        return validate(self.tracks, self.spec)

    # ------------------------------------------------------------------ edits

    def rename(self, name: str) -> None:
        name = " ".join(name.split())
        if not name:
            raise EditError("the playlist name must not be empty")
        self.name, self.named, self.dirty = name, True, True

    def move(self, track_id: int, position: int) -> None:
        current = self._require(track_id)
        self._check_position(position, len(self.entries))
        entry = self.entries.pop(current - 1)
        self.entries.insert(position - 1, entry)
        self.dirty = self.dirty or current != position

    def remove(self, track_id: int) -> Track:
        current = self._require(track_id)
        self.dirty = True
        self.marked.discard(track_id)
        return self.entries.pop(current - 1).track

    def toggle_mark(self, track_id: int) -> bool:
        """Mark or unmark a track; return whether it is marked now."""
        self._require(track_id)
        if track_id in self.marked:
            self.marked.discard(track_id)
            return False
        self.marked.add(track_id)
        return True

    def keep_marked(self) -> list[Track]:
        """Remove every track that is not marked, clear the marks, return the removed."""
        keep = self.marked & {t.id for t in self.tracks}
        if not keep:
            raise EditError("no tracks are marked (select a row and press C to mark it)")
        removed = [e.track for e in self.entries if e.track.id not in keep]
        self.entries = [e for e in self.entries if e.track.id in keep]
        self.marked.clear()
        self.dirty = self.dirty or bool(removed)
        return removed

    def add(self, track: Track, position: int | None = None) -> int:
        """Insert (default: append) and return the 1-based position used."""
        if self.position_of(track.id) is not None:
            raise EditError(f"track {track.id} is already in the playlist "
                            f"(position {self.position_of(track.id)})")
        if position is None:
            position = len(self.entries) + 1
        self._check_position(position, len(self.entries) + 1)
        self.entries.insert(position - 1, Entry(track, now_iso()))
        self.dirty = True
        return position

    def _require(self, track_id: int) -> int:
        pos = self.position_of(track_id)
        if pos is None:
            raise EditError(f"track {track_id} is not in the playlist")
        return pos

    @staticmethod
    def _check_position(position: int, last: int) -> None:
        if not 1 <= position <= last:
            raise EditError(f"position must be between 1 and {last}")
