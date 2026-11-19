"""Deterministic parsing of playlist commands typed at the prompt.

``parse_command`` returns a ``Command`` for recognized commands, raises
``CommandError`` for a recognized command word with bad arguments, and returns
None for anything else (which is then treated as a natural-language request).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class CommandError(ValueError):
    """A command word was recognized but its arguments are invalid."""


@dataclass(frozen=True)
class Command:
    action: str                 # name | rename | move | remove | add | save | playlists | open | ...
    text: str | None = None     # name argument
    track_id: int | None = None
    position: int | None = None


USAGE = {
    "name": "name <playlist name>",
    "rename": "rename <new name>",
    "move": "move <track ID> to <position>",
    "remove": "remove <track ID>",
    "add": "add <track ID> [at <position>]",
    "save": "save",
    "playlists": "playlists",
    "open": "open <playlist name>",
    "refine": "refine",
    "new": "new",
    "help": "help",
}

_ALIASES = {"rm": "remove", "delete": "remove", "ls": "playlists", "load": "open", "?": "help"}

_MOVE = re.compile(r"(\d+)\s+(?:to\s+)?(\d+)", re.IGNORECASE)
_ADD = re.compile(r"(\d+)(?:\s+(?:at\s+)?(\d+))?", re.IGNORECASE)
_ID = re.compile(r"(\d+)")


def parse_command(text: str) -> Command | None:
    text = text.strip()
    if not text:
        return None
    word, _, rest = text.partition(" ")
    action = _ALIASES.get(word.lower(), word.lower())
    rest = rest.strip()
    if action not in USAGE:
        return None

    def fail() -> CommandError:
        return CommandError(f"usage: {USAGE[action]}")

    if action in ("name", "rename", "open"):
        if not rest:
            raise fail()
        return Command(action, text=rest)
    if action in ("save", "playlists", "refine", "new", "help"):
        if rest:
            raise fail()
        return Command(action)
    if action == "move":
        if not (m := _MOVE.fullmatch(rest)):
            raise fail()
        return Command(action, track_id=int(m[1]), position=int(m[2]))
    if action == "remove":
        if not (m := _ID.fullmatch(rest)):
            raise fail()
        return Command(action, track_id=int(m[1]))
    if action == "add":
        if not (m := _ADD.fullmatch(rest)):
            raise fail()
        return Command(action, track_id=int(m[1]), position=int(m[2]) if m[2] else None)
    raise AssertionError(action)


HELP_TEXT = "Commands: " + " · ".join(USAGE.values()) + \
    " — anything else is a natural-language playlist request."
