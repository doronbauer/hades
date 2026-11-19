"""Minimal progress reporting to stderr (no third-party dependency)."""

from __future__ import annotations

import shutil
import sys
from typing import TextIO


class Progress:
    """Single updating line on a TTY; periodic plain lines otherwise."""

    def __init__(self, stream: TextIO | None = None, every: int = 100) -> None:
        self.stream = stream or sys.stderr
        self.tty = self.stream.isatty()
        self.every = every
        self._line_open = False

    def update(self, done: int, total: int | None, label: str) -> None:
        counter = f"[{done}/{total}]" if total is not None else f"[{done}]"
        if self.tty:
            width = shutil.get_terminal_size((100, 20)).columns - 1
            text = f"{counter} {label}"
            if len(text) > width:
                text = text[: max(0, width - 1)] + "…"
            self.stream.write("\r\x1b[K" + text)
            self.stream.flush()
            self._line_open = True
        elif done % self.every == 0 or done == total:
            self.stream.write(f"{counter} {label}\n")

    def note(self, message: str) -> None:
        self._clear()
        self.stream.write(message + "\n")
        self.stream.flush()

    def failure(self, path: str, error: str) -> None:
        self.note(f"FAILED {path}: {error}")

    def finish(self) -> None:
        self._clear()

    def _clear(self) -> None:
        if self._line_open:
            self.stream.write("\r\x1b[K")
            self._line_open = False


class NullProgress(Progress):
    def __init__(self) -> None:
        pass

    def update(self, done: int, total: int | None, label: str) -> None:
        pass

    def note(self, message: str) -> None:
        pass

    def finish(self) -> None:
        pass
