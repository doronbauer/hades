"""Audio preview playback behind a small interface, implemented with mpv.

mpv runs as a separate idle process controlled over its JSON IPC socket, so
decoding never blocks the terminal UI. Calls are synchronous but short (local
socket, bounded timeouts); the UI runs them off its event loop.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

PreviewMode = Literal["START", "MIDDLE"]
IPC_TIMEOUT = 1.0
STARTUP_TIMEOUT = 5.0


class PlayerError(Exception):
    """Playback failed or the backend is unavailable."""


@dataclass(frozen=True)
class PlayerState:
    path: str | None = None
    playing: bool = False         # a file is loaded (possibly paused)
    paused: bool = False
    position: float | None = None
    duration: float | None = None
    error: str | None = None      # last playback error, e.g. an undecodable file


class Player(Protocol):
    def play(self, path: str, mode: PreviewMode) -> None: ...
    def toggle_pause(self) -> None: ...
    def stop(self) -> None: ...
    def state(self) -> PlayerState: ...
    def close(self) -> None: ...


class NullPlayer:
    """Stands in when no backend is available; every play reports why."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def play(self, path: str, mode: PreviewMode) -> None:
        raise PlayerError(self.reason)

    def toggle_pause(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def state(self) -> PlayerState:
        return PlayerState()

    def close(self) -> None:
        pass


def start_offset(mode: PreviewMode) -> str:
    """mpv --start value: the beginning, or half of the file's duration."""
    return "50%" if mode == "MIDDLE" else "0"


class MpvPlayer:
    def __init__(self, mpv_path: str = "mpv") -> None:
        exe = shutil.which(mpv_path)
        if exe is None:
            raise PlayerError(f"mpv not found ({mpv_path!r}); install it, e.g. `brew install mpv`")
        self._exe = exe
        self._proc: subprocess.Popen[bytes] | None = None
        self._sock: socket.socket | None = None
        self._dir: str | None = None
        self._buf = b""
        self._next_id = 0
        self._lock = threading.Lock()
        self._path: str | None = None
        self._error: str | None = None

    # ------------------------------------------------------------------ process / IPC

    def _ensure(self) -> socket.socket:
        if self._sock is not None and self._proc is not None and self._proc.poll() is None:
            return self._sock
        self._shutdown()
        self._dir = tempfile.mkdtemp(prefix="hades-mpv-")
        ipc = os.path.join(self._dir, "ipc.sock")
        self._proc = subprocess.Popen(
            [self._exe, "--idle=yes", "--no-video", "--no-terminal", "--keep-open=no",
             "--audio-display=no", f"--input-ipc-server={ipc}"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while True:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(ipc)
                break
            except OSError:
                sock.close()
                if self._proc.poll() is not None or time.monotonic() > deadline:
                    self._shutdown()
                    raise PlayerError("mpv failed to start") from None
                time.sleep(0.05)
        sock.settimeout(IPC_TIMEOUT)
        self._sock, self._buf = sock, b""
        return sock

    def _command(self, *args: Any) -> Any:
        with self._lock:
            sock = self._ensure()
            self._next_id += 1
            rid = self._next_id
            try:
                sock.sendall(json.dumps({"command": list(args), "request_id": rid}).encode() + b"\n")
                while True:
                    while b"\n" not in self._buf:
                        chunk = sock.recv(65536)
                        if not chunk:
                            raise OSError("mpv closed the connection")
                        self._buf += chunk
                    line, self._buf = self._buf.split(b"\n", 1)
                    msg = json.loads(line)
                    if msg.get("request_id") == rid:
                        break
                    self._event(msg)
            except (OSError, ValueError) as exc:
                self._shutdown()
                raise PlayerError(f"lost contact with mpv: {exc}") from None
        if msg.get("error") != "success":
            raise PlayerError(str(msg.get("error")))
        return msg.get("data")

    def _event(self, msg: dict[str, Any]) -> None:
        if msg.get("event") == "end-file" and msg.get("reason") == "error":
            self._error = f"could not play file: {msg.get('file_error') or 'unknown error'}"

    def _get(self, prop: str) -> Any:
        try:
            return self._command("get_property", prop)
        except PlayerError:
            return None  # e.g. "property unavailable" while idle

    def _shutdown(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

    # ------------------------------------------------------------------ Player

    def play(self, path: str, mode: PreviewMode) -> None:
        if not os.path.isfile(path):
            raise PlayerError(f"file not found: {path}")
        if not os.access(path, os.R_OK):
            raise PlayerError(f"file not readable: {path}")
        self._error = None
        self._path = path
        self._command("set_property", "start", start_offset(mode))
        self._command("loadfile", path, "replace")
        self._command("set_property", "pause", False)

    def toggle_pause(self) -> None:
        if self._get("idle-active") is False:
            self._command("cycle", "pause")

    def stop(self) -> None:
        if self._sock is not None:
            self._command("stop")

    def state(self) -> PlayerState:
        if self._sock is None:
            return PlayerState(error=self._error)
        idle = self._get("idle-active")
        if idle or idle is None:
            return PlayerState(error=self._error)
        return PlayerState(
            path=self._path, playing=True, paused=bool(self._get("pause")),
            position=self._get("time-pos"), duration=self._get("duration"), error=self._error)

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.sendall(b'{"command": ["quit"]}\n')
                except OSError:
                    pass
            self._shutdown()


def make_player(mpv_path: str) -> Player:
    try:
        return MpvPlayer(mpv_path)
    except PlayerError as exc:
        return NullPlayer(f"playback unavailable: {exc}")
