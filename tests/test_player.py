"""mpv playback backend, exercised against a fake mpv that speaks the JSON IPC protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import hades.player
from hades.player import MpvPlayer, NullPlayer, PlayerError, make_player, snap_target

FAKE_MPV = r'''#!{python}
import json, os, socket, sys
path = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--input-ipc-server="))
log = os.environ["FAKE_MPV_LOG"]
srv = socket.socket(socket.AF_UNIX)
srv.bind(path)
srv.listen(1)
conn, _ = srv.accept()
f = conn.makefile("rwb")
state = {"idle-active": True, "pause": False, "start": None}
for line in f:
    msg = json.loads(line)
    cmd = msg["command"]
    with open(log, "a") as lf:
        lf.write(json.dumps(cmd) + "\n")
    out = {"request_id": msg.get("request_id"), "error": "success"}
    if cmd[0] == "quit":
        break
    if cmd[0] == "get_property":
        if state["idle-active"] and cmd[1] in ("time-pos", "duration"):
            out["error"] = "property unavailable"
        else:
            out["data"] = {"time-pos": 12.5, "duration": 300.0}.get(cmd[1], state.get(cmd[1]))
    elif cmd[0] == "set_property":
        state[cmd[1]] = cmd[2]
    elif cmd[0] == "loadfile":
        if "corrupt" in cmd[1]:
            event = {"event": "end-file", "reason": "error", "file_error": "unrecognized format"}
            f.write(json.dumps(event).encode() + b"\n")
        else:
            state["idle-active"] = False
    elif cmd[0] == "stop":
        state["idle-active"] = True
    elif cmd[0] == "cycle":
        state["pause"] = not state["pause"]
    elif cmd[0] == "seek":
        pass  # like a real seek still in progress: time-pos keeps its old value
    f.write(json.dumps(out).encode() + b"\n")
    f.flush()
'''


@pytest.fixture
def fake_mpv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    exe = tmp_path / "fake-mpv"
    exe.write_text(FAKE_MPV.replace("{python}", sys.executable))
    exe.chmod(0o755)
    log = tmp_path / "mpv.log"
    monkeypatch.setenv("FAKE_MPV_LOG", str(log))
    return exe, log


@pytest.fixture
def player(fake_mpv):
    p = MpvPlayer(str(fake_mpv[0]))
    yield p
    p.close()


def commands(log: Path) -> list[list]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def audio(tmp_path: Path, name: str = "a.mp3") -> str:
    path = tmp_path / name
    path.write_bytes(b"\x00")
    return str(path)


def test_play_start_and_middle(player, fake_mpv, tmp_path: Path) -> None:
    path = audio(tmp_path)
    player.play(path, "START")
    st = player.state()
    assert st.playing and not st.paused and st.position == 12.5 and st.duration == 300.0
    assert st.path == path
    player.play(path, "MIDDLE")
    sent = commands(fake_mpv[1])
    starts = [c[2] for c in sent if c[:2] == ["set_property", "start"]]
    assert starts == ["0", "50%"]
    assert ["loadfile", path, "replace"] in sent


def test_pause_resume_stop(player, tmp_path: Path) -> None:
    player.play(audio(tmp_path), "START")
    player.toggle_pause()
    assert player.state().paused
    player.toggle_pause()
    assert not player.state().paused
    player.stop()
    assert not player.state().playing


def test_toggle_pause_when_idle_does_nothing(player, fake_mpv, tmp_path: Path) -> None:
    player.play(audio(tmp_path), "START")
    player.stop()
    player.toggle_pause()
    assert ["cycle", "pause"] not in commands(fake_mpv[1])


def test_missing_and_undecodable_files(player, tmp_path: Path) -> None:
    with pytest.raises(PlayerError, match="file not found"):
        player.play(str(tmp_path / "nope.mp3"), "START")
    player.play(audio(tmp_path, "corrupt.mp3"), "START")
    st = player.state()
    assert not st.playing and "unrecognized format" in (st.error or "")


def test_backend_unavailable(tmp_path: Path) -> None:
    p = make_player(str(tmp_path / "no-such-mpv"))
    assert isinstance(p, NullPlayer)
    with pytest.raises(PlayerError, match="mpv not found"):
        p.play(audio(tmp_path), "START")
    assert not p.state().playing


@pytest.mark.parametrize(("position", "direction", "expected"), [
    (0.0, 1, 30.0), (12.5, 1, 30.0), (150.0, 1, 180.0), (159.0, 1, 180.0),
    (179.6, 1, 210.0), (179.4, 1, 180.0),  # mpv reads a bit low right after a seek
    (269.0, 1, 270.0), (270.0, 1, None), (295.0, 1, None),
    (189.0, -1, 180.0), (182.0, -1, 180.0),
    (180.0, -1, 150.0), (181.0, -1, 150.0),  # just jumped there: skip it
    (20.0, -1, 0.0), (1.0, -1, 0.0), (0.0, -1, None),
])
def test_snap_target(position: float, direction: int, expected: float | None) -> None:
    target = snap_target(position, 300.0, direction)
    assert target is None if expected is None else target == pytest.approx(expected)


def test_snap_target_unknown_length() -> None:
    assert snap_target(10.0, None, 1) is None and snap_target(10.0, 0.0, -1) is None
    assert snap_target(None, 300.0, 1) is None


def seeks(log: Path) -> list[float]:
    return [c[1] for c in commands(log) if c[0] == "seek"]


def test_repeated_seeks_step_from_the_last_target(player, fake_mpv, tmp_path: Path) -> None:
    player.play(audio(tmp_path), "START")  # fake mpv reports 12.5 of 300 throughout
    for _ in range(3):
        player.seek_step(1)
    player.seek_step(-1)
    assert seeks(fake_mpv[1]) == pytest.approx([30.0, 60.0, 90.0, 60.0])
    assert all(c[2] == "absolute+exact" for c in commands(fake_mpv[1]) if c[0] == "seek")


def test_seek_uses_the_position_once_settled(player, fake_mpv, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hades.player, "SEEK_SETTLE", 0.0)
    player.play(audio(tmp_path), "START")
    player.seek_step(1)
    player.seek_step(1)
    assert seeks(fake_mpv[1]) == pytest.approx([30.0, 30.0])


def test_new_track_forgets_the_last_seek(player, fake_mpv, tmp_path: Path) -> None:
    player.play(audio(tmp_path), "START")
    player.seek_step(1)
    player.seek_step(1)
    player.play(audio(tmp_path), "START")
    player.seek_step(1)
    assert seeks(fake_mpv[1]) == pytest.approx([30.0, 60.0, 30.0])


def test_seek_when_idle_does_nothing(player, fake_mpv, tmp_path: Path) -> None:
    player.seek_step(1)  # never started
    player.play(audio(tmp_path), "START")
    player.stop()
    player.seek_step(1)
    assert seeks(fake_mpv[1]) == []
    NullPlayer("no backend").seek_step(1)
