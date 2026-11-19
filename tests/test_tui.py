"""Terminal UI driven through Textual's pilot, with the LLM and the player mocked."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from conftest import FakeLLM, insert_track, spec_answer
from hades.llm import LLMError
from hades.player import PlayerError, PlayerState
from hades.tui import HadesApp, PlaylistTable
from textual.widgets import Input


class FakePlayer:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.lock = threading.Lock()
        self.current: str | None = None
        self.paused = False

    def play(self, path: str, mode: str) -> None:
        with self.lock:
            self.calls.append(("play", path, mode))
            self.current, self.paused = path, False

    def toggle_pause(self) -> None:
        with self.lock:
            self.calls.append(("toggle_pause",))
            self.paused = not self.paused

    def stop(self) -> None:
        with self.lock:
            self.calls.append(("stop",))
            self.current = None

    def state(self) -> PlayerState:
        with self.lock:
            if self.current is None:
                return PlayerState()
            return PlayerState(path=self.current, playing=True, paused=self.paused,
                               position=10.0, duration=300.0)

    def close(self) -> None:
        pass

    def actions(self) -> list[tuple]:
        with self.lock:
            return [c for c in self.calls if c[0] != "state"]


async def until(pilot, condition, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await pilot.pause(0.02)


async def submit(pilot, app: HadesApp, text: str) -> None:
    app.query_one("#prompt", Input).value = text
    app.query_one("#prompt", Input).focus()
    await pilot.press("enter")
    await pilot.pause()


def log_text(app: HadesApp) -> str:
    return "\n".join(line.text for line in app.query_one("#log").lines)


def make_app(conn, config, llm, player) -> HadesApp:
    return HadesApp(conn, config.database_path, lambda: llm, player)


def test_generate_navigate_play_edit_save(conn, config, music_dir: Path) -> None:
    ids = [insert_track(conn, music_dir, artist=f"Artist {i}", bpm=124 + i, duration=330)
           for i in range(8)]
    llm = FakeLLM(spec_answer(track_count=4,
                              ordering={"bpm_progression": "ascending",
                                        "harmonic_compatibility": False}))
    player = FakePlayer()

    async def scenario() -> None:
        app = make_app(conn, config, llm, player)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(pilot, app, "4 techno tracks getting faster")
            await until(pilot, lambda: len(app.playlist) == 4)
            table = app.query_one(PlaylistTable)
            assert table.row_count == 4
            assert app.playlist.name == "Untitled Playlist"
            assert "All hard constraints satisfied." in log_text(app)
            first, second, third = (t for t in app.playlist.tracks[:3])

            # Up/Down in the prompt edits text; they must not play anything.
            await pilot.press("up", "down")
            assert player.actions() == []

            await pilot.press("tab")
            assert app.focused is table
            await pilot.press("down")
            await until(pilot, lambda: player.actions())
            assert player.actions() == [("play", second.path, "START")]
            assert app.playing_id == second.id

            await pilot.press("m")  # affects the next selection only
            assert app.mode == "MIDDLE"
            await pilot.pause(0.1)
            assert len(player.actions()) == 1
            await pilot.press("down")
            await until(pilot, lambda: len(player.actions()) == 2)
            assert player.actions()[-1] == ("play", third.path, "MIDDLE")

            await pilot.press("space")
            await until(pilot, lambda: player.actions()[-1] == ("toggle_pause",))
            await until(pilot, lambda: app.player_state.paused)
            await pilot.press("s")
            await until(pilot, lambda: player.actions()[-1] == ("stop",))
            assert app.playing_id is None

            # Edits by stable track ID.
            await submit(pilot, app, f"move {third.id} to 1")
            assert [t.id for t in app.playlist.tracks[:3]] == [third.id, first.id, second.id]
            assert "BPM drop" in str(table.get_row_at(1)[6])  # transition indicator recalculated
            await submit(pilot, app, f"remove {first.id}")
            assert app.playlist.position_of(first.id) is None and table.row_count == 3
            spare = next(i for i in ids if app.playlist.position_of(i) is None)
            await submit(pilot, app, f"add {spare} at 2")
            assert app.playlist.position_of(spare) == 2
            await submit(pilot, app, "add 99999")
            assert "no track with ID 99999" in log_text(app)

            await submit(pilot, app, "save")
            assert "name the playlist first" in log_text(app)
            await submit(pilot, app, "name Warmup")
            await submit(pilot, app, "save")
            assert "Saved “Warmup” (4 tracks)" in log_text(app)
            saved = [t.id for t in app.playlist.tracks]

            await submit(pilot, app, "new")
            assert len(app.playlist) == 0
            await submit(pilot, app, "playlists")
            assert "Warmup  (4 tracks" in log_text(app)
            await submit(pilot, app, "open Warmup")
            assert [t.id for t in app.playlist.tracks] == saved

    asyncio.run(scenario())


def test_unsaved_changes_need_confirmation(conn, config, music_dir: Path) -> None:
    ids = [insert_track(conn, music_dir, artist=f"A{i}") for i in range(3)]

    async def scenario() -> None:
        app = make_app(conn, config, FakeLLM(), FakePlayer())
        async with app.run_test() as pilot:
            await submit(pilot, app, f"add {ids[0]}")
            await submit(pilot, app, "new")
            assert len(app.playlist) == 1 and "unsaved changes" in log_text(app)
            await submit(pilot, app, "new")
            assert len(app.playlist) == 0

    asyncio.run(scenario())


def test_errors_do_not_crash(conn, config, music_dir: Path) -> None:
    missing = insert_track(conn, music_dir, create_file=False)
    llm = FakeLLM(LLMError("could not reach the LLM service"),
                  {"error": "Below 120 and above 140 BPM cannot both hold.", "spec": None})

    class BrokenPlayer(FakePlayer):
        def play(self, path: str, mode: str) -> None:
            raise PlayerError("mpv exploded")

    async def scenario() -> None:
        app = make_app(conn, config, llm, BrokenPlayer())
        async with app.run_test() as pilot:
            await submit(pilot, app, "some techno")
            await until(pilot, lambda: "could not reach" in log_text(app))
            await submit(pilot, app, "below 120 and above 140")
            await until(pilot, lambda: "cannot both hold" in log_text(app))
            await submit(pilot, app, "move 1 to")
            assert "usage: move" in log_text(app)

            await submit(pilot, app, f"add {missing}")
            assert "currently unavailable" in log_text(app)
            ok = insert_track(conn, music_dir)
            await submit(pilot, app, f"add {ok}")
            await pilot.press("tab")
            await pilot.press("up")  # the unavailable track
            assert "is unavailable" in log_text(app)
            await pilot.press("down")  # the player fails
            await until(pilot, lambda: "mpv exploded" in log_text(app))
            assert app.is_running

    asyncio.run(scenario())
