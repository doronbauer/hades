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

    def seek_step(self, direction: int) -> None:
        with self.lock:
            self.calls.append(("seek_step", direction))

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

    def finish(self) -> None:
        """The current track plays to its end."""
        with self.lock:
            self.current = None

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
    return HadesApp(conn, config.database_path, lambda: llm, player, config.export_directory)


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
            await pilot.press("s")  # no stop key: pausing is enough
            await pilot.pause(0.1)
            assert player.actions()[-1] == ("toggle_pause",) and app.playing_id == third.id
            assert str(table.get_row_at(0)[6]) == "2026-05-10"  # filesystem creation date

            # Edits by stable track ID.
            await submit(pilot, app, f"move {third.id} to 1")
            assert [t.id for t in app.playlist.tracks[:3]] == [third.id, first.id, second.id]
            assert "BPM drop" in str(table.get_row_at(1)[7])  # transition indicator recalculated
            await submit(pilot, app, f"remove {first.id}")
            assert app.playlist.position_of(first.id) is None and table.row_count == 3
            spare = next(i for i in ids if app.playlist.position_of(i) is None)
            await submit(pilot, app, f"add {spare} at 2")
            assert app.playlist.position_of(spare) == 2
            await submit(pilot, app, "add 99999")
            assert "no track with ID 99999" in log_text(app)

            # Several IDs at once; a bad one is reported and the rest still apply.
            spares = [i for i in ids if app.playlist.position_of(i) is None][:2]
            await submit(pilot, app, f"add {spares[0]} 99999 {spares[1]} at 1")
            assert [t.id for t in app.playlist.tracks[:2]] == spares
            assert "no track with ID 99999" in log_text(app)
            await submit(pilot, app, f"remove {spares[0]} 99999 {spares[1]}")
            assert all(app.playlist.position_of(i) is None for i in spares)
            assert "track 99999 is not in the playlist" in log_text(app)
            assert len(app.playlist) == 4

            await submit(pilot, app, "save")
            assert "name the playlist first" in log_text(app)
            await submit(pilot, app, "name Warmup")
            await submit(pilot, app, "save")
            assert "Saved “Warmup” (4 tracks)" in log_text(app)
            saved = [t.id for t in app.playlist.tracks]
            m3u = config.export_directory / "Warmup.m3u"
            assert m3u.read_text().count(".mp3\n") == 4

            # Same request again: no LLM call, a different version, the saved one kept.
            calls = len(llm.calls)
            await submit(pilot, app, "again")
            await until(pilot, lambda: app.playlist.id is None and len(app.playlist) == 4)
            assert len(llm.calls) == calls
            assert app.playlist.prompt == "4 techno tracks getting faster"
            assert {t.id for t in app.playlist.tracks} != set(saved)
            assert "tracks are new" in log_text(app)

            await submit(pilot, app, "new")  # the new version is unsaved: confirm twice
            await submit(pilot, app, "new")
            assert len(app.playlist) == 0
            await submit(pilot, app, "playlists")
            assert "Warmup  (4 tracks" in log_text(app)
            await submit(pilot, app, "open Warmup")
            assert [t.id for t in app.playlist.tracks] == saved

    asyncio.run(scenario())


def test_mark_and_keep(conn, config, music_dir: Path) -> None:
    ids = [insert_track(conn, music_dir, artist=f"A{i}") for i in range(4)]

    async def scenario() -> None:
        app = make_app(conn, config, FakeLLM(), FakePlayer())
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(pilot, app, "add " + " ".join(map(str, ids)))
            assert [t.id for t in app.playlist.tracks] == ids
            await submit(pilot, app, "keep")
            assert "no tracks are marked" in log_text(app) and len(app.playlist) == 4

            table = app.query_one(PlaylistTable)
            await pilot.press("tab")
            table.move_cursor(row=0)
            await pilot.press("c", "down", "down", "c", "down", "c", "c")  # last one toggled off
            assert app.playlist.marked == {ids[0], ids[2]}
            assert "●" in str(table.get_row_at(0)[0]) and "●" not in str(table.get_row_at(1)[0])
            assert "2 marked" in str(app.query_one("#stats").render())

            await submit(pilot, app, "keep")
            assert [t.id for t in app.playlist.tracks] == [ids[0], ids[2]]
            assert "removed 2" in log_text(app) and app.playlist.marked == set()

    asyncio.run(scenario())


def test_grab_and_move_tracks(conn, config, music_dir: Path) -> None:
    ids = [insert_track(conn, music_dir, artist=f"A{i}") for i in range(4)]

    async def scenario() -> None:
        player = FakePlayer()
        app = make_app(conn, config, FakeLLM(), player)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(pilot, app, "add " + " ".join(map(str, ids)))
            app.playlist.dirty = False
            table = app.query_one(PlaylistTable)
            await pilot.press("tab")
            table.move_cursor(row=1)

            await pilot.press("g")
            assert "MOVING" in str(app.query_one("#help").render())
            assert "⇅" in str(table.get_row_at(1)[0])
            await pilot.press("down", "down", "down")  # the last press hits the bottom
            assert [t.id for t in app.playlist.tracks] == [ids[0], ids[2], ids[3], ids[1]]
            assert table.cursor_row == 3
            await pilot.press("escape")  # cancel: back to where it was, not dirty
            assert [t.id for t in app.playlist.tracks] == ids and not app.playlist.dirty
            assert table.cursor_row == 1 and "MOVING" not in str(app.query_one("#help").render())

            await pilot.press("g", "up", "up", "enter")  # drop at the top
            assert [t.id for t in app.playlist.tracks] == [ids[1], ids[0], ids[2], ids[3]]
            assert app.playlist.dirty and f"Moved {ids[1]} to position 1." in log_text(app)
            await pilot.press("down")  # plain navigation again
            assert table.cursor_row == 1
            await until(pilot, lambda: len(player.actions()) == 1)  # only that press played

            await pilot.press("g", "down", "tab")  # leaving the table drops the track
            assert [t.id for t in app.playlist.tracks] == [ids[1], ids[2], ids[0], ids[3]]
            assert app._grab is None and not table.moving

    asyncio.run(scenario())


def test_arrows_seek_the_playing_track(conn, config, music_dir: Path) -> None:
    ids = [insert_track(conn, music_dir, artist=f"A{i}") for i in range(3)]

    async def scenario() -> None:
        player = FakePlayer()
        app = make_app(conn, config, FakeLLM(), player)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(pilot, app, "add " + " ".join(map(str, ids)))
            await pilot.press("left", "right")  # in the prompt: text editing only
            await pilot.press("tab", "right")  # in the table, but nothing playing yet
            await pilot.pause(0.1)
            assert player.actions() == []

            await pilot.press("down")
            await pilot.press("right", "right", "right", "left")
            await until(pilot, lambda: len(player.actions()) == 5)
            assert player.actions()[1:] == [("seek_step", 1)] * 3 + [("seek_step", -1)]

            app.query_one(PlaylistTable).move_cursor(row=2)  # cursor off the playing track
            await pilot.press("right")
            await pilot.pause(0.1)
            assert len(player.actions()) == 5

    asyncio.run(scenario())


def test_auto_advance(conn, config, music_dir: Path) -> None:
    ids = [insert_track(conn, music_dir, artist="A0"),
           insert_track(conn, music_dir, artist="A1", create_file=False),
           insert_track(conn, music_dir, artist="A2"),
           insert_track(conn, music_dir, artist="A3")]

    async def scenario() -> None:
        player = FakePlayer()
        app = make_app(conn, config, FakeLLM(), player)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(pilot, app, "add " + " ".join(map(str, ids)))
            table = app.query_one(PlaylistTable)
            await pilot.press("tab")
            table.move_cursor(row=0)
            app.play_row(0)
            await until(pilot, lambda: app.player_state.playing)

            player.finish()  # skips the unavailable track
            await until(pilot, lambda: app.playing_id == ids[2])
            assert table.cursor_row == 2 and "Skipped 1 unavailable track." in log_text(app)
            await until(pilot, lambda: app.player_state.playing)

            player.finish()
            await until(pilot, lambda: app.playing_id == ids[3])
            await until(pilot, lambda: app.player_state.playing)
            player.finish()  # the last one: playback stops
            await until(pilot, lambda: app.playing_id is None)
            await pilot.pause(0.6)
            plays = [a[1] for a in player.actions() if a[0] == "play"]
            assert plays == [app.playlist.entries[r].track.path for r in (0, 2, 3)]
            assert table.cursor_row == 3

    asyncio.run(scenario())


def test_unsaved_changes_need_confirmation(conn, config, music_dir: Path) -> None:
    ids = [insert_track(conn, music_dir, artist=f"A{i}") for i in range(3)]

    async def scenario() -> None:
        app = make_app(conn, config, FakeLLM(), FakePlayer())
        async with app.run_test() as pilot:
            await submit(pilot, app, f"add {ids[0]}")
            await submit(pilot, app, "new")
            assert len(app.playlist) == 1 and "unsaved changes" in log_text(app)
            prompt = app.query_one("#prompt", Input)
            assert prompt.value == "new", "the input stays, so Enter confirms"
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.playlist) == 0 and prompt.value == ""

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
