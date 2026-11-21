"""Interactive terminal UI: prompt, playlist table and player status.

Slow work never runs on the UI event loop: LLM requests and candidate queries
run in worker threads (with their own database connection), and every player
call goes through a single background thread so commands stay in order.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import work
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widgets import DataTable, Input, RichLog, Static

from hades.builder import BuildResult, apply_defaults, fmt_duration
from hades.commands import HELP_TEXT, Command, CommandError, parse_command
from hades.db import connect, now_iso
from hades.llm import LLMClient, LLMError, refine_playlist
from hades.player import Player, PlayerError, PlayerState, PreviewMode
from hades.playlist import EditError, Entry, Playlist
from hades.query import find_candidates, load_track, track_violations
from hades.export import export_m3u
from hades.repository import RepositoryError, list_playlists, load_playlist
from hades.service import (
    ExportError,
    generate,
    generate_again,
    report_lines,
    save_and_export,
)

FIXED_COLUMNS = (("#", 5), ("ID", 6), ("BPM", 5), ("Key", 4), ("Time", 7), ("Added", 10),
                 ("Notes", 26))
COLUMNS = ("#", "ID", "Artist — Title", "BPM", "Key", "Time", "Added", "Notes")
HELP_KEYS = ("↑↓ Navigate & play   Space Pause/resume   M Preview mode   C Mark/unmark   "
             "G Move track   Tab Switch focus   Enter Submit   Ctrl+Q Quit   (type `help` for commands)")
GRAB_HELP = "MOVING ⇅   ↑↓ Move track   G/Enter Drop here   Esc Cancel"


class PlaylistTable(DataTable):
    """DataTable that reports cursor moves made by the user (not by refreshes)."""

    BINDINGS = [
        Binding("space", "app.toggle_pause", "Pause"),
        Binding("m", "app.toggle_mode", "Preview mode"),
        Binding("c", "app.toggle_mark", "Mark"),
        Binding("g", "app.toggle_grab", "Move track"),
        Binding("escape", "app.cancel_grab", "Cancel move", show=False),
    ]

    moving = False  # set by the app while a track is grabbed: ↑↓ move it instead

    class Navigated(Message):
        def __init__(self, row: int) -> None:
            super().__init__()
            self.row = row

    class Shift(Message):
        """Move the grabbed track by `delta` positions."""

        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    class Drop(Message):
        """Leave the grabbed track where it is."""

    def _moved(self, action: Callable[[], None]) -> None:
        before = self.cursor_row
        action()
        if self.row_count and self.cursor_row != before:
            self.post_message(self.Navigated(self.cursor_row))

    def action_cursor_up(self) -> None:
        if self.moving:
            self.post_message(self.Shift(-1))
        else:
            self._moved(super().action_cursor_up)

    def action_cursor_down(self) -> None:
        if self.moving:
            self.post_message(self.Shift(1))
        else:
            self._moved(super().action_cursor_down)

    def action_select_cursor(self) -> None:
        if self.moving:
            self.post_message(self.Drop())
        else:
            super().action_select_cursor()

    def on_blur(self) -> None:
        if self.moving:
            self.post_message(self.Drop())


class HadesApp(App[None]):
    TITLE = "Hades"
    CSS = """
    #title { height: 1; padding: 0 1; background: $primary; color: $text; text-style: bold; }
    #playlist { height: 1fr; }
    #stats { height: auto; max-height: 3; padding: 0 1; }
    #player { height: 2; padding: 0 1; background: $boost; }
    #log { height: 8; padding: 0 1; border-top: solid $primary-darken-2; }
    #help { height: 1; padding: 0 1; color: $text-muted; }
    """

    def __init__(self, conn: sqlite3.Connection, db_path: Path,
                 llm_factory: Callable[[], LLMClient], player: Player,
                 export_dir: Path = Path("exports"), export_extension: str = ".m3u") -> None:
        super().__init__()
        self.export_extension = export_extension
        self.conn = conn
        self.db_path = db_path
        self.export_dir = export_dir
        self._llm_factory = llm_factory
        self._llm: LLMClient | None = None
        self.player = player
        self.playlist = Playlist()
        self.mode: PreviewMode = "START"
        self.playing_id: int | None = None
        self.player_state = PlayerState()
        self.busy: str | None = None             # description of running background work
        self._seen_playing = False
        self._pending_discard: str | None = None
        self._last_playback_error: str | None = None
        self._player_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="player")
        self._poll: Future[PlayerState] | None = None
        self._grab: tuple[int, int, bool] | None = None  # track ID, start position, dirty before

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield Static(id="title")
        yield PlaylistTable(id="playlist", cursor_type="row", zebra_stripes=True)
        yield Static(id="stats")
        yield Static(id="player")
        yield RichLog(id="log", wrap=True, markup=False)
        yield Static(HELP_KEYS, id="help")
        yield Input(placeholder="Describe a playlist, or type a command (help)", id="prompt")

    def on_mount(self) -> None:
        table = self.query_one(PlaylistTable)
        widths = dict(FIXED_COLUMNS)
        for label in COLUMNS:
            table.add_column(label, key=label, width=widths.get(label, self._title_width()))
        self.refresh_view()
        self.query_one("#prompt", Input).focus()
        self.set_interval(0.5, self._poll_player)

    def on_unmount(self) -> None:
        self._player_thread.submit(self.player.close)
        self._player_thread.shutdown(wait=True)

    # ------------------------------------------------------------------ rendering

    def _title_width(self) -> int:
        # Each column has 1 cell of padding on both sides.
        used = sum(w + 2 for _, w in FIXED_COLUMNS) + 2 + 2
        return max(20, self.size.width - used)

    def on_resize(self) -> None:
        table = self.query_one(PlaylistTable)
        if "Artist — Title" in table.columns:
            table.columns["Artist — Title"].width = self._title_width()  # type: ignore[index]
            self.refresh_view()

    def log_line(self, text: str, style: str = "") -> None:
        self.query_one("#log", RichLog).write(Text(text, style=style))

    def refresh_view(self, focus_track: int | None = None) -> None:
        pl = self.playlist
        self.query_one("#title", Static).update(
            f"HADES  |  {pl.name}{' *' if pl.dirty else ''}")

        table = self.query_one(PlaylistTable)
        old_row = table.cursor_row
        table.clear()
        title_w = self._title_width()
        issues_into = [()] + [t.issues for t in pl.transitions()]
        for pos, (entry, issues) in enumerate(zip(pl.entries, issues_into), start=1):
            t = entry.track
            notes = list(issues)
            if pl.spec is not None:
                notes += track_violations(t, pl.spec)
            elif not t.available:
                notes.append("file unavailable")
            grabbed = self._grab is not None and t.id == self._grab[0]
            marker = "⇅" if grabbed else ("▶" if t.id == self.playing_id else " ")
            mark = ("●", "bold magenta") if t.id in pl.marked else (" ", "")
            bpm = "–" if t.bpm is None else f"{t.bpm:.0f}" + ("" if t.bpm_reliable else "?")
            key = "–" if t.camelot is None else t.camelot + ("" if t.key_reliable else "?")
            time = fmt_duration(t.duration) if t.duration else "–"
            added = t.created_date.isoformat() if t.created_date else "–"
            style = "bold yellow" if grabbed else "bold green" if t.id == self.playing_id \
                else ("red" if not t.available else "")
            table.add_row(
                Text.assemble((marker, style), mark, (str(pos), style)), str(t.id), _fit(t.display, title_w, style),
                bpm, key, time, added, _fit(", ".join(notes), 26, "yellow"), key=str(t.id))
        if pl.entries:
            row = (pl.position_of(focus_track) or 1) - 1 if focus_track is not None \
                else min(old_row, len(pl) - 1)
            table.move_cursor(row=max(row, 0))

        s = pl.stats
        stats = f"{s.count} tracks | {fmt_duration(s.total_seconds)} total | {s.bpm_range}"
        if s.unknown_duration:
            stats += f" | {s.unknown_duration} with unknown length"
        if marked := len(pl.marked & {t.id for t in pl.tracks}):
            stats += f" | {marked} marked (`keep` removes the rest)"
        if pl.prompt:
            stats += f" | “{pl.prompt}”"
        text = Text(stats)
        violations = [v for v in pl.violations() if not v.startswith("#")]
        if violations:
            text.append("\n⚠ " + "; ".join(violations), style="bold yellow")
        self.query_one("#stats", Static).update(text)
        self.query_one("#help", Static).update(GRAB_HELP if self._grab else HELP_KEYS)
        self._render_player()

    def _render_player(self) -> None:
        st = self.player_state
        track = None
        if self.playing_id is not None:
            pos = self.playlist.position_of(self.playing_id)
            track = self.playlist.entries[pos - 1].track if pos else load_track(
                self.conn, self.playing_id)
        now = f"Now playing: {track.display}" if track else "Now playing: —"
        if st.playing:
            state = "Paused" if st.paused else "Playing"
            clock = f"{_clock(st.position)} / {_clock(st.duration)}"
        else:
            state, clock = "Stopped", "--:-- / --:--"
        busy = f"    ⏳ {self.busy}…" if self.busy else ""
        self.query_one("#player", Static).update(
            f"{now}\n{clock}    Preview: {self.mode}    {state}{busy}")

    # ------------------------------------------------------------------ playback

    def _player_call(self, fn: Callable[..., Any], *args: Any) -> None:
        def run() -> None:
            try:
                fn(*args)
            except PlayerError as exc:
                self.call_from_thread(self._playback_failed, str(exc))
        self._player_thread.submit(run)

    def _playback_failed(self, message: str) -> None:
        if message != self._last_playback_error:  # don't repeat it on every keypress
            self.log_line(f"Playback: {message}", "red")
        self._last_playback_error = message
        self.playing_id = None
        self.refresh_view()

    def play_row(self, row: int) -> None:
        if not 0 <= row < len(self.playlist):
            return
        track = self.playlist.entries[row].track
        self.playing_id = track.id
        self._seen_playing = False
        if not track.available:
            self._player_call(self.player.stop)
            self._playback_failed(f"track {track.id} is unavailable ({track.path})")
            return
        self._player_call(self.player.play, track.path, self.mode)
        self.refresh_view()

    def on_playlist_table_navigated(self, event: PlaylistTable.Navigated) -> None:
        self.play_row(event.row)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.play_row(event.cursor_row)

    def action_toggle_pause(self) -> None:
        self._player_call(self.player.toggle_pause)

    def action_toggle_mode(self) -> None:
        # Affects the next selection only; the current track keeps playing.
        self.mode = "MIDDLE" if self.mode == "START" else "START"
        self._render_player()

    def action_toggle_mark(self) -> None:
        table = self.query_one(PlaylistTable)
        if not 0 <= table.cursor_row < len(self.playlist):
            return
        self.playlist.toggle_mark(self.playlist.entries[table.cursor_row].track.id)
        self.refresh_view()

    # ------------------------------------------------------------------ moving tracks

    def action_toggle_grab(self) -> None:
        if self._grab is not None:
            self._end_grab(cancel=False)
            return
        table = self.query_one(PlaylistTable)
        if not 0 <= table.cursor_row < len(self.playlist):
            return
        track_id = self.playlist.entries[table.cursor_row].track.id
        self._grab = (track_id, table.cursor_row + 1, self.playlist.dirty)
        table.moving = True
        self.refresh_view(focus_track=track_id)

    def action_cancel_grab(self) -> None:
        if self._grab is None:
            raise SkipAction()
        self._end_grab(cancel=True)

    def on_playlist_table_shift(self, event: PlaylistTable.Shift) -> None:
        if self._grab is None:
            return
        track_id = self._grab[0]
        pos = self.playlist.position_of(track_id)
        if pos is None:  # the playlist was replaced meanwhile
            self._end_grab(cancel=False)
            return
        if 1 <= pos + event.delta <= len(self.playlist):
            self.playlist.move(track_id, pos + event.delta)
        self.refresh_view(focus_track=track_id)

    def on_playlist_table_drop(self, event: PlaylistTable.Drop) -> None:
        if self._grab is not None:
            self._end_grab(cancel=False)

    def _end_grab(self, cancel: bool) -> None:
        assert self._grab is not None
        track_id, start, was_dirty = self._grab
        self._grab = None
        self.query_one(PlaylistTable).moving = False
        pos = self.playlist.position_of(track_id)
        if pos is None:
            self.refresh_view()
            return
        if cancel or pos == start:  # back where it was: undo the dirty flag too
            self.playlist.move(track_id, min(start, len(self.playlist)))
            self.playlist.dirty = was_dirty
        else:
            self.log_line(f"Moved {track_id} to position {pos}.")
        self.refresh_view(focus_track=track_id)

    def _poll_player(self) -> None:
        if self._poll is not None and not self._poll.done():
            return
        self._poll = self._player_thread.submit(self.player.state)
        self._poll.add_done_callback(
            lambda f: self.call_from_thread(self._update_player, f)
            if f.exception() is None else None)

    def _update_player(self, future: Future[PlayerState]) -> None:
        st = future.result()
        self.player_state = st
        if st.error:
            self.log_line(f"Playback: {st.error}", "red")
            self._player_thread.submit(self.player.stop)
            self.playing_id = None
            self.refresh_view()
            return
        if st.playing:
            self._seen_playing = True
            self._last_playback_error = None
        elif self.playing_id is not None and self._seen_playing:
            self.playing_id = None  # the track ended; playback stops (no auto-advance)
            self.refresh_view()
            return
        self._render_player()

    # ------------------------------------------------------------------ prompt

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text:
            return
        self.log_line(f"> {text}", "bold")
        try:
            command = parse_command(text)
        except CommandError as exc:
            self.log_line(str(exc), "red")
            return
        if command is None or command.action in ("open", "new", "again"):
            if not self._may_discard(text):
                return
        if command is None:
            self.start_generation(text)
        else:
            self.run_command(command)

    def _may_discard(self, text: str) -> bool:
        """Replacing a playlist with unsaved changes needs the same input twice.

        The input is put back in the prompt, so confirming is a single Enter.
        """
        if not (self.playlist.dirty and len(self.playlist)) or self._pending_discard == text:
            self._pending_discard = None
            return True
        self._pending_discard = text
        self.query_one("#prompt", Input).value = text
        self.log_line(f"“{self.playlist.name}” has unsaved changes. Press Enter again to discard "
                      "it and continue, or clear the prompt and type `save` to keep it.",
                      "yellow")
        return False

    def run_command(self, cmd: Command) -> None:
        pl = self.playlist
        try:
            if cmd.action in ("name", "rename"):
                pl.rename(cmd.text or "")
                self.log_line(f"Playlist is now called “{pl.name}” (unsaved until `save`).")
            elif cmd.action == "move":
                pl.move(cmd.track_id, cmd.position)  # type: ignore[arg-type]
                self.log_line(f"Moved {cmd.track_id} to position {cmd.position}.")
            elif cmd.action == "remove":
                for track_id in cmd.track_ids:
                    try:
                        track = pl.remove(track_id)
                    except EditError as exc:
                        self.log_line(str(exc), "red")
                    else:
                        self.log_line(f"Removed {track.id}: {track.display}.")
            elif cmd.action == "add":
                position = cmd.position
                for track_id in cmd.track_ids:
                    try:
                        position = self._add_track(track_id, position)
                    except EditError as exc:
                        self.log_line(str(exc), "red")
            elif cmd.action == "keep":
                removed = pl.keep_marked()
                self.log_line(f"Kept {len(pl)} marked tracks, removed {len(removed)}.")
            elif cmd.action == "save":
                path = save_and_export(self.conn, pl, self.export_dir, self.export_extension)
                self.log_line(f"Saved “{pl.name}” ({len(pl)} tracks) → {path}", "green")
                self._warn_unavailable()
            elif cmd.action == "export":
                target = Path(cmd.text).expanduser() if cmd.text else self.export_dir
                try:
                    path = export_m3u(pl, target, self.export_extension)
                except OSError as exc:
                    raise ExportError(f"could not write {target}: {exc.strerror or exc}") from None
                self.log_line(f"Exported {len(pl)} tracks → {path}", "green")
                self._warn_unavailable()
            elif cmd.action == "again":
                self.start_again()
            elif cmd.action == "playlists":
                items = list_playlists(self.conn)
                if not items:
                    self.log_line("No saved playlists yet.")
                for p in items:
                    self.log_line(f"  {p.name}  ({p.track_count} tracks, {p.generation_status}, "
                                  f"updated {p.updated_at})")
            elif cmd.action == "open":
                self.playlist = load_playlist(self.conn, cmd.text or "")
                self.log_line(f"Opened “{self.playlist.name}” ({len(self.playlist)} tracks).")
                for v in self.playlist.violations():
                    self.log_line(f"Warning: {v}", "yellow")
            elif cmd.action == "new":
                self.playlist = Playlist()
            elif cmd.action == "refine":
                self.start_refinement()
            elif cmd.action == "help":
                self.log_line(HELP_TEXT)
        except (EditError, RepositoryError, ExportError) as exc:
            self.log_line(str(exc), "red")
        focus = cmd.track_id if cmd.action == "move" else (
            cmd.track_ids[0] if cmd.action == "add" else None)
        self.refresh_view(focus_track=focus)

    def _add_track(self, track_id: int, position: int | None) -> int | None:
        """Add one track; return the position for the next one (None: append)."""
        track = load_track(self.conn, track_id)
        if track is None:
            raise EditError(f"no track with ID {track_id} in the library")
        pos = self.playlist.add(track, position)
        self.log_line(f"Added {track.id} at position {pos}: {track.display}.")
        if not track.available:
            self.log_line(f"Track {track.id} is currently unavailable ({track.path}).", "yellow")
        return None if position is None else pos + 1

    def _warn_unavailable(self) -> None:
        n = sum(1 for t in self.playlist.tracks if not t.available)
        if n:
            self.log_line(f"{n} track(s) in the file are currently unavailable on disk.",
                          "yellow")

    # ------------------------------------------------------------------ background work

    def _client(self) -> LLMClient:
        if self._llm is None:
            self._llm = self._llm_factory()
        return self._llm

    def _set_busy(self, what: str | None) -> None:
        self.busy = what
        self._render_player()

    def start_generation(self, request: str) -> None:
        if self.busy:
            self.log_line(f"Please wait: {self.busy}.", "yellow")
            return
        self._set_busy("interpreting request")
        self._generate(request)

    @work(thread=True, exclusive=True, group="llm")
    def _generate(self, request: str) -> None:
        try:
            client = self._client()
            conn = connect(self.db_path)
            try:
                gen = generate(conn, request, client)
            finally:
                conn.close()
        except LLMError as exc:
            self.call_from_thread(self._background_failed, str(exc))
            return
        except Exception as exc:  # never let a worker crash the UI
            self.call_from_thread(self._background_failed, f"generation failed: {exc!r}")
            return
        self.call_from_thread(self._generation_done, gen)

    def _background_failed(self, message: str) -> None:
        self._set_busy(None)
        self.log_line(message, "red")

    def _generation_done(self, gen: Any) -> None:
        self._set_busy(None)
        if gen.error:
            self.log_line(f"Could not create a playlist: {gen.error}", "red")
            return
        result: BuildResult = gen.result
        self.playlist = Playlist.from_result(result, gen.request)
        for line in report_lines(result):
            style = "red" if line.startswith("UNMET") else (
                "yellow" if line.startswith("Warning") else "")
            self.log_line(line, style)
        self.refresh_view()
        self.query_one(PlaylistTable).move_cursor(row=0)

    def start_again(self) -> None:
        if not self.playlist.generated:
            self.log_line("This playlist was not generated from a request, so there is "
                          "nothing to run again.", "yellow")
            return
        if self.busy:
            self.log_line(f"Please wait: {self.busy}.", "yellow")
            return
        self._set_busy("building another version")
        self._again(self.playlist)

    @work(thread=True, exclusive=True, group="llm")
    def _again(self, playlist: Playlist) -> None:
        try:
            conn = connect(self.db_path)
            try:
                result = generate_again(conn, playlist)
            finally:
                conn.close()
        except Exception as exc:  # never let a worker crash the UI
            self.call_from_thread(self._background_failed, f"could not run again: {exc!r}")
            return
        self.call_from_thread(self._again_done, playlist, result)

    def _again_done(self, previous: Playlist, result: BuildResult) -> None:
        self._set_busy(None)
        if previous is not self.playlist:
            self.log_line("Discarded: the playlist was replaced meanwhile.", "yellow")
            return
        before = {t.id for t in previous.tracks}
        self.playlist = Playlist.from_result(result, previous.prompt)
        new = sum(1 for t in result.tracks if t.id not in before)
        self.log_line(f"Another version of the same request: {new} of {len(result.tracks)} "
                      f"tracks are new. `again` for more; the previous one is "
                      f"{'still saved as “' + previous.name + '”' if previous.id else 'discarded'}.")
        for line in report_lines(result):
            style = "red" if line.startswith("UNMET") else (
                "yellow" if line.startswith("Warning") else "")
            self.log_line(line, style)
        self.refresh_view()
        self.query_one(PlaylistTable).move_cursor(row=0)

    def start_refinement(self) -> None:
        if not self.playlist.generated or not len(self.playlist):
            self.log_line("Only generated playlists can be refined.", "yellow")
            return
        if self.busy:
            self.log_line(f"Please wait: {self.busy}.", "yellow")
            return
        self._set_busy("refining with the LLM")
        self._refine(self.playlist)

    @work(thread=True, exclusive=True, group="llm")
    def _refine(self, playlist: Playlist) -> None:
        try:
            client = self._client()
            conn = connect(self.db_path)
            try:
                spec = playlist.spec
                assert spec is not None
                pool = find_candidates(conn, spec)
            finally:
                conn.close()
            effective, defaults = apply_defaults(spec)
            current = BuildResult(spec=spec, effective_spec=effective, tracks=playlist.tracks,
                                  candidate_count=len(pool.tracks), defaults=defaults)
            outcome = refine_playlist(current, pool.tracks, client, playlist.prompt or "")
        except LLMError as exc:
            self.call_from_thread(self._background_failed, str(exc))
            return
        except Exception as exc:
            self.call_from_thread(self._background_failed, f"refinement failed: {exc!r}")
            return
        self.call_from_thread(self._refinement_done, playlist, outcome)

    def _refinement_done(self, playlist: Playlist, outcome: Any) -> None:
        self._set_busy(None)
        if playlist is not self.playlist:
            self.log_line("Refinement discarded: the playlist was replaced meanwhile.", "yellow")
            return
        self.log_line(f"Refinement {outcome.message}", "green" if outcome.accepted else "yellow")
        if outcome.accepted:
            added = {e.track.id: e.added_at for e in playlist.entries}
            now = now_iso()
            playlist.entries = [Entry(t, added.get(t.id, now)) for t in outcome.result.tracks]
            playlist.dirty = True
            self.refresh_view()


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _fit(text: str, width: int, style: str = "") -> Text:
    out = Text(text, style=style, no_wrap=True, overflow="ellipsis")
    out.truncate(width, overflow="ellipsis")
    return out
