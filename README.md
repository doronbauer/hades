# hades

Local music-library indexer with independent BPM, key and loudness analysis, and a
terminal playlist creator driven by natural-language requests.

## Setup

```sh
uv sync
cp config.example.toml config.toml   # then edit music_directories
brew install mpv                     # optional: audio previews in `hades ui`
```

Library commands: `hades scan`, `analyze`, `reconcile`, `stats`, `inspect`, `search`,
`list`, `conflicts`, `export`. Run `hades --help` for details.

## Playlists

```sh
hades ui                                   # interactive terminal app
hades generate "a one-hour drum & bass set below 160 BPM, added after March 2026"
hades generate "15 techno tracks, 125-135 BPM, rising tempo" --refine --save "Peak"
hades regenerate "Peak" [--save "Peak 2"]  # another version of the same request, no LLM call
hades playlists [NAME]                     # list saved playlists, or show one
hades playlists "Peak" --export [PATH]     # write it as a .m3u playlist file
```

Requests are interpreted by an LLM configured under `[llm]` in `config.toml`:
`provider = "openai"` (default model `gpt-6.1-terra`) or `"anthropic"` (default
`claude-opus-5-5`). The API key comes from `llm.api_key` in `config.toml` (which is
git-ignored) or else from `$OPENAI_API_KEY` / `$ANTHROPIC_API_KEY` (or the variable
named by `llm.api_key_env`).
Only the request text and, for `refine`, a shortlist of track IDs, artists, titles,
BPM, keys, genres and durations are sent — never file paths or audio.

### How a request becomes a playlist

1. **Interpretation** (`llm.py`, `spec.py`): the LLM returns JSON constrained by a
   schema, which is validated again as a `PlaylistSpec`. Contradictory requests
   (e.g. min BPM above max) are reported, never silently changed.
   *Hard constraints*: genre, filesystem-creation date, BPM limits, folder,
   artists, track count, total duration. *Soft preferences*: BPM progression,
   harmonic compatibility.
2. **Retrieval** (`query.py`): parameterized SQLite filters on the effective
   (reconciled) values. A track with missing data never satisfies a constraint
   on that field; BPMs where tags and analysis CONFLICT (and no manual value
   exists) count as unknown under a BPM limit. Missing files are excluded. The
   number of tracks excluded for each reason is reported.
3. **Building** (`builder.py`): a deterministic greedy sequencer, run from several
   starting tracks, picks each next track by transition score while keeping
   count/duration satisfiable. Tempo follows a planned ramp for ascending or
   descending requests; Camelot-wheel neighbours score high; missing or
   unreliable keys get a neutral score; duplicate songs are skipped. The result
   is validated against every hard constraint.
   *Variety*: multi-artist credits are split into names ("Clark, Marcel Dettmann"
   counts as both), and two neighbouring tracks never share an artist or an album
   unless nothing else fits (e.g. a single-artist request). Repeating an artist
   elsewhere in the set is allowed but penalized; ask for "one track per artist"
   or "at most 2 per artist" to make it a hard limit.
4. **Refinement** (optional, `refine` / `--refine`): the LLM may reorder or swap
   tracks from a bounded shortlist. The proposal is accepted only if it passes
   all hard constraints and scores at least as well as the current order.
5. **Report**: track count, total duration, BPM range, eligible candidates,
   unmet constraints (`UNMET`) and warnings.

Interpretation rules worth knowing:

* Dates are calendar-based and exclusive: "after March 2026" means from 1 April 2026.
  They always refer to the file's creation date on disk.
* "About an hour" is a 60-minute target with ±5 minutes; "no longer than an hour"
  is a strict 60-minute maximum.
* With neither a count nor a duration, a 60 ± 5 minute length is assumed and
  reported as a default (falling short of it is a warning, not a failure).

### Terminal app

| Key | Action |
|---|---|
| ↑ / ↓ | select previous/next track and play it (playlist focused) |
| Space | pause/resume |
| M | toggle preview mode START (from 0:00) / MIDDLE (from 50%) for the next selection |
| Tab | switch focus between playlist and prompt |
| Enter | submit the prompt |
| Ctrl+Q | quit |

Commands at the prompt (anything else is a natural-language request):

| Command | Action |
|---|---|
| `name My Playlist` / `rename New Name` | set the name |
| `move 876 to 4` | move track ID 876 to position 4 |
| `remove 876` | remove track ID 876 |
| `add 1042` / `add 1042 at 3` | append / insert a track by ID |
| `save` | save (requires a name) and write `<export_directory>/<name>.m3u` |
| `export` / `export ~/Desktop/set.m3u` | write the current playlist file (default: export_directory) |
| `again` | another version of the same request (no LLM call), preferring other tracks |
| `playlists` | list saved playlists |
| `open My Playlist` | load a saved playlist |
| `new` | start an empty playlist |
| `refine` | ask the LLM to improve the current generated playlist |
| `help` | show commands |

Track IDs are the library's IDs (see `hades search`); positions start at 1.
Manual edits may break the original request; the Notes column and the status
line show what no longer holds. Replacing a playlist that has unsaved changes (with a
new request, `again`, `open` or `new`) asks first: press Enter a second time to
discard it.

The columns show position, library ID, artist and title, BPM and key (`?` marks
a value where tags and analysis disagree), length, the date the file was added
to disk, and notes (transition problems and constraint violations).

`again` rebuilds from the stored request with some randomness, steering away
from the current tracks; it picks the smoothest of several variants that replace
at least half of them. The result is a new, unsaved playlist; a saved one is kept.

Saved playlists are written as extended M3U files (`.m3u`, UTF-8, absolute
paths to your files), which Traktor 3, VLC, Music, Rekordbox and most players
open. Set `playlist_format = "m3u8"` for players that prefer that extension.
Renaming and saving replaces the old file. Set `export_directory` in
`config.toml` (default `./exports`, which git ignores).

Playlists are stored in the library database (`playlists`, `playlist_tracks`,
added by schema migration 2). They reference tracks by ID and keep the
original request and its parameters; they are not regenerated when library
metadata changes.

## Tests

```sh
uv run pytest
```

The LLM and mpv are mocked (a fake mpv speaks the real JSON IPC protocol).
