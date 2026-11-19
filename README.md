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
hades playlists [NAME]                     # list saved playlists, or show one
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
   unreliable keys get a neutral score; artists are varied and duplicate songs
   skipped. The result is validated against every hard constraint.
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
| S | stop |
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
| `save` | save (requires a name) |
| `playlists` | list saved playlists |
| `open My Playlist` | load a saved playlist |
| `new` | start an empty playlist |
| `refine` | ask the LLM to improve the current generated playlist |
| `help` | show commands |

Track IDs are the library's IDs (see `hades search`); positions start at 1.
Manual edits may break the original request; the Notes column and the status
line show what no longer holds. Replacing a playlist that has unsaved changes
requires submitting the same input twice.

Playlists are stored in the library database (`playlists`, `playlist_tracks`,
added by schema migration 2). They reference tracks by ID and keep the
original request and its parameters; they are not regenerated when library
metadata changes.

## Tests

```sh
uv run pytest
```

The LLM and mpv are mocked (a fake mpv speaks the real JSON IPC protocol).
