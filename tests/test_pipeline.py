"""Integration tests for scan/analyze against a real SQLite database (fake analyzer)."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest

from conftest import FakeAnalyzer, write_wav
from hades.config import Config
from hades.db import connect
from hades.pipeline import analyze, scan


def real(path: Path) -> str:
    return os.path.realpath(path)


def track(conn, path: Path):
    return conn.execute("SELECT * FROM tracks WHERE path = ?", (real(path),)).fetchone()


def bump_mtime(path: Path) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


def test_first_scan_indexes_everything(conn, config: Config, music_dir: Path) -> None:
    write_wav(music_dir / "a.wav")
    write_wav(music_dir / "deep" / "nested" / "b.wav")
    (music_dir / "readme.txt").write_text("hi")

    s = scan(conn, config)

    assert (s.discovered, s.new, s.extracted, s.failed) == (2, 2, 2, 0)
    t = track(conn, music_dir / "deep" / "nested" / "b.wav")
    assert t["status"] == "available"
    assert t["format"] == "WAV"
    assert t["filename"] == "b.wav"
    assert t["sample_rate"] == 44100 and t["channels"] == 2
    assert t["duration"] == pytest.approx(8.0, abs=0.01)
    assert t["metadata_signature"] == t["content_signature"]
    tags = conn.execute("SELECT * FROM track_tags WHERE track_id = ?", (t["id"],)).fetchone()
    assert tags["title"] is None and tags["artist"] is None  # never derived from filenames


def test_rescan_skips_unchanged_and_processes_changed(conn, config: Config, music_dir: Path) -> None:
    a, b = write_wav(music_dir / "a.wav"), write_wav(music_dir / "b.wav")
    scan(conn, config)

    s = scan(conn, config)
    assert (s.new, s.changed, s.skipped, s.extracted) == (0, 0, 2, 0)

    bump_mtime(b)
    write_wav(music_dir / "c.wav")
    s = scan(conn, config)
    assert (s.new, s.changed, s.skipped, s.extracted) == (1, 1, 1, 2)
    assert track(conn, a)["id"] < track(conn, b)["id"]  # ids are stable


def test_analysis_runs_once_and_again_after_change(conn, config: Config, music_dir: Path) -> None:
    a = write_wav(music_dir / "a.wav")
    write_wav(music_dir / "b.wav")
    scan(conn, config)
    analyzer = FakeAnalyzer()

    assert analyze(conn, config, analyzer).analyzed == 2
    assert analyze(conn, config, analyzer).analyzed == 0
    assert len(analyzer.calls) == 2

    bump_mtime(a)
    scan(conn, config)
    s = analyze(conn, config, analyzer)
    assert s.analyzed == 1
    assert analyzer.calls[-1] == real(a)


def test_new_analyzer_version_makes_tracks_pending(conn, config: Config, music_dir: Path) -> None:
    write_wav(music_dir / "a.wav")
    scan(conn, config)
    analyze(conn, config, FakeAnalyzer(version="v1"))
    assert analyze(conn, config, FakeAnalyzer(version="v2")).analyzed == 1
    assert conn.execute("SELECT COUNT(*) FROM audio_analysis").fetchone()[0] == 2
    assert analyze(conn, config, FakeAnalyzer(version="v2"), force=True).analyzed == 1


def test_limit(conn, config: Config, music_dir: Path) -> None:
    for i in range(5):
        write_wav(music_dir / f"{i}.wav")
    scan(conn, config)
    analyzer = FakeAnalyzer()
    assert analyze(conn, config, analyzer, limit=2).analyzed == 2
    assert analyze(conn, config, analyzer, limit=10).analyzed == 3


def test_reconciliation_after_analysis(conn, config: Config, music_dir: Path) -> None:
    a = write_wav(music_dir / "a.wav")
    scan(conn, config)
    analyze(conn, config, FakeAnalyzer(bpm=174.0))
    t = track(conn, a)
    # No embedded BPM/key in a bare WAV: analysis values become effective.
    assert (t["effective_bpm"], t["effective_bpm_source"], t["bpm_flag"]) == (174.0, "analysis", None)
    assert (t["effective_key"], t["effective_key_camelot"]) == ("Am", "8A")

    # Simulate tags as an embedded 87 BPM / 8A would be stored, then reconcile.
    from hades.pipeline import reconcile_track

    with conn:
        conn.execute("UPDATE track_tags SET bpm_raw = '87', bpm = 87, key_raw = '8A' WHERE track_id = ?",
                     (t["id"],))
        reconcile_track(conn, t["id"])
    t = track(conn, a)
    assert (t["effective_bpm"], t["effective_bpm_source"], t["bpm_flag"]) == (87.0, "tag", "HALF_DOUBLE")
    assert t["key_flag"] == "CONSISTENT"


def test_corrupt_files_do_not_stop_processing(conn, config: Config, music_dir: Path) -> None:
    write_wav(music_dir / "good.wav")
    (music_dir / "corrupt.wav").write_bytes(b"RIFF\x00\x00garbage")
    (music_dir / "corrupt.mp3").write_bytes(b"\x00" * 64)

    s = scan(conn, config)
    assert s.discovered == 3
    assert s.extracted + s.failed == 3
    assert track(conn, music_dir / "good.wav")["status"] == "available"

    a = analyze(conn, config, FakeAnalyzer())
    assert (a.analyzed, a.failed) == (1, 2)
    errors = conn.execute("SELECT error FROM audio_analysis WHERE status = 'error'").fetchall()
    assert all(e["error"] for e in errors)

    # Failures are not retried unless asked.
    analyzer = FakeAnalyzer()
    assert analyze(conn, config, analyzer).pending == 0
    assert analyze(conn, config, analyzer, retry_failed=True).pending == 2


def test_metadata_errors_are_recorded_and_retryable(conn, config: Config, music_dir: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    import hades.pipeline as pipeline
    from hades.metadata import MetadataError

    write_wav(music_dir / "a.wav")
    write_wav(music_dir / "b.wav")

    real_extract = pipeline.extract_metadata

    def flaky(path: str):
        if path.endswith("a.wav"):
            raise MetadataError("boom")
        return real_extract(path)

    monkeypatch.setattr(pipeline, "extract_metadata", flaky)
    s = scan(conn, config)
    assert (s.extracted, s.failed) == (1, 1)
    t = track(conn, music_dir / "a.wav")
    assert (t["status"], t["last_error"]) == ("error", "boom")

    assert scan(conn, config).failed == 0  # unchanged failure is not retried every scan

    monkeypatch.setattr(pipeline, "extract_metadata", real_extract)
    s = scan(conn, config, retry_failed=True)
    assert (s.extracted, s.failed) == (1, 0)
    assert track(conn, music_dir / "a.wav")["status"] == "available"


def test_deleted_file_marked_missing_and_restored(conn, config: Config, music_dir: Path) -> None:
    a = write_wav(music_dir / "a.wav")
    write_wav(music_dir / "b.wav")
    scan(conn, config)
    analyze(conn, config, FakeAnalyzer())

    moved = music_dir.parent / "a.wav.bak"
    shutil.move(a, moved)
    s = scan(conn, config)
    assert s.missing == 1
    t = track(conn, a)
    assert t["status"] == "missing"
    assert conn.execute("SELECT COUNT(*) FROM audio_analysis WHERE track_id = ?",
                        (t["id"],)).fetchone()[0] == 1  # history kept, row not deleted
    assert analyze(conn, config, FakeAnalyzer()).pending == 0  # missing tracks not analyzed

    shutil.move(moved, a)
    s = scan(conn, config)
    assert (s.skipped, s.extracted) == (2, 0)
    assert track(conn, a)["status"] == "available"


def test_unmounted_root_leaves_tracks_untouched(conn, config: Config, music_dir: Path) -> None:
    write_wav(music_dir / "a.wav")
    scan(conn, config)

    unplugged = music_dir.parent / "unplugged"
    music_dir.rename(unplugged)
    s = scan(conn, config)
    assert s.missing == 0
    assert len(s.unavailable_roots) == 1
    assert conn.execute("SELECT status FROM tracks").fetchone()[0] == "available"
    unplugged.rename(music_dir)


def test_empty_root_is_not_treated_as_mass_deletion(conn, config: Config, music_dir: Path) -> None:
    a = write_wav(music_dir / "a.wav")
    scan(conn, config)
    a.unlink()
    s = scan(conn, config)
    assert s.missing == 0
    assert s.warnings
    assert track(conn, a)["status"] == "available"


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs non-root POSIX")
def test_unreadable_subdirectory_is_not_marked_missing(conn, config: Config, music_dir: Path) -> None:
    write_wav(music_dir / "ok.wav")
    locked_file = write_wav(music_dir / "locked" / "x.wav")
    scan(conn, config)
    locked = music_dir / "locked"
    locked.chmod(0)
    try:
        s = scan(conn, config)
    finally:
        locked.chmod(0o755)
    assert s.missing == 0
    assert track(conn, locked_file)["status"] == "available"


def test_overlapping_roots_index_each_file_once(make_config, tmp_path: Path, music_dir: Path) -> None:
    write_wav(music_dir / "sub" / "a.wav")
    write_wav(music_dir / "b.wav")
    cfg = make_config(music_dir, music_dir / "sub", music_dir)
    c = connect(cfg.database_path)
    s = scan(c, cfg)
    assert s.discovered == 2
    assert c.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 2
    assert scan(c, cfg).missing == 0
    c.close()


def test_database_survives_restart(config: Config, music_dir: Path) -> None:
    write_wav(music_dir / "a.wav")
    c = connect(config.database_path)
    scan(c, config)
    analyze(c, config, FakeAnalyzer())
    c.close()

    c = connect(config.database_path)
    assert scan(c, config).skipped == 1
    assert analyze(c, config, FakeAnalyzer()).pending == 0
    runs = c.execute("SELECT kind, status FROM scan_runs ORDER BY id").fetchall()
    assert [tuple(r) for r in runs] == [("scan", "completed"), ("analyze", "completed")] * 2
    c.close()


def test_interrupted_analysis_resumes(conn, config: Config, music_dir: Path) -> None:
    for i in range(3):
        write_wav(music_dir / f"{i}.wav")
    scan(conn, config)

    class Interrupting(FakeAnalyzer):
        def analyze(self, path: str):
            if len(self.calls) == 1:
                raise KeyboardInterrupt
            return super().analyze(path)

    with pytest.raises(KeyboardInterrupt):
        analyze(conn, config, Interrupting())
    status = conn.execute("SELECT status FROM scan_runs ORDER BY id DESC").fetchone()[0]
    assert status == "interrupted"
    assert analyze(conn, config, FakeAnalyzer()).analyzed == 2


def test_original_files_are_never_modified(conn, config: Config, music_dir: Path) -> None:
    files = [write_wav(music_dir / "a.wav"), write_wav(music_dir / "sub" / "b.wav")]
    (music_dir / "corrupt.mp3").write_bytes(b"\x00" * 64)
    files.append(music_dir / "corrupt.mp3")

    def snapshot():
        return {f: (hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_mtime_ns) for f in files}

    before = snapshot()
    listing = sorted(p.name for p in music_dir.rglob("*"))
    scan(conn, config)
    analyze(conn, config, FakeAnalyzer())
    scan(conn, config, retry_failed=True)
    assert snapshot() == before
    assert sorted(p.name for p in music_dir.rglob("*")) == listing


def test_tags_from_older_extractor_are_reread(conn, config: Config, music_dir: Path) -> None:
    write_wav(music_dir / "a.wav")
    write_wav(music_dir / "b.wav")
    scan(conn, config)
    assert scan(conn, config).extracted == 0
    with conn:
        conn.execute("UPDATE track_tags SET extractor = 'tinytag 2.3.2' WHERE track_id = 1")
    s = scan(conn, config)
    assert (s.skipped, s.extracted) == (2, 1)
    assert scan(conn, config).extracted == 0
