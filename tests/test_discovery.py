import os
from pathlib import Path

from hades.discovery import check_root, is_under, iter_audio_files


def touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def names(root: Path) -> list[str]:
    unreadable: list[str] = []
    return [os.path.relpath(f.path, root) for f in iter_audio_files(str(root), unreadable)]


def test_finds_supported_formats_recursively(tmp_path: Path) -> None:
    for rel in ["a.mp3", "b/c.AIFF", "b/d/e.aif", "b/d/f.wav", "g.flac", "h.m4a"]:
        touch(tmp_path / rel)
    assert sorted(names(tmp_path)) == sorted(
        ["a.mp3", "b/c.AIFF", "b/d/e.aif", "b/d/f.wav", "g.flac", "h.m4a"])


def test_ignores_unsupported_hidden_and_resource_fork_files(tmp_path: Path) -> None:
    for rel in ["notes.txt", "cover.jpg", "x.ogg", "._a.mp3", ".hidden.mp3",
                ".Trashes/t.mp3", "real.mp3"]:
        touch(tmp_path / rel)
    assert names(tmp_path) == ["real.mp3"]


def test_does_not_follow_directory_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    touch(outside / "elsewhere.mp3")
    touch(root / "here.mp3")
    (root / "link").symlink_to(outside, target_is_directory=True)
    assert names(root) == ["here.mp3"]


def test_file_metadata(tmp_path: Path) -> None:
    touch(tmp_path / "a.flac", b"12345")
    (f,) = iter_audio_files(str(tmp_path), [])
    assert (f.format, f.size) == ("FLAC", 5)
    assert f.signature == f"5:{f.mtime_ns}"


def test_check_root(tmp_path: Path) -> None:
    assert check_root(str(tmp_path)) is None
    assert "does not exist" in check_root(str(tmp_path / "gone"))
    touch(tmp_path / "file.mp3")
    assert "not a directory" in check_root(str(tmp_path / "file.mp3"))
    assert check_root("/Volumes/hades-test-not-mounted/Tracks") is not None


def test_is_under() -> None:
    assert is_under("/a/b/c.mp3", ["/a/b"])
    assert is_under("/a/b", ["/a/b/"])
    assert not is_under("/a/bc/d.mp3", ["/a/b"])
