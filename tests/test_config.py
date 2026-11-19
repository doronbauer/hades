from pathlib import Path

import pytest

from hades.config import ConfigError, load_config, resolve_config_path


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_paths_are_expanded_and_resolved(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, 'music_directories = ["~/Music/DJ", "rel/dir"]\n'
                                      'database_path = "./data/lib.sqlite3"\n'))
    assert cfg.music_directories[0] == Path.home() / "Music" / "DJ"
    assert cfg.music_directories[1] == tmp_path.resolve() / "rel" / "dir"
    assert cfg.database_path == tmp_path.resolve() / "data" / "lib.sqlite3"
    assert cfg.analysis.sample_rate == 44100


def test_example_config_is_valid() -> None:
    cfg = load_config(Path(__file__).parent.parent / "config.example.toml")
    assert len(cfg.music_directories) == 2
    assert cfg.analysis.key_profile == "edma"


@pytest.mark.parametrize(
    "text",
    ["", "music_directories = []", 'music_directories = "x"', "music_directories = [1]",
     'music_directories = ["x"]\n[analysis]\nbpm_method = "magic"',
     'music_directories = ["x"]\n[analysis]\nsample_rate = -1', "not toml ="],
)
def test_invalid_configs(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_config_path_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HADES_CONFIG", raising=False)
    assert resolve_config_path(None) == Path("config.toml")
    monkeypatch.setenv("HADES_CONFIG", "/x/y.toml")
    assert resolve_config_path(None) == Path("/x/y.toml")
    assert resolve_config_path("a.toml") == Path("a.toml")
