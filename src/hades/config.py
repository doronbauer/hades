"""Load and validate the TOML configuration file."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_ENV_VAR = "HADES_CONFIG"
DEFAULT_CONFIG_PATH = Path("config.toml")
DEFAULT_DATABASE_PATH = "./data/library.sqlite3"

BPM_METHODS = ("multifeature", "degara")


class ConfigError(Exception):
    """The configuration file is missing or invalid."""


@dataclass(frozen=True)
class AnalysisSettings:
    sample_rate: int = 44100
    bpm_method: str = "multifeature"
    key_profile: str = "edma"
    max_duration_seconds: float = 1800.0


@dataclass(frozen=True)
class Config:
    path: Path
    music_directories: tuple[Path, ...]
    database_path: Path
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)


def resolve_config_path(explicit: str | None) -> Path:
    """Pick the config file: explicit argument, then $HADES_CONFIG, then ./config.toml."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get(CONFIG_ENV_VAR)
    return Path(env).expanduser() if env else DEFAULT_CONFIG_PATH


def load_config(path: Path) -> Config:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(
            f"config file not found: {path} (copy config.example.toml to config.toml)"
        ) from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from None

    base = path.resolve().parent

    dirs = data.get("music_directories")
    if (
        not isinstance(dirs, list)
        or not dirs
        or not all(isinstance(d, str) and d.strip() for d in dirs)
    ):
        raise ConfigError("music_directories must be a non-empty list of directory paths")

    database = data.get("database_path", DEFAULT_DATABASE_PATH)
    if not isinstance(database, str) or not database.strip():
        raise ConfigError("database_path must be a non-empty string")

    return Config(
        path=path,
        music_directories=tuple(_resolve_path(d, base) for d in dirs),
        database_path=_resolve_path(database, base),
        analysis=_load_analysis(data.get("analysis", {})),
    )


def _resolve_path(value: str, base: Path) -> Path:
    # normpath only: realpath would touch the filesystem, and external drives may be absent.
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base / p
    return Path(os.path.normpath(p))


def _load_analysis(section: Any) -> AnalysisSettings:
    if not isinstance(section, dict):
        raise ConfigError("[analysis] must be a table")
    defaults = AnalysisSettings()
    sample_rate = section.get("sample_rate", defaults.sample_rate)
    bpm_method = section.get("bpm_method", defaults.bpm_method)
    key_profile = section.get("key_profile", defaults.key_profile)
    max_duration = section.get("max_duration_seconds", defaults.max_duration_seconds)

    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ConfigError("analysis.sample_rate must be a positive integer")
    if bpm_method not in BPM_METHODS:
        raise ConfigError(f"analysis.bpm_method must be one of {', '.join(BPM_METHODS)}")
    if not isinstance(key_profile, str) or not key_profile:
        raise ConfigError("analysis.key_profile must be a string")
    if not isinstance(max_duration, (int, float)) or max_duration <= 0:
        raise ConfigError("analysis.max_duration_seconds must be a positive number")

    return AnalysisSettings(
        sample_rate=sample_rate,
        bpm_method=bpm_method,
        key_profile=key_profile,
        max_duration_seconds=float(max_duration),
    )
