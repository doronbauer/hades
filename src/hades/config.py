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
DEFAULT_EXPORT_DIRECTORY = "./exports"
PLAYLIST_FORMATS = ("m3u", "m3u8")

BPM_METHODS = ("multifeature", "degara")
LLM_PROVIDERS = ("anthropic", "openai")
# provider -> (default model, default API key environment variable)
LLM_DEFAULTS = {
    "anthropic": ("claude-opus-5-5", "ANTHROPIC_API_KEY"),
    "openai": ("gpt-6.1-terra", "OPENAI_API_KEY"),
}
LLM_EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ConfigError(Exception):
    """The configuration file is missing or invalid."""


@dataclass(frozen=True)
class AnalysisSettings:
    sample_rate: int = 44100
    bpm_method: str = "multifeature"
    key_profile: str = "edma"
    max_duration_seconds: float = 1800.0


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "anthropic"
    model: str = "claude-opus-5-5"
    api_key_env: str = "ANTHROPIC_API_KEY"  # unset: the SDK's own credential lookup applies
    effort: str = "low"                      # request interpretation is a small extraction task
    timeout_seconds: float = 60.0
    api_key: str | None = field(default=None, repr=False)  # from config.toml; wins over the env var

    @property
    def resolved_api_key(self) -> str | None:
        return self.api_key or os.environ.get(self.api_key_env) or None


@dataclass(frozen=True)
class PlayerSettings:
    mpv_path: str = "mpv"


@dataclass(frozen=True)
class Config:
    path: Path
    music_directories: tuple[Path, ...]
    database_path: Path
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    export_directory: Path = Path(DEFAULT_EXPORT_DIRECTORY)  # where saved playlists are written
    playlist_format: str = "m3u"                             # m3u (Traktor) | m3u8

    @property
    def export_extension(self) -> str:
        return "." + self.playlist_format
    llm: LLMSettings = field(default_factory=LLMSettings)
    player: PlayerSettings = field(default_factory=PlayerSettings)


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

    export_dir = data.get("export_directory", DEFAULT_EXPORT_DIRECTORY)
    if not isinstance(export_dir, str) or not export_dir.strip():
        raise ConfigError("export_directory must be a non-empty string")
    playlist_format = data.get("playlist_format", "m3u")
    if playlist_format not in PLAYLIST_FORMATS:
        raise ConfigError(f"playlist_format must be one of {', '.join(PLAYLIST_FORMATS)}")

    return Config(
        path=path,
        export_directory=_resolve_path(export_dir, base),
        playlist_format=playlist_format,
        music_directories=tuple(_resolve_path(d, base) for d in dirs),
        database_path=_resolve_path(database, base),
        analysis=_load_analysis(data.get("analysis", {})),
        llm=_load_llm(data.get("llm", {})),
        player=_load_player(data.get("player", {})),
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


def _load_llm(section: Any) -> LLMSettings:
    if not isinstance(section, dict):
        raise ConfigError("[llm] must be a table")
    defaults = LLMSettings()
    provider = section.get("provider", defaults.provider)
    if provider not in LLM_PROVIDERS:
        raise ConfigError(f"llm.provider must be one of {', '.join(LLM_PROVIDERS)}")
    default_model, default_env = LLM_DEFAULTS[provider]
    model = section.get("model", default_model)
    api_key_env = section.get("api_key_env", default_env)
    api_key = section.get("api_key")
    effort = section.get("effort", defaults.effort)
    timeout = section.get("timeout_seconds", defaults.timeout_seconds)

    if not isinstance(model, str) or not model.strip():
        raise ConfigError("llm.model must be a non-empty string")
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise ConfigError("llm.api_key_env must be the name of an environment variable")
    if effort not in LLM_EFFORTS:
        raise ConfigError(f"llm.effort must be one of {', '.join(LLM_EFFORTS)}")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ConfigError("llm.timeout_seconds must be a positive number")
    if api_key is not None and not isinstance(api_key, str):
        raise ConfigError("llm.api_key must be a string")

    return LLMSettings(provider=provider, model=model, api_key_env=api_key_env,
                       effort=effort, timeout_seconds=float(timeout),
                       api_key=api_key.strip() if api_key and api_key.strip() else None)


def _load_player(section: Any) -> PlayerSettings:
    if not isinstance(section, dict):
        raise ConfigError("[player] must be a table")
    mpv_path = section.get("mpv_path", PlayerSettings().mpv_path)
    if not isinstance(mpv_path, str) or not mpv_path.strip():
        raise ConfigError("player.mpv_path must be a non-empty string")
    return PlayerSettings(mpv_path=mpv_path)
