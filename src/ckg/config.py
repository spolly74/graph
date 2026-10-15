"""Load and validate config.yaml.

Every setting comes from the config file (DESIGN section 9). The models
below have no defaults for settings, so a missing key is an error rather
than a silent fallback. Relative paths resolve against the config file's
directory, not the current working directory.
"""

from __future__ import annotations

import os
from fnmatch import fnmatch
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CONFIG_ENV_VAR = "CKG_CONFIG"
CONFIG_FILENAME = "config.yaml"


class ConfigError(Exception):
    """Raised when the config file is missing or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CorpusConfig(_Strict):
    repo_root: Path
    base_dir: Path
    exclude: list[str]
    external_include_prefixes: list[str]
    include: list[str] | None = Field(
        default=None,
        description="Explicit file list relative to base_dir. Absent means the full scope.",
    )


class PathsConfig(_Strict):
    out_dir: Path
    review_dir: Path
    state_db: Path


class Config(_Strict):
    corpus: CorpusConfig
    paths: PathsConfig
    config_dir: Path = Field(exclude=True)

    def resolve(self, path: Path) -> Path:
        """Resolve a config path relative to the config file's directory."""
        return path if path.is_absolute() else (self.config_dir / path).resolve()

    @property
    def corpus_dir(self) -> Path:
        return self.resolve(self.corpus.repo_root) / self.corpus.base_dir

    def corpus_files(self, base: Path | None = None) -> list[Path]:
        """List in-scope Markdown files, sorted, relative to the corpus base."""
        base = base or self.corpus_dir
        if not base.is_dir():
            raise ConfigError(f"corpus directory not found: {base}")
        if self.corpus.include is not None:
            candidates = [Path(p) for p in self.corpus.include]
            missing = [p for p in candidates if not (base / p).is_file()]
            if missing:
                raise ConfigError(f"{len(missing)} included files not found, first: {missing[0]}")
        else:
            candidates = [p.relative_to(base) for p in base.rglob("*.md")]
        kept = [p for p in candidates if not any(fnmatch(p.as_posix(), pat) for pat in self.corpus.exclude)]
        return sorted(kept, key=lambda p: p.as_posix())


def find_config(explicit: Path | None = None) -> Path:
    """Find the config file: explicit path, then $CKG_CONFIG, then ./config.yaml."""
    if explicit is not None:
        path = explicit
    elif os.environ.get(CONFIG_ENV_VAR):
        path = Path(os.environ[CONFIG_ENV_VAR])
    else:
        path = Path.cwd() / CONFIG_FILENAME
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    return path


def load_config(path: Path | None = None) -> Config:
    path = find_config(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    try:
        return Config(**raw, config_dir=path.parent.resolve())
    except ValidationError as e:
        raise ConfigError(f"{path}: {e}") from e
