"""Read source files into normalized text and split off front matter.

Normalized text is what every offset in the pipeline refers to: UTF-8 with
any BOM removed and all line endings converted to LF. Normalizing makes
offsets and hashes identical across checkouts, whatever git's autocrlf
setting did to the working copy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?\n)?---[ \t]*(?:\n|\Z)", re.S)


def normalize(raw: bytes) -> str:
    text = raw.decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n")


@dataclass(frozen=True)
class SourceFile:
    path: str  # repo-relative, POSIX separators
    text: str  # normalized
    body_start: int  # offset of the first character after front matter
    front_matter: dict[str, Any] = field(default_factory=dict)


def split_front_matter(text: str) -> tuple[dict[str, Any], int]:
    m = FRONT_MATTER.match(text)
    if not m:
        return {}, 0
    try:
        data = yaml.safe_load(m.group(1) or "") or {}
    except yaml.YAMLError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {str(k): _clean(v) for k, v in data.items()}, m.end()


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


class Loader(Protocol):
    """One loader per format (DESIGN 5.1)."""

    def load(self, repo_root: Path, path: str) -> SourceFile: ...


class MarkdownLoader:
    """Loads Markdown with YAML front matter. Caches files by path."""

    def __init__(self) -> None:
        self._cache: dict[str, SourceFile] = {}

    def load(self, repo_root: Path, path: str) -> SourceFile:
        if path not in self._cache:
            text = normalize((repo_root / path).read_bytes())
            fm, body = split_front_matter(text)
            self._cache[path] = SourceFile(path=path, text=text, body_start=body, front_matter=fm)
        return self._cache[path]
