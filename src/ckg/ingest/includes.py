"""Resolve [!INCLUDE] directives into one text with a source map.

The resolved text is a list of pieces. Each piece is either a run of text
copied from a source file (with the offset it came from) or a placeholder
for an include that could not be resolved. Pieces let any span of the
resolved text, and any chunk cut from it, be traced back to the exact file
and offsets it came from (DESIGN 5.1).
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

from ckg.ingest.loader import Loader

INCLUDE = re.compile(r"\[!INCLUDE\s*\[[^\]]*\]\(\s*<?([^)\s>]+)>?\s*\)\]", re.I)


@dataclass(frozen=True)
class Piece:
    start: int
    end: int
    source: str | None  # repo-relative file, None for a placeholder
    source_start: int | None
    placeholder: str | None = None  # include target, set for placeholders

    def to_json(self) -> dict:
        d = {"start": self.start, "end": self.end}
        if self.placeholder is not None:
            d["placeholder"] = self.placeholder
        else:
            d["source"] = self.source
            d["source_start"] = self.source_start
        return d


@dataclass(frozen=True)
class Unresolved:
    article: str
    source: str  # file containing the directive (the article or an include)
    line: int
    target: str
    reason: str  # external, not_found, outside_repo, absolute, cycle, max_depth


@dataclass(frozen=True)
class Resolved:
    text: str
    pieces: tuple[Piece, ...]
    includes: tuple[str, ...]  # resolved include files, sorted
    unresolved: tuple[Unresolved, ...]


class IncludeResolver:
    def __init__(
        self,
        repo_root: Path,
        loader: Loader,
        external_prefixes: list[str],
        placeholder_format: str,
        max_depth: int,
    ) -> None:
        self.repo_root = repo_root
        self.loader = loader
        self.external_prefixes = tuple(external_prefixes)
        self.placeholder_format = placeholder_format
        self.max_depth = max_depth

    def resolve(self, article: str) -> Resolved:
        includes: set[str] = set()
        unresolved: list[Unresolved] = []
        segments = self._expand(article, article, (article,), includes, unresolved)

        pieces: list[Piece] = []
        parts: list[str] = []
        pos = 0
        for text, source, source_start, placeholder in segments:
            if not text:
                continue
            prev = pieces[-1] if pieces else None
            if (
                prev is not None
                and source is not None
                and prev.source == source
                and prev.source_start is not None
                and prev.source_start + (prev.end - prev.start) == source_start
            ):
                pieces[-1] = Piece(prev.start, pos + len(text), source, prev.source_start)
            else:
                pieces.append(Piece(pos, pos + len(text), source, source_start, placeholder))
            parts.append(text)
            pos += len(text)
        return Resolved("".join(parts), tuple(pieces), tuple(sorted(includes)), tuple(unresolved))

    # (text, source, source_start, placeholder)
    Segment = tuple[str, "str | None", "int | None", "str | None"]

    def _expand(self, article, path, stack, includes, unresolved) -> list[Segment]:
        src = self.loader.load(self.repo_root, path)
        text, pos = src.text, src.body_start
        out: list[IncludeResolver.Segment] = []
        for m in INCLUDE.finditer(text, pos):
            out.append((text[pos : m.start()], path, pos, None))
            target = m.group(1)
            line = text.count("\n", 0, m.start()) + 1
            located, reason = self._locate(target, path)
            if located is not None and located in stack:
                located, reason = None, "cycle"
            elif located is not None and len(stack) > self.max_depth:
                located, reason = None, "max_depth"
            if located is None:
                unresolved.append(Unresolved(article, path, line, target, reason))
                out.append((self.placeholder_format.format(path=target), None, None, target))
            else:
                includes.add(located)
                out.extend(_rstrip_newlines(self._expand(article, located, stack + (located,), includes, unresolved)))
            pos = m.end()
        out.append((text[pos:], path, pos, None))
        return out

    def _locate(self, target: str, current: str) -> tuple[str | None, str]:
        target = target.split("?")[0]
        if target.startswith(self.external_prefixes):
            return None, "external"
        if target.startswith("~/"):
            rel = posixpath.normpath(target[2:])
        elif target.startswith("/"):
            return None, "absolute"
        else:
            rel = posixpath.normpath(posixpath.join(posixpath.dirname(current), target))
        if rel.startswith("../") or rel == "..":
            return None, "outside_repo"
        if not (self.repo_root / rel).is_file():
            return None, "not_found"
        return rel, ""


def _rstrip_newlines(segments: list) -> list:
    """Drop trailing newlines from included text so the directive's own line ending stays."""
    segments = list(segments)
    while segments:
        text, source, start, ph = segments[-1]
        stripped = text.rstrip("\n")
        if stripped:
            segments[-1] = (stripped, source, start, ph)
            break
        segments.pop()
    return segments
