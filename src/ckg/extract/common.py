"""Shared pieces for the extraction tiers: chunk loading, masking, mention records."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ckg.config import MaskConfig
from ckg.ingest.markdown import FENCE, lines

LINK_TARGET = re.compile(r"\]\(([^)\n]*)\)")
BARE_URL = re.compile(r"https?://[^\s)>\]]+")
DIRECTIVE = re.compile(r":::.*?:::|^[ \t]*:::.*$", re.M)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
CALLOUT_MARK = re.compile(r"\[!(?:NOTE|TIP|IMPORTANT|CAUTION|WARNING)\]", re.I)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    reason: str

    def overlaps(self, start: int, end: int) -> bool:
        return self.start < end and start < self.end


def read_chunks(out_dir: Path) -> list[dict]:
    path = out_dir / "chunks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `ckg ingest` first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def masked_spans(chunk: dict, mask: MaskConfig) -> list[Span]:
    """Ranges of chunk text that extraction must not match inside."""
    text = chunk["text"]
    spans = [Span(p["start"], p["end"], "placeholder") for p in chunk["pieces"] if "placeholder" in p]
    if mask.code_blocks:
        spans += _fenced_code(text)
    if mask.link_targets:
        spans += [Span(m.start(1), m.end(1), "link_target") for m in LINK_TARGET.finditer(text)]
        spans += [Span(m.start(), m.end(), "link_target") for m in BARE_URL.finditer(text)]
    if mask.directives:
        spans += [Span(m.start(), m.end(), "directive") for m in DIRECTIVE.finditer(text)]
    if mask.html_comments:
        spans += [Span(m.start(), m.end(), "html_comment") for m in HTML_COMMENT.finditer(text)]
    if mask.callouts:
        spans += [Span(m.start(), m.end(), "callout") for m in CALLOUT_MARK.finditer(text)]
    return sorted(spans, key=lambda s: (s.start, s.end))


def _fenced_code(text: str) -> list[Span]:
    rows = lines(text)
    spans, i = [], 0
    while i < len(rows):
        m = FENCE.match(rows[i][2])
        if not m:
            i += 1
            continue
        mark, close_at = m.group(1), len(rows) - 1
        for j in range(i + 1, len(rows)):
            n = FENCE.match(rows[j][2])
            if n and n.group(1)[0] == mark[0] and len(n.group(1)) >= len(mark) and not rows[j][2][n.end():].strip():
                close_at = j
                break
        spans.append(Span(rows[i][0], rows[close_at][1], "code_block"))
        i = close_at + 1
    return spans


def first_overlap(spans: list[Span], start: int, end: int) -> Span | None:
    return next((s for s in spans if s.overlaps(start, end)), None)


def unmasked_segments(text: str, spans: list[Span]) -> list[tuple[int, int]]:
    """The complement of the masked spans within the text."""
    out, pos = [], 0
    for s in spans:
        if s.start > pos:
            out.append((pos, s.start))
        pos = max(pos, s.end)
    if pos < len(text):
        out.append((pos, len(text)))
    return out


def mention_id(chunk_id: str, start: int, end: int, method: str) -> str:
    digest = hashlib.sha256(f"{chunk_id}|{start}|{end}|{method}".encode("utf-8")).hexdigest()[:16]
    return f"ckg:mention-{digest}"


def mention(chunk_id: str, text: str, start: int, end: int, method: str, predicted_class: str,
            vocab_id: str | None = None, score: float | None = None, model: str | None = None) -> dict:
    record = {
        "id": mention_id(chunk_id, start, end, method),
        "chunk_id": chunk_id,
        "start": start,
        "end": end,
        "text": text[start:end],
        "predicted_class": predicted_class,
        "method": method,
    }
    if vocab_id is not None:
        record["vocab_id"] = vocab_id
    if score is not None:
        record["score"] = round(score, 4)
    if model is not None:
        record["model"] = model
    return record


def dropped(record: dict, reason: str) -> dict:
    return {k: record[k] for k in ("chunk_id", "start", "end", "text", "method", "predicted_class")} | {"reason": reason}
