"""Ingest stage: corpus files -> documents.jsonl, chunks.jsonl (DESIGN 5.1).

Outputs are deterministic: files are processed in sorted order, JSON keys
are sorted, and nothing time-dependent is written, so rerunning on
unchanged input produces byte-identical files.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from ckg.config import Config
from ckg.ingest.chunker import Chunker, ChunkPlan
from ckg.ingest.includes import IncludeResolver, Piece, Resolved, Unresolved
from ckg.ingest.loader import MarkdownLoader
from ckg.ingest.markdown import parse_sections

CHUNKS_FILE = "chunks.jsonl"
DOCUMENTS_FILE = "documents.jsonl"
UNRESOLVED_FILE = "unresolved_includes.csv"
CALLOUT = re.compile(r"^[ \t]*>[ \t]*\[!(NOTE|TIP|IMPORTANT|CAUTION|WARNING)\]", re.M | re.I)


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    unresolved: int = 0
    oversize: list[str] = field(default_factory=list)
    max_tokens: int = 0
    outputs: list[Path] = field(default_factory=list)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_ingest(cfg: Config, base: Path | None = None) -> IngestReport:
    repo_root = cfg.repo_root
    base = (base or cfg.corpus_dir).resolve()
    base_rel = base.relative_to(repo_root).as_posix()
    ic = cfg.ingest
    loader = MarkdownLoader()
    resolver = IncludeResolver(repo_root, loader, cfg.corpus.external_include_prefixes, ic.placeholder_format, ic.max_include_depth)
    chunker = Chunker(ic.chars_per_token, ic.chunk_target_tokens, ic.chunk_max_tokens)

    report = IngestReport()
    documents, chunks, unresolved = [], [], []
    for rel in cfg.corpus_files(base):
        doc_id = rel.as_posix()
        article = f"{base_rel}/{doc_id}" if base_rel != "." else doc_id
        resolved = resolver.resolve(article)
        sections = parse_sections(resolved.text)
        plans = chunker.plan(resolved.text, sections)
        front = loader.load(repo_root, article).front_matter

        title = str(front.get("title") or next((s.title for s in sections if s.level == 1), ""))
        preview_pieces = [
            p for p in resolved.pieces
            if p.source not in (None, article) and any(fnmatch(p.source, pat) for pat in ic.preview_include_patterns)
        ]
        first_h2 = min((s.start for s in sections if s.level == 2 and s.heading), default=len(resolved.text))
        article_preview = "(preview)" in title.lower() or any(p.start < first_h2 for p in preview_pieces)

        doc_chunks = build_chunks(doc_id, resolved, plans, chunker, article_preview, preview_pieces)
        for c in doc_chunks:
            if c["oversize"]:
                report.oversize.append(c["chunk_id"])
            report.max_tokens = max(report.max_tokens, c["token_estimate"])
        chunks.extend(doc_chunks)
        unresolved.extend(resolved.unresolved)
        documents.append({
            "doc_id": doc_id,
            "path": article,
            "title": title,
            "metadata": {k: front[k] for k in ic.front_matter_keys if k in front},
            "sha256": sha256(resolved.text),
            "includes": list(resolved.includes),
            "unresolved_includes": len(resolved.unresolved),
            "release_status_hint": "preview" if article_preview else None,
            "chunk_ids": [c["chunk_id"] for c in doc_chunks],
        })

    out_dir = cfg.resolve(cfg.paths.out_dir)
    review_dir = cfg.resolve(cfg.paths.review_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    report.outputs = [
        _write_jsonl(out_dir / DOCUMENTS_FILE, documents),
        _write_jsonl(out_dir / CHUNKS_FILE, chunks),
        _write_unresolved(review_dir / UNRESOLVED_FILE, unresolved),
    ]
    report.documents, report.chunks, report.unresolved = len(documents), len(chunks), len(unresolved)
    return report


def build_chunks(
    doc_id: str,
    resolved: Resolved,
    plans: list[ChunkPlan],
    chunker: Chunker,
    article_preview: bool,
    preview_pieces: list[Piece],
) -> list[dict]:
    starts = [p.start for p in resolved.pieces]
    used: set[str] = set()
    out = []
    for seq, plan in enumerate(plans):
        key = _unique(plan.key, used)
        text = "".join(resolved.text[s:e] for s, e in plan.segments)
        pieces = _map_pieces(resolved.pieces, starts, plan.segments)
        preview = article_preview or any(
            pp.start < e and s < pp.end for pp in preview_pieces for s, e in plan.segments
        )
        out.append({
            "chunk_id": f"{doc_id}#{key}",
            "doc_id": doc_id,
            "seq": seq,
            "heading_path": list(plan.heading_path),
            "text": text,
            "sha256": sha256(text),
            "token_estimate": chunker.tokens(len(text)),
            "oversize": plan.oversize,
            "pieces": [p.to_json() for p in pieces],
            "hints": {
                "interface": plan.interface,
                "release_status": "preview" if preview else None,
                "callouts": sorted({m.upper() for m in CALLOUT.findall(text)}),
                "has_table": any(line.lstrip().startswith("|") for line in text.splitlines()),
                "has_code": "```" in text or "~~~" in text,
            },
        })
    return out


def _unique(key: str, used: set[str]) -> str:
    candidate, n = key, 1
    while candidate in used:
        n += 1
        candidate = f"{key}-{n}"
    used.add(candidate)
    return candidate


def _map_pieces(doc_pieces, starts, segments) -> list[Piece]:
    """Translate resolved-text pieces covering the segments into chunk offsets."""
    out: list[Piece] = []
    offset = 0
    for s, e in segments:
        i = max(bisect.bisect_right(starts, s) - 1, 0)
        while i < len(doc_pieces) and doc_pieces[i].start < e:
            p = doc_pieces[i]
            lo, hi = max(p.start, s), min(p.end, e)
            if lo < hi:
                src_start = None if p.source is None else p.source_start + (lo - p.start)
                out.append(Piece(offset + lo - s, offset + hi - s, p.source, src_start, p.placeholder))
            i += 1
        offset += e - s
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _write_unresolved(path: Path, rows: list[Unresolved]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["article", "source", "line", "target", "reason"])
        for u in sorted(rows, key=lambda u: (u.article, u.source, u.line, u.target)):
            w.writerow([u.article, u.source, u.line, u.target, u.reason])
    return path
