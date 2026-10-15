"""Extract stage, tiers 1 and 2: chunks.jsonl -> mentions (DESIGN 5.2).

Outputs, in paths.out_dir:
    mentions.tier1.jsonl    every tier 1 mention that passed its checks
    mentions.tier2.jsonl    every tier 2 mention that passed its checks
    mentions.jsonl          tier 1 mentions plus tier 2 mentions that don't overlap them
    mentions.dropped.jsonl  every rejected mention, with the reason

With changed_only, a chunk is reused from the previous tier output when its
hash and the tier's settings are unchanged since the last run.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ckg.config import Config
from ckg.extract.common import dropped, first_overlap, masked_spans, read_chunks, Span
from ckg.extract.tier1_vocab import VocabMatcher
from ckg.extract.tier2_gliner import GlinerExtractor, class_labels, load_gliner
from ckg.extract.vocab import load_vocabulary
from ckg.state import StateDB

TIER_FILES = {1: "mentions.tier1.jsonl", 2: "mentions.tier2.jsonl"}
DROPPED_FILES = {1: "mentions.tier1.dropped.jsonl", 2: "mentions.tier2.dropped.jsonl"}
MERGED_FILE = "mentions.jsonl"
DROPPED_FILE = "mentions.dropped.jsonl"


@dataclass
class TierReport:
    tier: int
    processed: int = 0
    reused: int = 0
    mentions: int = 0
    dropped: Counter = field(default_factory=Counter)


@dataclass
class ExtractReport:
    tiers: list[TierReport] = field(default_factory=list)
    merged: int = 0
    dropped: Counter = field(default_factory=Counter)
    outputs: list[Path] = field(default_factory=list)


def run_extract(cfg: Config, tiers: list[int], changed_only: bool = False,
                gliner_factory: Callable | None = None) -> ExtractReport:
    out_dir = cfg.resolve(cfg.paths.out_dir)
    chunks = read_chunks(out_dir)
    ex = cfg.extract
    masked = {c["chunk_id"]: masked_spans(c, ex.mask) for c in chunks}
    state = StateDB(cfg.resolve(cfg.paths.state_db))
    report = ExtractReport()
    try:
        for tier in tiers:
            if tier == 1:
                vocab_path = cfg.resolve(ex.vocabulary)
                settings = _settings_hash(vocab_path.read_bytes(), ex.mask, ex.tier1)
                matcher = None

                def run1(todo: list[dict]):
                    nonlocal matcher
                    matcher = matcher or VocabMatcher(load_vocabulary(vocab_path), ex.tier1.case_sensitive_single_token, ex.tier1.match_plurals)
                    kept, rejected = [], []
                    for c in todo:
                        k, r = matcher.extract(c, masked[c["chunk_id"]])
                        kept += k
                        rejected += r
                    return kept, rejected

                report.tiers.append(_run_tier(1, chunks, out_dir, state, settings, changed_only, run1))
            elif tier == 2:
                t2 = ex.tier2
                from ckg.generated import model as schema_model

                labels = class_labels(schema_model, t2.label_overrides, t2.exclude_classes)
                settings = _settings_hash(t2, labels, ex.mask)
                extractor = None

                def run2(todo: list[dict]):
                    nonlocal extractor
                    if extractor is None:
                        model, name = (gliner_factory or load_gliner)(t2.model, t2.revision)
                        extractor = GlinerExtractor(model, labels, t2.threshold, t2.max_words, t2.batch_size, name)
                    return extractor.extract(todo, masked)

                report.tiers.append(_run_tier(2, chunks, out_dir, state, settings, changed_only, run2))
            else:
                raise ValueError(f"tier {tier} is not handled by this stage")
    finally:
        state.close()

    merged, merge_dropped = merge(chunks, out_dir)
    report.merged = len(merged)
    all_dropped = merge_dropped
    for tier in (1, 2):
        all_dropped += _read_jsonl(out_dir / DROPPED_FILES[tier])
    report.dropped = Counter(d["reason"] for d in all_dropped)
    report.outputs = [
        _write_jsonl(out_dir / MERGED_FILE, merged),
        _write_jsonl(out_dir / DROPPED_FILE, all_dropped),
    ]
    return report


def _run_tier(tier, chunks, out_dir, state, settings, changed_only, extract_fn) -> TierReport:
    stage = f"extract.tier{tier}"
    rep = TierReport(tier)
    previous = _by_chunk(_read_jsonl(out_dir / TIER_FILES[tier])) if changed_only else {}
    previous_dropped = _by_chunk(_read_jsonl(out_dir / DROPPED_FILES[tier])) if changed_only else {}
    seen = state.processed(stage) if changed_only else {}

    todo, reused = [], set()
    for c in chunks:
        cid = c["chunk_id"]
        if changed_only and seen.get(cid) == (c["sha256"], settings):
            reused.add(cid)
        else:
            todo.append(c)
    kept, rejected = extract_fn(todo) if todo else ([], [])
    new_kept, new_rejected = _by_chunk(kept), _by_chunk(rejected)

    mentions, drops = [], []
    for c in chunks:
        cid = c["chunk_id"]
        source_kept, source_drop = (previous, previous_dropped) if cid in reused else (new_kept, new_rejected)
        mentions += source_kept.get(cid, [])
        drops += source_drop.get(cid, [])
    _write_jsonl(out_dir / TIER_FILES[tier], mentions)
    _write_jsonl(out_dir / DROPPED_FILES[tier], drops)
    state.replace(stage, [(c["chunk_id"], c["sha256"], settings) for c in chunks])

    rep.processed, rep.reused, rep.mentions = len(todo), len(reused), len(mentions)
    rep.dropped = Counter(d["reason"] for d in drops)
    return rep


def merge(chunks: list[dict], out_dir: Path) -> tuple[list[dict], list[dict]]:
    """Tier 1 wins over tier 2 wherever their spans overlap (tier 1 has exact vocabulary IDs)."""
    t1 = _by_chunk(_read_jsonl(out_dir / TIER_FILES[1]))
    t2 = _by_chunk(_read_jsonl(out_dir / TIER_FILES[2]))
    merged, rejected = [], []
    for c in chunks:
        cid = c["chunk_id"]
        first = t1.get(cid, [])
        spans = [Span(m["start"], m["end"], "tier1") for m in first]
        extra = []
        for m in t2.get(cid, []):
            if first_overlap(spans, m["start"], m["end"]):
                rejected.append(dropped(m, "overlaps_tier1"))
            else:
                extra.append(m)
        merged += sorted(first + extra, key=lambda m: (m["start"], m["end"], m["method"]))
    return merged, rejected


def _settings_hash(*parts) -> str:
    def norm(p):
        if isinstance(p, bytes):
            return hashlib.sha256(p).hexdigest()
        if hasattr(p, "model_dump"):
            return p.model_dump(mode="json")
        return p
    return hashlib.sha256(json.dumps([norm(p) for p in parts], sort_keys=True).encode("utf-8")).hexdigest()


def _by_chunk(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["chunk_id"], []).append(r)
    return out


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path
