"""Pack sections into chunks (DESIGN 5.1).

- A chunk starts at a heading boundary. Consecutive small sections with the
  same interface are merged while the chunk stays within the target size.
- A section larger than the target is split between blocks. Later parts
  repeat the section's heading line so each chunk keeps its context.
- A block larger than the maximum is split by lines. A table is split by
  rows and each part repeats the header and separator rows. A table is
  never cut inside a row.

A chunk's text is the concatenation of one or more ranges of the resolved
text (its segments), so repeated headings and table headers still map back
to real source offsets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ckg.ingest.markdown import TABLE_SEPARATOR, Block, Section, lines


@dataclass(frozen=True)
class ChunkPlan:
    key: str
    heading_path: tuple[str, ...]
    interface: str | None
    segments: tuple[tuple[int, int], ...]
    oversize: bool = False  # a single line longer than the maximum


class Chunker:
    def __init__(self, chars_per_token: int, target_tokens: int, max_tokens: int) -> None:
        self.cpt = chars_per_token
        self.target = target_tokens
        self.max = max_tokens

    def tokens(self, n_chars: int) -> int:
        return math.ceil(n_chars / self.cpt)

    def plan(self, text: str, sections: list[Section]) -> list[ChunkPlan]:
        plans: list[ChunkPlan] = []
        group: list[Section] = []
        size = 0

        def flush() -> None:
            nonlocal group, size
            if group:
                first = group[0]
                body = next((s for s in group if s.blocks), first)
                segs = _join(text, [(s.start, s.end) for s in group])
                plans.append(ChunkPlan(first.key, first.heading_path, body.interface, segs))
            group, size = [], 0

        # Headings with no text of their own (an H2 directly above an H3 or a
        # tab group) always lead into the next section rather than ending up
        # as heading-only chunks. The chunk takes the key of the first heading
        # and the interface of the first section with content.
        for sec in sections:
            sec_size = self.tokens(sec.end - sec.start)
            headings_only = bool(group) and all(not s.blocks for s in group)
            if sec_size > self.target:
                lead: list[Section] = []
                if headings_only:
                    lead, group, size = group, [], 0
                flush()
                split = self._split_section(text, sec)
                if lead:
                    first = split[0]
                    segs = _join(text, [(s.start, s.end) for s in lead] + [first.segments[0]]) + first.segments[1:]
                    split[0] = ChunkPlan(lead[0].key, lead[0].heading_path, first.interface, segs, first.oversize)
                plans.extend(split)
                continue
            if group and not headings_only:
                body_interface = next((s.interface for s in group if s.blocks), group[0].interface)
                if body_interface != sec.interface or size + sec_size > self.target:
                    flush()
            group.append(sec)
            size += sec_size
        flush()
        return plans

    def _split_section(self, text: str, sec: Section) -> list[ChunkPlan]:
        head = [sec.heading] if sec.heading else []
        head_tokens = self.tokens(sec.heading[1] - sec.heading[0]) if sec.heading else 0
        budget = self.max - head_tokens
        parts: list[tuple[tuple[tuple[int, int], ...], bool]] = []
        current: list[Block] = []
        size = head_tokens

        def close() -> None:
            nonlocal current, size
            if current:
                ranges = [(b.start, b.end) for b in current]
                if not parts and sec.heading:
                    segs = _join(text, [sec.heading] + ranges)  # first part: contiguous with heading
                else:
                    segs = tuple(head) + _join(text, ranges)
                parts.append((segs, False))
            current, size = [], head_tokens

        for block in sec.blocks:
            b_tokens = self.tokens(block.end - block.start)
            if b_tokens > budget:
                close()
                for segs, oversize in self._split_block(text, block, budget):
                    parts.append((tuple(head) + segs, oversize))
                continue
            if current and size + b_tokens > self.target:
                close()
            current.append(block)
            size += b_tokens
        close()
        if not parts and sec.heading:  # heading with no blocks
            parts.append((tuple(head), False))
        return [ChunkPlan(sec.key, sec.heading_path, sec.interface, segs, oversize) for segs, oversize in parts]

    def _split_block(self, text: str, block: Block, budget: int):
        rows = [(s + block.start, e + block.start) for s, e, _ in lines(text[block.start : block.end])]
        header: list[tuple[int, int]] = []
        if block.kind == "table" and len(rows) > 2 and TABLE_SEPARATOR.match(text[rows[1][0] : rows[1][1]]):
            header, rows = rows[:2], rows[2:]
        header_tokens = sum(self.tokens(e - s) for s, e in header)
        group: list[tuple[int, int]] = []
        size = header_tokens
        out = []
        for row in rows:
            r_tokens = self.tokens(row[1] - row[0])
            if group and size + r_tokens > budget:
                out.append((tuple(header) + ((group[0][0], group[-1][1]),), size > budget))
                group, size = [], header_tokens
            group.append(row)
            size += r_tokens
        if group:
            out.append((tuple(header) + ((group[0][0], group[-1][1]),), size > budget))
        return out


def _join(text: str, ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Merge ranges separated only by whitespace into one contiguous range."""
    merged: list[list[int]] = []
    for s, e in ranges:
        if merged and not text[merged[-1][1] : s].strip():
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return tuple((s, e) for s, e in merged)
