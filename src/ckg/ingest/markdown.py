"""Split resolved Microsoft Learn Markdown into sections and blocks.

Docs-specific rules (DESIGN 5.1):
- `# [Label](#tab/id)` starts a tab, not a heading. A tab group ends at a
  line containing only `---`. Each tab is its own section; its label is
  the section's interface hint.
- Lines starting with `#` inside fenced code blocks are not headings.
- A Markdown table (consecutive lines starting with `|`) is one block, so
  the chunker can keep it whole.

All offsets refer to the resolved text. Block and section ranges are
line-aligned: they start at a line start and end after a newline (or at
the end of the text).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LINE = re.compile(r"[^\n]*\n|[^\n]+\Z")
FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
TAB = re.compile(r"^#[ \t]*\[([^\]]+)\]\(#tab/([^)\s]+)\)[ \t]*$")
TAB_END = re.compile(r"^---[ \t]*$")
TABLE_ROW = re.compile(r"^[ \t]*\|")
TABLE_SEPARATOR = re.compile(r"^[ \t]*\|?[ \t]*:?-{3,}")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
ANCHOR = re.compile(r"\s*\{#[^}]*\}\s*$")


@dataclass(frozen=True)
class Block:
    kind: str  # para, table, code
    start: int
    end: int


@dataclass
class Section:
    key: str  # slug used for the chunk ID
    title: str
    level: int  # heading level; tabs are one deeper than their parent
    heading_path: tuple[str, ...]
    interface: str | None  # tab label, if the section is inside a tab
    heading: tuple[int, int] | None  # range of the heading or tab line
    blocks: list[Block] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.heading[0] if self.heading else self.blocks[0].start

    @property
    def end(self) -> int:
        return self.blocks[-1].end if self.blocks else self.heading[1]

    @property
    def empty(self) -> bool:
        return self.heading is None and not self.blocks


def clean_title(raw: str) -> str:
    title = ANCHOR.sub("", raw)
    title = LINK.sub(r"\1", title)
    return re.sub(r"[`*_]", "", title).strip()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def lines(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group().rstrip("\n")) for m in LINE.finditer(text)]


def parse_sections(text: str) -> list[Section]:
    rows = lines(text)
    sections: list[Section] = []
    stack: list[tuple[int, str, str]] = []  # (level, title, key)
    cur = Section("intro", "", 0, (), None, None)
    pending: list = []  # [kind, start, end]
    tab_parent: tuple[list, str, int, tuple[str, ...]] | None = None
    tab_label: str | None = None

    def flush() -> None:
        if pending:
            cur.blocks.append(Block(*pending))
            pending.clear()

    def close() -> None:
        flush()
        if not cur.empty:
            sections.append(cur)

    def path() -> tuple[str, ...]:
        return tuple(t for _, t, _ in stack)

    i = 0
    while i < len(rows):
        start, end, line = rows[i]

        fence = FENCE.match(line)
        if fence:
            flush()
            mark = fence.group(1)
            close_at = len(rows) - 1
            for j in range(i + 1, len(rows)):
                m = FENCE.match(rows[j][2])
                if m and m.group(1)[0] == mark[0] and len(m.group(1)) >= len(mark) and not rows[j][2][m.end():].strip():
                    close_at = j
                    break
            cur.blocks.append(Block("code", start, rows[close_at][1]))
            i = close_at + 1
            continue

        tab = TAB.match(line)
        if tab:
            close()
            if tab_parent is None:
                tab_parent = (list(stack), cur.key, cur.level, path())
            tab_label = tab.group(1).strip()
            _, parent_key, parent_level, parent_path = tab_parent
            cur = Section(
                f"{parent_key}--{slugify(tab.group(2))}", tab_label, parent_level + 1,
                parent_path + (tab_label,), tab_label, (start, end),
            )
            i += 1
            continue

        if tab_parent is not None and TAB_END.match(line):
            close()
            stack, parent_key, parent_level, parent_path = tab_parent
            cur = Section(parent_key, parent_path[-1] if parent_path else "", parent_level, parent_path, None, None)
            tab_parent, tab_label = None, None
            i += 1
            continue

        heading = HEADING.match(line)
        if heading:
            close()
            level, title = len(heading.group(1)), clean_title(heading.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            key = slugify(title)
            stack.append((level, title, key))
            cur = Section(key, title, level, path(), tab_label, (start, end))
            i += 1
            continue

        if not line.strip():
            flush()
        else:
            kind = "table" if TABLE_ROW.match(line) else "para"
            if pending and pending[0] == kind:
                pending[2] = end
            else:
                flush()
                pending.extend([kind, start, end])
        i += 1

    close()
    return sections
