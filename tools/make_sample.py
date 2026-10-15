"""Copy sample articles and the in-repo include files they use into data/sample.

The sample mirrors the repo layout from its root, so relative include paths
such as ../includes/x.md and ~/articles/... resolve the same way they do in
the real checkout. Includes into external repos are not copied.

Usage:
    python tools/make_sample.py <repo root> <base dir> <dest> <article> [<article> ...]
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

INCLUDE = re.compile(r"\[!INCLUDE\s*\[[^\]]*\]\(([^)]+)\)\]", re.I)
EXTERNAL = ("~/reusable-content/", "~/azure-policy-autogen-docs/")


def resolve(target: str, current: Path, repo: Path) -> Path | None:
    target = target.strip().split("#")[0]
    if target.startswith(EXTERNAL):
        return None
    if target.startswith("~/"):
        return (repo / target[2:]).resolve()
    if target.startswith("/"):
        return None
    return (current.parent / target).resolve()


def main() -> None:
    repo, base, dest = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
    queue = [(repo / base / a).resolve() for a in sys.argv[4:]]
    seen: set[Path] = set()
    while queue:
        f = queue.pop()
        if f in seen:
            continue
        seen.add(f)
        if not f.is_file():
            print(f"missing: {f.relative_to(repo)}")
            continue
        out = dest / f.relative_to(repo)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, out)
        for m in INCLUDE.finditer(f.read_text(encoding="utf-8")):
            r = resolve(m.group(1), f, repo)
            if r is not None:
                queue.append(r)
    articles = len(sys.argv) - 4
    print(f"copied {articles} articles and {len(seen) - articles} include files to {dest}")


if __name__ == "__main__":
    main()
