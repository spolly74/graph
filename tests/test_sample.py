"""The sample corpus must be self-contained: every in-repo include it uses is present."""

from __future__ import annotations

import re

from conftest import SAMPLE_BASE, SAMPLE_REPO

INCLUDE = re.compile(r"\[!INCLUDE\s*\[[^\]]*\]\(([^)]+)\)\]", re.I)
EXTERNAL = ("~/reusable-content/", "~/azure-policy-autogen-docs/")


def test_sample_has_ten_articles():
    base = SAMPLE_REPO / SAMPLE_BASE
    articles = [p for p in base.rglob("*.md") if "includes" not in p.relative_to(base).parts]
    assert len(articles) == 10


def test_sample_includes_resolve():
    missing = []
    for f in SAMPLE_REPO.rglob("*.md"):
        for m in INCLUDE.finditer(f.read_text(encoding="utf-8")):
            target = m.group(1).strip().split("#")[0]
            if target.startswith(EXTERNAL):
                continue
            path = SAMPLE_REPO / target[2:] if target.startswith("~/") else f.parent / target
            if not path.is_file():
                missing.append(f"{f.relative_to(SAMPLE_REPO)} -> {target}")
    assert not missing, missing


def test_sample_covers_external_includes():
    """At least one sample article uses an external include, so Phase 1 can test placeholders."""
    text = "".join(p.read_text(encoding="utf-8") for p in SAMPLE_REPO.rglob("*.md"))
    assert any(prefix in text for prefix in EXTERNAL)
