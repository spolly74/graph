"""Name normalization for entity resolution (DESIGN 5.3, step 1)."""

from __future__ import annotations

import re

EDGE = " \t\n`*_\"'.,:;!?()[]{}<>"


def singular(name: str) -> str:
    """Singularize the last word with regular English rules."""
    head, _, last = name.rpartition(" ")
    if len(last) <= 3 or not last.isalpha():
        return name
    if last.endswith("ies") and len(last) > 4:
        last = last[:-3] + "y"
    elif last.endswith(("xes", "ches", "shes", "sses")):
        last = last[:-2]
    elif last.endswith("s") and not last.endswith(("ss", "us", "is")):
        last = last[:-1]
    return f"{head} {last}" if head else last


def normalize(text: str, singularize: bool) -> str:
    name = " ".join(text.split()).strip(EDGE).lower()
    name = re.sub(r"\s*([-/])\s*", r"\1", name)  # "Arc - enabled" == "Arc-enabled"
    return singular(name) if singularize else name


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "entity"
