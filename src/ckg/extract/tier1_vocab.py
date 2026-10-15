"""Tier 1: dictionary matching against the vocabulary with spaCy's PhraseMatcher (DESIGN 5.2)."""

from __future__ import annotations

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span as SpacySpan
from spacy.util import filter_spans

from ckg.extract.common import Span, dropped, first_overlap, mention
from ckg.extract.vocab import Concept

METHOD = "vocabulary"


def pluralize(label: str) -> str | None:
    """Regular English plural of a label ending in a lowercase word, else None."""
    words = label.split(" ")
    last = words[-1]
    if not last.isalpha() or not last.islower() or len(last) < 3 or last.endswith("s"):
        return None
    if last.endswith(("x", "ch", "sh")):
        plural = last + "es"
    elif last.endswith("y") and last[-2] not in "aeiou":
        plural = last[:-1] + "ies"
    else:
        plural = last + "s"
    return " ".join(words[:-1] + [plural])


class VocabMatcher:
    """Matches vocabulary labels. Overlapping matches keep the longest span."""

    def __init__(self, concepts: list[Concept], case_sensitive_single_token: bool, match_plurals: bool = False) -> None:
        self.nlp = spacy.blank("en")
        self.concepts = {c.iri: c for c in concepts}
        self.lower = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.exact = PhraseMatcher(self.nlp.vocab, attr="ORTH")
        for c in concepts:
            exact, lower = [], []
            for label in c.labels:
                doc = self.nlp.make_doc(label)
                is_exact = case_sensitive_single_token and len(doc) == 1 and any(ch.isupper() for ch in label)
                (exact if is_exact else lower).append(doc)
                if match_plurals and not is_exact and (plural := pluralize(label)):
                    lower.append(self.nlp.make_doc(plural))
            if exact:
                self.exact.add(c.iri, exact)
            if lower:
                self.lower.add(c.iri, lower)

    def matches(self, text: str) -> list[tuple[int, int, Concept]]:
        doc = self.nlp.make_doc(text)
        spans = [SpacySpan(doc, s, e, label=key) for key, s, e in list(self.lower(doc)) + list(self.exact(doc))]
        kept = sorted(filter_spans(spans), key=lambda s: s.start_char)
        return [(s.start_char, s.end_char, self.concepts[s.label_]) for s in kept]

    def extract(self, chunk: dict, masked: list[Span]) -> tuple[list[dict], list[dict]]:
        text, cid = chunk["text"], chunk["chunk_id"]
        kept, rejected = [], []
        for start, end, concept in self.matches(text):
            record = mention(cid, text, start, end, METHOD, concept.domain_class, vocab_id=concept.iri)
            hit = first_overlap(masked, start, end)
            if hit:
                rejected.append(dropped(record, f"masked:{hit.reason}"))
            else:
                kept.append(record)
        return kept, rejected
