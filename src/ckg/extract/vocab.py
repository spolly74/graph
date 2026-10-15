"""Load the SKOS vocabulary: each concept with its labels and domain class."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rdflib
from rdflib.namespace import RDF, SKOS


@dataclass(frozen=True)
class Concept:
    iri: str
    domain_class: str
    labels: tuple[str, ...]  # preferred label first, then alternates in sorted order


def load_vocabulary(path: Path) -> list[Concept]:
    g = rdflib.Graph().parse(path, format="turtle")
    domain_class = rdflib.URIRef(_terms_ns(g) + "domainClass")
    scheme_class = {str(s): str(o) for s, o in g.subject_objects(domain_class)}
    concepts = []
    for c in sorted(set(g.subjects(RDF.type, SKOS.Concept)), key=str):
        schemes = [str(s) for s in g.objects(c, SKOS.inScheme)]
        cls = next((scheme_class[s] for s in schemes if s in scheme_class), None)
        if cls is None:
            raise ValueError(f"{c}: its concept scheme has no ckg:domainClass")
        pref = [str(o) for o in g.objects(c, SKOS.prefLabel)]
        alts = sorted({str(o) for o in g.objects(c, SKOS.altLabel)} - set(pref))
        concepts.append(Concept(str(c), cls, tuple(pref + alts)))
    return concepts


def _terms_ns(g: rdflib.Graph) -> str:
    for prefix, ns in g.namespaces():
        if prefix == "ckg":
            return str(ns)
    raise ValueError("vocabulary has no ckg: prefix for its terms namespace")
