"""Turn resolved records into RDF quads in named graphs (DESIGN 5.4).

- `<graph_base>entities`: every entity's type, label, vocabulary match, and aliases.
- `<graph_base>assertions`: every assertion node with its conditions, plus the
  direct triple (subject, predicate, object). Negated assertions get no
  direct triple.
- `<graph_base>chunk/<chunk id>`: the chunk's mentions, and the provenance
  records of assertions the chunk supports (with the assertion -> provenance link).

Each triple is stored once, because Oxigraph's union default graph does not
de-duplicate triples repeated across graphs. Every graph is replaced as a
whole when its content changes, so a reload never leaves orphaned triples.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from pyoxigraph import Literal, NamedNode, Quad

RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"
XSD_ANYURI = "http://www.w3.org/2001/XMLSchema#anyURI"
ENTITIES = "entities"
ASSERTIONS = "assertions"
SHARED_GRAPHS = (ENTITIES, ASSERTIONS)


@dataclass(frozen=True)
class RdfMap:
    prefixes: dict[str, str]
    classes: dict[str, str]
    slots: dict[str, dict]

    @classmethod
    def load(cls, path: Path) -> "RdfMap":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["prefixes"], d["classes"], d["slots"])

    def expand(self, curie_or_iri: str) -> str:
        prefix, sep, local = curie_or_iri.partition(":")
        if sep and prefix in self.prefixes and not local.startswith("//"):
            return self.prefixes[prefix] + local
        return curie_or_iri

    def compact_slot(self, iri: str) -> str | None:
        """Slot name for a predicate IRI or CURIE."""
        full = self.expand(iri)
        return next((n for n, s in self.slots.items() if s["iri"] == full), None)

    def node(self, value: str) -> NamedNode:
        return NamedNode(self.expand(value))

    def cls(self, name: str) -> NamedNode:
        return NamedNode(self.classes[name])

    def pred(self, slot: str) -> NamedNode:
        return NamedNode(self.slots[slot]["iri"])

    def value(self, slot: str, value) -> NamedNode | Literal:
        """An RDF term for a slot value, typed by the slot's range."""
        s = self.slots[slot]
        if s["kind"] == "class" or s["datatype"] == XSD_ANYURI:
            return self.node(str(value))
        if s["kind"] == "enum" or s["datatype"] in (None, XSD_STRING):
            return Literal(str(value))
        if isinstance(value, bool):
            value = "true" if value else "false"
        return Literal(str(value), datatype=NamedNode(s["datatype"]))


def chunk_graph(base: str, key: str) -> NamedNode:
    """Graph IRI for a chunk ID, or for one of the shared graph names."""
    if key in SHARED_GRAPHS:
        return NamedNode(base + key)
    return NamedNode(base + "chunk/" + quote(key, safe=""))


def build_quads(rm: RdfMap, graph_base: str, entities: dict[str, dict], mentions: list[dict],
                assertions: list[dict], load_mentions: bool) -> dict[str, list[Quad]]:
    """Quads per graph key (a chunk ID or a shared graph name), sorted for stable hashing."""
    graphs: dict[str, set] = defaultdict(set)

    def add(key: str, s, p, o) -> None:
        graphs[key].add(Quad(s, p, o, chunk_graph(graph_base, key)))

    for eid, e in entities.items():
        node = rm.node(eid)
        add(ENTITIES, node, RDF_TYPE, rm.cls(e["entity_type"]))
        add(ENTITIES, node, rm.pred("label"), rm.value("label", e["label"]))
        if "vocab_id" in e:
            add(ENTITIES, node, rm.pred("vocab_id"), rm.value("vocab_id", e["vocab_id"]))
        for alias in e.get("aliases", []):
            add(ENTITIES, node, rm.pred("aliases"), rm.value("aliases", alias))

    if load_mentions:
        for m in mentions:
            node = rm.node(m["id"])
            add(m["chunk_id"], node, RDF_TYPE, rm.cls("Mention"))
            for slot in ("chunk_id", "start", "end", "text", "predicted_class", "vocab_id", "entity", "method", "score", "model"):
                if m.get(slot) is not None:
                    add(m["chunk_id"], node, rm.pred(slot), rm.value(slot, m[slot]))

    for a in assertions:
        node = rm.node(a["id"])
        slot = rm.compact_slot(a["predicate"])
        g = ASSERTIONS
        add(g, node, RDF_TYPE, rm.cls("Assertion"))
        add(g, node, rm.pred("subject"), rm.node(a["subject"]))
        add(g, node, rm.pred("predicate"), rm.node(a["predicate"]))
        add(g, node, rm.pred("negated"), rm.value("negated", a["negated"]))
        if "object" in a:
            add(g, node, rm.pred("object"), rm.node(a["object"]))
        else:
            add(g, node, rm.pred("object_value"), rm.value("object_value", a["object_value"]))
        if a.get("confidence") is not None:
            add(g, node, rm.pred("confidence"), rm.value("confidence", a["confidence"]))
        for i, c in enumerate(a["conditions"]):
            cnode = NamedNode(f"{node.value}/condition/{i}")
            add(g, node, rm.pred("conditions"), cnode)
            add(g, cnode, RDF_TYPE, rm.cls("Condition"))
            add(g, cnode, rm.pred("type"), rm.value("type", c["type"]))
            add(g, cnode, rm.pred("value"), rm.value("value", c["value"]))
        if not a["negated"] and slot is not None:
            obj = rm.node(a["object"]) if "object" in a else rm.value(slot, a["object_value"])
            add(g, rm.node(a["subject"]), rm.pred(slot), obj)
        for p in a["provenance"]:
            digest = hashlib.sha256(f"{p['chunk_id']}|{p['start']}|{p['end']}".encode()).hexdigest()[:12]
            pnode = NamedNode(f"{node.value}/provenance/{digest}")
            add(p["chunk_id"], node, rm.pred("provenance"), pnode)
            add(p["chunk_id"], pnode, RDF_TYPE, rm.cls("Provenance"))
            for field in ("source_doc", "include_file", "chunk_id", "start", "end", "evidence", "method", "model", "extracted_at"):
                if p.get(field) is not None:
                    add(p["chunk_id"], pnode, rm.pred(field), rm.value(field, p[field]))

    return {key: sorted(quads, key=str) for key, quads in sorted(graphs.items())}


def quads_hash(quads: list[Quad]) -> str:
    return hashlib.sha256("\n".join(str(q) for q in quads).encode("utf-8")).hexdigest()
