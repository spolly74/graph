"""Validate stage: SHACL plus a domain check over the loaded graph (DESIGN 5.5).

- SHACL: shapes generated from the LinkML schema, run with pySHACL over the
  union of all graphs. The ontology's rdfs:subClassOf axioms are mixed into
  the data graph, which is what SHACL needs for class checks on subclasses
  (no reasoner is run).
- Domain check: the generated shapes are open, so they can't flag a
  relationship used on the wrong subject class. Each direct triple's subject
  class is checked against the predicate's domain here.
- Datatype check: pySHACL doesn't check lexical forms, so an attribute value
  such as "September 16, 2024" typed xsd:date is checked here.

Every violation is written to the report CSV individually, linked back to
the assertion and chunk it came from where possible, and counted by type
in the summary.
"""

from __future__ import annotations

import csv
import inspect
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import rdflib
from pyoxigraph import BlankNode, Literal, NamedNode, Store
from pyshacl import validate as shacl_validate
from rdflib.namespace import RDF, SH

from ckg.config import Config
from ckg.load.pipeline import open_store
from ckg.load.rdf import RdfMap

LEXICAL = {
    "http://www.w3.org/2001/XMLSchema#date": re.compile(r"-?\d{4,}-\d{2}-\d{2}(Z|[+-]\d{2}:\d{2})?"),
    "http://www.w3.org/2001/XMLSchema#dateTime": re.compile(r"-?\d{4,}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?"),
    "http://www.w3.org/2001/XMLSchema#boolean": re.compile(r"true|false|1|0"),
    "http://www.w3.org/2001/XMLSchema#integer": re.compile(r"[+-]?\d+"),
    "http://www.w3.org/2001/XMLSchema#float": re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?|INF|-INF|NaN"),
}

COLUMNS = ["kind", "constraint", "severity", "focus", "focus_label", "focus_class", "path", "value",
           "value_label", "value_class", "message", "assertion", "evidence_chunk"]


@dataclass
class ValidateReport:
    conforms: bool = True
    triples: int = 0
    violations: int = 0
    by_kind: Counter = field(default_factory=Counter)
    by_constraint: Counter = field(default_factory=Counter)
    by_path: Counter = field(default_factory=Counter)
    outputs: list[Path] = field(default_factory=list)


def to_rdflib(term):
    if isinstance(term, NamedNode):
        return rdflib.URIRef(term.value)
    if isinstance(term, BlankNode):
        return rdflib.BNode(term.value)
    if isinstance(term, Literal):
        dt = term.datatype.value if term.datatype else None
        if term.language:
            return rdflib.Literal(term.value, lang=term.language)
        if dt in (None, "http://www.w3.org/2001/XMLSchema#string"):
            return rdflib.Literal(term.value)
        return rdflib.Literal(term.value, datatype=rdflib.URIRef(dt))
    raise TypeError(f"unexpected term {term!r}")


def store_graph(store: Store) -> rdflib.Graph:
    """The union of all named graphs as one rdflib graph (repeated triples collapse)."""
    g = rdflib.Graph()
    for q in store.quads_for_pattern(None, None, None, None):
        g.add((to_rdflib(q.subject), to_rdflib(q.predicate), to_rdflib(q.object)))
    return g


def run_validate(cfg: Config, store: Store | None = None) -> ValidateReport:
    vc = cfg.validation
    out = cfg.resolve(cfg.paths.out_dir)
    rm = RdfMap.load(cfg.resolve(cfg.store.rdf_map))
    entities = {e["id"]: e for e in _read_jsonl(out / "entities.jsonl")}
    assertions = _read_jsonl(out / "assertions.resolved.jsonl")
    labels = {rm.expand(i): e["label"] for i, e in entities.items()}
    classes = {rm.expand(i): e["entity_type"] for i, e in entities.items()}
    by_triple, by_focus, by_path = _assertion_index(rm, assertions)

    own = store is None
    store = open_store(cfg) if store is None else store
    try:
        data = store_graph(store)
    finally:
        if own:
            del store
    shapes = rdflib.Graph().parse(cfg.resolve(vc.shapes), format="turtle")
    ontology = rdflib.Graph().parse(cfg.resolve(vc.ontology), format="turtle")
    subclass = rdflib.Graph()
    for s, o in ontology.subject_objects(rdflib.RDFS.subClassOf):
        if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
            subclass.add((s, rdflib.RDFS.subClassOf, o))

    rep = ValidateReport(triples=len(data))
    conforms, results, _ = shacl_validate(data, shacl_graph=shapes, ont_graph=subclass, inference="none",
                                          abort_on_first=False, allow_warnings=True, advanced=False, meta_shacl=False)
    rows = []
    for r in results.subjects(RDF.type, SH.ValidationResult):
        focus, path, value = results.value(r, SH.focusNode), results.value(r, SH.resultPath), results.value(r, SH.value)
        focus_s, path_s, value_s = (str(x) if x is not None else "" for x in (focus, path, value))
        aid, chunk = (by_triple.get((focus_s, path_s, value_s)) or by_path.get((focus_s, path_s))
                      or by_focus.get(focus_s) or ("", ""))
        rows.append({
            "kind": "shacl",
            "constraint": _local(results.value(r, SH.sourceConstraintComponent)),
            "severity": _local(results.value(r, SH.resultSeverity)),
            "focus": focus_s, "focus_label": labels.get(focus_s, ""), "focus_class": classes.get(focus_s, ""),
            "path": _short(rm, path_s), "value": value_s, "value_label": labels.get(value_s, ""),
            "value_class": classes.get(value_s, ""), "message": str(results.value(r, SH.resultMessage) or ""),
            "assertion": aid, "evidence_chunk": chunk,
        })
    rows += _domain_violations(cfg, rm, assertions, entities)
    rows += _datatype_violations(rm, assertions, entities)

    # A violation can be reported by a class's shape and by its parent's; keep one.
    unique = {}
    for r in rows:
        unique.setdefault((r["kind"], r["constraint"], r["focus"], r["path"], r["value"]), r)
    rows = list(unique.values())
    rows.sort(key=lambda r: (r["kind"], r["constraint"], r["path"], r["focus"], r["value"]))
    rep.violations = len(rows)
    rep.conforms = conforms and not any(r["kind"] != "shacl" for r in rows)
    rep.by_kind = Counter(r["kind"] for r in rows)
    rep.by_constraint = Counter(f"{r['kind']}:{r['constraint']}" for r in rows)
    rep.by_path = Counter(r["path"] for r in rows)

    report_path = cfg.resolve(vc.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary_path = cfg.resolve(vc.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "conforms": rep.conforms, "triples": rep.triples, "violations": rep.violations,
        "by_kind": dict(sorted(rep.by_kind.items())),
        "by_constraint": dict(sorted(rep.by_constraint.items())),
        "by_path": dict(rep.by_path.most_common()),
    }, indent=2) + "\n", encoding="utf-8")
    rep.outputs = [report_path, summary_path]
    return rep


def _domain_violations(cfg: Config, rm: RdfMap, assertions: list[dict], entities: dict[str, dict]) -> list[dict]:
    """Direct triples whose subject class is outside the predicate's domain."""
    from ckg.generated import model as schema_model

    preds = {p["name"]: p for p in json.loads(cfg.resolve(cfg.extract.tier3.predicates_file).read_text(encoding="utf-8"))["predicates"]}
    model_classes = dict(inspect.getmembers(schema_model, inspect.isclass))
    rows = []
    for a in assertions:
        if a["negated"]:
            continue  # negated assertions produce no direct triple
        name = rm.compact_slot(a["predicate"])
        pred = preds.get(name or "")
        subject = entities.get(a["subject"])
        if pred is None or subject is None:
            continue
        domains = pred["domain"] if isinstance(pred["domain"], list) else [pred["domain"]]
        cls = model_classes.get(subject["entity_type"])
        if cls is None or any(d in model_classes and issubclass(cls, model_classes[d]) for d in domains if d):
            continue
        obj = entities.get(a.get("object", ""), {})
        rows.append({
            "kind": "domain", "constraint": "DomainConstraint", "severity": "Violation",
            "focus": rm.expand(a["subject"]), "focus_label": subject["label"], "focus_class": subject["entity_type"],
            "path": name, "value": rm.expand(a["object"]) if "object" in a else a.get("object_value", ""),
            "value_label": obj.get("label", ""), "value_class": obj.get("entity_type", ""),
            "message": f"{name} expects a subject of class {' or '.join(domains)}, got {subject['entity_type']}",
            "assertion": a["id"], "evidence_chunk": a["provenance"][0]["chunk_id"],
        })
    return rows


def _datatype_violations(rm: RdfMap, assertions: list[dict], entities: dict[str, dict]) -> list[dict]:
    """Attribute values whose text is not a valid lexical form for the slot's datatype."""
    rows = []
    for a in assertions:
        name = rm.compact_slot(a["predicate"])
        if a["negated"] or "object_value" not in a or name is None:
            continue
        pattern = LEXICAL.get(rm.slots[name]["datatype"] or "")
        if pattern is None or pattern.fullmatch(a["object_value"].strip()):
            continue
        subject = entities.get(a["subject"], {})
        rows.append({
            "kind": "datatype", "constraint": "LexicalFormConstraint", "severity": "Violation",
            "focus": rm.expand(a["subject"]), "focus_label": subject.get("label", ""),
            "focus_class": subject.get("entity_type", ""), "path": name, "value": a["object_value"],
            "value_label": "", "value_class": "",
            "message": f"{a['object_value']!r} is not a valid {rm.slots[name]['range']}",
            "assertion": a["id"], "evidence_chunk": a["provenance"][0]["chunk_id"],
        })
    return rows


def _assertion_index(rm: RdfMap, assertions: list[dict]):
    """Look up the assertion behind a direct triple, a subject and predicate, or an assertion node."""
    by_triple, by_focus, by_path = {}, {}, {}
    for a in assertions:
        chunk = a["provenance"][0]["chunk_id"]
        by_focus[rm.expand(a["id"])] = (a["id"], chunk)
        name = rm.compact_slot(a["predicate"])
        if name:
            value = rm.expand(a["object"]) if "object" in a else a.get("object_value", "")
            by_triple.setdefault((rm.expand(a["subject"]), rm.slots[name]["iri"], value), (a["id"], chunk))
            by_path.setdefault((rm.expand(a["subject"]), rm.slots[name]["iri"]), (a["id"], chunk))
    return by_triple, by_focus, by_path


def _local(term) -> str:
    s = str(term or "")
    return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _short(rm: RdfMap, iri: str) -> str:
    return next((n for n, s in rm.slots.items() if s["iri"] == iri), iri)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
