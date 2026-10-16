"""Phase 6: load into Oxigraph and validate (DESIGN 5.4-5.5, phase 6 done-when)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from pyoxigraph import NamedNode, Store

from ckg.config import load_config
from ckg.load.pipeline import run_load
from ckg.load.rdf import RdfMap, build_quads
from ckg.validate.pipeline import run_validate

from conftest import PROJECT, sample_config_dict

V = "https://example.org/ckg/vocab/"
E = "https://example.org/ckg/entity/"
QUERY = (PROJECT / "queries" / "assertions_with_conditions.rq").read_text(encoding="utf-8")


def entity(eid, cls, label, aliases=(), vocab=False):
    e = {"id": eid, "entity_type": cls, "label": label, "aliases": list(aliases)}
    if vocab:
        e["vocab_id"] = eid
    return e


def mention(i, chunk, text, eid):
    return {"id": f"ckg:mention-{i}", "chunk_id": chunk, "start": 0, "end": len(text), "text": text,
            "predicted_class": "Resource", "method": "vocabulary", "entity": eid}


def assertion(i, s, pred, chunks, obj=None, value=None, negated=False, conditions=()):
    a = {"id": f"ckg:assertion-{i}", "subject": s, "predicate": f"ckgd:{pred}", "negated": negated,
         "conditions": list(conditions),
         "provenance": [{"source_doc": c.split("#")[0], "chunk_id": c, "start": 0, "end": 9, "evidence": "evidence.",
                         "method": "llm", "model": "fake@0"} for c in chunks]}
    if obj:
        a["object"] = obj
    else:
        a["object_value"] = value
    return a


ENTITIES = [
    entity(V + "compute-cluster", "ComputeTarget", "compute cluster", ["AmlCompute"], vocab=True),
    entity(V + "managed-identity", "IdentityConcept", "managed identity", vocab=True),
    entity(V + "role-reader", "Role", "Reader", vocab=True),
    entity(E + "east-us-2", "Region", "East US 2"),
    entity(E + "old-sdk", "Product", "old SDK"),
]
MENTIONS = [mention(1, "a.md#x", "AmlCompute", V + "compute-cluster"), mention(2, "a.md#x", "managed identity", V + "managed-identity"),
            mention(3, "b.md#y", "compute cluster", V + "compute-cluster"), mention(4, "b.md#y", "East US 2", E + "east-us-2")]
ASSERTIONS = [
    assertion(1, V + "compute-cluster", "uses_identity", ["a.md#x", "b.md#y"], obj=V + "managed-identity",
              conditions=[{"type": "region", "value": "East US 2"}]),
    assertion(2, V + "compute-cluster", "available_in", ["b.md#y"], obj=E + "east-us-2"),
    assertion(3, V + "compute-cluster", "supports", ["a.md#x"], obj=V + "role-reader", negated=True),
    assertion(4, E + "old-sdk", "retirement_date", ["b.md#y"], value="2026-06-30"),
]


@pytest.fixture
def env(tmp_path: Path):
    data = sample_config_dict()
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(config)
    out = cfg.resolve(cfg.paths.out_dir)
    out.mkdir(parents=True)

    def write(entities=ENTITIES, mentions=MENTIONS, assertions=ASSERTIONS):
        for name, rows in (("entities.jsonl", entities), ("mentions.resolved.jsonl", mentions),
                           ("assertions.resolved.jsonl", assertions)):
            (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    write()
    return cfg, write, Store()


def expected_quads(cfg, entities=ENTITIES, mentions=MENTIONS, assertions=ASSERTIONS) -> set[str]:
    rm = RdfMap.load(cfg.resolve(cfg.store.rdf_map))
    per = build_quads(rm, cfg.store.graph_base, {e["id"]: e for e in entities}, mentions, assertions, cfg.store.load_mentions)
    return {str(q) for quads in per.values() for q in quads}


# --- done-when ------------------------------------------------------------------


def test_sparql_returns_assertions_with_conditions_and_provenance(env):
    cfg, _, store = env
    run_load(cfg, store)
    rows = list(store.query(QUERY, use_default_graph_as_union=True))
    assert {(r["subjectLabel"].value, r["conditionType"].value, r["conditionValue"].value, r["chunk"].value) for r in rows} == {
        ("compute cluster", "region", "East US 2", "a.md#x"), ("compute cluster", "region", "East US 2", "b.md#y")}
    assert all(r["objectLabel"].value == "managed identity" and r["evidence"].value == "evidence." for r in rows)


def test_reloading_a_changed_chunk_leaves_no_orphans(env):
    cfg, write, store = env
    run_load(cfg, store)
    # Chunk b.md#y changes: assertion 2 and the East US 2 mention disappear, assertion 4 loses nothing.
    mentions = [m for m in MENTIONS if m["id"] != "ckg:mention-4"]
    assertions = [a for a in ASSERTIONS if a["id"] != "ckg:assertion-2"]
    write(mentions=mentions, assertions=assertions)
    rep = run_load(cfg, store)
    assert (rep.loaded, rep.unchanged) == (2, 2)  # b.md#y and assertions reload; a.md#x and entities untouched
    actual = {str(q) for q in store.quads_for_pattern(None, None, None, None)}
    assert actual == expected_quads(cfg, mentions=mentions, assertions=assertions)
    assert not list(store.quads_for_pattern(NamedNode("https://example.org/ckg/assertion-2"), None, None, None))
    assert not list(store.quads_for_pattern(None, None, NamedNode("https://example.org/ckg/assertion-2"), None))


def test_removed_chunks_lose_their_graph(env):
    cfg, write, store = env
    run_load(cfg, store)
    mentions = [m for m in MENTIONS if m["chunk_id"] != "b.md#y"]
    assertions = [a | {"provenance": [p for p in a["provenance"] if p["chunk_id"] != "b.md#y"]} for a in ASSERTIONS]
    assertions = [a for a in assertions if a["provenance"]]
    write(mentions=mentions, assertions=assertions)
    rep = run_load(cfg, store)
    assert rep.removed == 1
    assert {str(q) for q in store.quads_for_pattern(None, None, None, None)} == expected_quads(cfg, mentions=mentions, assertions=assertions)


def test_rerun_changes_nothing(env):
    cfg, _, store = env
    run_load(cfg, store)
    rep = run_load(cfg, store)
    assert rep.loaded == 0 and rep.unchanged == rep.chunks


# --- RDF shape ------------------------------------------------------------------


def test_negated_assertions_have_no_direct_triple(env):
    cfg, _, store = env
    run_load(cfg, store)
    supports = NamedNode("https://example.org/ckg/domain/supports")
    assert not list(store.quads_for_pattern(None, supports, None, None))
    uses = NamedNode("https://example.org/ckg/domain/uses_identity")
    assert len(list(store.quads_for_pattern(NamedNode(V + "compute-cluster"), uses, None, None))) == 1


def test_each_triple_is_stored_once(env):
    cfg, _, store = env
    run_load(cfg, store)
    label = NamedNode("http://www.w3.org/2004/02/skos/core#prefLabel")
    assert len(list(store.quads_for_pattern(NamedNode(V + "compute-cluster"), label, None, None))) == 1
    asserted = NamedNode("https://example.org/ckg/assertion-1")
    subject = NamedNode("https://example.org/ckg/subject")
    assert len(list(store.quads_for_pattern(asserted, subject, None, None))) == 1  # supported by two chunks


# --- validation -------------------------------------------------------------------


def report_rows(cfg):
    with cfg.resolve(cfg.validation.report_file).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_valid_graph_conforms(env):
    cfg, _, store = env
    run_load(cfg, store)
    rep = run_validate(cfg, store)
    assert rep.conforms and rep.violations == 0 and report_rows(cfg) == []


def test_domain_range_and_datatype_violations_are_listed(env):
    cfg, write, store = env
    bad = ASSERTIONS + [
        assertion(10, V + "compute-cluster", "grants", ["a.md#x"], obj=V + "managed-identity"),   # domain: grants needs a Role
        assertion(11, V + "role-reader", "runs_on", ["a.md#x"], obj=V + "managed-identity"),      # range: runs_on needs a ComputeTarget
        assertion(12, E + "old-sdk", "retirement_date", ["b.md#y"], value="June 2025"),           # not an xsd:date
    ]
    write(assertions=bad)
    run_load(cfg, store)
    rep = run_validate(cfg, store)
    rows = report_rows(cfg)
    assert not rep.conforms
    found = {(r["kind"], r["constraint"], r["path"], r["assertion"]) for r in rows}
    assert ("domain", "DomainConstraint", "grants", "ckg:assertion-10") in found
    assert ("shacl", "ClassConstraintComponent", "runs_on", "ckg:assertion-11") in found
    assert ("datatype", "LexicalFormConstraint", "retirement_date", "ckg:assertion-12") in found
    assert all(r["evidence_chunk"] for r in rows if r["assertion"])
    summary = json.loads(cfg.resolve(cfg.validation.summary_file).read_text(encoding="utf-8"))
    assert summary["violations"] == len(rows) and summary["by_kind"]["domain"] >= 1


def test_vocabulary_graph_is_optional(env):
    cfg, _, store = env
    cfg = cfg.model_copy(update={"store": cfg.store.model_copy(update={"load_vocabulary": True})})
    rep = run_load(cfg, store)
    assert rep.vocabulary_loaded
    assert store.contains_named_graph(NamedNode(cfg.store.graph_base + "vocabulary"))
    assert not run_load(cfg, store).vocabulary_loaded  # unchanged file: not reloaded


def test_turning_vocabulary_off_drops_its_graph(env):
    cfg, _, store = env
    on = cfg.model_copy(update={"store": cfg.store.model_copy(update={"load_vocabulary": True})})
    run_load(on, store)
    run_load(cfg, store)
    assert not store.contains_named_graph(NamedNode(cfg.store.graph_base + "vocabulary"))
    assert {str(q) for q in store.quads_for_pattern(None, None, None, None)} == expected_quads(cfg)
