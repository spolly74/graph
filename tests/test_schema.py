"""Phase 2: core and domain schema, generated artifacts (DESIGN section 6)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ckg.generated import model as m

from conftest import PROJECT

SCHEMA_DIR = PROJECT / "schema"
GENERATED = SCHEMA_DIR / "generated"


def schema_view():
    pytest.importorskip("linkml")
    from linkml_runtime.utils.schemaview import SchemaView

    return SchemaView(str(SCHEMA_DIR / "domain.yaml"))


def test_generated_files_are_up_to_date():
    pytest.importorskip("linkml")
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_schema", PROJECT / "tools" / "gen_schema.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    stale = [p.name for p, text in gen.generate().items() if not p.exists() or p.read_text(encoding="utf-8") != text]
    assert not stale, f"rerun tools/gen_schema.py: {stale}"


def test_domain_condition_types_include_core_values():
    core = yaml.safe_load((SCHEMA_DIR / "core.yaml").read_text(encoding="utf-8"))
    required = set(core["enums"]["CoreConditionType"]["permissible_values"])
    assert required <= {e.value for e in m.ConditionType}


def test_relationship_slots_have_domain_range_and_owner():
    sv = schema_view()
    domain_slots = yaml.safe_load((SCHEMA_DIR / "domain.yaml").read_text(encoding="utf-8"))["slots"]
    classes = set(sv.all_classes())
    for name in domain_slots:
        slot = sv.get_slot(name)
        if slot.range not in classes:
            continue  # attribute slot
        assert slot.domain, f"{name} has no domain"
        assert name in sv.class_slots(slot.domain), f"{name} is not a slot of its domain class {slot.domain}"


def test_vocabulary_classes_exist_in_schema():
    ttl = (PROJECT / "vocab" / "vocabulary.ttl").read_text(encoding="utf-8")
    vocab_classes = set(re.findall(r'ckg:domainClass "(\w+)"', ttl))
    assert vocab_classes
    missing = [c for c in vocab_classes if not issubclass(getattr(m, c, object), m.DomainEntity)]
    assert not missing


def test_domain_classes_keep_their_type_when_serialized():
    ct = m.ComputeTarget(id="v:compute-cluster", label="compute cluster", runs_on=["v:x"])
    data = ct.model_dump(exclude_none=True)
    assert data["entity_type"] == "ComputeTarget"
    assert "has_gpu_support" in m.ComputeTarget.model_fields
    assert "grants" not in m.ComputeTarget.model_fields


def provenance(**kw) -> m.Provenance:
    base = dict(source_doc="d.md", chunk_id="d.md#s", evidence="text", method="llm", start=0, end=4)
    return m.Provenance(**(base | kw))


def test_assertion_with_conditions_and_provenance():
    a = m.Assertion(
        id="ckg:a1", subject="v:serverless-api-endpoint", predicate="ckgd:available_in", object="v:east-us-2",
        conditions=[m.Condition(type="release_status", value="preview")], provenance=[provenance()],
    )
    assert a.conditions[0].type == m.ConditionType.release_status.value


@pytest.mark.parametrize("bad", [
    lambda: m.Condition(type="planet", value="Mars"),
    lambda: m.Mention(id="ckg:m1", chunk_id="c", text="x", method="vocabulary", start=0),  # no end
    lambda: m.Mention(id="ckg:m1", chunk_id="c", text="x", method="vocabulary", start=0, end=1, score=1.5),
    lambda: m.Assertion(id="ckg:a1", subject="v:a", predicate="ckgd:supports", provenance=[]),  # at least one provenance record
    lambda: provenance(evidence=None),
])
def test_invalid_records_are_rejected(bad):
    with pytest.raises(ValidationError):
        bad()


def test_json_schema_validates_a_knowledge_graph():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((GENERATED / "ckg.schema.json").read_text(encoding="utf-8"))
    graph = {
        "entities": [{"id": "v:role-reader", "entity_type": "Role", "label": "Reader"}],
        "assertions": [{
            "id": "ckg:a1", "subject": "v:role-reader", "predicate": "ckgd:grants", "object_value": "read-only access",
            "provenance": [{"source_doc": "d.md", "chunk_id": "d.md#s", "evidence": "Read-only", "method": "llm"}],
        }],
    }
    jsonschema.validate(graph, schema)
    graph["assertions"][0]["conditions"] = [{"type": "planet", "value": "Mars"}]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(graph, schema)


def test_shacl_constrains_relationship_ranges():
    rdflib = pytest.importorskip("rdflib")
    g = rdflib.Graph().parse(GENERATED / "ckg.shacl.ttl", format="turtle")
    q = """
    PREFIX sh: <http://www.w3.org/ns/shacl#>
    SELECT ?cls WHERE {
      ?shape sh:targetClass ?target ; sh:property ?p .
      ?p sh:path <https://example.org/ckg/domain/runs_on> ; sh:class ?cls .
    }"""
    ranges = {str(r.cls).rsplit("/", 1)[-1] for r in g.query(q)}
    assert ranges == {"ComputeTarget"}
