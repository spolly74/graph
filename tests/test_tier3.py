"""Phase 4: tier 3 extraction (DESIGN 5.2, phase 4 done-when), using a fake LLM."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from ckg.config import load_config
from ckg.extract.common import Span
from ckg.extract.pipeline import MERGED_FILE, run_extract
from ckg.extract.tier3_llm import CODE_MARKER, LlmResult, Tier3Schema, Validator, find_evidence, user_message
from ckg.extract.tier3_run import ASSERTIONS_FILE, REJECTED_FILE, SKIPPED_FILE, TIMING_FILE, spread
from ckg.generated import model as schema_model
from ckg.ingest.pipeline import run_ingest

from conftest import PROJECT, SAMPLE_REPO, sample_config_dict

SCHEMA = Tier3Schema.load(PROJECT / "schema" / "generated" / "ckg.predicates.json")


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


class FakeLlm:
    """Returns one good assertion per chunk (built from the first two entities and the
    line the first one appears on) and one with paraphrased evidence."""

    model_id = "fake@0"

    def __init__(self):
        self.calls = 0

    def chat(self, system, user, schema):
        self.calls += 1
        entities = [l[2:].rsplit(" (", 1)[0] for l in user.split("\n") if l.startswith("- ")]
        text = user.split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0]
        line = next((l for l in text.splitlines() if entities[0] in l), "")
        good = {"subject": entities[0], "subject_class": "unknown", "predicate": "integrates_with",
                "object": entities[1], "object_class": "unknown", "object_value": "", "negated": False,
                "conditions": [], "evidence": line.strip()}
        bad = dict(good, evidence="A sentence the model made up.")
        return LlmResult(json.dumps({"assertions": [good, bad]}), 100, 20, 1.5)


@pytest.fixture
def extracted(tmp_path: Path):
    repo = tmp_path / "repo"
    shutil.copytree(SAMPLE_REPO, repo)
    data = sample_config_dict()
    data["corpus"]["repo_root"] = str(repo)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(config)
    run_ingest(cfg)
    run_extract(cfg, [1])
    return cfg


def run3(cfg, fake, limit=None):
    return run_extract(cfg, [3], llm_factory=lambda: fake, limit=limit).tier3


# --- done-when ------------------------------------------------------------------


def test_every_assertion_evidence_is_an_exact_substring(extracted):
    cfg = extracted
    rep = run3(cfg, FakeLlm())
    out = cfg.resolve(cfg.paths.out_dir)
    chunks = {c["chunk_id"]: c for c in read(out / "chunks.jsonl")}
    assertions = read(out / ASSERTIONS_FILE)
    assert assertions and rep.assertions == len(assertions)
    for a in assertions:
        for p in a["provenance"]:
            assert chunks[p["chunk_id"]]["text"][p["start"] : p["end"]] == p["evidence"]
        schema_model.Assertion(**a)
    rejected = read(out / REJECTED_FILE)
    assert {r["reason"] for r in rejected} >= {"evidence_not_in_chunk"}
    assert all("reason" in r for r in rejected)


def test_timing_is_recorded(extracted):
    cfg = extracted
    rep = run3(cfg, FakeLlm())
    timing = json.loads((cfg.resolve(cfg.paths.out_dir) / TIMING_FILE).read_text(encoding="utf-8"))
    assert timing["chunks"] == rep.gated and timing["median_seconds"] == 1.5 and timing["model"] == "fake@0"


# --- gating, references, resumability ----------------------------------------------


def test_gating_and_skips_are_logged(extracted):
    cfg = extracted
    rep = run3(cfg, FakeLlm())
    skipped = read(cfg.resolve(cfg.paths.out_dir) / SKIPPED_FILE)
    assert rep.gated + rep.skipped == len(read(cfg.resolve(cfg.paths.out_dir) / "chunks.jsonl"))
    assert len(skipped) == rep.skipped and all(s["distinct"] < 2 for s in skipped)


def test_assertions_refer_to_merged_mentions(extracted):
    cfg = extracted
    run3(cfg, FakeLlm())
    out = cfg.resolve(cfg.paths.out_dir)
    ids = {m["id"] for m in read(out / MERGED_FILE)}
    for a in read(out / ASSERTIONS_FILE):
        assert a["subject"] in ids and a["object"] in ids


def test_runs_resume_from_cache(extracted):
    cfg = extracted
    out = cfg.resolve(cfg.paths.out_dir)
    fake = FakeLlm()
    first = run3(cfg, fake, limit=3)
    assert first.called == 3 and first.missing == first.gated - 3
    second = run3(cfg, fake)
    assert second.called == first.gated - 3 and second.missing == 0
    snapshot = (out / ASSERTIONS_FILE).read_bytes()
    third = run3(cfg, fake)
    assert third.called == 0 and fake.calls == first.gated
    assert (out / ASSERTIONS_FILE).read_bytes() == snapshot


def test_spread_picks_evenly():
    assert spread(list(range(10)), 3) == [0, 3, 6]
    assert spread([1, 2], 5) == [1, 2]


# --- prompt and schema ------------------------------------------------------------


def test_prompt_lists_every_predicate_and_class():
    system = SCHEMA.render_system((PROJECT / "prompts" / "tier3_system.md").read_text(encoding="utf-8"))
    assert "{" + "predicates}" not in system and "(CQ" not in system
    for name in SCHEMA.predicates:
        assert f"- {name} (" in system
    for name in SCHEMA.class_names:
        assert f"- {name}:" in system
    schema = SCHEMA.output_schema()
    item = schema["properties"]["assertions"]["items"]["properties"]
    assert set(item["predicate"]["enum"]) == set(SCHEMA.predicates)


def test_code_blocks_are_omitted_from_the_prompt():
    text = "## Setup\n\nRun this:\n\n```bash\naz ml compute create\n```\n\nThen continue.\n"
    start = text.index("```")
    end = text.index("\n\nThen") + 1
    chunk = {"chunk_id": "d.md#setup", "heading_path": ["Doc", "Setup"], "text": text}
    msg = user_message(chunk, [{"text": "compute", "predicted_class": "Resource"}], [Span(start, end, "code_block")])
    assert "az ml compute create" not in msg and CODE_MARKER in msg and "Then continue." in msg
    assert "Section: Doc > Setup" in msg and "- compute (Resource)" in msg


# --- validator rules -----------------------------------------------------------------


def chunk_with(text: str) -> dict:
    return {"chunk_id": "d.md#s", "doc_id": "d.md", "text": text, "heading_path": ["D"],
            "pieces": [{"start": 0, "end": len(text), "source": "articles/x/d.md", "source_start": 0}]}


def item(**kw) -> dict:
    base = {"subject": "", "subject_class": "unknown", "predicate": "requires", "object": "", "object_class": "unknown",
            "object_value": "", "negated": False, "conditions": [], "evidence": ""}
    return base | kw


def validate(text, items, mentions=()):
    v = Validator(SCHEMA, "fake@0", whitespace_tolerant=True)
    return v.validate(chunk_with(text), "articles/x/d.md", list(mentions), json.dumps({"assertions": items}))


def test_whitespace_tolerant_evidence_is_stored_exactly():
    text = "Serverless compute   requires a\nmanaged identity.\n"
    span = find_evidence(text, "Serverless compute requires a managed identity.", True)
    assert text[span[0] : span[1]] == "Serverless compute   requires a\nmanaged identity."
    assert find_evidence(text, "Serverless compute requires an identity.", True) is None
    assert find_evidence(text, "Serverless compute requires a managed identity.", False) is None


def test_validator_rules():
    text = "Kubernetes compute doesn't support serverless compute. The Contributor role is generally available.\n"
    res = validate(text, [
        item(subject="Kubernetes compute", object="serverless compute", predicate="supports", negated=True,
             evidence="Kubernetes compute doesn't support serverless compute."),
        item(subject="Kubernetes compute", object="Azure Batch", evidence="Kubernetes compute doesn't support serverless compute."),
        item(subject="Contributor role", predicate="release_status", object_value="Generally available",
             evidence="The Contributor role is generally available."),
        item(subject="Contributor role", predicate="release_status", object_value="beta",
             evidence="The Contributor role is generally available."),
        item(subject="Kubernetes compute", predicate="flies_to", object="moon", evidence="Kubernetes compute"),
        item(subject="Kubernetes compute", object="Kubernetes compute", evidence="Kubernetes compute"),
        item(subject="Kubernetes compute", object="serverless compute", predicate="supports", negated=True,
             evidence="Kubernetes compute doesn't support serverless compute."),
    ])
    assert [r["reason"] for r in res.rejected] == [
        "object_not_in_chunk", "invalid_value", "unknown_predicate", "self_reference", "duplicate"]
    neg, status = res.assertions
    assert neg["negated"] is True and neg["predicate"] == "ckgd:supports"
    assert status["object_value"] == "generally_available" and "object" not in status
    assert {m["text"] for m in res.mentions} == {"Kubernetes compute", "serverless compute", "Contributor role"}
    assert all(m["method"] == "llm" for m in res.mentions)


def test_validator_prefers_existing_mentions_and_records_includes():
    text = "Use a compute cluster. A compute cluster requires a virtual network.\n"
    second = text.index("compute cluster", 10)
    mentions = [
        {"id": "m1", "text": "compute cluster", "start": text.index("compute cluster"), "end": text.index("compute cluster") + 15},
        {"id": "m2", "text": "compute cluster", "start": second, "end": second + 15},
        {"id": "m3", "text": "virtual network", "start": text.index("virtual network"), "end": text.index("virtual network") + 15},
    ]
    res = validate(text, [item(subject="compute cluster", object="virtual network",
                               evidence="A compute cluster requires a virtual network.")], mentions)
    a = res.assertions[0]
    assert (a["subject"], a["object"]) == ("m2", "m3") and not res.mentions
    assert "include_file" not in a["provenance"][0]  # evidence comes from the article itself


def test_invalid_json_is_rejected():
    v = Validator(SCHEMA, "fake@0", True)
    res = v.validate(chunk_with("text"), "articles/x/d.md", [], "not json")
    assert res.rejected[0]["reason"] == "invalid_json"


def test_truncated_output_keeps_complete_items():
    from ckg.extract.tier3_llm import parse_items

    good = item(subject="a", object="b", evidence="a b")
    raw = json.dumps({"assertions": [good, good]})[:-30]  # cut inside the second item
    items, truncated = parse_items(raw)
    assert truncated and items == [good]
    assert parse_items("nonsense") == (None, False)


def test_class_suffix_copied_from_entity_list_is_ignored():
    text = "The workspace requires a storage account.\n"
    res = validate(text, [item(subject="workspace (Resource)", object="storage account (unknown)",
                               evidence="The workspace requires a storage account.")])
    assert not res.rejected and {m["text"] for m in res.mentions} == {"workspace", "storage account"}


def test_code_include_lines_are_omitted():
    text = "## Test\n\nRun it:\n\n[!notebook-python[] (~/examples/run.ipynb?name=test_deployment)]\n\nDone.\n"
    chunk = {"chunk_id": "d.md#t", "heading_path": ["T"], "text": text}
    msg = user_message(chunk, [{"text": "x", "predicted_class": "Asset"}], [])
    assert "test_deployment" not in msg and CODE_MARKER in msg and "Done." in msg


def test_output_schema_caps_assertions():
    assert SCHEMA.output_schema(12)["properties"]["assertions"]["maxItems"] == 12
