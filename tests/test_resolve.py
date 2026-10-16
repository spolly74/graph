"""Phase 5: entity resolution and review queue (DESIGN 5.3, phase 5 done-when)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ckg.config import load_config
from ckg.generated import model as schema_model
from ckg.resolve.normalize import normalize, singular
from ckg.resolve.pipeline import run_resolve, variant_marks

from conftest import sample_config_dict

VOCAB = "https://example.org/ckg/vocab/"
ENTITY = "https://example.org/ckg/entity/"


class FakeEmbedder:
    """Character-trigram vectors: similar spellings get similar vectors, deterministically."""

    def encode(self, texts):
        out = np.zeros((len(texts), 256))
        for row, t in enumerate(texts):
            padded = f"  {t} "
            for i in range(len(padded) - 2):
                out[row, int(hashlib.md5(padded[i : i + 3].encode()).hexdigest(), 16) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def m(i: int, text: str, cls: str, method: str = "gliner", vocab: str | None = None, chunk: str = "d.md#s") -> dict:
    rec = {"id": f"ckg:mention-{i}", "chunk_id": chunk, "start": 0, "end": len(text), "text": text,
           "predicted_class": cls, "method": method}
    if vocab:
        rec["vocab_id"] = VOCAB + vocab
    return rec


def assertion(i: int, s: str, o: str, pred: str = "ckgd:supports", chunk: str = "d.md#s") -> dict:
    return {"id": f"ckg:assertion-{i}", "subject": s, "predicate": pred, "object": o, "negated": False,
            "conditions": [], "provenance": [{"source_doc": "d.md", "chunk_id": chunk, "start": 0, "end": 4,
                                              "evidence": "text", "method": "llm", "model": "fake@0"}]}


@pytest.fixture
def setup(tmp_path: Path):
    def _run(mentions, assertions=(), **resolution):
        data = sample_config_dict()
        data["resolution"].update(resolution)
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump(data), encoding="utf-8")
        cfg = load_config(config)
        out = cfg.resolve(cfg.paths.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in (("mentions.jsonl", mentions), ("assertions.jsonl", assertions)):
            (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        report = run_resolve(cfg, embedder_factory=FakeEmbedder)
        return cfg, report, Outputs(cfg)
    return _run


class Outputs:
    def __init__(self, cfg):
        out = cfg.resolve(cfg.paths.out_dir)
        read = lambda n: [json.loads(l) for l in (out / n).read_text(encoding="utf-8").splitlines()]
        self.entities = {e["id"]: e for e in read("entities.jsonl")}
        self.mentions = {x["id"]: x for x in read("mentions.resolved.jsonl")}
        self.assertions = read("assertions.resolved.jsonl")
        self.dropped = read("assertions.unresolved.jsonl")
        self.queue_path = cfg.resolve(cfg.resolution.queue_file)
        with self.queue_path.open(encoding="utf-8") as f:
            self.queue = list(csv.DictReader(f))

    def entity_of(self, mention_index: int) -> str:
        return self.mentions[f"ckg:mention-{mention_index}"]["entity"]


# --- done-when ------------------------------------------------------------------


def test_vocabulary_aliases_resolve_to_one_id(setup):
    _, _, out = setup([
        m(1, "AmlCompute", "ComputeTarget", "vocabulary", vocab="compute-cluster"),
        m(2, "compute clusters", "Resource"),          # plural of a vocabulary label, other predicted class
        m(3, "Compute Cluster", "ComputeTarget", "llm"),
        m(4, "training cluster", "Job", "vocabulary", vocab="compute-cluster"),
    ])
    assert {out.entity_of(i) for i in (1, 2, 3, 4)} == {VOCAB + "compute-cluster"}
    e = out.entities[VOCAB + "compute-cluster"]
    assert e["label"] == "compute cluster" and e["entity_type"] == "ComputeTarget"
    assert set(e["aliases"]) == {"AmlCompute", "compute clusters", "Compute Cluster", "training cluster"}


PAIR = {"quokka sampler", "quokka sampling"}  # Metric has no vocabulary concepts to interfere


def test_borderline_pairs_go_to_the_review_queue(setup):
    _, rep, out = setup([m(1, "quokka sampler", "Metric"), m(2, "quokka sampling", "Metric")],
                        merge_threshold=0.99, review_threshold=0.5)
    assert out.entity_of(1) != out.entity_of(2)
    row = next(r for r in out.queue if {r["label_a"], r["label_b"]} == PAIR)
    assert 0.5 <= float(row["score"]) < 0.99 and row["decision"] == "" and rep.queued >= 1


# --- review decisions ------------------------------------------------------------


def test_merge_decision_is_applied_and_kept(setup):
    mentions = [m(1, "quokka sampler", "Metric"), m(2, "quokka sampling", "Metric")]
    cfg, _, out = setup(mentions, merge_threshold=0.99, review_threshold=0.5)
    rows = out.queue
    for r in rows:
        r["decision"] = "merge" if {r["label_a"], r["label_b"]} == PAIR else ""
    with out.queue_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    rep = run_resolve(cfg, embedder_factory=FakeEmbedder)
    after = Outputs(cfg)
    assert after.entity_of(1) == after.entity_of(2) and rep.human_merges == 1
    assert [r["decision"] for r in after.queue] == ["merge"]


def test_separate_decision_overrides_an_automatic_merge(setup):
    mentions = [m(1, "quokka score", "Metric"), m(2, "quokka scores", "Metric"), m(3, "quokka-score", "Metric")]
    cfg, rep, out = setup(mentions, merge_threshold=0.5, review_threshold=0.3)
    assert out.entity_of(1) == out.entity_of(2) == out.entity_of(3) and rep.auto_merges == 1
    pair = {"class": "Metric", "name_a": "name:quokka score", "name_b": "name:quokka-score", "decision": "separate"}
    with out.queue_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, pair.keys(), lineterminator="\n")
        w.writeheader()
        w.writerow(pair)
    run_resolve(cfg, embedder_factory=FakeEmbedder)
    after = Outputs(cfg)
    assert after.entity_of(1) == after.entity_of(2) != after.entity_of(3)
    assert any(r["decision"] == "separate" for r in after.queue)


# --- guards ----------------------------------------------------------------------


def test_two_vocabulary_concepts_never_merge(setup):
    _, _, out = setup([m(1, "online endpoint", "Endpoint", "vocabulary", vocab="online-endpoint"),
                       m(2, "batch endpoint", "Endpoint", "vocabulary", vocab="batch-endpoint")],
                      merge_threshold=0.1, review_threshold=0.05)
    assert out.entity_of(1) != out.entity_of(2)


def test_names_with_different_numbers_are_queued_not_merged(setup):
    _, _, out = setup([m(1, "Llama 3.1 8B Instruct", "Asset"), m(2, "Llama 3.1 405B Instruct", "Asset")],
                      merge_threshold=0.5, review_threshold=0.3)
    assert out.entity_of(1) != out.entity_of(2)
    assert any(r["note"] == "numbers differ" for r in out.queue)
    assert variant_marks("Command R+") == ["+"]


def test_exact_match_classes_are_not_fuzzy_matched(setup):
    _, _, out = setup([m(1, "East US", "Region"), m(2, "East US 2", "Region"), m(3, "east us", "Region")],
                      merge_threshold=0.5, review_threshold=0.3)
    assert out.entity_of(1) == out.entity_of(3) != out.entity_of(2)
    assert not out.queue


def test_class_is_the_majority_vote(setup):
    _, _, out = setup([m(1, "Azure CLI", "Interface"), m(2, "Azure CLI", "Interface"), m(3, "azure cli", "PromptFlowTool")])
    e = out.entities[out.entity_of(1)]
    assert out.entity_of(3) == e["id"] and e["entity_type"] == "Interface" and e["label"] == "Azure CLI"
    assert e["id"] == ENTITY + "azure-cli"


def test_short_names_are_logged(setup):
    cfg, rep, out = setup([m(1, "H", "Metric"), m(2, "workspace", "Resource", "vocabulary", vocab="workspace")])
    assert rep.unresolved["name_too_short"] == 1 and "ckg:mention-1" not in out.mentions


# --- assertions --------------------------------------------------------------------


def test_assertions_move_to_entities_and_duplicates_merge(setup):
    mentions = [m(1, "AmlCompute", "ComputeTarget", "vocabulary", vocab="compute-cluster"),
                m(2, "compute cluster", "ComputeTarget", "vocabulary", vocab="compute-cluster", chunk="e.md#t"),
                m(3, "managed identity", "IdentityConcept", "vocabulary", vocab="managed-identity"),
                m(4, "managed identity", "IdentityConcept", "vocabulary", vocab="managed-identity", chunk="e.md#t")]
    assertions = [assertion(1, "ckg:mention-1", "ckg:mention-3"),
                  assertion(2, "ckg:mention-2", "ckg:mention-4", chunk="e.md#t"),
                  assertion(3, "ckg:mention-1", "ckg:mention-2")]
    _, rep, out = setup(mentions, assertions)
    assert len(out.assertions) == 1
    a = out.assertions[0]
    assert (a["subject"], a["object"]) == (VOCAB + "compute-cluster", VOCAB + "managed-identity")
    assert [p["chunk_id"] for p in a["provenance"]] == ["d.md#s", "e.md#t"]
    assert out.dropped[0]["reason"] == "self_reference_after_resolution"
    schema_model.Assertion(**a)


def test_outputs_conform_and_rerun_is_identical(setup):
    mentions = [m(1, "AmlCompute", "ComputeTarget", "vocabulary", vocab="compute-cluster"),
                m(2, "Azure CLI", "Interface"), m(3, "Azure CLIs", "Interface"), m(4, "TensorFlow", "Framework")]
    cfg, _, out = setup(mentions)
    for e in out.entities.values():
        getattr(schema_model, e["entity_type"])(**e)
    assert all(x["entity"] in out.entities for x in out.mentions.values())
    first = {p.name: p.read_bytes() for p in cfg.resolve(cfg.paths.out_dir).glob("*.resolved.jsonl")}
    run_resolve(cfg, embedder_factory=FakeEmbedder)
    assert {p.name: p.read_bytes() for p in cfg.resolve(cfg.paths.out_dir).glob("*.resolved.jsonl")} == first


# --- normalization --------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("Online Endpoints", "online endpoint"), ("**Policies**", "policy"), ("`uri_file`", "uri_file"),
    ("Azure Arc - enabled  workspace", "azure arc-enabled workspace"), ("indexes", "index"),
    ("access", "access"), ("status", "status"), ("analysis", "analysis"),
])
def test_normalize(raw, expected):
    assert normalize(raw, singularize=True) == expected


def test_singular_leaves_short_words():
    assert singular("vms") == "vms" and singular("nodes") == "node" and singular("gpus") == "gpus"
