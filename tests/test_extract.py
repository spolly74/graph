"""Phase 3: extraction tiers 1 and 2 (DESIGN 5.2, phase 3 done-when)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from ckg.config import MaskConfig, load_config
from ckg.extract.common import masked_spans, unmasked_segments
from ckg.extract.pipeline import DROPPED_FILE, MERGED_FILE, TIER_FILES, run_extract
from ckg.extract.tier1_vocab import VocabMatcher
from ckg.extract.tier2_gliner import WORDS, class_labels, windows
from ckg.extract.vocab import Concept, load_vocabulary
from ckg.generated import model as schema_model
from ckg.ingest.pipeline import run_ingest

from conftest import PROJECT, SAMPLE_BASE, SAMPLE_REPO, sample_config_dict

MASK = MaskConfig(code_blocks=True, link_targets=True, directives=True, html_comments=True, callouts=True)


class FakeGliner:
    """Stands in for GLiNER: tags every 'workspace' and 'secure' as entities."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def inference(self, texts, labels, threshold, batch_size):
        self.calls.append(texts)
        out = []
        for t in texts:
            ents = [{"start": m.start(), "end": m.end(), "text": m.group(), "label": "resource", "score": 0.9}
                    for m in re.finditer(r"\bworkspace\b", t)]
            ents += [{"start": m.start(), "end": m.end(), "text": m.group(), "label": "capability", "score": 0.6}
                     for m in re.finditer(r"\bsecure\b", t)]
            out.append(ents)
        return out


def fake_factory(fake: FakeGliner):
    return lambda model, revision: (fake, "fake@0")


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


@pytest.fixture
def ingested(tmp_path: Path):
    repo = tmp_path / "repo"
    shutil.copytree(SAMPLE_REPO, repo)
    data = sample_config_dict()
    data["corpus"]["repo_root"] = str(repo)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(config)
    run_ingest(cfg)
    return cfg, repo


def chunk_texts(cfg) -> dict[str, str]:
    return {c["chunk_id"]: c["text"] for c in read(cfg.resolve(cfg.paths.out_dir) / "chunks.jsonl")}


# --- done-when: offsets ----------------------------------------------------------


def test_every_mention_matches_its_chunk_text(ingested):
    cfg, _ = ingested
    run_extract(cfg, [1, 2], gliner_factory=fake_factory(FakeGliner()))
    out = cfg.resolve(cfg.paths.out_dir)
    texts = chunk_texts(cfg)
    mentions = read(out / MERGED_FILE)
    assert {m["method"] for m in mentions} == {"vocabulary", "gliner"}
    for m in mentions:
        assert m["text"] == texts[m["chunk_id"]][m["start"] : m["end"]]
        schema_model.Mention(**m)  # conforms to the generated model


# --- tier 1 ---------------------------------------------------------------------


def test_tier1_finds_vocabulary_terms(ingested):
    cfg, _ = ingested
    run_extract(cfg, [1])
    mentions = read(cfg.resolve(cfg.paths.out_dir) / TIER_FILES[1])
    roles = {m["vocab_id"].rsplit("/", 1)[1] for m in mentions if m["chunk_id"] == "how-to-assign-roles.md#default-roles"}
    assert {"role-azureml-data-scientist", "role-azureml-compute-operator", "role-reader"} <= roles


def test_tier1_case_rules():
    concepts = [
        Concept("v:role-reader", "Role", ("Reader",)),
        Concept("v:compute-cluster", "ComputeTarget", ("compute cluster", "AmlCompute")),
    ]
    vm = VocabMatcher(concepts, case_sensitive_single_token=True)
    text = "A reader can view a Compute Cluster. The Reader role. Use amlcompute or AmlCompute."
    found = [(text[s:e], c.iri) for s, e, c in vm.matches(text)]
    assert ("reader", "v:role-reader") not in found
    assert ("Reader", "v:role-reader") in found
    assert ("Compute Cluster", "v:compute-cluster") in found  # multi-word labels ignore case
    assert ("amlcompute", "v:compute-cluster") not in found
    assert ("AmlCompute", "v:compute-cluster") in found


def test_tier1_longest_match_wins():
    vm = VocabMatcher([
        Concept("v:aml", "Product", ("Azure Machine Learning",)),
        Concept("v:workspace", "Resource", ("workspace", "Azure Machine Learning workspace")),
    ], True)
    text = "Open your Azure Machine Learning workspace."
    assert [(text[s:e], c.iri) for s, e, c in vm.matches(text)] == [("Azure Machine Learning workspace", "v:workspace")]


def test_masked_text_is_not_matched():
    text = (
        "Create a compute cluster.\n\n```bash\naz ml compute create --type compute cluster\n```\n\n"
        "See [the guide](notes on compute cluster setup).\n"
        ":::image type=\"content\" alt-text= compute cluster :::\n"
        "<!-- compute cluster -->\n"
    )
    chunk = {"chunk_id": "d.md#x", "text": text, "pieces": [{"start": 0, "end": len(text), "source": "d.md", "source_start": 0}]}
    vm = VocabMatcher([Concept("v:cc", "ComputeTarget", ("compute cluster",))], True)
    kept, rejected = vm.extract(chunk, masked_spans(chunk, MASK))
    assert [m["start"] for m in kept] == [text.index("compute cluster")]
    assert {r["reason"] for r in rejected} == {"masked:code_block", "masked:link_target", "masked:directive", "masked:html_comment"}


def test_placeholders_are_masked():
    ph = "[[UNRESOLVED_INCLUDE: ~/reusable-content/workspace.md]]"
    text = f"Intro.\n{ph}\n"
    start = text.index(ph)
    chunk = {"chunk_id": "d.md#x", "text": text, "pieces": [
        {"start": 0, "end": start, "source": "d.md", "source_start": 0},
        {"start": start, "end": start + len(ph), "placeholder": "~/reusable-content/workspace.md"},
        {"start": start + len(ph), "end": len(text), "source": "d.md", "source_start": start},
    ]}
    vm = VocabMatcher([Concept("v:ws", "Resource", ("workspace",))], True)
    kept, rejected = vm.extract(chunk, masked_spans(chunk, MASK))
    spans = masked_spans(chunk, MASK)
    assert [(sp.start, sp.end, sp.reason) for sp in spans] == [(start, start + len(ph), "placeholder")]
    assert not kept and all(r["reason"] == "masked:placeholder" for r in rejected)
    assert unmasked_segments(text, spans) == [(0, start), (start + len(ph), len(text))]


def test_real_vocabulary_loads_with_classes():
    concepts = load_vocabulary(PROJECT / "vocab" / "vocabulary.ttl")
    assert len(concepts) > 300
    assert all(issubclass(getattr(schema_model, c.domain_class), schema_model.DomainEntity) for c in concepts)


# --- tier 2 ---------------------------------------------------------------------


def test_labels_come_from_concrete_domain_classes():
    labels = class_labels(schema_model, {"AzureService": "Azure service"}, ["Metric"])
    assert labels["compute target"] == "ComputeTarget"
    assert labels["Azure service"] == "AzureService"
    assert "DomainEntity" not in labels.values() and "Metric" not in labels.values()


def test_windows_respect_size_and_masks():
    text = "".join(f"Line {i} has several words in it for testing.\n" for i in range(100))
    text += "```\ncode that must be skipped\n```\n" + " ".join(["word"] * 700) + "\n"
    chunk = {"chunk_id": "x", "text": text, "pieces": []}
    spans = masked_spans(chunk, MASK)
    wins = windows(text, unmasked_segments(text, spans), max_words=60)
    for s, e in wins:
        assert len(WORDS.findall(text[s:e])) <= 60
        assert not any(sp.overlaps(s, e) for sp in spans)
    covered = "".join(text[s:e] for s, e in wins)
    assert "code that must be skipped" not in covered
    assert len(re.findall(r"\bword\b", covered)) == 700 and "Line 99" in covered


def test_tier2_offsets_rejects_and_merge(ingested):
    cfg, _ = ingested
    fake = FakeGliner()
    run_extract(cfg, [1, 2], gliner_factory=fake_factory(fake))
    out = cfg.resolve(cfg.paths.out_dir)
    texts = chunk_texts(cfg)
    t2 = read(out / TIER_FILES[2])
    assert t2 and all(texts[m["chunk_id"]][m["start"] : m["end"]] == m["text"] for m in t2)
    assert all(m["model"] == "fake@0" and m["predicted_class"] in ("Resource", "Capability") for m in t2)
    merged = read(out / MERGED_FILE)
    t1_spans = {(m["chunk_id"], m["start"], m["end"]) for m in merged if m["method"] == "vocabulary"}
    for m in (m for m in merged if m["method"] == "gliner"):
        assert not any(c == m["chunk_id"] and s < m["end"] and m["start"] < e for c, s, e in t1_spans)
    reasons = {d["reason"] for d in read(out / DROPPED_FILE)}
    assert "overlaps_tier1" in reasons  # "workspace" is also a vocabulary label


# --- reruns ---------------------------------------------------------------------


def outputs(cfg) -> dict[str, bytes]:
    out = cfg.resolve(cfg.paths.out_dir)
    return {p.name: p.read_bytes() for p in sorted(out.glob("mentions*.jsonl"))}


def test_rerun_is_identical(ingested):
    cfg, _ = ingested
    run_extract(cfg, [1, 2], gliner_factory=fake_factory(FakeGliner()))
    first = outputs(cfg)
    run_extract(cfg, [1, 2], gliner_factory=fake_factory(FakeGliner()))
    assert outputs(cfg) == first


def test_changed_only_reprocesses_only_changed_chunks(ingested):
    cfg, repo = ingested
    run_extract(cfg, [1, 2], gliner_factory=fake_factory(FakeGliner()))
    article = repo / SAMPLE_BASE / "concept-endpoints.md"
    article.write_text(article.read_text(encoding="utf-8") + "\n## Extra\n\nA secure Kubernetes cluster workspace.\n", encoding="utf-8")
    run_ingest(cfg)
    chunks = read(cfg.resolve(cfg.paths.out_dir) / "chunks.jsonl")

    report = run_extract(cfg, [1, 2], changed_only=True, gliner_factory=fake_factory(FakeGliner()))
    changed = report.tiers[0].processed
    assert 0 < changed < len(chunks) / 4
    assert all(t.processed == changed and t.reused == len(chunks) - changed for t in report.tiers)
    incremental = outputs(cfg)

    run_extract(cfg, [1, 2], gliner_factory=fake_factory(FakeGliner()))
    assert outputs(cfg) == incremental, "incremental run must equal a full run"


def test_tier1_matches_regular_plurals():
    vm = VocabMatcher([
        Concept("v:oe", "Endpoint", ("online endpoint",)),
        Concept("v:pol", "Policy", ("policy",)),
        Concept("v:kv", "AzureService", ("Azure Key Vault",)),
    ], True, match_plurals=True)
    text = "Two online endpoints, three policies, and Azure Key Vaults."
    found = {text[s:e]: c.iri for s, e, c in vm.matches(text)}
    assert found == {"online endpoints": "v:oe", "policies": "v:pol"}


def test_callout_markers_are_masked():
    text = "> [!NOTE]\n> Use a workspace.\n"
    chunk = {"chunk_id": "x", "text": text, "pieces": []}
    spans = masked_spans(chunk, MASK)
    assert [(text[s.start:s.end], s.reason) for s in spans] == [("[!NOTE]", "callout")]
