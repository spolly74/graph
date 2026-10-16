"""Phase 7: evaluation and the end-to-end run (DESIGN 5.6, phase 7 done-when)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
import yaml
from pyoxigraph import Store
from typer.testing import CliRunner

from ckg.cli import app
from ckg.config import load_config
from ckg.evaluate.checks import compare_gold, load_questions
from ckg.evaluate.pipeline import run_evaluate
from ckg.load.pipeline import run_load

from conftest import PROJECT, SAMPLE_REPO, sample_config_dict
from test_extract import FakeGliner
from test_load_validate import ASSERTIONS, ENTITIES, MENTIONS, assertion
from test_resolve import FakeEmbedder
from test_tier3 import FakeLlm


@pytest.fixture
def loaded(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(sample_config_dict()), encoding="utf-8")
    cfg = load_config(config)
    out = cfg.resolve(cfg.paths.out_dir)
    out.mkdir(parents=True)
    extra = [assertion(20, "https://example.org/ckg/entity/old-sdk", "release_status", ["b.md#y"], value="preview")]
    for name, rows in (("entities.jsonl", ENTITIES), ("mentions.resolved.jsonl", MENTIONS),
                       ("mentions.jsonl", MENTIONS), ("assertions.resolved.jsonl", ASSERTIONS + extra),
                       ("assertions.jsonl", [])):
        (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    store = Store()
    run_load(cfg, store)
    return cfg, store


# --- done-when: metrics and competency results from the command line --------------


def test_every_competency_question_has_a_query():
    questions = load_questions(PROJECT / "eval" / "competency_questions.md")
    queries = sorted((PROJECT / "eval" / "competency_questions").glob("cq*.rq"))
    assert len(questions) == 35 and [q.stem for q in queries] == [f"cq{n:02d}" for n in sorted(questions)]


def test_evaluate_reports_competency_results_and_metrics(loaded):
    cfg, store = loaded
    rep = run_evaluate(cfg, store)
    assert rep.questions == 35
    data = json.loads(cfg.resolve(cfg.evaluation.metrics_file).read_text(encoding="utf-8"))
    results = {r["n"]: r for r in data["competency_questions"]["results"]}
    assert results[14]["count"] >= 1  # uses_identity
    assert any(r["object"] == "East US 2" for r in results[8]["rows"])  # available_in
    assert any(r["subject"] == "old SDK" for r in results[32]["rows"])  # release_status preview
    assert rep.answered == sum(r["count"] > 0 for r in results.values()) and set(rep.unanswered) == {
        n for n, r in results.items() if r["count"] == 0}
    report = cfg.resolve(cfg.evaluation.report_file).read_text(encoding="utf-8")
    assert "## Competency questions" in report and "### CQ 14:" in report and "| Competency questions answered |" in report


# --- spot check -------------------------------------------------------------------------


def test_spot_check_is_stable_and_keeps_marks(loaded):
    cfg, store = loaded
    run_evaluate(cfg, store)
    path = cfg.resolve(cfg.evaluation.spot_check_file)
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(ASSERTIONS) + 1  # sample size is capped by the assertions available
    rows[0]["correct"], rows[1]["correct"], rows[2]["correct"] = "yes", "no", "yes"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    rep = run_evaluate(cfg, store)
    assert rep.spot == {"sample": len(rows), "marked": 3, "correct": 2, "incorrect": 1, "precision": 0.667}
    with path.open(encoding="utf-8") as f:
        again = {r["assertion"]: r["correct"] for r in csv.DictReader(f)}
    assert [again[r["assertion"]] for r in rows[:3]] == ["yes", "no", "yes"]


# --- unmapped mentions --------------------------------------------------------------------


def test_unmapped_mentions_list_non_vocabulary_entities(loaded):
    cfg, store = loaded
    rep = run_evaluate(cfg, store)
    assert rep.unmapped["entities"] == 1 and rep.unmapped["top"][0]["label"] == "East US 2"
    with cfg.resolve(cfg.evaluation.unmapped_file).open(encoding="utf-8") as f:
        assert [r["label"] for r in csv.DictReader(f)] == ["East US 2"]


# --- optional gold set ----------------------------------------------------------------------


def test_gold_comparison(tmp_path):
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "mentions.jsonl").write_text(
        json.dumps({"chunk_id": "a.md#x", "start": 0, "end": 10, "class": "ComputeTarget"}) + "\n"
        + json.dumps({"chunk_id": "a.md#x", "start": 20, "end": 30, "class": "Role"}) + "\n", encoding="utf-8")
    (gold / "assertions.jsonl").write_text(json.dumps(
        {"chunk_id": "a.md#x", "subject": "AmlCompute", "predicate": "uses_identity", "object": "Managed Identity"}) + "\n",
        encoding="utf-8")
    mentions = [{"id": "m1", "chunk_id": "a.md#x", "start": 0, "end": 10, "text": "AmlCompute", "entity": "e1"},
                {"id": "m2", "chunk_id": "a.md#x", "start": 12, "end": 18, "text": "managed identity", "entity": "e2"},
                {"id": "m3", "chunk_id": "z.md#q", "start": 0, "end": 3, "text": "out of scope", "entity": "e2"}]
    entities = {"e1": {"entity_type": "ComputeTarget"}, "e2": {"entity_type": "IdentityConcept"}}
    assertions = [{"subject": "m1", "predicate": "ckgd:uses_identity", "object": "m2", "negated": False,
                   "provenance": [{"chunk_id": "a.md#x"}]}]
    res = compare_gold(gold, mentions, assertions, entities)
    assert res["mentions"]["micro"] == {"gold": 2, "predicted": 2, "true_positive": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert res["assertions"]["micro"]["f1"] == 1.0
    assert compare_gold(tmp_path / "none", mentions, assertions, entities) is None


# --- end to end ------------------------------------------------------------------------------


def test_ckg_run_end_to_end_on_the_sample(tmp_path, monkeypatch):
    """Every stage through the CLI, with fake GLiNER, LLM, and embedding models."""
    import ckg.extract.pipeline as extract_pipeline
    import ckg.extract.tier3_run as tier3_run
    import ckg.resolve.pipeline as resolve_pipeline

    monkeypatch.setattr(extract_pipeline, "load_gliner", lambda model, revision: (FakeGliner(), "fake@0"))
    monkeypatch.setattr(tier3_run, "OllamaClient", lambda *a, **k: FakeLlm())
    monkeypatch.setattr(resolve_pipeline, "load_embedder", lambda model, revision: FakeEmbedder())
    repo = tmp_path / "repo"
    shutil.copytree(SAMPLE_REPO, repo)
    data = sample_config_dict()
    data["corpus"]["repo_root"] = str(repo)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = CliRunner().invoke(app, ["--config", str(config), "run"])
    assert result.exit_code == 0, result.output
    for line in ("10 documents ->", "tier 3:", "entities", "graphs:", "triples checked", "competency questions:"):
        assert line in result.output
    assert (tmp_path / "review" / "evaluation_report.md").is_file()

    again = CliRunner().invoke(app, ["--config", str(config), "run"])
    assert again.exit_code == 0 and "0 LLM calls this run" in again.output and "0 loaded" in again.output
