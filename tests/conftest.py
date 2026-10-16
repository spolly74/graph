from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT = Path(__file__).resolve().parent.parent
SAMPLE_REPO = PROJECT / "data" / "sample"
SAMPLE_BASE = Path("articles/machine-learning")


def sample_config_dict() -> dict:
    return {
        "paths": {"out_dir": "out", "review_dir": "review", "state_db": "out/state.sqlite"},
        "corpus": {
            "repo_root": str(SAMPLE_REPO),
            "base_dir": str(SAMPLE_BASE),
            "exclude": ["v1/*", "data-science-virtual-machine/*", "includes/*"],
            "external_include_prefixes": ["~/reusable-content/", "~/azure-policy-autogen-docs/"],
        },
        "ingest": {
            "chars_per_token": 4,
            "chunk_target_tokens": 600,
            "chunk_max_tokens": 1200,
            "max_include_depth": 10,
            "placeholder_format": "[[UNRESOLVED_INCLUDE: {path}]]",
            "preview_include_patterns": ["*preview*"],
            "front_matter_keys": ["title", "description", "ms.topic", "ms.subservice", "ms.date", "author"],
        },
        "extract": {
            "vocabulary": str(PROJECT / "vocab" / "vocabulary.ttl"),
            "mask": {"code_blocks": True, "link_targets": True, "directives": True, "html_comments": True, "callouts": True},
            "tier1": {"case_sensitive_single_token": True, "match_plurals": True},
            "tier2": {
                "model": "urchade/gliner_medium-v2.1",
                "revision": "40ec419335d09393f298636f471328b722c6da9e",
                "threshold": 0.5,
                "max_words": 250,
                "batch_size": 8,
                "label_overrides": {"AzureService": "Azure service"},
                "exclude_classes": [],
            },
            "tier3": {
                "ollama_url": "http://localhost:11434",
                "model": "qwen2.5:7b-instruct",
                "prompt_file": str(PROJECT / "prompts" / "tier3_system.md"),
                "predicates_file": str(PROJECT / "schema" / "generated" / "ckg.predicates.json"),
                "temperature": 0,
                "seed": 0,
                "num_ctx": 8192,
                "max_output_tokens": 1536,
                "max_assertions_per_chunk": 12,
                "timeout_seconds": 900,
                "min_distinct_mentions": 2,
                "omit_code_blocks": True,
                "evidence_whitespace_tolerant": True,
            },
        },
        "resolution": {
            "entity_base": "https://example.org/ckg/entity/",
            "singularize": True,
            "min_name_length": 2,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "string_weight": 0.5,
            "merge_threshold": 0.9,
            "review_threshold": 0.8,
            "exact_match_classes": ["Permission", "Region", "ErrorMessage", "Limit", "YamlSchema"],
            "queue_file": "review/resolution_queue.csv",
        },
        "store": {
            "path": "out/oxigraph",
            "rdf_map": str(PROJECT / "schema" / "generated" / "ckg.rdf_map.json"),
            "graph_base": "https://example.org/ckg/graph/",
            "load_mentions": True,
            "load_vocabulary": False,
        },
        "validation": {
            "shapes": str(PROJECT / "schema" / "generated" / "ckg.shacl.ttl"),
            "ontology": str(PROJECT / "schema" / "generated" / "ckg.owl.ttl"),
            "report_file": "review/validation_report.csv",
            "summary_file": "out/validation.summary.json",
        },
        "evaluation": {
            "questions_file": str(PROJECT / "eval" / "competency_questions.md"),
            "queries_dir": str(PROJECT / "eval" / "competency_questions"),
            "rows_in_report": 12,
            "spot_check_file": "review/spot_check.csv",
            "spot_check_size": 50,
            "seed": 42,
            "unmapped_file": "review/unmapped_mentions.csv",
            "report_file": "review/evaluation_report.md",
            "metrics_file": "out/evaluation.json",
            "gold_dir": "eval/gold",
        },
    }


@pytest.fixture
def write_config(tmp_path: Path):
    def _write(data: dict) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def sample_config(write_config) -> Path:
    return write_config(sample_config_dict())
