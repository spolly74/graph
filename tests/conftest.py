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
